import pandas as pd
import numpy as np
import os
from datetime import datetime, time, timedelta
import pytz
import warnings
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()


class ReverseStochasticRSIBacktester:
    def __init__(self, fyers_client_id, fyers_access_token, symbols=None,
                 rsi_period=14, stoch_period=14, stoch_smooth_k=3, stoch_smooth_d=3,
                 srsi_overbought=80, srsi_oversold=20,
                 rsi_declining_threshold=55, rsi_neutral_zone_low=45, rsi_neutral_zone_high=65,
                 divergence_lookback=10,
                 atr_period=14, stop_loss_atr_mult=1.5, target_atr_mult=3.0,
                 use_trailing_stop=False, trailing_stop_atr_mult=2.0,
                 volume_threshold_mult=1.2,
                 initial_capital=100000, square_off_time="15:20",
                 tick_interval='5', last_entry_time="14:30",
                 max_trades_per_day=3, min_risk_reward=1.5,
                 backtest_days=30, min_data_points=100,
                 long_only=None):
        """
        Reverse Stochastic RSI (SRSI) Cross-Under Strategy for Intraday Bearish Reversals

        STRATEGY CONCEPT:
        -----------------
        A Reverse Stochastic RSI cross-under occurs when the fast %K line of the
        Stochastic RSI crosses BELOW the slow %D line while both are positioned below
        a slowing or declining RSI. This signals that momentum-of-momentum is turning
        bearish faster than the underlying RSI, typically at a price top or after a
        failed rally.

        INDICATOR CALCULATION:
        ----------------------
        1. RSI (Relative Strength Index):
           - Standard RSI computed over rsi_period bars
           - RSI = 100 - (100 / (1 + RS)), RS = Avg Gain / Avg Loss

        2. Stochastic RSI (%K and %D):
           - Step 1: Calculate RSI over rsi_period
           - Step 2: Apply Stochastic formula to RSI values over stoch_period
             %K_raw = (RSI - RSI_lowest_N) / (RSI_highest_N - RSI_lowest_N) * 100
           - Step 3: Smooth %K with SMA(stoch_smooth_k) → StochRSI_%K
           - Step 4: Smooth %K further with SMA(stoch_smooth_d) → StochRSI_%D

        BEARISH SIGNAL (PRIMARY - SHORT):
        ----------------------------------
        The core setup triggers when ALL of the following align:
        1. Stochastic RSI %K crosses BELOW %D  (bearish crossover)
        2. Both %K and %D are below the declining/slowing RSI line
        3. RSI is in declining or slowing state (RSI < rsi_declining_threshold, or
           RSI turning down from a recent peak)
        4. Context: either in overbought zone (>80) OR neutral failed-rally zone (50-65)
        5. Volume surge confirms the reversal move

        DIVERGENCE CONFIRMATION (OPTIONAL BONUS):
        ------------------------------------------
        - Price makes a higher high while SRSI %K makes a lower high → Bearish divergence
        - Adds conviction to the bearish cross signal

        BULLISH SIGNAL (REVERSE / COUNTER-TREND LONG):
        -----------------------------------------------
        Mirror setup: %K crosses ABOVE %D while both are above a rising RSI,
        in oversold zone or after a failed decline — indicates bullish reversal.

        EXIT SIGNALS:
        -------------
        - Stop loss hit (ATR-based)
        - Profit target hit (ATR-based)
        - Trailing stop hit (if enabled)
        - SRSI reversal: %K crosses back above %D (for short) or below %D (for long)
        - End-of-day square-off

        PARAMETERS:
        -----------
        rsi_period              : RSI lookback period (default: 14)
        stoch_period            : Stochastic window applied on RSI (default: 14)
        stoch_smooth_k          : Smoothing for %K line (default: 3)
        stoch_smooth_d          : Smoothing for %D line (default: 3)
        srsi_overbought         : SRSI overbought level (default: 80)
        srsi_oversold           : SRSI oversold level (default: 20)
        rsi_declining_threshold : RSI level below which it's considered declining (default: 55)
        rsi_neutral_zone_low    : Lower bound of neutral failed-rally zone (default: 45)
        rsi_neutral_zone_high   : Upper bound of neutral failed-rally zone (default: 65)
        divergence_lookback     : Bars to look back for bearish divergence (default: 10)
        atr_period              : ATR period (default: 14)
        stop_loss_atr_mult      : Stop loss in ATR multiples (default: 1.5)
        target_atr_mult         : Target in ATR multiples (default: 3.0)
        use_trailing_stop       : Enable trailing stop (default: False)
        trailing_stop_atr_mult  : Trailing stop in ATR multiples (default: 2.0)
        volume_threshold_mult   : Volume multiplier for confirmation (default: 1.2)
        """
        # Fyers client
        self.fyers = fyersModel.FyersModel(
            client_id=fyers_client_id,
            token=fyers_access_token,
            is_async=False,
            log_path=""
        )

        # Date range
        self.backtest_days = backtest_days
        ist_tz = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist_tz)
        start_date = end_date - timedelta(days=backtest_days)
        self.range_from = int(start_date.timestamp())
        self.range_to = int(end_date.timestamp())

        # Indicator parameters
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.stoch_smooth_k = stoch_smooth_k
        self.stoch_smooth_d = stoch_smooth_d
        self.srsi_overbought = srsi_overbought
        self.srsi_oversold = srsi_oversold
        self.rsi_declining_threshold = rsi_declining_threshold
        self.rsi_neutral_zone_low = rsi_neutral_zone_low
        self.rsi_neutral_zone_high = rsi_neutral_zone_high
        self.divergence_lookback = divergence_lookback

        # ATR / risk management
        self.atr_period = atr_period
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.target_atr_mult = target_atr_mult
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_atr_mult = trailing_stop_atr_mult

        # Volume filter
        self.volume_threshold_mult = volume_threshold_mult

        # Trading rules
        self.initial_capital = initial_capital
        self.square_off_time = self._parse_time(square_off_time)
        self.last_entry_time = self._parse_time(last_entry_time)
        self.tick_interval = tick_interval
        self.min_data_points = min_data_points
        self.max_trades_per_day = max_trades_per_day
        self.min_risk_reward = min_risk_reward

        # Position filter: True=long only, False=short only, None=both
        self.long_only = long_only

        # Results
        self.results = {}
        self.combined_data = {}
        self.ist_tz = ist_tz

        print(f"{'='*100}")
        print(f"REVERSE STOCHASTIC RSI (SRSI) CROSS-UNDER STRATEGY - INTRADAY TRADING (FYERS DATA)")
        print(f"{'='*100}")
        print(f"Data Source        : Fyers API")
        print(f"Backtest Period    : Last {backtest_days} days")
        print(f"Date Range         : {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"\nIndicator Parameters:")
        print(f"  RSI Period       : {rsi_period}")
        print(f"  Stoch Period     : {stoch_period}")
        print(f"  Stoch Smooth %K  : {stoch_smooth_k}")
        print(f"  Stoch Smooth %D  : {stoch_smooth_d}")
        print(f"  SRSI Overbought  : {srsi_overbought}")
        print(f"  SRSI Oversold    : {srsi_oversold}")
        print(f"  RSI Declining Threshold : {rsi_declining_threshold}")
        print(f"  RSI Neutral Zone : {rsi_neutral_zone_low} - {rsi_neutral_zone_high}")
        print(f"  Divergence Lookback : {divergence_lookback} candles")
        print(f"\nRisk Management:")
        print(f"  ATR Period       : {atr_period}")
        print(f"  Stop Loss        : {stop_loss_atr_mult}x ATR")
        print(f"  Target           : {target_atr_mult}x ATR")
        print(f"  Trailing Stop    : {'Enabled (' + str(trailing_stop_atr_mult) + 'x ATR)' if use_trailing_stop else 'Disabled'}")
        print(f"  Volume Filter    : {volume_threshold_mult}x avg volume")
        print(f"\nTrading Rules:")
        print(f"  Tick Interval    : {tick_interval}")
        print(f"  Max Trades/Day   : {max_trades_per_day}")
        print(f"  Min Risk-Reward  : {min_risk_reward}:1")
        print(f"  Last Entry       : {last_entry_time}")
        print(f"  Square-off       : {square_off_time}")
        print(f"  Initial Capital  : ₹{initial_capital:,}")
        if long_only is True:
            print(f"  Position Filter  : LONG ONLY")
        elif long_only is False:
            print(f"  Position Filter  : SHORT ONLY")
        else:
            print(f"  Position Filter  : BOTH (Long & Short)")
        print(f"{'='*100}")

        if symbols is None:
            self.symbols = ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ", "NSE:TCS-EQ"]
            print(f"\nUsing default symbols: {self.symbols}")
        else:
            self.symbols = symbols
            print(f"\nSymbols to backtest: {len(self.symbols)}")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _parse_time(self, time_str):
        try:
            h, m = map(int, time_str.split(':'))
            return time(h, m)
        except Exception:
            return time(15, 20)

    def _is_valid_entry_time(self, ts):
        try:
            t = ts.time()
            return time(9, 15) <= t < self.last_entry_time
        except Exception:
            return False

    def _is_square_off_time(self, ts):
        try:
            return ts.time() >= self.square_off_time
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data_from_fyers(self, symbol):
        """Load historical OHLCV data from Fyers API."""
        try:
            print(f"  Fetching data from Fyers for {symbol}...")
            data = {
                "symbol": symbol,
                "resolution": self.tick_interval,
                "date_format": "0",
                "range_from": str(self.range_from),
                "range_to": str(self.range_to),
                "cont_flag": "1"
            }
            response = self.fyers.history(data=data)

            if response.get('s') != 'ok' or 'candles' not in response:
                print(f"  Error fetching data: {response.get('message', 'No candles data')}")
                return None

            candles = response['candles']
            if not candles:
                print(f"  No data available for {symbol}")
                return None

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')
            df.index = df.index.tz_localize('UTC').tz_convert(self.ist_tz)
            df.index = df.index.tz_localize(None)
            df = df.between_time('09:00', '15:30')
            df = df.dropna(subset=['close', 'high', 'low'])

            print(f"  Loaded {len(df)} candles from {df.index[0].date()} to {df.index[-1].date()}")
            return df[['open', 'high', 'low', 'close', 'volume']].copy()

        except Exception as e:
            print(f"  Error loading data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------

    def calculate_atr(self, df):
        """Average True Range."""
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = df[['high', 'prev_close']].max(axis=1) - df[['low', 'prev_close']].min(axis=1)
        df['atr'] = df['tr'].ewm(span=self.atr_period, adjust=False).mean()
        return df

    def calculate_rsi(self, series, period):
        """RSI for a given series."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_stoch_rsi(self, df):
        """
        Stochastic RSI:
          1. Compute RSI on close prices.
          2. Apply Stochastic formula over a rolling window on the RSI values.
             raw_%K = (RSI - RSI_min_N) / (RSI_max_N - RSI_min_N) * 100
          3. Smooth raw_%K with SMA(stoch_smooth_k)  → srsi_k
          4. Smooth srsi_k with SMA(stoch_smooth_d)  → srsi_d
        """
        rsi = self.calculate_rsi(df['close'], self.rsi_period)
        df['rsi'] = rsi

        rsi_min = rsi.rolling(window=self.stoch_period, min_periods=self.stoch_period).min()
        rsi_max = rsi.rolling(window=self.stoch_period, min_periods=self.stoch_period).max()

        rsi_range = rsi_max - rsi_min
        raw_k = np.where(rsi_range > 0, (rsi - rsi_min) / rsi_range * 100, 50.0)
        raw_k_series = pd.Series(raw_k, index=df.index)

        df['srsi_k'] = raw_k_series.rolling(window=self.stoch_smooth_k, min_periods=1).mean()
        df['srsi_d'] = df['srsi_k'].rolling(window=self.stoch_smooth_d, min_periods=1).mean()

        # Previous values for crossover detection
        df['srsi_k_prev'] = df['srsi_k'].shift(1)
        df['srsi_d_prev'] = df['srsi_d'].shift(1)

        # RSI slope (positive = rising, negative = falling)
        df['rsi_prev'] = df['rsi'].shift(1)
        df['rsi_slope'] = df['rsi'] - df['rsi_prev']

        # Rolling average volume for surge filter
        df['avg_volume'] = df['volume'].rolling(window=20, min_periods=5).mean()

        return df

    def detect_divergence(self, df):
        """
        Bearish divergence: price makes a higher high while SRSI %K makes a lower high
        over the past divergence_lookback bars — adds conviction to a bearish cross.

        Bullish divergence: price makes a lower low while SRSI %K makes a higher low.
        """
        lb = self.divergence_lookback
        price_highs = df['high'].rolling(window=lb, min_periods=lb).max()
        srsi_highs = df['srsi_k'].rolling(window=lb, min_periods=lb).max()
        price_lows = df['low'].rolling(window=lb, min_periods=lb).min()
        srsi_lows = df['srsi_k'].rolling(window=lb, min_periods=lb).min()

        # Bearish divergence: price higher high, SRSI lower high
        prev_price_high = df['high'].rolling(window=lb * 2, min_periods=lb).max().shift(lb)
        prev_srsi_high = df['srsi_k'].rolling(window=lb * 2, min_periods=lb).max().shift(lb)
        df['bearish_divergence'] = (price_highs > prev_price_high) & (srsi_highs < prev_srsi_high)

        # Bullish divergence: price lower low, SRSI higher low
        prev_price_low = df['low'].rolling(window=lb * 2, min_periods=lb).min().shift(lb)
        prev_srsi_low = df['srsi_k'].rolling(window=lb * 2, min_periods=lb).min().shift(lb)
        df['bullish_divergence'] = (price_lows < prev_price_low) & (srsi_lows > prev_srsi_low)

        return df

    def generate_signals(self, df):
        """
        Bearish (SHORT) signal — all conditions must hold:
          1. SRSI %K crosses BELOW %D  (was above, now below)
          2. Both %K and %D are below the current RSI value
          3. RSI is either:
             a. Declining (RSI < rsi_declining_threshold) OR
             b. In neutral failed-rally zone (rsi_neutral_zone_low < RSI < rsi_neutral_zone_high)
                with a negative slope
          4. Volume >= avg_volume * volume_threshold_mult
          5. Valid trading time

        Bullish (LONG) signal — mirror:
          1. SRSI %K crosses ABOVE %D
          2. Both %K and %D are above the current RSI value
          3. RSI is either rising from oversold OR in neutral with positive slope
          4. Volume surge
          5. Valid trading time
        """
        df['valid_time'] = df.index.map(self._is_valid_entry_time)

        # Volume surge
        df['volume_surge'] = df['volume'] >= (df['avg_volume'] * self.volume_threshold_mult)

        # --- BEARISH (SHORT) ---
        k_crosses_below_d = (df['srsi_k_prev'] >= df['srsi_d_prev']) & (df['srsi_k'] < df['srsi_d'])
        both_below_rsi = (df['srsi_k'] < df['rsi']) & (df['srsi_d'] < df['rsi'])
        rsi_declining = (df['rsi'] < self.rsi_declining_threshold)
        rsi_neutral_bearish = (
            (df['rsi'] >= self.rsi_neutral_zone_low) &
            (df['rsi'] <= self.rsi_neutral_zone_high) &
            (df['rsi_slope'] < 0)
        )
        rsi_context_bearish = rsi_declining | rsi_neutral_bearish

        df['short_signal'] = (
            k_crosses_below_d &
            both_below_rsi &
            rsi_context_bearish &
            df['volume_surge'] &
            df['valid_time'] &
            (~df['atr'].isna())
        )

        # --- BULLISH (LONG) ---
        k_crosses_above_d = (df['srsi_k_prev'] <= df['srsi_d_prev']) & (df['srsi_k'] > df['srsi_d'])
        both_above_rsi = (df['srsi_k'] > df['rsi']) & (df['srsi_d'] > df['rsi'])
        rsi_rising = (df['rsi'] > (100 - self.rsi_declining_threshold))
        rsi_neutral_bullish = (
            (df['rsi'] >= self.rsi_neutral_zone_low) &
            (df['rsi'] <= self.rsi_neutral_zone_high) &
            (df['rsi_slope'] > 0)
        )
        rsi_context_bullish = rsi_rising | rsi_neutral_bullish

        df['long_signal'] = (
            k_crosses_above_d &
            both_above_rsi &
            rsi_context_bullish &
            df['volume_surge'] &
            df['valid_time'] &
            (~df['atr'].isna())
        )

        # Apply position filter
        if self.long_only is True:
            df['short_signal'] = False
        elif self.long_only is False:
            df['long_signal'] = False

        return df

    # ------------------------------------------------------------------
    # Core backtesting loop
    # ------------------------------------------------------------------

    def backtest_single_symbol(self, symbol):
        """Run Reverse SRSI strategy backtest for a single symbol."""
        print(f"\n{'='*100}")
        print(f"Backtesting: {symbol}")
        print(f"{'='*100}")

        df = self.load_data_from_fyers(symbol)
        if df is None or len(df) < self.min_data_points:
            print(f"  Insufficient data for {symbol}. Skipping.")
            return None

        self.combined_data[symbol] = df.copy()

        # Build indicators
        df = self.calculate_atr(df)
        df = self.calculate_stoch_rsi(df)
        df = self.detect_divergence(df)
        df = self.generate_signals(df)

        df['is_square_off'] = df.index.map(self._is_square_off_time)
        df['trading_day'] = df.index.date

        # Trading state
        cash = self.initial_capital
        position = 0        # 0: flat, 1: long, -1: short
        entry_price = 0.0
        entry_time = None
        entry_atr = 0.0
        stop_loss = 0.0
        target = 0.0
        trailing_stop = 0.0
        trades = []
        trade_number = 0
        trades_today = {}

        for i in range(len(df)):
            row = df.iloc[i]
            current_time = df.index[i]
            current_price = row['close']
            current_high = row['high']
            current_low = row['low']
            current_atr = row['atr']
            is_sq_off = row['is_square_off']
            current_day = row['trading_day']

            if current_day not in trades_today:
                trades_today[current_day] = 0

            # ── Square-off at EOD ──────────────────────────────────────
            if position != 0 and is_sq_off:
                shares = trades[-1]['shares']
                if position == 1:
                    trade_pnl = shares * (current_price - entry_price)
                else:
                    trade_pnl = shares * (entry_price - current_price)
                trade_return = (trade_pnl / (shares * entry_price)) * 100
                trade_number += 1
                duration = (current_time - entry_time).total_seconds() / 60

                print(f"\n[Trade #{trade_number}] SQUARE-OFF @ {current_time.strftime('%H:%M:%S')}")
                print(f"  Direction : {'LONG' if position == 1 else 'SHORT'}")
                print(f"  Entry     : {entry_time.strftime('%H:%M:%S')} @ ₹{entry_price:.2f}")
                print(f"  Exit      : ₹{current_price:.2f} | P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
                print(f"  {'✅ PROFIT' if trade_pnl > 0 else '❌ LOSS'}")

                trades[-1].update({
                    'exit_time': current_time,
                    'exit_price': current_price,
                    'pnl': trade_pnl,
                    'return_pct': trade_return,
                    'exit_reason': 'SQUARE_OFF',
                    'duration_minutes': duration
                })
                position = 0
                trades_today[current_day] += 1
                continue

            # ── Entry ──────────────────────────────────────────────────
            if position == 0 and not is_sq_off:
                if trades_today[current_day] >= self.max_trades_per_day:
                    continue

                entered = False

                # SHORT entry: SRSI cross-under (bearish reversal)
                if row['short_signal']:
                    sl_price = current_price + (current_atr * self.stop_loss_atr_mult)
                    tgt_price = current_price - (current_atr * self.target_atr_mult)
                    risk = sl_price - current_price
                    reward = current_price - tgt_price
                    rr = reward / risk if risk > 0 else 0

                    if rr >= self.min_risk_reward:
                        position = -1
                        entry_price = current_price
                        entry_time = current_time
                        entry_atr = current_atr
                        stop_loss = sl_price
                        target = tgt_price
                        trailing_stop = sl_price if self.use_trailing_stop else 0.0
                        shares = int(cash / entry_price)

                        if shares > 0:
                            divergence_flag = bool(row['bearish_divergence'])
                            trades.append({
                                'trade_num': trade_number + 1,
                                'direction': 'SHORT',
                                'entry_time': entry_time,
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'target': target,
                                'shares': shares,
                                'entry_atr': entry_atr,
                                'risk_reward': rr,
                                'srsi_k': row['srsi_k'],
                                'srsi_d': row['srsi_d'],
                                'rsi_at_entry': row['rsi'],
                                'bearish_divergence': divergence_flag
                            })
                            print(f"\n[SRSI SHORT ENTRY] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"  Price  : ₹{entry_price:.2f} | Stop: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                            print(f"  Shares : {shares} | R:R = 1:{rr:.2f}")
                            print(f"  RSI    : {row['rsi']:.2f} | SRSI %K: {row['srsi_k']:.2f} | %D: {row['srsi_d']:.2f}")
                            if divergence_flag:
                                print(f"  Divergence: ✅ Bearish confirmed")
                            entered = True

                # LONG entry: SRSI cross-over (bullish reversal)
                if not entered and row['long_signal']:
                    sl_price = current_price - (current_atr * self.stop_loss_atr_mult)
                    tgt_price = current_price + (current_atr * self.target_atr_mult)
                    risk = current_price - sl_price
                    reward = tgt_price - current_price
                    rr = reward / risk if risk > 0 else 0

                    if rr >= self.min_risk_reward:
                        position = 1
                        entry_price = current_price
                        entry_time = current_time
                        entry_atr = current_atr
                        stop_loss = sl_price
                        target = tgt_price
                        trailing_stop = sl_price if self.use_trailing_stop else 0.0
                        shares = int(cash / entry_price)

                        if shares > 0:
                            divergence_flag = bool(row['bullish_divergence'])
                            trades.append({
                                'trade_num': trade_number + 1,
                                'direction': 'LONG',
                                'entry_time': entry_time,
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'target': target,
                                'shares': shares,
                                'entry_atr': entry_atr,
                                'risk_reward': rr,
                                'srsi_k': row['srsi_k'],
                                'srsi_d': row['srsi_d'],
                                'rsi_at_entry': row['rsi'],
                                'bullish_divergence': divergence_flag
                            })
                            print(f"\n[SRSI LONG ENTRY] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"  Price  : ₹{entry_price:.2f} | Stop: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                            print(f"  Shares : {shares} | R:R = 1:{rr:.2f}")
                            print(f"  RSI    : {row['rsi']:.2f} | SRSI %K: {row['srsi_k']:.2f} | %D: {row['srsi_d']:.2f}")
                            if divergence_flag:
                                print(f"  Divergence: ✅ Bullish confirmed")

            # ── Exit management ────────────────────────────────────────
            elif position != 0 and not is_sq_off:
                exit_signal = False
                exit_reason = ""
                exit_price = current_price

                if position == 1:  # Long
                    # Update trailing stop
                    if self.use_trailing_stop:
                        new_trail = current_price - (current_atr * self.trailing_stop_atr_mult)
                        trailing_stop = max(trailing_stop, new_trail)

                    if current_low <= stop_loss:
                        exit_signal, exit_reason, exit_price = True, "STOP_LOSS", stop_loss
                    elif self.use_trailing_stop and current_low <= trailing_stop:
                        exit_signal, exit_reason, exit_price = True, "TRAILING_STOP", trailing_stop
                    elif current_high >= target:
                        exit_signal, exit_reason, exit_price = True, "TARGET_HIT", target
                    # SRSI reversal exit: %K crosses back below %D
                    elif (row['srsi_k_prev'] >= row['srsi_d_prev']) and (row['srsi_k'] < row['srsi_d']):
                        exit_signal, exit_reason = True, "SRSI_REVERSAL"

                elif position == -1:  # Short
                    # Update trailing stop
                    if self.use_trailing_stop:
                        new_trail = current_price + (current_atr * self.trailing_stop_atr_mult)
                        trailing_stop = min(trailing_stop, new_trail)

                    if current_high >= stop_loss:
                        exit_signal, exit_reason, exit_price = True, "STOP_LOSS", stop_loss
                    elif self.use_trailing_stop and current_high >= trailing_stop:
                        exit_signal, exit_reason, exit_price = True, "TRAILING_STOP", trailing_stop
                    elif current_low <= target:
                        exit_signal, exit_reason, exit_price = True, "TARGET_HIT", target
                    # SRSI reversal exit: %K crosses back above %D
                    elif (row['srsi_k_prev'] <= row['srsi_d_prev']) and (row['srsi_k'] > row['srsi_d']):
                        exit_signal, exit_reason = True, "SRSI_REVERSAL"

                if exit_signal:
                    shares = trades[-1]['shares']
                    if position == 1:
                        trade_pnl = shares * (exit_price - entry_price)
                    else:
                        trade_pnl = shares * (entry_price - exit_price)
                    trade_return = (trade_pnl / (shares * entry_price)) * 100
                    trade_number += 1
                    duration = (current_time - entry_time).total_seconds() / 60

                    print(f"\n[Trade #{trade_number}] {exit_reason}")
                    print(f"  Direction : {'LONG' if position == 1 else 'SHORT'}")
                    print(f"  Entry     : {entry_time.strftime('%H:%M:%S')} @ ₹{entry_price:.2f}")
                    print(f"  Exit      : {current_time.strftime('%H:%M:%S')} @ ₹{exit_price:.2f}")
                    print(f"  Duration  : {duration:.1f} min | P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
                    print(f"  {'✅ PROFIT' if trade_pnl > 0 else '❌ LOSS'}")

                    trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'pnl': trade_pnl,
                        'return_pct': trade_return,
                        'exit_reason': exit_reason,
                        'duration_minutes': duration
                    })
                    position = 0
                    trades_today[current_day] += 1

        completed_trades = [t for t in trades if 'exit_time' in t]
        metrics = self.calculate_metrics(completed_trades)
        return {'symbol': symbol, 'data': df, 'trades': completed_trades, 'metrics': metrics}

    # ------------------------------------------------------------------
    # Metrics & reporting
    # ------------------------------------------------------------------

    def calculate_metrics(self, trades):
        if not trades:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0,
                'total_return': 0, 'avg_return': 0, 'best_trade': 0,
                'worst_trade': 0, 'avg_duration': 0, 'avg_win': 0,
                'avg_loss': 0, 'profit_factor': 0
            }

        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        losing_trades = total_trades - winning_trades
        total_pnl = sum(t['pnl'] for t in trades)
        avg_pnl = total_pnl / total_trades
        total_return = sum(t['return_pct'] for t in trades)
        avg_return = total_return / total_trades
        best_trade = max(t['pnl'] for t in trades)
        worst_trade = min(t['pnl'] for t in trades)
        avg_duration = sum(t['duration_minutes'] for t in trades) / total_trades

        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] <= 0]
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': (winning_trades / total_trades * 100),
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'total_return': total_return,
            'avg_return': avg_return,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_duration': avg_duration,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }

    def run_backtest(self):
        """Run backtest for all symbols."""
        print(f"\n{'='*100}")
        print("STARTING REVERSE STOCHASTIC RSI BACKTEST")
        print(f"{'='*100}")

        for symbol in self.symbols:
            try:
                result = self.backtest_single_symbol(symbol)
                if result:
                    self.results[symbol] = result
            except Exception as e:
                print(f"Error backtesting {symbol}: {e}")
                import traceback
                traceback.print_exc()

        self.print_summary()

    def print_summary(self):
        """Print and save summary results."""
        if not self.results:
            print("\nNo results to display.")
            return

        print(f"\n{'='*100}")
        print("REVERSE STOCHASTIC RSI STRATEGY RESULTS - ALL SYMBOLS")
        print(f"{'='*100}")

        summary_data = []
        for symbol, result in self.results.items():
            m = result['metrics']
            clean = symbol.split(':')[1].replace('-EQ', '') if ':' in symbol else symbol
            summary_data.append({
                'Symbol': clean,
                'Trades': m['total_trades'],
                'Win Rate': f"{m['win_rate']:.1f}%",
                'Total P&L': f"₹{m['total_pnl']:.2f}",
                'Avg P&L': f"₹{m['avg_pnl']:.2f}",
                'Profit Factor': f"{m['profit_factor']:.2f}",
                'Avg Duration': f"{m['avg_duration']:.1f}m",
                'Best Trade': f"₹{m['best_trade']:.2f}",
                'Worst Trade': f"₹{m['worst_trade']:.2f}"
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        print(f"\n{'='*100}")
        print("OVERALL STATISTICS")
        print(f"{'='*100}")

        total_trades = sum(r['metrics']['total_trades'] for r in self.results.values())
        total_pnl = sum(r['metrics']['total_pnl'] for r in self.results.values())
        winning_trades = sum(r['metrics']['winning_trades'] for r in self.results.values())
        profitable_symbols = sum(1 for r in self.results.values() if r['metrics']['total_pnl'] > 0)

        print(f"Symbols Tested     : {len(self.results)}")
        print(f"Total Trades       : {total_trades}")
        print(f"Total P&L          : ₹{total_pnl:.2f}")
        if total_trades > 0:
            print(f"Overall Win Rate   : {(winning_trades / total_trades * 100):.1f}%")
        print(f"Profitable Symbols : {profitable_symbols}/{len(self.results)}")

        summary_df.to_csv('reverse_srsi_backtest_results.csv', index=False)
        print(f"\n✅ Results saved to: reverse_srsi_backtest_results.csv")


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("REVERSE STOCHASTIC RSI (SRSI) CROSS-UNDER STRATEGY - INTRADAY BACKTEST WITH FYERS")
    print("=" * 100)

    FYERS_CLIENT_ID = os.environ.get('FYERS_CLIENT_ID')
    FYERS_ACCESS_TOKEN = os.environ.get('FYERS_ACCESS_TOKEN')

    if not FYERS_CLIENT_ID or not FYERS_ACCESS_TOKEN:
        print("\n❌ ERROR: Missing Fyers API credentials!")
        print("   Please set FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN in your .env file.")
        print("   Run: python main.py auth")
        exit(1)

    SYMBOLS = [
        "NSE:SBIN-EQ",
        "NSE:RELIANCE-EQ",
        "NSE:TCS-EQ",
        "NSE:INFY-EQ",
        "NSE:HDFCBANK-EQ",
        "NSE:STLTECH-EQ",
        "NSE:SKYGOLD-EQ",
        "NSE:AXISCADES-EQ",
        "BSE:SATTRIX-M",
        "NSE:AWHCL-EQ",
        "NSE:KAPSTON-EQ",
    ]

    backtester = ReverseStochasticRSIBacktester(
        fyers_client_id=FYERS_CLIENT_ID,
        fyers_access_token=FYERS_ACCESS_TOKEN,
        symbols=SYMBOLS,
        backtest_days=30,

        # Stochastic RSI parameters
        rsi_period=14,
        stoch_period=14,
        stoch_smooth_k=3,
        stoch_smooth_d=3,
        srsi_overbought=80,
        srsi_oversold=20,

        # RSI context thresholds
        rsi_declining_threshold=55,   # RSI below this is considered declining
        rsi_neutral_zone_low=45,      # Neutral zone lower bound
        rsi_neutral_zone_high=65,     # Neutral zone upper bound

        # Divergence lookback
        divergence_lookback=10,

        # Risk management
        atr_period=14,
        stop_loss_atr_mult=1.5,
        target_atr_mult=3.0,
        use_trailing_stop=True,
        trailing_stop_atr_mult=2.0,

        # Volume filter
        volume_threshold_mult=1.2,

        # Trading rules
        initial_capital=100000,
        square_off_time="15:20",
        tick_interval='30S',
        last_entry_time="14:30",
        max_trades_per_day=3,
        min_risk_reward=1.5,

        # Position filter: True=long only, False=short only, None=both
        long_only=None
    )

    backtester.run_backtest()

    print("\n" + "=" * 100)
    print("BACKTEST COMPLETE")
    print("=" * 100)
