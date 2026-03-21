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


class PivotPointBounceBacktester:
    def __init__(self, fyers_client_id, fyers_access_token, symbols=None,
                 bounce_confirmation_pct=0.1, volume_threshold_mult=1.2,
                 atr_period=14, rsi_period=14, rsi_oversold=40, rsi_overbought=60,
                 stop_loss_atr_mult=1.0, target_atr_mult=2.0,
                 use_trailing_stop=False, trailing_stop_atr_mult=1.0,
                 bounce_candles=2, pivot_touch_tolerance_pct=0.15,
                 use_r2_s2=True, initial_capital=100000,
                 square_off_time="15:20", last_entry_time="14:30",
                 max_trades_per_day=3, min_risk_reward=1.5,
                 tick_interval='5', backtest_days=7, long_only=None):
        """
        Pivot Point Bounce (Reversal) Strategy for Intraday Trading

        STRATEGY CONCEPT:
        -----------------
        Pivot Points are calculated from the previous day's High, Low, and Close
        and serve as key support/resistance levels for the current session. In a
        ranging market, price tends to "bounce" off these levels rather than
        breaking through them.

        This strategy exploits those reversals:
        - Price drops to S1 or S2 (support) → BUY expecting a bounce up
        - Price rises to R1 or R2 (resistance) → SELL expecting a drop down

        PIVOT POINT FORMULAS (Classic / Floor Trader Method):
        -----------------------------------------------------
        PP  = (Prev High + Prev Low + Prev Close) / 3
        R1  = 2 * PP - Prev Low
        R2  = PP + (Prev High - Prev Low)
        R3  = Prev High + 2 * (PP - Prev Low)
        S1  = 2 * PP - Prev High
        S2  = PP - (Prev High - Prev Low)
        S3  = Prev Low - 2 * (Prev High - PP)

        KEY COMPONENTS:
        ---------------
        1. Previous-Day OHLC Aggregation
           - Aggregate intraday bars to get each day's H/L/C
           - Use these to compute pivot levels for the following session

        2. Level Touch Detection
           - "Touch zone" around each pivot: ± pivot_touch_tolerance_pct %
           - A candle is considered to have touched a level when low ≤ level + zone
             (for support) or high ≥ level - zone (for resistance)

        3. Bounce / Rejection Confirmation
           - Support bounce: close > level (candle closes back above support)
           - Resistance rejection: close < level (candle closes back below resistance)
           - Requires bounce_candles consecutive closes in the reversal direction
           - Volume surge confirms conviction

        4. RSI Filter
           - Long at support: RSI < rsi_oversold → confirms oversold / reversal up
           - Short at resistance: RSI > rsi_overbought → confirms overbought / reversal down

        ENTRY SIGNALS:
        --------------
        LONG (buy the bounce):
        - Price touches S1 or S2 (within tolerance)
        - RSI below rsi_oversold threshold
        - Close forms a bullish reversal (close > touched support level)
        - Volume surge
        - Valid trading time and trade count

        SHORT (sell the rejection):
        - Price touches R1 or R2 (within tolerance)
        - RSI above rsi_overbought threshold
        - Close forms a bearish reversal (close < touched resistance level)
        - Volume surge
        - Valid trading time and trade count

        EXIT SIGNALS:
        -------------
        LONG exits:
        - Stop loss: low < entry - stop_loss_atr_mult * ATR
        - Target: high ≥ next pivot above entry (PP or R1)
        - Trailing stop (optional)
        - End-of-day square-off

        SHORT exits:
        - Stop loss: high > entry + stop_loss_atr_mult * ATR
        - Target: low ≤ next pivot below entry (PP or S1)
        - Trailing stop (optional)
        - End-of-day square-off

        PARAMETERS:
        -----------
        bounce_confirmation_pct  : Min % close above/below level to confirm bounce (default 0.1)
        volume_threshold_mult    : Volume multiplier for surge detection (default 1.2)
        atr_period               : ATR period (default 14)
        rsi_period               : RSI period (default 14)
        rsi_oversold             : RSI threshold for long entries (default 40)
        rsi_overbought           : RSI threshold for short entries (default 60)
        stop_loss_atr_mult       : Stop-loss distance in ATR multiples (default 1.0)
        target_atr_mult          : Target distance in ATR multiples (default 2.0)
        use_trailing_stop        : Enable trailing stop (default False)
        trailing_stop_atr_mult   : Trailing stop in ATR (default 1.0)
        bounce_candles           : Consecutive confirming candles needed (default 2)
        pivot_touch_tolerance_pct: Tolerance band around pivot level in % (default 0.15)
        use_r2_s2                : Also trade R2/S2 in addition to R1/S1 (default True)
        last_entry_time          : No new entries after this time (default "14:30")
        max_trades_per_day       : Max trades per symbol per day (default 3)
        min_risk_reward          : Minimum R:R ratio required (default 1.5)
        """
        self.fyers = fyersModel.FyersModel(
            client_id=fyers_client_id, token=fyers_access_token,
            is_async=False, log_path=""
        )

        # Date range
        self.backtest_days = backtest_days
        ist_tz = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist_tz)
        # Fetch extra days so we have prev-day data for the first trading day
        start_date = end_date - timedelta(days=backtest_days + 5)
        self.range_from = int(start_date.timestamp())
        self.range_to = int(end_date.timestamp())

        # Pivot parameters
        self.bounce_confirmation_pct = bounce_confirmation_pct / 100
        self.volume_threshold_mult = volume_threshold_mult
        self.pivot_touch_tolerance = pivot_touch_tolerance_pct / 100
        self.bounce_candles = bounce_candles
        self.use_r2_s2 = use_r2_s2

        # Technical indicators
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        # Risk management
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.target_atr_mult = target_atr_mult
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.min_risk_reward = min_risk_reward

        # Trading parameters
        self.initial_capital = initial_capital
        self.square_off_time = self._parse_time(square_off_time)
        self.last_entry_time = self._parse_time(last_entry_time)
        self.max_trades_per_day = max_trades_per_day
        self.tick_interval = tick_interval
        self.long_only = long_only

        # Storage
        self.results = {}
        self.combined_data = {}
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print(f"{'='*100}")
        print(f"PIVOT POINT BOUNCE (REVERSAL) STRATEGY - INTRADAY TRADING (FYERS DATA)")
        print(f"{'='*100}")
        print(f"Backtest Period: Last {backtest_days} days")
        print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"\nStrategy Parameters:")
        print(f"  Tick Interval: {tick_interval}")
        print(f"  Pivot Touch Tolerance: ±{pivot_touch_tolerance_pct}%")
        print(f"  Bounce Confirmation: {bounce_confirmation_pct}% close beyond level, {bounce_candles} candles")
        print(f"  Volume Surge: {volume_threshold_mult}x average")
        print(f"  Trade Levels: R1/S1" + (" + R2/S2" if use_r2_s2 else ""))
        print(f"  RSI Period: {rsi_period} | Oversold: {rsi_oversold} | Overbought: {rsi_overbought}")
        print(f"  Stop Loss: {stop_loss_atr_mult}x ATR | Target: {target_atr_mult}x ATR")
        print(f"  Trailing Stop: {'Enabled (' + str(trailing_stop_atr_mult) + 'x ATR)' if use_trailing_stop else 'Disabled'}")
        print(f"  Max Trades/Day: {max_trades_per_day} | Min R:R: {min_risk_reward}")
        print(f"  Last Entry: {last_entry_time} | Square-off: {square_off_time}")
        print(f"  Position Filter: {'LONG ONLY' if long_only is True else 'SHORT ONLY' if long_only is False else 'BOTH'}")
        print(f"  Initial Capital: ₹{initial_capital:,}")
        print(f"{'='*100}")

        if symbols is None:
            self.symbols = ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ", "NSE:TCS-EQ"]
            print(f"\nUsing default symbols: {self.symbols}")
        else:
            self.symbols = symbols
            print(f"\nSymbols to backtest: {len(self.symbols)}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_time(self, time_str):
        try:
            h, m = map(int, time_str.split(':'))
            return time(h, m)
        except Exception:
            return time(15, 20)

    def _is_valid_entry_time(self, ts):
        t = ts.time()
        return time(9, 16) <= t < self.last_entry_time

    def _is_square_off_time(self, ts):
        return ts.time() >= self.square_off_time

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def load_data_from_fyers(self, symbol):
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
    # Technical Indicators
    # ------------------------------------------------------------------

    def calculate_atr(self, df):
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = df[['high', 'prev_close']].max(axis=1) - df[['low', 'prev_close']].min(axis=1)
        df['atr'] = df['tr'].ewm(span=self.atr_period, adjust=False).mean()
        return df

    def calculate_rsi(self, df):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    # ------------------------------------------------------------------
    # Pivot Point Calculation
    # ------------------------------------------------------------------

    def calculate_pivot_points(self, df):
        """
        Calculate classic floor-trader pivot points from the previous day's OHLC.

        Steps:
        1. Extract each calendar day's H/L/C from the intraday data.
        2. For every intraday candle, look up the *previous* day's H/L/C.
        3. Compute PP, R1, R2, R3, S1, S2, S3 and attach them to each row.
        """
        df['trading_day'] = df.index.date

        # Daily OHLC from intraday bars
        daily = df.groupby('trading_day').agg(
            day_high=('high', 'max'),
            day_low=('low', 'min'),
            day_close=('close', 'last')
        ).reset_index()
        daily = daily.sort_values('trading_day').reset_index(drop=True)

        # Shift to get previous day's values
        daily['prev_high'] = daily['day_high'].shift(1)
        daily['prev_low'] = daily['day_low'].shift(1)
        daily['prev_close'] = daily['day_close'].shift(1)

        # Pivot formulas
        daily['pp'] = (daily['prev_high'] + daily['prev_low'] + daily['prev_close']) / 3
        daily['r1'] = 2 * daily['pp'] - daily['prev_low']
        daily['r2'] = daily['pp'] + (daily['prev_high'] - daily['prev_low'])
        daily['r3'] = daily['prev_high'] + 2 * (daily['pp'] - daily['prev_low'])
        daily['s1'] = 2 * daily['pp'] - daily['prev_high']
        daily['s2'] = daily['pp'] - (daily['prev_high'] - daily['prev_low'])
        daily['s3'] = daily['prev_low'] - 2 * (daily['prev_high'] - daily['pp'])

        # Map back to intraday rows
        day_map = daily.set_index('trading_day')[['pp', 'r1', 'r2', 'r3', 's1', 's2', 's3']]
        for col in ['pp', 'r1', 'r2', 'r3', 's1', 's2', 's3']:
            df[col] = df['trading_day'].map(day_map[col])

        # Average daily volume for volume-surge filter
        daily_vol = df.groupby('trading_day')['volume'].mean().rename('avg_day_volume')
        df['avg_day_volume'] = df['trading_day'].map(daily_vol)

        return df

    # ------------------------------------------------------------------
    # Signal Generation
    # ------------------------------------------------------------------

    def generate_signals(self, df):
        """
        Mark support-bounce and resistance-rejection signals.

        Support bounce (long):
        - Price low ≤ S1 (or S2) + tolerance band
        - Close > S1 (or S2) + bounce_confirmation_pct  → closed back above support
        - RSI < rsi_oversold
        - Volume surge

        Resistance rejection (short):
        - Price high ≥ R1 (or R2) - tolerance band
        - Close < R1 (or R2) - bounce_confirmation_pct  → closed back below resistance
        - RSI > rsi_overbought
        - Volume surge
        """
        tol = self.pivot_touch_tolerance
        conf = self.bounce_confirmation_pct
        vol_mult = self.volume_threshold_mult

        # Pivot levels to check
        support_levels = ['s1', 's2'] if self.use_r2_s2 else ['s1']
        resist_levels = ['r1', 'r2'] if self.use_r2_s2 else ['r1']

        df['volume_surge'] = df['volume'] > df['avg_day_volume'] * vol_mult

        # ---- Support Bounce (Long) ----
        long_cond = pd.Series(False, index=df.index)
        df['touched_support'] = np.nan
        for lvl in support_levels:
            touch = df['low'] <= df[lvl] * (1 + tol)
            confirm = df['close'] > df[lvl] * (1 + conf)
            lvl_cond = touch & confirm
            # record which level was touched (first match wins)
            df.loc[lvl_cond & df['touched_support'].isna(), 'touched_support'] = df.loc[lvl_cond & df['touched_support'].isna(), lvl]
            long_cond = long_cond | lvl_cond

        df['long_signal'] = (
            long_cond &
            (df['rsi'] < self.rsi_oversold) &
            df['volume_surge'] &
            (~df['pp'].isna()) &
            (~df['atr'].isna())
        )
        if self.long_only is False:
            df['long_signal'] = False

        # ---- Resistance Rejection (Short) ----
        short_cond = pd.Series(False, index=df.index)
        df['touched_resistance'] = np.nan
        for lvl in resist_levels:
            touch = df['high'] >= df[lvl] * (1 - tol)
            confirm = df['close'] < df[lvl] * (1 - conf)
            lvl_cond = touch & confirm
            df.loc[lvl_cond & df['touched_resistance'].isna(), 'touched_resistance'] = df.loc[lvl_cond & df['touched_resistance'].isna(), lvl]
            short_cond = short_cond | lvl_cond

        df['short_signal'] = (
            short_cond &
            (df['rsi'] > self.rsi_overbought) &
            df['volume_surge'] &
            (~df['pp'].isna()) &
            (~df['atr'].isna())
        )
        if self.long_only is True:
            df['short_signal'] = False

        return df

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def backtest_single_symbol(self, symbol):
        print(f"\n{'='*100}")
        print(f"Backtesting: {symbol}")
        print(f"{'='*100}")

        df = self.load_data_from_fyers(symbol)
        if df is None or len(df) < self.atr_period * 3:
            print(f"  Insufficient data for {symbol}. Skipping.")
            return None

        self.combined_data[symbol] = df.copy()

        df = self.calculate_atr(df)
        df = self.calculate_rsi(df)
        df = self.calculate_pivot_points(df)
        df = self.generate_signals(df)

        # Trading state
        cash = self.initial_capital
        position = 0           # 0: flat, 1: long, -1: short
        entry_price = 0.0
        stop_loss = 0.0
        target = 0.0
        trailing_stop = 0.0
        entry_time = None
        entry_atr = 0.0
        trades = []
        trade_number = 0
        trades_today = {}

        # Bounce confirmation tracking
        bounce_pending = False
        bounce_direction = 0
        bounce_count = 0
        pending_info = {}

        for i in range(len(df)):
            row = df.iloc[i]
            current_time = df.index[i]
            current_price = row['close']
            current_high = row['high']
            current_low = row['low']
            current_atr = row['atr']
            current_day = row['trading_day']
            is_sq_off = self._is_square_off_time(current_time)

            trades_today.setdefault(current_day, 0)

            # ---- Square-off ----
            if position != 0 and is_sq_off:
                shares = trades[-1]['shares']
                trade_pnl = shares * (current_price - entry_price) if position == 1 \
                    else shares * (entry_price - current_price)
                trade_return = (trade_pnl / (shares * entry_price)) * 100
                trade_number += 1
                duration = (current_time - entry_time).total_seconds() / 60

                print(f"\n[Trade #{trade_number}] SQUARE-OFF @ {current_time.strftime('%H:%M:%S')}")
                print(f"  {'LONG' if position == 1 else 'SHORT'} | Entry: {entry_time.strftime('%H:%M:%S')} @ ₹{entry_price:.2f}")
                print(f"  Exit: ₹{current_price:.2f} | P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
                print(f"  {'✅ PROFIT' if trade_pnl > 0 else '❌ LOSS'}")

                trades[-1].update({
                    'exit_time': current_time, 'exit_price': current_price,
                    'pnl': trade_pnl, 'return_pct': trade_return,
                    'exit_reason': 'SQUARE_OFF', 'duration_minutes': duration
                })
                position = 0
                trades_today[current_day] += 1
                bounce_pending = False

            # ---- Bounce confirmation pending ----
            elif bounce_pending:
                bounce_count += 1

                if bounce_direction == 1:  # Expecting bounce up
                    # Invalidate if price closes below the support level
                    if current_price < pending_info['level'] * (1 - self.bounce_confirmation_pct):
                        bounce_pending = False
                        bounce_count = 0
                    elif bounce_count >= self.bounce_candles:
                        bounce_pending = False
                        bounce_count = 0
                        if trades_today[current_day] < self.max_trades_per_day \
                                and self._is_valid_entry_time(current_time):
                            sl = current_price - current_atr * self.stop_loss_atr_mult
                            tgt = current_price + current_atr * self.target_atr_mult
                            risk = current_price - sl
                            reward = tgt - current_price
                            rr = reward / risk if risk > 0 else 0

                            if rr >= self.min_risk_reward:
                                position = 1
                                entry_price = current_price
                                entry_time = current_time
                                entry_atr = current_atr
                                stop_loss = sl
                                target = tgt
                                trailing_stop = sl if self.use_trailing_stop else 0
                                shares = int(cash / entry_price)

                                if shares > 0:
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
                                        'pivot_level': pending_info['level'],
                                        'pivot_type': pending_info['type'],
                                        'pp': row['pp'], 'r1': row['r1'], 's1': row['s1']
                                    })
                                    print(f"\n[PPB LONG ENTRY] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                                    print(f"  Bounced off {pending_info['type']} @ ₹{pending_info['level']:.2f}")
                                    print(f"  Entry: ₹{entry_price:.2f} | Stop: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                                    print(f"  Shares: {shares} | R:R = 1:{rr:.2f}")

                elif bounce_direction == -1:  # Expecting drop from resistance
                    if current_price > pending_info['level'] * (1 + self.bounce_confirmation_pct):
                        bounce_pending = False
                        bounce_count = 0
                    elif bounce_count >= self.bounce_candles:
                        bounce_pending = False
                        bounce_count = 0
                        if trades_today[current_day] < self.max_trades_per_day \
                                and self._is_valid_entry_time(current_time):
                            sl = current_price + current_atr * self.stop_loss_atr_mult
                            tgt = current_price - current_atr * self.target_atr_mult
                            risk = sl - current_price
                            reward = current_price - tgt
                            rr = reward / risk if risk > 0 else 0

                            if rr >= self.min_risk_reward:
                                position = -1
                                entry_price = current_price
                                entry_time = current_time
                                entry_atr = current_atr
                                stop_loss = sl
                                target = tgt
                                trailing_stop = sl if self.use_trailing_stop else 0
                                shares = int(cash / entry_price)

                                if shares > 0:
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
                                        'pivot_level': pending_info['level'],
                                        'pivot_type': pending_info['type'],
                                        'pp': row['pp'], 'r1': row['r1'], 's1': row['s1']
                                    })
                                    print(f"\n[PPB SHORT ENTRY] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                                    print(f"  Rejected at {pending_info['type']} @ ₹{pending_info['level']:.2f}")
                                    print(f"  Entry: ₹{entry_price:.2f} | Stop: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                                    print(f"  Shares: {shares} | R:R = 1:{rr:.2f}")

            # ---- New signal ----
            elif position == 0 and not is_sq_off:
                if trades_today[current_day] >= self.max_trades_per_day:
                    continue
                if not self._is_valid_entry_time(current_time):
                    continue

                if row['long_signal']:
                    bounce_pending = True
                    bounce_direction = 1
                    bounce_count = 0
                    # Which support was touched?
                    for lvl_name in (['s1', 's2'] if self.use_r2_s2 else ['s1']):
                        lvl_val = row[lvl_name]
                        if not np.isnan(lvl_val) and row['low'] <= lvl_val * (1 + self.pivot_touch_tolerance):
                            pending_info = {'level': lvl_val, 'type': lvl_name.upper()}
                            break
                    else:
                        bounce_pending = False

                elif row['short_signal']:
                    bounce_pending = True
                    bounce_direction = -1
                    bounce_count = 0
                    for lvl_name in (['r1', 'r2'] if self.use_r2_s2 else ['r1']):
                        lvl_val = row[lvl_name]
                        if not np.isnan(lvl_val) and row['high'] >= lvl_val * (1 - self.pivot_touch_tolerance):
                            pending_info = {'level': lvl_val, 'type': lvl_name.upper()}
                            break
                    else:
                        bounce_pending = False

            # ---- Exit management ----
            elif position != 0 and not is_sq_off:
                exit_signal = False
                exit_reason = ""
                exit_price = current_price

                if position == 1:
                    if self.use_trailing_stop:
                        new_ts = current_price - current_atr * self.trailing_stop_atr_mult
                        trailing_stop = max(trailing_stop, new_ts)
                    if current_low <= stop_loss:
                        exit_signal, exit_reason, exit_price = True, "STOP_LOSS", stop_loss
                    elif self.use_trailing_stop and current_low <= trailing_stop:
                        exit_signal, exit_reason, exit_price = True, "TRAILING_STOP", trailing_stop
                    elif current_high >= target:
                        exit_signal, exit_reason, exit_price = True, "TARGET_HIT", target

                elif position == -1:
                    if self.use_trailing_stop:
                        new_ts = current_price + current_atr * self.trailing_stop_atr_mult
                        trailing_stop = min(trailing_stop, new_ts)
                    if current_high >= stop_loss:
                        exit_signal, exit_reason, exit_price = True, "STOP_LOSS", stop_loss
                    elif self.use_trailing_stop and current_high >= trailing_stop:
                        exit_signal, exit_reason, exit_price = True, "TRAILING_STOP", trailing_stop
                    elif current_low <= target:
                        exit_signal, exit_reason, exit_price = True, "TARGET_HIT", target

                if exit_signal:
                    shares = trades[-1]['shares']
                    trade_pnl = shares * (exit_price - entry_price) if position == 1 \
                        else shares * (entry_price - exit_price)
                    trade_return = (trade_pnl / (shares * entry_price)) * 100
                    trade_number += 1
                    duration = (current_time - entry_time).total_seconds() / 60

                    print(f"\n[Trade #{trade_number}] {exit_reason}")
                    print(f"  {'LONG' if position == 1 else 'SHORT'}")
                    print(f"  Entry: {entry_time.strftime('%H:%M:%S')} @ ₹{entry_price:.2f}")
                    print(f"  Exit:  {current_time.strftime('%H:%M:%S')} @ ₹{exit_price:.2f}")
                    print(f"  Duration: {duration:.1f} min | P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
                    print(f"  {'✅ PROFIT' if trade_pnl > 0 else '❌ LOSS'}")

                    trades[-1].update({
                        'exit_time': current_time, 'exit_price': exit_price,
                        'pnl': trade_pnl, 'return_pct': trade_return,
                        'exit_reason': exit_reason, 'duration_minutes': duration
                    })
                    position = 0
                    trades_today[current_day] += 1

        completed_trades = [t for t in trades if 'exit_time' in t]
        metrics = self.calculate_metrics(completed_trades)
        return {'symbol': symbol, 'data': df, 'trades': completed_trades, 'metrics': metrics}

    # ------------------------------------------------------------------
    # Metrics
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

        total = len(trades)
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] <= 0]
        total_pnl = sum(t['pnl'] for t in trades)

        return {
            'total_trades': total,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / total * 100,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / total,
            'total_return': sum(t['return_pct'] for t in trades),
            'avg_return': sum(t['return_pct'] for t in trades) / total,
            'best_trade': max(t['pnl'] for t in trades),
            'worst_trade': min(t['pnl'] for t in trades),
            'avg_duration': sum(t['duration_minutes'] for t in trades) / total,
            'avg_win': sum(wins) / len(wins) if wins else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'profit_factor': sum(wins) / abs(sum(losses)) if losses else 0
        }

    # ------------------------------------------------------------------
    # Run & Summary
    # ------------------------------------------------------------------

    def run_backtest(self):
        print(f"\n{'='*100}")
        print("STARTING PIVOT POINT BOUNCE (REVERSAL) BACKTEST")
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
        if not self.results:
            print("\nNo results to display.")
            return

        print(f"\n{'='*100}")
        print("PIVOT POINT BOUNCE (REVERSAL) STRATEGY RESULTS - ALL SYMBOLS")
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

        print(pd.DataFrame(summary_data).to_string(index=False))

        print(f"\n{'='*100}")
        print("OVERALL STATISTICS")
        print(f"{'='*100}")

        total_trades = sum(r['metrics']['total_trades'] for r in self.results.values())
        total_pnl = sum(r['metrics']['total_pnl'] for r in self.results.values())
        winning = sum(r['metrics']['winning_trades'] for r in self.results.values())
        profitable = sum(1 for r in self.results.values() if r['metrics']['total_pnl'] > 0)

        print(f"Symbols Tested: {len(self.results)}")
        print(f"Total Trades: {total_trades}")
        print(f"Total P&L: ₹{total_pnl:.2f}")
        if total_trades > 0:
            print(f"Overall Win Rate: {winning / total_trades * 100:.1f}%")
        print(f"Profitable Symbols: {profitable}/{len(self.results)}")

        pd.DataFrame(summary_data).to_csv('output/pivot_bounce_backtest_results.csv', index=False)
        print(f"\n✅ Results saved to: output/pivot_bounce_backtest_results.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("PIVOT POINT BOUNCE (REVERSAL) STRATEGY - INTRADAY BACKTEST WITH FYERS")
    print("=" * 100)

    FYERS_CLIENT_ID = os.environ.get('FYERS_CLIENT_ID')
    FYERS_ACCESS_TOKEN = os.environ.get('FYERS_ACCESS_TOKEN')

    if not FYERS_CLIENT_ID or not FYERS_ACCESS_TOKEN:
        print("\n❌ ERROR: Missing Fyers API credentials!")
        print("   Please set both FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN environment variables")
        print("\n   Step 1: Create a .env file in the project root")
        print("   Step 2: Add your credentials:")
        print("           FYERS_CLIENT_ID=your_client_id_here")
        print("           FYERS_ACCESS_TOKEN=your_access_token_here")
        print("\n   How to get credentials:")
        print("   1. Login to: https://myapi.fyers.in/dashboard/")
        print("   2. Create an app to get your CLIENT_ID")
        print("   3. Run: python main.py auth")
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
    ]

    backtester = PivotPointBounceBacktester(
        fyers_client_id=FYERS_CLIENT_ID,
        fyers_access_token=FYERS_ACCESS_TOKEN,
        symbols=SYMBOLS,
        backtest_days=30,

        # Pivot touch and bounce parameters
        pivot_touch_tolerance_pct=0.15,   # ±0.15% tolerance band around pivot level
        bounce_confirmation_pct=0.1,      # 0.1% close beyond level to confirm bounce
        bounce_candles=2,                 # 2 confirming candles before entry
        use_r2_s2=True,                   # Also trade R2/S2 levels

        # Volume filter
        volume_threshold_mult=1.2,        # 1.2x average daily volume

        # Technical indicators
        atr_period=14,
        rsi_period=14,
        rsi_oversold=40,                  # More sensitive: RSI < 40 for longs
        rsi_overbought=60,                # More sensitive: RSI > 60 for shorts

        # Risk management
        stop_loss_atr_mult=1.0,
        target_atr_mult=2.0,
        use_trailing_stop=True,
        trailing_stop_atr_mult=1.0,

        # Trading rules
        initial_capital=100000,
        square_off_time="15:20",
        tick_interval='5S',                # 5-Seconds bars
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
