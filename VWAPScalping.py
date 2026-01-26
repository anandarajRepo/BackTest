import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, time, timedelta
import pytz
import warnings
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class VWAPScalpingBacktester:
    def __init__(self, fyers_client_id, fyers_access_token, symbols=None,
                 vwap_band_mult=1.5, atr_period=14,
                 rsi_period=14, rsi_oversold=35, rsi_overbought=65,
                 volume_threshold_mult=1.2, pullback_tolerance_pct=0.1,
                 stop_loss_atr_mult=0.75, target_atr_mult=1.5,
                 trailing_stop_atr_mult=0.5, use_trailing_stop=True,
                 use_rsi_filter=True, use_volume_filter=True,
                 initial_capital=100000, square_off_time="15:20",
                 min_data_points=100, tick_interval='5',
                 last_entry_time="14:30", max_trades_per_day=5,
                 min_risk_reward=1.5, scalp_mode='pullback',
                 vwap_trend_candles=5, backtest_days=7):
        """
        VWAP Scalping Strategy for Intraday Trading

        STRATEGY CONCEPT:
        ----------------
        VWAP (Volume Weighted Average Price) is the benchmark price that represents
        the average price a security has traded at throughout the day, weighted by volume.
        Institutional traders use VWAP to ensure they execute trades at favorable prices.

        This scalping strategy trades pullbacks to VWAP during trending conditions,
        capturing quick profits from price mean-reversion to this key level.

        KEY COMPONENTS:
        ---------------
        1. VWAP Calculation
           - Cumulative (Price * Volume) / Cumulative Volume
           - Resets daily at market open
           - Dynamic throughout the day

        2. VWAP Bands (Standard Deviation)
           - Upper Band: VWAP + (StdDev * Multiplier)
           - Lower Band: VWAP - (StdDev * Multiplier)
           - Acts as dynamic support/resistance

        3. Trend Identification
           - Price above VWAP = Bullish bias
           - Price below VWAP = Bearish bias
           - Trend strength measured by distance from VWAP

        4. Entry Modes
           a) Pullback Mode (Default):
              - LONG: Price in uptrend pulls back to VWAP
              - SHORT: Price in downtrend rallies back to VWAP
           b) Band Bounce Mode:
              - LONG: Price bounces from lower VWAP band
              - SHORT: Price rejects from upper VWAP band
           c) VWAP Cross Mode:
              - LONG: Price crosses above VWAP with momentum
              - SHORT: Price crosses below VWAP with momentum

        ENTRY SIGNALS:
        --------------
        LONG ENTRY (Pullback Mode):
        - Price has been above VWAP (bullish trend)
        - Price pulls back to touch/near VWAP
        - Volume confirmation present
        - RSI not overbought
        - Time within trading window

        SHORT ENTRY (Pullback Mode):
        - Price has been below VWAP (bearish trend)
        - Price rallies back to touch/near VWAP
        - Volume confirmation present
        - RSI not oversold
        - Time within trading window

        EXIT SIGNALS:
        -------------
        - Stop loss hit (ATR-based, tight for scalping)
        - Profit target hit (ATR-based, quick scalp targets)
        - Trailing stop hit (if enabled)
        - Price crosses VWAP against position
        - End of day square-off

        PARAMETERS:
        -----------
        - vwap_band_mult: VWAP band multiplier (default: 1.5)
        - atr_period: ATR calculation period (default: 14)
        - rsi_period: RSI calculation period (default: 14)
        - rsi_oversold: RSI oversold level for shorts (default: 35)
        - rsi_overbought: RSI overbought level for longs (default: 65)
        - volume_threshold_mult: Volume multiplier (default: 1.2x)
        - pullback_tolerance_pct: % from VWAP for entry (default: 0.1%)
        - stop_loss_atr_mult: Stop loss in ATR (default: 0.75 - tight for scalping)
        - target_atr_mult: Target in ATR (default: 1.5)
        - trailing_stop_atr_mult: Trailing stop in ATR (default: 0.5)
        - use_trailing_stop: Enable trailing stops (default: True)
        - scalp_mode: 'pullback', 'band_bounce', or 'vwap_cross' (default: 'pullback')
        - vwap_trend_candles: Candles to confirm trend direction (default: 5)
        """
        # Initialize Fyers client
        self.fyers = fyersModel.FyersModel(client_id=fyers_client_id, token=fyers_access_token, is_async=False, log_path="")

        # Date range for backtest (last N days)
        self.backtest_days = backtest_days
        ist_tz = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist_tz)
        start_date = end_date - timedelta(days=backtest_days)
        self.range_from = int(start_date.timestamp())
        self.range_to = int(end_date.timestamp())

        # VWAP parameters
        self.vwap_band_mult = vwap_band_mult
        self.pullback_tolerance_pct = pullback_tolerance_pct / 100
        self.scalp_mode = scalp_mode
        self.vwap_trend_candles = vwap_trend_candles

        # Technical indicator parameters
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_threshold_mult = volume_threshold_mult

        # Filters
        self.use_rsi_filter = use_rsi_filter
        self.use_volume_filter = use_volume_filter

        # ATR-based stop loss and targets (tight for scalping)
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.target_atr_mult = target_atr_mult
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.use_trailing_stop = use_trailing_stop

        # Trading parameters
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.tick_interval = tick_interval

        # Risk management
        self.last_entry_time = self.parse_square_off_time(last_entry_time)
        self.max_trades_per_day = max_trades_per_day
        self.min_risk_reward = min_risk_reward

        # Market timing
        self.market_open_time = time(9, 15)
        self.vwap_start_time = time(9, 30)  # Start trading after first 15 min

        # Results storage
        self.results = {}
        self.combined_data = {}

        # Timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print(f"{'='*100}")
        print(f"VWAP SCALPING STRATEGY - INTRADAY TRADING (FYERS DATA)")
        print(f"{'='*100}")
        print(f"Data Source: Fyers API")
        print(f"Backtest Period: Last {backtest_days} days")
        print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"\nStrategy Parameters:")
        print(f"  Tick Interval: {self.tick_interval} seconds")
        print(f"  Scalp Mode: {self.scalp_mode.upper()}")
        print(f"  VWAP Band Multiplier: {self.vwap_band_mult}")
        print(f"  Pullback Tolerance: {pullback_tolerance_pct}%")
        print(f"  Trend Confirmation: {self.vwap_trend_candles} candles")
        print(f"  Technical Indicators:")
        print(f"    - ATR Period: {self.atr_period}")
        print(f"    - RSI Period: {self.rsi_period} (Oversold: {self.rsi_oversold}, Overbought: {self.rsi_overbought})")
        print(f"    - Volume Threshold: {self.volume_threshold_mult}x")
        print(f"  Filters:")
        print(f"    - RSI Filter: {'Enabled' if use_rsi_filter else 'Disabled'}")
        print(f"    - Volume Filter: {'Enabled' if use_volume_filter else 'Disabled'}")
        print(f"  Risk Management:")
        print(f"    - Stop Loss: {self.stop_loss_atr_mult}x ATR (tight for scalping)")
        print(f"    - Target: {self.target_atr_mult}x ATR")
        print(f"    - Trailing Stop: {'Enabled' if use_trailing_stop else 'Disabled'}")
        if use_trailing_stop:
            print(f"    - Trailing Stop: {self.trailing_stop_atr_mult}x ATR")
        print(f"  Max Trades/Day: {self.max_trades_per_day}")
        print(f"  Min Risk-Reward: {self.min_risk_reward}:1")
        print(f"  Last Entry: {last_entry_time}")
        print(f"  Square-off: {square_off_time}")
        print(f"  Initial Capital: Rs.{self.initial_capital:,}")
        print(f"{'='*100}")

        # Set symbols
        if symbols is None:
            # Default symbols if none provided
            self.symbols = ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ", "NSE:TCS-EQ"]
            print(f"\nUsing default symbols: {self.symbols}")
        else:
            self.symbols = symbols
            print(f"\nSymbols to backtest: {len(self.symbols)}")

    def parse_square_off_time(self, time_str):
        """Parse time string to time object"""
        try:
            hour, minute = map(int, time_str.split(':'))
            return time(hour, minute)
        except:
            return time(15, 20)

    def load_data_from_fyers(self, symbol):
        """Load historical data from Fyers API"""
        try:
            print(f"  Fetching data from Fyers for {symbol}...")

            # Prepare data request
            data = {
                "symbol": symbol,
                "resolution": self.tick_interval,
                "date_format": "0",
                "range_from": str(self.range_from),
                "range_to": str(self.range_to),
                "cont_flag": "1"
            }

            # Fetch data from Fyers
            response = self.fyers.history(data=data)

            if response.get('s') != 'ok' or 'candles' not in response:
                print(f"  Error fetching data: {response.get('message', 'No candles data')}")
                return None

            candles = response['candles']
            if not candles or len(candles) == 0:
                print(f"  No data available for {symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')

            # Convert to IST timezone
            df.index = df.index.tz_localize('UTC').tz_convert(self.ist_tz)

            # Remove timezone info for easier handling
            df.index = df.index.tz_localize(None)

            # Filter to only include market hours (9:00 AM - 3:30 PM IST)
            df = df.between_time('09:00', '15:30')

            # Remove missing data
            df = df.dropna(subset=['close', 'high', 'low'])

            print(f"  Loaded {len(df)} candles from {df.index[0].date()} to {df.index[-1].date()}")

            return df[['open', 'high', 'low', 'close', 'volume']].copy()

        except Exception as e:
            print(f"  Error loading data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_vwap(self, df):
        """Calculate VWAP (Volume Weighted Average Price) with bands

        VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
        Typical Price = (High + Low + Close) / 3

        VWAP resets at the start of each trading day.
        """
        # Add trading day column
        df['trading_day'] = df.index.date

        # Calculate typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # Calculate VWAP components
        df['tp_volume'] = df['typical_price'] * df['volume']

        # Group by trading day and calculate cumulative VWAP
        df['cum_tp_volume'] = df.groupby('trading_day')['tp_volume'].cumsum()
        df['cum_volume'] = df.groupby('trading_day')['volume'].cumsum()

        # Calculate VWAP
        df['vwap'] = df['cum_tp_volume'] / df['cum_volume']

        # Calculate VWAP Standard Deviation for bands
        # Rolling standard deviation of typical price from VWAP
        df['vwap_diff_sq'] = (df['typical_price'] - df['vwap']) ** 2
        df['cum_vwap_diff_sq'] = df.groupby('trading_day')['vwap_diff_sq'].cumsum()
        df['candle_count'] = df.groupby('trading_day').cumcount() + 1
        df['vwap_std'] = np.sqrt(df['cum_vwap_diff_sq'] / df['candle_count'])

        # Calculate VWAP bands
        df['vwap_upper'] = df['vwap'] + (df['vwap_std'] * self.vwap_band_mult)
        df['vwap_lower'] = df['vwap'] - (df['vwap_std'] * self.vwap_band_mult)

        # Calculate distance from VWAP (percentage)
        df['vwap_distance_pct'] = (df['close'] - df['vwap']) / df['vwap'] * 100

        # Identify price position relative to VWAP
        df['above_vwap'] = df['close'] > df['vwap']
        df['below_vwap'] = df['close'] < df['vwap']

        # Calculate trend: count consecutive candles above/below VWAP
        df['trend_bullish'] = df['above_vwap'].rolling(window=self.vwap_trend_candles, min_periods=1).sum() >= self.vwap_trend_candles
        df['trend_bearish'] = df['below_vwap'].rolling(window=self.vwap_trend_candles, min_periods=1).sum() >= self.vwap_trend_candles

        return df

    def calculate_atr(self, df):
        """Calculate Average True Range"""
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        df['atr'] = df['tr'].ewm(span=self.atr_period, adjust=False).mean()

        return df

    def calculate_rsi(self, df):
        """Calculate Relative Strength Index (RSI)"""
        delta = df['close'].diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()

        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def calculate_volume_ma(self, df):
        """Calculate Volume Moving Average"""
        df['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_surge'] = df['volume'] > (df['volume_ma'] * self.volume_threshold_mult)

        return df

    def detect_pullback_signals(self, df):
        """Detect pullback entry signals to VWAP"""
        # Price touching or near VWAP
        df['near_vwap'] = abs(df['close'] - df['vwap']) / df['vwap'] <= self.pullback_tolerance_pct

        # Previous candles were away from VWAP (confirming pullback)
        df['was_above_vwap'] = df['close'].shift(1) > df['vwap'].shift(1) * (1 + self.pullback_tolerance_pct)
        df['was_below_vwap'] = df['close'].shift(1) < df['vwap'].shift(1) * (1 - self.pullback_tolerance_pct)

        # Bullish pullback: Was above VWAP, now pulling back to VWAP in uptrend
        df['bullish_pullback'] = (
            (df['trend_bullish']) &  # Overall bullish trend
            (df['near_vwap'] | df['was_above_vwap']) &  # Near VWAP or just above
            (df['low'] <= df['vwap'] * (1 + self.pullback_tolerance_pct)) &  # Low touched VWAP area
            (df['close'] >= df['vwap'])  # Closed at or above VWAP (bounce)
        )

        # Bearish pullback: Was below VWAP, now pulling back to VWAP in downtrend
        df['bearish_pullback'] = (
            (df['trend_bearish']) &  # Overall bearish trend
            (df['near_vwap'] | df['was_below_vwap']) &  # Near VWAP or just below
            (df['high'] >= df['vwap'] * (1 - self.pullback_tolerance_pct)) &  # High touched VWAP area
            (df['close'] <= df['vwap'])  # Closed at or below VWAP (rejection)
        )

        return df

    def detect_band_bounce_signals(self, df):
        """Detect band bounce entry signals"""
        # Bounce from lower band (long signal)
        df['lower_band_bounce'] = (
            (df['low'] <= df['vwap_lower']) &  # Low touched lower band
            (df['close'] > df['vwap_lower']) &  # Closed above lower band
            (df['close'] < df['vwap'])  # Still below VWAP (room to run)
        )

        # Rejection from upper band (short signal)
        df['upper_band_rejection'] = (
            (df['high'] >= df['vwap_upper']) &  # High touched upper band
            (df['close'] < df['vwap_upper']) &  # Closed below upper band
            (df['close'] > df['vwap'])  # Still above VWAP (room to fall)
        )

        return df

    def detect_vwap_cross_signals(self, df):
        """Detect VWAP crossover entry signals"""
        # Previous close
        df['prev_close_price'] = df['close'].shift(1)
        df['prev_vwap'] = df['vwap'].shift(1)

        # Bullish cross: Was below VWAP, now above
        df['vwap_cross_up'] = (
            (df['prev_close_price'] < df['prev_vwap']) &  # Was below
            (df['close'] > df['vwap']) &  # Now above
            (df['close'] - df['vwap']) / df['vwap'] > 0.001  # Meaningful cross (>0.1%)
        )

        # Bearish cross: Was above VWAP, now below
        df['vwap_cross_down'] = (
            (df['prev_close_price'] > df['prev_vwap']) &  # Was above
            (df['close'] < df['vwap']) &  # Now below
            (df['vwap'] - df['close']) / df['vwap'] > 0.001  # Meaningful cross (>0.1%)
        )

        return df

    def is_valid_trading_time(self, timestamp):
        """Check if time is valid for trading"""
        try:
            current_time = timestamp.time()

            # Must be after VWAP stabilization time (first 15 min)
            if current_time < self.vwap_start_time:
                return False

            # No new entries after last_entry_time
            if current_time >= self.last_entry_time:
                return False

            return True
        except:
            return False

    def generate_signals(self, df):
        """Generate trading signals based on selected mode"""
        # Valid trading time
        df['valid_time'] = df.index.map(self.is_valid_trading_time)

        # RSI filter conditions
        if self.use_rsi_filter:
            df['rsi_long_ok'] = df['rsi'] < self.rsi_overbought
            df['rsi_short_ok'] = df['rsi'] > self.rsi_oversold
        else:
            df['rsi_long_ok'] = True
            df['rsi_short_ok'] = True

        # Volume filter
        if self.use_volume_filter:
            df['volume_ok'] = df['volume_surge']
        else:
            df['volume_ok'] = True

        # Initialize signals
        df['long_signal'] = False
        df['short_signal'] = False

        if self.scalp_mode == 'pullback':
            df['long_signal'] = (
                (df['bullish_pullback']) &
                (df['valid_time']) &
                (df['rsi_long_ok']) &
                (df['volume_ok']) &
                (~df['atr'].isna())
            )

            df['short_signal'] = (
                (df['bearish_pullback']) &
                (df['valid_time']) &
                (df['rsi_short_ok']) &
                (df['volume_ok']) &
                (~df['atr'].isna())
            )

        elif self.scalp_mode == 'band_bounce':
            df['long_signal'] = (
                (df['lower_band_bounce']) &
                (df['valid_time']) &
                (df['rsi_long_ok']) &
                (df['volume_ok']) &
                (~df['atr'].isna())
            )

            df['short_signal'] = (
                (df['upper_band_rejection']) &
                (df['valid_time']) &
                (df['rsi_short_ok']) &
                (df['volume_ok']) &
                (~df['atr'].isna())
            )

        elif self.scalp_mode == 'vwap_cross':
            df['long_signal'] = (
                (df['vwap_cross_up']) &
                (df['valid_time']) &
                (df['rsi_long_ok']) &
                (df['volume_ok']) &
                (~df['atr'].isna())
            )

            df['short_signal'] = (
                (df['vwap_cross_down']) &
                (df['valid_time']) &
                (df['rsi_short_ok']) &
                (df['volume_ok']) &
                (~df['atr'].isna())
            )

        return df

    def is_square_off_time(self, timestamp):
        """Check if square-off time"""
        try:
            current_time = timestamp.time()
            return current_time >= self.square_off_time
        except:
            return False

    def backtest_single_symbol(self, symbol):
        """Backtest VWAP Scalping strategy for a symbol"""
        print(f"\n{'='*100}")
        print(f"Backtesting: {symbol}")
        print(f"{'='*100}")

        df = self.load_data_from_fyers(symbol)

        if df is None or len(df) < self.atr_period * 3:
            print(f"Insufficient data for {symbol}. Skipping.")
            return None

        self.combined_data[symbol] = df.copy()

        # Calculate all technical indicators
        df = self.calculate_vwap(df)
        df = self.calculate_atr(df)
        df = self.calculate_rsi(df)
        df = self.calculate_volume_ma(df)
        df = self.detect_pullback_signals(df)
        df = self.detect_band_bounce_signals(df)
        df = self.detect_vwap_cross_signals(df)
        df = self.generate_signals(df)

        df['is_square_off'] = df.index.map(self.is_square_off_time)

        # Trading variables
        cash = self.initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        stop_loss = 0
        target = 0
        trailing_stop = 0
        trades = []
        entry_time = None
        entry_atr = 0
        entry_vwap = 0
        trade_number = 0
        trades_today = {}

        # Backtest loop
        for i in range(len(df)):
            current_time = df.index[i]
            current_price = df.iloc[i]['close']
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            current_atr = df.iloc[i]['atr']
            current_vwap = df.iloc[i]['vwap']
            is_square_off = df.iloc[i]['is_square_off']
            current_day = df.iloc[i]['trading_day']

            # Initialize day counter
            if current_day not in trades_today:
                trades_today[current_day] = 0

            # Square off at EOD
            if position != 0 and is_square_off:
                shares = trades[-1]['shares']

                if position == 1:  # Long position
                    trade_pnl = shares * (current_price - entry_price)
                else:  # Short position
                    trade_pnl = shares * (entry_price - current_price)

                trade_return = (trade_pnl / (shares * entry_price)) * 100

                trade_number += 1
                duration = (current_time - entry_time).total_seconds() / 60

                print(f"\n[Trade #{trade_number}] SQUARE-OFF @ {current_time.strftime('%H:%M:%S')}")
                print(f"  Direction: {'LONG' if position == 1 else 'SHORT'}")
                print(f"  Entry: {entry_time.strftime('%H:%M:%S')} @ Rs.{entry_price:.2f}")
                print(f"  Exit: Rs.{current_price:.2f} | P&L: Rs.{trade_pnl:.2f} ({trade_return:+.2f}%)")
                print(f"  {'PROFIT' if trade_pnl > 0 else 'LOSS'}")

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

            # Entry signals
            elif position == 0 and not is_square_off:
                # Check trade frequency limit
                if trades_today[current_day] >= self.max_trades_per_day:
                    continue

                # LONG SIGNAL
                if df.iloc[i]['long_signal']:
                    # Calculate stop loss and target
                    stop_loss_price = current_price - (current_atr * self.stop_loss_atr_mult)
                    target_price = current_price + (current_atr * self.target_atr_mult)

                    risk = current_price - stop_loss_price
                    reward = target_price - current_price
                    risk_reward = reward / risk if risk > 0 else 0

                    if risk_reward >= self.min_risk_reward:
                        position = 1
                        entry_price = current_price
                        entry_time = current_time
                        entry_atr = current_atr
                        entry_vwap = current_vwap
                        stop_loss = stop_loss_price
                        target = target_price
                        trailing_stop = stop_loss_price if self.use_trailing_stop else 0

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
                                'entry_vwap': entry_vwap,
                                'risk_reward': risk_reward,
                                'scalp_mode': self.scalp_mode
                            })

                            print(f"\n[VWAP LONG ENTRY] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"  Mode: {self.scalp_mode.upper()}")
                            print(f"  Price: Rs.{entry_price:.2f} | VWAP: Rs.{current_vwap:.2f}")
                            print(f"  Stop: Rs.{stop_loss:.2f} | Target: Rs.{target:.2f}")
                            print(f"  Shares: {shares} | R:R = 1:{risk_reward:.2f}")

                # SHORT SIGNAL
                elif df.iloc[i]['short_signal']:
                    # Calculate stop loss and target
                    stop_loss_price = current_price + (current_atr * self.stop_loss_atr_mult)
                    target_price = current_price - (current_atr * self.target_atr_mult)

                    risk = stop_loss_price - current_price
                    reward = current_price - target_price
                    risk_reward = reward / risk if risk > 0 else 0

                    if risk_reward >= self.min_risk_reward:
                        position = -1
                        entry_price = current_price
                        entry_time = current_time
                        entry_atr = current_atr
                        entry_vwap = current_vwap
                        stop_loss = stop_loss_price
                        target = target_price
                        trailing_stop = stop_loss_price if self.use_trailing_stop else 0

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
                                'entry_vwap': entry_vwap,
                                'risk_reward': risk_reward,
                                'scalp_mode': self.scalp_mode
                            })

                            print(f"\n[VWAP SHORT ENTRY] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"  Mode: {self.scalp_mode.upper()}")
                            print(f"  Price: Rs.{entry_price:.2f} | VWAP: Rs.{current_vwap:.2f}")
                            print(f"  Stop: Rs.{stop_loss:.2f} | Target: Rs.{target:.2f}")
                            print(f"  Shares: {shares} | R:R = 1:{risk_reward:.2f}")

            # Exit management
            elif position != 0 and not is_square_off:
                exit_signal = False
                exit_reason = ""
                exit_price = current_price

                if position == 1:  # Long position
                    # Update trailing stop if enabled
                    if self.use_trailing_stop:
                        new_trailing = current_price - (current_atr * self.trailing_stop_atr_mult)
                        trailing_stop = max(trailing_stop, new_trailing)

                    # Check exits
                    if current_low <= stop_loss:
                        exit_signal = True
                        exit_reason = "STOP_LOSS"
                        exit_price = stop_loss
                    elif self.use_trailing_stop and current_low <= trailing_stop:
                        exit_signal = True
                        exit_reason = "TRAILING_STOP"
                        exit_price = trailing_stop
                    elif current_high >= target:
                        exit_signal = True
                        exit_reason = "TARGET_HIT"
                        exit_price = target
                    # VWAP cross exit: Price crosses below VWAP significantly
                    elif current_price < current_vwap * (1 - self.pullback_tolerance_pct * 2):
                        exit_signal = True
                        exit_reason = "VWAP_CROSS_EXIT"
                        exit_price = current_price

                elif position == -1:  # Short position
                    # Update trailing stop if enabled
                    if self.use_trailing_stop:
                        new_trailing = current_price + (current_atr * self.trailing_stop_atr_mult)
                        trailing_stop = min(trailing_stop, new_trailing)

                    # Check exits
                    if current_high >= stop_loss:
                        exit_signal = True
                        exit_reason = "STOP_LOSS"
                        exit_price = stop_loss
                    elif self.use_trailing_stop and current_high >= trailing_stop:
                        exit_signal = True
                        exit_reason = "TRAILING_STOP"
                        exit_price = trailing_stop
                    elif current_low <= target:
                        exit_signal = True
                        exit_reason = "TARGET_HIT"
                        exit_price = target
                    # VWAP cross exit: Price crosses above VWAP significantly
                    elif current_price > current_vwap * (1 + self.pullback_tolerance_pct * 2):
                        exit_signal = True
                        exit_reason = "VWAP_CROSS_EXIT"
                        exit_price = current_price

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
                    print(f"  Direction: {'LONG' if position == 1 else 'SHORT'}")
                    print(f"  Entry: {entry_time.strftime('%H:%M:%S')} @ Rs.{entry_price:.2f}")
                    print(f"  Exit: {current_time.strftime('%H:%M:%S')} @ Rs.{exit_price:.2f}")
                    print(f"  Duration: {duration:.1f} min | P&L: Rs.{trade_pnl:.2f} ({trade_return:+.2f}%)")
                    print(f"  {'PROFIT' if trade_pnl > 0 else 'LOSS'}")

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

        # Calculate metrics
        completed_trades = [t for t in trades if 'exit_time' in t]
        metrics = self.calculate_metrics(completed_trades)

        return {
            'symbol': symbol,
            'data': df,
            'trades': completed_trades,
            'metrics': metrics
        }

    def calculate_metrics(self, trades):
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'total_return': 0,
                'avg_return': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_duration': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }

        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        losing_trades = sum(1 for t in trades if t['pnl'] <= 0)

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

        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
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
        """Run backtest for all symbols"""
        print(f"\n{'='*100}")
        print("STARTING VWAP SCALPING BACKTEST")
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
        """Print summary of all results"""
        if not self.results:
            print("\nNo results to display.")
            return

        print(f"\n{'='*100}")
        print("VWAP SCALPING STRATEGY RESULTS - ALL SYMBOLS")
        print(f"{'='*100}")

        summary_data = []
        for symbol, result in self.results.items():
            metrics = result['metrics']
            clean_symbol = symbol.split(':')[1].replace('-EQ', '') if ':' in symbol else symbol.replace('-EQ', '')

            summary_data.append({
                'Symbol': clean_symbol,
                'Trades': metrics['total_trades'],
                'Win Rate': f"{metrics['win_rate']:.1f}%",
                'Total P&L': f"Rs.{metrics['total_pnl']:.2f}",
                'Avg P&L': f"Rs.{metrics['avg_pnl']:.2f}",
                'Profit Factor': f"{metrics['profit_factor']:.2f}",
                'Avg Duration': f"{metrics['avg_duration']:.1f}m",
                'Best Trade': f"Rs.{metrics['best_trade']:.2f}",
                'Worst Trade': f"Rs.{metrics['worst_trade']:.2f}"
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        # Overall stats
        print(f"\n{'='*100}")
        print("OVERALL STATISTICS")
        print(f"{'='*100}")

        total_trades = sum(r['metrics']['total_trades'] for r in self.results.values())
        total_pnl = sum(r['metrics']['total_pnl'] for r in self.results.values())
        winning_trades = sum(r['metrics']['winning_trades'] for r in self.results.values())
        profitable_symbols = sum(1 for r in self.results.values() if r['metrics']['total_pnl'] > 0)

        print(f"Symbols Tested: {len(self.results)}")
        print(f"Total Trades: {total_trades}")
        print(f"Total P&L: Rs.{total_pnl:.2f}")
        print(f"Overall Win Rate: {(winning_trades / total_trades * 100):.1f}%" if total_trades > 0 else "N/A")
        print(f"Profitable Symbols: {profitable_symbols}/{len(self.results)}")

        summary_df.to_csv('vwap_scalping_backtest_results.csv', index=False)
        print(f"\nResults saved to: vwap_scalping_backtest_results.csv")


if __name__ == "__main__":
    print("\n" + "="*100)
    print("VWAP SCALPING STRATEGY - INTRADAY BACKTEST WITH FYERS")
    print("="*100)

    # IMPORTANT: Set your Fyers credentials here
    FYERS_CLIENT_ID = os.environ.get('FYERS_CLIENT_ID')
    FYERS_ACCESS_TOKEN = os.environ.get('FYERS_ACCESS_TOKEN')

    # Validate credentials
    if not FYERS_CLIENT_ID or not FYERS_ACCESS_TOKEN:
        print("\nERROR: Missing Fyers API credentials!")
        print("   Please set both FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN environment variables")
        print("\n   Step 1: Create a .env file in the project root")
        print("   Step 2: Add your credentials:")
        print("           FYERS_CLIENT_ID=your_client_id_here")
        print("           FYERS_ACCESS_TOKEN=your_access_token_here")
        print("\n   How to get credentials:")
        print("   1. Login to: https://myapi.fyers.in/dashboard/")
        print("   2. Create an app to get your CLIENT_ID")
        print("   3. Generate access token using authentication flow")
        print("   4. Visit docs: https://myapi.fyers.in/docsv3/#tag/Authentication\n")
        exit(1)

    # Define symbols to backtest (same as OpenRangeBreakout.py)
    SYMBOLS = [
        "NSE:SBIN-EQ",      # State Bank of India
        "NSE:RELIANCE-EQ",  # Reliance Industries
        "NSE:TCS-EQ",       # Tata Consultancy Services
        "NSE:INFY-EQ",      # Infosys
        "NSE:HDFCBANK-EQ",  # HDFC Bank

        "NSE:URBANCO-EQ",
        "NSE:AMANTA-EQ",
        "NSE:VIKRAMSOLR-EQ",
        "NSE:SHREEJISPG-EQ",
        "NSE:PATELRMART-EQ",
        "NSE:REGAAL-EQ",
        "NSE:HILINFRA-EQ",
        "NSE:SAATVIKGL-EQ",
        "NSE:ATLANTAELE-EQ",
        "NSE:STYL-EQ",
        "NSE:SOLARWORLD-EQ",
        "NSE:TRUALT-EQ",
        "NSE:ADVANCE-EQ",
        "NSE:LGEINDIA-EQ",
        "NSE:RUBICON-EQ",
        "NSE:MIDWESTLTD-EQ",
        "NSE:ORKLAINDIA-EQ",
        "NSE:LENSKART-EQ",
        "NSE:GROWW-EQ",
        "NSE:SUDEEPPHRM-EQ",
        "NSE:EXCELSOFT-EQ",
        "NSE:TENNIND-EQ",
        "NSE:MEESHO-EQ",
        "NSE:AEQUS-EQ",
        "NSE:CORONA-EQ",

        # Favourite Stocks
        "NSE:STLTECH-EQ",
        "NSE:SKYGOLD-EQ",
        "NSE:AXISCADES-EQ",
        "BSE:SATTRIX-M"
    ]

    backtester = VWAPScalpingBacktester(
        fyers_client_id=FYERS_CLIENT_ID,
        fyers_access_token=FYERS_ACCESS_TOKEN,
        symbols=SYMBOLS,
        backtest_days=30,  # Last 30 days

        # VWAP parameters
        vwap_band_mult=1.5,  # VWAP band standard deviation multiplier
        pullback_tolerance_pct=0.1,  # 0.1% tolerance for VWAP touch
        scalp_mode='pullback',  # Options: 'pullback', 'band_bounce', 'vwap_cross'
        vwap_trend_candles=5,  # Candles to confirm trend direction

        # Technical indicators
        atr_period=14,  # ATR period for volatility measurement
        rsi_period=14,  # RSI calculation period
        rsi_oversold=35,  # RSI oversold level (less extreme for scalping)
        rsi_overbought=65,  # RSI overbought level (less extreme for scalping)
        volume_threshold_mult=1.2,  # 1.2x volume surge confirmation

        # Filters
        use_rsi_filter=True,  # Use RSI for entry filtering
        use_volume_filter=True,  # Use volume confirmation

        # Risk management (tight for scalping)
        stop_loss_atr_mult=0.75,  # Tight stop loss (0.75x ATR)
        target_atr_mult=1.5,  # Quick target (1.5x ATR)

        # Trailing stops
        use_trailing_stop=True,
        trailing_stop_atr_mult=0.5,  # Very tight trailing stop for scalping

        # Trading rules
        initial_capital=100000,
        square_off_time="15:20",
        tick_interval='5',  # 5 seconds resolution
        last_entry_time="14:30",
        max_trades_per_day=5,  # More trades allowed for scalping
        min_risk_reward=1.5
    )

    backtester.run_backtest()

    print("\n" + "="*100)
    print("BACKTEST COMPLETE")
    print("="*100)
