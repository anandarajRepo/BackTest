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

class VolumeProfileBreakoutBacktester:
    def __init__(self, fyers_client_id, fyers_access_token, symbols=None,
                 volume_profile_start_time="9:15", volume_profile_end_time="15:00",
                 signal_start_time="10:00", signal_end_time="15:00",
                 breakout_confirmation_pct=0.2, volume_threshold_mult=1.3,
                 atr_period=14, momentum_period=14, momentum_threshold=0,
                 rsi_period=14, rsi_oversold=30, rsi_overbought=70,
                 stop_loss_atr_mult=1.5, target_atr_mult=3.0,
                 use_atr_targets=True, trailing_stop_atr_mult=1.0,
                 use_trailing_stop=False, initial_capital=100000,
                 square_off_time="15:20", min_data_points=100,
                 tick_interval='5', last_entry_time="14:30",
                 max_trades_per_day=3, min_risk_reward=1.5,
                 false_breakout_candles=2, backtest_days=7,
                 value_area_pct=70, num_price_levels=50):
        """
        Volume Profile Breakout Strategy for Intraday Trading

        STRATEGY CONCEPT:
        ----------------
        The Volume Profile Breakout strategy identifies key price levels based on
        volume distribution throughout the day. It calculates the volume profile
        from 9:15 AM to 3:00 PM and generates trading signals from 10:00 AM to 3:00 PM.

        KEY COMPONENTS:
        ---------------
        1. Volume Profile Calculation
           - Calculate volume distribution across price levels from 9:15 AM to 3:00 PM
           - Identify Point of Control (POC): Price level with highest volume
           - Identify Value Area High (VAH): Upper bound of value area (70% volume)
           - Identify Value Area Low (VAL): Lower bound of value area (70% volume)

        2. Breakout Detection
           - Upward breakout: Price closes above VAH with volume confirmation
           - Downward breakout: Price closes below VAL with volume confirmation
           - Volume confirmation required
           - Price must sustain above/below value area

        3. Signal Generation
           - Signals generated only from 10:00 AM to 3:00 PM
           - Volume profile updated continuously throughout the day

        4. Risk Management
           - ATR-based stop losses and profit targets
           - Optional trailing stops
           - Time-based filters
           - Maximum trades per day limit

        ENTRY SIGNALS:
        --------------
        LONG ENTRY:
        - Volume profile calculated and valid
        - Price breaks above VAH with confirmation
        - Volume surge (> threshold)
        - Breakout sustained for N candles
        - Valid trading time (10:00 AM - 3:00 PM)
        - Risk-reward ratio acceptable

        SHORT ENTRY:
        - Volume profile calculated and valid
        - Price breaks below VAL with confirmation
        - Volume surge (> threshold)
        - Breakout sustained for N candles
        - Valid trading time (10:00 AM - 3:00 PM)
        - Risk-reward ratio acceptable

        EXIT SIGNALS:
        -------------
        - Stop loss hit (ATR-based)
        - Profit target hit
        - Trailing stop hit (if enabled)
        - End of day square-off
        - Re-entry into value area (failed breakout)

        PARAMETERS:
        -----------
        - volume_profile_start_time: Start time for volume profile calculation (default: "9:15")
        - volume_profile_end_time: End time for volume profile calculation (default: "15:00")
        - signal_start_time: Start time for signal generation (default: "10:00")
        - signal_end_time: End time for signal generation (default: "15:00")
        - breakout_confirmation_pct: % move beyond value area (default: 0.2%)
        - volume_threshold_mult: Volume multiplier (default: 1.3x)
        - false_breakout_candles: Candles to confirm breakout (default: 2)
        - value_area_pct: Percentage of volume in value area (default: 70%)
        - num_price_levels: Number of price levels for volume profile (default: 50)
        - atr_period: ATR calculation period (default: 14)
        - momentum_period: Momentum calculation period (default: 14)
        - momentum_threshold: Minimum momentum value for entry (default: 0)
        - rsi_period: RSI calculation period (default: 14)
        - rsi_oversold: RSI oversold level (default: 30)
        - rsi_overbought: RSI overbought level (default: 70)
        - stop_loss_atr_mult: Stop loss in ATR multiples (default: 1.5)
        - target_atr_mult: Target in ATR multiples (default: 3.0)
        - use_atr_targets: Use ATR for targets (default: True)
        - trailing_stop_atr_mult: Trailing stop in ATR (default: 1.0)
        - use_trailing_stop: Enable trailing stops (default: False)
        - last_entry_time: Last entry time (default: "14:30")
        - max_trades_per_day: Maximum trades per day (default: 3)
        - min_risk_reward: Minimum risk-reward ratio (default: 1.5)
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

        # Volume profile parameters
        self.volume_profile_start_time = self.parse_time(volume_profile_start_time)
        self.volume_profile_end_time = self.parse_time(volume_profile_end_time)
        self.signal_start_time = self.parse_time(signal_start_time)
        self.signal_end_time = self.parse_time(signal_end_time)
        self.value_area_pct = value_area_pct
        self.num_price_levels = num_price_levels

        # Breakout parameters
        self.breakout_confirmation_pct = breakout_confirmation_pct / 100
        self.volume_threshold_mult = volume_threshold_mult
        self.false_breakout_candles = false_breakout_candles

        # Technical indicator parameters
        self.atr_period = atr_period
        self.momentum_period = momentum_period
        self.momentum_threshold = momentum_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        # ATR-based stop loss and targets
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.target_atr_mult = target_atr_mult
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.use_trailing_stop = use_trailing_stop

        # Target calculation method
        self.use_atr_targets = use_atr_targets

        # Trading parameters
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_time(square_off_time)
        self.min_data_points = min_data_points
        self.tick_interval = tick_interval

        # Risk management
        self.last_entry_time = self.parse_time(last_entry_time)
        self.max_trades_per_day = max_trades_per_day
        self.min_risk_reward = min_risk_reward

        # Results storage
        self.results = {}
        self.combined_data = {}

        # Timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print(f"{'='*100}")
        print(f"VOLUME PROFILE BREAKOUT STRATEGY - INTRADAY TRADING (FYERS DATA)")
        print(f"{'='*100}")
        print(f"Data Source: Fyers API")
        print(f"Backtest Period: Last {backtest_days} days")
        print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"\nStrategy Parameters:")
        print(f"  Tick Interval: {self.tick_interval} seconds")
        print(f"  Volume Profile Calculation: {volume_profile_start_time} - {volume_profile_end_time}")
        print(f"  Signal Generation: {signal_start_time} - {signal_end_time}")
        print(f"  Value Area: {value_area_pct}% of volume")
        print(f"  Price Levels: {num_price_levels}")
        print(f"  Breakout Confirmation: {breakout_confirmation_pct}% move, {self.volume_threshold_mult}x volume")
        print(f"  False Breakout Filter: {self.false_breakout_candles} candles")
        print(f"  Technical Indicators:")
        print(f"    - ATR Period: {self.atr_period}")
        print(f"    - Momentum Period: {self.momentum_period} (Threshold: {self.momentum_threshold})")
        print(f"    - RSI Period: {self.rsi_period} (Oversold: {self.rsi_oversold}, Overbought: {self.rsi_overbought})")
        print(f"  Stop Loss: {self.stop_loss_atr_mult}x ATR")
        print(f"  Target: {self.target_atr_mult}x ATR")
        print(f"  Trailing Stop: {'Enabled' if use_trailing_stop else 'Disabled'}")
        if use_trailing_stop:
            print(f"  Trailing Stop: {self.trailing_stop_atr_mult}x ATR")
        print(f"  Max Trades/Day: {self.max_trades_per_day}")
        print(f"  Min Risk-Reward: {self.min_risk_reward}:1")
        print(f"  Last Entry: {last_entry_time}")
        print(f"  Square-off: {square_off_time}")
        print(f"  Initial Capital: ₹{self.initial_capital:,}")
        print(f"{'='*100}")

        # Set symbols
        if symbols is None:
            # Default symbols if none provided
            self.symbols = ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ", "NSE:TCS-EQ"]
            print(f"\nUsing default symbols: {self.symbols}")
        else:
            self.symbols = symbols
            print(f"\nSymbols to backtest: {len(self.symbols)}")

    def parse_time(self, time_str):
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
                "resolution": self.tick_interval,  # Resolution in seconds (e.g., "5" for 5 seconds)
                "date_format": "0",  # Unix timestamp
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

    def calculate_atr(self, df):
        """Calculate Average True Range"""
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        df['atr'] = df['tr'].ewm(span=self.atr_period, adjust=False).mean()

        return df

    def calculate_momentum(self, df):
        """Calculate Momentum Indicator

        Momentum measures the rate of change in price over a specified period.
        Formula: Current Price - Price N periods ago

        Positive momentum indicates upward price movement
        Negative momentum indicates downward price movement
        """
        df['momentum'] = df['close'] - df['close'].shift(self.momentum_period)
        return df

    def calculate_rsi(self, df):
        """Calculate Relative Strength Index (RSI)

        RSI is a momentum oscillator that measures the speed and magnitude of price changes.
        Formula: RSI = 100 - (100 / (1 + RS))
        Where RS = Average Gain / Average Loss over the period

        RSI ranges from 0 to 100:
        - Above 70: Overbought (potential reversal down)
        - Below 30: Oversold (potential reversal up)
        """
        # Calculate price changes
        delta = df['close'].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss using exponential moving average
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()

        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss

        # Calculate RSI
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def calculate_volume_profile(self, df):
        """Calculate volume profile for each trading day"""
        df['trading_day'] = df.index.date
        df['time_of_day'] = df.index.time

        # Identify volume profile calculation period
        df['in_vp_period'] = (df['time_of_day'] >= self.volume_profile_start_time) & \
                             (df['time_of_day'] <= self.volume_profile_end_time)

        # Calculate volume profile for each day
        volume_profiles = {}

        for day in df['trading_day'].unique():
            day_data = df[df['trading_day'] == day]
            vp_data = day_data[day_data['in_vp_period']]

            if not vp_data.empty and len(vp_data) >= 10:  # Need at least 10 candles
                # Get price range for the day
                high_price = vp_data['high'].max()
                low_price = vp_data['low'].min()
                price_range = high_price - low_price

                if price_range == 0:
                    volume_profiles[day] = {'valid': False}
                    continue

                # Create price levels
                price_levels = np.linspace(low_price, high_price, self.num_price_levels)
                level_size = price_range / (self.num_price_levels - 1)

                # Calculate volume at each price level
                volume_at_price = np.zeros(self.num_price_levels)

                for idx, row in vp_data.iterrows():
                    # Distribute volume across price levels touched by this candle
                    candle_low = row['low']
                    candle_high = row['high']
                    candle_volume = row['volume']

                    # Find which price levels this candle touched
                    for i, price in enumerate(price_levels):
                        if candle_low <= price <= candle_high:
                            volume_at_price[i] += candle_volume

                # Normalize volume distribution
                total_volume = volume_at_price.sum()
                if total_volume == 0:
                    volume_profiles[day] = {'valid': False}
                    continue

                volume_at_price_pct = volume_at_price / total_volume

                # Find Point of Control (POC) - price level with highest volume
                poc_idx = np.argmax(volume_at_price)
                poc = price_levels[poc_idx]

                # Calculate Value Area (VA) - price range containing value_area_pct% of volume
                # Sort price levels by volume
                sorted_indices = np.argsort(volume_at_price)[::-1]
                cumulative_volume = 0
                value_area_indices = []

                for idx in sorted_indices:
                    cumulative_volume += volume_at_price_pct[idx]
                    value_area_indices.append(idx)
                    if cumulative_volume >= (self.value_area_pct / 100):
                        break

                # Find VAH and VAL
                value_area_indices = sorted(value_area_indices)
                val_idx = value_area_indices[0]
                vah_idx = value_area_indices[-1]
                val = price_levels[val_idx]
                vah = price_levels[vah_idx]

                # Calculate average volume during VP period
                avg_volume = vp_data['volume'].mean()

                volume_profiles[day] = {
                    'poc': poc,
                    'vah': vah,
                    'val': val,
                    'high': high_price,
                    'low': low_price,
                    'avg_volume': avg_volume,
                    'valid': True,
                    'volume_distribution': volume_at_price,
                    'price_levels': price_levels
                }
            else:
                volume_profiles[day] = {'valid': False}

        # Add volume profile info to dataframe
        df['poc'] = df['trading_day'].map(lambda d: volume_profiles.get(d, {}).get('poc', np.nan))
        df['vah'] = df['trading_day'].map(lambda d: volume_profiles.get(d, {}).get('vah', np.nan))
        df['val'] = df['trading_day'].map(lambda d: volume_profiles.get(d, {}).get('val', np.nan))
        df['vp_valid'] = df['trading_day'].map(lambda d: volume_profiles.get(d, {}).get('valid', False))
        df['vp_avg_volume'] = df['trading_day'].map(lambda d: volume_profiles.get(d, {}).get('avg_volume', 0))

        return df

    def detect_breakouts(self, df):
        """Detect breakouts from volume profile value area"""
        # Check if in signal generation period
        df['in_signal_period'] = (df['time_of_day'] >= self.signal_start_time) & \
                                  (df['time_of_day'] <= self.signal_end_time)

        # Calculate volume surge
        df['volume_surge'] = df['volume'] > (df['vp_avg_volume'] * self.volume_threshold_mult)

        # Breakout above Value Area High (VAH)
        df['breakout_up'] = (
            (df['vp_valid']) &
            (df['in_signal_period']) &
            (df['close'] > df['vah'] * (1 + self.breakout_confirmation_pct)) &
            (df['volume_surge'])
        )

        # Breakout below Value Area Low (VAL)
        df['breakout_down'] = (
            (df['vp_valid']) &
            (df['in_signal_period']) &
            (df['close'] < df['val'] * (1 - self.breakout_confirmation_pct)) &
            (df['volume_surge'])
        )

        return df

    def is_valid_trading_time(self, timestamp):
        """Check if time is valid for trading"""
        try:
            current_time = timestamp.time()

            # Must be in signal generation period
            if current_time < self.signal_start_time:
                return False

            # No new entries after last_entry_time
            if current_time >= self.last_entry_time:
                return False

            return True
        except:
            return False

    def generate_signals(self, df):
        """Generate trading signals"""
        # Valid trading time
        df['valid_time'] = df.index.map(self.is_valid_trading_time)

        # LONG SIGNAL: Breakout above Value Area High
        df['long_signal'] = (
            (df['breakout_up']) &
            (df['valid_time']) &
            (~df['atr'].isna())
        )

        # SHORT SIGNAL: Breakout below Value Area Low
        df['short_signal'] = (
            (df['breakout_down']) &
            (df['valid_time']) &
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
        """Backtest Volume Profile Breakout strategy for a symbol"""
        print(f"\n{'='*100}")
        print(f"Backtesting: {symbol}")
        print(f"{'='*100}")

        df = self.load_data_from_fyers(symbol)

        if df is None or len(df) < self.atr_period * 3:
            print(f"Insufficient data for {symbol}. Skipping.")
            return None

        self.combined_data[symbol] = df.copy()

        # Calculate all technical indicators
        df = self.calculate_atr(df)
        df = self.calculate_momentum(df)
        df = self.calculate_rsi(df)
        df = self.calculate_volume_profile(df)
        df = self.detect_breakouts(df)
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
        trade_number = 0
        trades_today = {}

        # Breakout confirmation tracking
        breakout_pending = False
        breakout_direction = 0
        breakout_candles_count = 0
        breakout_info = {}

        # Backtest loop
        for i in range(len(df)):
            current_time = df.index[i]
            current_price = df.iloc[i]['close']
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            current_atr = df.iloc[i]['atr']
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
                print(f"  Entry: {entry_time.strftime('%H:%M:%S')} @ ₹{entry_price:.2f}")
                print(f"  Exit: ₹{current_price:.2f} | P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
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

            # Breakout confirmation logic
            elif breakout_pending:
                breakout_candles_count += 1

                # Check if breakout is sustained
                if breakout_direction == 1:  # Up breakout
                    if current_low < breakout_info['vah']:
                        # Failed breakout - price re-entered value area
                        breakout_pending = False
                        breakout_candles_count = 0
                    elif breakout_candles_count >= self.false_breakout_candles:
                        # Confirmed breakout - enter long
                        breakout_pending = False
                        breakout_candles_count = 0

                        if trades_today[current_day] < self.max_trades_per_day:
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
                                        'risk_reward': risk_reward,
                                        'vah': breakout_info['vah'],
                                        'val': breakout_info['val'],
                                        'poc': breakout_info['poc']
                                    })

                                    print(f"\n[VP LONG ENTRY] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                                    print(f"  Price: ₹{entry_price:.2f} | Stop: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                                    print(f"  Shares: {shares} | R:R = 1:{risk_reward:.2f}")
                                    print(f"  Value Area: ₹{breakout_info['val']:.2f} - ₹{breakout_info['vah']:.2f} | POC: ₹{breakout_info['poc']:.2f}")

                elif breakout_direction == -1:  # Down breakout
                    if current_high > breakout_info['val']:
                        # Failed breakout - price re-entered value area
                        breakout_pending = False
                        breakout_candles_count = 0
                    elif breakout_candles_count >= self.false_breakout_candles:
                        # Confirmed breakout - enter short
                        breakout_pending = False
                        breakout_candles_count = 0

                        if trades_today[current_day] < self.max_trades_per_day:
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
                                        'risk_reward': risk_reward,
                                        'vah': breakout_info['vah'],
                                        'val': breakout_info['val'],
                                        'poc': breakout_info['poc']
                                    })

                                    print(f"\n[VP SHORT ENTRY] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                                    print(f"  Price: ₹{entry_price:.2f} | Stop: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                                    print(f"  Shares: {shares} | R:R = 1:{risk_reward:.2f}")
                                    print(f"  Value Area: ₹{breakout_info['val']:.2f} - ₹{breakout_info['vah']:.2f} | POC: ₹{breakout_info['poc']:.2f}")

            # Entry signals
            elif position == 0 and not is_square_off:
                # Check trade frequency limit
                if trades_today[current_day] >= self.max_trades_per_day:
                    continue

                # VP LONG (pending confirmation)
                if df.iloc[i]['long_signal']:
                    breakout_pending = True
                    breakout_direction = 1
                    breakout_candles_count = 0
                    breakout_info = {
                        'vah': df.iloc[i]['vah'],
                        'val': df.iloc[i]['val'],
                        'poc': df.iloc[i]['poc']
                    }

                # VP SHORT (pending confirmation)
                elif df.iloc[i]['short_signal']:
                    breakout_pending = True
                    breakout_direction = -1
                    breakout_candles_count = 0
                    breakout_info = {
                        'vah': df.iloc[i]['vah'],
                        'val': df.iloc[i]['val'],
                        'poc': df.iloc[i]['poc']
                    }

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
                    # Re-entry into value area (failed breakout)
                    elif current_low <= df.iloc[i]['val']:
                        exit_signal = True
                        exit_reason = "RE_ENTRY_VA"
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
                    # Re-entry into value area (failed breakout)
                    elif current_high >= df.iloc[i]['vah']:
                        exit_signal = True
                        exit_reason = "RE_ENTRY_VA"
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
                    print(f"  Entry: {entry_time.strftime('%H:%M:%S')} @ ₹{entry_price:.2f}")
                    print(f"  Exit: {current_time.strftime('%H:%M:%S')} @ ₹{exit_price:.2f}")
                    print(f"  Duration: {duration:.1f} min | P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
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
        print("STARTING VOLUME PROFILE BREAKOUT BACKTEST")
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
        print("VOLUME PROFILE BREAKOUT STRATEGY RESULTS - ALL SYMBOLS")
        print(f"{'='*100}")

        summary_data = []
        for symbol, result in self.results.items():
            metrics = result['metrics']
            clean_symbol = symbol.split(':')[1].replace('-EQ', '') if ':' in symbol else symbol.replace('-EQ', '')

            summary_data.append({
                'Symbol': clean_symbol,
                'Trades': metrics['total_trades'],
                'Win Rate': f"{metrics['win_rate']:.1f}%",
                'Total P&L': f"₹{metrics['total_pnl']:.2f}",
                'Avg P&L': f"₹{metrics['avg_pnl']:.2f}",
                'Profit Factor': f"{metrics['profit_factor']:.2f}",
                'Avg Duration': f"{metrics['avg_duration']:.1f}m",
                'Best Trade': f"₹{metrics['best_trade']:.2f}",
                'Worst Trade': f"₹{metrics['worst_trade']:.2f}"
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
        print(f"Total P&L: ₹{total_pnl:.2f}")
        print(f"Overall Win Rate: {(winning_trades / total_trades * 100):.1f}%" if total_trades > 0 else "N/A")
        print(f"Profitable Symbols: {profitable_symbols}/{len(self.results)}")

        summary_df.to_csv('volume_profile_backtest_results.csv', index=False)
        print(f"\n✅ Results saved to: volume_profile_backtest_results.csv")


if __name__ == "__main__":
    print("\n" + "="*100)
    print("VOLUME PROFILE BREAKOUT STRATEGY - INTRADAY BACKTEST WITH FYERS")
    print("="*100)

    # IMPORTANT: Set your Fyers credentials here
    # Get these from Fyers API authentication
    FYERS_CLIENT_ID = os.environ.get('FYERS_CLIENT_ID')
    FYERS_ACCESS_TOKEN = os.environ.get('FYERS_ACCESS_TOKEN')

    # Validate credentials
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
        print("   3. Generate access token using authentication flow")
        print("   4. Visit docs: https://myapi.fyers.in/docsv3/#tag/Authentication\n")
        exit(1)

    # Define symbols to backtest (Fyers format: NSE:SYMBOL-EQ)
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

    backtester = VolumeProfileBreakoutBacktester(
        fyers_client_id=FYERS_CLIENT_ID,
        fyers_access_token=FYERS_ACCESS_TOKEN,
        symbols=SYMBOLS,
        backtest_days=30,  # Last 30 days

        # Volume profile parameters
        volume_profile_start_time="9:15",  # Start calculating volume profile from 9:15 AM
        volume_profile_end_time="15:00",   # Calculate until 3:00 PM
        signal_start_time="10:00",         # Generate signals from 10:00 AM
        signal_end_time="15:00",           # Generate signals until 3:00 PM
        value_area_pct=70,                 # 70% volume for value area
        num_price_levels=50,               # Number of price levels for volume profile

        # Breakout parameters
        breakout_confirmation_pct=0.2,  # 0.2% move beyond value area
        volume_threshold_mult=1.3,      # 1.3x average volume
        false_breakout_candles=2,       # Wait 2 candles for confirmation

        # Technical indicators
        atr_period=14,            # ATR period for volatility measurement
        momentum_period=14,       # Momentum calculation period
        momentum_threshold=0,     # Minimum momentum for entry (0 = disabled)
        rsi_period=14,            # RSI calculation period
        rsi_oversold=30,          # RSI oversold level
        rsi_overbought=70,        # RSI overbought level

        # Risk management
        stop_loss_atr_mult=1.5,
        target_atr_mult=3.0,
        use_atr_targets=True,     # Use ATR for targets

        # Trailing stops
        use_trailing_stop=True,
        trailing_stop_atr_mult=1.0,

        # Trading rules
        initial_capital=100000,
        square_off_time="15:20",
        tick_interval='5S',       # 5 seconds resolution (Fyers format)
        last_entry_time="14:30",
        max_trades_per_day=3,
        min_risk_reward=1.5
    )

    backtester.run_backtest()

    print("\n" + "="*100)
    print("BACKTEST COMPLETE")
    print("="*100)
