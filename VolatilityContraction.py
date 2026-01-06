import sqlite3
import pandas as pd
import numpy as np
import json
import glob
import os
from datetime import datetime, time
import pytz
import warnings

warnings.filterwarnings('ignore')


class VolatilityContractionBacktester:
    def __init__(self, data_folder="data", symbols=None,
                 bb_period=20, bb_std=2.0, kc_period=20, kc_atr_mult=1.5,
                 atr_period=14, squeeze_threshold=0.8, min_squeeze_candles=6,
                 volatility_expansion_threshold=1.3, rsi_period=14,
                 initial_capital=100000, square_off_time="15:20",
                 min_data_points=100, tick_interval='1min',
                 stop_loss_atr_mult=2.0, target_atr_mult=3.0,
                 avoid_opening_mins=15, last_entry_time="14:45",
                 max_trades_per_day=3, min_risk_reward=1.5):
        """
        Volatility Contraction Pattern Strategy for Intraday Trading

        STRATEGY CONCEPT:
        ----------------
        Markets alternate between periods of low and high volatility.
        This strategy identifies volatility contraction (squeeze) and
        trades the subsequent expansion (breakout).

        KEY COMPONENTS:
        ---------------
        1. Bollinger Bands Squeeze Detection
        2. Keltner Channel for confirmation
        3. ATR-based volatility measurement
        4. Momentum confirmation with RSI
        5. Directional breakout validation

        ENTRY SIGNALS:
        --------------
        - Bollinger Bands inside Keltner Channels (squeeze)
        - Minimum consecutive candles in squeeze
        - Volatility expansion (ATR increasing)
        - Clear directional breakout
        - RSI confirmation

        EXIT SIGNALS:
        -------------
        - ATR-based stop loss
        - ATR-based profit target
        - End of day square-off

        PARAMETERS:
        -----------
        - bb_period: Bollinger Bands period (default: 20)
        - bb_std: Bollinger Bands standard deviation (default: 2.0)
        - kc_period: Keltner Channel period (default: 20)
        - kc_atr_mult: Keltner Channel ATR multiplier (default: 1.5)
        - atr_period: ATR calculation period (default: 14)
        - squeeze_threshold: BB/KC width ratio for squeeze (default: 0.8)
        - min_squeeze_candles: Minimum candles in squeeze (default: 6)
        - volatility_expansion_threshold: ATR increase for breakout (default: 1.3x)
        - rsi_period: RSI period for momentum (default: 14)
        - stop_loss_atr_mult: Stop loss in ATR multiples (default: 2.0)
        - target_atr_mult: Target in ATR multiples (default: 3.0)
        - avoid_opening_mins: Skip first N minutes (default: 15)
        - last_entry_time: Last entry time (default: "14:45")
        - max_trades_per_day: Maximum trades per day (default: 3)
        - min_risk_reward: Minimum risk-reward ratio (default: 1.5)
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()

        # Bollinger Bands parameters
        self.bb_period = bb_period
        self.bb_std = bb_std

        # Keltner Channel parameters
        self.kc_period = kc_period
        self.kc_atr_mult = kc_atr_mult

        # Volatility parameters
        self.atr_period = atr_period
        self.squeeze_threshold = squeeze_threshold
        self.min_squeeze_candles = min_squeeze_candles
        self.volatility_expansion_threshold = volatility_expansion_threshold

        # Momentum parameters
        self.rsi_period = rsi_period

        # Trading parameters
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.tick_interval = tick_interval
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.target_atr_mult = target_atr_mult

        # Risk management
        self.avoid_opening_mins = avoid_opening_mins
        self.last_entry_time = self.parse_square_off_time(last_entry_time)
        self.max_trades_per_day = max_trades_per_day
        self.min_risk_reward = min_risk_reward

        # Results storage
        self.results = {}
        self.combined_data = {}

        # Timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print(f"{'='*100}")
        print(f"VOLATILITY CONTRACTION PATTERN STRATEGY - INTRADAY TRADING")
        print(f"{'='*100}")
        print(f"Strategy Parameters:")
        print(f"  Tick Interval: {self.tick_interval}")
        print(f"  Bollinger Bands: {self.bb_period} period, {self.bb_std} std dev")
        print(f"  Keltner Channel: {self.kc_period} period, {self.kc_atr_mult}x ATR")
        print(f"  ATR Period: {self.atr_period}")
        print(f"  Squeeze Threshold: {self.squeeze_threshold} (BB/KC ratio)")
        print(f"  Min Squeeze Candles: {self.min_squeeze_candles}")
        print(f"  Volatility Expansion: {self.volatility_expansion_threshold}x ATR increase")
        print(f"  RSI Period: {self.rsi_period}")
        print(f"  Stop Loss: {self.stop_loss_atr_mult}x ATR")
        print(f"  Target: {self.target_atr_mult}x ATR")
        print(f"  Max Trades/Day: {self.max_trades_per_day}")
        print(f"  Min Risk-Reward: {self.min_risk_reward}:1")
        print(f"  Avoid Opening: {self.avoid_opening_mins} minutes")
        print(f"  Last Entry: {last_entry_time}")
        print(f"  Square-off: {square_off_time}")
        print(f"  Initial Capital: ₹{self.initial_capital:,}")
        print(f"{'='*100}")

        # Auto-detect symbols
        if symbols is None:
            print("\nAuto-detecting symbols from databases...")
            self.symbols = self.auto_detect_symbols()
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

    def find_database_files(self):
        """Find all database files"""
        if not os.path.exists(self.data_folder):
            return []
        db_files = glob.glob(os.path.join(self.data_folder, "*.db"))
        return sorted(db_files)

    def auto_detect_symbols(self):
        """Auto-detect symbols from databases"""
        all_symbols = set()
        symbol_stats = {}

        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                query = """
                SELECT symbol, COUNT(*) as record_count,
                       MIN(timestamp) as first_record,
                       MAX(timestamp) as last_record
                FROM market_data
                GROUP BY symbol
                HAVING COUNT(*) >= ?
                ORDER BY record_count DESC
                """
                symbols_df = pd.read_sql_query(query, conn, params=(self.min_data_points,))
                conn.close()

                for _, row in symbols_df.iterrows():
                    symbol = row['symbol']
                    all_symbols.add(symbol)

                    if symbol not in symbol_stats:
                        symbol_stats[symbol] = {
                            'total_records': 0,
                            'databases': []
                        }

                    symbol_stats[symbol]['total_records'] += row['record_count']
                    symbol_stats[symbol]['databases'].append(os.path.basename(db_file))
            except:
                continue

        # Filter and sort symbols
        filtered_symbols = [s for s, stats in symbol_stats.items()
                          if stats['total_records'] >= self.min_data_points]

        return sorted(filtered_symbols,
                     key=lambda s: symbol_stats[s]['total_records'],
                     reverse=True)[:20]

    def load_data_from_all_databases(self, symbol):
        """Load and combine data from all databases"""
        combined_df = pd.DataFrame()

        for db_file in self.db_files:
            try:
                df = self.load_data_from_single_db(db_file, symbol)
                if df is not None and not df.empty:
                    df['source_db'] = os.path.basename(db_file)
                    combined_df = pd.concat([combined_df, df], ignore_index=False)
            except:
                continue

        if combined_df.empty:
            return None

        # Sort and remove duplicates
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        # Resample tick data
        if self.tick_interval is not None:
            combined_df = self.resample_tick_data(combined_df, self.tick_interval)

        return combined_df

    def load_data_from_single_db(self, db_path, symbol):
        """Load data from a single database"""
        try:
            conn = sqlite3.connect(db_path)

            check_query = "SELECT COUNT(*) FROM market_data WHERE symbol = ?"
            count_result = pd.read_sql_query(check_query, conn, params=(symbol,))

            if count_result.iloc[0, 0] == 0:
                conn.close()
                return None

            query = """
            SELECT timestamp, symbol, ltp, high_price, low_price, close_price,
                   volume, raw_data
            FROM market_data
            WHERE symbol = ?
            ORDER BY timestamp
            """

            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()

            if df.empty:
                return None

            # Parse raw_data
            def parse_raw_data(raw_data_str):
                try:
                    if raw_data_str:
                        raw_data = json.loads(raw_data_str)
                        return pd.Series({
                            'high_raw': raw_data.get('high_price', np.nan),
                            'low_raw': raw_data.get('low_price', np.nan),
                            'volume_raw': raw_data.get('vol_traded_today', 0)
                        })
                    else:
                        return pd.Series({'high_raw': np.nan, 'low_raw': np.nan, 'volume_raw': 0})
                except:
                    return pd.Series({'high_raw': np.nan, 'low_raw': np.nan, 'volume_raw': 0})

            raw_parsed = df['raw_data'].apply(parse_raw_data)
            df = pd.concat([df, raw_parsed], axis=1)

            # Create OHLC
            df['open'] = df['ltp']
            df['high'] = df['high_raw'].fillna(df['high_price']).fillna(df['ltp'])
            df['low'] = df['low_raw'].fillna(df['low_price']).fillna(df['ltp'])
            df['close'] = df['close_price'].fillna(df['ltp'])
            df['volume'] = df['volume_raw'].fillna(df['volume']).fillna(0)

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            # Remove missing data
            df = df.dropna(subset=['close', 'high', 'low'])

            return df[['open', 'high', 'low', 'close', 'volume']].copy()

        except Exception as e:
            return None

    def resample_tick_data(self, df, interval):
        """Resample tick data to specified time interval"""
        if df is None or df.empty:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        resampled = df.resample(interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        resampled = resampled.dropna(subset=['close'])
        resampled['open'] = resampled['open'].fillna(resampled['close'])
        resampled['high'] = resampled['high'].fillna(resampled['close'])
        resampled['low'] = resampled['low'].fillna(resampled['close'])
        resampled['volume'] = resampled['volume'].fillna(0)

        return resampled

    def calculate_bollinger_bands(self, df):
        """Calculate Bollinger Bands"""
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        df['bb_std'] = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (self.bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (self.bb_std * df['bb_std'])
        df['bb_width'] = df['bb_upper'] - df['bb_lower']

        return df

    def calculate_keltner_channels(self, df):
        """Calculate Keltner Channels"""
        df['kc_middle'] = df['close'].rolling(window=self.kc_period).mean()

        # Calculate True Range
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # ATR for Keltner
        df['kc_atr'] = df['tr'].rolling(window=self.kc_period).mean()

        df['kc_upper'] = df['kc_middle'] + (self.kc_atr_mult * df['kc_atr'])
        df['kc_lower'] = df['kc_middle'] - (self.kc_atr_mult * df['kc_atr'])
        df['kc_width'] = df['kc_upper'] - df['kc_lower']

        return df

    def calculate_atr(self, df):
        """Calculate Average True Range"""
        if 'tr' not in df.columns:
            df['prev_close'] = df['close'].shift(1)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['prev_close'])
            df['tr3'] = abs(df['low'] - df['prev_close'])
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        df['atr'] = df['tr'].ewm(span=self.atr_period, adjust=False).mean()
        df['atr_change'] = df['atr'].pct_change()

        # ATR slope for expansion detection
        df['atr_slope'] = df['atr'].diff()

        return df

    def calculate_rsi(self, df):
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def detect_squeeze(self, df):
        """Detect Bollinger Bands squeeze (inside Keltner Channels)"""
        # Squeeze occurs when BB is inside KC
        df['squeeze'] = (
            (df['bb_upper'] < df['kc_upper']) &
            (df['bb_lower'] > df['kc_lower'])
        )

        # Alternative: BB width / KC width ratio
        df['bb_kc_ratio'] = df['bb_width'] / df['kc_width']
        df['squeeze_ratio'] = df['bb_kc_ratio'] < self.squeeze_threshold

        # Combined squeeze signal
        df['squeeze_active'] = df['squeeze'] | df['squeeze_ratio']

        # Count consecutive squeeze candles
        df['squeeze_count'] = 0
        count = 0
        for i in range(len(df)):
            if df.iloc[i]['squeeze_active']:
                count += 1
            else:
                count = 0
            df.iloc[i, df.columns.get_loc('squeeze_count')] = count

        return df

    def detect_volatility_expansion(self, df):
        """Detect volatility expansion after contraction"""
        # ATR increasing
        df['atr_increasing'] = df['atr_slope'] > 0

        # ATR significantly higher than recent low
        df['atr_min_5'] = df['atr'].rolling(window=5).min()
        df['atr_expansion'] = df['atr'] > (df['atr_min_5'] * self.volatility_expansion_threshold)

        # Candle range increasing
        df['candle_range'] = df['high'] - df['low']
        df['avg_range_5'] = df['candle_range'].rolling(window=5).mean()
        df['range_expansion'] = df['candle_range'] > df['avg_range_5']

        # Combined expansion signal
        df['expansion'] = (
            df['atr_increasing'] &
            df['atr_expansion'] &
            df['range_expansion']
        )

        return df

    def is_valid_trading_time(self, timestamp):
        """Check if time is valid for trading"""
        try:
            current_time = timestamp.time()

            # Market opens at 9:15, avoid first N minutes
            market_open = time(9, 15)
            avoid_until = time(9, 15 + self.avoid_opening_mins)

            # Check conditions
            if current_time < avoid_until:
                return False

            # No new entries after last_entry_time
            if current_time >= self.last_entry_time:
                return False

            return True
        except:
            return False

    def generate_signals(self, df):
        """Generate trading signals"""
        # Detect breakout direction after squeeze
        df['price_above_bb_mid'] = df['close'] > df['bb_middle']
        df['price_below_bb_mid'] = df['close'] < df['bb_middle']

        # Momentum confirmation
        df['rsi_bullish'] = (df['rsi'] > 50) & (df['rsi'] < 80)
        df['rsi_bearish'] = (df['rsi'] < 50) & (df['rsi'] > 20)

        # Valid trading time
        df['valid_time'] = df.index.map(self.is_valid_trading_time)

        # Previous squeeze condition
        df['prev_squeeze'] = df['squeeze_count'].shift(1) >= self.min_squeeze_candles
        df['squeeze_releasing'] = df['prev_squeeze'] & (~df['squeeze_active'])

        # BUY SIGNAL: Breakout above after squeeze with expansion
        df['buy_signal'] = (
            (df['squeeze_releasing']) &
            (df['expansion']) &
            (df['price_above_bb_mid']) &
            (df['close'] > df['bb_middle']) &
            (df['rsi_bullish']) &
            (df['valid_time']) &
            (~df['atr'].isna())
        )

        # SELL SIGNAL: Breakout below after squeeze with expansion
        df['sell_signal'] = (
            (df['squeeze_releasing']) &
            (df['expansion']) &
            (df['price_below_bb_mid']) &
            (df['close'] < df['bb_middle']) &
            (df['rsi_bearish']) &
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
        """Backtest volatility contraction strategy for a symbol"""
        print(f"\n{'='*100}")
        print(f"Backtesting: {symbol}")
        print(f"{'='*100}")

        df = self.load_data_from_all_databases(symbol)

        if df is None or len(df) < max(self.bb_period, self.kc_period, self.atr_period) * 3:
            print(f"Insufficient data for {symbol}. Skipping.")
            return None

        self.combined_data[symbol] = df.copy()

        # Calculate all indicators
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_keltner_channels(df)
        df = self.calculate_atr(df)
        df = self.calculate_rsi(df)
        df = self.detect_squeeze(df)
        df = self.detect_volatility_expansion(df)
        df = self.generate_signals(df)

        df['trading_day'] = df.index.date
        df['is_square_off'] = df.index.map(self.is_square_off_time)

        # Trading variables
        cash = self.initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        stop_loss = 0
        target = 0
        trades = []
        entry_time = None
        entry_atr = 0
        trade_number = 0
        trades_today = {}

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

            # Entry signals
            elif position == 0 and not is_square_off:
                # Check trade frequency limit
                if trades_today[current_day] >= self.max_trades_per_day:
                    continue

                # LONG entry
                if df.iloc[i]['buy_signal']:
                    stop_loss_price = current_price - (current_atr * self.stop_loss_atr_mult)
                    target_price = current_price + (current_atr * self.target_atr_mult)

                    # Risk-reward validation
                    risk = current_price - stop_loss_price
                    reward = target_price - current_price
                    risk_reward = reward / risk if risk > 0 else 0

                    if risk_reward < self.min_risk_reward:
                        continue

                    position = 1
                    entry_price = current_price
                    entry_time = current_time
                    entry_atr = current_atr
                    stop_loss = stop_loss_price
                    target = target_price

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
                            'squeeze_duration': df.iloc[i-1]['squeeze_count'] if i > 0 else 0,
                            'entry_rsi': df.iloc[i]['rsi'],
                            'risk_reward': risk_reward
                        })

                        print(f"\n[LONG ENTRY] {current_time.strftime('%H:%M:%S')}")
                        print(f"  Price: ₹{entry_price:.2f} | Stop: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                        print(f"  Shares: {shares} | R:R = 1:{risk_reward:.2f}")
                        print(f"  Squeeze Duration: {trades[-1]['squeeze_duration']} candles")

                # SHORT entry
                elif df.iloc[i]['sell_signal']:
                    stop_loss_price = current_price + (current_atr * self.stop_loss_atr_mult)
                    target_price = current_price - (current_atr * self.target_atr_mult)

                    # Risk-reward validation
                    risk = stop_loss_price - current_price
                    reward = current_price - target_price
                    risk_reward = reward / risk if risk > 0 else 0

                    if risk_reward < self.min_risk_reward:
                        continue

                    position = -1
                    entry_price = current_price
                    entry_time = current_time
                    entry_atr = current_atr
                    stop_loss = stop_loss_price
                    target = target_price

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
                            'squeeze_duration': df.iloc[i-1]['squeeze_count'] if i > 0 else 0,
                            'entry_rsi': df.iloc[i]['rsi'],
                            'risk_reward': risk_reward
                        })

                        print(f"\n[SHORT ENTRY] {current_time.strftime('%H:%M:%S')}")
                        print(f"  Price: ₹{entry_price:.2f} | Stop: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                        print(f"  Shares: {shares} | R:R = 1:{risk_reward:.2f}")
                        print(f"  Squeeze Duration: {trades[-1]['squeeze_duration']} candles")

            # Exit management
            elif position != 0 and not is_square_off:
                exit_signal = False
                exit_reason = ""
                exit_price = current_price

                if position == 1:  # Long position
                    if current_low <= stop_loss:
                        exit_signal = True
                        exit_reason = "STOP_LOSS"
                        exit_price = stop_loss
                    elif current_high >= target:
                        exit_signal = True
                        exit_reason = "TARGET_HIT"
                        exit_price = target

                elif position == -1:  # Short position
                    if current_high >= stop_loss:
                        exit_signal = True
                        exit_reason = "STOP_LOSS"
                        exit_price = stop_loss
                    elif current_low <= target:
                        exit_signal = True
                        exit_reason = "TARGET_HIT"
                        exit_price = target

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
                'profit_factor': 0,
                'avg_squeeze_duration': 0,
                'long_trades': 0,
                'short_trades': 0,
                'long_win_rate': 0,
                'short_win_rate': 0
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

        avg_squeeze_duration = sum(t.get('squeeze_duration', 0) for t in trades) / total_trades

        # Direction analysis
        long_trades = [t for t in trades if t['direction'] == 'LONG']
        short_trades = [t for t in trades if t['direction'] == 'SHORT']

        long_wins = sum(1 for t in long_trades if t['pnl'] > 0)
        short_wins = sum(1 for t in short_trades if t['pnl'] > 0)

        long_win_rate = (long_wins / len(long_trades) * 100) if long_trades else 0
        short_win_rate = (short_wins / len(short_trades) * 100) if short_trades else 0

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
            'profit_factor': profit_factor,
            'avg_squeeze_duration': avg_squeeze_duration,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate
        }

    def run_backtest(self):
        """Run backtest for all symbols"""
        print(f"\n{'='*100}")
        print("STARTING VOLATILITY CONTRACTION PATTERN BACKTEST")
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
        print("VOLATILITY CONTRACTION STRATEGY RESULTS - ALL SYMBOLS")
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
                'Avg Squeeze': f"{metrics['avg_squeeze_duration']:.0f}",
                'Long Trades': metrics['long_trades'],
                'Short Trades': metrics['short_trades'],
                'Avg Duration': f"{metrics['avg_duration']:.1f}m"
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

        total_long = sum(r['metrics']['long_trades'] for r in self.results.values())
        total_short = sum(r['metrics']['short_trades'] for r in self.results.values())

        print(f"Symbols Tested: {len(self.results)}")
        print(f"Total Trades: {total_trades}")
        print(f"Total P&L: ₹{total_pnl:.2f}")
        print(f"Overall Win Rate: {(winning_trades / total_trades * 100):.1f}%" if total_trades > 0 else "N/A")
        print(f"Profitable Symbols: {profitable_symbols}/{len(self.results)}")
        print(f"Long Trades: {total_long} | Short Trades: {total_short}")

        summary_df.to_csv('volatility_contraction_results.csv', index=False)
        print(f"\n✅ Results saved to: volatility_contraction_results.csv")


if __name__ == "__main__":
    print("\n" + "="*100)
    print("VOLATILITY CONTRACTION PATTERN STRATEGY - INTRADAY BACKTEST")
    print("="*100)

    backtester = VolatilityContractionBacktester(
        data_folder="data/symbolupdate",
        symbols=None,  # Auto-detect

        # Squeeze detection parameters
        bb_period=20,
        bb_std=2.0,
        kc_period=20,
        kc_atr_mult=1.5,
        squeeze_threshold=0.8,
        min_squeeze_candles=6,

        # Volatility parameters
        atr_period=14,
        volatility_expansion_threshold=1.3,

        # Momentum
        rsi_period=14,

        # Risk management
        stop_loss_atr_mult=2.0,
        target_atr_mult=3.0,
        min_risk_reward=1.5,

        # Trading rules
        initial_capital=100000,
        square_off_time="15:20",
        tick_interval='1min',
        avoid_opening_mins=15,
        last_entry_time="14:45",
        max_trades_per_day=3
    )

    backtester.run_backtest()

    print("\n" + "="*100)
    print("BACKTEST COMPLETE")
    print("="*100)
