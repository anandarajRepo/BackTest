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


class SupertrendHeikinAshiBacktester:
    def __init__(self, data_folder="data", symbols=None,
                 supertrend_period=10, supertrend_multiplier=3.0,
                 ha_smoothing=3, atr_period=14,
                 trailing_stop_atr_mult=1.5, breakeven_profit_pct=1.0,
                 initial_capital=100000, square_off_time="15:20",
                 last_entry_time="14:30", min_data_points=100,
                 tick_interval=None, max_trades_per_day=3):
        """
        Supertrend + Heikin Ashi Strategy with Trailing Stop Loss

        Strategy Logic:
        ---------------
        ENTRY CONDITIONS (ALL must be true):
        1. Supertrend turns bullish (price crosses above Supertrend line)
        2. Heikin Ashi candle is bullish (close > open) for confirmation
        3. Heikin Ashi body is significant (not a doji)
        4. Before last entry time (avoid late entries)
        5. Max trades per day not exceeded

        EXIT CONDITIONS (ANY triggers exit):
        1. Supertrend turns bearish (trend reversal)
        2. Heikin Ashi turns bearish (momentum loss)
        3. ATR-based trailing stop loss (dynamic risk management)
        4. Breakeven stop (after profit threshold reached)
        5. 3:20 PM square-off (mandatory)

        Parameters:
        -----------
        - data_folder: Folder containing database files
        - symbols: List of symbols (None = auto-detect)
        - supertrend_period: Period for ATR in Supertrend calculation (default: 10)
        - supertrend_multiplier: Multiplier for ATR in Supertrend (default: 3.0)
        - ha_smoothing: EMA period for Heikin Ashi smoothing (default: 3)
        - atr_period: Period for ATR calculation (default: 14)
        - trailing_stop_atr_mult: ATR multiplier for trailing stop (default: 1.5)
        - breakeven_profit_pct: Profit % to trigger breakeven stop (default: 1.0)
        - initial_capital: Starting capital per symbol (default: 100000)
        - square_off_time: Time to square off (HH:MM format, default: "15:20")
        - last_entry_time: Last time to enter new trades (default: "14:30")
        - min_data_points: Minimum data points per symbol (default: 100)
        - tick_interval: Time interval for resampling (e.g., '30s', '1min', None for raw)
        - max_trades_per_day: Maximum trades allowed per day (default: 3)
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier
        self.ha_smoothing = ha_smoothing
        self.atr_period = atr_period
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.breakeven_profit_pct = breakeven_profit_pct / 100
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.last_entry_time = self.parse_square_off_time(last_entry_time)
        self.min_data_points = min_data_points
        self.tick_interval = tick_interval
        self.max_trades_per_day = max_trades_per_day
        self.results = {}
        self.combined_data = {}

        # Set up IST timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print(f"{'='*100}")
        print(f"SUPERTREND + HEIKIN ASHI STRATEGY - Trend Following with Confirmation")
        print(f"{'='*100}")
        print(f"Strategy Parameters:")
        print(f"  Tick Interval: {self.tick_interval if self.tick_interval else 'Raw tick data (no resampling)'}")
        print(f"  Supertrend Period: {self.supertrend_period}, Multiplier: {self.supertrend_multiplier}x")
        print(f"  Heikin Ashi Smoothing: {self.ha_smoothing}")
        print(f"  ATR Period: {self.atr_period}")
        print(f"  Trailing Stop: {self.trailing_stop_atr_mult}x ATR")
        print(f"  Breakeven Trigger: {breakeven_profit_pct}%")
        print(f"  Max Trades Per Day: {self.max_trades_per_day}")
        print(f"  Last Entry Time: {last_entry_time} IST")
        print(f"  Square-off Time: {square_off_time} IST")
        print(f"  Initial Capital: ₹{self.initial_capital:,}")
        print(f"{'='*100}")

        # Auto-detect symbols if not provided
        if symbols is None:
            print("\nAuto-detecting symbols from databases...")
            self.symbols = self.auto_detect_symbols()
        else:
            self.symbols = symbols

        print(f"\nSymbols to backtest: {len(self.symbols)}")

    def parse_square_off_time(self, time_str):
        """Parse square-off time string"""
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
                     reverse=True)[:20]  # Top 20 symbols by data quality

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

        # Resample tick data to specified interval if tick_interval is set
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

        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        # Resample using standard OHLCV aggregation
        resampled = df.resample(interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Remove rows with no data
        resampled = resampled.dropna(subset=['close'])

        # Forward fill any remaining NaN values
        resampled['open'] = resampled['open'].fillna(resampled['close'])
        resampled['high'] = resampled['high'].fillna(resampled['close'])
        resampled['low'] = resampled['low'].fillna(resampled['close'])
        resampled['volume'] = resampled['volume'].fillna(0)

        return resampled

    def calculate_heikin_ashi(self, df):
        """Calculate Heikin Ashi candles with EMA smoothing"""
        ha_df = df.copy()

        # Calculate standard Heikin Ashi
        ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_df['ha_open'] = (df['open'] + df['close']) / 2

        # Calculate subsequent HA opens
        for i in range(1, len(ha_df)):
            ha_df.iloc[i, ha_df.columns.get_loc('ha_open')] = (
                ha_df.iloc[i-1]['ha_open'] + ha_df.iloc[i-1]['ha_close']
            ) / 2

        ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
        ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)

        # Apply EMA smoothing
        if self.ha_smoothing > 1:
            ha_df['ha_open_smooth'] = ha_df['ha_open'].ewm(span=self.ha_smoothing, adjust=False).mean()
            ha_df['ha_close_smooth'] = ha_df['ha_close'].ewm(span=self.ha_smoothing, adjust=False).mean()
            ha_df['ha_high_smooth'] = ha_df['ha_high'].ewm(span=self.ha_smoothing, adjust=False).mean()
            ha_df['ha_low_smooth'] = ha_df['ha_low'].ewm(span=self.ha_smoothing, adjust=False).mean()
        else:
            ha_df['ha_open_smooth'] = ha_df['ha_open']
            ha_df['ha_close_smooth'] = ha_df['ha_close']
            ha_df['ha_high_smooth'] = ha_df['ha_high']
            ha_df['ha_low_smooth'] = ha_df['ha_low']

        # Determine candle color and body size
        ha_df['ha_bullish'] = ha_df['ha_close_smooth'] > ha_df['ha_open_smooth']
        ha_df['ha_bearish'] = ha_df['ha_close_smooth'] < ha_df['ha_open_smooth']
        ha_df['ha_body'] = abs(ha_df['ha_close_smooth'] - ha_df['ha_open_smooth'])

        # Identify significant candles (not dojis)
        avg_body = ha_df['ha_body'].rolling(20).mean()
        ha_df['ha_significant'] = ha_df['ha_body'] > (avg_body * 0.3)

        return ha_df

    def calculate_atr(self, df, period=None):
        """Calculate Average True Range"""
        if period is None:
            period = self.atr_period

        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Calculate ATR using EMA
        df['atr'] = df['tr'].ewm(span=period, adjust=False).mean()

        return df

    def calculate_supertrend(self, df):
        """
        Calculate Supertrend Indicator

        Supertrend is a trend-following indicator that uses ATR to set dynamic support/resistance.
        - When price is above Supertrend line: Bullish trend
        - When price is below Supertrend line: Bearish trend
        """
        # Calculate ATR for Supertrend (using supertrend_period)
        df = self.calculate_atr(df, period=self.supertrend_period)

        # Calculate basic upper and lower bands
        hl_avg = (df['high'] + df['low']) / 2
        df['basic_ub'] = hl_avg + (self.supertrend_multiplier * df['atr'])
        df['basic_lb'] = hl_avg - (self.supertrend_multiplier * df['atr'])

        # Initialize Supertrend columns
        df['supertrend'] = 0.0
        df['supertrend_direction'] = 1  # 1 = bullish, -1 = bearish

        # Calculate Supertrend
        for i in range(1, len(df)):
            # Previous values
            prev_close = df['close'].iloc[i-1]
            prev_supertrend = df['supertrend'].iloc[i-1]
            prev_direction = df['supertrend_direction'].iloc[i-1]

            # Current values
            curr_close = df['close'].iloc[i]
            basic_ub = df['basic_ub'].iloc[i]
            basic_lb = df['basic_lb'].iloc[i]

            # Calculate final bands
            final_ub = basic_ub
            final_lb = basic_lb

            # Adjust upper band
            if basic_ub < df['basic_ub'].iloc[i-1] or prev_close > df['basic_ub'].iloc[i-1]:
                final_ub = basic_ub
            else:
                final_ub = df['basic_ub'].iloc[i-1]

            # Adjust lower band
            if basic_lb > df['basic_lb'].iloc[i-1] or prev_close < df['basic_lb'].iloc[i-1]:
                final_lb = basic_lb
            else:
                final_lb = df['basic_lb'].iloc[i-1]

            # Determine Supertrend and direction
            if prev_direction == 1:
                if curr_close <= final_lb:
                    df.loc[df.index[i], 'supertrend'] = final_ub
                    df.loc[df.index[i], 'supertrend_direction'] = -1
                else:
                    df.loc[df.index[i], 'supertrend'] = final_lb
                    df.loc[df.index[i], 'supertrend_direction'] = 1
            else:
                if curr_close >= final_ub:
                    df.loc[df.index[i], 'supertrend'] = final_lb
                    df.loc[df.index[i], 'supertrend_direction'] = 1
                else:
                    df.loc[df.index[i], 'supertrend'] = final_ub
                    df.loc[df.index[i], 'supertrend_direction'] = -1

        # Store final bands for reference
        df['final_ub'] = df.apply(
            lambda row: row['basic_ub'] if row['supertrend_direction'] == -1 else np.nan,
            axis=1
        )
        df['final_lb'] = df.apply(
            lambda row: row['basic_lb'] if row['supertrend_direction'] == 1 else np.nan,
            axis=1
        )

        # Identify Supertrend signals
        df['st_bullish'] = df['supertrend_direction'] == 1
        df['st_bearish'] = df['supertrend_direction'] == -1

        # Detect trend changes
        df['st_direction_change'] = df['supertrend_direction'] != df['supertrend_direction'].shift(1)
        df['st_buy_signal'] = (df['supertrend_direction'] == 1) & (df['supertrend_direction'].shift(1) == -1)
        df['st_sell_signal'] = (df['supertrend_direction'] == -1) & (df['supertrend_direction'].shift(1) == 1)

        return df

    def generate_signals(self, df):
        """
        Generate trading signals combining Supertrend and Heikin Ashi

        BUY SIGNAL: Supertrend turns bullish + Heikin Ashi confirms
        SELL SIGNAL: Supertrend turns bearish OR Heikin Ashi turns bearish
        """
        # BUY SIGNAL: All conditions must be true
        df['buy_signal'] = (
            (df['st_buy_signal']) &                    # Supertrend just turned bullish
            (df['ha_bullish']) &                       # Heikin Ashi is bullish
            (df['ha_significant']) &                   # Significant candle (not doji)
            (~df['supertrend'].isna()) &              # Valid Supertrend
            (~df['atr'].isna())                       # Valid ATR
        )

        # SELL SIGNAL: Either indicator turns bearish
        df['sell_signal'] = (
            (df['st_sell_signal']) |                   # Supertrend turns bearish
            (df['ha_bearish'])                         # OR Heikin Ashi turns bearish
        )

        return df

    def is_square_off_time(self, timestamp):
        """Check if it's square-off time"""
        try:
            current_time = timestamp.time()
            return current_time >= self.square_off_time
        except:
            return False

    def is_last_entry_time(self, timestamp):
        """Check if it's past last entry time"""
        try:
            current_time = timestamp.time()
            return current_time >= self.last_entry_time
        except:
            return False

    def backtest_single_symbol(self, symbol):
        """Backtest strategy for a single symbol"""
        print(f"\n{'='*100}")
        print(f"Backtesting: {symbol}")
        print(f"{'='*100}")

        # Load data
        df = self.load_data_from_all_databases(symbol)

        if df is None or len(df) < max(self.supertrend_period, self.atr_period, self.ha_smoothing) * 3:
            print(f"Insufficient data for {symbol}. Skipping.")
            return None

        # Store combined data
        self.combined_data[symbol] = df.copy()

        # Calculate all indicators
        df = self.calculate_heikin_ashi(df)
        df = self.calculate_supertrend(df)
        df = self.calculate_atr(df)  # Recalculate ATR with atr_period for trailing stops
        df = self.generate_signals(df)

        # Add trading day info
        df['trading_day'] = df.index.date
        df['is_square_off'] = df.index.map(self.is_square_off_time)
        df['is_last_entry'] = df.index.map(self.is_last_entry_time)

        # Initialize trading variables
        cash = self.initial_capital
        position = 0  # 0: no position, 1: long
        entry_price = 0
        trailing_stop = 0
        breakeven_triggered = False
        trades = []
        entry_time = None
        entry_atr = 0
        trade_number = 0
        daily_trades = {}  # Track trades per day

        # Backtest loop
        for i in range(len(df)):
            current_time = df.index[i]
            current_price = df.iloc[i]['close']
            current_atr = df.iloc[i]['atr']
            is_square_off = df.iloc[i]['is_square_off']
            is_last_entry = df.iloc[i]['is_last_entry']
            current_day = df.iloc[i]['trading_day']

            # Initialize daily trade counter
            if current_day not in daily_trades:
                daily_trades[current_day] = 0

            # Square off at end of day
            if position == 1 and is_square_off:
                shares = int(cash / entry_price) if entry_price > 0 else 0
                trade_pnl = shares * (current_price - entry_price)
                trade_return = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

                trade_number += 1
                duration = (current_time - entry_time).total_seconds() / 60

                print(f"\n[Trade #{trade_number}] SQUARE-OFF @ 3:20 PM")
                print(f"  Entry:  {entry_time.strftime('%Y-%m-%d %H:%M:%S')} @ ₹{entry_price:.2f}")
                print(f"  Exit:   {current_time.strftime('%Y-%m-%d %H:%M:%S')} @ ₹{current_price:.2f}")
                print(f"  Duration: {duration:.1f} min | P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
                print(f"  {'✅ PROFIT' if trade_pnl > 0 else '❌ LOSS'}")

                trades.append({
                    'trade_num': trade_number,
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': shares,
                    'duration_minutes': duration,
                    'pnl': trade_pnl,
                    'return_pct': trade_return,
                    'exit_reason': 'SQUARE_OFF',
                    'atr': current_atr,
                    'supertrend': df.iloc[i]['supertrend'],
                    'breakeven_triggered': breakeven_triggered
                })

                position = 0
                breakeven_triggered = False

            # Entry signal
            elif position == 0 and df.iloc[i]['buy_signal'] and not is_square_off and not is_last_entry:
                # Check max trades per day limit
                if daily_trades[current_day] >= self.max_trades_per_day:
                    continue

                position = 1
                entry_price = current_price
                entry_time = current_time
                entry_atr = current_atr
                # ATR-based trailing stop
                trailing_stop = entry_price - (entry_atr * self.trailing_stop_atr_mult)
                breakeven_triggered = False
                daily_trades[current_day] += 1

            # Exit signals
            elif position == 1:
                # Update trailing stop with ATR
                if current_price > entry_price:
                    new_stop = current_price - (current_atr * self.trailing_stop_atr_mult)

                    # Breakeven protection
                    if current_price >= entry_price * (1 + self.breakeven_profit_pct):
                        if not breakeven_triggered:
                            breakeven_triggered = True
                        # Ensure stop is at least at entry price
                        trailing_stop = max(trailing_stop, entry_price, new_stop)
                    else:
                        trailing_stop = max(trailing_stop, new_stop)

                exit_signal = False
                exit_reason = ""

                # Check exit conditions
                if current_price <= trailing_stop:
                    exit_signal = True
                    exit_reason = "TRAILING_STOP" if not breakeven_triggered else "BREAKEVEN_STOP"
                elif df.iloc[i]['st_sell_signal']:
                    exit_signal = True
                    exit_reason = "SUPERTREND_BEARISH"
                elif df.iloc[i]['ha_bearish']:
                    exit_signal = True
                    exit_reason = "HA_BEARISH"

                if exit_signal:
                    shares = int(cash / entry_price) if entry_price > 0 else 0
                    trade_pnl = shares * (current_price - entry_price)
                    trade_return = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

                    trade_number += 1
                    duration = (current_time - entry_time).total_seconds() / 60

                    print(f"\n[Trade #{trade_number}] {exit_reason}")
                    print(f"  Entry:  {entry_time.strftime('%Y-%m-%d %H:%M:%S')} @ ₹{entry_price:.2f}")
                    print(f"  Exit:   {current_time.strftime('%Y-%m-%d %H:%M:%S')} @ ₹{current_price:.2f}")
                    print(f"  Duration: {duration:.1f} min | P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
                    print(f"  {'✅ PROFIT' if trade_pnl > 0 else '❌ LOSS'}")

                    trades.append({
                        'trade_num': trade_number,
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': shares,
                        'duration_minutes': duration,
                        'pnl': trade_pnl,
                        'return_pct': trade_return,
                        'exit_reason': exit_reason,
                        'atr': current_atr,
                        'supertrend': df.iloc[i]['supertrend'],
                        'breakeven_triggered': breakeven_triggered
                    })

                    position = 0
                    breakeven_triggered = False

        # Calculate metrics
        metrics = self.calculate_metrics(trades)

        return {
            'symbol': symbol,
            'data': df,
            'trades': trades,
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
                'breakeven_exits': 0,
                'trailing_stop_exits': 0,
                'supertrend_exits': 0,
                'ha_bearish_exits': 0,
                'square_off_exits': 0
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

        # Exit reason breakdown
        breakeven_exits = sum(1 for t in trades if 'BREAKEVEN' in t['exit_reason'])
        trailing_stop_exits = sum(1 for t in trades if 'TRAILING' in t['exit_reason'])
        supertrend_exits = sum(1 for t in trades if 'SUPERTREND' in t['exit_reason'])
        ha_bearish_exits = sum(1 for t in trades if 'HA_BEARISH' in t['exit_reason'])
        square_off_exits = sum(1 for t in trades if 'SQUARE_OFF' in t['exit_reason'])

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
            'breakeven_exits': breakeven_exits,
            'trailing_stop_exits': trailing_stop_exits,
            'supertrend_exits': supertrend_exits,
            'ha_bearish_exits': ha_bearish_exits,
            'square_off_exits': square_off_exits
        }

    def run_backtest(self):
        """Run backtest for all symbols"""
        print(f"\n{'='*100}")
        print("STARTING SUPERTREND + HEIKIN ASHI BACKTEST")
        print(f"{'='*100}")

        for symbol in self.symbols:
            try:
                result = self.backtest_single_symbol(symbol)
                if result:
                    self.results[symbol] = result
            except Exception as e:
                print(f"Error backtesting {symbol}: {e}")

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print overall summary"""
        if not self.results:
            print("\nNo results to display.")
            return

        print(f"\n{'='*100}")
        print("BACKTEST SUMMARY - ALL SYMBOLS")
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
                'Avg Return': f"{metrics['avg_return']:.2f}%",
                'Avg Duration': f"{metrics['avg_duration']:.1f} min",
                'ST Exits': metrics['supertrend_exits'],
                'HA Exits': metrics['ha_bearish_exits'],
                'Trail Exits': metrics['trailing_stop_exits']
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        # Overall statistics
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
        print(f"Overall Win Rate: {(winning_trades / total_trades * 100):.1f}%")
        print(f"Profitable Symbols: {profitable_symbols}/{len(self.results)}")

        # Export to CSV
        summary_df.to_csv('supertrend_heikinashi_results.csv', index=False)
        print(f"\n✅ Results exported to: supertrend_heikinashi_results.csv")


# Main execution
if __name__ == "__main__":
    backtester = SupertrendHeikinAshiBacktester(
        data_folder="data/symbolupdate",
        symbols=None,  # Auto-detect
        supertrend_period=10,  # ATR period for Supertrend
        supertrend_multiplier=3.0,  # Multiplier for Supertrend bands
        ha_smoothing=3,  # Heikin Ashi smoothing
        atr_period=14,  # ATR period for trailing stops
        trailing_stop_atr_mult=1.5,  # 1.5x ATR for trailing stop
        breakeven_profit_pct=1.0,  # Move to breakeven at 1% profit
        initial_capital=100000,
        square_off_time="15:20",
        last_entry_time="14:30",
        max_trades_per_day=3,  # Limit trades to avoid overtrading
        tick_interval='30s'  # Options: None (raw), '5s', '10s', '30s', '1min', '5min'
    )

    backtester.run_backtest()
