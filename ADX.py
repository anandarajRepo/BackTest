import sqlite3
import pandas as pd
import numpy as np
import json
import glob
import os
from datetime import datetime, time
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')


class MultiDatabaseDIBacktester:
    def __init__(self, data_folder="data", symbols=None, period=14, volume_threshold_percentile=50,
                 trailing_stop_pct=3.0, initial_capital=100000, square_off_time="15:20",
                 min_data_points=100, breakeven_profit_pct=1.0):
        """
        Initialize the Multi-Database Multi-Symbol DI Crossover Backtester

        Parameters:
        - data_folder: Folder containing database files (default: "data")
        - symbols: List of trading symbols to backtest (if None, auto-detect from databases)
        - period: Period for DI calculation (default 14)
        - volume_threshold_percentile: Volume percentile threshold for good volume (default 50)
        - trailing_stop_pct: Trailing stop loss percentage (default 3%)
        - initial_capital: Starting capital for backtesting (per symbol)
        - square_off_time: Time to square off all positions in HH:MM format (default "15:20" for 3:20 PM)
        - min_data_points: Minimum data points required for a symbol to be included (default 100)
        - breakeven_profit_pct: Profit percentage to trigger breakeven stop (default 1%)
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()
        self.period = period
        self.volume_threshold_percentile = volume_threshold_percentile
        self.trailing_stop_pct = trailing_stop_pct / 100
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.breakeven_profit_pct = breakeven_profit_pct / 100  # Convert to decimal
        self.results = {}
        self.combined_data = {}

        # Set up IST timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print(f"Data folder: {self.data_folder}")
        print(f"Square-off time: {square_off_time} IST")
        print(f"Minimum data points required per symbol: {self.min_data_points}")
        print(f"Breakeven stop trigger: {breakeven_profit_pct}% profit")
        print(f"Found {len(self.db_files)} database files:")
        for db_file in self.db_files:
            print(f"  - {os.path.basename(db_file)}")

        # Auto-detect symbols if not provided
        if symbols is None:
            print("\nAuto-detecting symbols from database files...")
            self.symbols = self.auto_detect_symbols()
        else:
            self.symbols = symbols

        print(f"\nSymbols to backtest: {len(self.symbols)}")
        for symbol in self.symbols:
            print(f"  - {symbol}")

    def parse_square_off_time(self, time_str):
        """Parse square-off time string to time object"""
        try:
            hour, minute = map(int, time_str.split(':'))
            return time(hour, minute)
        except:
            print(f"Invalid square-off time format: {time_str}. Using default 15:20")
            return time(15, 20)

    def find_database_files(self):
        """Find all database files in the data folder"""
        if not os.path.exists(self.data_folder):
            print(f"Data folder '{self.data_folder}' not found!")
            return []

        # Look for .db files in the data folder
        db_pattern = os.path.join(self.data_folder, "*.db")
        db_files = glob.glob(db_pattern)

        if not db_files:
            print(f"No .db files found in '{self.data_folder}' folder!")
            return []

        # Sort files to ensure chronological order
        db_files.sort()
        return db_files

    def auto_detect_symbols(self):
        """Automatically detect all symbols from database files with data quality filtering"""
        all_symbols = set()
        symbol_stats = {}

        print("Scanning database files for symbols...")

        for db_file in self.db_files:
            try:
                print(f"  Scanning {os.path.basename(db_file)}...")
                conn = sqlite3.connect(db_file)

                # Get all unique symbols and their record counts
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

                if not symbols_df.empty:
                    print(f"    Found {len(symbols_df)} symbols with >= {self.min_data_points} records")

                    for _, row in symbols_df.iterrows():
                        symbol = row['symbol']
                        all_symbols.add(symbol)

                        # Track symbol statistics across databases
                        if symbol not in symbol_stats:
                            symbol_stats[symbol] = {
                                'total_records': 0,
                                'databases': [],
                                'first_seen': row['first_record'],
                                'last_seen': row['last_record']
                            }

                        symbol_stats[symbol]['total_records'] += row['record_count']
                        symbol_stats[symbol]['databases'].append(os.path.basename(db_file))

                        # Update date range
                        if row['first_record'] < symbol_stats[symbol]['first_seen']:
                            symbol_stats[symbol]['first_seen'] = row['first_record']
                        if row['last_record'] > symbol_stats[symbol]['last_seen']:
                            symbol_stats[symbol]['last_seen'] = row['last_record']
                else:
                    print(f"    No symbols found with >= {self.min_data_points} records")

            except Exception as e:
                print(f"    Error scanning {db_file}: {e}")
                continue

        # Filter symbols based on data quality
        filtered_symbols = []
        print(f"\nSymbol Quality Analysis:")
        print("-" * 80)
        print(f"{'Symbol':<25} {'Records':<10} {'Databases':<12} {'Date Range'}")
        print("-" * 80)

        # Sort symbols by total records (best data first)
        sorted_symbols = sorted(symbol_stats.items(),
                                key=lambda x: x[1]['total_records'],
                                reverse=True)

        for symbol, stats in sorted_symbols:
            date_range = f"{stats['first_seen'][:10]} to {stats['last_seen'][:10]}"
            databases_count = len(stats['databases'])

            print(f"{symbol:<25} {stats['total_records']:<10} {databases_count:<12} {date_range}")

            # Include symbol if it has sufficient data
            if stats['total_records'] >= self.min_data_points:
                filtered_symbols.append(symbol)

        print("-" * 80)
        print(f"Selected {len(filtered_symbols)} symbols for backtesting")

        if not filtered_symbols:
            print("\nWARNING: No symbols found with sufficient data!")
            print(f"Consider reducing min_data_points (currently {self.min_data_points})")

        return filtered_symbols

    def get_symbol_info(self, symbol):
        """Get detailed information about a specific symbol across all databases"""
        symbol_info = {
            'databases': [],
            'total_records': 0,
            'date_range': None,
            'avg_volume': 0
        }

        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                query = """
                SELECT COUNT(*) as count,
                       MIN(timestamp) as min_date,
                       MAX(timestamp) as max_date,
                       AVG(volume) as avg_vol
                FROM market_data 
                WHERE symbol = ?
                """

                result = pd.read_sql_query(query, conn, params=(symbol,))
                conn.close()

                if result.iloc[0]['count'] > 0:
                    symbol_info['databases'].append(os.path.basename(db_file))
                    symbol_info['total_records'] += result.iloc[0]['count']

                    if symbol_info['date_range'] is None:
                        symbol_info['date_range'] = [result.iloc[0]['min_date'], result.iloc[0]['max_date']]
                    else:
                        if result.iloc[0]['min_date'] < symbol_info['date_range'][0]:
                            symbol_info['date_range'][0] = result.iloc[0]['min_date']
                        if result.iloc[0]['max_date'] > symbol_info['date_range'][1]:
                            symbol_info['date_range'][1] = result.iloc[0]['max_date']

                    symbol_info['avg_volume'] += result.iloc[0]['avg_vol'] or 0

            except Exception as e:
                continue

        if symbol_info['databases']:
            symbol_info['avg_volume'] /= len(symbol_info['databases'])

        return symbol_info

    def load_data_from_all_databases(self, symbol):
        """Load and combine data from all database files for a specific symbol"""
        combined_df = pd.DataFrame()

        for db_file in self.db_files:
            try:
                print(f"  Loading {symbol} from {os.path.basename(db_file)}")
                df = self.load_data_from_single_db(db_file, symbol)

                if df is not None and not df.empty:
                    # Add source database info
                    df['source_db'] = os.path.basename(db_file)
                    combined_df = pd.concat([combined_df, df], ignore_index=False)

            except Exception as e:
                print(f"    Error loading from {db_file}: {e}")
                continue

        if combined_df.empty:
            print(f"  No data found for {symbol} across all databases")
            return None

        # Sort by timestamp and remove duplicates
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        print(f"  Combined {len(combined_df)} data points for {symbol} from {len(self.db_files)} databases")
        return combined_df

    def load_data_from_single_db(self, db_path, symbol):
        """Load data from a single database file"""
        try:
            conn = sqlite3.connect(db_path)

            # First check if symbol exists in this database
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

            # Parse raw_data to get additional fields
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

            # Apply parsing and fill missing values
            raw_parsed = df['raw_data'].apply(parse_raw_data)
            df = pd.concat([df, raw_parsed], axis=1)

            # Use raw data values where available, otherwise use main columns
            df['high'] = df['high_raw'].fillna(df['high_price']).fillna(df['ltp'])
            df['low'] = df['low_raw'].fillna(df['low_price']).fillna(df['ltp'])
            df['close'] = df['close_price'].fillna(df['ltp'])
            df['volume_final'] = df['volume_raw'].fillna(df['volume']).fillna(0)

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            # Remove rows with missing critical data
            df = df.dropna(subset=['close', 'high', 'low'])

            return df[['close', 'high', 'low', 'volume_final']].copy()

        except Exception as e:
            print(f"    Error accessing database {db_path}: {e}")
            return None

    def get_database_date_ranges(self):
        """Get date ranges for each database file"""
        date_ranges = {}

        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                query = "SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM market_data"
                result = pd.read_sql_query(query, conn)
                conn.close()

                if not result.empty:
                    date_ranges[os.path.basename(db_file)] = {
                        'min_date': result.iloc[0]['min_date'],
                        'max_date': result.iloc[0]['max_date'],
                        'file_path': db_file
                    }
            except Exception as e:
                print(f"Error getting date range for {db_file}: {e}")

        return date_ranges

    def calculate_di(self, df):
        """Calculate +DI and -DI indicators"""
        # Calculate True Range (TR)
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Calculate Directional Movements
        df['dm_plus'] = np.where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
            np.maximum(df['high'] - df['high'].shift(1), 0),
            0
        )

        df['dm_minus'] = np.where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
            np.maximum(df['low'].shift(1) - df['low'], 0),
            0
        )

        # Smooth the values using Wilder's smoothing (EMA with alpha = 1/period)
        alpha = 1 / self.period

        df['tr_smooth'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
        df['dm_plus_smooth'] = df['dm_plus'].ewm(alpha=alpha, adjust=False).mean()
        df['dm_minus_smooth'] = df['dm_minus'].ewm(alpha=alpha, adjust=False).mean()

        # Calculate +DI and -DI
        df['di_plus'] = 100 * (df['dm_plus_smooth'] / df['tr_smooth'])
        df['di_minus'] = 100 * (df['dm_minus_smooth'] / df['tr_smooth'])

        # Calculate ADX for reference
        df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
        df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()

        return df

    def identify_good_volume(self, df):
        """Identify periods with good volume based on percentile threshold"""
        volume_threshold = df['volume_final'].quantile(self.volume_threshold_percentile / 100)
        df['good_volume'] = df['volume_final'] > volume_threshold
        return df

    def generate_signals(self, df):
        """Generate buy/sell signals based on DI crossover"""
        df['di_plus_prev'] = df['di_plus'].shift(1)
        df['di_minus_prev'] = df['di_minus'].shift(1)

        # Buy signal: +DI crosses above -DI with good volume
        df['buy_signal'] = (
                (df['di_plus'] > df['di_minus']) &
                (df['di_plus_prev'] <= df['di_minus_prev']) &
                (df['good_volume'])
        )

        # Sell signal: -DI crosses above +DI
        df['sell_signal'] = (
                (df['di_minus'] > df['di_plus']) &
                (df['di_minus_prev'] <= df['di_plus_prev'])
        )

        return df

    def is_square_off_time(self, timestamp):
        """Check if the given timestamp is at or after the square-off time"""
        try:
            # Convert to IST if timezone info is available
            if timestamp.tz is None:
                # Assume the timestamp is already in IST if no timezone info
                ist_time = timestamp
            else:
                ist_time = timestamp.astimezone(self.ist_tz)

            current_time = ist_time.time()
            return current_time >= self.square_off_time
        except:
            # Fallback: use simple time comparison
            return timestamp.time() >= self.square_off_time

    def backtest_single_symbol(self, symbol):
        """Backtest strategy for a single symbol with 3:20 PM IST square-off and breakeven stop"""
        print(f"\nStarting backtest for {symbol}")

        # Load and combine data from all databases
        df = self.load_data_from_all_databases(symbol)

        if df is None or len(df) < self.period * 2:
            print(f"Insufficient data for {symbol}. Skipping.")
            return None

        # Store combined data for later analysis
        self.combined_data[symbol] = df.copy()

        # Calculate indicators
        df = self.calculate_di(df)
        df = self.identify_good_volume(df)
        df = self.generate_signals(df)

        # Add trading day and square-off time information
        df['trading_day'] = df.index.date
        df['is_square_off_time'] = df.index.map(self.is_square_off_time)

        # Initialize trading variables
        cash = self.initial_capital
        position = 0  # 0: no position, 1: long position
        entry_price = 0
        trailing_stop_price = 0
        breakeven_triggered = False  # Track if breakeven has been triggered
        portfolio_value = []
        trades = []
        entry_day = None

        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            current_date = df.index[i]
            current_day = df.iloc[i]['trading_day']
            is_square_off = df.iloc[i]['is_square_off_time']

            # Check if we're at or past square-off time (3:20 PM IST)
            if position == 1 and is_square_off:
                # Force square-off at 3:20 PM IST
                shares = trades[-1]['shares']  # Get shares from last buy
                proceeds = shares * current_price
                cash += proceeds
                position = 0

                trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': round(current_price, 2),
                    'shares': shares,
                    'cash': round(cash, 2),
                    'reason': 'SQUARE_OFF_3_20_PM',
                    'di_plus': round(df.iloc[i]['di_plus'], 2),
                    'di_minus': round(df.iloc[i]['di_minus'], 2),
                    'source_db': df.iloc[i].get('source_db', 'Unknown'),
                    'exit_time': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_price': round(current_price, 2),
                    'entry_price': round(trades[-1]['entry_price'], 2) if trades else 0,
                    'trade_return_pct': round(((current_price - trades[-1]['entry_price']) / trades[-1]['entry_price'] * 100), 2) if trades else 0,
                    'trade_pnl': round((shares * (current_price - trades[-1]['entry_price'])), 2) if trades else 0,
                    'breakeven_triggered': breakeven_triggered
                })

                # Reset for next day
                entry_day = None
                breakeven_triggered = False

            # Check for buy signal (only if not in position and not at/after square-off time)
            elif df.iloc[i]['buy_signal'] and position == 0 and not is_square_off:
                # Enter long position
                shares = int(cash / current_price)
                if shares > 0:
                    position = 1
                    entry_price = current_price
                    entry_day = current_day
                    trailing_stop_price = entry_price * (1 - self.trailing_stop_pct)
                    breakeven_triggered = False
                    cost = shares * current_price
                    cash -= cost

                    trades.append({
                        'date': current_date,
                        'action': 'BUY',
                        'price': round(current_price, 2),
                        'shares': shares,
                        'cash': round(cash, 2),
                        'di_plus': round(df.iloc[i]['di_plus'], 2),
                        'di_minus': round(df.iloc[i]['di_minus'], 2),
                        'source_db': df.iloc[i].get('source_db', 'Unknown'),
                        'entry_day': entry_day,
                        'entry_time': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'entry_price': round(current_price, 2)
                    })

            # Check for sell signals or trailing stop (only if in position and same day and before square-off)
            elif position == 1 and entry_day == current_day and not is_square_off:
                # Update trailing stop with breakeven protection
                if current_price > entry_price:
                    # Calculate new trailing stop based on current price
                    new_trailing_stop = current_price * (1 - self.trailing_stop_pct)

                    # If position is profitable by more than breakeven_profit_pct, move stop to breakeven (entry price)
                    if current_price >= entry_price * (1 + self.breakeven_profit_pct):
                        if not breakeven_triggered:
                            breakeven_triggered = True
                            print(f"  {symbol}: Breakeven triggered at {current_price:.2f} (entry: {entry_price:.2f})")
                        # Ensure stop loss is at least at entry price (breakeven)
                        trailing_stop_price = max(trailing_stop_price, entry_price)

                    # Update trailing stop to highest value
                    trailing_stop_price = max(trailing_stop_price, new_trailing_stop)

                # Check exit conditions (excluding square-off as it's handled above)
                should_exit = (
                        df.iloc[i]['sell_signal'] or  # DI crossover sell signal
                        current_price <= trailing_stop_price  # Trailing stop hit
                )

                if should_exit:
                    # Determine exit reason
                    if df.iloc[i]['sell_signal']:
                        exit_reason = 'SELL_SIGNAL'
                    elif breakeven_triggered and abs(current_price - entry_price) < 0.01:
                        exit_reason = 'BREAKEVEN_STOP'
                    else:
                        exit_reason = 'TRAILING_STOP'

                    # Exit position
                    shares = trades[-1]['shares']  # Get shares from last buy
                    proceeds = shares * current_price
                    cash += proceeds
                    position = 0

                    trades.append({
                        'date': current_date,
                        'action': 'SELL',
                        'price': round(current_price, 2),
                        'shares': shares,
                        'cash': round(cash, 2),
                        'reason': exit_reason,
                        'di_plus': round(df.iloc[i]['di_plus'], 2),
                        'di_minus': round(df.iloc[i]['di_minus'], 2),
                        'source_db': df.iloc[i].get('source_db', 'Unknown'),
                        'exit_time': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'exit_price': round(current_price, 2),
                        'entry_price': round(trades[-1]['entry_price'], 2) if trades else 0,
                        'trade_return_pct': round(((current_price - trades[-1]['entry_price']) / trades[-1]['entry_price'] * 100), 2) if trades else 0,
                        'trade_pnl': round((shares * (current_price - trades[-1]['entry_price'])), 2) if trades else 0,
                        'breakeven_triggered': breakeven_triggered
                    })

                    # Reset for potential new trade same day
                    entry_day = None
                    breakeven_triggered = False

            # Calculate portfolio value
            if position == 1:
                shares = trades[-1]['shares']
                portfolio_val = cash + (shares * current_price)
            else:
                portfolio_val = cash

            portfolio_value.append({
                'date': current_date,
                'value': round(portfolio_val, 2),
                'position': position,
                'trading_day': current_day,
                'is_square_off_time': is_square_off
            })

        # Final check - ensure no overnight positions
        if position == 1:
            print(f"Warning: {symbol} ended with open position. This should not happen with square-off strategy.")

        # Calculate metrics
        portfolio_df = pd.DataFrame(portfolio_value)
        metrics = self.calculate_metrics(portfolio_df, trades)

        # Add intraday-specific metrics
        intraday_metrics = self.calculate_intraday_metrics(trades)
        metrics.update(intraday_metrics)

        return {
            'symbol': symbol,
            'data': df,
            'portfolio': portfolio_df,
            'trades': trades,
            'metrics': metrics,
            'data_summary': self.get_data_summary(df)
        }

    def calculate_intraday_metrics(self, trades):
        """Calculate intraday-specific performance metrics including 3:20 PM square-offs and breakeven stops"""
        if not trades:
            return {
                'total_intraday_trades': 0,
                'avg_trade_duration_minutes': 0,
                'same_day_trades': 0,
                'square_off_3_20_closures': 0,
                'signal_exits': 0,
                'trailing_stop_exits': 0,
                'breakeven_stop_exits': 0,
                'intraday_win_rate': 0,
                'avg_intraday_return': 0
            }

        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']

        # Calculate trade durations and categorize exits
        trade_durations = []
        same_day_count = 0
        square_off_3_20_count = 0
        signal_exit_count = 0
        trailing_stop_count = 0
        breakeven_stop_count = 0
        intraday_returns = []

        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_trade = buy_trades[i]
            sell_trade = sell_trades[i]

            # Calculate duration
            buy_time = pd.to_datetime(buy_trade['date'])
            sell_time = pd.to_datetime(sell_trade['date'])
            duration_minutes = (sell_time - buy_time).total_seconds() / 60
            trade_durations.append(duration_minutes)

            # Check if same day
            if buy_time.date() == sell_time.date():
                same_day_count += 1

            # Categorize exit reasons
            exit_reason = sell_trade.get('reason', 'UNKNOWN')
            if exit_reason == 'SQUARE_OFF_3_20_PM':
                square_off_3_20_count += 1
            elif exit_reason == 'SELL_SIGNAL':
                signal_exit_count += 1
            elif exit_reason == 'BREAKEVEN_STOP':
                breakeven_stop_count += 1
            elif exit_reason == 'TRAILING_STOP':
                trailing_stop_count += 1

            # Calculate return
            trade_return = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
            intraday_returns.append(trade_return)

        # Calculate metrics
        avg_duration = np.mean(trade_durations) if trade_durations else 0
        intraday_win_rate = sum(1 for r in intraday_returns if r > 0) / len(intraday_returns) if intraday_returns else 0
        avg_intraday_return = np.mean(intraday_returns) if intraday_returns else 0

        return {
            'total_intraday_trades': len(buy_trades),
            'avg_trade_duration_minutes': round(avg_duration, 2),
            'same_day_trades': same_day_count,
            'square_off_3_20_closures': square_off_3_20_count,
            'signal_exits': signal_exit_count,
            'trailing_stop_exits': trailing_stop_count,
            'breakeven_stop_exits': breakeven_stop_count,
            'intraday_win_rate': round(intraday_win_rate, 4),
            'avg_intraday_return': round(avg_intraday_return, 4)
        }

    def get_data_summary(self, df):
        """Get summary statistics for the data"""
        unique_days = len(df['trading_day'].unique()) if 'trading_day' in df.columns else len(df.index.date)

        return {
            'total_records': len(df),
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'trading_days': unique_days,
            'avg_daily_volume': round(df['volume_final'].mean(), 2),
            'databases_used': df['source_db'].nunique() if 'source_db' in df.columns else 0,
            'avg_records_per_day': round(len(df) / unique_days, 2) if unique_days > 0 else 0
        }

    def calculate_metrics(self, portfolio_df, trades):
        """Calculate performance metrics"""
        total_return = (portfolio_df.iloc[-1]['value'] - self.initial_capital) / self.initial_capital

        # Calculate trade-based metrics
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']

        trade_returns = []
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_price = buy_trades[i]['price']
            sell_price = sell_trades[i]['price']
            trade_return = (sell_price - buy_price) / buy_price
            trade_returns.append(trade_return)

        if trade_returns:
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
            avg_trade_return = np.mean(trade_returns)
            max_trade_return = max(trade_returns)
            min_trade_return = min(trade_returns)
        else:
            win_rate = avg_trade_return = max_trade_return = min_trade_return = 0

        # Calculate portfolio-based metrics
        portfolio_df['returns'] = portfolio_df['value'].pct_change()
        sharpe_ratio = portfolio_df['returns'].mean() / portfolio_df['returns'].std() * np.sqrt(252) if portfolio_df['returns'].std() > 0 else 0

        max_dd = self.calculate_max_drawdown(portfolio_df['value'])

        return {
            'total_return': round(total_return, 4),
            'total_trades': len(buy_trades),
            'win_rate': round(win_rate, 4),
            'avg_trade_return': round(avg_trade_return, 4),
            'max_trade_return': round(max_trade_return, 4),
            'min_trade_return': round(min_trade_return, 4),
            'sharpe_ratio': round(sharpe_ratio, 4),
            'max_drawdown': round(max_dd, 4),
            'final_value': round(portfolio_df.iloc[-1]['value'], 2)
        }

    def calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return abs(drawdown.min())

    def run_backtest_sequential(self):
        """Run backtest for all symbols sequentially"""
        print(f"\nStarting multi-database backtest for {len(self.symbols)} symbols")
        print(f"Parameters: Period={self.period}, Volume Threshold={self.volume_threshold_percentile}th percentile")
        print(f"Trailing Stop={self.trailing_stop_pct * 100}%, Breakeven Trigger={self.breakeven_profit_pct * 100}%")
        print(f"Database files: {len(self.db_files)}")

        for symbol in self.symbols:
            try:
                result = self.backtest_single_symbol(symbol)
                if result:
                    self.results[symbol] = result
            except Exception as exc:
                print(f'{symbol} generated an exception: {exc}')
                import traceback
                traceback.print_exc()

        return self.results

    def create_summary_report(self):
        """Create a summary report of all backtests with 3:20 PM square-off metrics"""
        if not self.results:
            print("No results to summarize.")
            return None

        summary_data = []
        for symbol, result in self.results.items():
            metrics = result['metrics']
            data_summary = result['data_summary']

            summary_data.append({
                'Symbol': symbol.split(':')[1].replace('-EQ', '') if ':' in symbol else symbol.replace('-EQ', ''),  # Clean symbol name
                'Total Return (%)': metrics['total_return'] * 100,
                'Final Value': metrics['final_value'],
                'Total Trades': metrics['total_trades'],
                'Win Rate (%)': metrics['win_rate'] * 100,
                'Avg Trade Return (%)': metrics['avg_trade_return'] * 100,
                'Best Trade (%)': metrics['max_trade_return'] * 100,
                'Worst Trade (%)': metrics['min_trade_return'] * 100,
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Max Drawdown (%)': metrics['max_drawdown'] * 100,
                'Intraday Trades': metrics.get('total_intraday_trades', 0),
                'Same Day Trades': metrics.get('same_day_trades', 0),
                '3:20 PM Square-offs': metrics.get('square_off_3_20_closures', 0),
                'Signal Exits': metrics.get('signal_exits', 0),
                'Trailing Stop Exits': metrics.get('trailing_stop_exits', 0),
                'Breakeven Stop Exits': metrics.get('breakeven_stop_exits', 0),
                'Avg Duration (min)': metrics.get('avg_trade_duration_minutes', 0),
                'Intraday Win Rate (%)': metrics.get('intraday_win_rate', 0) * 100,
                'Data Points': data_summary['total_records'],
                'Trading Days': data_summary['trading_days'],
                'Avg Records/Day': data_summary.get('avg_records_per_day', 0)
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Total Return (%)', ascending=False)

        return summary_df

    def create_database_analysis_report(self):
        """Create analysis report for database coverage"""
        print("\n" + "=" * 80)
        print("DATABASE ANALYSIS REPORT")
        print("=" * 80)

        # Get date ranges for each database
        date_ranges = self.get_database_date_ranges()

        print(f"\nDatabase Files Found: {len(self.db_files)}")
        print("-" * 50)

        for db_name, info in date_ranges.items():
            print(f"File: {db_name}")
            print(f"  Date Range: {info['min_date']} to {info['max_date']}")
            print(f"  Full Path: {info['file_path']}")
            print()

        # Analyze symbol coverage across databases
        if self.combined_data:
            print("Symbol Data Coverage:")
            print("-" * 30)

            for symbol, df in self.combined_data.items():
                if 'source_db' in df.columns:
                    db_counts = df['source_db'].value_counts()
                    print(f"\n{symbol}:")
                    for db_file, count in db_counts.items():
                        print(f"  {db_file}: {count} records")
                    print(f"  Total: {len(df)} records")
                    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    def export_results(self, filename_prefix="multi_db_di_crossover_breakeven"):
        """Export all results to CSV files"""
        if not self.results:
            print("No results to export.")
            return

        # Export summary
        summary_df = self.create_summary_report()
        summary_filename = f"{filename_prefix}_summary.csv"
        summary_df.to_csv(summary_filename, index=False)
        print(f"Summary exported to {summary_filename}")

        # Export individual trades for each symbol
        all_trades = []
        for symbol, result in self.results.items():
            trades_df = pd.DataFrame(result['trades'])
            if not trades_df.empty:
                trades_df['symbol'] = symbol
                all_trades.append(trades_df)

        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)
            trades_filename = f"{filename_prefix}_all_trades.csv"
            combined_trades.to_csv(trades_filename, index=False)
            print(f"All trades exported to {trades_filename}")

        # Export database analysis
        date_ranges = self.get_database_date_ranges()
        if date_ranges:
            db_analysis = pd.DataFrame.from_dict(date_ranges, orient='index')
            db_filename = f"{filename_prefix}_database_analysis.csv"
            db_analysis.to_csv(db_filename)
            print(f"Database analysis exported to {db_filename}")

        # Export symbol detection analysis
        self.export_symbol_analysis(filename_prefix)

    def export_symbol_analysis(self, filename_prefix):
        """Export symbol detection and quality analysis"""
        symbol_analysis = []

        for symbol in self.symbols:
            symbol_info = self.get_symbol_info(symbol)

            symbol_analysis.append({
                'Symbol': symbol,
                'Clean_Name': symbol.split(':')[1].replace('-EQ', '') if ':' in symbol else symbol.replace('-EQ', ''),
                'Total_Records': symbol_info['total_records'],
                'Databases_Count': len(symbol_info['databases']),
                'Databases': ', '.join(symbol_info['databases']),
                'Date_Range_Start': symbol_info['date_range'][0] if symbol_info['date_range'] else '',
                'Date_Range_End': symbol_info['date_range'][1] if symbol_info['date_range'] else '',
                'Avg_Volume': round(symbol_info['avg_volume'], 2),
                'Backtested': symbol in self.results
            })

        symbol_df = pd.DataFrame(symbol_analysis)
        symbol_df = symbol_df.sort_values('Total_Records', ascending=False)

        symbol_filename = f"{filename_prefix}_symbol_analysis.csv"
        symbol_df.to_csv(symbol_filename, index=False)
        print(f"Symbol analysis exported to {symbol_filename}")


# Example usage
if __name__ == "__main__":
    # Initialize multi-database backtester with AUTO-DETECTION of symbols and BREAKEVEN STOP
    backtester = MultiDatabaseDIBacktester(
        data_folder="data/symbolupdate",  # Folder containing database files
        symbols=None,  # Auto-detect symbols from databases (set to None)
        period=60,  # DI calculation period
        volume_threshold_percentile=60,  # Volume filter (30th percentile)
        trailing_stop_pct=6,  # 6% trailing stop loss
        initial_capital=100000,  # Starting capital per symbol
        square_off_time="15:20",  # 3:20 PM IST square-off time
        min_data_points=100,  # Minimum data points required per symbol
        breakeven_profit_pct=1.0  # Move stop to breakeven at 1% profit
    )

    print("=" * 100)
    print("AUTO-DETECTED SYMBOLS WITH BREAKEVEN STOP - 3:20 PM IST SQUARE-OFF DI CROSSOVER STRATEGY")
    print("=" * 100)
    print("Strategy Rules:")
    print("1. Symbols automatically detected from database files")
    print("2. Only symbols with sufficient data included")
    print("3. All positions MUST be squared off at 3:20 PM IST daily")
    print("4. No overnight/positional trades allowed")
    print("5. Automatic square-off at 3:20 PM regardless of P&L")
    print("6. Buy signals ignored at or after 3:20 PM")
    print("7. Multiple intraday trades allowed before 3:20 PM")
    print("8. BREAKEVEN STOP: When position is up 1%, stop loss moves to entry price")
    print("9. Trailing stop continues to trail above breakeven level")
    print("10. Data read from 'data' folder containing .db files")
    print("=" * 100)

    try:
        # Check if data folder and databases exist
        if not backtester.db_files:
            print("No database files found!")
            print("Please ensure:")
            print("1. Create a 'data' folder in the script directory")
            print("2. Place your .db files in the 'data' folder")
            print("3. Expected files: fyers_market_data_*.db")
            exit(1)

        # Check if symbols were detected
        if not backtester.symbols:
            print("No symbols detected with sufficient data!")
            print("Please check:")
            print("1. Database files contain market_data table")
            print("2. Reduce min_data_points if needed")
            print("3. Verify database schema matches expected format")
            exit(1)

        # Create database analysis report
        backtester.create_database_analysis_report()

        # Run backtest
        print(f"\nStarting 3:20 PM square-off backtest with breakeven stop for {len(backtester.symbols)} auto-detected symbols...")
        results = backtester.run_backtest_sequential()

        if results:
            print(f"\nBacktest completed for {len(results)} symbols across {len(backtester.db_files)} databases")

            # Create and display summary report
            summary_df = backtester.create_summary_report()
            print("\n" + "=" * 150)
            print("AUTO-DETECTED SYMBOLS WITH BREAKEVEN STOP - INTRADAY 3:20 PM SQUARE-OFF STRATEGY BACKTEST SUMMARY")
            print("=" * 150)
            print(summary_df.round(2).to_string(index=False))

            # Export results
            print("\nExporting results to CSV files...")
            backtester.export_results("auto_detected_breakeven_intraday_3_20_square_off_strategy")

            # Performance ranking
            print("\n" + "=" * 100)
            print("TOP PERFORMERS BY TOTAL RETURN (WITH BREAKEVEN STOP):")
            print("=" * 100)
            top_performers = summary_df.head(5)[['Symbol', 'Total Return (%)', 'Intraday Win Rate (%)', 'Breakeven Stop Exits', '3:20 PM Square-offs', 'Avg Duration (min)']]
            print(top_performers.to_string(index=False))

            print("\n" + "=" * 100)
            print("BREAKEVEN STOP EFFECTIVENESS ANALYSIS:")
            print("=" * 100)

            # Calculate overall strategy statistics
            total_symbols = len(results)
            profitable_symbols = len([r for r in results.values() if r['metrics']['total_return'] > 0])
            avg_return = summary_df['Total Return (%)'].mean()
            avg_win_rate = summary_df['Intraday Win Rate (%)'].mean()
            avg_trades = summary_df['Intraday Trades'].mean()
            avg_duration = summary_df['Avg Duration (min)'].mean()
            total_same_day_trades = summary_df['Same Day Trades'].sum()
            total_square_offs = summary_df['3:20 PM Square-offs'].sum()
            total_signal_exits = summary_df['Signal Exits'].sum()
            total_trailing_stops = summary_df['Trailing Stop Exits'].sum()
            total_breakeven_stops = summary_df['Breakeven Stop Exits'].sum()

            print(f"Symbols Auto-Detected: {len(backtester.symbols)}")
            print(f"Symbols Successfully Backtested: {total_symbols}")
            print(f"Profitable Symbols: {profitable_symbols} ({profitable_symbols / total_symbols * 100:.1f}%)")
            print(f"Average Return: {avg_return:.2f}%")
            print(f"Average Intraday Win Rate: {avg_win_rate:.1f}%")
            print(f"Average Intraday Trades per Symbol: {avg_trades:.1f}")
            print(f"Average Trade Duration: {avg_duration:.1f} minutes")
            print(f"Total Same-Day Trades: {total_same_day_trades}")
            print(f"Total 3:20 PM Square-offs: {total_square_offs}")
            print(f"Total Signal-based Exits: {total_signal_exits}")
            print(f"Total Trailing Stop Exits: {total_trailing_stops}")
            print(f"Total Breakeven Stop Exits: {total_breakeven_stops}")

            # Calculate exit breakdown percentages
            total_exits = total_square_offs + total_signal_exits + total_trailing_stops + total_breakeven_stops
            if total_exits > 0:
                print(f"\nExit Breakdown:")
                print(f"  3:20 PM Square-offs: {total_square_offs} ({total_square_offs / total_exits * 100:.1f}%)")
                print(f"  Signal-based Exits: {total_signal_exits} ({total_signal_exits / total_exits * 100:.1f}%)")
                print(f"  Trailing Stop Exits: {total_trailing_stops} ({total_trailing_stops / total_exits * 100:.1f}%)")
                print(f"  Breakeven Stop Exits: {total_breakeven_stops} ({total_breakeven_stops / total_exits * 100:.1f}%)")

            # Breakeven stop insights
            if total_breakeven_stops > 0:
                print(f"\nBreakeven Stop Impact:")
                print(f"  Total trades protected by breakeven: {total_breakeven_stops}")
                print(f"  Percentage of all exits: {total_breakeven_stops / total_exits * 100:.1f}%")
                print(f"  This feature prevented potential losses on trades that reached 1% profit")

            # Strategy insights
            print("\n" + "=" * 100)
            print("STRATEGY INSIGHTS:")
            print("=" * 100)

            # Most breakeven-dependent symbols
            if 'Breakeven Stop Exits' in summary_df.columns:
                high_breakeven = summary_df.nlargest(3, 'Breakeven Stop Exits')[['Symbol', 'Breakeven Stop Exits', 'Total Return (%)']]
                print("Symbols with Most Breakeven Stop Exits:")
                print(high_breakeven.to_string(index=False))

            # Most square-off dependent symbols
            high_square_off = summary_df.nlargest(3, '3:20 PM Square-offs')[['Symbol', '3:20 PM Square-offs', 'Total Return (%)']]
            best_signal_exits = summary_df.nlargest(3, 'Signal Exits')[['Symbol', 'Signal Exits', 'Total Return (%)']]

            print("\nSymbols with Most 3:20 PM Square-offs:")
            print(high_square_off.to_string(index=False))

            print("\nSymbols with Most Signal-based Exits:")
            print(best_signal_exits.to_string(index=False))

            # Time efficiency analysis
            fastest_trades = summary_df.nsmallest(3, 'Avg Duration (min)')[['Symbol', 'Avg Duration (min)', 'Intraday Win Rate (%)']]
            print("\nFastest Average Trade Execution:")
            print(fastest_trades.to_string(index=False))

            # Database coverage analysis
            print("\n" + "=" * 100)
            print("DATABASE COVERAGE SUMMARY:")
            print("=" * 100)

            total_data_points = summary_df['Data Points'].sum()
            total_trading_days = summary_df['Trading Days'].sum()

            print(f"Total Data Points Processed: {total_data_points:,}")
            print(f"Total Trading Days Covered: {total_trading_days}")
            print(f"Databases Used: {len(backtester.db_files)}")
            print(f"Data Folder: {backtester.data_folder}")

            # Best performing symbol details
            best_symbol = summary_df.iloc[0]['Symbol']
            best_return = summary_df.iloc[0]['Total Return (%)']
            best_square_offs = summary_df.iloc[0]['3:20 PM Square-offs']
            best_breakeven = summary_df.iloc[0]['Breakeven Stop Exits']
            print(f"\nBest Performing Symbol: {best_symbol} (+{best_return:.2f}%)")
            print(f"3:20 PM Square-offs: {best_square_offs}, Breakeven Stops: {best_breakeven}")

            print("\n" + "=" * 100)
            print("FILES EXPORTED:")
            print("=" * 100)
            print("1. auto_detected_breakeven_intraday_3_20_square_off_strategy_summary.csv - Performance summary")
            print("2. auto_detected_breakeven_intraday_3_20_square_off_strategy_all_trades.csv - All trade details")
            print("3. auto_detected_breakeven_intraday_3_20_square_off_strategy_database_analysis.csv - Database info")
            print("4. auto_detected_breakeven_intraday_3_20_square_off_strategy_symbol_analysis.csv - Symbol detection analysis")
            print("=" * 100)

        else:
            print("No successful backtests completed.")
            print("Please check:")
            print("1. Database files exist in 'data' folder")
            print("2. Symbols have sufficient data (min_data_points)")
            print("3. Database schema matches expected format")
            print("4. Sufficient data available for analysis")

    except Exception as e:
        print(f"Error running auto-detected symbols with breakeven stop backtest: {e}")
        import traceback

        traceback.print_exc()