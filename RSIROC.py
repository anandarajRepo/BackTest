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


class RSIROCScalpingBacktester:
    def __init__(self, data_folder="data", symbols=None, rsi_period=14, roc_period=10,
                 rsi_threshold=70, roc_lookback_period=50, trailing_stop_pct=0.3,
                 initial_capital=100000, square_off_time="15:20", min_data_points=100):
        """
        Initialize the RSI & ROC Scalping Strategy Backtester

        Parameters:
        - data_folder: Folder containing database files (default: "data")
        - symbols: List of trading symbols to backtest (if None, auto-detect from databases)
        - rsi_period: Period for RSI calculation (default 14)
        - roc_period: Period for ROC calculation (default 10)
        - rsi_threshold: RSI threshold for entry signal (default 70)
        - roc_lookback_period: Period to calculate ROC average (default 50)
        - trailing_stop_pct: Trailing stop loss percentage (default 0.3%)
        - initial_capital: Starting capital for backtesting (per symbol)
        - square_off_time: Time to square off all positions in HH:MM format (default "15:20")
        - min_data_points: Minimum data points required for a symbol to be included
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()
        self.rsi_period = rsi_period
        self.roc_period = roc_period
        self.rsi_threshold = rsi_threshold
        self.roc_lookback_period = roc_lookback_period
        self.trailing_stop_pct = trailing_stop_pct / 100
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.results = {}
        self.combined_data = {}

        # Set up IST timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print(f"RSI & ROC Scalping Strategy Parameters:")
        print(f"  Data folder: {self.data_folder}")
        print(f"  RSI Period: {self.rsi_period}")
        print(f"  ROC Period: {self.roc_period}")
        print(f"  RSI Threshold: {self.rsi_threshold}")
        print(f"  ROC Lookback for Average: {self.roc_lookback_period}")
        print(f"  Trailing Stop: {trailing_stop_pct}%")
        print(f"  Square-off time: {square_off_time} IST")
        print(f"  Minimum data points: {self.min_data_points}")

        print(f"\nFound {len(self.db_files)} database files:")
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

        db_pattern = os.path.join(self.data_folder, "*.db")
        db_files = glob.glob(db_pattern)

        if not db_files:
            print(f"No .db files found in '{self.data_folder}' folder!")
            return []

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

                        if symbol not in symbol_stats:
                            symbol_stats[symbol] = {
                                'total_records': 0,
                                'databases': [],
                                'first_seen': row['first_record'],
                                'last_seen': row['last_record']
                            }

                        symbol_stats[symbol]['total_records'] += row['record_count']
                        symbol_stats[symbol]['databases'].append(os.path.basename(db_file))

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

        sorted_symbols = sorted(symbol_stats.items(),
                                key=lambda x: x[1]['total_records'],
                                reverse=True)

        for symbol, stats in sorted_symbols:
            date_range = f"{stats['first_seen'][:10]} to {stats['last_seen'][:10]}"
            databases_count = len(stats['databases'])

            print(f"{symbol:<25} {stats['total_records']:<10} {databases_count:<12} {date_range}")

            if stats['total_records'] >= self.min_data_points:
                filtered_symbols.append(symbol)

        print("-" * 80)
        print(f"Selected {len(filtered_symbols)} symbols for backtesting")

        if not filtered_symbols:
            print("\nWARNING: No symbols found with sufficient data!")
            print(f"Consider reducing min_data_points (currently {self.min_data_points})")

        return filtered_symbols

    def load_data_from_all_databases(self, symbol):
        """Load and combine data from all database files for a specific symbol"""
        combined_df = pd.DataFrame()

        for db_file in self.db_files:
            try:
                print(f"  Loading {symbol} from {os.path.basename(db_file)}")
                df = self.load_data_from_single_db(db_file, symbol)

                if df is not None and not df.empty:
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

        print(f"  Combined {len(combined_df)} data points for {symbol}")
        return combined_df

    def load_data_from_single_db(self, db_path, symbol):
        """Load data from a single database file"""
        try:
            conn = sqlite3.connect(db_path)

            # Check if symbol exists
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

            # Parse raw_data for additional fields
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

            # Use best available data
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

    def calculate_rsi(self, df):
        """Calculate RSI (Relative Strength Index)"""
        # Calculate price changes
        df['price_change'] = df['close'].diff()

        # Separate gains and losses
        df['gain'] = np.where(df['price_change'] > 0, df['price_change'], 0)
        df['loss'] = np.where(df['price_change'] < 0, -df['price_change'], 0)

        # Calculate rolling averages using Wilder's smoothing
        alpha = 1 / self.rsi_period
        df['avg_gain'] = df['gain'].ewm(alpha=alpha, adjust=False).mean()
        df['avg_loss'] = df['loss'].ewm(alpha=alpha, adjust=False).mean()

        # Calculate RS and RSI
        df['rs'] = df['avg_gain'] / df['avg_loss']
        df['rsi'] = 100 - (100 / (1 + df['rs']))

        return df

    def calculate_roc(self, df):
        """Calculate ROC (Rate of Change) and its rolling average"""
        # Calculate ROC
        df['roc'] = ((df['close'] - df['close'].shift(self.roc_period)) / df['close'].shift(self.roc_period)) * 100

        # Calculate rolling average of ROC
        df['roc_avg'] = df['roc'].rolling(window=self.roc_lookback_period).mean()

        return df

    def generate_signals(self, df):
        """Generate buy/sell signals based on RSI > 70 and ROC > ROC_average"""
        # Buy signal: RSI > threshold AND ROC > ROC_average
        df['buy_signal'] = (
                (df['rsi'] > self.rsi_threshold) &
                (df['roc'] > df['roc_avg']) &
                (~df['rsi'].isna()) &
                (~df['roc'].isna()) &
                (~df['roc_avg'].isna())
        )

        # For scalping, we'll mainly rely on trailing stop for exits
        # But we can add a condition where RSI drops significantly
        df['sell_signal'] = False  # We'll primarily use trailing stops

        return df

    def is_square_off_time(self, timestamp):
        """Check if the given timestamp is at or after the square-off time"""
        try:
            if timestamp.tz is None:
                ist_time = timestamp
            else:
                ist_time = timestamp.astimezone(self.ist_tz)

            current_time = ist_time.time()
            return current_time >= self.square_off_time
        except:
            return timestamp.time() >= self.square_off_time

    def backtest_single_symbol(self, symbol):
        """Backtest RSI & ROC scalping strategy for a single symbol"""
        print(f"\nStarting RSI & ROC scalping backtest for {symbol}")

        # Load data
        df = self.load_data_from_all_databases(symbol)

        if df is None or len(df) < max(self.rsi_period, self.roc_period, self.roc_lookback_period) * 2:
            print(f"Insufficient data for {symbol}. Skipping.")
            return None

        # Store combined data
        self.combined_data[symbol] = df.copy()

        # Calculate indicators
        df = self.calculate_rsi(df)
        df = self.calculate_roc(df)
        df = self.generate_signals(df)

        # Add trading day and square-off time information
        df['trading_day'] = df.index.date
        df['is_square_off_time'] = df.index.map(self.is_square_off_time)

        # Initialize trading variables
        cash = self.initial_capital
        position = 0  # 0: no position, 1: long position
        entry_price = 0
        trailing_stop_price = 0
        portfolio_value = []
        trades = []
        entry_day = None

        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            current_date = df.index[i]
            current_day = df.iloc[i]['trading_day']
            is_square_off = df.iloc[i]['is_square_off_time']

            # Force square-off at 3:20 PM IST
            if position == 1 and is_square_off:
                shares = trades[-1]['shares']
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
                    'rsi': round(df.iloc[i]['rsi'], 2),
                    'roc': round(df.iloc[i]['roc'], 2),
                    'roc_avg': round(df.iloc[i]['roc_avg'], 2),
                    'source_db': df.iloc[i].get('source_db', 'Unknown'),
                    'exit_time': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_price': round(current_price, 2),
                    'entry_price': round(trades[-1]['entry_price'], 2) if trades else 0,
                    'trade_return_pct': round(((current_price - trades[-1]['entry_price']) / trades[-1]['entry_price'] * 100), 2) if trades else 0,
                    'trade_pnl': round((shares * (current_price - trades[-1]['entry_price'])), 2) if trades else 0
                })

                entry_day = None

            # Check for buy signal (only if not in position and not at/after square-off time)
            elif df.iloc[i]['buy_signal'] and position == 0 and not is_square_off:
                shares = int(cash / current_price)
                if shares > 0:
                    position = 1
                    entry_price = current_price
                    entry_day = current_day
                    trailing_stop_price = entry_price * (1 - self.trailing_stop_pct)
                    cost = shares * current_price
                    cash -= cost

                    trades.append({
                        'date': current_date,
                        'action': 'BUY',
                        'price': round(current_price, 2),
                        'shares': shares,
                        'cash': round(cash, 2),
                        'rsi': round(df.iloc[i]['rsi'], 2),
                        'roc': round(df.iloc[i]['roc'], 2),
                        'roc_avg': round(df.iloc[i]['roc_avg'], 2),
                        'source_db': df.iloc[i].get('source_db', 'Unknown'),
                        'entry_day': entry_day,
                        'entry_time': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'entry_price': round(current_price, 2)
                    })

            # Check for trailing stop exit (only if in position, same day, and before square-off)
            elif position == 1 and entry_day == current_day and not is_square_off:
                # Update trailing stop if price moved up
                if current_price > entry_price:
                    new_trailing_stop = current_price * (1 - self.trailing_stop_pct)
                    trailing_stop_price = max(trailing_stop_price, new_trailing_stop)

                # Check if trailing stop hit
                if current_price <= trailing_stop_price:
                    shares = trades[-1]['shares']
                    proceeds = shares * current_price
                    cash += proceeds
                    position = 0

                    trades.append({
                        'date': current_date,
                        'action': 'SELL',
                        'price': round(current_price, 2),
                        'shares': shares,
                        'cash': round(cash, 2),
                        'reason': 'TRAILING_STOP',
                        'rsi': round(df.iloc[i]['rsi'], 2),
                        'roc': round(df.iloc[i]['roc'], 2),
                        'roc_avg': round(df.iloc[i]['roc_avg'], 2),
                        'source_db': df.iloc[i].get('source_db', 'Unknown'),
                        'exit_time': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'exit_price': round(current_price, 2),
                        'entry_price': round(trades[-1]['entry_price'], 2) if trades else 0,
                        'trade_return_pct': round(((current_price - trades[-1]['entry_price']) / trades[-1]['entry_price'] * 100), 2) if trades else 0,
                        'trade_pnl': round((shares * (current_price - trades[-1]['entry_price'])), 2) if trades else 0
                    })

                    entry_day = None

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
                'is_square_off_time': is_square_off,
                'rsi': round(df.iloc[i]['rsi'], 2) if not pd.isna(df.iloc[i]['rsi']) else 0,
                'roc': round(df.iloc[i]['roc'], 2) if not pd.isna(df.iloc[i]['roc']) else 0
            })

        # Calculate metrics
        portfolio_df = pd.DataFrame(portfolio_value)
        metrics = self.calculate_metrics(portfolio_df, trades)

        # Add scalping-specific metrics
        scalping_metrics = self.calculate_scalping_metrics(trades)
        metrics.update(scalping_metrics)

        return {
            'symbol': symbol,
            'data': df,
            'portfolio': portfolio_df,
            'trades': trades,
            'metrics': metrics,
            'data_summary': self.get_data_summary(df)
        }

    def calculate_scalping_metrics(self, trades):
        """Calculate scalping-specific performance metrics"""
        if not trades:
            return {
                'total_scalping_trades': 0,
                'avg_trade_duration_minutes': 0,
                'trailing_stop_exits': 0,
                'square_off_3_20_closures': 0,
                'scalping_win_rate': 0,
                'avg_scalping_return': 0,
                'best_scalp_return': 0,
                'worst_scalp_return': 0,
                'avg_rsi_entry': 0,
                'avg_roc_entry': 0
            }

        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']

        # Calculate trade metrics
        trade_durations = []
        trailing_stop_count = 0
        square_off_count = 0
        scalping_returns = []
        entry_rsi_values = []
        entry_roc_values = []

        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_trade = buy_trades[i]
            sell_trade = sell_trades[i]

            # Duration
            buy_time = pd.to_datetime(buy_trade['date'])
            sell_time = pd.to_datetime(sell_trade['date'])
            duration_minutes = (sell_time - buy_time).total_seconds() / 60
            trade_durations.append(duration_minutes)

            # Exit reasons
            exit_reason = sell_trade.get('reason', 'UNKNOWN')
            if exit_reason == 'TRAILING_STOP':
                trailing_stop_count += 1
            elif exit_reason == 'SQUARE_OFF_3_20_PM':
                square_off_count += 1

            # Returns
            trade_return = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
            scalping_returns.append(trade_return)

            # Entry conditions
            entry_rsi_values.append(buy_trade.get('rsi', 0))
            entry_roc_values.append(buy_trade.get('roc', 0))

        # Calculate metrics
        avg_duration = np.mean(trade_durations) if trade_durations else 0
        scalping_win_rate = sum(1 for r in scalping_returns if r > 0) / len(scalping_returns) if scalping_returns else 0
        avg_scalping_return = np.mean(scalping_returns) if scalping_returns else 0
        best_scalp = max(scalping_returns) if scalping_returns else 0
        worst_scalp = min(scalping_returns) if scalping_returns else 0
        avg_entry_rsi = np.mean(entry_rsi_values) if entry_rsi_values else 0
        avg_entry_roc = np.mean(entry_roc_values) if entry_roc_values else 0

        return {
            'total_scalping_trades': len(buy_trades),
            'avg_trade_duration_minutes': round(avg_duration, 2),
            'trailing_stop_exits': trailing_stop_count,
            'square_off_3_20_closures': square_off_count,
            'scalping_win_rate': round(scalping_win_rate, 4),
            'avg_scalping_return': round(avg_scalping_return, 4),
            'best_scalp_return': round(best_scalp, 4),
            'worst_scalp_return': round(worst_scalp, 4),
            'avg_rsi_entry': round(avg_entry_rsi, 2),
            'avg_roc_entry': round(avg_entry_roc, 2)
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
            'avg_records_per_day': round(len(df) / unique_days, 2) if unique_days > 0 else 0,
            'avg_rsi': round(df['rsi'].mean(), 2),
            'avg_roc': round(df['roc'].mean(), 2)
        }

    def calculate_metrics(self, portfolio_df, trades):
        """Calculate performance metrics"""
        total_return = (portfolio_df.iloc[-1]['value'] - self.initial_capital) / self.initial_capital

        # Trade-based metrics
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

        # Portfolio-based metrics
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
        print(f"\nStarting RSI & ROC scalping backtest for {len(self.symbols)} symbols")
        print(f"Strategy: RSI > {self.rsi_threshold} AND ROC > ROC_avg, Exit: {self.trailing_stop_pct * 100}% trailing stop")

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
        """Create a summary report of all backtests"""
        if not self.results:
            print("No results to summarize.")
            return None

        summary_data = []
        for symbol, result in self.results.items():
            metrics = result['metrics']
            data_summary = result['data_summary']

            summary_data.append({
                'Symbol': symbol.split(':')[1].replace('-EQ', '') if ':' in symbol else symbol.replace('-EQ', ''),
                'Total Return (%)': metrics['total_return'] * 100,
                'Final Value': metrics['final_value'],
                'Total Trades': metrics['total_trades'],
                'Win Rate (%)': metrics['win_rate'] * 100,
                'Avg Trade Return (%)': metrics['avg_trade_return'] * 100,
                'Best Trade (%)': metrics['max_trade_return'] * 100,
                'Worst Trade (%)': metrics['min_trade_return'] * 100,
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Max Drawdown (%)': metrics['max_drawdown'] * 100,
                'Scalping Trades': metrics.get('total_scalping_trades', 0),
                'Avg Duration (min)': metrics.get('avg_trade_duration_minutes', 0),
                'Trailing Stop Exits': metrics.get('trailing_stop_exits', 0),
                '3:20 PM Square-offs': metrics.get('square_off_3_20_closures', 0),
                'Scalping Win Rate (%)': metrics.get('scalping_win_rate', 0) * 100,
                'Best Scalp (%)': metrics.get('best_scalp_return', 0) * 100,
                'Worst Scalp (%)': metrics.get('worst_scalp_return', 0) * 100,
                'Avg Entry RSI': metrics.get('avg_rsi_entry', 0),
                'Avg Entry ROC': metrics.get('avg_roc_entry', 0),
                'Data Points': data_summary['total_records'],
                'Trading Days': data_summary['trading_days'],
                'Avg RSI': data_summary.get('avg_rsi', 0),
                'Avg ROC': data_summary.get('avg_roc', 0)
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Total Return (%)', ascending=False)

        return summary_df

    def export_results(self, filename_prefix="rsi_roc_scalping"):
        """Export all results to CSV files"""
        if not self.results:
            print("No results to export.")
            return

        # Export summary
        summary_df = self.create_summary_report()
        summary_filename = f"{filename_prefix}_summary.csv"
        summary_df.to_csv(summary_filename, index=False)
        print(f"Summary exported to {summary_filename}")

        # Export individual trades
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

        # Export detailed trade analysis
        self.export_trade_analysis(filename_prefix)

    def export_trade_analysis(self, filename_prefix):
        """Export detailed trade analysis"""
        trade_analysis = []

        for symbol, result in self.results.items():
            trades = result['trades']
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']

            for i in range(min(len(buy_trades), len(sell_trades))):
                buy_trade = buy_trades[i]
                sell_trade = sell_trades[i]

                buy_time = pd.to_datetime(buy_trade['date'])
                sell_time = pd.to_datetime(sell_trade['date'])
                duration_minutes = (sell_time - buy_time).total_seconds() / 60

                trade_return = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
                trade_pnl = sell_trade.get('trade_pnl', 0)

                trade_analysis.append({
                    'Symbol': symbol,
                    'Entry_Date': buy_trade['date'],
                    'Exit_Date': sell_trade['date'],
                    'Entry_Price': buy_trade['price'],
                    'Exit_Price': sell_trade['price'],
                    'Shares': buy_trade['shares'],
                    'Duration_Minutes': round(duration_minutes, 2),
                    'Trade_Return_Pct': round(trade_return * 100, 2),
                    'Trade_PnL': round(trade_pnl, 2),
                    'Exit_Reason': sell_trade.get('reason', 'UNKNOWN'),
                    'Entry_RSI': buy_trade.get('rsi', 0),
                    'Entry_ROC': buy_trade.get('roc', 0),
                    'Entry_ROC_Avg': buy_trade.get('roc_avg', 0),
                    'Exit_RSI': sell_trade.get('rsi', 0),
                    'Exit_ROC': sell_trade.get('roc', 0),
                    'Same_Day': buy_time.date() == sell_time.date(),
                    'Source_DB': buy_trade.get('source_db', 'Unknown')
                })

        if trade_analysis:
            trade_df = pd.DataFrame(trade_analysis)
            trade_filename = f"{filename_prefix}_detailed_trades.csv"
            trade_df.to_csv(trade_filename, index=False)
            print(f"Detailed trade analysis exported to {trade_filename}")

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

    def create_parameter_optimization_report(self):
        """Create recommendations for parameter optimization based on results"""
        if not self.results:
            return

        summary_df = self.create_summary_report()

        print("\n" + "=" * 120)
        print("PARAMETER OPTIMIZATION INSIGHTS:")
        print("=" * 120)

        # Analyze current parameters effectiveness
        avg_win_rate = summary_df['Scalping Win Rate (%)'].mean()
        avg_return = summary_df['Total Return (%)'].mean()
        avg_duration = summary_df['Avg Duration (min)'].mean()
        avg_entry_rsi = summary_df['Avg Entry RSI'].mean()

        print(f"Current Strategy Performance Summary:")
        print(f"  RSI Threshold: {self.rsi_threshold}")
        print(f"  ROC Period: {self.roc_period}")
        print(f"  ROC Lookback: {self.roc_lookback_period}")
        print(f"  Trailing Stop: {self.trailing_stop_pct * 100}%")
        print(f"  Average Win Rate: {avg_win_rate:.1f}%")
        print(f"  Average Return: {avg_return:.2f}%")
        print(f"  Average Duration: {avg_duration:.1f} minutes")
        print(f"  Average Entry RSI: {avg_entry_rsi:.1f}")

        print(f"\nOptimization Recommendations:")

        # RSI threshold recommendations
        if avg_entry_rsi > 85:
            print(f"🔧 RSI Threshold: Consider lowering to 75-80 (current avg entry: {avg_entry_rsi:.1f})")
        elif avg_entry_rsi < 72:
            print(f"🔧 RSI Threshold: Consider raising to 75-80 (current avg entry: {avg_entry_rsi:.1f})")
        else:
            print(f"✅ RSI Threshold: Current level appears optimal")

        # Trailing stop recommendations
        if avg_duration < 15:
            print(f"🔧 Trailing Stop: Consider loosening to 0.4-0.5% (trades too quick: {avg_duration:.1f} min avg)")
        elif avg_duration > 45:
            print(f"🔧 Trailing Stop: Consider tightening to 0.2-0.25% (trades too long: {avg_duration:.1f} min avg)")
        else:
            print(f"✅ Trailing Stop: Current 0.3% appears optimal")

        # Win rate recommendations
        if avg_win_rate < 35:
            print(f"🔧 Entry Criteria: Consider stricter conditions (low win rate: {avg_win_rate:.1f}%)")
            print(f"   - Add volume filter")
            print(f"   - Increase ROC lookback period")
            print(f"   - Add ADX filter for trending markets")
        elif avg_win_rate > 65:
            print(f"🔧 Entry Criteria: Consider more aggressive entries (high win rate: {avg_win_rate:.1f}%)")
            print(f"   - Lower RSI threshold slightly")
            print(f"   - Reduce ROC lookback period")
        else:
            print(f"✅ Win Rate: Current level is healthy")

        # Return recommendations
        if avg_return < 0:
            print(f"⚠️  Negative Average Return: Strategy needs significant improvement")
            print(f"   - Consider reversing strategy (short on RSI > 70)")
            print(f"   - Add market regime filter")
            print(f"   - Test different time frames")
        elif avg_return < 5:
            print(f"🔧 Low Average Return: Consider parameter adjustments")
            print(f"   - Test different RSI periods (21, 9)")
            print(f"   - Adjust ROC calculation period")
        else:
            print(f"✅ Good Average Return: Strategy performing well")

        # Additional recommendations based on analysis
        profitable_symbols = len([r for r in self.results.values() if r['metrics']['total_return'] > 0])
        profitability_rate = profitable_symbols / len(self.results)

        if profitability_rate < 0.4:
            print(f"🔧 Symbol Selection: Low profitability rate ({profitability_rate:.1%})")
            print(f"   - Add symbol screening criteria")
            print(f"   - Filter by average volume")
            print(f"   - Consider sector-specific parameters")
        elif profitability_rate > 0.7:
            print(f"✅ Excellent symbol profitability rate: {profitability_rate:.1%}")

        # Risk management recommendations
        avg_max_dd = summary_df['Max Drawdown (%)'].mean()
        if avg_max_dd > 15:
            print(f"⚠️  High Average Drawdown: {avg_max_dd:.1f}%")
            print(f"   - Consider position sizing rules")
            print(f"   - Add daily loss limits")
            print(f"   - Implement correlation filters")
        else:
            print(f"✅ Drawdown levels acceptable: {avg_max_dd:.1f}%")

    def run_parameter_sensitivity_analysis(self):
        """Provide insights on parameter sensitivity (conceptual analysis)"""
        print("\n" + "=" * 120)
        print("PARAMETER SENSITIVITY ANALYSIS:")
        print("=" * 120)

        print("Based on the current results, here's how different parameters might affect performance:")
        print()

        print("RSI Threshold Sensitivity:")
        print("  • Lower (65-70): More trades, potentially lower win rate, faster entries")
        print("  • Higher (75-85): Fewer trades, potentially higher win rate, later entries")
        print("  • Current (70): Balanced approach for momentum scalping")
        print()

        print("Trailing Stop Sensitivity:")
        print("  • Tighter (0.1-0.2%): Faster exits, lower per-trade returns, higher win rate")
        print("  • Looser (0.4-0.6%): Longer holds, higher per-trade returns, lower win rate")
        print("  • Current (0.3%): Moderate risk/reward balance")
        print()

        print("ROC Period Sensitivity:")
        print("  • Shorter (5-8): More responsive to price changes, more signals")
        print("  • Longer (12-20): Smoother signals, fewer false positives")
        print("  • Current (10): Good balance for intraday momentum")
        print()

        print("ROC Lookback Sensitivity:")
        print("  • Shorter (20-35): More recent average, more adaptive")
        print("  • Longer (60-100): More stable average, fewer signals")
        print("  • Current (50): Reasonable trend context")

        print("\nRecommended Testing Scenarios:")
        print("1. Conservative: RSI > 75, Trailing Stop 0.2%, ROC Period 12")
        print("2. Aggressive: RSI > 65, Trailing Stop 0.4%, ROC Period 8")
        print("3. Stable: RSI > 72, Trailing Stop 0.25%, ROC Lookback 75")
        print("4. Quick Scalp: RSI > 70, Trailing Stop 0.15%, early cutoff time")


# Enhanced main execution with comprehensive analysis


# Example usage and main execution
if __name__ == "__main__":
    # Initialize RSI & ROC Scalping Backtester
    backtester = RSIROCScalpingBacktester(
        data_folder="data/symbolupdate",  # Folder containing database files
        symbols=None,  # Auto-detect symbols from databases
        rsi_period=14,  # RSI calculation period
        roc_period=10,  # ROC calculation period
        rsi_threshold=70,  # RSI threshold for entry (RSI > 70)
        roc_lookback_period=50,  # Period to calculate ROC average
        trailing_stop_pct=0.3,  # 0.3% trailing stop loss
        initial_capital=100000,  # Starting capital per symbol
        square_off_time="15:20",  # 3:20 PM IST square-off time
        min_data_points=100  # Minimum data points required per symbol
    )

    print("=" * 120)
    print("RSI & ROC SCALPING STRATEGY BACKTESTER")
    print("=" * 120)
    print("Strategy Rules:")
    print("1. Entry Signal: RSI > 70 AND ROC > ROC_rolling_average")
    print("2. Exit Signal: 0.3% trailing stop loss")
    print("3. Mandatory square-off at 3:20 PM IST (no overnight positions)")
    print("4. Intraday scalping only - multiple trades allowed per day")
    print("5. Auto-detection of symbols with sufficient data")
    print("6. Data sourced from multiple database files in 'data' folder")
    print("=" * 120)

    try:
        # Check prerequisites
        if not backtester.db_files:
            print("ERROR: No database files found!")
            print("Setup Instructions:")
            print("1. Create a 'data' folder in the script directory")
            print("2. Place your .db files in the 'data' folder")
            print("3. Expected schema: market_data table with OHLCV data")
            exit(1)

        if not backtester.symbols:
            print("ERROR: No symbols detected with sufficient data!")
            print("Troubleshooting:")
            print("1. Check database files contain market_data table")
            print("2. Reduce min_data_points if needed")
            print("3. Verify database schema matches expected format")
            exit(1)

        # Run the backtest
        print(f"\nStarting RSI & ROC scalping backtest...")
        print(f"Entry Condition: RSI > {backtester.rsi_threshold} AND ROC > ROC_avg({backtester.roc_lookback_period})")
        print(f"Exit Condition: {backtester.trailing_stop_pct * 100}% trailing stop loss")
        print(f"Square-off Time: {backtester.square_off_time} IST")

        results = backtester.run_backtest_sequential()

        if results:
            print(f"\nBacktest completed for {len(results)} symbols")

            # Create and display summary
            summary_df = backtester.create_summary_report()
            print("\n" + "=" * 150)
            print("RSI & ROC SCALPING STRATEGY BACKTEST SUMMARY")
            print("=" * 150)
            print(summary_df.round(2).to_string(index=False))

            # Export results
            print("\nExporting results to CSV files...")
            backtester.export_results("rsi_roc_scalping_strategy")

            # Performance analysis
            print("\n" + "=" * 120)
            print("TOP PERFORMERS BY TOTAL RETURN:")
            print("=" * 120)
            top_performers = summary_df.head(5)[['Symbol', 'Total Return (%)', 'Scalping Win Rate (%)',
                                                 'Avg Duration (min)', 'Trailing Stop Exits', '3:20 PM Square-offs']]
            print(top_performers.to_string(index=False))

            # Strategy effectiveness analysis
            print("\n" + "=" * 120)
            print("RSI & ROC SCALPING STRATEGY EFFECTIVENESS ANALYSIS:")
            print("=" * 120)

            total_symbols = len(results)
            profitable_symbols = len([r for r in results.values() if r['metrics']['total_return'] > 0])
            avg_return = summary_df['Total Return (%)'].mean()
            avg_win_rate = summary_df['Scalping Win Rate (%)'].mean()
            avg_trades = summary_df['Scalping Trades'].mean()
            avg_duration = summary_df['Avg Duration (min)'].mean()
            total_trailing_stops = summary_df['Trailing Stop Exits'].sum()
            total_square_offs = summary_df['3:20 PM Square-offs'].sum()
            avg_entry_rsi = summary_df['Avg Entry RSI'].mean()
            avg_entry_roc = summary_df['Avg Entry ROC'].mean()

            print(f"Symbols Backtested: {total_symbols}")
            print(f"Profitable Symbols: {profitable_symbols} ({profitable_symbols / total_symbols * 100:.1f}%)")
            print(f"Average Return: {avg_return:.2f}%")
            print(f"Average Scalping Win Rate: {avg_win_rate:.1f}%")
            print(f"Average Scalping Trades per Symbol: {avg_trades:.1f}")
            print(f"Average Trade Duration: {avg_duration:.1f} minutes")
            print(f"Total Trailing Stop Exits: {total_trailing_stops}")
            print(f"Total 3:20 PM Square-offs: {total_square_offs}")
            print(f"Average Entry RSI: {avg_entry_rsi:.1f}")
            print(f"Average Entry ROC: {avg_entry_roc:.2f}%")

            # Exit analysis
            total_exits = total_trailing_stops + total_square_offs
            if total_exits > 0:
                print(f"\nExit Method Breakdown:")
                print(f"  Trailing Stop (0.3%): {total_trailing_stops} ({total_trailing_stops / total_exits * 100:.1f}%)")
                print(f"  3:20 PM Square-off: {total_square_offs} ({total_square_offs / total_exits * 100:.1f}%)")

            # Best and worst performers
            print("\n" + "=" * 120)
            print("PERFORMANCE INSIGHTS:")
            print("=" * 120)

            best_performer = summary_df.iloc[0]
            worst_performer = summary_df.iloc[-1]

            print(f"Best Performer: {best_performer['Symbol']} (+{best_performer['Total Return (%)']:.2f}%)")
            print(f"  - Scalping Win Rate: {best_performer['Scalping Win Rate (%)']:.1f}%")
            print(f"  - Average Trade Duration: {best_performer['Avg Duration (min)']:.1f} minutes")
            print(f"  - Average Entry RSI: {best_performer['Avg Entry RSI']:.1f}")

            print(f"\nWorst Performer: {worst_performer['Symbol']} ({worst_performer['Total Return (%)']:.2f}%)")
            print(f"  - Scalping Win Rate: {worst_performer['Scalping Win Rate (%)']:.1f}%")
            print(f"  - Average Trade Duration: {worst_performer['Avg Duration (min)']:.1f} minutes")
            print(f"  - Average Entry RSI: {worst_performer['Avg Entry RSI']:.1f}")

            # Strategy parameter analysis
            print("\n" + "=" * 120)
            print("STRATEGY PARAMETER EFFECTIVENESS:")
            print("=" * 120)

            high_rsi_entries = summary_df[summary_df['Avg Entry RSI'] > 75]
            moderate_rsi_entries = summary_df[(summary_df['Avg Entry RSI'] >= 70) & (summary_df['Avg Entry RSI'] <= 75)]

            if not high_rsi_entries.empty:
                print(f"High RSI Entries (>75): {len(high_rsi_entries)} symbols, Avg Return: {high_rsi_entries['Total Return (%)'].mean():.2f}%")
            if not moderate_rsi_entries.empty:
                print(f"Moderate RSI Entries (70-75): {len(moderate_rsi_entries)} symbols, Avg Return: {moderate_rsi_entries['Total Return (%)'].mean():.2f}%")

            # Quick trades vs longer holds
            quick_trades = summary_df[summary_df['Avg Duration (min)'] < 30]
            longer_trades = summary_df[summary_df['Avg Duration (min)'] >= 30]

            if not quick_trades.empty:
                print(f"Quick Scalps (<30 min): {len(quick_trades)} symbols, Avg Return: {quick_trades['Total Return (%)'].mean():.2f}%")
            if not longer_trades.empty:
                print(f"Longer Holds (≥30 min): {len(longer_trades)} symbols, Avg Return: {longer_trades['Total Return (%)'].mean():.2f}%")

            print("\n" + "=" * 120)
            print("FILES EXPORTED:")
            print("=" * 120)
            print("1. rsi_roc_scalping_strategy_summary.csv - Performance summary")
            print("2. rsi_roc_scalping_strategy_all_trades.csv - All trade records")
            print("3. rsi_roc_scalping_strategy_detailed_trades.csv - Detailed trade analysis")
            print("=" * 120)

            # Additional analysis for scalping effectiveness
            print("\n" + "=" * 120)
            print("SCALPING STRATEGY DETAILED ANALYSIS:")
            print("=" * 120)

            # Analyze by trade duration
            very_quick = summary_df[summary_df['Avg Duration (min)'] < 15]
            quick = summary_df[(summary_df['Avg Duration (min)'] >= 15) & (summary_df['Avg Duration (min)'] < 30)]
            medium = summary_df[(summary_df['Avg Duration (min)'] >= 30) & (summary_df['Avg Duration (min)'] < 60)]
            long_scalps = summary_df[summary_df['Avg Duration (min)'] >= 60]

            print("Performance by Trade Duration:")
            if not very_quick.empty:
                print(
                    f"  Very Quick (<15 min): {len(very_quick)} symbols, Avg Return: {very_quick['Total Return (%)'].mean():.2f}%, Avg Win Rate: {very_quick['Scalping Win Rate (%)'].mean():.1f}%")
            if not quick.empty:
                print(f"  Quick (15-30 min): {len(quick)} symbols, Avg Return: {quick['Total Return (%)'].mean():.2f}%, Avg Win Rate: {quick['Scalping Win Rate (%)'].mean():.1f}%")
            if not medium.empty:
                print(
                    f"  Medium (30-60 min): {len(medium)} symbols, Avg Return: {medium['Total Return (%)'].mean():.2f}%, Avg Win Rate: {medium['Scalping Win Rate (%)'].mean():.1f}%")
            if not long_scalps.empty:
                print(
                    f"  Long (>60 min): {len(long_scalps)} symbols, Avg Return: {long_scalps['Total Return (%)'].mean():.2f}%, Avg Win Rate: {long_scalps['Scalping Win Rate (%)'].mean():.1f}%")

            # Analyze exit methods effectiveness
            print("\nExit Method Effectiveness:")
            trailing_heavy = summary_df[summary_df['Trailing Stop Exits'] > summary_df['3:20 PM Square-offs']]
            square_off_heavy = summary_df[summary_df['3:20 PM Square-offs'] > summary_df['Trailing Stop Exits']]
            balanced = summary_df[summary_df['Trailing Stop Exits'] == summary_df['3:20 PM Square-offs']]

            if not trailing_heavy.empty:
                print(f"  Trailing Stop Dominant: {len(trailing_heavy)} symbols, Avg Return: {trailing_heavy['Total Return (%)'].mean():.2f}%")
            if not square_off_heavy.empty:
                print(f"  Square-off Dominant: {len(square_off_heavy)} symbols, Avg Return: {square_off_heavy['Total Return (%)'].mean():.2f}%")
            if not balanced.empty:
                print(f"  Balanced Exits: {len(balanced)} symbols, Avg Return: {balanced['Total Return (%)'].mean():.2f}%")

            # RSI entry level analysis
            print("\nRSI Entry Level Analysis:")
            high_rsi = summary_df[summary_df['Avg Entry RSI'] > 75]
            very_high_rsi = summary_df[summary_df['Avg Entry RSI'] > 80]
            moderate_rsi = summary_df[(summary_df['Avg Entry RSI'] >= 70) & (summary_df['Avg Entry RSI'] <= 75)]

            if not moderate_rsi.empty:
                print(f"  Moderate RSI (70-75): {len(moderate_rsi)} symbols, Avg Return: {moderate_rsi['Total Return (%)'].mean():.2f}%")
            if not high_rsi.empty:
                print(f"  High RSI (75-80): {len(high_rsi)} symbols, Avg Return: {high_rsi['Total Return (%)'].mean():.2f}%")
            if not very_high_rsi.empty:
                print(f"  Very High RSI (>80): {len(very_high_rsi)} symbols, Avg Return: {very_high_rsi['Total Return (%)'].mean():.2f}%")

            # Trade frequency analysis
            print("\nTrade Frequency Analysis:")
            low_freq = summary_df[summary_df['Scalping Trades'] < 10]
            medium_freq = summary_df[(summary_df['Scalping Trades'] >= 10) & (summary_df['Scalping Trades'] < 25)]
            high_freq = summary_df[summary_df['Scalping Trades'] >= 25]

            if not low_freq.empty:
                print(f"  Low Frequency (<10 trades): {len(low_freq)} symbols, Avg Return: {low_freq['Total Return (%)'].mean():.2f}%")
            if not medium_freq.empty:
                print(f"  Medium Frequency (10-25 trades): {len(medium_freq)} symbols, Avg Return: {medium_freq['Total Return (%)'].mean():.2f}%")
            if not high_freq.empty:
                print(f"  High Frequency (≥25 trades): {len(high_freq)} symbols, Avg Return: {high_freq['Total Return (%)'].mean():.2f}%")

            # Risk-adjusted performance
            print("\nRisk-Adjusted Performance (Sharpe Ratio Analysis):")
            good_sharpe = summary_df[summary_df['Sharpe Ratio'] > 1.0]
            decent_sharpe = summary_df[(summary_df['Sharpe Ratio'] > 0.5) & (summary_df['Sharpe Ratio'] <= 1.0)]
            poor_sharpe = summary_df[summary_df['Sharpe Ratio'] <= 0.5]

            if not good_sharpe.empty:
                print(f"  Excellent Risk-Adj Return (Sharpe > 1.0): {len(good_sharpe)} symbols")
                print(f"    Top symbol: {good_sharpe.iloc[0]['Symbol']} (Sharpe: {good_sharpe.iloc[0]['Sharpe Ratio']:.2f})")
            if not decent_sharpe.empty:
                print(f"  Good Risk-Adj Return (Sharpe 0.5-1.0): {len(decent_sharpe)} symbols")
            if not poor_sharpe.empty:
                print(f"  Poor Risk-Adj Return (Sharpe ≤ 0.5): {len(poor_sharpe)} symbols")

            # Maximum drawdown analysis
            print("\nDrawdown Analysis:")
            low_dd = summary_df[summary_df['Max Drawdown (%)'] < 5]
            medium_dd = summary_df[(summary_df['Max Drawdown (%)'] >= 5) & (summary_df['Max Drawdown (%)'] < 10)]
            high_dd = summary_df[summary_df['Max Drawdown (%)'] >= 10]

            if not low_dd.empty:
                print(f"  Low Drawdown (<5%): {len(low_dd)} symbols, Avg Return: {low_dd['Total Return (%)'].mean():.2f}%")
            if not medium_dd.empty:
                print(f"  Medium Drawdown (5-10%): {len(medium_dd)} symbols, Avg Return: {medium_dd['Total Return (%)'].mean():.2f}%")
            if not high_dd.empty:
                print(f"  High Drawdown (≥10%): {len(high_dd)} symbols, Avg Return: {high_dd['Total Return (%)'].mean():.2f}%")

            print("=" * 120)

            # Strategy recommendations
            print("\n" + "=" * 120)
            print("STRATEGY OPTIMIZATION RECOMMENDATIONS:")
            print("=" * 120)

            if avg_win_rate < 40:
                print("⚠️  Low win rate detected. Consider:")
                print("   - Increasing RSI threshold (try 75-80)")
                print("   - Adding volume filter for better entry quality")
                print("   - Adjusting ROC lookback period")

            if avg_duration > 60:
                print("⚠️  Long average trade duration. Consider:")
                print("   - Tightening trailing stop (try 0.2%)")
                print("   - Adding profit target alongside trailing stop")

            if total_square_offs > total_trailing_stops:
                print("📊 Many trades closed at 3:20 PM. Consider:")
                print("   - Earlier entry cutoff (e.g., 2:30 PM)")
                print("   - Faster exit conditions for late entries")

            profitable_rate = profitable_symbols / total_symbols
            if profitable_rate > 0.6:
                print("✅ Strong strategy performance across symbols!")
                print("   - Consider increasing position sizing")
                print("   - Test with higher capital allocation")
            elif profitable_rate < 0.4:
                print("⚠️  Low profitability rate. Consider:")
                print("   - Stricter symbol selection criteria")
                print("   - Additional confluence factors")
                print("   - Different time frames for signals")

        else:
            print("No successful backtests completed.")
            print("Troubleshooting:")
            print("1. Check database files exist and are accessible")
            print("2. Verify sufficient data for RSI and ROC calculations")
            print("3. Ensure database schema matches expected format")
            print("4. Check symbol detection and filtering criteria")

    except Exception as e:
        print(f"Error running RSI & ROC scalping backtest: {e}")
        import traceback

        traceback.print_exc()