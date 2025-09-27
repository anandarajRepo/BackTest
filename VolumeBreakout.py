import sqlite3
import pandas as pd
import numpy as np
import json
import glob
import os
from datetime import datetime, time, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')


class First15SecondsVolumeBreakoutBacktester:
    def __init__(self, data_folder="data", symbols=None, volume_lookback=30,
                 trailing_stop_pct=2.0, initial_capital=100000, square_off_time="15:20",
                 min_data_points=100, entry_window_seconds=15):
        """
        Initialize the First 15 Seconds Volume Breakout Strategy Backtester

        Parameters:
        - data_folder: Folder containing database files (default: "data")
        - symbols: List of trading symbols to backtest (if None, auto-detect from databases)
        - volume_lookback: Number of 1-minute candles to calculate volume average (default 30)
        - trailing_stop_pct: Trailing stop loss percentage (default 2%)
        - initial_capital: Starting capital for backtesting (per symbol)
        - square_off_time: Time to square off all positions in HH:MM format (default "15:20")
        - min_data_points: Minimum data points required for a symbol to be included
        - entry_window_seconds: Entry window in seconds from candle start (default 15)
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()
        self.volume_lookback = volume_lookback
        self.trailing_stop_pct = trailing_stop_pct / 100
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.entry_window_seconds = entry_window_seconds
        self.results = {}
        self.combined_data = {}

        # Set up IST timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print(f"First 15 Seconds Volume Breakout Strategy Parameters:")
        print(f"  Data folder: {self.data_folder}")
        print(f"  Volume lookback: {self.volume_lookback} candles")
        print(f"  Entry window: {self.entry_window_seconds} seconds from candle start")
        print(f"  Trailing stop: {trailing_stop_pct}%")
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

    def create_1min_candles(self, df):
        """Convert tick data to 1-minute candles"""
        print("  Converting tick data to 1-minute candles...")

        # Resample to 1-minute candles
        ohlc_dict = {
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume_final': 'sum'
        }

        # Create 1-minute OHLC data
        candle_df = df.resample('1T').agg(ohlc_dict)

        # Add open price (first close of the period)
        candle_df['open'] = df['close'].resample('1T').first()

        # Drop rows with no data
        candle_df = candle_df.dropna()

        # Reorder columns
        candle_df = candle_df[['open', 'high', 'low', 'close', 'volume_final']]

        print(f"    Created {len(candle_df)} 1-minute candles from {len(df)} tick records")
        return candle_df

    def calculate_volume_average(self, df):
        """Calculate rolling average volume for the last N candles"""
        df['volume_avg'] = df['volume_final'].rolling(window=self.volume_lookback, min_periods=1).mean()
        return df

    def is_within_entry_window(self, timestamp):
        """Check if timestamp is within the first 15 seconds of a minute"""
        return timestamp.second < self.entry_window_seconds

    def generate_signals(self, df, tick_data):
        """Generate buy signals based on volume breakout within first 15 seconds"""
        df['buy_signal'] = False

        # For each 1-minute candle, check if volume > average
        for candle_time in df.index:
            if pd.isna(df.loc[candle_time, 'volume_avg']):
                continue

            volume_condition = df.loc[candle_time, 'volume_final'] > df.loc[candle_time, 'volume_avg']

            if volume_condition:
                # Check if we have tick data within the first 15 seconds of this candle
                candle_start = candle_time
                candle_end = candle_start + timedelta(seconds=self.entry_window_seconds)

                # Find tick data within the first 15 seconds
                tick_window = tick_data[(tick_data.index >= candle_start) &
                                        (tick_data.index < candle_end)]

                if not tick_window.empty:
                    # Volume breakout confirmed within first 15 seconds
                    df.loc[candle_time, 'buy_signal'] = True

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
        """Backtest First 15 Seconds Volume Breakout strategy for a single symbol"""
        print(f"\nStarting First 15 Seconds Volume Breakout backtest for {symbol}")

        # Load tick data
        tick_data = self.load_data_from_all_databases(symbol)

        if tick_data is None or len(tick_data) < self.volume_lookback * 10:
            print(f"Insufficient data for {symbol}. Skipping.")
            return None

        # Store combined data
        self.combined_data[symbol] = tick_data.copy()

        # Create 1-minute candles
        candle_data = self.create_1min_candles(tick_data)

        if len(candle_data) < self.volume_lookback:
            print(f"Insufficient candle data for {symbol}. Skipping.")
            return None

        # Calculate volume average
        candle_data = self.calculate_volume_average(candle_data)

        # Generate signals using both candle and tick data
        candle_data = self.generate_signals(candle_data, tick_data)

        # Add trading day and square-off time information
        candle_data['trading_day'] = candle_data.index.date
        candle_data['is_square_off_time'] = candle_data.index.map(self.is_square_off_time)

        # Initialize trading variables
        cash = self.initial_capital
        position = 0  # 0: no position, 1: long position
        entry_price = 0
        trailing_stop_price = 0
        portfolio_value = []
        trades = []
        entry_day = None
        entry_candle_time = None

        for i in range(len(candle_data)):
            current_candle = candle_data.iloc[i]
            current_price = current_candle['close']
            current_time = candle_data.index[i]
            current_day = current_candle['trading_day']
            is_square_off = current_candle['is_square_off_time']

            # Force square-off at 3:20 PM IST
            if position == 1 and is_square_off:
                shares = trades[-1]['shares']
                proceeds = shares * current_price
                cash += proceeds
                position = 0

                trades.append({
                    'date': current_time,
                    'action': 'SELL',
                    'price': round(current_price, 2),
                    'shares': shares,
                    'cash': round(cash, 2),
                    'reason': 'SQUARE_OFF_3_20_PM',
                    'volume': round(current_candle['volume_final'], 2),
                    'volume_avg': round(current_candle['volume_avg'], 2),
                    'volume_ratio': round(current_candle['volume_final'] / current_candle['volume_avg'], 2) if current_candle['volume_avg'] > 0 else 0,
                    'source_db': 'Combined',
                    'exit_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_price': round(current_price, 2),
                    'entry_price': round(trades[-1]['entry_price'], 2) if trades else 0,
                    'trade_return_pct': round(((current_price - trades[-1]['entry_price']) / trades[-1]['entry_price'] * 100), 2) if trades else 0,
                    'trade_pnl': round((shares * (current_price - trades[-1]['entry_price'])), 2) if trades else 0
                })

                entry_day = None
                entry_candle_time = None

            # Check for buy signal (only if not in position and not at/after square-off time)
            elif current_candle['buy_signal'] and position == 0 and not is_square_off:
                shares = int(cash / current_price)
                if shares > 0:
                    position = 1
                    entry_price = current_price
                    entry_day = current_day
                    entry_candle_time = current_time
                    trailing_stop_price = entry_price * (1 - self.trailing_stop_pct)
                    cost = shares * current_price
                    cash -= cost

                    trades.append({
                        'date': current_time,
                        'action': 'BUY',
                        'price': round(current_price, 2),
                        'shares': shares,
                        'cash': round(cash, 2),
                        'volume': round(current_candle['volume_final'], 2),
                        'volume_avg': round(current_candle['volume_avg'], 2),
                        'volume_ratio': round(current_candle['volume_final'] / current_candle['volume_avg'], 2) if current_candle['volume_avg'] > 0 else 0,
                        'source_db': 'Combined',
                        'entry_day': entry_day,
                        'entry_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'entry_price': round(current_price, 2),
                        'candle_open': round(current_candle['open'], 2),
                        'candle_high': round(current_candle['high'], 2),
                        'candle_low': round(current_candle['low'], 2),
                        'entry_within_15_sec': True  # This is guaranteed by our signal generation
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
                        'date': current_time,
                        'action': 'SELL',
                        'price': round(current_price, 2),
                        'shares': shares,
                        'cash': round(cash, 2),
                        'reason': 'TRAILING_STOP',
                        'volume': round(current_candle['volume_final'], 2),
                        'volume_avg': round(current_candle['volume_avg'], 2),
                        'volume_ratio': round(current_candle['volume_final'] / current_candle['volume_avg'], 2) if current_candle['volume_avg'] > 0 else 0,
                        'source_db': 'Combined',
                        'exit_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'exit_price': round(current_price, 2),
                        'entry_price': round(trades[-1]['entry_price'], 2) if trades else 0,
                        'trade_return_pct': round(((current_price - trades[-1]['entry_price']) / trades[-1]['entry_price'] * 100), 2) if trades else 0,
                        'trade_pnl': round((shares * (current_price - trades[-1]['entry_price'])), 2) if trades else 0
                    })

                    entry_day = None
                    entry_candle_time = None

            # Calculate portfolio value
            if position == 1:
                shares = trades[-1]['shares']
                portfolio_val = cash + (shares * current_price)
            else:
                portfolio_val = cash

            portfolio_value.append({
                'date': current_time,
                'value': round(portfolio_val, 2),
                'position': position,
                'trading_day': current_day,
                'is_square_off_time': is_square_off,
                'volume': round(current_candle['volume_final'], 2),
                'volume_avg': round(current_candle['volume_avg'], 2)
            })

        # Calculate metrics
        portfolio_df = pd.DataFrame(portfolio_value)
        metrics = self.calculate_metrics(portfolio_df, trades, candle_data)

        # Add volume breakout specific metrics
        volume_metrics = self.calculate_volume_breakout_metrics(trades)
        metrics.update(volume_metrics)

        return {
            'symbol': symbol,
            'candle_data': candle_data,
            'tick_data': tick_data,
            'portfolio': portfolio_df,
            'trades': trades,
            'metrics': metrics,
            'data_summary': self.get_data_summary(candle_data, tick_data)
        }

    def calculate_volume_breakout_metrics(self, trades):
        """Calculate volume breakout specific performance metrics"""
        if not trades:
            return {
                'total_volume_breakout_trades': 0,
                'avg_volume_ratio_entry': 0,
                'min_volume_ratio_entry': 0,
                'max_volume_ratio_entry': 0,
                'avg_trade_duration_minutes': 0,
                'trailing_stop_exits': 0,
                'square_off_3_20_closures': 0,
                'volume_breakout_win_rate': 0,
                'first_15_sec_entries': 0
            }

        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']

        # Calculate trade metrics
        trade_durations = []
        trailing_stop_count = 0
        square_off_count = 0
        volume_ratios = []
        first_15_sec_count = 0

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

            # Volume metrics
            volume_ratios.append(buy_trade.get('volume_ratio', 0))

            # Entry timing
            if buy_trade.get('entry_within_15_sec', False):
                first_15_sec_count += 1

        # Calculate win rate
        trade_returns = []
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_price = buy_trades[i]['price']
            sell_price = sell_trades[i]['price']
            trade_return = (sell_price - buy_price) / buy_price
            trade_returns.append(trade_return)

        volume_breakout_win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) if trade_returns else 0

        return {
            'total_volume_breakout_trades': len(buy_trades),
            'avg_volume_ratio_entry': round(np.mean(volume_ratios), 2) if volume_ratios else 0,
            'min_volume_ratio_entry': round(min(volume_ratios), 2) if volume_ratios else 0,
            'max_volume_ratio_entry': round(max(volume_ratios), 2) if volume_ratios else 0,
            'avg_trade_duration_minutes': round(np.mean(trade_durations), 2) if trade_durations else 0,
            'trailing_stop_exits': trailing_stop_count,
            'square_off_3_20_closures': square_off_count,
            'volume_breakout_win_rate': round(volume_breakout_win_rate, 4),
            'first_15_sec_entries': first_15_sec_count
        }

    def get_data_summary(self, candle_data, tick_data):
        """Get summary statistics for the data"""
        unique_days = len(candle_data['trading_day'].unique()) if 'trading_day' in candle_data.columns else len(candle_data.index.date)

        return {
            'total_candles': len(candle_data),
            'total_ticks': len(tick_data),
            'date_range': f"{candle_data.index.min()} to {candle_data.index.max()}",
            'trading_days': unique_days,
            'avg_daily_volume': round(candle_data['volume_final'].mean(), 2),
            'avg_volume_ratio': round((candle_data['volume_final'] / candle_data['volume_avg']).mean(), 2),
            'volume_breakout_candles': len(candle_data[candle_data['volume_final'] > candle_data['volume_avg']]),
            'avg_candles_per_day': round(len(candle_data) / unique_days, 2) if unique_days > 0 else 0
        }

    def calculate_metrics(self, portfolio_df, trades, candle_data):
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
        print(f"\nStarting First 15 Seconds Volume Breakout backtest for {len(self.symbols)} symbols")
        print(f"Strategy: Volume > {self.volume_lookback}-candle average within first {self.entry_window_seconds} seconds")
        print(f"Exit: {self.trailing_stop_pct * 100}% trailing stop or 3:20 PM square-off")

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
                'Volume Breakout Trades': metrics.get('total_volume_breakout_trades', 0),
                'Avg Duration (min)': metrics.get('avg_trade_duration_minutes', 0),
                'Avg Volume Ratio': metrics.get('avg_volume_ratio_entry', 0),
                'Max Volume Ratio': metrics.get('max_volume_ratio_entry', 0),
                'Trailing Stop Exits': metrics.get('trailing_stop_exits', 0),
                '3:20 PM Square-offs': metrics.get('square_off_3_20_closures', 0),
                'First 15-Sec Entries': metrics.get('first_15_sec_entries', 0),
                'Volume Breakout Win Rate (%)': metrics.get('volume_breakout_win_rate', 0) * 100,
                'Total Candles': data_summary['total_candles'],
                'Total Ticks': data_summary['total_ticks'],
                'Trading Days': data_summary['trading_days'],
                'Volume Breakout Candles': data_summary['volume_breakout_candles'],
                'Avg Daily Volume': data_summary['avg_daily_volume']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Total Return (%)', ascending=False)

        return summary_df

    def export_results(self, filename_prefix="first_15sec_volume_breakout"):
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

        # Export volume analysis
        self.export_volume_analysis(filename_prefix)

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
                    'Entry_Volume': buy_trade.get('volume', 0),
                    'Entry_Volume_Avg': buy_trade.get('volume_avg', 0),
                    'Volume_Ratio': buy_trade.get('volume_ratio', 0),
                    'Entry_Within_15_Sec': buy_trade.get('entry_within_15_sec', False),
                    'Candle_Open': buy_trade.get('candle_open', 0),
                    'Candle_High': buy_trade.get('candle_high', 0),
                    'Candle_Low': buy_trade.get('candle_low', 0),
                    'Same_Day': buy_time.date() == sell_time.date(),
                    'Source_DB': buy_trade.get('source_db', 'Combined')
                })

        if trade_analysis:
            trade_df = pd.DataFrame(trade_analysis)
            trade_filename = f"{filename_prefix}_detailed_trades.csv"
            trade_df.to_csv(trade_filename, index=False)
            print(f"Detailed trade analysis exported to {trade_filename}")

    def export_volume_analysis(self, filename_prefix):
        """Export volume breakout analysis"""
        volume_analysis = []

        for symbol, result in self.results.items():
            candle_data = result['candle_data']

            # Analyze volume breakouts
            volume_breakouts = candle_data[candle_data['volume_final'] > candle_data['volume_avg']]

            for idx, candle in volume_breakouts.iterrows():
                volume_analysis.append({
                    'Symbol': symbol,
                    'DateTime': idx,
                    'Volume': candle['volume_final'],
                    'Volume_Avg': candle['volume_avg'],
                    'Volume_Ratio': candle['volume_final'] / candle['volume_avg'] if candle['volume_avg'] > 0 else 0,
                    'Open': candle['open'],
                    'High': candle['high'],
                    'Low': candle['low'],
                    'Close': candle['close'],
                    'Had_Buy_Signal': candle.get('buy_signal', False),
                    'Price_Change_Pct': ((candle['close'] - candle['open']) / candle['open'] * 100) if candle['open'] > 0 else 0
                })

        if volume_analysis:
            volume_df = pd.DataFrame(volume_analysis)
            volume_filename = f"{filename_prefix}_volume_analysis.csv"
            volume_df.to_csv(volume_filename, index=False)
            print(f"Volume analysis exported to {volume_filename}")

    def create_performance_insights(self):
        """Create detailed performance insights"""
        if not self.results:
            return

        summary_df = self.create_summary_report()

        print("\n" + "=" * 120)
        print("FIRST 15 SECONDS VOLUME BREAKOUT STRATEGY INSIGHTS:")
        print("=" * 120)

        # Overall performance
        total_symbols = len(self.results)
        profitable_symbols = len([r for r in self.results.values() if r['metrics']['total_return'] > 0])
        avg_return = summary_df['Total Return (%)'].mean()
        avg_win_rate = summary_df['Volume Breakout Win Rate (%)'].mean()
        avg_volume_ratio = summary_df['Avg Volume Ratio'].mean()
        avg_duration = summary_df['Avg Duration (min)'].mean()

        print(f"Strategy Performance Summary:")
        print(f"  Symbols Tested: {total_symbols}")
        print(f"  Profitable Symbols: {profitable_symbols} ({profitable_symbols / total_symbols * 100:.1f}%)")
        print(f"  Average Return: {avg_return:.2f}%")
        print(f"  Average Win Rate: {avg_win_rate:.1f}%")
        print(f"  Average Volume Ratio at Entry: {avg_volume_ratio:.2f}x")
        print(f"  Average Trade Duration: {avg_duration:.1f} minutes")

        # Volume ratio analysis
        print(f"\nVolume Ratio Analysis:")
        high_volume_trades = summary_df[summary_df['Avg Volume Ratio'] > 2.0]
        medium_volume_trades = summary_df[(summary_df['Avg Volume Ratio'] >= 1.5) & (summary_df['Avg Volume Ratio'] <= 2.0)]
        low_volume_trades = summary_df[summary_df['Avg Volume Ratio'] < 1.5]

        if not high_volume_trades.empty:
            print(f"  High Volume Ratio (>2.0x): {len(high_volume_trades)} symbols, Avg Return: {high_volume_trades['Total Return (%)'].mean():.2f}%")
        if not medium_volume_trades.empty:
            print(f"  Medium Volume Ratio (1.5-2.0x): {len(medium_volume_trades)} symbols, Avg Return: {medium_volume_trades['Total Return (%)'].mean():.2f}%")
        if not low_volume_trades.empty:
            print(f"  Low Volume Ratio (<1.5x): {len(low_volume_trades)} symbols, Avg Return: {low_volume_trades['Total Return (%)'].mean():.2f}%")

        # Trade duration analysis
        print(f"\nTrade Duration Analysis:")
        quick_trades = summary_df[summary_df['Avg Duration (min)'] < 30]
        medium_trades = summary_df[(summary_df['Avg Duration (min)'] >= 30) & (summary_df['Avg Duration (min)'] < 60)]
        long_trades = summary_df[summary_df['Avg Duration (min)'] >= 60]

        if not quick_trades.empty:
            print(f"  Quick Trades (<30 min): {len(quick_trades)} symbols, Avg Return: {quick_trades['Total Return (%)'].mean():.2f}%")
        if not medium_trades.empty:
            print(f"  Medium Trades (30-60 min): {len(medium_trades)} symbols, Avg Return: {medium_trades['Total Return (%)'].mean():.2f}%")
        if not long_trades.empty:
            print(f"  Long Trades (≥60 min): {len(long_trades)} symbols, Avg Return: {long_trades['Total Return (%)'].mean():.2f}%")

        # Exit method analysis
        total_trailing_stops = summary_df['Trailing Stop Exits'].sum()
        total_square_offs = summary_df['3:20 PM Square-offs'].sum()
        total_exits = total_trailing_stops + total_square_offs

        print(f"\nExit Method Analysis:")
        if total_exits > 0:
            print(f"  Trailing Stop Exits: {total_trailing_stops} ({total_trailing_stops / total_exits * 100:.1f}%)")
            print(f"  3:20 PM Square-offs: {total_square_offs} ({total_square_offs / total_exits * 100:.1f}%)")

        # Best performers
        print(f"\nTop 3 Performers:")
        top_3 = summary_df.head(3)
        for idx, row in top_3.iterrows():
            print(f"  {row['Symbol']}: +{row['Total Return (%)']:.2f}% (Win Rate: {row['Volume Breakout Win Rate (%)']:.1f}%, Avg Vol Ratio: {row['Avg Volume Ratio']:.2f}x)")

        # Strategy effectiveness recommendations
        print(f"\nStrategy Optimization Recommendations:")

        if avg_volume_ratio < 1.5:
            print("🔧 Volume Threshold: Consider requiring higher volume ratios (>1.5x) for entry")
        elif avg_volume_ratio > 3.0:
            print("🔧 Volume Threshold: Current high volume requirement may be too restrictive")
        else:
            print("✅ Volume Threshold: Current level appears optimal")

        if avg_duration > 90:
            print("🔧 Exit Strategy: Consider tighter trailing stops (trades holding too long)")
        elif avg_duration < 15:
            print("🔧 Exit Strategy: Consider looser trailing stops (trades exiting too quickly)")
        else:
            print("✅ Exit Strategy: Current trailing stop level appears optimal")

        if avg_win_rate < 40:
            print("🔧 Entry Criteria: Consider additional filters to improve win rate")
            print("   - Add price momentum confirmation")
            print("   - Require volume spike during specific market hours")
            print("   - Add moving average filters")
        elif avg_win_rate > 70:
            print("🔧 Entry Criteria: Excellent win rate - consider more aggressive position sizing")
        else:
            print("✅ Win Rate: Healthy level for the strategy")

        if profitable_symbols / total_symbols < 0.5:
            print("⚠️  Symbol Selection: Low profitability rate suggests need for better symbol screening")
            print("   - Filter by average daily volume")
            print("   - Screen for stocks with consistent volume patterns")
            print("   - Consider sector-specific parameters")
        else:
            print("✅ Good symbol profitability rate")


# Example usage and main execution
if __name__ == "__main__":
    # Initialize First 15 Seconds Volume Breakout Backtester
    backtester = First15SecondsVolumeBreakoutBacktester(
        data_folder="data/symbolupdate",  # Folder containing database files
        symbols=None,  # Auto-detect symbols from databases
        volume_lookback=30,  # Look at last 30 1-minute candles for volume average
        trailing_stop_pct=2.0,  # 2% trailing stop loss
        initial_capital=100000,  # Starting capital per symbol
        square_off_time="15:20",  # 3:20 PM IST square-off time
        min_data_points=100,  # Minimum data points required per symbol
        entry_window_seconds=15  # Entry within first 15 seconds of candle
    )

    print("=" * 120)
    print("FIRST 15 SECONDS VOLUME BREAKOUT STRATEGY BACKTESTER")
    print("=" * 120)
    print("Strategy Rules:")
    print("1. Convert tick data to 1-minute candles")
    print("2. Calculate rolling 30-candle volume average")
    print("3. Entry Signal: Volume > 30-candle average AND entry within first 15 seconds")
    print("4. Exit Signal: 2% trailing stop loss")
    print("5. Mandatory square-off at 3:20 PM IST (no overnight positions)")
    print("6. Intraday only - multiple trades allowed per day")
    print("7. Auto-detection of symbols with sufficient tick data")
    print("=" * 120)

    try:
        # Check prerequisites
        if not backtester.db_files:
            print("ERROR: No database files found!")
            print("Setup Instructions:")
            print("1. Create a 'data' folder in the script directory")
            print("2. Place your .db files in the 'data' folder")
            print("3. Expected schema: market_data table with tick-level OHLCV data")
            exit(1)

        if not backtester.symbols:
            print("ERROR: No symbols detected with sufficient data!")
            print("Troubleshooting:")
            print("1. Check database files contain market_data table")
            print("2. Reduce min_data_points if needed")
            print("3. Verify database schema matches expected format")
            print("4. Ensure tick-level data is available")
            exit(1)

        # Run the backtest
        print(f"\nStarting First 15 Seconds Volume Breakout backtest...")
        print(f"Entry Condition: Volume > {backtester.volume_lookback}-candle average within first {backtester.entry_window_seconds} seconds")
        print(f"Exit Condition: {backtester.trailing_stop_pct * 100}% trailing stop loss")
        print(f"Square-off Time: {backtester.square_off_time} IST")

        results = backtester.run_backtest_sequential()

        if results:
            print(f"\nBacktest completed for {len(results)} symbols")

            # Create and display summary
            summary_df = backtester.create_summary_report()
            print("\n" + "=" * 150)
            print("FIRST 15 SECONDS VOLUME BREAKOUT STRATEGY BACKTEST SUMMARY")
            print("=" * 150)
            print(summary_df.round(2).to_string(index=False))

            # Export results
            print("\nExporting results to CSV files...")
            backtester.export_results("first_15sec_volume_breakout_strategy")

            # Performance analysis
            print("\n" + "=" * 120)
            print("TOP PERFORMERS BY TOTAL RETURN:")
            print("=" * 120)
            top_performers = summary_df.head(5)[['Symbol', 'Total Return (%)', 'Volume Breakout Win Rate (%)',
                                                 'Avg Volume Ratio', 'Avg Duration (min)', 'First 15-Sec Entries']]
            print(top_performers.to_string(index=False))

            # Strategy effectiveness analysis
            backtester.create_performance_insights()

            print("\n" + "=" * 120)
            print("FILES EXPORTED:")
            print("=" * 120)
            print("1. first_15sec_volume_breakout_strategy_summary.csv - Performance summary")
            print("2. first_15sec_volume_breakout_strategy_all_trades.csv - All trade records")
            print("3. first_15sec_volume_breakout_strategy_detailed_trades.csv - Detailed trade analysis")
            print("4. first_15sec_volume_breakout_strategy_volume_analysis.csv - Volume breakout analysis")
            print("=" * 120)

            # Additional insights
            print("\n" + "=" * 120)
            print("VOLUME BREAKOUT PATTERN ANALYSIS:")
            print("=" * 120)

            total_volume_breakout_trades = summary_df['Volume Breakout Trades'].sum()
            total_first_15_sec = summary_df['First 15-Sec Entries'].sum()
            avg_max_volume_ratio = summary_df['Max Volume Ratio'].mean()

            print(f"Total Volume Breakout Trades: {total_volume_breakout_trades}")
            print(f"Entries Within First 15 Seconds: {total_first_15_sec}")
            print(f"Average Maximum Volume Ratio: {avg_max_volume_ratio:.2f}x")

            # Volume ratio distribution
            volume_ratio_stats = summary_df['Avg Volume Ratio'].describe()
            print(f"\nVolume Ratio Distribution:")
            print(f"  Minimum: {volume_ratio_stats['min']:.2f}x")
            print(f"  25th Percentile: {volume_ratio_stats['25%']:.2f}x")
            print(f"  Median: {volume_ratio_stats['50%']:.2f}x")
            print(f"  75th Percentile: {volume_ratio_stats['75%']:.2f}x")
            print(f"  Maximum: {volume_ratio_stats['max']:.2f}x")

            # Best volume ratio performers
            high_volume_performers = summary_df[summary_df['Avg Volume Ratio'] > volume_ratio_stats['75%']]
            if not high_volume_performers.empty:
                print(f"\nTop Volume Ratio Performers (>{volume_ratio_stats['75%']:.2f}x):")
                for _, row in high_volume_performers.head(3).iterrows():
                    print(f"  {row['Symbol']}: {row['Avg Volume Ratio']:.2f}x ratio, {row['Total Return (%)']:.2f}% return")

            print("\n" + "=" * 120)
            print("TIMING ANALYSIS - FIRST 15 SECONDS ENTRY EFFECTIVENESS:")
            print("=" * 120)

            symbols_with_entries = summary_df[summary_df['First 15-Sec Entries'] > 0]
            if not symbols_with_entries.empty:
                avg_return_with_entries = symbols_with_entries['Total Return (%)'].mean()
                print(f"Symbols with First 15-Sec Entries: {len(symbols_with_entries)}")
                print(f"Average Return for Symbols with Entries: {avg_return_with_entries:.2f}%")

                best_timing_symbol = symbols_with_entries.loc[symbols_with_entries['Total Return (%)'].idxmax()]
                print(f"Best Timing Performance: {best_timing_symbol['Symbol']} (+{best_timing_symbol['Total Return (%)']:.2f}%)")
                print(f"  - First 15-Sec Entries: {best_timing_symbol['First 15-Sec Entries']}")
                print(f"  - Average Volume Ratio: {best_timing_symbol['Avg Volume Ratio']:.2f}x")
                print(f"  - Win Rate: {best_timing_symbol['Volume Breakout Win Rate (%)']:.1f}%")

            print("\n" + "=" * 120)
            print("STRATEGY VALIDATION SUMMARY:")
            print("=" * 120)

            validation_score = 0
            max_score = 5

            # Scoring criteria
            if summary_df['Total Return (%)'].mean() > 0:
                validation_score += 1
                print("✅ Positive average returns")
            else:
                print("❌ Negative average returns")

            if summary_df['Volume Breakout Win Rate (%)'].mean() > 50:
                validation_score += 1
                print("✅ Win rate > 50%")
            else:
                print("❌ Win rate ≤ 50%")

            if len([r for r in results.values() if r['metrics']['total_return'] > 0]) / len(results) > 0.5:
                validation_score += 1
                print("✅ >50% of symbols profitable")
            else:
                print("❌ ≤50% of symbols profitable")

            if summary_df['Avg Volume Ratio'].mean() > 1.5:
                validation_score += 1
                print("✅ Good volume breakout signals (>1.5x average)")
            else:
                print("❌ Weak volume breakout signals (≤1.5x average)")

            if summary_df['First 15-Sec Entries'].sum() > 0:
                validation_score += 1
                print("✅ Successfully captured first 15-second entries")
            else:
                print("❌ No first 15-second entries captured")

            print(f"\nStrategy Validation Score: {validation_score}/{max_score}")

            if validation_score >= 4:
                print("🎉 EXCELLENT: Strategy shows strong potential")
            elif validation_score >= 3:
                print("👍 GOOD: Strategy shows promise with room for optimization")
            elif validation_score >= 2:
                print("⚠️ FAIR: Strategy needs significant improvement")
            else:
                print("❌ POOR: Strategy requires major revision or different approach")

        else:
            print("No successful backtests completed.")
            print("Troubleshooting:")
            print("1. Check database files exist and are accessible")
            print("2. Verify sufficient tick-level data for 1-minute candle creation")
            print("3. Ensure database schema matches expected format")
            print("4. Check symbol detection and filtering criteria")
            print("5. Verify volume data is available and non-zero")

    except Exception as e:
        print(f"Error running First 15 Seconds Volume Breakout backtest: {e}")
        import traceback

        traceback.print_exc()