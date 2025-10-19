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


class OrderFlowBacktester:
    def __init__(self, data_folder="data", symbols=None, volume_lookback=20,
                 volume_imbalance_threshold=0.65, order_size_percentile=75,
                 momentum_period=5, trailing_stop_pct=0.4, initial_capital=100000,
                 square_off_time="15:20", min_data_points=100, debug_mode=False):
        """
        Initialize the Order Flow Trading Strategy Backtester

        Parameters:
        - data_folder: Folder containing database files (default: "data")
        - symbols: List of trading symbols to backtest (if None, auto-detect from databases)
        - volume_lookback: Period to calculate volume moving average (default 20)
        - volume_imbalance_threshold: Threshold for buy/sell volume imbalance (default 0.65 = 65%)
        - order_size_percentile: Percentile for large order detection (default 75)
        - momentum_period: Period for price momentum calculation (default 5)
        - trailing_stop_pct: Trailing stop loss percentage (default 0.4%)
        - initial_capital: Starting capital for backtesting (per symbol)
        - square_off_time: Time to square off all positions in HH:MM format (default "15:20")
        - min_data_points: Minimum data points required for a symbol to be included
        - debug_mode: Enable detailed diagnostic output (default False)
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()
        self.volume_lookback = volume_lookback
        self.volume_imbalance_threshold = volume_imbalance_threshold
        self.order_size_percentile = order_size_percentile
        self.momentum_period = momentum_period
        self.trailing_stop_pct = trailing_stop_pct / 100
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.debug_mode = debug_mode
        self.results = {}
        self.combined_data = {}

        # Set up IST timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print(f"Order Flow Trading Strategy Parameters:")
        print(f"  Data folder: {self.data_folder}")
        print(f"  Volume Lookback: {self.volume_lookback}")
        print(f"  Volume Imbalance Threshold: {volume_imbalance_threshold * 100}%")
        print(f"  Large Order Percentile: {order_size_percentile}th")
        print(f"  Momentum Period: {momentum_period}")
        print(f"  Trailing Stop: {trailing_stop_pct}%")
        print(f"  Square-off time: {square_off_time} IST")
        print(f"  Minimum data points: {self.min_data_points}")
        print(f"  Debug mode: {debug_mode}")

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
                if self.debug_mode:
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
                            'volume_raw': raw_data.get('vol_traded_today', 0),
                            'bid_price': raw_data.get('bid_price', np.nan),
                            'ask_price': raw_data.get('ask_price', np.nan),
                            'bid_size': raw_data.get('bid_size', 0),
                            'ask_size': raw_data.get('ask_size', 0),
                            'total_buy_qty': raw_data.get('total_buy_qty', 0),
                            'total_sell_qty': raw_data.get('total_sell_qty', 0)
                        })
                    else:
                        return pd.Series({
                            'high_raw': np.nan, 'low_raw': np.nan, 'volume_raw': 0,
                            'bid_price': np.nan, 'ask_price': np.nan,
                            'bid_size': 0, 'ask_size': 0,
                            'total_buy_qty': 0, 'total_sell_qty': 0
                        })
                except:
                    return pd.Series({
                        'high_raw': np.nan, 'low_raw': np.nan, 'volume_raw': 0,
                        'bid_price': np.nan, 'ask_price': np.nan,
                        'bid_size': 0, 'ask_size': 0,
                        'total_buy_qty': 0, 'total_sell_qty': 0
                    })

            raw_parsed = df['raw_data'].apply(parse_raw_data)
            df = pd.concat([df, raw_parsed], axis=1)

            # Use best available data
            df['high'] = df['high_raw'].fillna(df['high_price']).fillna(df['ltp'])
            df['low'] = df['low_raw'].fillna(df['low_price']).fillna(df['ltp'])
            df['close'] = df['close_price'].fillna(df['ltp'])
            df['volume_final'] = df['volume_raw'].fillna(df['volume']).fillna(0)

            # Handle order book data
            df['bid_price'] = df['bid_price'].fillna(df['close'] * 0.999)
            df['ask_price'] = df['ask_price'].fillna(df['close'] * 1.001)
            df['bid_size'] = df['bid_size'].fillna(0)
            df['ask_size'] = df['ask_size'].fillna(0)
            df['total_buy_qty'] = df['total_buy_qty'].fillna(0)
            df['total_sell_qty'] = df['total_sell_qty'].fillna(0)

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            # Remove rows with missing critical data
            df = df.dropna(subset=['close', 'high', 'low'])

            return df[['close', 'high', 'low', 'volume_final', 'bid_price', 'ask_price',
                       'bid_size', 'ask_size', 'total_buy_qty', 'total_sell_qty']].copy()

        except Exception as e:
            if self.debug_mode:
                print(f"    Error accessing database {db_path}: {e}")
            return None

    def validate_order_flow_data(self, df):
        """Validate that order flow data exists and is usable"""
        issues = []

        # Check for order book data
        bid_ask_available = ((df['bid_size'] > 0) | (df['ask_size'] > 0)).sum()
        bid_ask_coverage = bid_ask_available / len(df) if len(df) > 0 else 0

        # Check for buy/sell volume data
        buy_sell_available = ((df['total_buy_qty'] > 0) | (df['total_sell_qty'] > 0)).sum()
        buy_sell_coverage = buy_sell_available / len(df) if len(df) > 0 else 0

        if self.debug_mode:
            print(f"  Order book data coverage: {bid_ask_coverage * 100:.1f}% ({bid_ask_available}/{len(df)} rows)")
            print(f"  Buy/sell volume coverage: {buy_sell_coverage * 100:.1f}% ({buy_sell_available}/{len(df)} rows)")

        if bid_ask_coverage < 0.01:
            issues.append("No meaningful bid/ask size data (< 1% coverage)")
        if buy_sell_coverage < 0.01:
            issues.append("No meaningful buy/sell quantity data (< 1% coverage)")

        return issues, bid_ask_coverage, buy_sell_coverage

    def calculate_order_flow_indicators(self, df):
        """Calculate order flow and volume analysis indicators"""

        # Volume moving average
        df['volume_ma'] = df['volume_final'].rolling(window=self.volume_lookback).mean()

        # Volume ratio (current volume vs average)
        df['volume_ratio'] = df['volume_final'] / df['volume_ma']

        # Large volume threshold (based on percentile)
        df['volume_threshold'] = df['volume_final'].rolling(window=self.volume_lookback).quantile(self.order_size_percentile / 100)
        df['large_volume'] = df['volume_final'] > df['volume_threshold']

        # Order book imbalance (bid vs ask sizes)
        df['total_order_size'] = df['bid_size'] + df['ask_size']
        df['bid_ask_imbalance'] = np.where(
            df['total_order_size'] > 0,
            (df['bid_size'] - df['ask_size']) / df['total_order_size'],
            0
        )

        # Buy/Sell quantity imbalance
        df['total_order_qty'] = df['total_buy_qty'] + df['total_sell_qty']
        df['buy_sell_imbalance'] = np.where(
            df['total_order_qty'] > 0,
            df['total_buy_qty'] / df['total_order_qty'],
            0.5  # Neutral if no data
        )

        # Price momentum
        df['price_change'] = df['close'].pct_change()
        df['momentum'] = df['price_change'].rolling(window=self.momentum_period).mean()

        # Spread analysis
        df['spread'] = df['ask_price'] - df['bid_price']
        df['spread_pct'] = df['spread'] / df['close']
        df['avg_spread'] = df['spread_pct'].rolling(window=self.volume_lookback).mean()

        # Volume-Price Trend (VPT)
        df['vpt'] = (df['price_change'] * df['volume_final']).cumsum()
        df['vpt_ma'] = df['vpt'].rolling(window=self.volume_lookback).mean()

        # Order flow momentum
        df['order_flow_momentum'] = df['buy_sell_imbalance'].rolling(window=self.momentum_period).mean()

        return df

    def generate_signals(self, df, has_order_book, has_buy_sell):
        """Generate buy/sell signals with adaptive logic based on data availability"""

        if self.debug_mode:
            print(f"  Generating signals - Order book: {has_order_book}, Buy/Sell data: {has_buy_sell}")

        # Core conditions that always apply
        core_buy_conditions = (
                (df['large_volume'] == True) &
                (df['momentum'] > 0) &
                (df['volume_ratio'] > 1.1) &  # Relaxed from 1.2
                (~df['momentum'].isna()) &
                (~df['volume_ratio'].isna())
        )

        core_sell_conditions = (
                (df['large_volume'] == True) &
                (df['momentum'] < 0) &
                (df['volume_ratio'] > 1.1) &
                (~df['momentum'].isna()) &
                (~df['volume_ratio'].isna())
        )

        # Add order flow conditions if available
        if has_buy_sell:
            # Use buy/sell imbalance - relaxed threshold to 0.60 from 0.65
            buy_flow_condition = (
                    (df['buy_sell_imbalance'] > 0.60) &
                    (~df['buy_sell_imbalance'].isna())
            )
            sell_flow_condition = (
                    (df['buy_sell_imbalance'] < 0.40) &
                    (~df['buy_sell_imbalance'].isna())
            )

            df['buy_signal'] = core_buy_conditions & buy_flow_condition
            df['sell_signal'] = core_sell_conditions & sell_flow_condition

            if self.debug_mode:
                print(f"    Using buy/sell imbalance (threshold: 0.60/0.40)")
        else:
            # Fallback: use VPT when no order flow data
            vpt_buy_condition = (
                    (df['vpt'] > df['vpt_ma']) &
                    (~df['vpt'].isna()) &
                    (~df['vpt_ma'].isna())
            )
            vpt_sell_condition = (
                    (df['vpt'] < df['vpt_ma']) &
                    (~df['vpt'].isna()) &
                    (~df['vpt_ma'].isna())
            )

            df['buy_signal'] = core_buy_conditions & vpt_buy_condition
            df['sell_signal'] = core_sell_conditions & vpt_sell_condition

            if self.debug_mode:
                print(f"    Using VPT fallback (no buy/sell data)")

        # Only add bid-ask filter if we have meaningful order book data (>10% coverage)
        if has_order_book > 0.10:
            df['buy_signal'] = df['buy_signal'] & (df['bid_ask_imbalance'] > 0.05)  # Relaxed from 0.1
            df['sell_signal'] = df['sell_signal'] & (df['bid_ask_imbalance'] < -0.05)  # Relaxed from -0.1
            if self.debug_mode:
                print(f"    Applied bid-ask imbalance filter (±0.05)")
        else:
            if self.debug_mode:
                print(f"    Skipped bid-ask filter (insufficient data)")

        buy_count = df['buy_signal'].sum()
        sell_count = df['sell_signal'].sum()

        if self.debug_mode:
            print(f"  Generated {buy_count} buy signals and {sell_count} sell signals")

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
        """Backtest order flow strategy for a single symbol"""
        print(f"\n{'=' * 80}")
        print(f"Backtesting {symbol}")
        print(f"{'=' * 80}")

        # Load data
        df = self.load_data_from_all_databases(symbol)

        if df is None or len(df) < max(self.volume_lookback, self.momentum_period) * 2:
            print(f"Insufficient data for {symbol}. Skipping.")
            return None

        # Validate data quality
        issues, bid_ask_cov, buy_sell_cov = self.validate_order_flow_data(df)

        if issues:
            print(f"⚠️  Data Quality Issues:")
            for issue in issues:
                print(f"  - {issue}")

        # Store combined data
        self.combined_data[symbol] = df.copy()

        # Calculate indicators
        df = self.calculate_order_flow_indicators(df)

        # Debug: Print indicator statistics
        if self.debug_mode:
            print(f"\nIndicator Statistics:")
            print(f"  Valid rows (no NaN in key indicators): {(~df['momentum'].isna() & ~df['volume_ratio'].isna()).sum()} / {len(df)}")
            print(f"  large_volume == True: {(df['large_volume'] == True).sum()}")
            print(f"  momentum > 0: {(df['momentum'] > 0).sum()}")
            print(f"  momentum < 0: {(df['momentum'] < 0).sum()}")
            print(f"  volume_ratio > 1.1: {(df['volume_ratio'] > 1.1).sum()}")
            print(f"  buy_sell_imbalance > 0.60: {(df['buy_sell_imbalance'] > 0.60).sum()}")
            print(f"  buy_sell_imbalance < 0.40: {(df['buy_sell_imbalance'] < 0.40).sum()}")
            print(f"  buy_sell_imbalance range: [{df['buy_sell_imbalance'].min():.3f}, {df['buy_sell_imbalance'].max():.3f}]")
            print(f"  bid_ask_imbalance range: [{df['bid_ask_imbalance'].min():.3f}, {df['bid_ask_imbalance'].max():.3f}]")

        # Generate signals with adaptive logic
        df = self.generate_signals(df, bid_ask_cov, buy_sell_cov)

        # Add trading day and square-off time information
        df['trading_day'] = df.index.date
        df['is_square_off_time'] = df.index.map(self.is_square_off_time)

        # Initialize trading variables
        cash = self.initial_capital
        position = 0
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
            if position != 0 and is_square_off:
                if position == 1:
                    shares = trades[-1]['shares']
                    proceeds = shares * current_price
                    cash += proceeds
                elif position == -1:
                    shares = trades[-1]['shares']
                    cost = shares * current_price
                    cash -= cost

                position = 0

                trades.append({
                    'date': current_date,
                    'action': 'COVER' if position == -1 else 'SELL',
                    'price': round(current_price, 2),
                    'shares': shares,
                    'cash': round(cash, 2),
                    'reason': 'SQUARE_OFF_3_20_PM',
                    'buy_sell_imbalance': round(df.iloc[i]['buy_sell_imbalance'], 3),
                    'bid_ask_imbalance': round(df.iloc[i]['bid_ask_imbalance'], 3),
                    'volume_ratio': round(df.iloc[i]['volume_ratio'], 2),
                    'momentum': round(df.iloc[i]['momentum'], 4),
                    'source_db': df.iloc[i].get('source_db', 'Unknown'),
                    'exit_time': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_price': round(current_price, 2),
                    'entry_price': round(trades[-1]['entry_price'], 2) if trades else 0,
                    'trade_return_pct': round(((current_price - trades[-1]['entry_price']) / trades[-1]['entry_price'] * 100), 2) if trades else 0,
                    'trade_pnl': round((shares * (current_price - trades[-1]['entry_price'])), 2) if trades else 0
                })

                entry_day = None

            # Check for buy signal
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
                        'buy_sell_imbalance': round(df.iloc[i]['buy_sell_imbalance'], 3),
                        'bid_ask_imbalance': round(df.iloc[i]['bid_ask_imbalance'], 3),
                        'volume_ratio': round(df.iloc[i]['volume_ratio'], 2),
                        'momentum': round(df.iloc[i]['momentum'], 4),
                        'large_volume': df.iloc[i]['large_volume'],
                        'source_db': df.iloc[i].get('source_db', 'Unknown'),
                        'entry_day': entry_day,
                        'entry_time': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'entry_price': round(current_price, 2)
                    })

            # Check for sell signal
            elif df.iloc[i]['sell_signal'] and position == 0 and not is_square_off:
                shares = int(cash / current_price)
                if shares > 0:
                    position = -1
                    entry_price = current_price
                    entry_day = current_day
                    trailing_stop_price = entry_price * (1 + self.trailing_stop_pct)
                    proceeds = shares * current_price
                    cash += proceeds

                    trades.append({
                        'date': current_date,
                        'action': 'SHORT',
                        'price': round(current_price, 2),
                        'shares': shares,
                        'cash': round(cash, 2),
                        'buy_sell_imbalance': round(df.iloc[i]['buy_sell_imbalance'], 3),
                        'bid_ask_imbalance': round(df.iloc[i]['bid_ask_imbalance'], 3),
                        'volume_ratio': round(df.iloc[i]['volume_ratio'], 2),
                        'momentum': round(df.iloc[i]['momentum'], 4),
                        'large_volume': df.iloc[i]['large_volume'],
                        'source_db': df.iloc[i].get('source_db', 'Unknown'),
                        'entry_day': entry_day,
                        'entry_time': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'entry_price': round(current_price, 2)
                    })

            # Check for exit conditions
            elif position != 0 and entry_day == current_day and not is_square_off:
                should_exit = False
                exit_reason = ''

                if position == 1:
                    if current_price > entry_price:
                        new_trailing_stop = current_price * (1 - self.trailing_stop_pct)
                        trailing_stop_price = max(trailing_stop_price, new_trailing_stop)

                    if current_price <= trailing_stop_price:
                        should_exit = True
                        exit_reason = 'TRAILING_STOP'
                    elif df.iloc[i]['sell_signal']:
                        should_exit = True
                        exit_reason = 'SELL_SIGNAL'

                elif position == -1:
                    if current_price < entry_price:
                        new_trailing_stop = current_price * (1 + self.trailing_stop_pct)
                        trailing_stop_price = min(trailing_stop_price, new_trailing_stop)

                    if current_price >= trailing_stop_price:
                        should_exit = True
                        exit_reason = 'TRAILING_STOP'
                    elif df.iloc[i]['buy_signal']:
                        should_exit = True
                        exit_reason = 'BUY_SIGNAL'

                if should_exit:
                    shares = trades[-1]['shares']

                    if position == 1:
                        proceeds = shares * current_price
                        cash += proceeds
                        action = 'SELL'
                    else:
                        cost = shares * current_price
                        cash -= cost
                        action = 'COVER'

                    position = 0

                    pnl_multiplier = 1 if action == 'SELL' else -1
                    trade_pnl = shares * (current_price - trades[-1]['entry_price']) * pnl_multiplier

                    trades.append({
                        'date': current_date,
                        'action': action,
                        'price': round(current_price, 2),
                        'shares': shares,
                        'cash': round(cash, 2),
                        'reason': exit_reason,
                        'buy_sell_imbalance': round(df.iloc[i]['buy_sell_imbalance'], 3),
                        'bid_ask_imbalance': round(df.iloc[i]['bid_ask_imbalance'], 3),
                        'volume_ratio': round(df.iloc[i]['volume_ratio'], 2),
                        'momentum': round(df.iloc[i]['momentum'], 4),
                        'source_db': df.iloc[i].get('source_db', 'Unknown'),
                        'exit_time': current_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'exit_price': round(current_price, 2),
                        'entry_price': round(trades[-1]['entry_price'], 2) if trades else 0,
                        'trade_return_pct': round((trade_pnl / (shares * trades[-1]['entry_price']) * 100), 2) if trades else 0,
                        'trade_pnl': round(trade_pnl, 2)
                    })

                    entry_day = None

            # Calculate portfolio value
            if position == 1:
                shares = trades[-1]['shares']
                portfolio_val = cash + (shares * current_price)
            elif position == -1:
                shares = trades[-1]['shares']
                portfolio_val = cash - (shares * (current_price - trades[-1]['entry_price']))
            else:
                portfolio_val = cash

            portfolio_value.append({
                'date': current_date,
                'value': round(portfolio_val, 2),
                'position': position,
                'trading_day': current_day,
                'is_square_off_time': is_square_off,
                'buy_sell_imbalance': round(df.iloc[i]['buy_sell_imbalance'], 3) if not pd.isna(df.iloc[i]['buy_sell_imbalance']) else 0,
                'volume_ratio': round(df.iloc[i]['volume_ratio'], 2) if not pd.isna(df.iloc[i]['volume_ratio']) else 0
            })

        # Calculate metrics
        portfolio_df = pd.DataFrame(portfolio_value)
        metrics = self.calculate_metrics(portfolio_df, trades)

        # Add order flow specific metrics
        order_flow_metrics = self.calculate_order_flow_metrics(trades)
        metrics.update(order_flow_metrics)

        # Print trade summary
        print(f"✓ Completed: {len(trades)} total actions ({metrics['total_trades']} complete trades)")
        if metrics['total_trades'] > 0:
            print(f"  Return: {metrics['total_return'] * 100:.2f}% | Win Rate: {metrics['win_rate'] * 100:.1f}%")

        return {
            'symbol': symbol,
            'data': df,
            'portfolio': portfolio_df,
            'trades': trades,
            'metrics': metrics,
            'data_summary': self.get_data_summary(df)
        }

    def calculate_order_flow_metrics(self, trades):
        """Calculate order flow specific performance metrics"""
        if not trades:
            return {
                'total_order_flow_trades': 0,
                'long_trades': 0,
                'short_trades': 0,
                'avg_trade_duration_minutes': 0,
                'trailing_stop_exits': 0,
                'signal_exits': 0,
                'square_off_3_20_closures': 0,
                'order_flow_win_rate': 0,
                'avg_order_flow_return': 0,
                'best_order_flow_return': 0,
                'worst_order_flow_return': 0,
                'avg_buy_sell_imbalance_entry': 0,
                'avg_volume_ratio_entry': 0,
                'avg_momentum_entry': 0
            }

        buy_trades = [t for t in trades if t['action'] in ['BUY', 'SHORT']]
        sell_trades = [t for t in trades if t['action'] in ['SELL', 'COVER']]

        # Calculate trade metrics
        trade_durations = []
        long_count = 0
        short_count = 0
        trailing_stop_count = 0
        signal_exit_count = 0
        square_off_count = 0
        order_flow_returns = []
        entry_buy_sell_imbalance = []
        entry_volume_ratio = []
        entry_momentum = []

        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_trade = buy_trades[i]
            sell_trade = sell_trades[i]

            # Duration
            buy_time = pd.to_datetime(buy_trade['date'])
            sell_time = pd.to_datetime(sell_trade['date'])
            duration_minutes = (sell_time - buy_time).total_seconds() / 60
            trade_durations.append(duration_minutes)

            # Trade type
            if buy_trade['action'] == 'BUY':
                long_count += 1
            elif buy_trade['action'] == 'SHORT':
                short_count += 1

            # Exit reasons
            exit_reason = sell_trade.get('reason', 'UNKNOWN')
            if exit_reason == 'TRAILING_STOP':
                trailing_stop_count += 1
            elif exit_reason in ['SELL_SIGNAL', 'BUY_SIGNAL']:
                signal_exit_count += 1
            elif exit_reason == 'SQUARE_OFF_3_20_PM':
                square_off_count += 1

            # Returns
            if 'trade_return_pct' in sell_trade:
                trade_return = sell_trade['trade_return_pct'] / 100
            else:
                if buy_trade['action'] == 'BUY':
                    trade_return = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
                else:
                    trade_return = (buy_trade['price'] - sell_trade['price']) / buy_trade['price']

            order_flow_returns.append(trade_return)

            # Entry conditions
            entry_buy_sell_imbalance.append(buy_trade.get('buy_sell_imbalance', 0))
            entry_volume_ratio.append(buy_trade.get('volume_ratio', 0))
            entry_momentum.append(buy_trade.get('momentum', 0))

        # Calculate metrics
        avg_duration = np.mean(trade_durations) if trade_durations else 0
        order_flow_win_rate = sum(1 for r in order_flow_returns if r > 0) / len(order_flow_returns) if order_flow_returns else 0
        avg_order_flow_return = np.mean(order_flow_returns) if order_flow_returns else 0
        best_order_flow = max(order_flow_returns) if order_flow_returns else 0
        worst_order_flow = min(order_flow_returns) if order_flow_returns else 0
        avg_entry_imbalance = np.mean(entry_buy_sell_imbalance) if entry_buy_sell_imbalance else 0
        avg_entry_vol_ratio = np.mean(entry_volume_ratio) if entry_volume_ratio else 0
        avg_entry_momentum = np.mean(entry_momentum) if entry_momentum else 0

        return {
            'total_order_flow_trades': len(buy_trades),
            'long_trades': long_count,
            'short_trades': short_count,
            'avg_trade_duration_minutes': round(avg_duration, 2),
            'trailing_stop_exits': trailing_stop_count,
            'signal_exits': signal_exit_count,
            'square_off_3_20_closures': square_off_count,
            'order_flow_win_rate': round(order_flow_win_rate, 4),
            'avg_order_flow_return': round(avg_order_flow_return, 4),
            'best_order_flow_return': round(best_order_flow, 4),
            'worst_order_flow_return': round(worst_order_flow, 4),
            'avg_buy_sell_imbalance_entry': round(avg_entry_imbalance, 3),
            'avg_volume_ratio_entry': round(avg_entry_vol_ratio, 2),
            'avg_momentum_entry': round(avg_entry_momentum, 4)
        }

    def get_data_summary(self, df):
        """Get summary statistics for the data"""
        unique_days = len(df['trading_day'].unique()) if 'trading_day' in df.columns else len(set(df.index.date))

        return {
            'total_records': len(df),
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'trading_days': unique_days,
            'avg_daily_volume': round(df['volume_final'].mean(), 2),
            'databases_used': df['source_db'].nunique() if 'source_db' in df.columns else 0,
            'avg_records_per_day': round(len(df) / unique_days, 2) if unique_days > 0 else 0,
            'avg_buy_sell_imbalance': round(df['buy_sell_imbalance'].mean(), 3),
            'avg_volume_ratio': round(df['volume_ratio'].mean(), 2),
            'avg_spread_pct': round(df['spread_pct'].mean(), 4)
        }

    def calculate_metrics(self, portfolio_df, trades):
        """Calculate performance metrics"""
        total_return = (portfolio_df.iloc[-1]['value'] - self.initial_capital) / self.initial_capital

        # Trade-based metrics
        buy_trades = [t for t in trades if t['action'] in ['BUY', 'SHORT']]
        sell_trades = [t for t in trades if t['action'] in ['SELL', 'COVER']]

        trade_returns = []
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_trade = buy_trades[i]
            sell_trade = sell_trades[i]

            if 'trade_return_pct' in sell_trade:
                trade_return = sell_trade['trade_return_pct'] / 100
            else:
                if buy_trade['action'] == 'BUY':
                    trade_return = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
                else:
                    trade_return = (buy_trade['price'] - sell_trade['price']) / buy_trade['price']

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
        print(f"\n{'=' * 80}")
        print(f"Starting order flow backtest for {len(self.symbols)} symbols")
        print(f"{'=' * 80}")

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
                'Order Flow Trades': metrics.get('total_order_flow_trades', 0),
                'Long Trades': metrics.get('long_trades', 0),
                'Short Trades': metrics.get('short_trades', 0),
                'Avg Duration (min)': metrics.get('avg_trade_duration_minutes', 0),
                'Trailing Stop Exits': metrics.get('trailing_stop_exits', 0),
                'Signal Exits': metrics.get('signal_exits', 0),
                '3:20 PM Square-offs': metrics.get('square_off_3_20_closures', 0),
                'Order Flow Win Rate (%)': metrics.get('order_flow_win_rate', 0) * 100,
                'Best Order Flow (%)': metrics.get('best_order_flow_return', 0) * 100,
                'Worst Order Flow (%)': metrics.get('worst_order_flow_return', 0) * 100,
                'Avg Entry Imbalance': metrics.get('avg_buy_sell_imbalance_entry', 0),
                'Avg Entry Vol Ratio': metrics.get('avg_volume_ratio_entry', 0),
                'Avg Entry Momentum': metrics.get('avg_momentum_entry', 0),
                'Data Points': data_summary['total_records'],
                'Trading Days': data_summary['trading_days'],
                'Avg Buy/Sell Imbalance': data_summary.get('avg_buy_sell_imbalance', 0),
                'Avg Volume Ratio': data_summary.get('avg_volume_ratio', 0),
                'Avg Spread (%)': data_summary.get('avg_spread_pct', 0) * 100
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Total Return (%)', ascending=False)

        return summary_df

    def export_results(self, filename_prefix="order_flow_strategy"):
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
            buy_trades = [t for t in trades if t['action'] in ['BUY', 'SHORT']]
            sell_trades = [t for t in trades if t['action'] in ['SELL', 'COVER']]

            for i in range(min(len(buy_trades), len(sell_trades))):
                buy_trade = buy_trades[i]
                sell_trade = sell_trades[i]

                buy_time = pd.to_datetime(buy_trade['date'])
                sell_time = pd.to_datetime(sell_trade['date'])
                duration_minutes = (sell_time - buy_time).total_seconds() / 60

                if 'trade_return_pct' in sell_trade:
                    trade_return = sell_trade['trade_return_pct']
                    trade_pnl = sell_trade.get('trade_pnl', 0)
                else:
                    if buy_trade['action'] == 'BUY':
                        trade_return = (sell_trade['price'] - buy_trade['price']) / buy_trade['price'] * 100
                    else:
                        trade_return = (buy_trade['price'] - sell_trade['price']) / buy_trade['price'] * 100
                    trade_pnl = buy_trade['shares'] * (sell_trade['price'] - buy_trade['price'])
                    if buy_trade['action'] == 'SHORT':
                        trade_pnl *= -1

                trade_analysis.append({
                    'Symbol': symbol,
                    'Entry_Date': buy_trade['date'],
                    'Exit_Date': sell_trade['date'],
                    'Trade_Type': buy_trade['action'],
                    'Entry_Price': buy_trade['price'],
                    'Exit_Price': sell_trade['price'],
                    'Shares': buy_trade['shares'],
                    'Duration_Minutes': round(duration_minutes, 2),
                    'Trade_Return_Pct': round(trade_return, 2),
                    'Trade_PnL': round(trade_pnl, 2),
                    'Exit_Reason': sell_trade.get('reason', 'UNKNOWN'),
                    'Entry_Buy_Sell_Imbalance': buy_trade.get('buy_sell_imbalance', 0),
                    'Entry_Bid_Ask_Imbalance': buy_trade.get('bid_ask_imbalance', 0),
                    'Entry_Volume_Ratio': buy_trade.get('volume_ratio', 0),
                    'Entry_Momentum': buy_trade.get('momentum', 0),
                    'Entry_Large_Volume': buy_trade.get('large_volume', False),
                    'Exit_Buy_Sell_Imbalance': sell_trade.get('buy_sell_imbalance', 0),
                    'Exit_Volume_Ratio': sell_trade.get('volume_ratio', 0),
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

        date_ranges = self.get_database_date_ranges()

        print(f"\nDatabase Files Found: {len(self.db_files)}")
        print("-" * 50)

        for db_name, info in date_ranges.items():
            print(f"File: {db_name}")
            print(f"  Date Range: {info['min_date']} to {info['max_date']}")
            print(f"  Full Path: {info['file_path']}")
            print()

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


# Example usage and main execution
if __name__ == "__main__":
    # Initialize with debug_mode=True to see detailed diagnostics
    backtester = OrderFlowBacktester(
        data_folder="data/symbolupdate",
        symbols=None,
        volume_lookback=200,
        volume_imbalance_threshold=0.60,  # Relaxed from 0.65
        order_size_percentile=75,
        momentum_period=5,
        trailing_stop_pct=0.4,
        initial_capital=100000,
        square_off_time="15:20",
        min_data_points=100,
        debug_mode=True  # Enable diagnostics
    )

    print("=" * 120)
    print("ORDER FLOW TRADING STRATEGY BACKTESTER - DIAGNOSTIC MODE")
    print("=" * 120)
    print("Key Changes:")
    print("✓ Relaxed buy/sell imbalance threshold: 60%/40% (was 65%/35%)")
    print("✓ Relaxed volume ratio requirement: 1.1x (was 1.2x)")
    print("✓ Relaxed bid-ask imbalance filter: ±0.05 (was ±0.1)")
    print("✓ Adaptive signal generation based on data availability")
    print("✓ Falls back to VPT when no buy/sell data available")
    print("✓ Detailed diagnostic output enabled")
    print("=" * 120)

    try:
        if not backtester.db_files:
            print("ERROR: No database files found!")
            exit(1)

        if not backtester.symbols:
            print("ERROR: No symbols detected with sufficient data!")
            exit(1)

        backtester.create_database_analysis_report()

        print(f"\nStarting backtest with relaxed parameters...")
        results = backtester.run_backtest_sequential()

        if results:
            print(f"\n{'=' * 120}")
            print(f"Backtest completed for {len(results)} symbols")
            print(f"{'=' * 120}")

            summary_df = backtester.create_summary_report()
            print("\nSUMMARY REPORT")
            print("=" * 180)
            print(summary_df.round(2).to_string(index=False))

            print("\nExporting results...")
            backtester.export_results("order_flow_trading_strategy")
            backtester.export_symbol_analysis("order_flow_trading_strategy")

            print("\n" + "=" * 120)
            print("FILES EXPORTED:")
            print("=" * 120)
            print("1. order_flow_trading_strategy_summary.csv")
            print("2. order_flow_trading_strategy_all_trades.csv")
            print("3. order_flow_trading_strategy_detailed_trades.csv")
            print("4. order_flow_trading_strategy_symbol_analysis.csv")

        else:
            print("\n⚠️  No successful backtests completed.")
            print("\nTroubleshooting steps:")
            print("1. Check if debug output shows any signals being generated")
            print("2. Verify your raw_data contains order flow fields")
            print("3. Try further relaxing parameters (threshold=0.55, volume_ratio=1.0)")
            print("4. Check if data has sufficient variability")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()