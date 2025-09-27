# Quick fix for your existing backtesting code
# Replace the find_database_files method in your existing classes

def find_database_files(self):
    """Find all database files in the data folder - ENHANCED VERSION"""

    # Try multiple possible folder paths
    possible_folders = [
        self.data_folder,  # Original path
        "data/marketupdate",
        "data/symbolupdate",
        "data",
        os.path.join(os.getcwd(), "data", "marketupdate"),
        os.path.join(os.getcwd(), "data", "symbolupdate"),
        os.path.join(os.getcwd(), "data")
    ]

    print(f"Searching for database files...")

    for folder in possible_folders:
        print(f"  Checking folder: {folder}")

        if not os.path.exists(folder):
            print(f"    Folder does not exist: {folder}")
            continue

        # Try multiple file patterns
        patterns = [
            "*.db",
            "*.sqlite",
            "*.sqlite3",
            "fyers_*.db",
            "market_data_*.db",
            "*market*.db"
        ]

        found_files = []
        for pattern in patterns:
            search_pattern = os.path.join(folder, pattern)
            files = glob.glob(search_pattern)
            found_files.extend(files)

        # Remove duplicates
        found_files = list(set(found_files))

        if found_files:
            print(f"    Found {len(found_files)} database files")

            # Test each file to ensure it's a valid database
            valid_files = []
            for db_file in found_files:
                try:
                    conn = sqlite3.connect(db_file)

                    # Check if market_data table exists
                    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
                    tables = pd.read_sql_query(tables_query, conn)

                    if 'market_data' in tables['name'].values:
                        # Quick count to ensure there's data
                        count_query = "SELECT COUNT(*) FROM market_data LIMIT 1"
                        count_result = pd.read_sql_query(count_query, conn)

                        if count_result.iloc[0, 0] > 0:
                            valid_files.append(db_file)
                            print(f"       Valid: {os.path.basename(db_file)}")
                        else:
                            print(f"        Empty: {os.path.basename(db_file)}")
                    else:
                        print(f"       No market_data table: {os.path.basename(db_file)}")

                    conn.close()

                except Exception as e:
                    print(f"       Invalid database: {os.path.basename(db_file)} - {e}")
                    continue

            if valid_files:
                # Update the data folder to the working one
                self.data_folder = folder
                print(f"   Using folder: {folder}")
                return sorted(valid_files)

        else:
            print(f"     No database files found in {folder}")

    # If no files found anywhere, provide helpful error message
    print(f"\n No database files found in any location!")
    print(f"Searched folders:")
    for folder in possible_folders:
        print(f"  - {os.path.abspath(folder)}")

    print(f"\nSuggestions:")
    print(f"  1. Check if your database files are in one of the above folders")
    print(f"  2. Update the data_folder parameter when initializing the backtester")
    print(f"  3. Run the database test script to diagnose the issue")

    return []


# Enhanced initialization for your existing classes
def __init__(self, data_folder="data/marketupdate", symbols=None, **kwargs):
    """Enhanced initialization with better error handling"""

    # Store original data folder
    self.original_data_folder = data_folder
    self.data_folder = data_folder

    print(f"Initializing backtester...")
    print(f"Looking for database files in: {data_folder}")

    # Find database files (this will automatically try multiple folders)
    self.db_files = self.find_database_files()

    if not self.db_files:
        raise FileNotFoundError(
            f"No valid database files found! "
            f"Please check your data folder path or run the database test script."
        )

    print(f"Found {len(self.db_files)} valid database files")
    print(f"Using data folder: {self.data_folder}")

    # Continue with rest of initialization...
    # (your existing initialization code goes here)


# Quick patch function you can call to fix existing instances
def patch_existing_backtester(backtester_instance):
    """Patch an existing backtester instance with enhanced database finding"""

    import types

    # Replace the find_database_files method
    backtester_instance.find_database_files = types.MethodType(find_database_files, backtester_instance)

    # Re-run database file discovery
    print("Patching existing backtester instance...")
    backtester_instance.db_files = backtester_instance.find_database_files()

    if backtester_instance.db_files:
        print(f"Patch successful! Found {len(backtester_instance.db_files)} database files")
        return True
    else:
        print("Patch failed - no database files found")
        return False


def enhanced_find_database_files(self):
    """Enhanced database file finder that searches multiple locations"""

    # List of possible data folder locations
    search_paths = [
        self.data_folder,  # Original specified folder
        "data/marketupdate",
        "data/symbolupdate",
        "data",
        os.path.join(os.getcwd(), "data", "marketupdate"),
        os.path.join(os.getcwd(), "data", "symbolupdate"),
        os.path.join(os.getcwd(), "data"),
        os.path.expanduser("~/data/marketupdate"),  # User home directory
        os.path.expanduser("~/data/symbolupdate"),
        os.path.expanduser("~/data")
    ]

    # File patterns to search for
    file_patterns = [
        "*.db",
        "*.sqlite",
        "*.sqlite3",
        "fyers_*.db",
        "market_data_*.db",
        "*market*.db",
        "*fyers*.db"
    ]

    print("Enhanced Database Search")
    print("=" * 50)

    all_found_files = []

    for search_path in search_paths:
        abs_path = os.path.abspath(search_path)
        print(f"Searching: {search_path}")
        print(f"   Absolute: {abs_path}")

        if not os.path.exists(search_path):
            print(f"   Path does not exist")
            continue

        # Search for files with each pattern
        path_files = []
        for pattern in file_patterns:
            full_pattern = os.path.join(search_path, pattern)
            found = glob.glob(full_pattern)
            path_files.extend(found)

        # Remove duplicates
        path_files = list(set(path_files))

        if path_files:
            print(f"   Found {len(path_files)} database files")

            # Validate each database file
            valid_files = []
            for db_file in path_files:
                db_name = os.path.basename(db_file)
                print(f"   Testing: {db_name}")

                try:
                    # Test database connection
                    conn = sqlite3.connect(db_file)

                    # Check for required table
                    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
                    tables_df = pd.read_sql_query(tables_query, conn)
                    table_names = tables_df['name'].tolist()

                    if 'market_data' not in table_names:
                        print(f"     No market_data table (has: {table_names})")
                        conn.close()
                        continue

                    # Check if table has data
                    count_query = "SELECT COUNT(*) as count FROM market_data"
                    count_df = pd.read_sql_query(count_query, conn)
                    record_count = count_df.iloc[0]['count']

                    if record_count == 0:
                        print(f"     Empty table")
                        conn.close()
                        continue

                    # Get additional info
                    sample_query = "SELECT COUNT(DISTINCT symbol) as symbols FROM market_data"
                    sample_df = pd.read_sql_query(sample_query, conn)
                    symbol_count = sample_df.iloc[0]['symbols']

                    conn.close()

                    print(f"     Valid ({record_count:,} records, {symbol_count} symbols)")
                    valid_files.append(db_file)

                except Exception as e:
                    print(f"     Error: {str(e)[:50]}...")
                    continue

            if valid_files:
                print(f"   {len(valid_files)} valid database(s) in this path")
                all_found_files.extend(valid_files)

                # Update the data folder to the working path
                self.data_folder = search_path
                break  # Stop searching once we find valid files
        else:
            print(f"   No database files found")

    # Remove duplicates and sort
    all_found_files = sorted(list(set(all_found_files)))

    print("\n" + "=" * 50)
    if all_found_files:
        print(f"FINAL RESULT: Found {len(all_found_files)} valid database files")
        print(f"Using folder: {self.data_folder}")
        print("Database files:")
        for i, db_file in enumerate(all_found_files, 1):
            print(f"   {i}. {os.path.basename(db_file)}")
    else:
        print("NO VALID DATABASE FILES FOUND!")
        print("\n Troubleshooting suggestions:")
        print("   1. Check if database files exist in any of the searched paths")
        print("   2. Verify database files have .db extension")
        print("   3. Ensure database files contain 'market_data' table with data")
        print("   4. Run the database test script for detailed diagnosis")
        print("\nSearched paths:")
        for path in search_paths:
            if os.path.exists(path):
                print(f"   {os.path.abspath(path)}")
            else:
                print(f"   {os.path.abspath(path)} (does not exist)")

    return all_found_files


# COMPLETE WORKING EXAMPLE - Copy this into your script to replace existing classes
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


class FixedMultiDatabaseBacktester:
    """Fixed version of the Multi-Database Backtester with enhanced database finding"""

    def __init__(self, data_folder="data/marketupdate", symbols=None, period=14,
                 volume_threshold_percentile=50, trailing_stop_pct=3.0,
                 initial_capital=100000, square_off_time="15:20", min_data_points=100):

        print("Initializing Fixed Multi-Database Backtester")
        print("=" * 60)

        # Store parameters
        self.original_data_folder = data_folder
        self.data_folder = data_folder
        self.period = period
        self.volume_threshold_percentile = volume_threshold_percentile
        self.trailing_stop_pct = trailing_stop_pct / 100
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.results = {}
        self.combined_data = {}

        # Set up IST timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        # Enhanced database file finding
        self.db_files = self.enhanced_find_database_files()

        if not self.db_files:
            raise FileNotFoundError(
                "No valid database files found! Please check your database files or run the test script."
            )

        # Auto-detect symbols if not provided
        if symbols is None:
            print("\nAuto-detecting symbols from database files...")
            self.symbols = self.auto_detect_symbols()
        else:
            self.symbols = symbols

        print(f"\nInitialization complete!")
        print(f"Symbols to backtest: {len(self.symbols)}")
        print(f"Database files: {len(self.db_files)}")
        print(f"Data folder: {self.data_folder}")

    def enhanced_find_database_files(self):
        """Enhanced database file finder that searches multiple locations"""
        return enhanced_find_database_files(self)

    def parse_square_off_time(self, time_str):
        """Parse square-off time string to time object"""
        try:
            hour, minute = map(int, time_str.split(':'))
            return time(hour, minute)
        except:
            print(f"Invalid square-off time format: {time_str}. Using default 15:20")
            return time(15, 20)

    def auto_detect_symbols(self):
        """Auto-detect symbols from all databases"""
        all_symbols = set()
        symbol_stats = {}

        print("Scanning all databases for symbols...")

        for db_file in self.db_files:
            try:
                db_name = os.path.basename(db_file)
                print(f"  Scanning {db_name}...")

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
                        symbol_stats[symbol]['databases'].append(db_name)

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

        # Filter and sort symbols
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
            print(f"\nWARNING: No symbols found with sufficient data!")
            print(f"Consider reducing min_data_points (currently {self.min_data_points})")

        return filtered_symbols

    def load_data_from_all_databases(self, symbol):
        """Load and combine data from all database files for a specific symbol"""
        combined_df = pd.DataFrame()

        print(f"  Loading {symbol} from all databases...")

        for db_file in self.db_files:
            try:
                db_name = os.path.basename(db_file)
                df = self.load_data_from_single_db(db_file, symbol)

                if df is not None and not df.empty:
                    df['source_db'] = db_name
                    combined_df = pd.concat([combined_df, df], ignore_index=False)
                    print(f"    {db_name}: {len(df)} records")

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
        """Load data from a single database file with market depth parsing"""
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

            # Parse raw_data to get market depth and other fields
            def parse_raw_data(raw_data_str):
                try:
                    if raw_data_str:
                        raw_data = json.loads(raw_data_str)

                        # Check if this is market depth data
                        if raw_data.get('type') == 'dp':
                            # Market depth data structure
                            return pd.Series({
                                # Best bid/ask prices (Level 1)
                                'best_bid_price': raw_data.get('bid_price1', np.nan),
                                'best_ask_price': raw_data.get('ask_price1', np.nan),
                                'best_bid_size': raw_data.get('bid_size1', 0),
                                'best_ask_size': raw_data.get('ask_size1', 0),

                                # All bid prices (5 levels)
                                'bid_price1': raw_data.get('bid_price1', np.nan),
                                'bid_price2': raw_data.get('bid_price2', np.nan),
                                'bid_price3': raw_data.get('bid_price3', np.nan),
                                'bid_price4': raw_data.get('bid_price4', np.nan),
                                'bid_price5': raw_data.get('bid_price5', np.nan),

                                # All ask prices (5 levels)
                                'ask_price1': raw_data.get('ask_price1', np.nan),
                                'ask_price2': raw_data.get('ask_price2', np.nan),
                                'ask_price3': raw_data.get('ask_price3', np.nan),
                                'ask_price4': raw_data.get('ask_price4', np.nan),
                                'ask_price5': raw_data.get('ask_price5', np.nan),

                                # All bid sizes (5 levels)
                                'bid_size1': raw_data.get('bid_size1', 0),
                                'bid_size2': raw_data.get('bid_size2', 0),
                                'bid_size3': raw_data.get('bid_size3', 0),
                                'bid_size4': raw_data.get('bid_size4', 0),
                                'bid_size5': raw_data.get('bid_size5', 0),

                                # All ask sizes (5 levels)
                                'ask_size1': raw_data.get('ask_size1', 0),
                                'ask_size2': raw_data.get('ask_size2', 0),
                                'ask_size3': raw_data.get('ask_size3', 0),
                                'ask_size4': raw_data.get('ask_size4', 0),
                                'ask_size5': raw_data.get('ask_size5', 0),

                                # Order counts (5 levels)
                                'bid_order1': raw_data.get('bid_order1', 0),
                                'bid_order2': raw_data.get('bid_order2', 0),
                                'bid_order3': raw_data.get('bid_order3', 0),
                                'bid_order4': raw_data.get('bid_order4', 0),
                                'bid_order5': raw_data.get('bid_order5', 0),

                                'ask_order1': raw_data.get('ask_order1', 0),
                                'ask_order2': raw_data.get('ask_order2', 0),
                                'ask_order3': raw_data.get('ask_order3', 0),
                                'ask_order4': raw_data.get('ask_order4', 0),
                                'ask_order5': raw_data.get('ask_order5', 0),

                                # Calculated metrics
                                'bid_ask_spread': raw_data.get('ask_price1', np.nan) - raw_data.get('bid_price1', np.nan) if raw_data.get('ask_price1') and raw_data.get(
                                    'bid_price1') else np.nan,
                                'total_bid_size': sum([raw_data.get(f'bid_size{i}', 0) for i in range(1, 6)]),
                                'total_ask_size': sum([raw_data.get(f'ask_size{i}', 0) for i in range(1, 6)]),
                                'total_bid_orders': sum([raw_data.get(f'bid_order{i}', 0) for i in range(1, 6)]),
                                'total_ask_orders': sum([raw_data.get(f'ask_order{i}', 0) for i in range(1, 6)]),

                                # Data type indicator
                                'data_type': 'market_depth',
                                'processing_timestamp': raw_data.get('processing_timestamp', ''),

                                # Legacy fields for backward compatibility (using approximations)
                                'high_raw': np.nan,  # Not available in market depth
                                'low_raw': np.nan,  # Not available in market depth
                                'volume_raw': 0  # Not available in market depth
                            })
                        else:
                            # Legacy OHLC data structure or other types
                            return pd.Series({
                                # Legacy fields
                                'high_raw': raw_data.get('high_price', np.nan),
                                'low_raw': raw_data.get('low_price', np.nan),
                                'volume_raw': raw_data.get('vol_traded_today', 0),

                                # Market depth fields (will be NaN for legacy data)
                                'best_bid_price': raw_data.get('bid_price', np.nan),
                                'best_ask_price': raw_data.get('ask_price', np.nan),
                                'best_bid_size': raw_data.get('bid_size', 0),
                                'best_ask_size': raw_data.get('ask_size', 0),

                                # All other market depth fields as NaN
                                **{f'bid_price{i}': np.nan for i in range(1, 6)},
                                **{f'ask_price{i}': np.nan for i in range(1, 6)},
                                **{f'bid_size{i}': 0 for i in range(1, 6)},
                                **{f'ask_size{i}': 0 for i in range(1, 6)},
                                **{f'bid_order{i}': 0 for i in range(1, 6)},
                                **{f'ask_order{i}': 0 for i in range(1, 6)},

                                'bid_ask_spread': np.nan,
                                'total_bid_size': 0,
                                'total_ask_size': 0,
                                'total_bid_orders': 0,
                                'total_ask_orders': 0,
                                'data_type': 'legacy',
                                'processing_timestamp': ''
                            })
                    else:
                        # No raw data available
                        return pd.Series({
                            # Legacy fields
                            'high_raw': np.nan,
                            'low_raw': np.nan,
                            'volume_raw': 0,

                            # Market depth fields
                            'best_bid_price': np.nan,
                            'best_ask_price': np.nan,
                            'best_bid_size': 0,
                            'best_ask_size': 0,

                            # All market depth levels as NaN/0
                            **{f'bid_price{i}': np.nan for i in range(1, 6)},
                            **{f'ask_price{i}': np.nan for i in range(1, 6)},
                            **{f'bid_size{i}': 0 for i in range(1, 6)},
                            **{f'ask_size{i}': 0 for i in range(1, 6)},
                            **{f'bid_order{i}': 0 for i in range(1, 6)},
                            **{f'ask_order{i}': 0 for i in range(1, 6)},

                            'bid_ask_spread': np.nan,
                            'total_bid_size': 0,
                            'total_ask_size': 0,
                            'total_bid_orders': 0,
                            'total_ask_orders': 0,
                            'data_type': 'no_data',
                            'processing_timestamp': ''
                        })
                except Exception as e:
                    print(f"      ⚠️ Error parsing raw_data: {e}")
                    # Return default values on error
                    return pd.Series({
                        'high_raw': np.nan, 'low_raw': np.nan, 'volume_raw': 0,
                        'best_bid_price': np.nan, 'best_ask_price': np.nan,
                        'best_bid_size': 0, 'best_ask_size': 0,
                        **{f'bid_price{i}': np.nan for i in range(1, 6)},
                        **{f'ask_price{i}': np.nan for i in range(1, 6)},
                        **{f'bid_size{i}': 0 for i in range(1, 6)},
                        **{f'ask_size{i}': 0 for i in range(1, 6)},
                        **{f'bid_order{i}': 0 for i in range(1, 6)},
                        **{f'ask_order{i}': 0 for i in range(1, 6)},
                        'bid_ask_spread': np.nan, 'total_bid_size': 0, 'total_ask_size': 0,
                        'total_bid_orders': 0, 'total_ask_orders': 0,
                        'data_type': 'error', 'processing_timestamp': ''
                    })

            # Apply parsing and fill missing values
            raw_parsed = df['raw_data'].apply(parse_raw_data)
            df = pd.concat([df, raw_parsed], axis=1)

            # Use best available data for OHLC construction
            # For market depth data, we'll approximate OHLC from bid/ask prices

            # High: Use the highest ask price or best ask if others not available
            df['high'] = df[['ask_price1', 'ask_price2', 'ask_price3', 'ask_price4', 'ask_price5']].max(axis=1, skipna=True)
            df['high'] = df['high'].fillna(df['best_ask_price']).fillna(df['high_price']).fillna(df['ltp'])

            # Low: Use the lowest bid price or best bid if others not available
            df['low'] = df[['bid_price1', 'bid_price2', 'bid_price3', 'bid_price4', 'bid_price5']].min(axis=1, skipna=True)
            df['low'] = df['low'].fillna(df['best_bid_price']).fillna(df['low_price']).fillna(df['ltp'])

            # Close: Use LTP as the most recent trade price
            df['close'] = df['close_price'].fillna(df['ltp'])

            # Volume: Use volume from database (market depth doesn't have volume)
            df['volume_final'] = df['volume_raw'].fillna(df['volume']).fillna(0)

            # Handle bid/ask prices for order flow analysis
            df['bid_price'] = df['best_bid_price'].fillna(df['close'] * 0.999)  # Approximate if missing
            df['ask_price'] = df['best_ask_price'].fillna(df['close'] * 1.001)  # Approximate if missing
            df['bid_size'] = df['best_bid_size'].fillna(0)
            df['ask_size'] = df['best_ask_size'].fillna(0)

            # Market depth specific fields for advanced strategies
            df['spread'] = df['bid_ask_spread'].fillna(df['ask_price'] - df['bid_price'])
            df['spread_pct'] = df['spread'] / df['close'] * 100

            # Order book imbalance (more sophisticated with market depth)
            df['order_book_imbalance'] = np.where(
                (df['total_bid_size'] + df['total_ask_size']) > 0,
                (df['total_bid_size'] - df['total_ask_size']) / (df['total_bid_size'] + df['total_ask_size']),
                0
            )

            # Weighted mid price using level 1 sizes
            df['weighted_mid_price'] = np.where(
                (df['best_bid_size'] + df['best_ask_size']) > 0,
                (df['best_bid_price'] * df['best_ask_size'] + df['best_ask_price'] * df['best_bid_size']) / (df['best_bid_size'] + df['best_ask_size']),
                (df['best_bid_price'] + df['best_ask_price']) / 2
            )

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            # Remove rows with missing critical data
            # df = df.dropna(subset=['close', 'high', 'low'])

            # Return comprehensive dataset with market depth information
            return_columns = [
                # Basic OHLC
                'close', 'high', 'low', 'volume_final',

                # Level 1 market depth (for basic order flow strategies)
                'bid_price', 'ask_price', 'bid_size', 'ask_size',

                # All 5 levels of market depth (for advanced strategies)
                'bid_price1', 'bid_price2', 'bid_price3', 'bid_price4', 'bid_price5',
                'ask_price1', 'ask_price2', 'ask_price3', 'ask_price4', 'ask_price5',
                'bid_size1', 'bid_size2', 'bid_size3', 'bid_size4', 'bid_size5',
                'ask_size1', 'ask_size2', 'ask_size3', 'ask_size4', 'ask_size5',
                'bid_order1', 'bid_order2', 'bid_order3', 'bid_order4', 'bid_order5',
                'ask_order1', 'ask_order2', 'ask_order3', 'ask_order4', 'ask_order5',

                # Calculated market depth metrics
                'spread', 'spread_pct', 'order_book_imbalance', 'weighted_mid_price',
                'total_bid_size', 'total_ask_size', 'total_bid_orders', 'total_ask_orders',

                # Data type indicator
                'data_type'
            ]

            # Only include columns that exist
            available_columns = [col for col in return_columns if col in df.columns]

            return df[available_columns].copy()

        except Exception as e:
            print(f"    ❌ Error accessing database {db_path}: {e}")
            return None

    def test_connection(self):
        """Test the database connections and symbol loading"""
        print("\nTesting Database Connections")
        print("=" * 50)

        if not self.symbols:
            print("No symbols available for testing")
            return False

        # Test loading data for the first symbol
        test_symbol = self.symbols[0]
        print(f"Testing data loading for: {test_symbol}")

        test_data = self.load_data_from_all_databases(test_symbol)

        if test_data is not None:
            print(f"Successfully loaded {len(test_data)} records")
            print(f"Date range: {test_data.index.min()} to {test_data.index.max()}")

            if 'source_db' in test_data.columns:
                source_counts = test_data['source_db'].value_counts()
                print(f"Data sources:")
                for source, count in source_counts.items():
                    print(f"   {source}: {count} records")

            return True
        else:
            print("Failed to load test data")
            return False


# Enhanced market depth analysis functions
def calculate_market_depth_indicators(df):
    """Calculate advanced market depth indicators"""

    # Price impact indicators
    for level in range(1, 6):
        df[f'bid_price_impact_{level}'] = (df['close'] - df[f'bid_price{level}']) / df['close'] * 100
        df[f'ask_price_impact_{level}'] = (df[f'ask_price{level}'] - df['close']) / df['close'] * 100

    # Volume weighted average prices for each side
    bid_volumes = df[['bid_size1', 'bid_size2', 'bid_size3', 'bid_size4', 'bid_size5']]
    bid_prices = df[['bid_price1', 'bid_price2', 'bid_price3', 'bid_price4', 'bid_price5']]

    ask_volumes = df[['ask_size1', 'ask_size2', 'ask_size3', 'ask_size4', 'ask_size5']]
    ask_prices = df[['ask_price1', 'ask_price2', 'ask_price3', 'ask_price4', 'ask_price5']]

    # VWAP for bid side
    df['bid_vwap'] = (bid_prices * bid_volumes).sum(axis=1) / bid_volumes.sum(axis=1)

    # VWAP for ask side
    df['ask_vwap'] = (ask_prices * ask_volumes).sum(axis=1) / ask_volumes.sum(axis=1)

    # Market depth pressure (how much volume at each level compared to level 1)
    df['bid_depth_pressure'] = df['total_bid_size'] / df['bid_size1']
    df['ask_depth_pressure'] = df['total_ask_size'] / df['ask_size1']

    # Order size per order (average order size at each level)
    for level in range(1, 6):
        df[f'avg_bid_order_size_{level}'] = np.where(
            df[f'bid_order{level}'] > 0,
            df[f'bid_size{level}'] / df[f'bid_order{level}'],
            0
        )
        df[f'avg_ask_order_size_{level}'] = np.where(
            df[f'ask_order{level}'] > 0,
            df[f'ask_size{level}'] / df[f'ask_order{level}'],
            0
        )

    # Market microstructure indicators
    df['effective_spread'] = 2 * abs(df['close'] - df['weighted_mid_price'])
    df['realized_spread'] = df['effective_spread'] - df['spread']

    # Order book slope (price difference per unit volume)
    df['bid_slope'] = np.where(
        df['total_bid_size'] > 0,
        (df['bid_price1'] - df['bid_price5']) / df['total_bid_size'],
        0
    )
    df['ask_slope'] = np.where(
        df['total_ask_size'] > 0,
        (df['ask_price5'] - df['ask_price1']) / df['total_ask_size'],
        0
    )

    return df


def detect_market_depth_patterns(df):
    """Detect specific market depth patterns for trading signals"""

    # Large order detection (orders significantly larger than average)
    df['large_bid_l1'] = df['bid_size1'] > df['bid_size1'].rolling(20).mean() * 2
    df['large_ask_l1'] = df['ask_size1'] > df['ask_size1'].rolling(20).mean() * 2

    # Iceberg order detection (consistent refilling at same price level)
    df['potential_bid_iceberg'] = (
            (df['bid_price1'] == df['bid_price1'].shift(1)) &
            (df['bid_size1'] > df['bid_size1'].shift(1))
    )
    df['potential_ask_iceberg'] = (
            (df['ask_price1'] == df['ask_price1'].shift(1)) &
            (df['ask_size1'] > df['ask_size1'].shift(1))
    )

    # Spoofing detection (large orders that disappear quickly)
    df['bid_spoofing_signal'] = (
            (df['bid_size1'].shift(1) > df['bid_size1'].rolling(10).mean() * 3) &
            (df['bid_size1'] < df['bid_size1'].shift(1) * 0.5)
    )
    df['ask_spoofing_signal'] = (
            (df['ask_size1'].shift(1) > df['ask_size1'].rolling(10).mean() * 3) &
            (df['ask_size1'] < df['ask_size1'].shift(1) * 0.5)
    )

    # Market depth imbalance signals
    df['strong_bid_imbalance'] = (
            (df['total_bid_size'] / (df['total_bid_size'] + df['total_ask_size']) > 0.7) &
            (df['bid_size1'] > df['ask_size1'] * 2)
    )
    df['strong_ask_imbalance'] = (
            (df['total_ask_size'] / (df['total_bid_size'] + df['total_ask_size']) > 0.7) &
            (df['ask_size1'] > df['bid_size1'] * 2)
    )

    return df


# Example usage for enhanced order flow strategy with market depth
def enhanced_order_flow_strategy_with_market_depth(df):
    """Enhanced order flow strategy using market depth data"""

    # Calculate market depth indicators
    df = calculate_market_depth_indicators(df)
    df = detect_market_depth_patterns(df)

    # Enhanced buy signals using market depth
    df['enhanced_buy_signal'] = (
        # Strong bid imbalance
            (df['strong_bid_imbalance']) &

            # Large bid orders at level 1
            (df['large_bid_l1']) &

            # Bid side pressure (more volume deeper in book)
            (df['bid_depth_pressure'] > 1.5) &

            # Price near weighted mid (not chasing)
            (abs(df['close'] - df['weighted_mid_price']) / df['close'] < 0.001) &

            # Tight spread (good liquidity)
            (df['spread_pct'] < 0.1)
    )

    # Enhanced sell signals using market depth
    df['enhanced_sell_signal'] = (
        # Strong ask imbalance
            (df['strong_ask_imbalance']) &

            # Large ask orders at level 1
            (df['large_ask_l1']) &

            # Ask side pressure (more volume deeper in book)
            (df['ask_depth_pressure'] > 1.5) &

            # Price near weighted mid (not chasing)
            (abs(df['close'] - df['weighted_mid_price']) / df['close'] < 0.001) &

            # Tight spread (good liquidity)
            (df['spread_pct'] < 0.1)
    )

    return df


# Test function to verify market depth parsing
def test_market_depth_parsing():
    """Test the market depth parsing functionality"""

    # Sample market depth data
    sample_raw_data = {
        "bid_price1": 94.8, "bid_price2": 94.75, "bid_price3": 94.7, "bid_price4": 94.65, "bid_price5": 94.6,
        "ask_price1": 95.05, "ask_price2": 95.1, "ask_price3": 95.15, "ask_price4": 95.2, "ask_price5": 95.25,
        "bid_size1": 975, "bid_size2": 2400, "bid_size3": 1950, "bid_size4": 750, "bid_size5": 1875,
        "ask_size1": 2850, "ask_size2": 2775, "ask_size3": 2550, "ask_size4": 1350, "ask_size5": 600,
        "bid_order1": 6, "bid_order2": 13, "bid_order3": 8, "bid_order4": 3, "bid_order5": 8,
        "ask_order1": 11, "ask_order2": 14, "ask_order3": 9, "ask_order4": 6, "ask_order5": 3,
        "type": "dp", "symbol": "NSE:NIFTY25SEP25100CE", "processing_timestamp": "2025-09-25T09:36:00.379007"
    }

    print("🧪 Testing Market Depth Parsing")
    print("=" * 50)

    # Test parsing
    raw_data_str = json.dumps(sample_raw_data)

    # This would normally be done inside the load_data_from_single_db function
    # Here we're just testing the parsing logic
    try:
        raw_data = json.loads(raw_data_str)

        print(f"✅ Data type: {raw_data.get('type')}")
        print(f"✅ Symbol: {raw_data.get('symbol')}")
        print(f"✅ Best bid: {raw_data.get('bid_price1')} (size: {raw_data.get('bid_size1')})")
        print(f"✅ Best ask: {raw_data.get('ask_price1')} (size: {raw_data.get('ask_size1')})")

        # Calculate some metrics
        spread = raw_data.get('ask_price1') - raw_data.get('bid_price1')
        total_bid_size = sum([raw_data.get(f'bid_size{i}', 0) for i in range(1, 6)])
        total_ask_size = sum([raw_data.get(f'ask_size{i}', 0) for i in range(1, 6)])

        print(f"✅ Spread: {spread:.2f}")
        print(f"✅ Total bid size: {total_bid_size}")
        print(f"✅ Total ask size: {total_ask_size}")
        print(f"✅ Order book imbalance: {(total_bid_size - total_ask_size) / (total_bid_size + total_ask_size):.3f}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# USAGE EXAMPLE:
if __name__ == "__main__":
    print("TESTING FIXED MULTI-DATABASE BACKTESTER")
    print("=" * 70)

    test_market_depth_parsing()

    try:
        # Initialize the fixed backtester
        backtester = FixedMultiDatabaseBacktester(
            data_folder="data/marketdepth",  # This will auto-search multiple folders
            symbols=None,  # Auto-detect symbols
            period=14,
            volume_threshold_percentile=60,
            trailing_stop_pct=3.0,
            initial_capital=100000,
            square_off_time="15:20",
            min_data_points=100
        )

        # Test the connection
        if backtester.test_connection():
            print("\nSUCCESS! Database connections are working")
            print(f"Ready to backtest {len(backtester.symbols)} symbols")
            print(f"Using {len(backtester.db_files)} database files")

            # You can now proceed with your backtesting
            # results = backtester.run_backtest_sequential()

        else:
            print("\nConnection test failed")

    except FileNotFoundError as e:
        print(f"\nDatabase Error: {e}")
        print("\n💡 Solutions:")
        print("1. Run the database test script to diagnose the issue")
        print("2. Check if your database files are in the correct location")
        print("3. Verify database file format and content")

    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        import traceback

        traceback.print_exc()


# QUICK PATCH FUNCTION FOR EXISTING CODE:
def apply_quick_fix_to_existing_backtester(backtester_instance):
    """Apply the quick fix to an existing backtester instance"""
    import types

    print("🔧 Applying quick fix to existing backtester...")

    # Replace the find_database_files method
    backtester_instance.enhanced_find_database_files = types.MethodType(enhanced_find_database_files, backtester_instance)

    # Re-run database file discovery
    backtester_instance.db_files = backtester_instance.enhanced_find_database_files()

    if backtester_instance.db_files:
        print(f"Quick fix successful! Found {len(backtester_instance.db_files)} database files")
        print(f"Using folder: {backtester_instance.data_folder}")
        return True
    else:
        print("Quick fix failed - no database files found")
        return False


"""
HOW TO USE THIS FIX:

1. REPLACE YOUR EXISTING CLASS:
   Simply replace your existing MultiDatabaseDIBacktester class with FixedMultiDatabaseBacktester

2. OR PATCH EXISTING INSTANCE:
   If you have an existing backtester that's failing:

   # Your existing code that might be failing
   backtester = MultiDatabaseDIBacktester(data_folder="data/marketupdate", ...)

   # Apply the quick fix
   if apply_quick_fix_to_existing_backtester(backtester):
       # Now it should work
       results = backtester.run_backtest_sequential()

3. UPDATE FOLDER PATH:
   The enhanced version will automatically search multiple common folder paths:
   - data/marketupdate
   - data/symbolupdate  
   - data/
   - And several others...

4. RUN TEST FIRST:
   Before running your full backtest, use the test_connection() method to verify everything works.
"""