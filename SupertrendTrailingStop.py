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


class SupertrendTrailingStopBacktester:
    def __init__(self, data_folder="data", symbols=None,
                 supertrend_period=10, supertrend_multiplier=3.0,
                 initial_capital=100000, square_off_time="15:20",
                 last_entry_time="14:30", min_data_points=100,
                 tick_interval=None, max_trades_per_day=5,
                 trailing_stop_pct=3.0):
        """
        Supertrend Trailing Stop Strategy

        Strategy Logic:
        ---------------
        This strategy uses the Supertrend indicator line as a dynamic trailing stop loss.

        ENTRY CONDITIONS (ALL must be true):
        1. Supertrend turns bullish (price crosses above Supertrend line)
        2. Valid Supertrend and ATR values exist
        3. Before last entry time (avoid late entries)
        4. Max trades per day not exceeded

        EXIT CONDITIONS (ANY triggers exit):
        1. Price drops 3% below highest price since entry (trailing stop hit)
        2. 3:20 PM square-off (mandatory end-of-day exit)

        KEY CONCEPT:
        ------------
        This strategy uses a fixed 3% trailing stop loss:
        - Tracks the highest price achieved since entry
        - Exit triggered when price drops 3% below the highest price
        - Locks in profits while allowing trend continuation
        - Provides clear, objective exit points

        Parameters:
        -----------
        - data_folder: Folder containing database files
        - symbols: List of symbols (None = auto-detect)
        - supertrend_period: Period for ATR in Supertrend calculation (default: 10)
        - supertrend_multiplier: Multiplier for ATR in Supertrend (default: 3.0)
        - initial_capital: Starting capital per symbol (default: 100000)
        - square_off_time: Time to square off (HH:MM format, default: "15:20")
        - last_entry_time: Last time to enter new trades (default: "14:30")
        - min_data_points: Minimum data points per symbol (default: 100)
        - tick_interval: Time interval for resampling (e.g., '30s', '1min', None for raw)
        - max_trades_per_day: Maximum trades allowed per day (default: 5)
        - trailing_stop_pct: Trailing stop loss percentage (default: 3.0%)
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.last_entry_time = self.parse_square_off_time(last_entry_time)
        self.min_data_points = min_data_points
        self.tick_interval = tick_interval
        self.max_trades_per_day = max_trades_per_day
        self.trailing_stop_pct = trailing_stop_pct
        self.results = {}
        self.combined_data = {}

        # Set up IST timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print(f"{'='*100}")
        print(f"SUPERTREND TRAILING STOP STRATEGY - Pure Trend Following")
        print(f"{'='*100}")
        print(f"Strategy Parameters:")
        print(f"  Tick Interval: {self.tick_interval if self.tick_interval else 'Raw tick data (no resampling)'}")
        print(f"  Supertrend Period: {self.supertrend_period}, Multiplier: {self.supertrend_multiplier}x")
        print(f"  Trailing Stop: {self.trailing_stop_pct}%")
        print(f"  Max Trades Per Day: {self.max_trades_per_day}")
        print(f"  Last Entry Time: {last_entry_time} IST")
        print(f"  Square-off Time: {square_off_time} IST")
        print(f"  Initial Capital: ₹{self.initial_capital:,}")
        print(f"{'='*100}")
        print(f"\n💡 KEY CONCEPT: {self.trailing_stop_pct}% trailing stop loss")
        print(f"   - Entry: When price crosses above Supertrend")
        print(f"   - Exit: When price drops {self.trailing_stop_pct}% below highest price since entry")
        print(f"   - Trailing stop moves up with price, locking in profits")
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

    def calculate_atr(self, df, period=None):
        """Calculate Average True Range"""
        if period is None:
            period = self.supertrend_period

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
        - When price is above Supertrend line: Bullish trend (Supertrend acts as support)
        - When price is below Supertrend line: Bearish trend (Supertrend acts as resistance)

        In this strategy, the Supertrend line acts as our trailing stop loss.
        """
        # Calculate ATR for Supertrend
        df = self.calculate_atr(df, period=self.supertrend_period)

        # Calculate basic upper and lower bands
        hl_avg = (df['high'] + df['low']) / 2
        df['basic_ub'] = hl_avg + (self.supertrend_multiplier * df['atr'])
        df['basic_lb'] = hl_avg - (self.supertrend_multiplier * df['atr'])

        # Initialize Supertrend columns
        df['supertrend'] = np.nan
        df['supertrend_direction'] = 0  # 1 = bullish, -1 = bearish, 0 = not set

        # Set initial direction based on first candle
        if len(df) > 0:
            first_close = df['close'].iloc[0]
            first_hl_avg = (df['high'].iloc[0] + df['low'].iloc[0]) / 2
            # Start bearish if close is below average, otherwise bullish
            if first_close < first_hl_avg:
                df.loc[df.index[0], 'supertrend_direction'] = -1
                df.loc[df.index[0], 'supertrend'] = df['basic_ub'].iloc[0]
            else:
                df.loc[df.index[0], 'supertrend_direction'] = 1
                df.loc[df.index[0], 'supertrend'] = df['basic_lb'].iloc[0]

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
        Generate trading signals using Supertrend

        BUY SIGNAL: Supertrend turns bullish (price crosses above Supertrend line)
        SELL SIGNAL: Price drops 3% below highest price since entry (trailing stop hit)
        """
        # BUY SIGNAL: Supertrend just turned bullish
        df['buy_signal'] = (
            (df['st_buy_signal']) &                    # Supertrend just turned bullish
            (~df['supertrend'].isna()) &               # Valid Supertrend
            (~df['atr'].isna())                        # Valid ATR
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

    def print_signal_diagnostics(self, df, symbol):
        """Print diagnostic information about signals"""
        print(f"\n📊 SIGNAL DIAGNOSTICS for {symbol}")
        print(f"{'-'*100}")

        # Count various signal conditions
        total_rows = len(df)
        st_buy_signals = df['st_buy_signal'].sum()
        st_sell_signals = df['st_sell_signal'].sum()
        buy_signals = df['buy_signal'].sum()

        # Count direction states
        st_bullish_count = (df['supertrend_direction'] == 1).sum()
        st_bearish_count = (df['supertrend_direction'] == -1).sum()

        print(f"Total Rows: {total_rows}")
        print(f"\nSupertrend:")
        print(f"  Bullish Periods: {st_bullish_count} ({st_bullish_count/total_rows*100:.1f}%)")
        print(f"  Bearish Periods: {st_bearish_count} ({st_bearish_count/total_rows*100:.1f}%)")
        print(f"  Buy Signals (Bearish→Bullish): {st_buy_signals}")
        print(f"  Sell Signals (Bullish→Bearish): {st_sell_signals}")

        print(f"\nCombined Signals:")
        print(f"  ✅ FINAL BUY SIGNALS: {buy_signals}")

        # If no buy signals, show why
        if buy_signals == 0 and st_buy_signals > 0:
            print(f"\n⚠️  No buy signals generated! Let's check why...")
            print(f"\nSample ST Buy Signal instances (first 3):")
            st_buy_rows = df[df['st_buy_signal']].head(3)
            for idx, row in st_buy_rows.iterrows():
                print(f"\n  Time: {idx}")
                print(f"    ST Direction: {row['supertrend_direction']}")
                print(f"    Close: {row['close']:.2f}, ST: {row['supertrend']:.2f}")
                print(f"    ATR: {row['atr']:.2f}")

        print(f"{'-'*100}")

    def backtest_single_symbol(self, symbol):
        """Backtest strategy for a single symbol"""
        print(f"\n{'='*100}")
        print(f"Backtesting: {symbol}")
        print(f"{'='*100}")

        # Load data
        df = self.load_data_from_all_databases(symbol)

        if df is None or len(df) < self.supertrend_period * 3:
            print(f"Insufficient data for {symbol}. Skipping.")
            return None

        # Store combined data
        self.combined_data[symbol] = df.copy()

        # Calculate all indicators
        df = self.calculate_supertrend(df)
        df = self.generate_signals(df)

        # Diagnostic output for signal analysis
        self.print_signal_diagnostics(df, symbol)

        # Add trading day info
        df['trading_day'] = df.index.date
        df['is_square_off'] = df.index.map(self.is_square_off_time)
        df['is_last_entry'] = df.index.map(self.is_last_entry_time)

        # Initialize trading variables
        cash = self.initial_capital
        position = 0  # 0: no position, 1: long
        entry_price = 0
        entry_supertrend = 0  # Store Supertrend value at entry
        highest_price = 0  # Track highest price since entry for trailing stop
        trades = []
        entry_time = None
        trade_number = 0
        daily_trades = {}  # Track trades per day

        # Backtest loop
        for i in range(len(df)):
            current_time = df.index[i]
            current_price = df.iloc[i]['close']
            current_supertrend = df.iloc[i]['supertrend']
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
                trailing_stop_price = highest_price * (1 - self.trailing_stop_pct / 100)

                trade_number += 1
                duration = (current_time - entry_time).total_seconds() / 60

                print(f"\n[Trade #{trade_number}] SQUARE-OFF @ {current_time.strftime('%H:%M:%S')}")
                print(f"  Entry:  {entry_time.strftime('%Y-%m-%d %H:%M:%S')} @ ₹{entry_price:.2f}")
                print(f"  Exit:   {current_time.strftime('%Y-%m-%d %H:%M:%S')} @ ₹{current_price:.2f}")
                print(f"  Highest: ₹{highest_price:.2f}, Trailing Stop: ₹{trailing_stop_price:.2f} (not hit)")
                print(f"  Duration: {duration:.1f} min | P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
                print(f"  {'✅ PROFIT' if trade_pnl > 0 else '❌ LOSS'}")

                trades.append({
                    'trade_num': trade_number,
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'highest_price': highest_price,
                    'trailing_stop_price': trailing_stop_price,
                    'entry_supertrend': entry_supertrend,
                    'exit_supertrend': current_supertrend,
                    'shares': shares,
                    'duration_minutes': duration,
                    'pnl': trade_pnl,
                    'return_pct': trade_return,
                    'exit_reason': 'SQUARE_OFF',
                    'atr': current_atr
                })

                position = 0

            # Entry signal
            elif position == 0 and df.iloc[i]['buy_signal'] and not is_square_off and not is_last_entry:
                # Check max trades per day limit
                if daily_trades[current_day] >= self.max_trades_per_day:
                    continue

                position = 1
                entry_price = current_price
                entry_time = current_time
                entry_supertrend = current_supertrend
                highest_price = current_price  # Initialize highest price at entry
                daily_trades[current_day] += 1

                trailing_stop_price = highest_price * (1 - self.trailing_stop_pct / 100)
                print(f"\n🟢 ENTRY SIGNAL #{daily_trades[current_day]}")
                print(f"  Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Entry Price: ₹{entry_price:.2f}")
                print(f"  Initial Trailing Stop: ₹{trailing_stop_price:.2f} ({self.trailing_stop_pct}% below entry)")
                print(f"  ATR: ₹{current_atr:.2f}")

            # Exit signals (for open positions)
            elif position == 1:
                # Update highest price if current price is higher
                if current_price > highest_price:
                    highest_price = current_price

                # Calculate trailing stop price (3% below highest price)
                trailing_stop_price = highest_price * (1 - self.trailing_stop_pct / 100)

                exit_signal = False
                exit_reason = ""

                # Check if price hit trailing stop (dropped 3% below highest)
                if current_price < trailing_stop_price:
                    exit_signal = True
                    exit_reason = "TRAILING_STOP"

                if exit_signal:
                    shares = int(cash / entry_price) if entry_price > 0 else 0
                    trade_pnl = shares * (current_price - entry_price)
                    trade_return = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

                    trade_number += 1
                    duration = (current_time - entry_time).total_seconds() / 60

                    # Calculate how much profit was locked in
                    max_potential_return = ((highest_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

                    print(f"\n[Trade #{trade_number}] {exit_reason} ({self.trailing_stop_pct}% HIT)")
                    print(f"  Entry:  {entry_time.strftime('%Y-%m-%d %H:%M:%S')} @ ₹{entry_price:.2f}")
                    print(f"  Highest: ₹{highest_price:.2f} (Max potential: {max_potential_return:+.2f}%)")
                    print(f"  Exit:   {current_time.strftime('%Y-%m-%d %H:%M:%S')} @ ₹{current_price:.2f}")
                    print(f"  Trailing Stop: ₹{trailing_stop_price:.2f}")
                    print(f"  Duration: {duration:.1f} min | P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
                    print(f"  {'✅ PROFIT' if trade_pnl > 0 else '❌ LOSS'}")

                    trades.append({
                        'trade_num': trade_number,
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'highest_price': highest_price,
                        'trailing_stop_price': trailing_stop_price,
                        'max_potential_return': max_potential_return,
                        'entry_supertrend': entry_supertrend,
                        'exit_supertrend': current_supertrend,
                        'shares': shares,
                        'duration_minutes': duration,
                        'pnl': trade_pnl,
                        'return_pct': trade_return,
                        'exit_reason': exit_reason,
                        'atr': current_atr
                    })

                    position = 0

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
                'trailing_stop_exits': 0,
                'square_off_exits': 0,
                'avg_st_movement': 0
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
        trailing_stop_exits = sum(1 for t in trades if 'TRAILING' in t['exit_reason'])
        square_off_exits = sum(1 for t in trades if 'SQUARE_OFF' in t['exit_reason'])

        # Calculate average Supertrend movement
        st_movements = [t.get('st_movement', 0) for t in trades if 'st_movement' in t]
        avg_st_movement = sum(st_movements) / len(st_movements) if st_movements else 0

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
            'trailing_stop_exits': trailing_stop_exits,
            'square_off_exits': square_off_exits,
            'avg_st_movement': avg_st_movement
        }

    def run_backtest(self):
        """Run backtest for all symbols"""
        print(f"\n{'='*100}")
        print("STARTING SUPERTREND TRAILING STOP BACKTEST")
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
                'ST Exits': metrics['trailing_stop_exits'],
                'Square-off': metrics['square_off_exits'],
                'Avg ST Move': f"₹{metrics['avg_st_movement']:.2f}"
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
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        print(f"Overall Win Rate: {win_rate:.1f}%")
        print(f"Profitable Symbols: {profitable_symbols}/{len(self.results)}")

        # Export to CSV
        summary_df.to_csv('supertrend_trailing_stop_results.csv', index=False)
        print(f"\n✅ Results exported to: supertrend_trailing_stop_results.csv")


# Main execution
if __name__ == "__main__":
    backtester = SupertrendTrailingStopBacktester(
        data_folder="data/symbolupdate",
        symbols=None,  # Auto-detect
        supertrend_period=10,  # ATR period for Supertrend
        supertrend_multiplier=3.0,  # Multiplier for Supertrend bands
        initial_capital=100000,
        square_off_time="15:20",
        last_entry_time="14:30",
        max_trades_per_day=5,  # Allow more trades since this is a pure trend-following system
        tick_interval='30s'  # Options: None (raw), '5s', '10s', '30s', '1min', '5min'
    )

    backtester.run_backtest()
