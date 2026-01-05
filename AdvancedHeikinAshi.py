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


class AdvancedHeikinAshiBacktester:
    def __init__(self, data_folder="data", symbols=None,
                 ha_smoothing=3, adx_period=14, adx_threshold=25,
                 volume_percentile=60, atr_period=14, atr_multiplier=2.0,
                 breakeven_profit_pct=1.0, consecutive_candles=2,
                 initial_capital=100000, square_off_time="15:20",
                 min_data_points=100, tick_interval=None):
        """
        Advanced Heikin Ashi Strategy with Multi-Indicator Confirmation

        Strategy Logic:
        ---------------
        ENTRY CONDITIONS (ALL must be true):
        1. Heikin Ashi turns bullish (close > open)
        2. Multiple consecutive bullish HA candles (momentum)
        3. ADX > threshold (trending market, not choppy)
        4. Volume > percentile threshold (good liquidity)
        5. Before square-off time

        EXIT CONDITIONS (ANY triggers exit):
        1. ATR-based dynamic trailing stop (adapts to volatility)
        2. Breakeven stop (after 1% profit, stop moves to entry)
        3. Heikin Ashi turns bearish
        4. 3:20 PM square-off (mandatory)

        Parameters:
        -----------
        - data_folder: Folder containing database files
        - symbols: List of symbols (None = auto-detect)
        - ha_smoothing: EMA period for Heikin Ashi smoothing
        - adx_period: Period for ADX calculation
        - adx_threshold: Minimum ADX for trend confirmation
        - volume_percentile: Volume percentile threshold
        - atr_period: Period for ATR calculation
        - atr_multiplier: Multiplier for ATR trailing stop
        - breakeven_profit_pct: Profit % to trigger breakeven stop
        - consecutive_candles: Number of consecutive bullish candles required
        - initial_capital: Starting capital per symbol
        - square_off_time: Time to square off (HH:MM format)
        - min_data_points: Minimum data points per symbol
        - tick_interval: Time interval for resampling tick data (e.g., '5s', '10s', '30s', '1min', '5min')
                        None = use raw tick data without resampling
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()
        self.ha_smoothing = ha_smoothing
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.volume_percentile = volume_percentile
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.breakeven_profit_pct = breakeven_profit_pct / 100
        self.consecutive_candles = consecutive_candles
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.tick_interval = tick_interval
        self.results = {}
        self.combined_data = {}

        # Set up IST timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print(f"{'='*100}")
        print(f"ADVANCED HEIKIN ASHI STRATEGY - Multi-Indicator Confirmation")
        print(f"{'='*100}")
        print(f"Strategy Parameters:")
        print(f"  Tick Interval: {self.tick_interval if self.tick_interval else 'Raw tick data (no resampling)'}")
        print(f"  HA Smoothing Period: {self.ha_smoothing}")
        print(f"  ADX Period: {self.adx_period}, Threshold: {self.adx_threshold}")
        print(f"  Volume Percentile: {self.volume_percentile}%")
        print(f"  ATR Period: {self.atr_period}, Multiplier: {self.atr_multiplier}x")
        print(f"  Breakeven Trigger: {breakeven_profit_pct}%")
        print(f"  Consecutive Candles: {self.consecutive_candles}")
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
        """
        Resample tick data to specified time interval

        Parameters:
        -----------
        df: DataFrame with tick data (must have timestamp index and OHLCV columns)
        interval: Time interval string (e.g., '5s', '10s', '30s', '1min', '5min')

        Returns:
        --------
        DataFrame resampled to specified interval with proper OHLCV aggregation
        """
        if df is None or df.empty:
            return df

        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        # Resample using standard OHLCV aggregation
        resampled = df.resample(interval).agg({
            'open': 'first',    # First price in interval
            'high': 'max',      # Highest price in interval
            'low': 'min',       # Lowest price in interval
            'close': 'last',    # Last price in interval
            'volume': 'sum'     # Total volume in interval
        })

        # Remove rows with no data (NaN close prices)
        resampled = resampled.dropna(subset=['close'])

        # Forward fill any remaining NaN values in open/high/low
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

        return ha_df

    def calculate_adx(self, df):
        """Calculate ADX for trend strength"""
        # Calculate True Range
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

        # Smooth using Wilder's smoothing
        alpha = 1 / self.adx_period
        df['tr_smooth'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
        df['dm_plus_smooth'] = df['dm_plus'].ewm(alpha=alpha, adjust=False).mean()
        df['dm_minus_smooth'] = df['dm_minus'].ewm(alpha=alpha, adjust=False).mean()

        # Calculate DI
        df['di_plus'] = 100 * (df['dm_plus_smooth'] / df['tr_smooth'])
        df['di_minus'] = 100 * (df['dm_minus_smooth'] / df['tr_smooth'])

        # Calculate ADX
        df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
        df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()

        return df

    def calculate_atr(self, df):
        """Calculate Average True Range for dynamic stops"""
        if 'tr' not in df.columns:
            df['prev_close'] = df['close'].shift(1)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['prev_close'])
            df['tr3'] = abs(df['low'] - df['prev_close'])
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Calculate ATR using EMA
        df['atr'] = df['tr'].ewm(span=self.atr_period, adjust=False).mean()

        return df

    def identify_good_volume(self, df):
        """Identify periods with good volume"""
        volume_threshold = df['volume'].quantile(self.volume_percentile / 100)
        df['good_volume'] = df['volume'] > volume_threshold
        return df

    def generate_signals(self, df):
        """Generate trading signals with multiple confirmations"""
        # Count consecutive bullish candles
        df['bullish_streak'] = 0
        streak = 0
        for i in range(len(df)):
            if df.iloc[i]['ha_bullish']:
                streak += 1
            else:
                streak = 0
            df.iloc[i, df.columns.get_loc('bullish_streak')] = streak

        # BUY SIGNAL: All conditions must be true
        df['buy_signal'] = (
            (df['ha_bullish']) &                                    # Bullish HA candle
            (df['bullish_streak'] >= self.consecutive_candles) &   # Momentum (consecutive candles)
            (df['adx'] > self.adx_threshold) &                     # Trending market
            (df['good_volume']) &                                   # Good volume
            (~df['adx'].isna()) &                                  # Valid ADX
            (~df['atr'].isna())                                    # Valid ATR
        )

        # SELL SIGNAL: Heikin Ashi turns bearish
        df['sell_signal'] = (
            (df['ha_bearish']) &
            (df['ha_bullish'].shift(1))  # Was bullish, now bearish
        )

        return df

    def is_square_off_time(self, timestamp):
        """Check if it's square-off time"""
        try:
            current_time = timestamp.time()
            return current_time >= self.square_off_time
        except:
            return False

    def backtest_single_symbol(self, symbol):
        """Backtest strategy for a single symbol"""
        print(f"\n{'='*100}")
        print(f"Backtesting: {symbol}")
        print(f"{'='*100}")

        # Load data
        df = self.load_data_from_all_databases(symbol)

        if df is None or len(df) < max(self.adx_period, self.atr_period, self.ha_smoothing) * 3:
            print(f"Insufficient data for {symbol}. Skipping.")
            return None

        # Store combined data
        self.combined_data[symbol] = df.copy()

        # Calculate all indicators
        df = self.calculate_heikin_ashi(df)
        df = self.calculate_adx(df)
        df = self.calculate_atr(df)
        df = self.identify_good_volume(df)
        df = self.generate_signals(df)

        # Add trading day info
        df['trading_day'] = df.index.date
        df['is_square_off'] = df.index.map(self.is_square_off_time)

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

        # Backtest loop
        for i in range(len(df)):
            current_time = df.index[i]
            current_price = df.iloc[i]['close']
            current_atr = df.iloc[i]['atr']
            is_square_off = df.iloc[i]['is_square_off']

            # Square off at end of day
            if position == 1 and is_square_off:
                shares = int(cash / entry_price) if entry_price > 0 else 0
                trade_pnl = shares * (current_price - entry_price)
                trade_return = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

                trade_number += 1
                duration = (current_time - entry_time).total_seconds() / 60

                print(f"\n[Trade #{trade_number}] SQUARE-OFF @ 3:20 PM")
                print(f"  Entry:  {entry_time.strftime('%H:%M:%S')} @ Rs.{entry_price:.2f}")
                print(f"  Exit:   {current_time.strftime('%H:%M:%S')} @ Rs.{current_price:.2f}")
                print(f"  Duration: {duration:.1f} min | P&L: Rs.{trade_pnl:.2f} ({trade_return:+.2f}%)")
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
                    'adx': df.iloc[i]['adx'],
                    'atr': current_atr,
                    'breakeven_triggered': breakeven_triggered
                })

                position = 0
                breakeven_triggered = False

            # Entry signal
            elif position == 0 and df.iloc[i]['buy_signal'] and not is_square_off:
                position = 1
                entry_price = current_price
                entry_time = current_time
                entry_atr = current_atr
                # ATR-based trailing stop (more dynamic than fixed %)
                trailing_stop = entry_price - (entry_atr * self.atr_multiplier)
                breakeven_triggered = False

            # Exit signals
            elif position == 1:
                # Update trailing stop with ATR
                if current_price > entry_price:
                    new_stop = current_price - (current_atr * self.atr_multiplier)

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
                    exit_reason = "TRAILING_STOP_ATR" if not breakeven_triggered else "BREAKEVEN_STOP"
                elif df.iloc[i]['sell_signal']:
                    exit_signal = True
                    exit_reason = "HA_BEARISH"

                if exit_signal:
                    shares = int(cash / entry_price) if entry_price > 0 else 0
                    trade_pnl = shares * (current_price - entry_price)
                    trade_return = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

                    trade_number += 1
                    duration = (current_time - entry_time).total_seconds() / 60

                    print(f"\n[Trade #{trade_number}] {exit_reason}")
                    print(f"  Entry:  {entry_time.strftime('%H:%M:%S')} @ ₹{entry_price:.2f}")
                    print(f"  Exit:   {current_time.strftime('%H:%M:%S')} @ ₹{current_price:.2f}")
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
                        'adx': df.iloc[i]['adx'],
                        'atr': current_atr,
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
            'ha_bearish_exits': ha_bearish_exits,
            'square_off_exits': square_off_exits
        }

    def run_backtest(self):
        """Run backtest for all symbols"""
        print(f"\n{'='*100}")
        print("STARTING ADVANCED HEIKIN ASHI BACKTEST")
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
                'Breakeven Exits': metrics['breakeven_exits'],
                'Trailing Exits': metrics['trailing_stop_exits']
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
        summary_df.to_csv('advanced_heikin_ashi_results.csv', index=False)
        print(f"\n✅ Results exported to: advanced_heikin_ashi_results.csv")


# Main execution
if __name__ == "__main__":
    backtester = AdvancedHeikinAshiBacktester(
        data_folder="data/symbolupdate",
        symbols=None,  # Auto-detect
        ha_smoothing=3,  # Light smoothing for responsiveness
        adx_period=14,  # Standard ADX period
        adx_threshold=25,  # Trending market filter
        volume_percentile=60,  # Above average volume
        atr_period=14,  # Standard ATR period
        atr_multiplier=2.0,  # 2x ATR for trailing stop
        breakeven_profit_pct=1.0,  # Move to breakeven at 1% profit
        consecutive_candles=2,  # Require 2 consecutive bullish candles
        initial_capital=100000,
        square_off_time="15:20",
        tick_interval='30s'  # Options: None (raw ticks), '5s', '10s', '30s', '1min', '5min', etc.
    )

    backtester.run_backtest()
