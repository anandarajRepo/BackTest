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


class AdvanceDeclineBacktester:
    def __init__(self, data_folder="data", symbols=None,
                 ad_lookback=20, ad_threshold=0.6,
                 mcclellan_fast=19, mcclellan_slow=39,
                 trailing_stop_pct=2.0, initial_capital=100000,
                 square_off_time="15:20", min_data_points=100,
                 candle_timeframe="5T", min_symbols_required=10):
        """
        Initialize the Advance-Decline Market Breadth Strategy Backtester

        The Advance-Decline strategy uses market breadth indicators to identify:
        - Strong bullish momentum (many stocks advancing)
        - Strong bearish momentum (many stocks declining)
        - Market divergences and reversals

        Parameters:
        - data_folder: Folder containing database files (default: "data")
        - symbols: List of trading symbols (if None, auto-detect)
        - ad_lookback: Lookback period for A/D ratio smoothing (default 20)
        - ad_threshold: A/D ratio threshold for signal (default 0.6 = 60% advancing)
        - mcclellan_fast: Fast EMA period for McClellan Oscillator (default 19)
        - mcclellan_slow: Slow EMA period for McClellan Oscillator (default 39)
        - trailing_stop_pct: Trailing stop loss percentage (default 2%)
        - initial_capital: Starting capital per symbol
        - square_off_time: Square-off time HH:MM format (default "15:20")
        - min_data_points: Minimum data points required per symbol
        - candle_timeframe: Candle period - "1T" (1min), "5T" (5min), "15T" (15min)
        - min_symbols_required: Minimum symbols needed for breadth calculation (default 10)
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()
        self.ad_lookback = ad_lookback
        self.ad_threshold = ad_threshold
        self.mcclellan_fast = mcclellan_fast
        self.mcclellan_slow = mcclellan_slow
        self.trailing_stop_pct = trailing_stop_pct / 100
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.candle_timeframe = candle_timeframe
        self.min_symbols_required = min_symbols_required
        self.results = {}

        # Set up IST timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        # Timeframe descriptions
        timeframe_names = {
            "1T": "1 minute",
            "5T": "5 minutes",
            "15T": "15 minutes",
            "30T": "30 minutes",
            "1H": "1 hour"
        }
        self.timeframe_name = timeframe_names.get(candle_timeframe, candle_timeframe)

        print("=" * 80)
        print("ADVANCE-DECLINE MARKET BREADTH STRATEGY BACKTESTER")
        print("=" * 80)
        print(f"Candle Timeframe: {self.timeframe_name}")
        print(f"A/D Lookback: {ad_lookback} periods")
        print(f"A/D Threshold: {ad_threshold * 100}% advancing stocks")
        print(f"McClellan Fast/Slow: {mcclellan_fast}/{mcclellan_slow}")
        print(f"Trailing Stop: {trailing_stop_pct}%")
        print(f"Square-off Time: {square_off_time} IST")
        print(f"Minimum Symbols Required: {min_symbols_required}")
        print("=" * 80)

        # Auto-detect symbols if not provided
        if symbols is None:
            print("\nAuto-detecting symbols...")
            self.symbols = self.auto_detect_symbols()
        else:
            self.symbols = symbols

        print(f"\nSymbols to analyze: {len(self.symbols)}")
        if len(self.symbols) < self.min_symbols_required:
            print(f"⚠️  WARNING: Only {len(self.symbols)} symbols found, but {self.min_symbols_required} recommended")
            print("   Market breadth indicators work best with more symbols")

    def parse_square_off_time(self, time_str):
        """Parse square-off time"""
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
        """Auto-detect symbols with sufficient data"""
        all_symbols = set()
        symbol_stats = {}

        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                query = """
                SELECT symbol, COUNT(*) as record_count
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
                        symbol_stats[symbol] = 0
                    symbol_stats[symbol] += row['record_count']
            except:
                continue

        # Return symbols sorted by data availability
        sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1], reverse=True)
        return [symbol for symbol, count in sorted_symbols]

    def load_data(self, symbol):
        """Load tick data for a symbol"""
        combined_df = pd.DataFrame()

        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                query = """
                SELECT timestamp, ltp, close_price, high_price, low_price, volume, raw_data
                FROM market_data 
                WHERE symbol = ? 
                ORDER BY timestamp
                """
                df = pd.read_sql_query(query, conn, params=(symbol,))
                conn.close()

                if not df.empty:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
            except:
                continue

        if combined_df.empty:
            return None

        # Process tick data
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.set_index('timestamp')
        combined_df['close'] = combined_df['close_price'].fillna(combined_df['ltp'])
        combined_df['high'] = combined_df['high_price'].fillna(combined_df['ltp'])
        combined_df['low'] = combined_df['low_price'].fillna(combined_df['ltp'])
        combined_df = combined_df.dropna(subset=['close'])
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        return combined_df

    def convert_to_candles(self, tick_data):
        """Convert tick data to OHLC candles"""
        ohlc_dict = {
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume': 'sum'
        }

        candles = tick_data.resample(self.candle_timeframe).agg(ohlc_dict)
        candles['open'] = tick_data['close'].resample(self.candle_timeframe).first()
        candles = candles.dropna()
        candles = candles[['open', 'high', 'low', 'close', 'volume']]

        return candles

    def load_all_symbols_data(self):
        """Load and convert data for all symbols"""
        print(f"\nLoading data for {len(self.symbols)} symbols...")
        all_candles = {}

        for symbol in self.symbols:
            try:
                tick_data = self.load_data(symbol)
                if tick_data is not None and len(tick_data) > 50:
                    candles = self.convert_to_candles(tick_data)
                    if len(candles) > max(self.mcclellan_slow, self.ad_lookback):
                        all_candles[symbol] = candles
                        print(f"  ✓ {symbol}: {len(candles)} candles")
            except Exception as e:
                print(f"  ✗ {symbol}: Error - {e}")
                continue

        print(f"\nSuccessfully loaded {len(all_candles)} symbols")
        return all_candles

    def calculate_advance_decline_indicators(self, all_candles_dict):
        """Calculate market breadth indicators from all symbols"""
        print(f"\nCalculating Advance-Decline indicators...")

        # Get all unique timestamps across all symbols
        all_timestamps = set()
        for candles in all_candles_dict.values():
            all_timestamps.update(candles.index)

        all_timestamps = sorted(list(all_timestamps))

        # Create breadth dataframe
        breadth_df = pd.DataFrame(index=all_timestamps)
        breadth_df['advancing'] = 0
        breadth_df['declining'] = 0
        breadth_df['unchanged'] = 0
        breadth_df['total_symbols'] = 0

        # Calculate advancing/declining for each timestamp
        for timestamp in all_timestamps:
            advancing = 0
            declining = 0
            unchanged = 0
            total = 0

            for symbol, candles in all_candles_dict.items():
                if timestamp in candles.index:
                    try:
                        # Get current and previous close
                        current_idx = candles.index.get_loc(timestamp)
                        if current_idx > 0:
                            current_close = candles.iloc[current_idx]['close']
                            prev_close = candles.iloc[current_idx - 1]['close']

                            if current_close > prev_close:
                                advancing += 1
                            elif current_close < prev_close:
                                declining += 1
                            else:
                                unchanged += 1

                            total += 1
                    except:
                        continue

            breadth_df.loc[timestamp, 'advancing'] = advancing
            breadth_df.loc[timestamp, 'declining'] = declining
            breadth_df.loc[timestamp, 'unchanged'] = unchanged
            breadth_df.loc[timestamp, 'total_symbols'] = total

        # Calculate breadth indicators

        # 1. Advance-Decline Ratio
        breadth_df['ad_ratio'] = np.where(
            breadth_df['total_symbols'] > 0,
            breadth_df['advancing'] / breadth_df['total_symbols'],
            0.5
        )

        # 2. Advance-Decline Line (cumulative)
        breadth_df['ad_line'] = (breadth_df['advancing'] - breadth_df['declining']).cumsum()

        # 3. Advance-Decline Spread
        breadth_df['ad_spread'] = breadth_df['advancing'] - breadth_df['declining']

        # 4. McClellan Oscillator (based on A/D spread)
        alpha_fast = 2 / (self.mcclellan_fast + 1)
        alpha_slow = 2 / (self.mcclellan_slow + 1)

        breadth_df['ema_fast'] = breadth_df['ad_spread'].ewm(alpha=alpha_fast, adjust=False).mean()
        breadth_df['ema_slow'] = breadth_df['ad_spread'].ewm(alpha=alpha_slow, adjust=False).mean()
        breadth_df['mcclellan_oscillator'] = breadth_df['ema_fast'] - breadth_df['ema_slow']

        # 5. Smoothed A/D Ratio
        breadth_df['ad_ratio_ma'] = breadth_df['ad_ratio'].rolling(window=self.ad_lookback).mean()

        # 6. A/D Momentum
        breadth_df['ad_momentum'] = breadth_df['ad_ratio'].diff(5)

        print(f"  Calculated breadth indicators for {len(breadth_df)} periods")
        print(f"  Average symbols per period: {breadth_df['total_symbols'].mean():.1f}")

        return breadth_df

    def generate_market_signals(self, breadth_df):
        """Generate buy/sell signals based on advance-decline indicators"""

        # Initialize signals
        breadth_df['market_signal'] = 0  # 0 = neutral, 1 = bullish, -1 = bearish

        # Signal Logic:
        # BULLISH:
        # - A/D Ratio > threshold (e.g., 60% stocks advancing)
        # - McClellan Oscillator > 0
        # - A/D Momentum positive

        bullish_conditions = (
                (breadth_df['ad_ratio'] > self.ad_threshold) &
                (breadth_df['mcclellan_oscillator'] > 0) &
                (breadth_df['ad_momentum'] > 0) &
                (~breadth_df['ad_ratio'].isna())
        )

        # BEARISH:
        # - A/D Ratio < (1 - threshold) (e.g., <40% stocks advancing)
        # - McClellan Oscillator < 0
        # - A/D Momentum negative

        bearish_conditions = (
                (breadth_df['ad_ratio'] < (1 - self.ad_threshold)) &
                (breadth_df['mcclellan_oscillator'] < 0) &
                (breadth_df['ad_momentum'] < 0) &
                (~breadth_df['ad_ratio'].isna())
        )

        breadth_df.loc[bullish_conditions, 'market_signal'] = 1
        breadth_df.loc[bearish_conditions, 'market_signal'] = -1

        # Signal changes (for entry/exit)
        breadth_df['signal_change'] = breadth_df['market_signal'].diff()
        breadth_df['buy_signal'] = breadth_df['signal_change'] == 1
        breadth_df['sell_signal'] = breadth_df['signal_change'] == -1

        bullish_count = (breadth_df['market_signal'] == 1).sum()
        bearish_count = (breadth_df['market_signal'] == -1).sum()

        print(f"\nMarket signals generated:")
        print(f"  Bullish periods: {bullish_count} ({bullish_count / len(breadth_df) * 100:.1f}%)")
        print(f"  Bearish periods: {bearish_count} ({bearish_count / len(breadth_df) * 100:.1f}%)")
        print(f"  Buy signals: {breadth_df['buy_signal'].sum()}")
        print(f"  Sell signals: {breadth_df['sell_signal'].sum()}")

        return breadth_df

    def is_square_off_time(self, timestamp):
        """Check if square-off time"""
        try:
            return timestamp.time() >= self.square_off_time
        except:
            return False

    def backtest_symbol_with_breadth(self, symbol, symbol_candles, breadth_df):
        """Backtest a single symbol using market breadth signals"""
        print(f"\n{'=' * 80}")
        print(f"BACKTESTING: {symbol}")
        print(f"{'=' * 80}")

        # Align symbol candles with breadth data
        aligned_candles = symbol_candles.copy()

        # Add breadth indicators to symbol candles
        for col in ['ad_ratio', 'ad_line', 'mcclellan_oscillator', 'market_signal', 'buy_signal', 'sell_signal']:
            aligned_candles[col] = breadth_df[col].reindex(aligned_candles.index, method='ffill')

        # Add trading day and square-off info
        aligned_candles['trading_day'] = aligned_candles.index.date
        aligned_candles['is_square_off'] = aligned_candles.index.map(self.is_square_off_time)

        # Initialize trading variables
        cash = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        trailing_stop = 0
        trades = []
        entry_day = None

        # Backtest loop
        for i in range(len(aligned_candles)):
            current_time = aligned_candles.index[i]
            current_candle = aligned_candles.iloc[i]
            current_price = current_candle['close']
            current_day = current_candle['trading_day']
            is_square_off = current_candle['is_square_off']

            # Skip if no breadth data
            if pd.isna(current_candle['market_signal']):
                continue

            # Square-off at 3:20 PM
            if position != 0 and is_square_off:
                shares = trades[-1]['shares']

                if position == 1:  # Long
                    proceeds = shares * current_price
                    cash += proceeds
                    pnl = shares * (current_price - entry_price)
                else:  # Short
                    cost = shares * current_price
                    cash -= cost
                    pnl = shares * (entry_price - current_price)

                pnl_pct = (pnl / (shares * entry_price)) * 100

                trades[-1].update({
                    'exit_time': current_time,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'SQUARE_OFF_3:20PM',
                    'exit_ad_ratio': current_candle['ad_ratio'],
                    'exit_mcclellan': current_candle['mcclellan_oscillator']
                })

                self.print_trade(trades[-1])
                position = 0
                entry_day = None

            # Entry signals
            elif position == 0 and not is_square_off:
                # Long entry on bullish breadth
                if current_candle['buy_signal']:
                    shares = int(cash / current_price)
                    if shares > 0:
                        position = 1
                        entry_price = current_price
                        entry_day = current_day
                        trailing_stop = entry_price * (1 - self.trailing_stop_pct)
                        cost = shares * current_price
                        cash -= cost

                        trades.append({
                            'direction': 'LONG',
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'shares': shares,
                            'entry_ad_ratio': current_candle['ad_ratio'],
                            'entry_mcclellan': current_candle['mcclellan_oscillator'],
                            'entry_ad_line': current_candle['ad_line']
                        })

                        print(f"\n🟢 LONG ENTRY at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"   Price: ₹{entry_price:.2f} | Shares: {shares}")
                        print(f"   A/D Ratio: {current_candle['ad_ratio']:.2%}")
                        print(f"   McClellan: {current_candle['mcclellan_oscillator']:.2f}")

                # Short entry on bearish breadth
                elif current_candle['sell_signal']:
                    shares = int(cash / current_price)
                    if shares > 0:
                        position = -1
                        entry_price = current_price
                        entry_day = current_day
                        trailing_stop = entry_price * (1 + self.trailing_stop_pct)
                        proceeds = shares * current_price
                        cash += proceeds

                        trades.append({
                            'direction': 'SHORT',
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'shares': shares,
                            'entry_ad_ratio': current_candle['ad_ratio'],
                            'entry_mcclellan': current_candle['mcclellan_oscillator'],
                            'entry_ad_line': current_candle['ad_line']
                        })

                        print(f"\n🔴 SHORT ENTRY at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"   Price: ₹{entry_price:.2f} | Shares: {shares}")
                        print(f"   A/D Ratio: {current_candle['ad_ratio']:.2%}")
                        print(f"   McClellan: {current_candle['mcclellan_oscillator']:.2f}")

            # Exit management
            elif position != 0 and entry_day == current_day and not is_square_off:
                exit_triggered = False
                exit_reason = ""

                if position == 1:  # Long position
                    # Update trailing stop
                    if current_price > entry_price:
                        new_stop = current_price * (1 - self.trailing_stop_pct)
                        trailing_stop = max(trailing_stop, new_stop)

                    # Check exits
                    if current_price <= trailing_stop:
                        exit_triggered = True
                        exit_reason = "TRAILING_STOP"
                    elif current_candle['sell_signal']:
                        exit_triggered = True
                        exit_reason = "BEARISH_BREADTH"

                elif position == -1:  # Short position
                    # Update trailing stop
                    if current_price < entry_price:
                        new_stop = current_price * (1 + self.trailing_stop_pct)
                        trailing_stop = min(trailing_stop, new_stop)

                    # Check exits
                    if current_price >= trailing_stop:
                        exit_triggered = True
                        exit_reason = "TRAILING_STOP"
                    elif current_candle['buy_signal']:
                        exit_triggered = True
                        exit_reason = "BULLISH_BREADTH"

                if exit_triggered:
                    shares = trades[-1]['shares']

                    if position == 1:
                        proceeds = shares * current_price
                        cash += proceeds
                        pnl = shares * (current_price - entry_price)
                    else:
                        cost = shares * current_price
                        cash -= cost
                        pnl = shares * (entry_price - current_price)

                    pnl_pct = (pnl / (shares * entry_price)) * 100

                    trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'exit_ad_ratio': current_candle['ad_ratio'],
                        'exit_mcclellan': current_candle['mcclellan_oscillator']
                    })

                    self.print_trade(trades[-1])
                    position = 0
                    entry_day = None

        # Summary
        completed_trades = [t for t in trades if 'exit_time' in t]
        self.print_summary(symbol, completed_trades, cash, aligned_candles)

        return {
            'symbol': symbol,
            'trades': completed_trades,
            'final_capital': cash,
            'candles': aligned_candles
        }

    def print_trade(self, trade):
        """Print trade details"""
        duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60

        print(f"\n📊 TRADE CLOSED")
        print(f"   Direction: {trade['direction']}")
        print(f"   Entry:  {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')} @ ₹{trade['entry_price']:.2f}")
        print(f"           A/D Ratio: {trade['entry_ad_ratio']:.2%} | McClellan: {trade['entry_mcclellan']:.2f}")
        print(f"   Exit:   {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')} @ ₹{trade['exit_price']:.2f}")
        print(f"           A/D Ratio: {trade['exit_ad_ratio']:.2%} | McClellan: {trade['exit_mcclellan']:.2f}")
        print(f"   Duration: {duration:.1f} minutes")
        print(f"   P&L: ₹{trade['pnl']:.2f} ({trade['pnl_pct']:+.2f}%)")
        print(f"   Exit Reason: {trade['exit_reason']}")

        if trade['pnl'] > 0:
            print(f"   Result: ✅ PROFIT")
        else:
            print(f"   Result: ❌ LOSS")

    def print_summary(self, symbol, trades, final_cash, candles):
        """Print summary for a symbol"""
        if not trades:
            print(f"\n⚠️  No completed trades for {symbol}")
            return

        total_pnl = sum(t['pnl'] for t in trades)
        total_return = ((final_cash - self.initial_capital) / self.initial_capital) * 100
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        win_rate = (len(winning_trades) / len(trades)) * 100

        long_trades = [t for t in trades if t['direction'] == 'LONG']
        short_trades = [t for t in trades if t['direction'] == 'SHORT']

        print(f"\n{'=' * 80}")
        print(f"SUMMARY: {symbol}")
        print(f"{'=' * 80}")
        print(f"Strategy:           Advance-Decline Market Breadth")
        print(f"Candle Timeframe:   {self.timeframe_name}")
        print(f"Total Candles:      {len(candles)}")
        print(f"")
        print(f"Initial Capital:    ₹{self.initial_capital:,.2f}")
        print(f"Final Capital:      ₹{final_cash:,.2f}")
        print(f"Total P&L:          ₹{total_pnl:,.2f}")
        print(f"Total Return:       {total_return:+.2f}%")
        print(f"")
        print(f"Total Trades:       {len(trades)}")
        print(f"Long Trades:        {len(long_trades)}")
        print(f"Short Trades:       {len(short_trades)}")
        print(f"Winning Trades:     {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losing Trades:      {len(losing_trades)} ({100 - win_rate:.1f}%)")
        print(f"{'=' * 80}")

    def run_backtest(self):
        """Run backtest for all symbols"""
        print(f"\n🚀 Starting Advance-Decline Strategy Backtest")
        print(f"Analyzing {len(self.symbols)} symbols for market breadth\n")

        # Load all symbols data
        all_candles = self.load_all_symbols_data()

        if len(all_candles) < self.min_symbols_required:
            print(f"\n❌ ERROR: Only {len(all_candles)} symbols loaded")
            print(f"   Minimum {self.min_symbols_required} symbols required for breadth analysis")
            return {}

        # Calculate market breadth indicators
        breadth_df = self.calculate_advance_decline_indicators(all_candles)
        breadth_df = self.generate_market_signals(breadth_df)

        # Backtest each symbol using breadth signals
        print(f"\n{'=' * 80}")
        print("BACKTESTING INDIVIDUAL SYMBOLS WITH MARKET BREADTH SIGNALS")
        print(f"{'=' * 80}")

        all_results = []

        for symbol, candles in all_candles.items():
            try:
                result = self.backtest_symbol_with_breadth(symbol, candles, breadth_df)
                if result and result['trades']:
                    all_results.append(result)
                    self.results[symbol] = result
            except Exception as e:
                print(f"❌ Error with {symbol}: {e}")
                import traceback
                traceback.print_exc()

        if all_results:
            self.print_overall_summary(all_results, breadth_df)

        return all_results

    def print_overall_summary(self, results, breadth_df):
        """Print overall strategy performance"""
        print(f"\n{'=' * 80}")
        print(f"OVERALL ADVANCE-DECLINE STRATEGY PERFORMANCE")
        print(f"{'=' * 80}")
        print(f"Strategy:              Market Breadth (Advance-Decline)")
        print(f"Candle Timeframe:      {self.timeframe_name}")
        print(f"Symbols Analyzed:      {len(self.symbols)}")
        print(f"Symbols Traded:        {len(results)}")
        print(f"")

        total_trades = sum(len(r['trades']) for r in results)
        total_pnl = sum(r['final_capital'] - self.initial_capital for r in results)
        profitable_symbols = sum(1 for r in results if r['final_capital'] > self.initial_capital)

        # Aggregate all trades
        all_trades = []
        for r in results:
            all_trades.extend(r['trades'])

        winning_trades = [t for t in all_trades if t['pnl'] > 0]
        overall_win_rate = (len(winning_trades) / len(all_trades) * 100) if all_trades else 0

        long_trades = [t for t in all_trades if t['direction'] == 'LONG']
        short_trades = [t for t in all_trades if t['direction'] == 'SHORT']

        print(f"Total Trades:          {total_trades}")
        print(f"Long Trades:           {len(long_trades)} ({len(long_trades) / total_trades * 100:.1f}%)")
        print(f"Short Trades:          {len(short_trades)} ({len(short_trades) / total_trades * 100:.1f}%)")
        print(f"Overall Win Rate:      {overall_win_rate:.1f}%")
        print(f"Profitable Symbols:    {profitable_symbols} ({profitable_symbols / len(results) * 100:.1f}%)")
        print(f"Total P&L:             ₹{total_pnl:,.2f}")
        print(f"Average Return/Symbol: {(total_pnl / (self.initial_capital * len(results))) * 100:+.2f}%")
        print(f"{'=' * 80}")

        # Market breadth statistics
        print(f"\nMARKET BREADTH STATISTICS:")
        print(f"Average A/D Ratio:     {breadth_df['ad_ratio'].mean():.2%}")
        print(f"Average Advancing:     {breadth_df['advancing'].mean():.1f} stocks")
        print(f"Average Declining:     {breadth_df['declining'].mean():.1f} stocks")
        print(f"McClellan Range:       [{breadth_df['mcclellan_oscillator'].min():.2f}, {breadth_df['mcclellan_oscillator'].max():.2f}]")

        # Best and worst performers
        if results:
            best = max(results, key=lambda x: x['final_capital'])
            worst = min(results, key=lambda x: x['final_capital'])

            print(f"\n🏆 Best Performer:  {best['symbol']}")
            print(f"   Final Capital: ₹{best['final_capital']:,.2f}")
            print(f"   Return: {((best['final_capital'] - self.initial_capital) / self.initial_capital) * 100:+.2f}%")
            print(f"   Trades: {len(best['trades'])}")

            print(f"\n📉 Worst Performer: {worst['symbol']}")
            print(f"   Final Capital: ₹{worst['final_capital']:,.2f}")
            print(f"   Return: {((worst['final_capital'] - self.initial_capital) / self.initial_capital) * 100:+.2f}%")
            print(f"   Trades: {len(worst['trades'])}")

        # Direction analysis
        if long_trades:
            long_wins = len([t for t in long_trades if t['pnl'] > 0])
            print(f"\nLONG TRADE ANALYSIS:")
            print(f"  Total: {len(long_trades)} trades")
            print(f"  Win Rate: {(long_wins / len(long_trades) * 100):.1f}%")
            print(f"  Total P&L: ₹{sum(t['pnl'] for t in long_trades):,.2f}")

        if short_trades:
            short_wins = len([t for t in short_trades if t['pnl'] > 0])
            print(f"\nSHORT TRADE ANALYSIS:")
            print(f"  Total: {len(short_trades)} trades")
            print(f"  Win Rate: {(short_wins / len(short_trades) * 100):.1f}%")
            print(f"  Total P&L: ₹{sum(t['pnl'] for t in short_trades):,.2f}")


# Main execution
if __name__ == "__main__":
    """
    ADVANCE-DECLINE MARKET BREADTH STRATEGY

    This strategy uses market breadth indicators to identify strong trends:

    KEY INDICATORS:
    1. Advance-Decline Ratio: % of stocks advancing vs total
    2. Advance-Decline Line: Cumulative breadth momentum
    3. McClellan Oscillator: Fast vs Slow EMA of A/D spread
    4. A/D Momentum: Rate of change in breadth

    ENTRY SIGNALS:
    - LONG: >60% stocks advancing + McClellan > 0 + positive momentum
    - SHORT: <40% stocks advancing + McClellan < 0 + negative momentum

    EXIT SIGNALS:
    - Trailing stop (2%)
    - Opposite breadth signal
    - 3:20 PM square-off

    ADVANTAGES:
    - Confirms market-wide trends
    - Reduces false signals in choppy markets
    - Works well for index trading or sector leaders
    - Early warning of trend reversals
    """

    # Initialize backtester
    backtester = AdvanceDeclineBacktester(
        data_folder="data/symbolupdate",  # Update with your path
        symbols=None,  # Auto-detect symbols
        ad_lookback=20,  # 20-period smoothing for A/D ratio
        ad_threshold=0.6,  # 60% advancing stocks threshold
        mcclellan_fast=19,  # Fast EMA for McClellan
        mcclellan_slow=39,  # Slow EMA for McClellan
        trailing_stop_pct=2.0,  # 2% trailing stop
        initial_capital=100000,
        square_off_time="15:20",
        candle_timeframe="5T",  # 5-minute candles
        min_symbols_required=10  # Need at least 10 symbols for breadth
    )

    # Run backtest
    results = backtester.run_backtest()

    if results:
        print(f"\n✅ Backtest Complete!")
        print(f"Tested Advance-Decline strategy on {len(results)} symbols")
        print(f"Timeframe: {backtester.timeframe_name}")
    else:
        print(f"\n⚠️  No successful trades executed")
        print("Possible reasons:")
        print("1. Insufficient symbols for breadth calculation")
        print("2. No strong breadth signals in the data period")
        print("3. All signals filtered by tight conditions")

    print(f"\n💡 OPTIMIZATION TIPS:")
    print(f"1. Adjust 'ad_threshold' (currently {backtester.ad_threshold:.0%}) for more/fewer signals")
    print(f"2. Try different timeframes: 1T for scalping, 15T for swing")
    print(f"3. Include more symbols for better breadth representation")
    print(f"4. Combine with sector rotation analysis")
    print(f"5. Add volume confirmation for stronger signals")