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


class SmoothedHeikinAshiScalpingBacktester:
    def __init__(self, data_folder="data", symbols=None, smoothing_period=5,
                 trailing_stop_pct=1.5, initial_capital=100000,
                 square_off_time="15:20", min_data_points=100):
        """
        Initialize the Smoothed Heikin Ashi Scalping Strategy Backtester

        Parameters:
        - data_folder: Folder containing database files
        - symbols: List of trading symbols (if None, auto-detect)
        - smoothing_period: EMA period for smoothing Heikin Ashi candles
        - trailing_stop_pct: Trailing stop loss percentage
        - initial_capital: Starting capital per symbol
        - square_off_time: Square-off time in HH:MM format
        - min_data_points: Minimum data points required per symbol
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()
        self.smoothing_period = smoothing_period
        self.trailing_stop_pct = trailing_stop_pct / 100
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.results = {}

        # Set up IST timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print(f"Smoothed Heikin Ashi Scalping Strategy Parameters:")
        print(f"  Data folder: {self.data_folder}")
        print(f"  Smoothing Period: {self.smoothing_period}")
        print(f"  Trailing Stop: {trailing_stop_pct}%")
        print(f"  Square-off time: {square_off_time} IST")

        # Auto-detect symbols if not provided
        if symbols is None:
            print("\nAuto-detecting symbols...")
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

        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                query = """
                SELECT symbol, COUNT(*) as count
                FROM market_data 
                GROUP BY symbol
                HAVING COUNT(*) >= ?
                ORDER BY count DESC
                LIMIT 10
                """
                symbols_df = pd.read_sql_query(query, conn, params=(self.min_data_points,))
                conn.close()

                for symbol in symbols_df['symbol']:
                    all_symbols.add(symbol)
            except:
                continue

        return list(all_symbols)

    def load_data(self, symbol):
        """Load data for a symbol from all databases"""
        combined_df = pd.DataFrame()

        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                query = """
                SELECT timestamp, ltp, high_price, low_price, close_price, volume
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

        # Process data
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.set_index('timestamp')

        # Create OHLC
        combined_df['open'] = combined_df['ltp']
        combined_df['high'] = combined_df['high_price'].fillna(combined_df['ltp'])
        combined_df['low'] = combined_df['low_price'].fillna(combined_df['ltp'])
        combined_df['close'] = combined_df['close_price'].fillna(combined_df['ltp'])

        combined_df = combined_df.dropna(subset=['close'])
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        return combined_df[['open', 'high', 'low', 'close', 'volume']]

    def calculate_heikin_ashi(self, df):
        """Calculate Heikin Ashi candles"""
        ha_df = df.copy()

        # Initialize first Heikin Ashi candle
        ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_df['ha_open'] = (df['open'] + df['close']) / 2

        # Calculate subsequent Heikin Ashi candles
        for i in range(1, len(ha_df)):
            ha_df.iloc[i, ha_df.columns.get_loc('ha_open')] = (
                                                                      ha_df.iloc[i - 1]['ha_open'] + ha_df.iloc[i - 1]['ha_close']
                                                              ) / 2

        ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
        ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)

        return ha_df

    def smooth_heikin_ashi(self, df):
        """Apply EMA smoothing to Heikin Ashi candles"""
        df['sha_open'] = df['ha_open'].ewm(span=self.smoothing_period, adjust=False).mean()
        df['sha_close'] = df['ha_close'].ewm(span=self.smoothing_period, adjust=False).mean()
        df['sha_high'] = df['ha_high'].ewm(span=self.smoothing_period, adjust=False).mean()
        df['sha_low'] = df['ha_low'].ewm(span=self.smoothing_period, adjust=False).mean()

        return df

    def generate_signals(self, df):
        """Generate trading signals based on smoothed Heikin Ashi"""
        # Determine candle color
        df['bullish'] = df['sha_close'] > df['sha_open']
        df['bearish'] = df['sha_close'] < df['sha_open']

        # Buy signal: Change from bearish to bullish
        df['buy_signal'] = (df['bullish']) & (df['bearish'].shift(1))

        # Sell signal: Change from bullish to bearish
        df['sell_signal'] = (df['bearish']) & (df['bullish'].shift(1))

        return df

    def is_square_off_time(self, timestamp):
        """Check if it's square-off time"""
        try:
            current_time = timestamp.time()
            return current_time >= self.square_off_time
        except:
            return False

    def backtest_symbol(self, symbol):
        """Backtest strategy for a single symbol"""
        print(f"\n{'=' * 80}")
        print(f"Backtesting: {symbol}")
        print(f"{'=' * 80}")

        # Load data
        df = self.load_data(symbol)
        if df is None or len(df) < self.smoothing_period * 3:
            print(f"Insufficient data for {symbol}")
            return None

        # Calculate Heikin Ashi and smooth
        df = self.calculate_heikin_ashi(df)
        df = self.smooth_heikin_ashi(df)
        df = self.generate_signals(df)

        # Add trading day info
        df['trading_day'] = df.index.date
        df['is_square_off'] = df.index.map(self.is_square_off_time)

        # Initialize variables
        cash = self.initial_capital
        position = 0  # 0: no position, 1: long
        entry_price = 0
        trailing_stop = 0
        trades = []
        entry_time = None
        trade_number = 0

        # Backtest loop
        for i in range(self.smoothing_period * 2, len(df)):
            current_time = df.index[i]
            current_price = df.iloc[i]['close']
            is_square_off = df.iloc[i]['is_square_off']

            # Square off at end of day
            if position == 1 and is_square_off:
                exit_reason = "SQUARE_OFF_3:20PM"
                shares = int(cash / entry_price) if entry_price > 0 else 0

                trade_pnl = shares * (current_price - entry_price)
                trade_return = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

                trade_number += 1
                duration = (current_time - entry_time).total_seconds() / 60

                print(f"\nTrade #{trade_number} - {exit_reason}")
                print(f"  Entry:  {entry_time.strftime('%Y-%m-%d %H:%M:%S')} @ ₹{entry_price:.2f}")
                print(f"  Exit:   {current_time.strftime('%Y-%m-%d %H:%M:%S')} @ ₹{current_price:.2f}")
                print(f"  Duration: {duration:.1f} minutes")
                print(f"  P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
                print(f"  Result: {'✅ PROFIT' if trade_pnl > 0 else '❌ LOSS'}")

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
                    'exit_reason': exit_reason
                })

                position = 0

            # Entry signal
            elif position == 0 and df.iloc[i]['buy_signal'] and not is_square_off:
                position = 1
                entry_price = current_price
                entry_time = current_time
                trailing_stop = entry_price * (1 - self.trailing_stop_pct)

            # Exit signals
            elif position == 1:
                # Update trailing stop
                if current_price > entry_price:
                    new_stop = current_price * (1 - self.trailing_stop_pct)
                    trailing_stop = max(trailing_stop, new_stop)

                exit_signal = False
                exit_reason = ""

                # Trailing stop
                if current_price <= trailing_stop:
                    exit_signal = True
                    exit_reason = "TRAILING_STOP"
                # Opposite signal
                elif df.iloc[i]['sell_signal']:
                    exit_signal = True
                    exit_reason = "OPPOSITE_SIGNAL"

                if exit_signal:
                    shares = int(cash / entry_price) if entry_price > 0 else 0

                    trade_pnl = shares * (current_price - entry_price)
                    trade_return = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

                    trade_number += 1
                    duration = (current_time - entry_time).total_seconds() / 60

                    print(f"\nTrade #{trade_number} - {exit_reason}")
                    print(f"  Entry:  {entry_time.strftime('%Y-%m-%d %H:%M:%S')} @ ₹{entry_price:.2f}")
                    print(f"  Exit:   {current_time.strftime('%Y-%m-%d %H:%M:%S')} @ ₹{current_price:.2f}")
                    print(f"  Duration: {duration:.1f} minutes")
                    print(f"  P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
                    print(f"  Result: {'✅ PROFIT' if trade_pnl > 0 else '❌ LOSS'}")

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
                        'exit_reason': exit_reason
                    })

                    position = 0

        return {
            'symbol': symbol,
            'trades': trades,
            'metrics': self.calculate_metrics(trades)
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
                'avg_duration': 0
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
            'avg_duration': avg_duration
        }

    def run_backtest(self):
        """Run backtest for all symbols"""
        print(f"\n{'=' * 80}")
        print("SMOOTHED HEIKIN ASHI SCALPING STRATEGY BACKTEST")
        print(f"{'=' * 80}")
        print(f"Initial Capital: ₹{self.initial_capital:,.0f}")
        print(f"Smoothing Period: {self.smoothing_period}")
        print(f"Trailing Stop: {self.trailing_stop_pct * 100}%")
        print(f"Square-off Time: {self.square_off_time}")
        print(f"{'=' * 80}")

        for symbol in self.symbols:
            try:
                result = self.backtest_symbol(symbol)
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

        print(f"\n{'=' * 80}")
        print("BACKTEST SUMMARY - ALL SYMBOLS")
        print(f"{'=' * 80}")

        for symbol, result in self.results.items():
            metrics = result['metrics']

            print(f"\n{symbol}:")
            print(f"  Total Trades: {metrics['total_trades']}")
            print(f"  Winning Trades: {metrics['winning_trades']}")
            print(f"  Losing Trades: {metrics['losing_trades']}")
            print(f"  Win Rate: {metrics['win_rate']:.1f}%")
            print(f"  Total P&L: ₹{metrics['total_pnl']:.2f}")
            print(f"  Average P&L per Trade: ₹{metrics['avg_pnl']:.2f}")
            print(f"  Total Return: {metrics['total_return']:.2f}%")
            print(f"  Average Return per Trade: {metrics['avg_return']:.2f}%")
            print(f"  Best Trade: ₹{metrics['best_trade']:.2f}")
            print(f"  Worst Trade: ₹{metrics['worst_trade']:.2f}")
            print(f"  Average Duration: {metrics['avg_duration']:.1f} minutes")

        # Overall statistics
        print(f"\n{'=' * 80}")
        print("OVERALL STATISTICS")
        print(f"{'=' * 80}")

        total_trades = sum(r['metrics']['total_trades'] for r in self.results.values())
        total_pnl = sum(r['metrics']['total_pnl'] for r in self.results.values())
        winning_trades = sum(r['metrics']['winning_trades'] for r in self.results.values())

        print(f"Symbols Tested: {len(self.results)}")
        print(f"Total Trades: {total_trades}")
        print(f"Total P&L: ₹{total_pnl:.2f}")
        print(f"Overall Win Rate: {(winning_trades / total_trades * 100):.1f}%")

        profitable_symbols = sum(1 for r in self.results.values() if r['metrics']['total_pnl'] > 0)
        print(f"Profitable Symbols: {profitable_symbols}/{len(self.results)}")


# Main execution
if __name__ == "__main__":
    backtester = SmoothedHeikinAshiScalpingBacktester(
        data_folder="data/symbolupdate",
        symbols=None,  # Auto-detect
        smoothing_period=500,  # 5-period EMA smoothing
        trailing_stop_pct=1,  # 1.5% trailing stop
        initial_capital=100000,
        square_off_time="15:20"
    )

    backtester.run_backtest()