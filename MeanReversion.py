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


class SimpleMeanReversionBacktester:
    def __init__(self, data_folder="data", symbols=None, lookback_period=20,
                 entry_threshold=2.0, exit_threshold=0.5, stop_loss_pct=1.0,
                 initial_capital=100000, square_off_time="15:20", min_data_points=100):
        """
        Initialize Simple Mean Reversion Strategy Backtester

        Parameters:
        - lookback_period: Period for calculating moving average and standard deviation
        - entry_threshold: Number of standard deviations from mean for entry
        - exit_threshold: Number of standard deviations from mean for exit
        - stop_loss_pct: Stop loss percentage
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_pct = stop_loss_pct / 100
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.results = {}

        # Set up IST timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print("Mean Reversion Strategy Parameters:")
        print(f"  Lookback Period: {self.lookback_period}")
        print(f"  Entry Threshold: {self.entry_threshold} std deviations")
        print(f"  Exit Threshold: {self.exit_threshold} std deviations")
        print(f"  Stop Loss: {stop_loss_pct}%")
        print(f"  Square-off Time: {square_off_time}")

        # Auto-detect symbols if not provided
        if symbols is None:
            self.symbols = self.auto_detect_symbols()
        else:
            self.symbols = symbols

        print(f"\nSymbols to backtest: {len(self.symbols)}")

    def parse_square_off_time(self, time_str):
        """Parse square-off time string to time object"""
        try:
            hour, minute = map(int, time_str.split(':'))
            return time(hour, minute)
        except:
            return time(15, 20)

    def find_database_files(self):
        """Find all database files in the data folder"""
        # Try multiple possible paths
        possible_paths = [
            self.data_folder,
            "data/symbolupdate",
            "data/marketupdate",
            "data"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                db_pattern = os.path.join(path, "*.db")
                db_files = glob.glob(db_pattern)
                if db_files:
                    db_files.sort()
                    return db_files

        return []

    def auto_detect_symbols(self):
        """Auto-detect symbols from database files"""
        all_symbols = set()

        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                query = """
                SELECT DISTINCT symbol 
                FROM market_data 
                LIMIT 10
                """
                symbols_df = pd.read_sql_query(query, conn)
                conn.close()

                for symbol in symbols_df['symbol']:
                    all_symbols.add(symbol)

            except Exception as e:
                continue

        return list(all_symbols)[:5]  # Limit to 5 symbols for simplicity

    def load_data(self, symbol):
        """Load data for a symbol from database"""
        combined_df = pd.DataFrame()

        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                query = """
                SELECT timestamp, ltp, close_price, high_price, low_price, volume
                FROM market_data 
                WHERE symbol = ? 
                ORDER BY timestamp
                """
                df = pd.read_sql_query(query, conn, params=(symbol,))
                conn.close()

                if not df.empty:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)

            except Exception as e:
                continue

        if combined_df.empty:
            return None

        # Process data
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.set_index('timestamp')
        combined_df['close'] = combined_df['close_price'].fillna(combined_df['ltp'])
        combined_df = combined_df.dropna(subset=['close'])
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        return combined_df

    def calculate_indicators(self, df):
        """Calculate mean reversion indicators"""
        # Calculate moving average
        df['ma'] = df['close'].rolling(window=self.lookback_period).mean()

        # Calculate standard deviation
        df['std'] = df['close'].rolling(window=self.lookback_period).std()

        # Calculate z-score (distance from mean in standard deviations)
        df['z_score'] = (df['close'] - df['ma']) / df['std']

        # Calculate upper and lower bands
        df['upper_band'] = df['ma'] + (self.entry_threshold * df['std'])
        df['lower_band'] = df['ma'] - (self.entry_threshold * df['std'])
        df['exit_upper'] = df['ma'] + (self.exit_threshold * df['std'])
        df['exit_lower'] = df['ma'] - (self.exit_threshold * df['std'])

        return df

    def is_square_off_time(self, timestamp):
        """Check if it's time to square off"""
        try:
            current_time = timestamp.time()
            return current_time >= self.square_off_time
        except:
            return False

    def backtest_symbol(self, symbol):
        """Backtest mean reversion strategy for a single symbol"""
        print(f"\n{'=' * 60}")
        print(f"Backtesting: {symbol}")
        print(f"{'=' * 60}")

        # Load data
        df = self.load_data(symbol)
        if df is None or len(df) < self.lookback_period * 2:
            print(f"Insufficient data for {symbol}")
            return None

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Add trading day info
        df['trading_day'] = df.index.date
        df['is_square_off'] = df.index.map(self.is_square_off_time)

        # Initialize variables
        cash = self.initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        stop_loss_price = 0
        trades = []

        # Backtest loop
        for i in range(self.lookback_period, len(df)):
            current_time = df.index[i]
            current_price = df.iloc[i]['close']
            z_score = df.iloc[i]['z_score']
            is_square_off = df.iloc[i]['is_square_off']

            # Skip if indicators not ready
            if pd.isna(z_score):
                continue

            # Square off at end of day
            if position != 0 and is_square_off:
                exit_reason = "SQUARE_OFF"
                self.close_position(position, current_time, current_price,
                                    entry_price, exit_reason, trades)
                position = 0
                continue

            # Entry signals
            if position == 0 and not is_square_off:
                # Long entry: Price below lower band (oversold)
                if z_score < -self.entry_threshold:
                    position = 1
                    entry_price = current_price
                    stop_loss_price = entry_price * (1 - self.stop_loss_pct)

                    print(f"\nLONG ENTRY:")
                    print(f"  Time: {current_time}")
                    print(f"  Price: {entry_price:.2f}")
                    print(f"  Z-Score: {z_score:.2f}")
                    print(f"  Stop Loss: {stop_loss_price:.2f}")

                # Short entry: Price above upper band (overbought)
                elif z_score > self.entry_threshold:
                    position = -1
                    entry_price = current_price
                    stop_loss_price = entry_price * (1 + self.stop_loss_pct)

                    print(f"\nSHORT ENTRY:")
                    print(f"  Time: {current_time}")
                    print(f"  Price: {entry_price:.2f}")
                    print(f"  Z-Score: {z_score:.2f}")
                    print(f"  Stop Loss: {stop_loss_price:.2f}")

            # Exit signals
            elif position != 0:
                exit_signal = False
                exit_reason = ""

                # Long position exits
                if position == 1:
                    # Take profit: Price returns to mean
                    if z_score > -self.exit_threshold:
                        exit_signal = True
                        exit_reason = "MEAN_REVERSION"
                    # Stop loss
                    elif current_price <= stop_loss_price:
                        exit_signal = True
                        exit_reason = "STOP_LOSS"

                # Short position exits
                elif position == -1:
                    # Take profit: Price returns to mean
                    if z_score < self.exit_threshold:
                        exit_signal = True
                        exit_reason = "MEAN_REVERSION"
                    # Stop loss
                    elif current_price >= stop_loss_price:
                        exit_signal = True
                        exit_reason = "STOP_LOSS"

                if exit_signal:
                    self.close_position(position, current_time, current_price,
                                        entry_price, exit_reason, trades)
                    position = 0

        # Force close any open position at end
        if position != 0:
            self.close_position(position, df.index[-1], df.iloc[-1]['close'],
                                entry_price, "END_OF_DATA", trades)

        return trades

    def close_position(self, position, exit_time, exit_price, entry_price,
                       exit_reason, trades):
        """Close a position and record the trade"""
        if position == 1:
            trade_type = "LONG"
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            trade_type = "SHORT"
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        trade = {
            'type': trade_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct
        }
        trades.append(trade)

        print(f"\n{trade_type} EXIT:")
        print(f"  Time: {exit_time}")
        print(f"  Exit Price: {exit_price:.2f}")
        print(f"  Entry Price: {entry_price:.2f}")
        print(f"  Exit Reason: {exit_reason}")
        print(f"  P&L: {pnl_pct:.2f}%")

        if pnl_pct > 0:
            print(f"  Result: PROFIT")
        else:
            print(f"  Result: LOSS")

    def run_backtest(self):
        """Run backtest for all symbols"""
        print("\n" + "=" * 60)
        print("STARTING MEAN REVERSION BACKTEST")
        print("=" * 60)

        all_trades = {}

        for symbol in self.symbols:
            trades = self.backtest_symbol(symbol)
            if trades:
                all_trades[symbol] = trades

        # Print summary
        self.print_summary(all_trades)

        return all_trades

    def print_summary(self, all_trades):
        """Print backtest summary"""
        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)

        for symbol, trades in all_trades.items():
            if not trades:
                continue

            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t['pnl_pct'] > 0)
            losing_trades = sum(1 for t in trades if t['pnl_pct'] <= 0)

            total_pnl = sum(t['pnl_pct'] for t in trades)
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            # Separate long and short trades
            long_trades = [t for t in trades if t['type'] == 'LONG']
            short_trades = [t for t in trades if t['type'] == 'SHORT']

            print(f"\n{symbol}:")
            print(f"  Total Trades: {total_trades}")
            print(f"  Long Trades: {len(long_trades)}")
            print(f"  Short Trades: {len(short_trades)}")
            print(f"  Winning Trades: {winning_trades}")
            print(f"  Losing Trades: {losing_trades}")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Total P&L: {total_pnl:.2f}%")
            print(f"  Average P&L per Trade: {avg_pnl:.2f}%")

            if trades:
                best_trade = max(trades, key=lambda x: x['pnl_pct'])
                worst_trade = min(trades, key=lambda x: x['pnl_pct'])
                print(f"  Best Trade: {best_trade['pnl_pct']:.2f}%")
                print(f"  Worst Trade: {worst_trade['pnl_pct']:.2f}%")

            # Exit reason breakdown
            exit_reasons = {}
            for trade in trades:
                reason = trade['exit_reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

            print(f"  Exit Reasons:")
            for reason, count in exit_reasons.items():
                print(f"    {reason}: {count}")


# Main execution
if __name__ == "__main__":
    # Initialize backtester
    backtester = SimpleMeanReversionBacktester(
        data_folder="data/symbolupdate",  # Adjust path as needed
        symbols=None,  # Auto-detect symbols
        lookback_period=20,  # 20-period moving average
        entry_threshold=2.0,  # Enter when 2 std deviations from mean
        exit_threshold=0.5,  # Exit when 0.5 std deviations from mean
        stop_loss_pct=1.0,  # 1% stop loss
        initial_capital=100000,
        square_off_time="15:20"  # 3:20 PM IST square-off
    )

    # Run backtest
    results = backtester.run_backtest()