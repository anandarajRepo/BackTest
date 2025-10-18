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


class SMAScalpingBacktester:
    def __init__(self, data_folder="data", symbols=None, fast_sma=5, slow_sma=20,
                 trailing_stop_pct=1.5, initial_capital=100000, square_off_time="15:20",
                 min_data_points=100, candle_timeframe="1T"):
        """
        Simple SMA Crossover Scalping Strategy Backtester with Candle Conversion

        Parameters:
        - data_folder: Folder containing database files
        - symbols: List of symbols (if None, auto-detect)
        - fast_sma: Fast SMA period (default 5)
        - slow_sma: Slow SMA period (default 20)
        - trailing_stop_pct: Trailing stop loss % (default 1.5%)
        - initial_capital: Starting capital per symbol
        - square_off_time: Daily square-off time (default "15:20")
        - min_data_points: Minimum data points required
        - candle_timeframe: Candle period - "15S" (15sec), "30S" (30sec), "1T" (1min), "5T" (5min)
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()
        self.fast_sma = fast_sma
        self.slow_sma = slow_sma
        self.trailing_stop_pct = trailing_stop_pct / 100
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.candle_timeframe = candle_timeframe
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        # Timeframe descriptions
        timeframe_names = {
            "15S": "15 seconds",
            "30S": "30 seconds",
            "1T": "1 minute",
            "5T": "5 minutes"
        }
        self.timeframe_name = timeframe_names.get(candle_timeframe, candle_timeframe)

        print("=" * 80)
        print("SMA CROSSOVER SCALPING STRATEGY (WITH CANDLE CONVERSION)")
        print("=" * 80)
        print(f"Candle Timeframe: {self.timeframe_name}")
        print(f"Fast SMA: {fast_sma} candles | Slow SMA: {slow_sma} candles")
        print(f"Trailing Stop: {trailing_stop_pct}%")
        print(f"Square-off Time: {square_off_time} IST")
        print(f"Initial Capital: ₹{initial_capital:,}")
        print("=" * 80)

        # Auto-detect symbols
        if symbols is None:
            self.symbols = self.auto_detect_symbols()
        else:
            self.symbols = symbols

        print(f"\nSymbols to backtest: {len(self.symbols)}")

    def parse_square_off_time(self, time_str):
        """Parse square-off time"""
        try:
            hour, minute = map(int, time_str.split(':'))
            return time(hour, minute)
        except:
            return time(15, 20)

    def find_database_files(self):
        """Find database files"""
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
                HAVING count >= ?
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
        """Convert streaming tick data to OHLC candles"""
        print(f"  Converting {len(tick_data)} ticks to {self.timeframe_name} candles...")

        # Define aggregation for OHLC
        ohlc_dict = {
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume': 'sum'
        }

        # Resample to specified timeframe
        candles = tick_data.resample(self.candle_timeframe).agg(ohlc_dict)

        # Add open price (first close of the period)
        candles['open'] = tick_data['close'].resample(self.candle_timeframe).first()

        # Drop rows with no data (weekends, holidays)
        candles = candles.dropna()

        # Reorder columns
        candles = candles[['open', 'high', 'low', 'close', 'volume']]

        print(f"  Created {len(candles)} {self.timeframe_name} candles")

        return candles

    def calculate_sma(self, candles):
        """Calculate SMAs on candle data"""
        candles['sma_fast'] = candles['close'].rolling(window=self.fast_sma, min_periods=self.fast_sma).mean()
        candles['sma_slow'] = candles['close'].rolling(window=self.slow_sma, min_periods=self.slow_sma).mean()

        print(f"  Calculated Fast SMA({self.fast_sma}) and Slow SMA({self.slow_sma}) on {self.timeframe_name} candles")

        return candles

    def generate_signals(self, df):
        """Generate buy/sell signals"""
        # Buy: Fast SMA crosses above Slow SMA
        df['buy_signal'] = (
                (df['sma_fast'] > df['sma_slow']) &
                (df['sma_fast'].shift(1) <= df['sma_slow'].shift(1)) &
                (~df['sma_fast'].isna()) &
                (~df['sma_slow'].isna())
        )

        # Sell: Fast SMA crosses below Slow SMA
        df['sell_signal'] = (
                (df['sma_fast'] < df['sma_slow']) &
                (df['sma_fast'].shift(1) >= df['sma_slow'].shift(1)) &
                (~df['sma_fast'].isna()) &
                (~df['sma_slow'].isna())
        )

        return df

    def is_square_off_time(self, timestamp):
        """Check if square-off time"""
        try:
            return timestamp.time() >= self.square_off_time
        except:
            return False

    def backtest_symbol(self, symbol):
        """Backtest strategy for a symbol"""
        print(f"\n{'=' * 80}")
        print(f"BACKTESTING: {symbol}")
        print(f"{'=' * 80}")

        # Load tick data
        tick_data = self.load_data(symbol)
        if tick_data is None or len(tick_data) < self.slow_sma * 10:
            print(f"❌ Insufficient tick data for {symbol}")
            return None

        print(f"  Loaded {len(tick_data)} tick records")

        # Convert to candles
        candles = self.convert_to_candles(tick_data)

        if len(candles) < self.slow_sma * 2:
            print(f"❌ Insufficient candles for {symbol} (need at least {self.slow_sma * 2})")
            return None

        # Calculate indicators on candles
        candles = self.calculate_sma(candles)
        candles = self.generate_signals(candles)
        candles['trading_day'] = candles.index.date
        candles['is_square_off'] = candles.index.map(self.is_square_off_time)

        # Initialize
        cash = self.initial_capital
        position = 0
        entry_price = 0
        trailing_stop = 0
        trades = []
        entry_day = None

        # Backtest loop on candles
        for i in range(len(candles)):
            current_time = candles.index[i]
            current_price = candles.iloc[i]['close']
            current_day = candles.iloc[i]['trading_day']
            is_square_off = candles.iloc[i]['is_square_off']

            # Square-off at 3:20 PM
            if position == 1 and is_square_off:
                shares = trades[-1]['shares']
                proceeds = shares * current_price
                cash += proceeds
                pnl = shares * (current_price - entry_price)
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                position = 0

                trades.append({
                    'entry_time': trades[-1]['entry_time'],
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': shares,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'SQUARE_OFF_3:20PM',
                    'sma_fast': candles.iloc[i]['sma_fast'],
                    'sma_slow': candles.iloc[i]['sma_slow']
                })

                self.print_trade(trades[-1])
                entry_day = None

            # Buy signal
            elif candles.iloc[i]['buy_signal'] and position == 0 and not is_square_off:
                shares = int(cash / current_price)
                if shares > 0:
                    position = 1
                    entry_price = current_price
                    entry_day = current_day
                    trailing_stop = entry_price * (1 - self.trailing_stop_pct)
                    cost = shares * current_price
                    cash -= cost

                    trades.append({
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'shares': shares,
                        'entry_sma_fast': candles.iloc[i]['sma_fast'],
                        'entry_sma_slow': candles.iloc[i]['sma_slow']
                    })

                    print(f"\n🟢 BUY SIGNAL at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   Price: ₹{entry_price:.2f} | Shares: {shares}")
                    print(f"   Fast SMA: {candles.iloc[i]['sma_fast']:.2f} | Slow SMA: {candles.iloc[i]['sma_slow']:.2f}")

            # Exit conditions
            elif position == 1 and entry_day == current_day and not is_square_off:
                # Update trailing stop
                if current_price > entry_price:
                    new_stop = current_price * (1 - self.trailing_stop_pct)
                    trailing_stop = max(trailing_stop, new_stop)

                # Check exit
                exit_triggered = False
                exit_reason = ""

                if current_price <= trailing_stop:
                    exit_triggered = True
                    exit_reason = "TRAILING_STOP"
                elif candles.iloc[i]['sell_signal']:
                    exit_triggered = True
                    exit_reason = "SMA_CROSSOVER_SELL"

                if exit_triggered:
                    shares = trades[-1]['shares']
                    proceeds = shares * current_price
                    cash += proceeds
                    pnl = shares * (current_price - entry_price)
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    position = 0

                    trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'sma_fast': candles.iloc[i]['sma_fast'],
                        'sma_slow': candles.iloc[i]['sma_slow']
                    })

                    self.print_trade(trades[-1])
                    entry_day = None

        # Summary
        self.print_summary(symbol, trades, cash, candles)

        return {
            'symbol': symbol,
            'trades': [t for t in trades if 'exit_time' in t],
            'final_capital': cash,
            'candles': candles
        }

    def print_trade(self, trade):
        """Print individual trade details"""
        duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60

        print(f"\n📊 TRADE EXECUTED")
        print(f"   Entry:  {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')} @ ₹{trade['entry_price']:.2f}")
        if 'entry_sma_fast' in trade:
            print(f"           Fast SMA: {trade['entry_sma_fast']:.2f} | Slow SMA: {trade['entry_sma_slow']:.2f}")
        print(f"   Exit:   {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')} @ ₹{trade['exit_price']:.2f}")
        if 'sma_fast' in trade:
            print(f"           Fast SMA: {trade['sma_fast']:.2f} | Slow SMA: {trade['sma_slow']:.2f}")
        print(f"   Shares: {trade['shares']}")
        print(f"   Duration: {duration:.1f} minutes")
        print(f"   P&L: ₹{trade['pnl']:.2f} ({trade['pnl_pct']:+.2f}%)")
        print(f"   Exit Reason: {trade['exit_reason']}")

        if trade['pnl'] > 0:
            print(f"   Result: ✅ PROFIT")
        else:
            print(f"   Result: ❌ LOSS")

    def print_summary(self, symbol, trades, final_cash, candles):
        """Print strategy summary"""
        completed_trades = [t for t in trades if 'exit_time' in t]

        if not completed_trades:
            print(f"\n⚠️  No completed trades for {symbol}")
            return

        total_pnl = sum(t['pnl'] for t in completed_trades)
        total_return = ((final_cash - self.initial_capital) / self.initial_capital) * 100
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] <= 0]
        win_rate = (len(winning_trades) / len(completed_trades)) * 100

        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0

        print(f"\n{'=' * 80}")
        print(f"SUMMARY: {symbol}")
        print(f"{'=' * 80}")
        print(f"Candle Timeframe:   {self.timeframe_name}")
        print(f"Total Candles:      {len(candles)}")
        print(f"SMA Configuration:  Fast({self.fast_sma}) / Slow({self.slow_sma}) candles")
        print(f"")
        print(f"Initial Capital:    ₹{self.initial_capital:,.2f}")
        print(f"Final Capital:      ₹{final_cash:,.2f}")
        print(f"Total P&L:          ₹{total_pnl:,.2f}")
        print(f"Total Return:       {total_return:+.2f}%")
        print(f"")
        print(f"Total Trades:       {len(completed_trades)}")
        print(f"Winning Trades:     {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losing Trades:      {len(losing_trades)} ({100 - win_rate:.1f}%)")
        print(f"")
        print(f"Average Win:        ₹{avg_win:,.2f}")
        print(f"Average Loss:       ₹{avg_loss:,.2f}")
        print(f"Best Trade:         ₹{max(t['pnl'] for t in completed_trades):,.2f}")
        print(f"Worst Trade:        ₹{min(t['pnl'] for t in completed_trades):,.2f}")
        print(f"{'=' * 80}")

        # Exit reasons breakdown
        exit_reasons = {}
        for trade in completed_trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        print(f"\nExit Reasons:")
        for reason, count in exit_reasons.items():
            pct = (count / len(completed_trades)) * 100
            print(f"  {reason:25} {count:3d} trades ({pct:.1f}%)")

    def run_backtest(self):
        """Run backtest for all symbols"""
        print(f"\n🚀 Starting SMA Crossover Scalping Backtest")
        print(f"Candle Timeframe: {self.timeframe_name}")
        print(f"Strategy: Fast SMA({self.fast_sma}) crosses Slow SMA({self.slow_sma}) on {self.timeframe_name} candles")
        print(f"Exit: {self.trailing_stop_pct * 100}% trailing stop or SMA crossover\n")

        all_results = []

        for symbol in self.symbols:
            try:
                result = self.backtest_symbol(symbol)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"❌ Error with {symbol}: {e}")
                import traceback
                traceback.print_exc()

        # Overall summary
        if all_results:
            self.print_overall_summary(all_results)

        return all_results

    def print_overall_summary(self, results):
        """Print overall strategy performance"""
        print(f"\n{'=' * 80}")
        print(f"OVERALL STRATEGY PERFORMANCE")
        print(f"{'=' * 80}")
        print(f"Candle Timeframe:      {self.timeframe_name}")
        print(f"SMA Configuration:     Fast({self.fast_sma}) / Slow({self.slow_sma})")
        print(f"")

        total_trades = sum(len(r['trades']) for r in results)
        total_pnl = sum(r['final_capital'] - self.initial_capital for r in results)
        profitable_symbols = sum(1 for r in results if r['final_capital'] > self.initial_capital)

        print(f"Symbols Tested:        {len(results)}")
        print(f"Profitable Symbols:    {profitable_symbols} ({profitable_symbols / len(results) * 100:.1f}%)")
        print(f"Total Trades:          {total_trades}")
        print(f"Total P&L:             ₹{total_pnl:,.2f}")
        print(f"Average Return/Symbol: {(total_pnl / (self.initial_capital * len(results))) * 100:+.2f}%")
        print(f"{'=' * 80}")

        # Best and worst performers
        best = max(results, key=lambda x: x['final_capital'])
        worst = min(results, key=lambda x: x['final_capital'])

        print(f"\n🏆 Best Performer:  {best['symbol']}")
        print(f"   Final Capital: ₹{best['final_capital']:,.2f}")
        print(f"   Return: {((best['final_capital'] - self.initial_capital) / self.initial_capital) * 100:+.2f}%")

        print(f"\n📉 Worst Performer: {worst['symbol']}")
        print(f"   Final Capital: ₹{worst['final_capital']:,.2f}")
        print(f"   Return: {((worst['final_capital'] - self.initial_capital) / self.initial_capital) * 100:+.2f}%")


# Main execution
if __name__ == "__main__":
    """
    CANDLE TIMEFRAME OPTIONS:
    - "15S"  = 15 seconds candles
    - "30S"  = 30 seconds candles
    - "1T"   = 1 minute candles
    - "5T"   = 5 minutes candles

    The strategy will:
    1. Load streaming tick data
    2. Convert to specified candle timeframe
    3. Calculate SMAs on the candle data
    4. Generate buy/sell signals based on SMA crossovers
    """

    # Initialize backtester with candle conversion
    backtester = SMAScalpingBacktester(
        data_folder="data/symbolupdate",  # Update with your path
        symbols=None,  # Auto-detect symbols
        fast_sma=20,  # Fast SMA: 5 candles
        slow_sma=100,  # Slow SMA: 20 candles
        trailing_stop_pct=1.5,  # 1.5% trailing stop
        initial_capital=100000,
        square_off_time="15:20",
        candle_timeframe="15S"  # ⭐ CHANGE THIS: "15S", "30S", "1T", "5T"
    )

    # Run backtest
    results = backtester.run_backtest()

    print(f"\n✅ Backtest Complete!")
    print(f"Tested {len(results)} symbols with {backtester.fast_sma}/{backtester.slow_sma} SMA crossover")
    print(f"on {backtester.timeframe_name} candles")

    # Example: To test with 30-second candles, use:
    # backtester = SMAScalpingBacktester(..., candle_timeframe="30S")

    # Example: To test with 15-second candles, use:
    # backtester = SMAScalpingBacktester(..., candle_timeframe="15S")