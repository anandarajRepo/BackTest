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


class DoubleRSIScalpingBacktester:
    def __init__(self, data_folder="data", symbols=None,
                 fast_rsi_period=7, slow_rsi_period=21,
                 fast_rsi_overbought=70, fast_rsi_oversold=30,
                 slow_rsi_overbought=60, slow_rsi_oversold=40,
                 trailing_stop_pct=1.0, initial_capital=100000,
                 square_off_time="15:20", min_data_points=100,
                 candle_timeframe="1T"):
        """
        Double RSI Scalping Strategy Backtester

        Strategy Logic:
        - BUY: Fast RSI crosses above oversold level (30) AND Slow RSI is above 40
        - SELL: Fast RSI crosses below overbought level (70) OR trailing stop hit

        Parameters:
        - fast_rsi_period: Fast RSI period (default 7)
        - slow_rsi_period: Slow RSI period (default 21)
        - fast_rsi_overbought: Fast RSI overbought threshold (default 70)
        - fast_rsi_oversold: Fast RSI oversold threshold (default 30)
        - slow_rsi_overbought: Slow RSI overbought threshold (default 60)
        - slow_rsi_oversold: Slow RSI oversold threshold (default 40)
        - trailing_stop_pct: Trailing stop % (default 1%)
        - candle_timeframe: Candle period - "15S" (15sec), "30S" (30sec), "1T" (1min), "5T" (5min)
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()
        self.fast_rsi_period = fast_rsi_period
        self.slow_rsi_period = slow_rsi_period
        self.fast_rsi_overbought = fast_rsi_overbought
        self.fast_rsi_oversold = fast_rsi_oversold
        self.slow_rsi_overbought = slow_rsi_overbought
        self.slow_rsi_oversold = slow_rsi_oversold
        self.trailing_stop_pct = trailing_stop_pct / 100
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.candle_timeframe = candle_timeframe
        self.results = {}

        # IST timezone
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
        print("DOUBLE RSI SCALPING STRATEGY (WITH CANDLE CONVERSION)")
        print("=" * 80)
        print(f"Candle Timeframe: {self.timeframe_name}")
        print(f"Fast RSI Period: {fast_rsi_period} candles | Slow RSI Period: {slow_rsi_period} candles")
        print(f"Fast RSI Levels: Oversold < {fast_rsi_oversold}, Overbought > {fast_rsi_overbought}")
        print(f"Slow RSI Levels: Oversold < {slow_rsi_oversold}, Overbought > {slow_rsi_overbought}")
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
        for symbol in self.symbols:
            print(f"  - {symbol}")

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
                LIMIT 5
                """
                symbols_df = pd.read_sql_query(query, conn, params=(self.min_data_points,))
                conn.close()
                for symbol in symbols_df['symbol']:
                    all_symbols.add(symbol)
            except:
                continue
        return list(all_symbols)

    def load_data(self, symbol):
        """Load tick data for a symbol from all databases"""
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

    def calculate_rsi(self, prices, period):
        """Calculate RSI from price series"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

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

        # Drop rows with no data
        candles = candles.dropna()

        # Reorder columns
        candles = candles[['open', 'high', 'low', 'close', 'volume']]

        print(f"  Created {len(candles)} {self.timeframe_name} candles")

        return candles

    def calculate_indicators(self, candles):
        """Calculate both RSI indicators on candle data"""
        candles['fast_rsi'] = self.calculate_rsi(candles['close'], self.fast_rsi_period)
        candles['slow_rsi'] = self.calculate_rsi(candles['close'], self.slow_rsi_period)

        print(f"  Calculated Fast RSI({self.fast_rsi_period}) and Slow RSI({self.slow_rsi_period}) on {self.timeframe_name} candles")

        return candles

    def generate_signals(self, df):
        """Generate buy/sell signals"""
        # BUY: Fast RSI crosses above oversold AND Slow RSI is in favorable zone
        df['buy_signal'] = (
                (df['fast_rsi'] > self.fast_rsi_oversold) &
                (df['fast_rsi'].shift(1) <= self.fast_rsi_oversold) &
                (df['slow_rsi'] > self.slow_rsi_oversold) &
                (~df['fast_rsi'].isna()) &
                (~df['slow_rsi'].isna())
        )

        # SELL: Fast RSI crosses below overbought
        df['sell_signal'] = (
                (df['fast_rsi'] < self.fast_rsi_overbought) &
                (df['fast_rsi'].shift(1) >= self.fast_rsi_overbought) &
                (~df['fast_rsi'].isna())
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
        if tick_data is None or len(tick_data) < max(self.fast_rsi_period, self.slow_rsi_period) * 10:
            print(f"❌ Insufficient tick data for {symbol}")
            return None

        print(f"  Loaded {len(tick_data)} tick records")

        # Convert to candles
        candles = self.convert_to_candles(tick_data)

        if len(candles) < max(self.fast_rsi_period, self.slow_rsi_period) * 2:
            print(f"❌ Insufficient candles for {symbol} (need at least {max(self.fast_rsi_period, self.slow_rsi_period) * 2})")
            return None

        # Calculate indicators on candles
        candles = self.calculate_indicators(candles)
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
        trade_number = 0

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
                duration = (current_time - trades[-1]['entry_time']).total_seconds() / 60

                trade_number += 1

                print(f"\n{'─' * 80}")
                print(f"TRADE #{trade_number} - SQUARE-OFF")
                print(f"{'─' * 80}")
                print(f"  Entry:    {trades[-1]['entry_time'].strftime('%Y-%m-%d %H:%M:%S')} @ ₹{entry_price:.2f}")
                print(f"            Fast RSI: {trades[-1]['entry_fast_rsi']:.1f} | Slow RSI: {trades[-1]['entry_slow_rsi']:.1f}")
                print(f"  Exit:     {current_time.strftime('%Y-%m-%d %H:%M:%S')} @ ₹{current_price:.2f}")
                print(f"            Fast RSI: {candles.iloc[i]['fast_rsi']:.1f} | Slow RSI: {candles.iloc[i]['slow_rsi']:.1f}")
                print(f"  Duration: {duration:.1f} minutes")
                print(f"  Shares:   {shares}")
                print(f"  P&L:      ₹{pnl:,.2f} ({pnl_pct:+.2f}%)")
                print(f"  Reason:   Square-off at 3:20 PM")

                if pnl > 0:
                    print(f"  Result:   ✅ PROFIT")
                else:
                    print(f"  Result:   ❌ LOSS")

                position = 0
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
                        'entry_fast_rsi': candles.iloc[i]['fast_rsi'],
                        'entry_slow_rsi': candles.iloc[i]['slow_rsi']
                    })

                    print(f"\n🟢 BUY SIGNAL at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   Price: ₹{entry_price:.2f} | Shares: {shares}")
                    print(f"   Fast RSI: {candles.iloc[i]['fast_rsi']:.1f} (crossed above {self.fast_rsi_oversold})")
                    print(f"   Slow RSI: {candles.iloc[i]['slow_rsi']:.1f}")

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
                    exit_reason = "Trailing Stop"
                elif candles.iloc[i]['sell_signal']:
                    exit_triggered = True
                    exit_reason = "Fast RSI Overbought"

                if exit_triggered:
                    shares = trades[-1]['shares']
                    proceeds = shares * current_price
                    cash += proceeds
                    pnl = shares * (current_price - entry_price)
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    duration = (current_time - trades[-1]['entry_time']).total_seconds() / 60

                    trade_number += 1

                    print(f"\n{'─' * 80}")
                    print(f"TRADE #{trade_number} - {exit_reason.upper()}")
                    print(f"{'─' * 80}")
                    print(f"  Entry:    {trades[-1]['entry_time'].strftime('%Y-%m-%d %H:%M:%S')} @ ₹{entry_price:.2f}")
                    print(f"            Fast RSI: {trades[-1]['entry_fast_rsi']:.1f} | Slow RSI: {trades[-1]['entry_slow_rsi']:.1f}")
                    print(f"  Exit:     {current_time.strftime('%Y-%m-%d %H:%M:%S')} @ ₹{current_price:.2f}")
                    print(f"            Fast RSI: {candles.iloc[i]['fast_rsi']:.1f} | Slow RSI: {candles.iloc[i]['slow_rsi']:.1f}")
                    print(f"  Duration: {duration:.1f} minutes")
                    print(f"  Shares:   {shares}")
                    print(f"  P&L:      ₹{pnl:,.2f} ({pnl_pct:+.2f}%)")
                    print(f"  Reason:   {exit_reason}")

                    if pnl > 0:
                        print(f"  Result:   ✅ PROFIT")
                    else:
                        print(f"  Result:   ❌ LOSS")

                    position = 0
                    entry_day = None

        # Summary
        self.print_summary(symbol, trade_number, cash, candles)

        return {
            'symbol': symbol,
            'total_trades': trade_number,
            'final_capital': cash,
            'candles': candles
        }

    def print_summary(self, symbol, total_trades, final_cash, candles):
        """Print strategy summary"""
        total_return = ((final_cash - self.initial_capital) / self.initial_capital) * 100

        print(f"\n{'=' * 80}")
        print(f"SUMMARY: {symbol}")
        print(f"{'=' * 80}")
        print(f"Candle Timeframe:   {self.timeframe_name}")
        print(f"Total Candles:      {len(candles)}")
        print(f"Strategy:           Double RSI Scalping")
        print(f"Fast RSI:           {self.fast_rsi_period}-period (Oversold: {self.fast_rsi_oversold}, Overbought: {self.fast_rsi_overbought})")
        print(f"Slow RSI:           {self.slow_rsi_period}-period (Oversold: {self.slow_rsi_oversold}, Overbought: {self.slow_rsi_overbought})")
        print(f"Trailing Stop:      {self.trailing_stop_pct * 100}%")
        print(f"")
        print(f"Initial Capital:    ₹{self.initial_capital:,.2f}")
        print(f"Final Capital:      ₹{final_cash:,.2f}")
        print(f"Total Return:       {total_return:+.2f}%")
        print(f"Total P&L:          ₹{final_cash - self.initial_capital:,.2f}")
        print(f"Total Trades:       {total_trades}")
        print(f"{'=' * 80}")

    def run_backtest(self):
        """Run backtest for all symbols"""
        print(f"\n🚀 Starting Double RSI Scalping Backtest")
        print(f"Candle Timeframe: {self.timeframe_name}")
        print(f"Strategy: Fast RSI({self.fast_rsi_period}) + Slow RSI({self.slow_rsi_period}) on {self.timeframe_name} candles")
        print(f"Entry: Fast RSI > {self.fast_rsi_oversold} AND Slow RSI > {self.slow_rsi_oversold}")
        print(f"Exit: Fast RSI < {self.fast_rsi_overbought} OR {self.trailing_stop_pct * 100}% trailing stop\n")

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
        print(f"OVERALL DOUBLE RSI STRATEGY PERFORMANCE")
        print(f"{'=' * 80}")
        print(f"Candle Timeframe:      {self.timeframe_name}")
        print(f"RSI Configuration:     Fast({self.fast_rsi_period}) / Slow({self.slow_rsi_period})")
        print(f"")

        total_trades = sum(r['total_trades'] for r in results)
        total_pnl = sum(r['final_capital'] - self.initial_capital for r in results)
        avg_return = (total_pnl / (self.initial_capital * len(results))) * 100
        profitable_symbols = sum(1 for r in results if r['final_capital'] > self.initial_capital)

        print(f"Symbols Tested:        {len(results)}")
        print(f"Profitable Symbols:    {profitable_symbols} ({profitable_symbols / len(results) * 100:.1f}%)")
        print(f"Total Trades:          {total_trades}")
        print(f"Avg Trades per Symbol: {total_trades / len(results):.1f}")
        print(f"Total P&L:             ₹{total_pnl:,.2f}")
        print(f"Average Return:        {avg_return:+.2f}%")
        print(f"{'=' * 80}")

        # Best and worst performers
        if results:
            best = max(results, key=lambda x: x['final_capital'])
            worst = min(results, key=lambda x: x['final_capital'])

            best_return = ((best['final_capital'] - self.initial_capital) / self.initial_capital) * 100
            worst_return = ((worst['final_capital'] - self.initial_capital) / self.initial_capital) * 100

            print(f"\n🏆 Best Performer:  {best['symbol']}")
            print(f"   Return: {best_return:+.2f}% | Trades: {best['total_trades']}")

            print(f"\n📉 Worst Performer: {worst['symbol']}")
            print(f"   Return: {worst_return:+.2f}% | Trades: {worst['total_trades']}")


# Main execution
if __name__ == "__main__":
    """
    DOUBLE RSI SCALPING STRATEGY WITH CANDLE CONVERSION

    Entry Signal:
    - Fast RSI (7) crosses above oversold level (30)
    - AND Slow RSI (21) is above oversold level (40)

    Exit Signal:
    - Fast RSI crosses below overbought level (70)
    - OR Trailing stop (1%) hit
    - OR Square-off at 3:20 PM IST

    CANDLE TIMEFRAME OPTIONS:
    - "15S"  = 15 seconds candles
    - "30S"  = 30 seconds candles
    - "1T"   = 1 minute candles
    - "5T"   = 5 minutes candles

    The strategy will:
    1. Load streaming tick data
    2. Convert to specified candle timeframe
    3. Calculate RSI indicators on the candle data
    4. Generate buy/sell signals based on Double RSI crossovers
    """

    # Initialize backtester with candle conversion
    backtester = DoubleRSIScalpingBacktester(
        data_folder="data/symbolupdate",  # Update with your path
        symbols=None,  # Auto-detect symbols
        fast_rsi_period=7,  # Fast RSI: 7 candles
        slow_rsi_period=21,  # Slow RSI: 21 candles
        fast_rsi_overbought=70,  # Fast RSI overbought level
        fast_rsi_oversold=30,  # Fast RSI oversold level
        slow_rsi_overbought=60,  # Slow RSI overbought level
        slow_rsi_oversold=40,  # Slow RSI oversold level
        trailing_stop_pct=1.0,  # 1% trailing stop
        initial_capital=100000,
        square_off_time="15:20",
        candle_timeframe="15S"  # ⭐ CHANGE THIS: "15S", "30S", "1T", "5T"
    )

    # Run backtest
    results = backtester.run_backtest()

    print(f"\n✅ Backtest Complete!")
    print(f"Tested {len(results)} symbols with Double RSI strategy")
    print(f"Fast RSI({backtester.fast_rsi_period}) / Slow RSI({backtester.slow_rsi_period})")
    print(f"on {backtester.timeframe_name} candles")

    # Example: To test with 30-second candles, use:
    # backtester = DoubleRSIScalpingBacktester(..., candle_timeframe="30S")

    # Example: To test with 15-second candles, use:
    # backtester = DoubleRSIScalpingBacktester(..., candle_timeframe="15S")