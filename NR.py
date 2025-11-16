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


class NR4NR7Backtester:
    def __init__(self, data_folder="data", symbols=None, strategy_type="NR7",
                 stop_loss_pct=2.0, target_multiplier=2.0, initial_capital=100000,
                 square_off_time="15:20", min_data_points=100, candle_timeframe="1T",
                 breakout_confirmation=True):
        """
        NR4 & NR7 Breakout Strategy Backtester with Candle Conversion

        Parameters:
        - data_folder: Folder containing database files
        - symbols: List of symbols (if None, auto-detect)
        - strategy_type: "NR4", "NR7", or "BOTH" (default "NR7")
        - stop_loss_pct: Stop loss % below NR day low for longs (default 2%)
        - target_multiplier: Target as multiple of NR range (default 2x)
        - initial_capital: Starting capital per symbol
        - square_off_time: Daily square-off time (default "15:20")
        - min_data_points: Minimum data points required
        - candle_timeframe: Candle period - "1T" (1min), "5T" (5min), "15T" (15min), "1H" (1hour), "1D" (1day)
        - breakout_confirmation: Wait for candle close beyond breakout level (default True)
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()
        self.strategy_type = strategy_type.upper()
        self.stop_loss_pct = stop_loss_pct / 100
        self.target_multiplier = target_multiplier
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.candle_timeframe = candle_timeframe
        self.breakout_confirmation = breakout_confirmation
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        # Timeframe descriptions
        timeframe_names = {
            "1T": "1 minute", "5T": "5 minutes", "15T": "15 minutes",
            "30T": "30 minutes", "1H": "1 hour", "1D": "1 day"
        }
        self.timeframe_name = timeframe_names.get(candle_timeframe, candle_timeframe)

        print("=" * 80)
        print(f"NR4 & NR7 BREAKOUT STRATEGY BACKTESTER")
        print("=" * 80)
        print(f"Strategy Type: {self.strategy_type}")
        print(f"Candle Timeframe: {self.timeframe_name}")
        print(f"Stop Loss: {stop_loss_pct}% below NR day low")
        print(f"Target: {target_multiplier}x NR day range")
        print(f"Breakout Confirmation: {'YES (wait for close)' if breakout_confirmation else 'NO (immediate)'}")
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

        print(f"  Created {len(candles)} {self.timeframe_name} candles")

        return candles

    def calculate_nr_indicators(self, candles):
        """Calculate NR4 and NR7 indicators"""
        # Calculate range for each candle
        candles['range'] = candles['high'] - candles['low']

        # NR4: Check if current range is smallest of last 4 days
        candles['nr4'] = False
        for i in range(3, len(candles)):
            current_range = candles.iloc[i]['range']
            last_4_ranges = [candles.iloc[j]['range'] for j in range(i - 3, i + 1)]
            if current_range == min(last_4_ranges) and current_range > 0:
                candles.iloc[i, candles.columns.get_loc('nr4')] = True

        # NR7: Check if current range is smallest of last 7 days
        candles['nr7'] = False
        for i in range(6, len(candles)):
            current_range = candles.iloc[i]['range']
            last_7_ranges = [candles.iloc[j]['range'] for j in range(i - 6, i + 1)]
            if current_range == min(last_7_ranges) and current_range > 0:
                candles.iloc[i, candles.columns.get_loc('nr7')] = True

        print(f"  NR4 Days detected: {candles['nr4'].sum()}")
        print(f"  NR7 Days detected: {candles['nr7'].sum()}")

        return candles

    def is_square_off_time(self, timestamp):
        """Check if square-off time"""
        try:
            return timestamp.time() >= self.square_off_time
        except:
            return False

    def backtest_symbol(self, symbol):
        """Backtest NR4/NR7 strategy for a symbol"""
        print(f"\n{'=' * 80}")
        print(f"BACKTESTING: {symbol}")
        print(f"{'=' * 80}")

        # Load tick data
        tick_data = self.load_data(symbol)
        if tick_data is None or len(tick_data) < 100:
            print(f"❌ Insufficient tick data for {symbol}")
            return None

        print(f"  Loaded {len(tick_data)} tick records")

        # Convert to candles
        candles = self.convert_to_candles(tick_data)

        if len(candles) < 10:
            print(f"❌ Insufficient candles for {symbol}")
            return None

        # Calculate NR indicators
        candles = self.calculate_nr_indicators(candles)
        candles['trading_day'] = candles.index.date
        candles['is_square_off'] = candles.index.map(self.is_square_off_time)

        # Initialize trading variables
        cash = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        stop_loss = 0
        target = 0
        trades = []
        nr_day_idx = None
        nr_day_high = 0
        nr_day_low = 0
        nr_day_range = 0
        nr_type = ""

        # Backtest loop
        for i in range(len(candles)):
            current_time = candles.index[i]
            current_candle = candles.iloc[i]
            current_price = current_candle['close']
            current_high = current_candle['high']
            current_low = current_candle['low']
            is_square_off = current_candle['is_square_off']

            # Identify NR day
            is_nr_day = False
            if self.strategy_type == "NR4" and current_candle['nr4']:
                is_nr_day = True
                nr_type = "NR4"
            elif self.strategy_type == "NR7" and current_candle['nr7']:
                is_nr_day = True
                nr_type = "NR7"
            elif self.strategy_type == "BOTH" and (current_candle['nr4'] or current_candle['nr7']):
                is_nr_day = True
                nr_type = "NR7" if current_candle['nr7'] else "NR4"

            # Mark NR day for next candle breakout
            if is_nr_day and position == 0 and not is_square_off:
                nr_day_idx = i
                nr_day_high = current_candle['high']
                nr_day_low = current_candle['low']
                nr_day_range = nr_day_high - nr_day_low

                print(f"\n🔍 {nr_type} DAY DETECTED at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Range: ₹{nr_day_range:.2f} | High: ₹{nr_day_high:.2f} | Low: ₹{nr_day_low:.2f}")

            # Square-off existing position
            if position != 0 and is_square_off:
                shares = trades[-1]['shares']
                proceeds = shares * current_price if position == 1 else shares * (2 * entry_price - current_price)
                cash += proceeds
                pnl = shares * (current_price - entry_price) if position == 1 else shares * (entry_price - current_price)
                pnl_pct = ((current_price - entry_price) / entry_price) * 100 if position == 1 else ((entry_price - current_price) / entry_price) * 100

                trades[-1].update({
                    'exit_time': current_time,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'SQUARE_OFF_3:20PM'
                })

                self.print_trade(trades[-1])
                position = 0
                nr_day_idx = None

            # Check for breakout after NR day
            elif nr_day_idx is not None and position == 0 and i > nr_day_idx and not is_square_off:
                breakout_long = False
                breakout_short = False

                # Check for breakout with or without confirmation
                if self.breakout_confirmation:
                    # Wait for candle close beyond breakout level
                    if current_price > nr_day_high:
                        breakout_long = True
                    elif current_price < nr_day_low:
                        breakout_short = True
                else:
                    # Immediate breakout on high/low breach
                    if current_high > nr_day_high:
                        breakout_long = True
                    elif current_low < nr_day_low:
                        breakout_short = True

                # Execute LONG breakout
                if breakout_long:
                    entry_price = nr_day_high if self.breakout_confirmation else current_high
                    stop_loss = nr_day_low * (1 - self.stop_loss_pct)
                    target = entry_price + (nr_day_range * self.target_multiplier)
                    shares = int(cash / entry_price)

                    if shares > 0:
                        position = 1
                        cost = shares * entry_price
                        cash -= cost

                        trades.append({
                            'nr_type': nr_type,
                            'direction': 'LONG',
                            'nr_day_time': candles.index[nr_day_idx],
                            'nr_day_range': nr_day_range,
                            'nr_day_high': nr_day_high,
                            'nr_day_low': nr_day_low,
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'target': target,
                            'shares': shares
                        })

                        print(f"\n🟢 LONG BREAKOUT (Above {nr_type} High)")
                        print(f"   Entry: ₹{entry_price:.2f} | Stop: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                        print(f"   Shares: {shares} | Risk: {((entry_price - stop_loss) / entry_price) * 100:.2f}%")

                # Execute SHORT breakout
                elif breakout_short:
                    entry_price = nr_day_low if self.breakout_confirmation else current_low
                    stop_loss = nr_day_high * (1 + self.stop_loss_pct)
                    target = entry_price - (nr_day_range * self.target_multiplier)
                    shares = int(cash / entry_price)

                    if shares > 0:
                        position = -1
                        cost = shares * entry_price
                        cash -= cost

                        trades.append({
                            'nr_type': nr_type,
                            'direction': 'SHORT',
                            'nr_day_time': candles.index[nr_day_idx],
                            'nr_day_range': nr_day_range,
                            'nr_day_high': nr_day_high,
                            'nr_day_low': nr_day_low,
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'target': target,
                            'shares': shares
                        })

                        print(f"\n🔴 SHORT BREAKOUT (Below {nr_type} Low)")
                        print(f"   Entry: ₹{entry_price:.2f} | Stop: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                        print(f"   Shares: {shares} | Risk: {((stop_loss - entry_price) / entry_price) * 100:.2f}%")

            # Monitor existing position
            elif position != 0 and not is_square_off:
                exit_triggered = False
                exit_reason = ""
                exit_price = current_price

                if position == 1:  # Long position
                    if current_low <= stop_loss:
                        exit_triggered = True
                        exit_reason = "STOP_LOSS"
                        exit_price = stop_loss
                    elif current_high >= target:
                        exit_triggered = True
                        exit_reason = "TARGET_HIT"
                        exit_price = target

                elif position == -1:  # Short position
                    if current_high >= stop_loss:
                        exit_triggered = True
                        exit_reason = "STOP_LOSS"
                        exit_price = stop_loss
                    elif current_low <= target:
                        exit_triggered = True
                        exit_reason = "TARGET_HIT"
                        exit_price = target

                if exit_triggered:
                    shares = trades[-1]['shares']
                    proceeds = shares * exit_price if position == 1 else shares * (2 * entry_price - exit_price)
                    cash += proceeds
                    pnl = shares * (exit_price - entry_price) if position == 1 else shares * (entry_price - exit_price)
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if position == 1 else ((entry_price - exit_price) / entry_price) * 100

                    trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason
                    })

                    self.print_trade(trades[-1])
                    position = 0
                    nr_day_idx = None

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

        print(f"\n📊 TRADE CLOSED")
        print(f"   {trade['nr_type']} {trade['direction']} Breakout")
        print(f"   NR Day: {trade['nr_day_time'].strftime('%Y-%m-%d %H:%M:%S')} | Range: ₹{trade['nr_day_range']:.2f}")
        print(f"   Entry:  {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')} @ ₹{trade['entry_price']:.2f}")
        print(f"   Exit:   {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')} @ ₹{trade['exit_price']:.2f}")
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
        print(f"Strategy:           {self.strategy_type} Breakout")
        print(f"Candle Timeframe:   {self.timeframe_name}")
        print(f"Total Candles:      {len(candles)}")
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

        if avg_loss != 0:
            print(f"Profit Factor:      {abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades)):.2f}")
        print(f"{'=' * 80}")

        # Breakdown by NR type
        nr4_trades = [t for t in completed_trades if t['nr_type'] == 'NR4']
        nr7_trades = [t for t in completed_trades if t['nr_type'] == 'NR7']

        if nr4_trades:
            print(f"\nNR4 Trades: {len(nr4_trades)}")
            nr4_wins = len([t for t in nr4_trades if t['pnl'] > 0])
            print(f"  Win Rate: {(nr4_wins / len(nr4_trades)) * 100:.1f}%")

        if nr7_trades:
            print(f"\nNR7 Trades: {len(nr7_trades)}")
            nr7_wins = len([t for t in nr7_trades if t['pnl'] > 0])
            print(f"  Win Rate: {(nr7_wins / len(nr7_trades)) * 100:.1f}%")

        # Direction breakdown
        long_trades = [t for t in completed_trades if t['direction'] == 'LONG']
        short_trades = [t for t in completed_trades if t['direction'] == 'SHORT']

        print(f"\nDirection Breakdown:")
        if long_trades:
            long_wins = len([t for t in long_trades if t['pnl'] > 0])
            print(f"  LONG:  {len(long_trades)} trades ({(long_wins / len(long_trades)) * 100:.1f}% win rate)")
        if short_trades:
            short_wins = len([t for t in short_trades if t['pnl'] > 0])
            print(f"  SHORT: {len(short_trades)} trades ({(short_wins / len(short_trades)) * 100:.1f}% win rate)")

        # Exit reasons
        exit_reasons = {}
        for trade in completed_trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        print(f"\nExit Reasons:")
        for reason, count in exit_reasons.items():
            pct = (count / len(completed_trades)) * 100
            print(f"  {reason:20} {count:3d} trades ({pct:.1f}%)")

    def run_backtest(self):
        """Run backtest for all symbols"""
        print(f"\n🚀 Starting {self.strategy_type} Breakout Backtest")
        print(f"Candle Timeframe: {self.timeframe_name}")
        print(f"Stop Loss: {self.stop_loss_pct * 100}% | Target: {self.target_multiplier}x range\n")

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

        if all_results:
            self.print_overall_summary(all_results)

        return all_results

    def print_overall_summary(self, results):
        """Print overall strategy performance"""
        print(f"\n{'=' * 80}")
        print(f"OVERALL STRATEGY PERFORMANCE")
        print(f"{'=' * 80}")
        print(f"Strategy:              {self.strategy_type} Breakout")
        print(f"Candle Timeframe:      {self.timeframe_name}")
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
        if results:
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
    NR4 & NR7 BREAKOUT STRATEGY

    Strategy Logic:
    1. Identify NR4 (Narrowest Range 4) or NR7 (Narrowest Range 7) days
    2. Wait for breakout above the high or below the low
    3. Enter trade with stop loss below/above the NR day's low/high
    4. Target is a multiple of the NR day's range

    STRATEGY OPTIONS:
    - strategy_type: "NR4", "NR7", or "BOTH"
    - stop_loss_pct: % below NR low for longs (default 2%)
    - target_multiplier: Target as multiple of NR range (default 2x)
    - breakout_confirmation: Wait for candle close (True) or immediate (False)

    CANDLE TIMEFRAME OPTIONS:
    - "1T"   = 1 minute candles
    - "5T"   = 5 minute candles
    - "15T"  = 15 minute candles
    - "1H"   = 1 hour candles
    - "1D"   = 1 day candles (recommended for NR4/NR7)
    """

    # Initialize backtester
    backtester = NR4NR7Backtester(
        data_folder="data/symbolupdate",  # Update with your path
        symbols=None,  # Auto-detect symbols
        strategy_type="NR7",  # "NR4", "NR7", or "BOTH"
        stop_loss_pct=2.0,  # 2% stop loss
        target_multiplier=2.0,  # 2x NR range as target
        initial_capital=100000,
        square_off_time="15:20",
        candle_timeframe="15S",  # Daily candles recommended for NR4/NR7
        breakout_confirmation=True  # Wait for candle close beyond breakout
    )

    # Run backtest
    results = backtester.run_backtest()

    print(f"\n✅ Backtest Complete!")
    print(f"Strategy: {backtester.strategy_type} on {backtester.timeframe_name} candles")
    print(f"Tested {len(results)} symbols")

    # Save results to CSV (optional)
    if results:
        all_trades = []
        for result in results:
            for trade in result['trades']:
                trade_data = {
                    'symbol': result['symbol'],
                    **trade
                }
                all_trades.append(trade_data)

        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            output_file = f"nr4_nr7_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            trades_df.to_csv(output_file, index=False)
            print(f"\n💾 Trades saved to: {output_file}")

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
Example 1: Test NR7 strategy on daily candles (RECOMMENDED)
------------------------------------------------------------
backtester = NR4NR7Backtester(
    data_folder="data/symbolupdate",
    strategy_type="NR7",
    stop_loss_pct=2.0,
    target_multiplier=2.0,
    candle_timeframe="1D",
    breakout_confirmation=True
)


Example 2: Test NR4 strategy with tighter stop loss
----------------------------------------------------
backtester = NR4NR7Backtester(
    data_folder="data/symbolupdate",
    strategy_type="NR4",
    stop_loss_pct=1.5,
    target_multiplier=2.5,
    candle_timeframe="1D",
    breakout_confirmation=True
)


Example 3: Test BOTH NR4 and NR7 on hourly candles
---------------------------------------------------
backtester = NR4NR7Backtester(
    data_folder="data/symbolupdate",
    strategy_type="BOTH",
    stop_loss_pct=2.0,
    target_multiplier=2.0,
    candle_timeframe="1H",
    breakout_confirmation=False  # Immediate breakout
)


Example 4: Aggressive scalping with NR4 on 5-min candles
---------------------------------------------------------
backtester = NR4NR7Backtester(
    data_folder="data/symbolupdate",
    strategy_type="NR4",
    stop_loss_pct=1.0,
    target_multiplier=1.5,
    candle_timeframe="5T",
    breakout_confirmation=True
)


Example 5: Conservative NR7 with larger targets
------------------------------------------------
backtester = NR4NR7Backtester(
    data_folder="data/symbolupdate",
    strategy_type="NR7",
    stop_loss_pct=3.0,
    target_multiplier=3.0,
    candle_timeframe="1D",
    breakout_confirmation=True
)


Example 6: Test specific symbols only
--------------------------------------
backtester = NR4NR7Backtester(
    data_folder="data/symbolupdate",
    symbols=["RELIANCE", "TCS", "INFY", "HDFCBANK"],
    strategy_type="NR7",
    candle_timeframe="1D"
)


Example 7: Intraday NR strategy on 15-min candles
--------------------------------------------------
backtester = NR4NR7Backtester(
    data_folder="data/symbolupdate",
    strategy_type="BOTH",
    stop_loss_pct=1.5,
    target_multiplier=2.0,
    candle_timeframe="15T",
    square_off_time="15:15",  # Exit before market close
    breakout_confirmation=True
)
"""