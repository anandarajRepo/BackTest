import sqlite3
import pandas as pd
import numpy as np
import json
import glob
import os
from datetime import datetime, time
import pytz
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional

warnings.filterwarnings('ignore')


@dataclass
class ArbitrageTrade:
    """Data class to store arbitrage trade information"""
    entry_time: datetime
    exit_time: datetime
    symbol_spot: str
    symbol_futures: str
    position_type: str  # 'long_futures_short_spot' or 'short_futures_long_spot'
    entry_spot_price: float
    entry_futures_price: float
    exit_spot_price: float
    exit_futures_price: float
    entry_basis: float
    exit_basis: float
    entry_basis_pct: float
    exit_basis_pct: float
    entry_z_score: float
    exit_z_score: float
    quantity: int
    spot_pnl: float
    futures_pnl: float
    total_pnl: float
    total_pnl_pct: float
    exit_reason: str
    holding_period_minutes: float


class SpotFuturesArbitrageBacktester:
    def __init__(self, data_folder="data", symbols=None,
                 basis_lookback=50, entry_zscore_threshold=2.0,
                 exit_zscore_threshold=0.5, stop_loss_pct=1.0,
                 initial_capital=100000, square_off_time="15:20",
                 min_data_points=100, lot_size=1):
        """
        Spot-Futures Arbitrage Strategy Backtester

        Strategy Logic:
        - Calculate basis = futures_price - spot_price
        - Calculate basis percentage = (basis / spot_price) * 100
        - Calculate z-score of basis percentage
        - ENTRY LONG FUTURES + SHORT SPOT: z-score < -entry_threshold (futures underpriced)
        - ENTRY SHORT FUTURES + LONG SPOT: z-score > +entry_threshold (futures overpriced)
        - EXIT: |z-score| < exit_threshold (basis converged to mean)

        Parameters:
        - basis_lookback: Period for calculating basis mean and std dev (default 50)
        - entry_zscore_threshold: Z-score threshold for entry (default 2.0)
        - exit_zscore_threshold: Z-score threshold for exit (default 0.5)
        - stop_loss_pct: Stop loss on total position value (default 1%)
        - lot_size: Futures lot size multiplier (default 1)
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()
        self.basis_lookback = basis_lookback
        self.entry_zscore_threshold = entry_zscore_threshold
        self.exit_zscore_threshold = exit_zscore_threshold
        self.stop_loss_pct = stop_loss_pct / 100
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.lot_size = lot_size
        self.results = {}

        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print("=" * 100)
        print("SPOT-FUTURES ARBITRAGE STRATEGY BACKTESTER")
        print("=" * 100)
        print(f"Strategy: Market-Neutral Basis Trading")
        print(f"Basis Lookback: {basis_lookback} periods")
        print(f"Entry Z-Score Threshold: ±{entry_zscore_threshold}")
        print(f"Exit Z-Score Threshold: ±{exit_zscore_threshold}")
        print(f"Stop Loss: {stop_loss_pct}% on total position")
        print(f"Square-off Time: {square_off_time} IST")
        print(f"Lot Size: {lot_size}")
        print("=" * 100)

        # Auto-detect spot-futures pairs
        if symbols is None:
            print("\nAuto-detecting spot-futures pairs...")
            self.symbol_pairs = self.auto_detect_spot_futures_pairs()
        else:
            self.symbol_pairs = symbols

        print(f"\nSpot-Futures Pairs to backtest: {len(self.symbol_pairs)}")
        for pair in self.symbol_pairs:
            print(f"  - {pair['spot']} <-> {pair['futures']}")

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

    def auto_detect_spot_futures_pairs(self):
        """Auto-detect spot and futures symbol pairs from databases"""
        all_symbols = set()

        # Collect all symbols
        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                query = """
                SELECT DISTINCT symbol 
                FROM market_data 
                WHERE symbol LIKE '%NSE:%'
                """
                symbols_df = pd.read_sql_query(query, conn)
                conn.close()

                for symbol in symbols_df['symbol']:
                    all_symbols.add(symbol)
            except:
                continue

        # Match spot and futures pairs
        pairs = []
        spot_symbols = [s for s in all_symbols if s.endswith('-EQ')]

        for spot in spot_symbols:
            # Extract base symbol name
            # Format: NSE:SYMBOL-EQ -> SYMBOL
            base_name = spot.split(':')[1].replace('-EQ', '') if ':' in spot else spot.replace('-EQ', '')

            # Look for corresponding futures
            # Format: NSE:SYMBOLFUT or NSE:SYMBOL-FUT or NSE:SYMBOLYYMM
            possible_futures = [
                f"NSE:{base_name}FUT",
                f"NSE:{base_name}-FUT",
                f"NSE:{base_name}24DEC",  # Current month
                f"NSE:{base_name}25JAN",  # Next month
            ]

            for futures in possible_futures:
                if futures in all_symbols:
                    pairs.append({
                        'spot': spot,
                        'futures': futures,
                        'base_name': base_name
                    })
                    break  # Take first match

        # Filter pairs with sufficient data
        valid_pairs = []
        for pair in pairs:
            spot_count = self.count_symbol_data(pair['spot'])
            futures_count = self.count_symbol_data(pair['futures'])

            if spot_count >= self.min_data_points and futures_count >= self.min_data_points:
                valid_pairs.append(pair)

        return valid_pairs

    def count_symbol_data(self, symbol):
        """Count data points for a symbol"""
        total_count = 0
        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                query = "SELECT COUNT(*) FROM market_data WHERE symbol = ?"
                result = pd.read_sql_query(query, conn, params=(symbol,))
                conn.close()
                total_count += result.iloc[0, 0]
            except:
                continue
        return total_count

    def load_spot_futures_data(self, spot_symbol, futures_symbol):
        """Load data for both spot and futures"""
        print(f"\n  Loading data for {spot_symbol} and {futures_symbol}...")

        spot_data = self.load_symbol_data(spot_symbol)
        futures_data = self.load_symbol_data(futures_symbol)

        if spot_data is None or futures_data is None:
            return None, None

        print(f"  Spot: {len(spot_data)} records | Futures: {len(futures_data)} records")

        return spot_data, futures_data

    def load_symbol_data(self, symbol):
        """Load data for a single symbol from all databases"""
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
            except:
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

        return combined_df[['close']]

    def align_spot_futures_data(self, spot_data, futures_data):
        """Align spot and futures data on common timestamps"""
        # Merge on index (timestamp)
        aligned = pd.merge(
            spot_data, futures_data,
            left_index=True, right_index=True,
            suffixes=('_spot', '_futures'),
            how='inner'
        )

        print(f"  Aligned data: {len(aligned)} common timestamps")

        return aligned

    def calculate_basis_statistics(self, df):
        """Calculate basis and z-score"""
        # Basis = Futures - Spot
        df['basis'] = df['close_futures'] - df['close_spot']

        # Basis percentage = (basis / spot) * 100
        df['basis_pct'] = (df['basis'] / df['close_spot']) * 100

        # Rolling statistics
        df['basis_mean'] = df['basis_pct'].rolling(window=self.basis_lookback).mean()
        df['basis_std'] = df['basis_pct'].rolling(window=self.basis_lookback).std()

        # Z-score = (current_basis - mean) / std
        df['z_score'] = (df['basis_pct'] - df['basis_mean']) / df['basis_std']

        # Fair value = spot + (spot * basis_mean / 100)
        df['fair_value'] = df['close_spot'] + (df['close_spot'] * df['basis_mean'] / 100)

        # Mispricing = (futures - fair_value) / fair_value
        df['mispricing_pct'] = ((df['close_futures'] - df['fair_value']) / df['fair_value']) * 100

        return df

    def generate_arbitrage_signals(self, df):
        """Generate arbitrage entry and exit signals"""
        # Entry signals
        # Long Futures + Short Spot: When futures are underpriced (negative z-score)
        df['long_futures_signal'] = (
                (df['z_score'] < -self.entry_zscore_threshold) &
                (~df['z_score'].isna()) &
                (df['basis_std'] > 0)
        )

        # Short Futures + Long Spot: When futures are overpriced (positive z-score)
        df['short_futures_signal'] = (
                (df['z_score'] > self.entry_zscore_threshold) &
                (~df['z_score'].isna()) &
                (df['basis_std'] > 0)
        )

        # Exit signals: Basis converged (z-score near zero)
        df['exit_signal'] = (
                (abs(df['z_score']) < self.exit_zscore_threshold) &
                (~df['z_score'].isna())
        )

        return df

    def is_square_off_time(self, timestamp):
        """Check if square-off time"""
        try:
            return timestamp.time() >= self.square_off_time
        except:
            return False

    def backtest_pair(self, pair):
        """Backtest arbitrage strategy for a spot-futures pair"""
        spot_symbol = pair['spot']
        futures_symbol = pair['futures']
        base_name = pair['base_name']

        print(f"\n{'=' * 100}")
        print(f"BACKTESTING: {base_name}")
        print(f"Spot: {spot_symbol} | Futures: {futures_symbol}")
        print(f"{'=' * 100}")

        # Load data
        spot_data, futures_data = self.load_spot_futures_data(spot_symbol, futures_symbol)

        if spot_data is None or futures_data is None:
            print(f" Insufficient data for {base_name}")
            return None

        # Align timestamps
        df = self.align_spot_futures_data(spot_data, futures_data)

        if len(df) < self.basis_lookback * 2:
            print(f" Insufficient aligned data for {base_name}")
            return None

        # Calculate basis statistics
        df = self.calculate_basis_statistics(df)
        df = self.generate_arbitrage_signals(df)

        # Add trading day info
        df['trading_day'] = df.index.date
        df['is_square_off'] = df.index.map(self.is_square_off_time)

        # Initialize trading
        cash = self.initial_capital
        position = None  # 'long_futures' or 'short_futures' or None
        entry_price_spot = 0
        entry_price_futures = 0
        entry_basis = 0
        entry_z_score = 0
        entry_time = None
        trades: List[ArbitrageTrade] = []

        # Backtest loop
        for i in range(self.basis_lookback, len(df)):
            current_time = df.index[i]
            spot_price = df.iloc[i]['close_spot']
            futures_price = df.iloc[i]['close_futures']
            current_basis = df.iloc[i]['basis']
            current_z_score = df.iloc[i]['z_score']
            is_square_off = df.iloc[i]['is_square_off']

            # Skip if z-score is NaN
            if pd.isna(current_z_score):
                continue

            # Force square-off at 3:20 PM
            if position and is_square_off:
                quantity = int((cash * 0.5) / entry_price_spot)  # 50% per leg

                # Calculate P&L for both legs
                if position == 'long_futures':
                    spot_pnl = quantity * (entry_price_spot - spot_price)  # Short spot
                    futures_pnl = quantity * (futures_price - entry_price_futures) * self.lot_size  # Long futures
                else:  # short_futures
                    spot_pnl = quantity * (spot_price - entry_price_spot)  # Long spot
                    futures_pnl = quantity * (entry_price_futures - futures_price) * self.lot_size  # Short futures

                total_pnl = spot_pnl + futures_pnl
                total_pnl_pct = (total_pnl / (quantity * entry_price_spot * 2)) * 100

                trade = ArbitrageTrade(
                    entry_time=entry_time,
                    exit_time=current_time,
                    symbol_spot=spot_symbol,
                    symbol_futures=futures_symbol,
                    position_type=position,
                    entry_spot_price=entry_price_spot,
                    entry_futures_price=entry_price_futures,
                    exit_spot_price=spot_price,
                    exit_futures_price=futures_price,
                    entry_basis=entry_basis,
                    exit_basis=current_basis,
                    entry_basis_pct=(entry_basis / entry_price_spot) * 100,
                    exit_basis_pct=(current_basis / spot_price) * 100,
                    entry_z_score=entry_z_score,
                    exit_z_score=current_z_score,
                    quantity=quantity,
                    spot_pnl=spot_pnl,
                    futures_pnl=futures_pnl,
                    total_pnl=total_pnl,
                    total_pnl_pct=total_pnl_pct,
                    exit_reason='SQUARE_OFF_3:20PM',
                    holding_period_minutes=(current_time - entry_time).total_seconds() / 60
                )

                trades.append(trade)
                cash += total_pnl
                position = None

                self.print_trade(trade)

            # Entry signals
            elif not position and not is_square_off:
                # Long Futures + Short Spot (futures underpriced)
                if df.iloc[i]['long_futures_signal']:
                    position = 'long_futures'
                    entry_price_spot = spot_price
                    entry_price_futures = futures_price
                    entry_basis = current_basis
                    entry_z_score = current_z_score
                    entry_time = current_time

                    print(f"\n LONG FUTURES + SHORT SPOT")
                    print(f"   Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   Spot: ₹{spot_price:.2f} | Futures: ₹{futures_price:.2f}")
                    print(f"   Basis: ₹{current_basis:.2f} ({(current_basis / spot_price) * 100:.3f}%)")
                    print(f"   Z-Score: {current_z_score:.2f} (Futures UNDERPRICED)")

                # Short Futures + Long Spot (futures overpriced)
                elif df.iloc[i]['short_futures_signal']:
                    position = 'short_futures'
                    entry_price_spot = spot_price
                    entry_price_futures = futures_price
                    entry_basis = current_basis
                    entry_z_score = current_z_score
                    entry_time = current_time

                    print(f"\n SHORT FUTURES + LONG SPOT")
                    print(f"   Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   Spot: ₹{spot_price:.2f} | Futures: ₹{futures_price:.2f}")
                    print(f"   Basis: ₹{current_basis:.2f} ({(current_basis / spot_price) * 100:.3f}%)")
                    print(f"   Z-Score: {current_z_score:.2f} (Futures OVERPRICED)")

            # Exit signals
            elif position and not is_square_off:
                should_exit = False
                exit_reason = ""

                # Exit on basis convergence
                if df.iloc[i]['exit_signal']:
                    should_exit = True
                    exit_reason = "BASIS_CONVERGENCE"

                # Stop loss check
                quantity = int((cash * 0.5) / entry_price_spot)
                if position == 'long_futures':
                    spot_pnl = quantity * (entry_price_spot - spot_price)
                    futures_pnl = quantity * (futures_price - entry_price_futures) * self.lot_size
                else:
                    spot_pnl = quantity * (spot_price - entry_price_spot)
                    futures_pnl = quantity * (entry_price_futures - futures_price) * self.lot_size

                total_pnl = spot_pnl + futures_pnl
                loss_pct = abs(total_pnl / (quantity * entry_price_spot * 2))

                if total_pnl < 0 and loss_pct >= self.stop_loss_pct:
                    should_exit = True
                    exit_reason = "STOP_LOSS"

                if should_exit:
                    total_pnl_pct = (total_pnl / (quantity * entry_price_spot * 2)) * 100

                    trade = ArbitrageTrade(
                        entry_time=entry_time,
                        exit_time=current_time,
                        symbol_spot=spot_symbol,
                        symbol_futures=futures_symbol,
                        position_type=position,
                        entry_spot_price=entry_price_spot,
                        entry_futures_price=entry_price_futures,
                        exit_spot_price=spot_price,
                        exit_futures_price=futures_price,
                        entry_basis=entry_basis,
                        exit_basis=current_basis,
                        entry_basis_pct=(entry_basis / entry_price_spot) * 100,
                        exit_basis_pct=(current_basis / spot_price) * 100,
                        entry_z_score=entry_z_score,
                        exit_z_score=current_z_score,
                        quantity=quantity,
                        spot_pnl=spot_pnl,
                        futures_pnl=futures_pnl,
                        total_pnl=total_pnl,
                        total_pnl_pct=total_pnl_pct,
                        exit_reason=exit_reason,
                        holding_period_minutes=(current_time - entry_time).total_seconds() / 60
                    )

                    trades.append(trade)
                    cash += total_pnl
                    position = None

                    self.print_trade(trade)

        # Calculate metrics
        metrics = self.calculate_metrics(trades, df)

        return {
            'pair': pair,
            'trades': trades,
            'metrics': metrics,
            'final_capital': cash,
            'data': df
        }

    def print_trade(self, trade: ArbitrageTrade):
        """Print trade details"""
        print(f"\n ARBITRAGE TRADE CLOSED")
        print(f"   Position: {trade.position_type.replace('_', ' ').upper()}")
        print(f"   Entry:  {trade.entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Exit:   {trade.exit_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Duration: {trade.holding_period_minutes:.1f} minutes")
        print(f"")
        print(f"   Entry Prices  - Spot: ₹{trade.entry_spot_price:.2f} | Futures: ₹{trade.entry_futures_price:.2f}")
        print(f"   Exit Prices   - Spot: ₹{trade.exit_spot_price:.2f} | Futures: ₹{trade.exit_futures_price:.2f}")
        print(f"   Entry Basis:  ₹{trade.entry_basis:.2f} ({trade.entry_basis_pct:.3f}%) | Z-Score: {trade.entry_z_score:.2f}")
        print(f"   Exit Basis:   ₹{trade.exit_basis:.2f} ({trade.exit_basis_pct:.3f}%) | Z-Score: {trade.exit_z_score:.2f}")
        print(f"   Basis Change: {trade.exit_basis - trade.entry_basis:.2f} ({trade.exit_basis_pct - trade.entry_basis_pct:.3f}%)")
        print(f"")
        print(f"   Spot Leg P&L:    ₹{trade.spot_pnl:,.2f}")
        print(f"   Futures Leg P&L: ₹{trade.futures_pnl:,.2f}")
        print(f"   Total P&L:       ₹{trade.total_pnl:,.2f} ({trade.total_pnl_pct:+.2f}%)")
        print(f"   Exit Reason:     {trade.exit_reason}")

        if trade.total_pnl > 0:
            print(f"   Result: PROFIT")
        else:
            print(f"   Result: LOSS")

    def calculate_metrics(self, trades: List[ArbitrageTrade], df):
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl_per_trade': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_holding_period': 0,
                'avg_entry_z_score': 0,
                'avg_exit_z_score': 0,
                'avg_basis_convergence': 0,
                'basis_convergence_trades': 0,
                'stop_loss_trades': 0,
                'square_off_trades': 0,
                'long_futures_trades': 0,
                'short_futures_trades': 0,
                'avg_spot_pnl': 0,
                'avg_futures_pnl': 0
            }

        winning_trades = [t for t in trades if t.total_pnl > 0]
        losing_trades = [t for t in trades if t.total_pnl <= 0]

        basis_convergence = [t for t in trades if t.exit_reason == 'BASIS_CONVERGENCE']
        stop_loss = [t for t in trades if t.exit_reason == 'STOP_LOSS']
        square_off = [t for t in trades if t.exit_reason == 'SQUARE_OFF_3:20PM']

        long_futures = [t for t in trades if t.position_type == 'long_futures']
        short_futures = [t for t in trades if t.position_type == 'short_futures']

        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(trades)) * 100,
            'total_pnl': sum(t.total_pnl for t in trades),
            'avg_pnl_per_trade': np.mean([t.total_pnl for t in trades]),
            'best_trade': max(t.total_pnl for t in trades),
            'worst_trade': min(t.total_pnl for t in trades),
            'avg_holding_period': np.mean([t.holding_period_minutes for t in trades]),
            'avg_entry_z_score': np.mean([abs(t.entry_z_score) for t in trades]),
            'avg_exit_z_score': np.mean([abs(t.exit_z_score) for t in trades]),
            'avg_basis_convergence': np.mean([abs(t.entry_basis_pct - t.exit_basis_pct) for t in trades]),
            'basis_convergence_trades': len(basis_convergence),
            'stop_loss_trades': len(stop_loss),
            'square_off_trades': len(square_off),
            'long_futures_trades': len(long_futures),
            'short_futures_trades': len(short_futures),
            'avg_spot_pnl': np.mean([t.spot_pnl for t in trades]),
            'avg_futures_pnl': np.mean([t.futures_pnl for t in trades]),
            'long_futures_pnl': sum(t.total_pnl for t in long_futures),
            'short_futures_pnl': sum(t.total_pnl for t in short_futures)
        }

    def run_backtest(self):
        """Run backtest for all pairs"""
        print(f"\nStarting Spot-Futures Arbitrage Backtest")
        print(f"Pairs: {len(self.symbol_pairs)}")
        print(f"Entry Threshold: Z-score ±{self.entry_zscore_threshold}")
        print(f"Exit Threshold: Z-score ±{self.exit_zscore_threshold}\n")

        for pair in self.symbol_pairs:
            try:
                result = self.backtest_pair(pair)
                if result:
                    self.results[pair['base_name']] = result
            except Exception as e:
                print(f"Error with {pair['base_name']}: {e}")
                import traceback
                traceback.print_exc()

        if self.results:
            self.print_overall_summary()

        return self.results

    def print_overall_summary(self):
        """Print overall performance summary"""
        print(f"\n{'=' * 100}")
        print("OVERALL SPOT-FUTURES ARBITRAGE PERFORMANCE")
        print(f"{'=' * 100}")

        total_pairs = len(self.results)
        total_trades = sum(r['metrics']['total_trades'] for r in self.results.values())
        total_pnl = sum(r['final_capital'] - self.initial_capital for r in self.results.values())
        avg_return = (total_pnl / (self.initial_capital * total_pairs)) * 100

        profitable_pairs = sum(1 for r in self.results.values() if r['final_capital'] > self.initial_capital)

        print(f"Pairs Tested:          {total_pairs}")
        print(f"Profitable Pairs:      {profitable_pairs} ({(profitable_pairs / total_pairs) * 100:.1f}%)")
        print(f"Total Trades:          {total_trades}")
        print(f"Total P&L:             ₹{total_pnl:,.2f}")
        print(f"Average Return/Pair:   {avg_return:+.2f}%")

        if total_trades > 0:
            all_trades = []
            for result in self.results.values():
                all_trades.extend(result['trades'])

            winning = [t for t in all_trades if t.total_pnl > 0]
            overall_win_rate = (len(winning) / len(all_trades)) * 100
            avg_holding = np.mean([t.holding_period_minutes for t in all_trades])

            print(f"Overall Win Rate:      {overall_win_rate:.1f}%")
            print(f"Avg Holding Period:    {avg_holding:.1f} minutes")

            # Position type breakdown
            long_fut = [t for t in all_trades if t.position_type == 'long_futures']
            short_fut = [t for t in all_trades if t.position_type == 'short_futures']

            print(f"\nPosition Type Breakdown:")
            if long_fut:
                long_fut_pnl = sum(t.total_pnl for t in long_fut)
                long_fut_win = len([t for t in long_fut if t.total_pnl > 0])
                print(f"  Long Futures:  {len(long_fut)} trades | P&L: ₹{long_fut_pnl:,.2f} | Win Rate: {(long_fut_win / len(long_fut)) * 100:.1f}%")

            if short_fut:
                short_fut_pnl = sum(t.total_pnl for t in short_fut)
                short_fut_win = len([t for t in short_fut if t.total_pnl > 0])
                print(f"  Short Futures: {len(short_fut)} trades | P&L: ₹{short_fut_pnl:,.2f} | Win Rate: {(short_fut_win / len(short_fut)) * 100:.1f}%")

            # Exit reason breakdown
            print(f"\nExit Reason Breakdown:")
            convergence = sum(r['metrics']['basis_convergence_trades'] for r in self.results.values())
            stop_loss = sum(r['metrics']['stop_loss_trades'] for r in self.results.values())
            square_off = sum(r['metrics']['square_off_trades'] for r in self.results.values())

            print(f"  Basis Convergence: {convergence} ({(convergence / total_trades) * 100:.1f}%)")
            print(f"  Stop Loss:         {stop_loss} ({(stop_loss / total_trades) * 100:.1f}%)")
            print(f"  Square-off:        {square_off} ({(square_off / total_trades) * 100:.1f}%)")

        # Best and worst pairs
        best_pair = max(self.results.items(), key=lambda x: x[1]['final_capital'])
        worst_pair = min(self.results.items(), key=lambda x: x[1]['final_capital'])

        print(f"\n Best Pair:  {best_pair[0]}")
        print(f"   Return: {((best_pair[1]['final_capital'] - self.initial_capital) / self.initial_capital) * 100:+.2f}%")
        print(f"   Trades: {best_pair[1]['metrics']['total_trades']}")

        print(f"\n Worst Pair: {worst_pair[0]}")
        print(f"   Return: {((worst_pair[1]['final_capital'] - self.initial_capital) / self.initial_capital) * 100:+.2f}%")
        print(f"   Trades: {worst_pair[1]['metrics']['total_trades']}")

        print(f"{'=' * 100}")

    def export_results(self, filename_prefix="spot_futures_arbitrage"):
        """Export results to CSV"""
        if not self.results:
            print("No results to export.")
            return

        # Collect all trades
        all_trades = []
        for pair_name, result in self.results.items():
            for trade in result['trades']:
                all_trades.append({
                    'pair': pair_name,
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'position_type': trade.position_type,
                    'entry_spot_price': trade.entry_spot_price,
                    'entry_futures_price': trade.entry_futures_price,
                    'exit_spot_price': trade.exit_spot_price,
                    'exit_futures_price': trade.exit_futures_price,
                    'entry_basis': trade.entry_basis,
                    'exit_basis': trade.exit_basis,
                    'entry_basis_pct': trade.entry_basis_pct,
                    'exit_basis_pct': trade.exit_basis_pct,
                    'entry_z_score': trade.entry_z_score,
                    'exit_z_score': trade.exit_z_score,
                    'quantity': trade.quantity,
                    'spot_pnl': trade.spot_pnl,
                    'futures_pnl': trade.futures_pnl,
                    'total_pnl': trade.total_pnl,
                    'total_pnl_pct': trade.total_pnl_pct,
                    'exit_reason': trade.exit_reason,
                    'holding_period_minutes': trade.holding_period_minutes
                })

        trades_df = pd.DataFrame(all_trades)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_trades_{timestamp}.csv"
        trades_df.to_csv(filename, index=False)
        print(f"\nTrades exported to: {filename}")

        # Summary by pair
        summary_data = []
        for pair_name, result in self.results.items():
            metrics = result['metrics']
            returns = ((result['final_capital'] - self.initial_capital) / self.initial_capital) * 100

            summary_data.append({
                'pair': pair_name,
                'total_trades': metrics['total_trades'],
                'winning_trades': metrics['winning_trades'],
                'losing_trades': metrics['losing_trades'],
                'win_rate': metrics['win_rate'],
                'total_pnl': metrics['total_pnl'],
                'avg_pnl_per_trade': metrics['avg_pnl_per_trade'],
                'best_trade': metrics['best_trade'],
                'worst_trade': metrics['worst_trade'],
                'return_pct': returns,
                'final_capital': result['final_capital'],
                'avg_holding_period': metrics['avg_holding_period'],
                'basis_convergence_trades': metrics['basis_convergence_trades'],
                'stop_loss_trades': metrics['stop_loss_trades'],
                'square_off_trades': metrics['square_off_trades'],
                'long_futures_trades': metrics['long_futures_trades'],
                'short_futures_trades': metrics['short_futures_trades']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"{filename_prefix}_summary_{timestamp}.csv"
        summary_df.to_csv(summary_filename, index=False)
        print(f"Summary exported to: {summary_filename}")


# Main execution
if __name__ == "__main__":
    """
    SPOT-FUTURES ARBITRAGE STRATEGY

    Market-Neutral Strategy that profits from basis (spread) convergence

    Entry Logic:
    - Long Futures + Short Spot: When basis z-score < -2.0 (futures underpriced)
    - Short Futures + Long Spot: When basis z-score > +2.0 (futures overpriced)

    Exit Logic:
    - Basis Convergence: |z-score| < 0.5
    - Stop Loss: 1% on total position value
    - Square-off: 3:20 PM IST

    Risk Profile:
    - Market-neutral (hedged position)
    - Lower risk than directional strategies
    - Profits from basis convergence, not price movement
    """

    backtester = SpotFuturesArbitrageBacktester(
        data_folder="data",
        symbols=[
            {'spot': 'RELIANCE', 'futures': 'RELIANCE_FUT', 'base_name': 'RELIANCE'},
            {'spot': 'HDFCBANK', 'futures': 'HDFCBANK_FUT', 'base_name': 'HDFCBANK'},
            {'spot': 'INFY', 'futures': 'INFY_FUT', 'base_name': 'INFY'}
        ],  # Auto-detect spot-futures pairs
        basis_lookback=50,  # 50-period lookback for basis statistics
        entry_zscore_threshold=2.0,  # Enter when z-score > 2 or < -2
        exit_zscore_threshold=0.5,  # Exit when |z-score| < 0.5
        stop_loss_pct=10.0,  # 1% stop loss on total position
        initial_capital=100000,
        square_off_time="15:20",
        lot_size=1  # Adjust for actual futures lot size
    )

    # Run backtest
    results = backtester.run_backtest()

    # Export results
    if results:
        backtester.export_results()
        print(f"\n Backtest Complete!")
        print(f"Tested {len(results)} spot-futures pairs")