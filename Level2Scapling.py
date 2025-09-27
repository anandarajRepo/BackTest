# Complete Level 2 Scalping Backtester with Market Depth Analysis
import sqlite3
import pandas as pd
import numpy as np
import json
import glob
import os
from datetime import datetime, time, timedelta
import pytz
import warnings
from typing import Dict, List, Tuple, Optional
import concurrent.futures
from dataclasses import dataclass

warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Trade record for tracking individual trades"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    holding_period: timedelta


class CompleteLevel2ScalpingBacktester:
    """Complete Level 2 Scalping Backtester with Market Depth Analysis"""

    def __init__(self, data_folder="data/marketupdate", symbols=None,
                 initial_capital=100000, position_size_pct=10,
                 min_spread_bps=5, max_spread_bps=50,
                 min_volume_imbalance=0.3, trailing_stop_pct=2.0,
                 square_off_time="15:20", min_data_points=100):

        print("Initializing Complete Level 2 Scalping Backtester")
        print("=" * 60)

        # Store parameters
        self.data_folder = data_folder
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct / 100
        self.min_spread_bps = min_spread_bps
        self.max_spread_bps = max_spread_bps
        self.min_volume_imbalance = min_volume_imbalance
        self.trailing_stop_pct = trailing_stop_pct / 100
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points

        # Trading state
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> position info
        self.trades = []
        self.daily_pnl = {}

        # Set up IST timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        # Enhanced database file finding
        self.db_files = self.enhanced_find_database_files()

        if not self.db_files:
            raise FileNotFoundError(
                "No valid database files found! Please check your database files."
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

    def enhanced_find_database_files(self):
        """Enhanced database file finder"""
        possible_folders = [
            self.data_folder,
            "data/marketupdate", "data/symbolupdate", "data",
            os.path.join(os.getcwd(), "data", "marketupdate"),
            os.path.join(os.getcwd(), "data", "symbolupdate"),
            os.path.join(os.getcwd(), "data")
        ]

        patterns = ["*.db", "*.sqlite", "*.sqlite3", "fyers_*.db", "market_data_*.db"]

        for folder in possible_folders:
            if not os.path.exists(folder):
                continue

            found_files = []
            for pattern in patterns:
                found_files.extend(glob.glob(os.path.join(folder, pattern)))

            if found_files:
                valid_files = []
                for db_file in found_files:
                    try:
                        conn = sqlite3.connect(db_file)
                        tables = pd.read_sql_query(
                            "SELECT name FROM sqlite_master WHERE type='table';", conn
                        )
                        if 'market_data' in tables['name'].values:
                            count = pd.read_sql_query(
                                "SELECT COUNT(*) FROM market_data", conn
                            ).iloc[0, 0]
                            if count > 0:
                                valid_files.append(db_file)
                        conn.close()
                    except:
                        continue

                if valid_files:
                    self.data_folder = folder
                    return sorted(valid_files)

        return []

    def parse_square_off_time(self, time_str):
        """Parse square-off time"""
        try:
            hour, minute = map(int, time_str.split(':'))
            return time(hour, minute)
        except:
            return time(15, 20)

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

            except Exception as e:
                print(f"Error scanning {db_file}: {e}")
                continue

        # Return top symbols by data availability
        sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1], reverse=True)
        return [symbol for symbol, count in sorted_symbols[:20]]  # Top 20 symbols

    def load_market_data(self, symbol):
        """Load and process market data for a symbol"""
        combined_df = pd.DataFrame()

        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                query = """
                SELECT timestamp, symbol, ltp, high_price, low_price, close_price, 
                       volume, raw_data
                FROM market_data 
                WHERE symbol = ? 
                ORDER BY timestamp
                """
                df = pd.read_sql_query(query, conn, params=(symbol,))
                conn.close()

                if not df.empty:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)

            except Exception as e:
                print(f"Error loading from {db_file}: {e}")
                continue

        if combined_df.empty:
            return None

        # Process the data
        combined_df = self.process_market_data(combined_df)
        return combined_df

    def process_market_data(self, df):
        """Process raw market data into trading features"""
        # Convert timestamp and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        # Parse raw market depth data
        def parse_market_depth(raw_data_str):
            try:
                if not raw_data_str:
                    return pd.Series({
                        'bid_price1': np.nan, 'ask_price1': np.nan,
                        'bid_size1': 0, 'ask_size1': 0,
                        'total_bid_size': 0, 'total_ask_size': 0
                    })

                raw_data = json.loads(raw_data_str)

                if raw_data.get('type') == 'dp':
                    # Market depth data
                    total_bid = sum([raw_data.get(f'bid_size{i}', 0) for i in range(1, 6)])
                    total_ask = sum([raw_data.get(f'ask_size{i}', 0) for i in range(1, 6)])

                    return pd.Series({
                        'bid_price1': raw_data.get('bid_price1', np.nan),
                        'ask_price1': raw_data.get('ask_price1', np.nan),
                        'bid_size1': raw_data.get('bid_size1', 0),
                        'ask_size1': raw_data.get('ask_size1', 0),
                        'total_bid_size': total_bid,
                        'total_ask_size': total_ask
                    })
                else:
                    # Legacy data - approximate bid/ask from LTP
                    ltp = raw_data.get('ltp', np.nan)
                    return pd.Series({
                        'bid_price1': ltp * 0.999 if not np.isnan(ltp) else np.nan,
                        'ask_price1': ltp * 1.001 if not np.isnan(ltp) else np.nan,
                        'bid_size1': 100, 'ask_size1': 100,
                        'total_bid_size': 500, 'total_ask_size': 500
                    })
            except:
                return pd.Series({
                    'bid_price1': np.nan, 'ask_price1': np.nan,
                    'bid_size1': 0, 'ask_size1': 0,
                    'total_bid_size': 0, 'total_ask_size': 0
                })

        # Parse market depth
        depth_data = df['raw_data'].apply(parse_market_depth)
        df = pd.concat([df, depth_data], axis=1)

        # Fill missing bid/ask with LTP approximations
        df['bid_price1'] = df['bid_price1'].fillna(df['ltp'] * 0.999)
        df['ask_price1'] = df['ask_price1'].fillna(df['ltp'] * 1.001)

        # Calculate trading features
        df = self.calculate_trading_features(df)

        return df

    def calculate_trading_features(self, df):
        """Calculate Level 2 trading features"""

        # Basic price features
        df['price'] = df['ltp'].fillna(df['close_price'])
        df['high'] = df['high_price'].fillna(df['price'])
        df['low'] = df['low_price'].fillna(df['price'])

        # Spread analysis
        df['spread'] = df['ask_price1'] - df['bid_price1']
        df['spread_bps'] = (df['spread'] / df['price']) * 10000
        df['mid_price'] = (df['bid_price1'] + df['ask_price1']) / 2

        # Volume imbalance
        total_volume = df['total_bid_size'] + df['total_ask_size']
        df['volume_imbalance'] = np.where(
            total_volume > 0,
            (df['total_bid_size'] - df['total_ask_size']) / total_volume,
            0
        )

        # Level 1 imbalance
        level1_volume = df['bid_size1'] + df['ask_size1']
        df['level1_imbalance'] = np.where(
            level1_volume > 0,
            (df['bid_size1'] - df['ask_size1']) / level1_volume,
            0
        )

        # Price pressure
        df['price_pressure'] = np.where(
            df['mid_price'] != 0,
            (df['price'] - df['mid_price']) / df['mid_price'],
            0
        )

        # Momentum features
        df['price_change'] = df['price'].pct_change()
        df['price_momentum_5'] = df['price'].pct_change(5)
        df['spread_momentum'] = df['spread_bps'].pct_change()

        # Volatility estimate
        df['volatility_5min'] = df['price_change'].rolling(5).std()

        # Market microstructure features
        df['effective_spread'] = 2 * abs(df['price'] - df['mid_price'])
        df['relative_effective_spread'] = df['effective_spread'] / df['price']

        # Order flow toxicity (simplified)
        df['order_flow_toxicity'] = abs(df['volume_imbalance']) * abs(df['price_pressure'])

        return df

    def generate_signals(self, df):
        """Generate Level 2 scalping signals"""

        # Initialize signal columns
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['exit_signal'] = False

        # Buy conditions (expect price to go up)
        buy_conditions = (
            # Spread is reasonable (not too wide, indicating good liquidity)
                (df['spread_bps'] >= self.min_spread_bps) &
                (df['spread_bps'] <= self.max_spread_bps) &

                # Strong buying pressure in order book
                (df['volume_imbalance'] > self.min_volume_imbalance) &
                (df['level1_imbalance'] > 0.2) &

                # Price is near or below mid (good entry)
                (df['price_pressure'] <= 0.001) &

                # Recent momentum is not too extreme
                (abs(df['price_momentum_5']) < 0.01) &

                # Volatility is manageable
                (df['volatility_5min'] < 0.02) &

                # Order flow is not too toxic
                (df['order_flow_toxicity'] < 0.5)
        )

        # Sell conditions (expect price to go down)
        sell_conditions = (
            # Spread is reasonable
                (df['spread_bps'] >= self.min_spread_bps) &
                (df['spread_bps'] <= self.max_spread_bps) &

                # Strong selling pressure
                (df['volume_imbalance'] < -self.min_volume_imbalance) &
                (df['level1_imbalance'] < -0.2) &

                # Price is near or above mid
                (df['price_pressure'] >= -0.001) &

                # Momentum conditions
                (abs(df['price_momentum_5']) < 0.01) &
                (df['volatility_5min'] < 0.02) &
                (df['order_flow_toxicity'] < 0.5)
        )

        df.loc[buy_conditions, 'buy_signal'] = True
        df.loc[sell_conditions, 'sell_signal'] = True

        # Exit conditions (for both long and short positions)
        df['exit_signal'] = (
            # Spread becomes too wide (liquidity drying up)
                (df['spread_bps'] > self.max_spread_bps * 1.5) |

                # Volume imbalance reverses significantly
                (abs(df['volume_imbalance'].shift(1) - df['volume_imbalance']) > 0.3) |

                # High order flow toxicity
                (df['order_flow_toxicity'] > 0.8) |

                # High volatility
                (df['volatility_5min'] > 0.03)
        )

        return df

    def execute_strategy(self, symbol, df):
        """Execute the Level 2 scalping strategy for a symbol"""

        if df is None or len(df) < 20:
            return []

        # Generate signals
        df = self.generate_signals(df)

        # Track positions and trades for this symbol
        trades = []
        position = None  # Current position info

        for i, (timestamp, row) in enumerate(df.iterrows()):
            current_time = timestamp.time()

            # Skip if market is closing soon
            if current_time >= self.square_off_time:
                if position:
                    # Force exit before market close
                    trade = self.close_position(position, row, "square_off", timestamp)
                    if trade:
                        trades.append(trade)
                    position = None
                continue

            # Entry logic
            if position is None:
                if row['buy_signal']:
                    position = self.open_position(symbol, 'long', row, timestamp)
                elif row['sell_signal']:
                    position = self.open_position(symbol, 'short', row, timestamp)

            # Exit logic
            elif position is not None:
                exit_reason = None

                # Check for exit signals
                if row['exit_signal']:
                    exit_reason = "exit_signal"

                # Check for trailing stop
                elif self.check_trailing_stop(position, row):
                    exit_reason = "trailing_stop"

                # Check for profit target (quick scalp)
                elif self.check_profit_target(position, row):
                    exit_reason = "profit_target"

                # Check for maximum holding time (5 minutes)
                elif (timestamp - position['entry_time']).total_seconds() > 300:
                    exit_reason = "max_time"

                if exit_reason:
                    trade = self.close_position(position, row, exit_reason, timestamp)
                    if trade:
                        trades.append(trade)
                    position = None
                else:
                    # Update trailing stop
                    self.update_trailing_stop(position, row)

        return trades

    def open_position(self, symbol, side, row, timestamp):
        """Open a new position"""

        # Calculate position size based on available capital
        position_value = self.current_capital * self.position_size_pct
        price = row['ask_price1'] if side == 'long' else row['bid_price1']
        quantity = int(position_value / price)

        if quantity <= 0:
            return None

        position = {
            'symbol': symbol,
            'side': side,
            'entry_time': timestamp,
            'entry_price': price,
            'quantity': quantity,
            'trailing_stop': None,
            'highest_profit': 0,
            'lowest_profit': 0
        }

        # Set initial trailing stop
        if side == 'long':
            position['trailing_stop'] = price * (1 - self.trailing_stop_pct)
        else:
            position['trailing_stop'] = price * (1 + self.trailing_stop_pct)

        return position

    def close_position(self, position, row, exit_reason, timestamp):
        """Close an existing position"""

        side = position['side']
        exit_price = row['bid_price1'] if side == 'long' else row['ask_price1']

        # Calculate P&L
        if side == 'long':
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - exit_price) * position['quantity']

        pnl_pct = pnl / (position['entry_price'] * position['quantity'])

        # Update capital
        self.current_capital += pnl

        # Create trade record
        trade = Trade(
            entry_time=position['entry_time'],
            exit_time=timestamp,
            symbol=position['symbol'],
            side=side,
            entry_price=position['entry_price'],
            exit_price=exit_price,
            quantity=position['quantity'],
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            holding_period=timestamp - position['entry_time']
        )

        return trade

    def check_trailing_stop(self, position, row):
        """Check if trailing stop should trigger"""
        current_price = row['bid_price1'] if position['side'] == 'long' else row['ask_price1']

        if position['side'] == 'long':
            return current_price <= position['trailing_stop']
        else:
            return current_price >= position['trailing_stop']

    def update_trailing_stop(self, position, row):
        """Update trailing stop based on current price"""
        current_price = row['ltp']

        if position['side'] == 'long':
            new_stop = current_price * (1 - self.trailing_stop_pct)
            position['trailing_stop'] = max(position['trailing_stop'], new_stop)
        else:
            new_stop = current_price * (1 + self.trailing_stop_pct)
            position['trailing_stop'] = min(position['trailing_stop'], new_stop)

    def check_profit_target(self, position, row):
        """Check if quick profit target is hit (0.5% for scalping)"""
        current_price = row['ltp']
        entry_price = position['entry_price']

        if position['side'] == 'long':
            profit_pct = (current_price - entry_price) / entry_price
            return profit_pct >= 0.005  # 0.5% profit target
        else:
            profit_pct = (entry_price - current_price) / entry_price
            return profit_pct >= 0.005

    def run_backtest(self, parallel=True):
        """Run the complete backtest"""

        print(f"\nRunning Level 2 Scalping Backtest")
        print(f"Symbols: {len(self.symbols)}")
        print(f"Initial Capital: ₹{self.initial_capital:,.0f}")
        print("=" * 50)

        all_trades = []

        if parallel and len(self.symbols) > 1:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_symbol = {}

                for symbol in self.symbols:
                    future = executor.submit(self._backtest_symbol, symbol)
                    future_to_symbol[future] = symbol

                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        trades = future.result()
                        all_trades.extend(trades)
                        print(f"✓ {symbol}: {len(trades)} trades")
                    except Exception as e:
                        print(f"✗ {symbol}: Error - {e}")
        else:
            # Sequential processing
            for symbol in self.symbols:
                try:
                    trades = self._backtest_symbol(symbol)
                    all_trades.extend(trades)
                    print(f"✓ {symbol}: {len(trades)} trades")
                except Exception as e:
                    print(f"✗ {symbol}: Error - {e}")

        self.trades = all_trades
        return self.analyze_results()

    def _backtest_symbol(self, symbol):
        """Backtest a single symbol (helper method for parallel processing)"""
        df = self.load_market_data(symbol)
        if df is not None:
            return self.execute_strategy(symbol, df)
        return []

    def analyze_results(self):
        """Analyze backtest results"""

        if not self.trades:
            print("No trades executed!")
            return {}

        # Convert trades to DataFrame for analysis
        trades_df = pd.DataFrame([
            {
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'symbol': trade.symbol,
                'side': trade.side,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'exit_reason': trade.exit_reason,
                'holding_period_seconds': trade.holding_period.total_seconds()
            }
            for trade in self.trades
        ])

        # Calculate performance metrics
        total_pnl = trades_df['pnl'].sum()
        total_return_pct = (total_pnl / self.initial_capital) * 100

        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]

        win_rate = len(winning_trades) / len(trades_df) * 100

        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else float('inf')

        max_drawdown = self.calculate_max_drawdown(trades_df)

        # Print results
        print(f"\n{'=' * 60}")
        print(f"LEVEL 2 SCALPING BACKTEST RESULTS")
        print(f"{'=' * 60}")
        print(f"Initial Capital:     ₹{self.initial_capital:,.0f}")
        print(f"Final Capital:       ₹{self.current_capital:,.0f}")
        print(f"Total P&L:           ₹{total_pnl:,.0f}")
        print(f"Total Return:        {total_return_pct:.2f}%")
        print(f"")
        print(f"Total Trades:        {len(trades_df)}")
        print(f"Winning Trades:      {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losing Trades:       {len(losing_trades)} ({100 - win_rate:.1f}%)")
        print(f"")
        print(f"Average Win:         ₹{avg_win:.0f}")
        print(f"Average Loss:        ₹{avg_loss:.0f}")
        print(f"Profit Factor:       {profit_factor:.2f}")
        print(f"Max Drawdown:        {max_drawdown:.2f}%")
        print(f"")
        print(f"Avg Holding Period:  {trades_df['holding_period_seconds'].mean() / 60:.1f} minutes")

        # Exit reason analysis
        print(f"\nEXIT REASON BREAKDOWN:")
        exit_reasons = trades_df['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            print(f"{reason:15}: {count:3d} trades ({count / len(trades_df) * 100:.1f}%)")

        # Symbol performance
        print(f"\nTOP PERFORMING SYMBOLS:")
        symbol_pnl = trades_df.groupby('symbol')['pnl'].agg(['sum', 'count']).sort_values('sum', ascending=False)
        for symbol, (pnl, count) in symbol_pnl.head(10).iterrows():
            print(f"{symbol:20}: ₹{pnl:6.0f} ({count:2d} trades)")

        results = {
            'trades_df': trades_df,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'avg_holding_period_minutes': trades_df['holding_period_seconds'].mean() / 60
        }

        return results

    def calculate_max_drawdown(self, trades_df):
        """Calculate maximum drawdown"""
        trades_df = trades_df.sort_values('exit_time')
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        trades_df['cumulative_capital'] = self.initial_capital + trades_df['cumulative_pnl']

        peak = trades_df['cumulative_capital'].expanding().max()
        drawdown = (trades_df['cumulative_capital'] - peak) / peak * 100

        return abs(drawdown.min())


# Usage Example
if __name__ == "__main__":
    print("LEVEL 2 SCALPING BACKTESTER")
    print("=" * 50)

    try:
        # Initialize backtester
        backtester = CompleteLevel2ScalpingBacktester(
            data_folder="data/marketupdate",
            symbols=None,  # Auto-detect
            initial_capital=100000,
            position_size_pct=10,
            min_spread_bps=5,
            max_spread_bps=50,
            min_volume_imbalance=0.3,
            trailing_stop_pct=2.0,
            square_off_time="15:20"
        )

        # Run backtest
        results = backtester.run_backtest(parallel=True)

        print(f"\nBacktest completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()