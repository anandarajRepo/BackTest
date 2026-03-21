import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, time, timedelta
import pytz
import warnings
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()


class LongOnlyGridBacktester:
    def __init__(self, fyers_client_id, fyers_access_token, symbols=None,
                 grid_levels=10,
                 grid_spacing_pct=1.0,
                 use_atr_spacing=False,
                 atr_period=14,
                 atr_spacing_mult=1.0,
                 position_size_pct=10.0,
                 stop_loss_levels=3,
                 max_grid_positions=10,
                 initial_capital=100000,
                 square_off_time="15:20",
                 min_data_points=50,
                 tick_interval='5',
                 backtest_days=30,
                 grid_anchor='open'):
        """
        Long-Only Grid Strategy (WunderTrading Style)

        STRATEGY CONCEPT:
        -----------------
        A long-only grid strategy places buy orders at each designated price level
        within a grid. It is primarily used for accumulating a position across multiple
        price levels during a downtrend or sideways market, profiting from bounces back
        up to the next grid level.

        HOW THE GRID IS BUILT:
        ----------------------
        - An anchor price (default: day's opening price) defines the top of the grid.
        - Levels are spaced downward by a fixed percentage (grid_spacing_pct) or by
          a multiple of ATR (if use_atr_spacing=True).
        - Example with 1% spacing and anchor=100:
            Level 9 (top): 100.00
            Level 8      :  99.00
            Level 7      :  98.01
            Level 6      :  97.03
            ...

        BUY LOGIC (Accumulation):
        -------------------------
        - A buy is triggered when the candle's price CROSSES DOWN through a grid level
          (previous close was above the level, current candle low dips at or below it).
        - Also buys if price crosses UP through a level from below (e.g., after a reset),
          treating any level interaction as an opportunity to accumulate.
        - Only one open position is allowed per grid level at any time.
        - Position size per level = position_size_pct % of initial capital.

        TAKE PROFIT LOGIC:
        ------------------
        - Each buy at level[i] has its take-profit target at level[i+1] (one level above).
        - When the candle's high touches or crosses the target level, that specific position
          is closed at the target price.
        - Other positions at lower levels remain open until their own targets are hit or
          until a stop-loss event.

        STOP LOSS / EXIT LOGIC:
        -----------------------
        - If the candle's low crosses the level that is exactly 3 levels below the FIRST
          buy level for the current grid session, ALL open positions are closed at that level.
        - The grid is then immediately RESET with the stop-loss level as the new anchor,
          and fresh buying resumes from the new grid.
        - This limits drawdown to 3 grid spacings from the initial buy point while
          allowing the strategy to re-accumulate at lower prices.

        END-OF-DAY SQUARE-OFF:
        ----------------------
        - All positions are closed at the current market price at square_off_time.
        - The grid resets fresh at the next day's open.

        PARAMETERS:
        -----------
        - grid_levels        : Total number of downward price levels to create (default: 10)
        - grid_spacing_pct   : % distance between each grid level (default: 1.0%)
        - use_atr_spacing    : Use ATR * atr_spacing_mult as grid spacing instead of % (default: False)
        - atr_period         : ATR calculation period (default: 14)
        - atr_spacing_mult   : Multiplier on ATR for spacing (default: 1.0)
        - position_size_pct  : % of initial capital to deploy per grid level (default: 10%)
        - stop_loss_levels   : Number of levels below first buy to trigger stop loss (default: 3)
        - max_grid_positions : Maximum concurrent open grid positions (default: 10)
        - initial_capital    : Starting capital in ₹ (default: 100,000)
        - square_off_time    : Intraday square-off time as "HH:MM" (default: "15:20")
        - tick_interval      : Candle interval in minutes for Fyers API (default: '5')
        - backtest_days      : Number of historical days to backtest (default: 30)
        - grid_anchor        : Price used to anchor the grid top each day (default: 'open')
        """
        # Fyers client
        self.fyers = fyersModel.FyersModel(
            client_id=fyers_client_id,
            token=fyers_access_token,
            is_async=False,
            log_path=""
        )

        # Date range
        self.backtest_days = backtest_days
        ist_tz = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist_tz)
        start_date = end_date - timedelta(days=backtest_days)
        self.range_from = int(start_date.timestamp())
        self.range_to = int(end_date.timestamp())

        # Grid parameters
        self.grid_levels = grid_levels
        self.grid_spacing_pct = grid_spacing_pct / 100
        self.use_atr_spacing = use_atr_spacing
        self.atr_period = atr_period
        self.atr_spacing_mult = atr_spacing_mult
        self.position_size_pct = position_size_pct / 100
        self.stop_loss_levels = stop_loss_levels
        self.max_grid_positions = max_grid_positions
        self.grid_anchor = grid_anchor

        # Trading parameters
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_time(square_off_time)
        self.min_data_points = min_data_points
        self.tick_interval = tick_interval

        # Results
        self.results = {}
        self.combined_data = {}
        self.ist_tz = ist_tz

        print(f"{'='*100}")
        print(f"LONG-ONLY GRID STRATEGY (WUNDERTRADING STYLE) - INTRADAY BACKTEST")
        print(f"{'='*100}")
        print(f"Data Source      : Fyers API")
        print(f"Backtest Period  : Last {backtest_days} days")
        print(f"Date Range       : {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"\nGrid Parameters:")
        print(f"  Grid Levels        : {grid_levels}")
        if use_atr_spacing:
            print(f"  Grid Spacing       : {atr_spacing_mult}x ATR (ATR period {atr_period})")
        else:
            print(f"  Grid Spacing       : {grid_spacing_pct:.2f}% per level")
        print(f"  Grid Anchor        : Day {grid_anchor.upper()}")
        print(f"  Max Positions      : {max_grid_positions}")
        print(f"\nTrade Parameters:")
        print(f"  Position Size      : {position_size_pct:.1f}% of capital per level")
        print(f"  Stop Loss Levels   : {stop_loss_levels} levels below first buy")
        print(f"  Take Profit        : Next level above each buy")
        print(f"  Square-off Time    : {square_off_time}")
        print(f"  Initial Capital    : ₹{initial_capital:,}")
        print(f"{'='*100}")

        if symbols is None:
            self.symbols = ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ", "NSE:TCS-EQ"]
            print(f"\nUsing default symbols: {self.symbols}")
        else:
            self.symbols = symbols
            print(f"\nSymbols to backtest: {len(self.symbols)}")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def parse_time(self, time_str):
        """Parse 'HH:MM' string to a time object."""
        try:
            h, m = map(int, time_str.split(':'))
            return time(h, m)
        except Exception:
            return time(15, 20)

    def is_square_off_time(self, timestamp):
        """Return True if current time >= square_off_time."""
        try:
            return timestamp.time() >= self.square_off_time
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data_from_fyers(self, symbol):
        """Load historical OHLCV data from Fyers API."""
        try:
            print(f"  Fetching data from Fyers for {symbol}...")

            data = {
                "symbol": symbol,
                "resolution": self.tick_interval,
                "date_format": "0",
                "range_from": str(self.range_from),
                "range_to": str(self.range_to),
                "cont_flag": "1"
            }

            response = self.fyers.history(data=data)

            if response.get('s') != 'ok' or 'candles' not in response:
                print(f"  Error fetching data: {response.get('message', 'No candles data')}")
                return None

            candles = response['candles']
            if not candles:
                print(f"  No data available for {symbol}")
                return None

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')
            df.index = df.index.tz_localize('UTC').tz_convert(self.ist_tz)
            df.index = df.index.tz_localize(None)
            df = df.between_time('09:00', '15:30')
            df = df.dropna(subset=['close', 'high', 'low'])

            print(f"  Loaded {len(df)} candles from {df.index[0].date()} to {df.index[-1].date()}")
            return df[['open', 'high', 'low', 'close', 'volume']].copy()

        except Exception as e:
            print(f"  Error loading data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------

    def calculate_atr(self, df):
        """Calculate Average True Range (ATR)."""
        df = df.copy()
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = df[['high', 'prev_close']].max(axis=1) - df[['low', 'prev_close']].min(axis=1)
        df['atr'] = df['tr'].ewm(span=self.atr_period, adjust=False).mean()
        return df

    # ------------------------------------------------------------------
    # Grid computation
    # ------------------------------------------------------------------

    def compute_grid_levels(self, anchor_price, atr=None):
        """
        Build an ascending list of grid price levels.

        Level index 0  = lowest (furthest below anchor)
        Level index N  = anchor (top of downward grid)

        The grid extends 'grid_levels' steps below anchor and one step
        above (used as the take-profit for the top level).
        """
        if self.use_atr_spacing and atr is not None and atr > 0:
            spacing = atr * self.atr_spacing_mult
            # Build levels: anchor at top, going down
            raw = [anchor_price - spacing * i for i in range(self.grid_levels + 2)]
        else:
            raw = [anchor_price * ((1 - self.grid_spacing_pct) ** i)
                   for i in range(self.grid_levels + 2)]

        raw.sort()   # ascending: index 0 = lowest price
        return raw

    def find_level_crosses(self, prev_close, curr_low, curr_high, grid_levels):
        """
        Identify which grid levels the current candle crossed.

        Returns:
            crossed_down : list of level indices where price crossed downward
                           (prev_close > level >= curr_low)
            crossed_up   : list of level indices where price crossed upward
                           (prev_close < level <= curr_high)
        """
        crossed_down = []
        crossed_up = []
        for idx, lvl in enumerate(grid_levels):
            if prev_close > lvl >= curr_low:
                crossed_down.append(idx)
            if prev_close < lvl <= curr_high:
                crossed_up.append(idx)
        return crossed_down, crossed_up

    # ------------------------------------------------------------------
    # Core backtesting logic
    # ------------------------------------------------------------------

    def backtest_single_symbol(self, symbol):
        """Run the Long-Only Grid backtest for a single symbol."""
        print(f"\n{'='*100}")
        print(f"Backtesting: {symbol}")
        print(f"{'='*100}")

        df = self.load_data_from_fyers(symbol)
        if df is None or len(df) < max(self.atr_period * 2, self.min_data_points):
            print(f"  Insufficient data for {symbol}. Skipping.")
            return None

        self.combined_data[symbol] = df.copy()
        df = self.calculate_atr(df)

        # ----------------------------------------------------------------
        # State variables (reset per session / per day)
        # ----------------------------------------------------------------
        cash = self.initial_capital
        trades = []           # All completed trades
        trade_number = 0

        # Active grid state
        grid_levels = []      # Current sorted list of price levels
        active_positions = [] # Each dict: {level_idx, level_price, buy_price, shares,
                              #              entry_time, target_price, target_level_idx}
        active_level_idxs = set()
        first_buy_level_idx = None   # Level index of the first buy this session
        current_day = None

        position_value = self.initial_capital * self.position_size_pct

        # ----------------------------------------------------------------
        # Candle-by-candle loop (start from index 1 for prev_close)
        # ----------------------------------------------------------------
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            ts = df.index[i]
            curr_day = ts.date()

            # ---- Day change: reset grid --------------------------------
            if curr_day != current_day:
                current_day = curr_day
                grid_levels = []
                active_positions = []
                active_level_idxs = set()
                first_buy_level_idx = None

            # ---- Build grid if not set for this day --------------------
            if not grid_levels:
                if self.grid_anchor == 'open':
                    anchor = row['open']
                else:
                    anchor = row['close']
                atr_val = row['atr'] if self.use_atr_spacing else None
                grid_levels = self.compute_grid_levels(anchor, atr_val)

            curr_low = row['low']
            curr_high = row['high']
            curr_close = row['close']
            prev_close = prev_row['close']
            curr_atr = row['atr']

            # ---- Square-off: close all at EOD --------------------------
            if self.is_square_off_time(ts):
                if active_positions:
                    for pos in active_positions:
                        pnl = pos['shares'] * (curr_close - pos['buy_price'])
                        ret_pct = (pnl / (pos['shares'] * pos['buy_price'])) * 100

                        trade_number += 1
                        duration = (ts - pos['entry_time']).total_seconds() / 60

                        print(f"\n  [Trade #{trade_number}] SQUARE-OFF @ {ts.strftime('%H:%M')}")
                        print(f"    Level: {pos['level_idx']} | Entry ₹{pos['buy_price']:.2f} → Exit ₹{curr_close:.2f}")
                        print(f"    Shares: {pos['shares']} | P&L: ₹{pnl:.2f} ({ret_pct:+.2f}%)")
                        print(f"    {'PROFIT' if pnl > 0 else 'LOSS'}")

                        trades.append({
                            'trade_num': trade_number,
                            'symbol': symbol,
                            'direction': 'LONG',
                            'entry_time': pos['entry_time'],
                            'entry_price': pos['buy_price'],
                            'exit_time': ts,
                            'exit_price': curr_close,
                            'shares': pos['shares'],
                            'pnl': pnl,
                            'return_pct': ret_pct,
                            'exit_reason': 'SQUARE_OFF',
                            'grid_level': pos['level_idx'],
                            'target_price': pos['target_price'],
                            'duration_minutes': duration
                        })

                    active_positions = []
                    active_level_idxs = set()
                    first_buy_level_idx = None
                continue

            # ---- Take Profit: check each position's target -------------
            positions_to_close = []
            for pos in active_positions:
                if curr_high >= pos['target_price']:
                    tp_price = pos['target_price']
                    pnl = pos['shares'] * (tp_price - pos['buy_price'])
                    ret_pct = (pnl / (pos['shares'] * pos['buy_price'])) * 100
                    duration = (ts - pos['entry_time']).total_seconds() / 60

                    trade_number += 1
                    print(f"\n  [Trade #{trade_number}] TAKE PROFIT @ {ts.strftime('%H:%M')}")
                    print(f"    Level: {pos['level_idx']} → TP Level: {pos['target_level_idx']}")
                    print(f"    Entry ₹{pos['buy_price']:.2f} → TP ₹{tp_price:.2f}")
                    print(f"    Shares: {pos['shares']} | P&L: ₹{pnl:.2f} ({ret_pct:+.2f}%) | Duration: {duration:.1f}m")
                    print(f"    PROFIT")

                    trades.append({
                        'trade_num': trade_number,
                        'symbol': symbol,
                        'direction': 'LONG',
                        'entry_time': pos['entry_time'],
                        'entry_price': pos['buy_price'],
                        'exit_time': ts,
                        'exit_price': tp_price,
                        'shares': pos['shares'],
                        'pnl': pnl,
                        'return_pct': ret_pct,
                        'exit_reason': 'TAKE_PROFIT',
                        'grid_level': pos['level_idx'],
                        'target_price': pos['target_price'],
                        'duration_minutes': duration
                    })
                    positions_to_close.append(pos)

            for pos in positions_to_close:
                active_positions.remove(pos)
                active_level_idxs.discard(pos['level_idx'])

            # Update first_buy_level_idx if the original first buy was closed
            if active_positions:
                first_buy_level_idx = min(pos['level_idx'] for pos in active_positions)
            else:
                first_buy_level_idx = None

            # ---- Stop Loss: 3 levels below first buy -------------------
            if active_positions and first_buy_level_idx is not None:
                sl_level_idx = first_buy_level_idx - self.stop_loss_levels
                if sl_level_idx >= 0:
                    sl_price = grid_levels[sl_level_idx]
                    if curr_low <= sl_price:
                        print(f"\n  [STOP LOSS] @ {ts.strftime('%H:%M')} — Price hit "
                              f"₹{sl_price:.2f} ({self.stop_loss_levels} levels below first buy "
                              f"at level {first_buy_level_idx})")

                        for pos in active_positions:
                            pnl = pos['shares'] * (sl_price - pos['buy_price'])
                            ret_pct = (pnl / (pos['shares'] * pos['buy_price'])) * 100
                            duration = (ts - pos['entry_time']).total_seconds() / 60

                            trade_number += 1
                            print(f"    [Trade #{trade_number}] Level {pos['level_idx']}: "
                                  f"Entry ₹{pos['buy_price']:.2f} → SL ₹{sl_price:.2f} | "
                                  f"P&L ₹{pnl:.2f} ({ret_pct:+.2f}%)")

                            trades.append({
                                'trade_num': trade_number,
                                'symbol': symbol,
                                'direction': 'LONG',
                                'entry_time': pos['entry_time'],
                                'entry_price': pos['buy_price'],
                                'exit_time': ts,
                                'exit_price': sl_price,
                                'shares': pos['shares'],
                                'pnl': pnl,
                                'return_pct': ret_pct,
                                'exit_reason': 'STOP_LOSS',
                                'grid_level': pos['level_idx'],
                                'target_price': pos['target_price'],
                                'duration_minutes': duration
                            })

                        # Reset grid anchored at stop-loss level
                        print(f"  >> Grid RESET anchored at ₹{sl_price:.2f} (SL level {sl_level_idx})")
                        active_positions = []
                        active_level_idxs = set()
                        first_buy_level_idx = None

                        atr_val = curr_atr if self.use_atr_spacing else None
                        grid_levels = self.compute_grid_levels(sl_price, atr_val)
                        continue  # Skip buy checks this candle after reset

            # ---- Buy: price crosses down (or up) through a grid level --
            crossed_down, crossed_up = self.find_level_crosses(
                prev_close, curr_low, curr_high, grid_levels
            )

            # Combine: any level interaction triggers a buy
            candidate_levels = sorted(set(crossed_down + crossed_up))

            for level_idx in candidate_levels:
                # Skip if already have a position at this level
                if level_idx in active_level_idxs:
                    continue
                # Respect maximum concurrent positions
                if len(active_positions) >= self.max_grid_positions:
                    break

                level_price = grid_levels[level_idx]

                # Take-profit is one level above the buy level
                tp_level_idx = level_idx + 1
                if tp_level_idx < len(grid_levels):
                    tp_price = grid_levels[tp_level_idx]
                else:
                    # Extend one step above anchor if at the top
                    if self.use_atr_spacing and curr_atr > 0:
                        tp_price = level_price + (curr_atr * self.atr_spacing_mult)
                    else:
                        tp_price = level_price * (1 + self.grid_spacing_pct)

                shares = int(position_value / level_price)
                if shares <= 0:
                    continue

                pos = {
                    'level_idx': level_idx,
                    'level_price': level_price,
                    'buy_price': level_price,
                    'shares': shares,
                    'entry_time': ts,
                    'target_price': tp_price,
                    'target_level_idx': tp_level_idx
                }
                active_positions.append(pos)
                active_level_idxs.add(level_idx)

                # Track first buy level
                if first_buy_level_idx is None:
                    first_buy_level_idx = level_idx
                else:
                    first_buy_level_idx = min(first_buy_level_idx, level_idx)

                direction = "DOWN" if level_idx in crossed_down else "UP"
                print(f"\n  [GRID BUY] {ts.strftime('%Y-%m-%d %H:%M')} — Level {level_idx} "
                      f"(cross {direction})")
                print(f"    Entry ₹{level_price:.2f} | TP ₹{tp_price:.2f} "
                      f"(Level {tp_level_idx}) | Shares: {shares}")
                print(f"    Active Positions: {len(active_positions)} | "
                      f"First Buy Level: {first_buy_level_idx} | "
                      f"SL Trigger: Level {first_buy_level_idx - self.stop_loss_levels if first_buy_level_idx is not None else 'N/A'}")

        # ----------------------------------------------------------------
        # Calculate metrics and return
        # ----------------------------------------------------------------
        metrics = self.calculate_metrics(trades)
        return {
            'symbol': symbol,
            'data': df,
            'trades': trades,
            'metrics': metrics
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def calculate_metrics(self, trades):
        """Calculate performance metrics from completed trades."""
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
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'tp_count': 0,
                'sl_count': 0,
                'sq_off_count': 0
            }

        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        losing_trades = total_trades - winning_trades

        pnls = [t['pnl'] for t in trades]
        rets = [t['return_pct'] for t in trades]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        profit_factor = (sum(wins) / abs(sum(losses))) if losses else 0

        by_reason = {}
        for t in trades:
            reason = t.get('exit_reason', 'UNKNOWN')
            by_reason[reason] = by_reason.get(reason, 0) + 1

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades else 0,
            'total_pnl': sum(pnls),
            'avg_pnl': sum(pnls) / total_trades,
            'total_return': sum(rets),
            'avg_return': sum(rets) / total_trades,
            'best_trade': max(pnls),
            'worst_trade': min(pnls),
            'avg_duration': sum(t['duration_minutes'] for t in trades) / total_trades,
            'avg_win': sum(wins) / len(wins) if wins else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'profit_factor': profit_factor,
            'tp_count': by_reason.get('TAKE_PROFIT', 0),
            'sl_count': by_reason.get('STOP_LOSS', 0),
            'sq_off_count': by_reason.get('SQUARE_OFF', 0)
        }

    # ------------------------------------------------------------------
    # Run & summary
    # ------------------------------------------------------------------

    def run_backtest(self):
        """Run the Long-Only Grid backtest for all configured symbols."""
        print(f"\n{'='*100}")
        print("STARTING LONG-ONLY GRID BACKTEST")
        print(f"{'='*100}")

        for symbol in self.symbols:
            try:
                result = self.backtest_single_symbol(symbol)
                if result:
                    self.results[symbol] = result
            except Exception as e:
                print(f"Error backtesting {symbol}: {e}")
                import traceback
                traceback.print_exc()

        self.print_summary()

    def print_summary(self):
        """Print a consolidated summary table and save results to CSV."""
        if not self.results:
            print("\nNo results to display.")
            return

        print(f"\n{'='*100}")
        print("LONG-ONLY GRID STRATEGY — BACKTEST RESULTS SUMMARY")
        print(f"{'='*100}")

        rows = []
        for symbol, result in self.results.items():
            m = result['metrics']
            clean = symbol.split(':')[1].replace('-EQ', '') if ':' in symbol else symbol
            rows.append({
                'Symbol':       clean,
                'Trades':       m['total_trades'],
                'Win Rate':     f"{m['win_rate']:.1f}%",
                'Total P&L':    f"₹{m['total_pnl']:.2f}",
                'Avg P&L':      f"₹{m['avg_pnl']:.2f}",
                'Profit Factor':f"{m['profit_factor']:.2f}",
                'TP / SL / SQ': f"{m['tp_count']} / {m['sl_count']} / {m['sq_off_count']}",
                'Best':         f"₹{m['best_trade']:.2f}",
                'Worst':        f"₹{m['worst_trade']:.2f}",
                'Avg Dur (m)':  f"{m['avg_duration']:.1f}"
            })

        summary_df = pd.DataFrame(rows)
        print(summary_df.to_string(index=False))

        print(f"\n{'='*100}")
        print("OVERALL STATISTICS")
        print(f"{'='*100}")

        all_trades = [t for r in self.results.values() for t in r['trades']]
        total_trades = len(all_trades)
        total_pnl = sum(t['pnl'] for t in all_trades)
        winning = sum(1 for t in all_trades if t['pnl'] > 0)
        profitable_syms = sum(1 for r in self.results.values() if r['metrics']['total_pnl'] > 0)
        tp_total = sum(r['metrics']['tp_count'] for r in self.results.values())
        sl_total = sum(r['metrics']['sl_count'] for r in self.results.values())
        sq_total = sum(r['metrics']['sq_off_count'] for r in self.results.values())

        print(f"  Symbols Tested   : {len(self.results)}")
        print(f"  Profitable Syms  : {profitable_syms}/{len(self.results)}")
        print(f"  Total Trades     : {total_trades}")
        if total_trades:
            print(f"  Overall Win Rate : {winning / total_trades * 100:.1f}%")
        print(f"  Total P&L        : ₹{total_pnl:.2f}")
        print(f"  Exit Breakdown   : TP={tp_total}  SL={sl_total}  EOD={sq_total}")

        output_file = 'long_only_grid_results.csv'
        summary_df.to_csv(output_file, index=False)
        print(f"\n  Results saved to: {output_file}")


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("LONG-ONLY GRID STRATEGY (WUNDERTRADING) — INTRADAY BACKTEST")
    print("=" * 100)

    FYERS_CLIENT_ID = os.environ.get('FYERS_CLIENT_ID')
    FYERS_ACCESS_TOKEN = os.environ.get('FYERS_ACCESS_TOKEN')

    if not FYERS_CLIENT_ID or not FYERS_ACCESS_TOKEN:
        print("\nERROR: Missing Fyers API credentials!")
        print("  Set FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN in your .env file.")
        print("  Run: python main.py auth   to generate an access token.")
        exit(1)

    # ----------------------------------------------------------------
    # Symbol universe
    # ----------------------------------------------------------------
    SYMBOLS = [
        "NSE:SBIN-EQ",
        "NSE:RELIANCE-EQ",
        "NSE:TCS-EQ",
        "NSE:INFY-EQ",
        "NSE:HDFCBANK-EQ",
        "NSE:STLTECH-EQ",
        "NSE:SKYGOLD-EQ",
        "NSE:AXISCADES-EQ",
    ]

    # ----------------------------------------------------------------
    # Strategy configuration
    # ----------------------------------------------------------------
    backtester = LongOnlyGridBacktester(
        fyers_client_id=FYERS_CLIENT_ID,
        fyers_access_token=FYERS_ACCESS_TOKEN,
        symbols=SYMBOLS,

        # Grid structure
        grid_levels=10,           # 10 price levels below anchor
        grid_spacing_pct=1.0,     # 1% apart (use use_atr_spacing=True for dynamic)
        use_atr_spacing=False,
        atr_period=14,
        atr_spacing_mult=1.0,
        grid_anchor='open',       # Anchor grid to each day's opening price

        # Position sizing & risk
        position_size_pct=10.0,   # Deploy 10% of capital per level
        stop_loss_levels=3,       # Exit all if price falls 3 levels below first buy
        max_grid_positions=10,    # Never hold more than 10 concurrent positions

        # Capital & timing
        initial_capital=100000,
        square_off_time="15:20",
        tick_interval='5S',        # 5-minute candles
        backtest_days=30,
    )

    backtester.run_backtest()
