"""
Simple Super DOM (Depth of Market) Trading Strategy.

STRATEGY CONCEPT
----------------
"Super DOM" strategies read the price ladder / order book and trade on
imbalances between resting bid and ask liquidity. When buyers are clearly
stacking bids above sellers' offers, price tends to drift up, and vice versa.

This is a deliberately simple implementation:

  1. Poll Fyers market depth for a symbol on a fixed interval.
  2. Sum the top-N bid and top-N ask quantities.
  3. Compute the order-book imbalance:
         imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
     Range: -1 (all asks) .. +1 (all bids).
  4. Signals:
         imbalance >  +threshold  -> LONG
         imbalance <  -threshold  -> SHORT
  5. Exits: fixed tick-based stop loss / target, or opposite imbalance flip.

Run live paper mode:
    python SuperDOMStrategy.py --symbol "NSE:SBIN-EQ"

Run backtest mode:
    python SuperDOMStrategy.py --backtest
"""

import argparse
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from datetime import time as dtime

import pandas as pd
import numpy as np
import pytz

from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

load_dotenv()
warnings.filterwarnings('ignore')


@dataclass
class Position:
    side: str           # "LONG" or "SHORT"
    entry_price: float
    stop_price: float
    target_price: float
    entered_at: datetime = field(default_factory=datetime.now)


class SuperDOMStrategy:
    def __init__(self, fyers_client_id, fyers_access_token, symbol,
                 depth_levels=5, imbalance_threshold=0.35,
                 stop_ticks=5, target_ticks=10, tick_size=0.05,
                 poll_interval_sec=1.0, max_runtime_sec=None):
        self.fyers = fyersModel.FyersModel(
            client_id=fyers_client_id,
            token=fyers_access_token,
            is_async=False,
            log_path="",
        )
        self.symbol = symbol
        self.depth_levels = depth_levels
        self.imbalance_threshold = imbalance_threshold
        self.stop_ticks = stop_ticks
        self.target_ticks = target_ticks
        self.tick_size = tick_size
        self.poll_interval_sec = poll_interval_sec
        self.max_runtime_sec = max_runtime_sec

        self.position: Position | None = None
        self.trades: list[dict] = []

    def fetch_depth(self):
        """Return (bids, asks, ltp) where bids/asks are lists of {price, volume}."""
        response = self.fyers.depth({"symbol": self.symbol, "ohlcv_flag": "1"})
        if not isinstance(response, dict) or response.get("s") != "ok":
            return None, None, None

        data = response.get("d", {}).get(self.symbol, {})
        bids = data.get("bids", [])[: self.depth_levels]
        asks = data.get("ask", [])[: self.depth_levels]
        ltp = data.get("ltp") or data.get("lp")
        return bids, asks, ltp

    @staticmethod
    def compute_imbalance(bids, asks):
        bid_qty = sum(b.get("volume", 0) for b in bids)
        ask_qty = sum(a.get("volume", 0) for a in asks)
        total = bid_qty + ask_qty
        if total == 0:
            return 0.0, bid_qty, ask_qty
        return (bid_qty - ask_qty) / total, bid_qty, ask_qty

    def enter_long(self, price):
        self.position = Position(
            side="LONG",
            entry_price=price,
            stop_price=price - self.stop_ticks * self.tick_size,
            target_price=price + self.target_ticks * self.tick_size,
        )
        print(f"[{datetime.now():%H:%M:%S}] LONG  entry @ {price:.2f} "
              f"SL {self.position.stop_price:.2f} TP {self.position.target_price:.2f}")

    def enter_short(self, price):
        self.position = Position(
            side="SHORT",
            entry_price=price,
            stop_price=price + self.stop_ticks * self.tick_size,
            target_price=price - self.target_ticks * self.tick_size,
        )
        print(f"[{datetime.now():%H:%M:%S}] SHORT entry @ {price:.2f} "
              f"SL {self.position.stop_price:.2f} TP {self.position.target_price:.2f}")

    def exit_position(self, price, reason):
        pos = self.position
        pnl = (price - pos.entry_price) if pos.side == "LONG" else (pos.entry_price - price)
        self.trades.append({
            "side": pos.side,
            "entry": pos.entry_price,
            "exit": price,
            "pnl": pnl,
            "reason": reason,
            "entered_at": pos.entered_at,
            "exited_at": datetime.now(),
        })
        print(f"[{datetime.now():%H:%M:%S}] EXIT  {pos.side} @ {price:.2f} "
              f"PnL {pnl:+.2f} ({reason})")
        self.position = None

    def check_exit(self, ltp, imbalance):
        pos = self.position
        if pos.side == "LONG":
            if ltp <= pos.stop_price:
                self.exit_position(ltp, "stop")
            elif ltp >= pos.target_price:
                self.exit_position(ltp, "target")
            elif imbalance < -self.imbalance_threshold:
                self.exit_position(ltp, "imbalance_flip")
        else:
            if ltp >= pos.stop_price:
                self.exit_position(ltp, "stop")
            elif ltp <= pos.target_price:
                self.exit_position(ltp, "target")
            elif imbalance > self.imbalance_threshold:
                self.exit_position(ltp, "imbalance_flip")

    def step(self):
        bids, asks, ltp = self.fetch_depth()
        if not bids or not asks or ltp is None:
            return

        imbalance, bid_qty, ask_qty = self.compute_imbalance(bids, asks)
        print(f"[{datetime.now():%H:%M:%S}] LTP {ltp:.2f} "
              f"bids={bid_qty} asks={ask_qty} imbalance={imbalance:+.2f}")

        if self.position is not None:
            self.check_exit(ltp, imbalance)
            return

        if imbalance > self.imbalance_threshold:
            self.enter_long(ltp)
        elif imbalance < -self.imbalance_threshold:
            self.enter_short(ltp)

    def run(self):
        print(f"Super DOM strategy running on {self.symbol} "
              f"(threshold {self.imbalance_threshold}, "
              f"SL {self.stop_ticks}t / TP {self.target_ticks}t)")
        started = time.time()
        try:
            while True:
                self.step()
                if self.max_runtime_sec and (time.time() - started) >= self.max_runtime_sec:
                    break
                time.sleep(self.poll_interval_sec)
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            self.summary()

    def summary(self):
        if not self.trades:
            print("No trades taken.")
            return
        total_pnl = sum(t["pnl"] for t in self.trades)
        wins = sum(1 for t in self.trades if t["pnl"] > 0)
        print(f"\nTrades: {len(self.trades)}  Wins: {wins}  "
              f"Total PnL: {total_pnl:+.2f}")


class SuperDOMBacktester:
    """
    Backtest the Super DOM imbalance strategy using Fyers historical OHLCV data.

    Since historical order-book snapshots are unavailable via the Fyers API, the
    per-bar DOM imbalance is proxied from OHLCV data using the close-location value:

        imbalance = ((close - low) / (high - low)) * 2 - 1

    This ranges from -1 (close at low → sellers dominated) to +1 (close at high →
    buyers dominated), mirroring the live imbalance formula.  A rolling smoothing
    window (imbalance_smooth_periods) reduces bar-to-bar noise.
    """

    def __init__(self, fyers_client_id, fyers_access_token, symbols=None,
                 imbalance_threshold=0.35, imbalance_smooth_periods=3,
                 stop_ticks=5, target_ticks=10, tick_size=0.05,
                 initial_capital=100000, square_off_time="15:20",
                 tick_interval='5', backtest_days=7,
                 max_trades_per_day=5, last_entry_time="14:30",
                 min_data_points=100):
        self.fyers = fyersModel.FyersModel(
            client_id=fyers_client_id,
            token=fyers_access_token,
            is_async=False,
            log_path="",
        )

        # Date range
        self.backtest_days = backtest_days
        ist_tz = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist_tz)
        start_date = end_date - timedelta(days=backtest_days)
        self.range_from = int(start_date.timestamp())
        self.range_to = int(end_date.timestamp())
        self.ist_tz = ist_tz

        # Strategy parameters
        self.imbalance_threshold = imbalance_threshold
        self.imbalance_smooth_periods = imbalance_smooth_periods
        self.stop_ticks = stop_ticks
        self.target_ticks = target_ticks
        self.tick_size = tick_size

        # Portfolio / risk parameters
        self.initial_capital = initial_capital
        self.tick_interval = tick_interval
        self.min_data_points = min_data_points
        self.max_trades_per_day = max_trades_per_day
        self.square_off_time = self._parse_time(square_off_time)
        self.last_entry_time = self._parse_time(last_entry_time)

        self.results = {}
        self.combined_data = {}

        if symbols is None:
            self.symbols = ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ", "NSE:TCS-EQ"]
            print(f"\nUsing default symbols: {self.symbols}")
        else:
            self.symbols = symbols

        print(f"{'='*100}")
        print("SUPER DOM STRATEGY - INTRADAY BACKTEST (FYERS DATA)")
        print(f"{'='*100}")
        print(f"Data Source      : Fyers API")
        print(f"Backtest Period  : Last {backtest_days} days")
        print(f"Date Range       : {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Tick Interval    : {tick_interval} seconds")
        print(f"Imbalance Proxy  : Close-Location Value (OHLCV)")
        print(f"Imbalance Smooth : {imbalance_smooth_periods} bars")
        print(f"Threshold        : ±{imbalance_threshold}")
        print(f"Stop             : {stop_ticks} ticks  ({stop_ticks * tick_size:.4f})")
        print(f"Target           : {target_ticks} ticks ({target_ticks * tick_size:.4f})")
        print(f"Last Entry       : {last_entry_time}")
        print(f"Square-off       : {square_off_time}")
        print(f"Max Trades/Day   : {max_trades_per_day}")
        print(f"Initial Capital  : ₹{initial_capital:,}")
        print(f"Symbols          : {len(self.symbols)}")
        print(f"{'='*100}")

    @staticmethod
    def _parse_time(time_str):
        try:
            h, m = map(int, time_str.split(':'))
            return dtime(h, m)
        except Exception:
            return dtime(15, 20)

    def load_data_from_fyers(self, symbol):
        """Load historical OHLCV data from Fyers API (identical to OpenRangeBreakout)."""
        try:
            print(f"  Fetching data from Fyers for {symbol}...")

            data = {
                "symbol": symbol,
                "resolution": self.tick_interval,
                "date_format": "0",
                "range_from": str(self.range_from),
                "range_to": str(self.range_to),
                "cont_flag": "1",
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

    @staticmethod
    def compute_imbalance_from_ohlcv(df, smooth_periods=3):
        """
        Proxy DOM imbalance from OHLCV using the close-location value:
            raw_imbalance = ((close - low) / (high - low)) * 2 - 1
        Smoothed over `smooth_periods` bars to reduce noise.
        """
        bar_range = df['high'] - df['low']
        raw = np.where(
            bar_range > 0,
            ((df['close'] - df['low']) / bar_range) * 2 - 1,
            0.0
        )
        df['raw_imbalance'] = raw
        df['imbalance'] = (
            df['raw_imbalance']
            .rolling(window=smooth_periods, min_periods=1)
            .mean()
        )
        return df

    def backtest_single_symbol(self, symbol):
        """Run Super DOM backtest for one symbol."""
        print(f"\n{'='*100}")
        print(f"Backtesting: {symbol}")
        print(f"{'='*100}")

        df = self.load_data_from_fyers(symbol)
        if df is None or len(df) < self.min_data_points:
            print(f"  Insufficient data for {symbol}. Skipping.")
            return None

        self.combined_data[symbol] = df.copy()
        df = self.compute_imbalance_from_ohlcv(df, self.imbalance_smooth_periods)

        df['trading_day'] = df.index.date
        df['time_of_day'] = df.index.time

        cash = self.initial_capital
        position = 0          # 0=flat, 1=long, -1=short
        entry_price = 0.0
        stop_price = 0.0
        target_price = 0.0
        entry_time = None
        trades = []
        trade_number = 0
        trades_today: dict = {}

        stop_dist = self.stop_ticks * self.tick_size
        target_dist = self.target_ticks * self.tick_size

        for i in range(len(df)):
            row = df.iloc[i]
            current_time = df.index[i]
            ltp = row['close']
            high = row['high']
            low = row['low']
            imbalance = row['imbalance']
            current_day = row['trading_day']
            current_tod = row['time_of_day']

            trades_today.setdefault(current_day, 0)

            is_square_off = current_tod >= self.square_off_time

            # ── EOD square-off ──────────────────────────────────────────────
            if position != 0 and is_square_off:
                pnl = (ltp - entry_price) if position == 1 else (entry_price - ltp)
                shares = trades[-1]['shares']
                trade_pnl = shares * pnl
                trade_return = (trade_pnl / (shares * entry_price)) * 100
                trade_number += 1
                duration = (current_time - entry_time).total_seconds() / 60

                print(f"\n[Trade #{trade_number}] SQUARE-OFF @ {current_time.strftime('%H:%M:%S')}")
                print(f"  Direction: {'LONG' if position == 1 else 'SHORT'}")
                print(f"  Entry: {entry_time.strftime('%H:%M:%S')} @ ₹{entry_price:.2f}")
                print(f"  Exit: ₹{ltp:.2f} | P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
                print(f"  {'✅ PROFIT' if trade_pnl > 0 else '❌ LOSS'}")

                trades[-1].update({
                    'exit_time': current_time,
                    'exit_price': ltp,
                    'pnl': trade_pnl,
                    'return_pct': trade_return,
                    'exit_reason': 'SQUARE_OFF',
                    'duration_minutes': duration,
                })
                position = 0
                trades_today[current_day] += 1
                continue

            # ── Exit management ─────────────────────────────────────────────
            if position != 0 and not is_square_off:
                exit_signal = False
                exit_reason = ""
                exit_price_val = ltp

                if position == 1:
                    if low <= stop_price:
                        exit_signal, exit_reason, exit_price_val = True, "STOP_LOSS", stop_price
                    elif high >= target_price:
                        exit_signal, exit_reason, exit_price_val = True, "TARGET_HIT", target_price
                    elif imbalance < -self.imbalance_threshold:
                        exit_signal, exit_reason = True, "IMBALANCE_FLIP"
                else:
                    if high >= stop_price:
                        exit_signal, exit_reason, exit_price_val = True, "STOP_LOSS", stop_price
                    elif low <= target_price:
                        exit_signal, exit_reason, exit_price_val = True, "TARGET_HIT", target_price
                    elif imbalance > self.imbalance_threshold:
                        exit_signal, exit_reason = True, "IMBALANCE_FLIP"

                if exit_signal:
                    pnl = (exit_price_val - entry_price) if position == 1 else (entry_price - exit_price_val)
                    shares = trades[-1]['shares']
                    trade_pnl = shares * pnl
                    trade_return = (trade_pnl / (shares * entry_price)) * 100
                    trade_number += 1
                    duration = (current_time - entry_time).total_seconds() / 60

                    print(f"\n[Trade #{trade_number}] {exit_reason}")
                    print(f"  Direction: {'LONG' if position == 1 else 'SHORT'}")
                    print(f"  Entry: {entry_time.strftime('%H:%M:%S')} @ ₹{entry_price:.2f}")
                    print(f"  Exit: {current_time.strftime('%H:%M:%S')} @ ₹{exit_price_val:.2f}")
                    print(f"  Duration: {duration:.1f} min | P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
                    print(f"  {'✅ PROFIT' if trade_pnl > 0 else '❌ LOSS'}")

                    trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': exit_price_val,
                        'pnl': trade_pnl,
                        'return_pct': trade_return,
                        'exit_reason': exit_reason,
                        'duration_minutes': duration,
                    })
                    position = 0
                    trades_today[current_day] += 1
                continue

            # ── Entry signals ───────────────────────────────────────────────
            if (position == 0
                    and not is_square_off
                    and current_tod < self.last_entry_time
                    and trades_today[current_day] < self.max_trades_per_day):

                if imbalance > self.imbalance_threshold:
                    direction = 1
                    sl = ltp - stop_dist
                    tp = ltp + target_dist
                    label = "LONG"
                elif imbalance < -self.imbalance_threshold:
                    direction = -1
                    sl = ltp + stop_dist
                    tp = ltp - target_dist
                    label = "SHORT"
                else:
                    continue

                shares = int(cash / ltp)
                if shares <= 0:
                    continue

                position = direction
                entry_price = ltp
                stop_price = sl
                target_price = tp
                entry_time = current_time

                trades.append({
                    'trade_num': trade_number + 1,
                    'direction': label,
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'stop_loss': stop_price,
                    'target': target_price,
                    'shares': shares,
                    'imbalance_at_entry': imbalance,
                })

                print(f"\n[DOM {label} ENTRY] {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Price: ₹{entry_price:.2f} | Stop: ₹{stop_price:.2f} | Target: ₹{target_price:.2f}")
                print(f"  Imbalance: {imbalance:+.3f} | Shares: {shares}")

        completed = [t for t in trades if 'exit_time' in t]
        metrics = self.calculate_metrics(completed)

        return {'symbol': symbol, 'data': df, 'trades': completed, 'metrics': metrics}

    def calculate_metrics(self, trades):
        """Calculate performance metrics (same structure as OpenRangeBreakout)."""
        if not trades:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0,
                'total_return': 0, 'avg_return': 0, 'best_trade': 0,
                'worst_trade': 0, 'avg_duration': 0, 'avg_win': 0,
                'avg_loss': 0, 'profit_factor': 0,
            }

        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        losing_trades = total_trades - winning_trades

        total_pnl = sum(t['pnl'] for t in trades)
        avg_pnl = total_pnl / total_trades
        total_return = sum(t['return_pct'] for t in trades)
        avg_return = total_return / total_trades
        best_trade = max(t['pnl'] for t in trades)
        worst_trade = min(t['pnl'] for t in trades)
        avg_duration = sum(t['duration_minutes'] for t in trades) / total_trades

        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] <= 0]
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades * 100,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'total_return': total_return,
            'avg_return': avg_return,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_duration': avg_duration,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
        }

    def run_backtest(self):
        """Run backtest for all symbols."""
        print(f"\n{'='*100}")
        print("STARTING SUPER DOM BACKTEST")
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
        """Print summary table and save CSV (mirrors OpenRangeBreakout)."""
        if not self.results:
            print("\nNo results to display.")
            return

        print(f"\n{'='*100}")
        print("SUPER DOM STRATEGY RESULTS - ALL SYMBOLS")
        print(f"{'='*100}")

        summary_data = []
        for symbol, result in self.results.items():
            m = result['metrics']
            clean = symbol.split(':')[1].replace('-EQ', '') if ':' in symbol else symbol
            summary_data.append({
                'Symbol': clean,
                'Trades': m['total_trades'],
                'Win Rate': f"{m['win_rate']:.1f}%",
                'Total P&L': f"₹{m['total_pnl']:.2f}",
                'Avg P&L': f"₹{m['avg_pnl']:.2f}",
                'Profit Factor': f"{m['profit_factor']:.2f}",
                'Avg Duration': f"{m['avg_duration']:.1f}m",
                'Best Trade': f"₹{m['best_trade']:.2f}",
                'Worst Trade': f"₹{m['worst_trade']:.2f}",
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        print(f"\n{'='*100}")
        print("OVERALL STATISTICS")
        print(f"{'='*100}")

        total_trades = sum(r['metrics']['total_trades'] for r in self.results.values())
        total_pnl = sum(r['metrics']['total_pnl'] for r in self.results.values())
        winning_trades = sum(r['metrics']['winning_trades'] for r in self.results.values())
        profitable_symbols = sum(1 for r in self.results.values() if r['metrics']['total_pnl'] > 0)

        print(f"Symbols Tested     : {len(self.results)}")
        print(f"Total Trades       : {total_trades}")
        print(f"Total P&L          : ₹{total_pnl:.2f}")
        if total_trades > 0:
            print(f"Overall Win Rate   : {winning_trades / total_trades * 100:.1f}%")
        print(f"Profitable Symbols : {profitable_symbols}/{len(self.results)}")

        os.makedirs('output', exist_ok=True)
        summary_df.to_csv('output/superdom_backtest_results.csv', index=False)
        print(f"\n✅ Results saved to: output/superdom_backtest_results.csv")


# ---------------------------------------------------------------------------
# Symbols shared with OpenRangeBreakout strategy
# ---------------------------------------------------------------------------
SYMBOLS = [
    "NSE:SBIN-EQ",
    "NSE:RELIANCE-EQ",
    "NSE:TCS-EQ",
    "NSE:INFY-EQ",
    "NSE:HDFCBANK-EQ",

    "NSE:URBANCO-EQ",
    "NSE:AMANTA-EQ",
    "NSE:VIKRAMSOLR-EQ",
    "NSE:SHREEJISPG-EQ",
    "NSE:PATELRMART-EQ",
    "NSE:REGAAL-EQ",
    "NSE:HILINFRA-EQ",
    "NSE:SAATVIKGL-EQ",
    "NSE:ATLANTAELE-EQ",
    "NSE:STYL-EQ",
    "NSE:SOLARWORLD-EQ",
    "NSE:TRUALT-EQ",
    "NSE:ADVANCE-EQ",
    "NSE:LGEINDIA-EQ",
    "NSE:RUBICON-EQ",
    "NSE:MIDWESTLTD-EQ",
    "NSE:ORKLAINDIA-EQ",
    "NSE:LENSKART-EQ",
    "NSE:GROWW-EQ",
    "NSE:SUDEEPPHRM-EQ",
    "NSE:EXCELSOFT-EQ",
    "NSE:TENNIND-EQ",
    "NSE:MEESHO-EQ",
    "NSE:AEQUS-EQ",
    "NSE:CORONA-EQ",

    # Favourite Stocks
    "NSE:STLTECH-EQ",
    "NSE:SKYGOLD-EQ",
    "NSE:AXISCADES-EQ",
    "BSE:SATTRIX-M",
    "NSE:AWHCL-EQ",
    "NSE:KAPSTON-EQ",

    "NSE:NIFTY2621025800CE",
    "NSE:NIFTY2621025800PE",
    "NSE:NIFTY2621025600CE",
    "NSE:NIFTY2621025600PE",
    "NSE:FINNIFTY26FEB27700CE",
    "NSE:FINNIFTY26FEB27700PE",
    "NSE:MIDCPNIFTY26FEB14000CE",
    "NSE:MIDCPNIFTY26FEB14000PE",
    "NSE:BANKNIFTY26FEB60000CE",
    "NSE:BANKNIFTY26FEB60000PE",
]


def main():
    parser = argparse.ArgumentParser(description="Simple Super DOM strategy")
    parser.add_argument("--backtest", action="store_true",
                        help="Run in backtest mode instead of live paper trading")
    parser.add_argument("--symbol", default="NSE:SBIN-EQ")
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--stop-ticks", type=int, default=5)
    parser.add_argument("--target-ticks", type=int, default=10)
    parser.add_argument("--tick-size", type=float, default=0.05)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--runtime", type=int, default=None,
                        help="Max runtime in seconds (default: run until Ctrl+C)")
    # Backtest-specific args
    parser.add_argument("--backtest-days", type=int, default=30)
    parser.add_argument("--smooth", type=int, default=3,
                        help="Imbalance smoothing window (backtest only)")
    parser.add_argument("--max-trades-day", type=int, default=5)
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--tick-interval", default="5",
                        help="Fyers resolution string, e.g. '5' or '5S' (backtest only)")
    args = parser.parse_args()

    client_id = os.environ.get("FYERS_CLIENT_ID", "").strip()
    access_token = os.environ.get("FYERS_ACCESS_TOKEN", "").strip()
    if not client_id or not access_token:
        raise SystemExit("Set FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN in .env (run `python main.py auth`).")

    if args.backtest:
        backtester = SuperDOMBacktester(
            fyers_client_id=client_id,
            fyers_access_token=access_token,
            symbols=SYMBOLS,
            imbalance_threshold=args.threshold,
            imbalance_smooth_periods=args.smooth,
            stop_ticks=args.stop_ticks,
            target_ticks=args.target_ticks,
            tick_size=args.tick_size,
            initial_capital=args.capital,
            square_off_time="15:20",
            tick_interval=args.tick_interval,
            backtest_days=args.backtest_days,
            max_trades_per_day=args.max_trades_day,
            last_entry_time="14:30",
        )
        backtester.run_backtest()
    else:
        strat = SuperDOMStrategy(
            fyers_client_id=client_id,
            fyers_access_token=access_token,
            symbol=args.symbol,
            depth_levels=args.levels,
            imbalance_threshold=args.threshold,
            stop_ticks=args.stop_ticks,
            target_ticks=args.target_ticks,
            tick_size=args.tick_size,
            poll_interval_sec=args.interval,
            max_runtime_sec=args.runtime,
        )
        strat.run()


if __name__ == "__main__":
    main()
