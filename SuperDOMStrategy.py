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
"""

import argparse
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

load_dotenv()


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


def main():
    parser = argparse.ArgumentParser(description="Simple Super DOM strategy")
    parser.add_argument("--symbol", default="NSE:SBIN-EQ")
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--stop-ticks", type=int, default=5)
    parser.add_argument("--target-ticks", type=int, default=10)
    parser.add_argument("--tick-size", type=float, default=0.05)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--runtime", type=int, default=None,
                        help="Max runtime in seconds (default: run until Ctrl+C)")
    args = parser.parse_args()

    client_id = os.environ.get("FYERS_CLIENT_ID", "").strip()
    access_token = os.environ.get("FYERS_ACCESS_TOKEN", "").strip()
    if not client_id or not access_token:
        raise SystemExit("Set FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN in .env (run `python main.py auth`).")

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
