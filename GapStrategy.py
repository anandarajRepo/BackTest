import pandas as pd
import numpy as np
import os
from datetime import datetime, time, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings('ignore')


class GapStrategyBacktester:
    """
    Gap Up / Gap Down Strategy Backtester with Historical Behaviour Analysis

    STRATEGY CONCEPT:
    -----------------
    This strategy trades gaps at market open (9:15 AM IST) but ONLY after
    analysing how the specific stock has historically reacted to similar gaps
    over the past 30 trading days.

    PHASE 1 — Historical Gap Behaviour Analysis (30-Day Lookback):
    ---------------------------------------------------------------
    For each trading day being backtested, look back 30 calendar days and:
      1. Identify all gap-up and gap-down events (open vs previous close).
      2. For each gap event, measure whether price CONTINUED in the gap
         direction (continuation) or REVERSED back (gap fill) by end of day.
         - Gap Up  + day closed above open  → CONTINUATION
         - Gap Up  + day closed below open  → REVERSAL
         - Gap Down + day closed below open → CONTINUATION
         - Gap Down + day closed above open → REVERSAL
      3. Calculate a continuation win-rate and a reversal win-rate.
      4. Select the dominant behaviour (must exceed `continuation_threshold`
         or `reversal_threshold`; otherwise skip the day — no trade).

    PHASE 2 — Trade Entry:
    ----------------------
    On the current day, if a qualifying gap is detected (between `gap_threshold_pct`
    and `max_gap_pct`):
      - Historical bias = CONTINUATION → trade IN the gap direction
          Gap Up  → BUY  (long)
          Gap Down → SELL (short)
      - Historical bias = REVERSAL → trade AGAINST the gap direction
          Gap Up  → SELL (short)
          Gap Down → BUY  (long)
    Entry is taken within `entry_candles` candles after open on `candle_interval`
    candles.

    PHASE 3 — Exit with Trailing Stop:
    -----------------------------------
    - Initial stop loss:   ATR × `initial_stop_atr_mult`
    - Trailing stop loss:  ATR × `trailing_stop_atr_mult` (ratchets with price)
    - Profit target:       ATR × `target_atr_mult`
    - Hard square-off:     `square_off_time` (default 15:20 IST)
    - Max 1 trade per day per symbol.
    """

    def __init__(
        self,
        fyers_client_id,
        fyers_access_token,
        symbols=None,
        # Gap detection
        gap_threshold_pct=0.3,
        max_gap_pct=5.0,
        # Historical behaviour analysis
        behavior_lookback_days=30,
        min_gap_history=5,
        continuation_threshold=0.55,
        reversal_threshold=0.55,
        # Entry timing
        entry_candles=3,
        candle_interval="5min",
        last_entry_time="09:45",
        # Risk management
        atr_period=14,
        trailing_stop_atr_mult=1.5,
        initial_stop_atr_mult=2.0,
        target_atr_mult=3.0,
        # Capital / session
        initial_capital=100000,
        square_off_time="15:20",
        min_data_points=100,
        max_trades_per_day=1,
        # Backtest window
        backtest_days=30,
    ):
        # Initialize Fyers client
        self.fyers = fyersModel.FyersModel(client_id=fyers_client_id, token=fyers_access_token, is_async=False, log_path="")

        # Date range: fetch enough history for behaviour lookback + backtest window
        self.backtest_days = backtest_days
        ist_tz = pytz.timezone('Asia/Kolkata')
        end_date = datetime.now(ist_tz)
        total_days = backtest_days + behavior_lookback_days + 30  # 30-day buffer
        start_date = end_date - timedelta(days=total_days)
        self.range_from = int(start_date.timestamp())
        self.range_to = int(end_date.timestamp())

        # Derive Fyers API resolution from candle_interval (e.g. "5min" → "5")
        self.fyers_resolution = candle_interval.replace('min', '').replace('T', '').strip()

        # Gap parameters
        self.gap_threshold_pct = gap_threshold_pct / 100
        self.max_gap_pct = max_gap_pct / 100

        # Behaviour analysis parameters
        self.behavior_lookback_days = behavior_lookback_days
        self.min_gap_history = min_gap_history
        self.continuation_threshold = continuation_threshold
        self.reversal_threshold = reversal_threshold

        # Entry parameters
        self.entry_candles = entry_candles
        self.candle_interval = candle_interval
        self.last_entry_time = self._parse_time(last_entry_time)

        # Risk parameters
        self.atr_period = atr_period
        self.trailing_stop_atr_mult = trailing_stop_atr_mult
        self.initial_stop_atr_mult = initial_stop_atr_mult
        self.target_atr_mult = target_atr_mult

        # Session parameters
        self.initial_capital = initial_capital
        self.square_off_time = self._parse_time(square_off_time)
        self.min_data_points = min_data_points
        self.max_trades_per_day = max_trades_per_day
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.results = {}

        print("=" * 70)
        print("     GAP UP / GAP DOWN STRATEGY BACKTESTER")
        print("=" * 70)
        print(f"  Data source              : Fyers API")
        print(f"  Backtest period          : last {backtest_days} trading days")
        print(f"  Date range               : {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"  Gap threshold            : {gap_threshold_pct}% – {max_gap_pct}%")
        print(f"  Behaviour lookback       : {behavior_lookback_days} days")
        print(f"  Min gap history required : {min_gap_history} events")
        print(f"  Continuation threshold   : {continuation_threshold * 100:.0f}%")
        print(f"  Reversal threshold       : {reversal_threshold * 100:.0f}%")
        print(f"  Entry window             : first {entry_candles} × {candle_interval} candles")
        print(f"  Last entry time          : {last_entry_time} IST")
        print(f"  ATR period               : {atr_period}")
        print(f"  Initial stop (ATR×)      : {initial_stop_atr_mult}")
        print(f"  Trailing stop (ATR×)     : {trailing_stop_atr_mult}")
        print(f"  Profit target (ATR×)     : {target_atr_mult}")
        print(f"  Initial capital          : ₹{initial_capital:,}")
        print(f"  Square-off time          : {square_off_time} IST")
        print(f"  Max trades per day       : {max_trades_per_day}")
        print()

        if symbols is None:
            self.symbols = ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ", "NSE:TCS-EQ"]
            print(f"Using default symbols: {self.symbols}")
        else:
            self.symbols = symbols

        print(f"\nSymbols selected for backtest: {len(self.symbols)}")
        for s in self.symbols:
            print(f"  - {s}")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _parse_time(self, time_str):
        try:
            h, m = map(int, time_str.split(":"))
            return time(h, m)
        except Exception:
            print(f"  WARNING: Invalid time '{time_str}', using 15:20 as fallback.")
            return time(15, 20)

    # ------------------------------------------------------------------
    # Data loading — Fyers API
    # ------------------------------------------------------------------

    def _load_data_from_fyers(self, symbol):
        """Load historical OHLCV candles from Fyers API."""
        try:
            print(f"  Fetching data from Fyers for {symbol}...")
            data = {
                "symbol": symbol,
                "resolution": self.fyers_resolution,
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
            df = df.dropna(subset=['open', 'close', 'high', 'low'])

            print(f"  Loaded {len(df)} candles from {df.index[0].date()} to {df.index[-1].date()}")
            return df[['open', 'high', 'low', 'close', 'volume']].copy()

        except Exception as e:
            print(f"  Error loading data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _to_daily_candles(self, df):
        """Build daily OHLCV candles from intraday OHLCV, restricted to NSE session."""
        session = df.between_time('09:15', '15:30')
        if session.empty:
            session = df
        agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        daily = session.resample('D').agg(agg)
        return daily.dropna(subset=['open', 'close'])

    # ------------------------------------------------------------------
    # ATR calculation
    # ------------------------------------------------------------------

    def _calc_atr(self, candles):
        high  = candles['high']
        low   = candles['low']
        close = candles['close']
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(self.atr_period, min_periods=1).mean()

    # ------------------------------------------------------------------
    # PHASE 1 — Historical gap behaviour analysis
    # ------------------------------------------------------------------

    def _analyze_gap_behavior(self, daily_candles, analysis_date):
        """
        Look back `behavior_lookback_days` days from `analysis_date` and
        compute gap-up and gap-down continuation/reversal rates.

        Returns
        -------
        dict with keys:
            gap_up_continuation_rate   : float 0-1
            gap_up_reversal_rate       : float 0-1
            gap_up_count               : int
            gap_down_continuation_rate : float 0-1
            gap_down_reversal_rate     : float 0-1
            gap_down_count             : int
        """
        cutoff_start = pd.Timestamp(analysis_date) - pd.Timedelta(days=self.behavior_lookback_days)
        cutoff_end   = pd.Timestamp(analysis_date) - pd.Timedelta(days=1)

        # Only dates in the lookback window, BEFORE analysis_date (no lookahead)
        historical = daily_candles[
            (daily_candles.index >= cutoff_start) &
            (daily_candles.index <  pd.Timestamp(analysis_date))
        ].copy()

        if len(historical) < 2:
            return None

        historical['prev_close'] = historical['close'].shift(1)
        historical = historical.dropna(subset=['prev_close'])

        historical['gap_pct'] = (historical['open'] - historical['prev_close']) / historical['prev_close']

        # Classify each row
        historical['gap_type'] = np.where(
            historical['gap_pct'] >= self.gap_threshold_pct, 'UP',
            np.where(historical['gap_pct'] <= -self.gap_threshold_pct, 'DOWN', 'NONE')
        )
        # Exclude extreme gaps
        historical.loc[historical['gap_pct'].abs() > self.max_gap_pct, 'gap_type'] = 'NONE'

        # Behaviour: did price close above or below open?
        # continuation for gap-up  → close > open
        # continuation for gap-down → close < open
        historical['gap_up_continuation']   = (historical['gap_type'] == 'UP')   & (historical['close'] > historical['open'])
        historical['gap_up_reversal']        = (historical['gap_type'] == 'UP')   & (historical['close'] < historical['open'])
        historical['gap_down_continuation']  = (historical['gap_type'] == 'DOWN') & (historical['close'] < historical['open'])
        historical['gap_down_reversal']      = (historical['gap_type'] == 'DOWN') & (historical['close'] > historical['open'])

        gap_up_rows   = historical[historical['gap_type'] == 'UP']
        gap_down_rows = historical[historical['gap_type'] == 'DOWN']

        def _rate(mask_series, total_df):
            if len(total_df) == 0:
                return 0.0
            return mask_series.sum() / len(total_df)

        result = {
            'gap_up_count'               : len(gap_up_rows),
            'gap_up_continuation_rate'   : _rate(gap_up_rows['gap_up_continuation'],   gap_up_rows),
            'gap_up_reversal_rate'        : _rate(gap_up_rows['gap_up_reversal'],        gap_up_rows),
            'gap_down_count'             : len(gap_down_rows),
            'gap_down_continuation_rate' : _rate(gap_down_rows['gap_down_continuation'], gap_down_rows),
            'gap_down_reversal_rate'      : _rate(gap_down_rows['gap_down_reversal'],     gap_down_rows),
        }
        return result

    def _decide_direction(self, gap_type, stats):
        """
        Given current day gap_type ('UP' or 'DOWN') and historical stats,
        return 'LONG', 'SHORT', or None (no trade).

        Logic
        -----
        Gap UP:
          - If continuation_rate > threshold → stock tends to keep running → LONG
          - If reversal_rate     > threshold → stock tends to fill gap      → SHORT
          - Otherwise no trade (ambiguous)

        Gap DOWN:
          - If continuation_rate > threshold → stock tends to keep falling  → SHORT
          - If reversal_rate     > threshold → stock tends to recover        → LONG
          - Otherwise no trade (ambiguous)
        """
        if stats is None:
            return None

        if gap_type == 'UP':
            count = stats['gap_up_count']
            if count < self.min_gap_history:
                return None
            cont = stats['gap_up_continuation_rate']
            rev  = stats['gap_up_reversal_rate']
            if cont >= self.continuation_threshold and cont > rev:
                return 'LONG'
            if rev >= self.reversal_threshold and rev > cont:
                return 'SHORT'
            return None

        if gap_type == 'DOWN':
            count = stats['gap_down_count']
            if count < self.min_gap_history:
                return None
            cont = stats['gap_down_continuation_rate']
            rev  = stats['gap_down_reversal_rate']
            if cont >= self.continuation_threshold and cont > rev:
                return 'SHORT'
            if rev >= self.reversal_threshold and rev > cont:
                return 'LONG'
            return None

        return None

    # ------------------------------------------------------------------
    # PHASE 2 & 3 — Intraday backtest for a single symbol
    # ------------------------------------------------------------------

    def backtest_single_symbol(self, symbol):
        print(f"\n{'='*60}")
        print(f"  Backtesting: {symbol}")
        print(f"{'='*60}")

        # Load data from Fyers
        intra = self._load_data_from_fyers(symbol)
        if intra is None or len(intra) < self.min_data_points:
            print(f"  Insufficient data for {symbol}. Skipping.")
            return None

        # Build daily candles from intraday data
        daily = self._to_daily_candles(intra)
        intra = intra.between_time('09:15', '15:30')

        if daily.empty or len(daily) < self.behavior_lookback_days + 2:
            print(f"  Not enough daily data for {symbol}. Skipping.")
            return None

        daily['atr'] = self._calc_atr(daily)
        intra['atr'] = self._calc_atr(intra)

        # Add previous close to daily
        daily['prev_close'] = daily['close'].shift(1)
        daily = daily.dropna(subset=['prev_close'])

        # All unique trading dates in intraday candles — restrict to last N days
        trading_dates = sorted(set(intra.index.date))
        if self.backtest_days and len(trading_dates) > self.backtest_days:
            trading_dates = trading_dates[-self.backtest_days:]
        print(f"  Backtesting {len(trading_dates)} day(s): {trading_dates[0]} → {trading_dates[-1]}"
              if trading_dates else "  No trading dates found.")

        # ----------------------------------------------------------------
        # Back-test loop
        # ----------------------------------------------------------------
        cash        = self.initial_capital
        trades      = []
        portfolio   = []
        position    = None

        for tdate in trading_dates:
            ts_date = pd.Timestamp(tdate)

            # Daily row for this date
            if ts_date not in daily.index:
                continue
            day_row    = daily.loc[ts_date]
            day_open   = day_row['open']
            prev_close = day_row['prev_close']

            if prev_close == 0 or np.isnan(prev_close) or np.isnan(day_open):
                continue

            # Detect today's gap
            gap_pct = (day_open - prev_close) / prev_close

            if abs(gap_pct) < self.gap_threshold_pct or abs(gap_pct) > self.max_gap_pct:
                continue  # No qualifying gap today

            gap_type = 'UP' if gap_pct > 0 else 'DOWN'

            # Phase 1: Historical behaviour analysis
            stats = self._analyze_gap_behavior(daily, tdate)
            direction = self._decide_direction(gap_type, stats)

            if direction is None:
                # Ambiguous history or insufficient history → skip
                continue

            # Intraday candles for this date
            day_candles = intra[intra.index.date == tdate].copy()
            if day_candles.empty:
                continue

            # Get ATR from most recent intraday candle of prior day or daily ATR
            atr_val = day_row['atr']
            if np.isnan(atr_val) or atr_val == 0:
                continue

            # ---- Entry ----
            trades_today = 0
            position     = None   # None | dict
            entry_candle_count = 0

            for idx, row in day_candles.iterrows():
                candle_time = idx.time()
                is_sqoff    = candle_time >= self.square_off_time

                # ---- Mandatory square-off ----
                if position is not None and is_sqoff:
                    exit_price = row['close']
                    pnl = self._calc_pnl(position, exit_price)
                    cash += position['cash_committed'] + pnl
                    trades.append(self._make_trade_record(
                        position, exit_price, idx, 'SQUARE_OFF', pnl, cash
                    ))
                    position = None
                    break

                # ---- Manage open position ----
                if position is not None:
                    price = row['close']

                    # Update trailing stop
                    if position['direction'] == 'LONG':
                        new_trail = price - self.trailing_stop_atr_mult * atr_val
                        position['trailing_stop'] = max(position['trailing_stop'], new_trail)

                        # Check stop
                        if price <= position['trailing_stop'] or price <= position['initial_stop']:
                            pnl = self._calc_pnl(position, price)
                            cash += position['cash_committed'] + pnl
                            reason = 'INITIAL_STOP' if price <= position['initial_stop'] else 'TRAILING_STOP'
                            trades.append(self._make_trade_record(
                                position, price, idx, reason, pnl, cash
                            ))
                            position = None
                            continue

                        # Check target
                        if price >= position['target']:
                            pnl = self._calc_pnl(position, price)
                            cash += position['cash_committed'] + pnl
                            trades.append(self._make_trade_record(
                                position, price, idx, 'TARGET_HIT', pnl, cash
                            ))
                            position = None
                            continue

                    else:  # SHORT
                        new_trail = price + self.trailing_stop_atr_mult * atr_val
                        position['trailing_stop'] = min(position['trailing_stop'], new_trail)

                        # Check stop
                        if price >= position['trailing_stop'] or price >= position['initial_stop']:
                            pnl = self._calc_pnl(position, price)
                            cash += position['cash_committed'] + pnl
                            reason = 'INITIAL_STOP' if price >= position['initial_stop'] else 'TRAILING_STOP'
                            trades.append(self._make_trade_record(
                                position, price, idx, reason, pnl, cash
                            ))
                            position = None
                            continue

                        # Check target
                        if price <= position['target']:
                            pnl = self._calc_pnl(position, price)
                            cash += position['cash_committed'] + pnl
                            trades.append(self._make_trade_record(
                                position, price, idx, 'TARGET_HIT', pnl, cash
                            ))
                            position = None
                            continue

                    portfolio.append({'date': idx, 'value': cash + position['cash_committed'] + self._calc_pnl(position, price) if position else cash})
                    continue

                # ---- Entry logic ----
                if (trades_today < self.max_trades_per_day
                        and candle_time <= self.last_entry_time
                        and entry_candle_count < self.entry_candles
                        and not is_sqoff):

                    entry_candle_count += 1
                    entry_price = row['close']

                    if direction == 'LONG':
                        initial_stop  = entry_price - self.initial_stop_atr_mult  * atr_val
                        trailing_stop = entry_price - self.trailing_stop_atr_mult * atr_val
                        target        = entry_price + self.target_atr_mult * atr_val
                    else:  # SHORT
                        initial_stop  = entry_price + self.initial_stop_atr_mult  * atr_val
                        trailing_stop = entry_price + self.trailing_stop_atr_mult * atr_val
                        target        = entry_price - self.target_atr_mult * atr_val

                    shares = max(1, int(self.initial_capital / entry_price))
                    cost   = shares * entry_price
                    cash  -= cost

                    position = {
                        'symbol'          : symbol,
                        'date'            : tdate,
                        'direction'       : direction,
                        'entry_time'      : idx,
                        'entry_price'     : entry_price,
                        'shares'          : shares,
                        'cash_committed'  : cost,
                        'initial_stop'    : initial_stop,
                        'trailing_stop'   : trailing_stop,
                        'target'          : target,
                        'gap_type'        : gap_type,
                        'gap_pct'         : gap_pct,
                        'gap_stats'       : stats,
                    }
                    trades_today += 1

                    print(
                        f"  {idx.strftime('%Y-%m-%d %H:%M')} | {direction:5s} {shares:4d} @ "
                        f"₹{entry_price:.2f} | Gap {gap_type} {gap_pct*100:+.2f}% | "
                        f"Stop ₹{initial_stop:.2f} | Target ₹{target:.2f} | "
                        f"Hist: cont={stats.get(f'gap_{gap_type.lower()}_continuation_rate', 0)*100:.0f}% "
                        f"rev={stats.get(f'gap_{gap_type.lower()}_reversal_rate', 0)*100:.0f}%"
                    )

                portfolio.append({'date': idx, 'value': cash})

        # Close any open position at last known price
        if position is not None:
            last_price = intra[intra.index.date == position['date']]['close'].iloc[-1]
            pnl = self._calc_pnl(position, last_price)
            cash += position['cash_committed'] + pnl
            trades.append(self._make_trade_record(
                position, last_price, position['entry_time'], 'SESSION_END', pnl, cash
            ))

        result = self._compute_summary(symbol, trades, cash)
        self.results[symbol] = result
        return result

    # ------------------------------------------------------------------
    # Helpers — P&L, trade records, summary
    # ------------------------------------------------------------------

    def _calc_pnl(self, position, exit_price):
        if position['direction'] == 'LONG':
            return position['shares'] * (exit_price - position['entry_price'])
        else:
            return position['shares'] * (position['entry_price'] - exit_price)

    def _make_trade_record(self, position, exit_price, exit_time, reason, pnl, cash_after):
        return {
            'symbol'          : position['symbol'],
            'trade_date'      : str(position['date']),
            'direction'       : position['direction'],
            'gap_type'        : position['gap_type'],
            'gap_pct'         : round(position['gap_pct'] * 100, 3),
            'entry_time'      : position['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'exit_time'       : exit_time.strftime('%Y-%m-%d %H:%M:%S'),
            'entry_price'     : round(position['entry_price'], 2),
            'exit_price'      : round(exit_price, 2),
            'shares'          : position['shares'],
            'pnl'             : round(pnl, 2),
            'return_pct'      : round(pnl / position['cash_committed'] * 100, 3),
            'exit_reason'     : reason,
            'cash_after'      : round(cash_after, 2),
            'hist_cont_rate'  : round(
                position['gap_stats'].get(
                    f"gap_{position['gap_type'].lower()}_continuation_rate", 0) * 100, 1),
            'hist_rev_rate'   : round(
                position['gap_stats'].get(
                    f"gap_{position['gap_type'].lower()}_reversal_rate", 0) * 100, 1),
            'hist_gap_count'  : position['gap_stats'].get(
                    f"gap_{position['gap_type'].lower()}_count", 0),
        }

    def _compute_summary(self, symbol, trades, final_cash):
        if not trades:
            return {
                'symbol': symbol, 'total_trades': 0, 'win_rate': 0,
                'net_pnl': 0, 'final_capital': final_cash,
            }

        pnl_list  = [t['pnl'] for t in trades]
        wins      = [p for p in pnl_list if p > 0]
        losses    = [p for p in pnl_list if p < 0]
        net_pnl   = sum(pnl_list)
        win_rate  = len(wins) / len(trades) * 100
        avg_win   = np.mean(wins) if wins else 0
        avg_loss  = np.mean(losses) if losses else 0
        pf        = abs(sum(wins) / sum(losses)) if losses else float('inf')

        exit_counts = {}
        for t in trades:
            r = t['exit_reason']
            exit_counts[r] = exit_counts.get(r, 0) + 1

        long_trades  = [t for t in trades if t['direction'] == 'LONG']
        short_trades = [t for t in trades if t['direction'] == 'SHORT']
        gapup_trades = [t for t in trades if t['gap_type'] == 'UP']
        gapdn_trades = [t for t in trades if t['gap_type'] == 'DOWN']

        return {
            'symbol'           : symbol,
            'total_trades'     : len(trades),
            'wins'             : len(wins),
            'losses'           : len(losses),
            'win_rate'         : round(win_rate, 2),
            'net_pnl'          : round(net_pnl, 2),
            'avg_win'          : round(avg_win, 2),
            'avg_loss'         : round(avg_loss, 2),
            'profit_factor'    : round(pf, 2),
            'best_trade'       : round(max(pnl_list), 2),
            'worst_trade'      : round(min(pnl_list), 2),
            'final_capital'    : round(final_cash, 2),
            'return_pct'       : round((final_cash - self.initial_capital) / self.initial_capital * 100, 2),
            'long_trades'      : len(long_trades),
            'short_trades'     : len(short_trades),
            'gap_up_trades'    : len(gapup_trades),
            'gap_down_trades'  : len(gapdn_trades),
            'exit_breakdown'   : exit_counts,
            'trade_log'        : trades,
        }

    # ------------------------------------------------------------------
    # Run all symbols
    # ------------------------------------------------------------------

    def run_backtest(self, max_workers=4):
        if not self.symbols:
            print("No symbols to backtest.")
            return {}

        print(f"\nRunning backtest for {len(self.symbols)} symbol(s) …\n")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.backtest_single_symbol, sym): sym
                for sym in self.symbols
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  ERROR backtesting {sym}: {e}")

        self.print_summary()
        self._save_results()
        return self.results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_summary(self):
        if not self.results:
            print("\nNo backtest results to display.")
            return

        print("\n")
        print("=" * 90)
        print("                     GAP STRATEGY BACKTEST SUMMARY")
        print("=" * 90)
        print(
            f"  {'Symbol':<25} {'Trades':>6} {'WinRate':>8} {'NetPnL':>10} "
            f"{'PF':>6} {'AvgWin':>9} {'AvgLoss':>9} {'Return%':>8}"
        )
        print("  " + "-" * 86)

        total_pnl    = 0
        total_trades = 0

        for sym, r in sorted(self.results.items()):
            if r['total_trades'] == 0:
                print(f"  {sym:<25} {'0':>6}  {'N/A':>8}  {'N/A':>10}")
                continue

            print(
                f"  {sym:<25} {r['total_trades']:>6} "
                f"{r['win_rate']:>7.1f}% "
                f"₹{r['net_pnl']:>9,.0f} "
                f"{r['profit_factor']:>6.2f} "
                f"₹{r['avg_win']:>8,.0f} "
                f"₹{r['avg_loss']:>8,.0f} "
                f"{r['return_pct']:>7.2f}%"
            )
            total_pnl    += r['net_pnl']
            total_trades += r['total_trades']

        print("  " + "-" * 86)
        print(f"  {'TOTAL':<25} {total_trades:>6}                ₹{total_pnl:>9,.0f}")
        print("=" * 90)

        # Per-symbol exit breakdown
        print("\n  Exit Reason Breakdown:")
        print("  " + "-" * 60)
        for sym, r in sorted(self.results.items()):
            if r['total_trades'] == 0:
                continue
            bd = r.get('exit_breakdown', {})
            bd_str = "  ".join(f"{k}:{v}" for k, v in bd.items())
            print(f"  {sym:<25}  {bd_str}")

        # Gap type breakdown
        print("\n  Gap Type Breakdown:")
        print("  " + "-" * 60)
        for sym, r in sorted(self.results.items()):
            if r['total_trades'] == 0:
                continue
            print(
                f"  {sym:<25}  GapUp:{r['gap_up_trades']}  GapDown:{r['gap_down_trades']}  "
                f"Long:{r['long_trades']}  Short:{r['short_trades']}"
            )
        print("=" * 90)

    def _save_results(self):
        os.makedirs("output", exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Flatten all trade logs into one CSV
        all_trades = []
        for r in self.results.values():
            all_trades.extend(r.get('trade_log', []))

        if all_trades:
            csv_path = os.path.join("output", f"gap_strategy_trades_{ts}.csv")
            pd.DataFrame(all_trades).to_csv(csv_path, index=False)
            print(f"\n  Trade log saved → {csv_path}")

        # Summary CSV
        summary_rows = []
        for r in self.results.values():
            row = {k: v for k, v in r.items() if k not in ('trade_log', 'exit_breakdown')}
            row['exit_breakdown'] = str(r.get('exit_breakdown', {}))
            summary_rows.append(row)

        if summary_rows:
            sum_path = os.path.join("output", f"gap_strategy_summary_{ts}.csv")
            pd.DataFrame(summary_rows).to_csv(sum_path, index=False)
            print(f"  Summary saved    → {sum_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    FYERS_CLIENT_ID = os.environ.get('FYERS_CLIENT_ID')
    FYERS_ACCESS_TOKEN = os.environ.get('FYERS_ACCESS_TOKEN')

    if not FYERS_CLIENT_ID or not FYERS_ACCESS_TOKEN:
        print("\nERROR: Missing Fyers API credentials!")
        print("  Set FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN in your .env file.")
        exit(1)

    SYMBOLS = [
        "NSE:SBIN-EQ",      # State Bank of India
        "NSE:RELIANCE-EQ",  # Reliance Industries
        "NSE:TCS-EQ",       # Tata Consultancy Services
        "NSE:INFY-EQ",      # Infosys
        "NSE:HDFCBANK-EQ",   # HDFC Bank

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
        "BSE:SATTRIX-M"

        "NSE:AWHCL-EQ",
        "NSE:KAPSTON-EQ",

        # "NSE:NIFTY2621025800CE",
        # "NSE:NIFTY2621025800PE",
        # "NSE:NIFTY2621025600CE",
        # "NSE:NIFTY2621025600PE",
        # "NSE:FINNIFTY26FEB27700CE",
        # "NSE:FINNIFTY26FEB27700PE",
        # "NSE:MIDCPNIFTY26FEB14000CE",
        # "NSE:MIDCPNIFTY26FEB14000PE",
        # "NSE:BANKNIFTY26FEB60000CE",
        # "NSE:BANKNIFTY26FEB60000PE"
    ]

    backtester = GapStrategyBacktester(
        fyers_client_id=FYERS_CLIENT_ID,
        fyers_access_token=FYERS_ACCESS_TOKEN,
        symbols=SYMBOLS,

        # Gap detection
        gap_threshold_pct=1.0,    # minimum 0.3% gap to qualify
        max_gap_pct=5.0,          # ignore gaps larger than 5% (news/earnings)

        # Historical behaviour analysis
        behavior_lookback_days=30,
        min_gap_history=5,         # need at least 5 prior gap events to decide
        continuation_threshold=0.55,
        reversal_threshold=0.55,

        # Entry
        entry_candles=12,           # enter within first 3 × 5-min candles after open
        candle_interval="5S",
        last_entry_time="09:45",   # no new entries after 9:45 AM

        # Risk
        atr_period=14,
        initial_stop_atr_mult=2.0,
        trailing_stop_atr_mult=1.5,
        target_atr_mult=3.0,

        # Session
        initial_capital=100000,
        square_off_time="15:20",
        max_trades_per_day=1,

        # Backtest window — last N trading days (mirrors OpenRangeBreakout.py)
        backtest_days=30,
    )

    results = backtester.run_backtest(max_workers=4)
