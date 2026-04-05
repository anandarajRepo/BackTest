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
            # Gap calculation mode:
            #   "full"    — Full Gap only   (open beyond prev high/low)
            #   "partial" — Partial Gap only (open beyond prev close but within prev range)
            #   "both"    — Both Full and Partial gaps (default)
            gap_calculation_mode="both",
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
            # Stock performance filter thresholds
            min_win_rate=50.0,
            min_profit_factor=1.0,
            min_trades_required=3,
            max_loss_rate=60.0,
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

        # Gap calculation mode validation
        valid_modes = ("full", "partial", "both")
        if gap_calculation_mode not in valid_modes:
            raise ValueError(f"gap_calculation_mode must be one of {valid_modes}, got '{gap_calculation_mode}'")
        self.gap_calculation_mode = gap_calculation_mode

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

        # Stock filter thresholds
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor
        self.min_trades_required = min_trades_required
        self.max_loss_rate = max_loss_rate

        print("=" * 70)
        print("     GAP UP / GAP DOWN STRATEGY BACKTESTER")
        print("=" * 70)
        print(f"  Data source              : Fyers API")
        print(f"  Backtest period          : last {backtest_days} trading days")
        print(f"  Date range               : {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"  Gap threshold            : {gap_threshold_pct}% – {max_gap_pct}%")
        gap_mode_labels = {
            "full":    "Full Gap only  (open beyond prev high/low)",
            "partial": "Partial Gap only (open beyond prev close, within prev range)",
            "both":    "Both Full & Partial gaps",
        }
        print(f"  Gap calculation mode     : {gap_mode_labels[gap_calculation_mode]}")
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
        print(f"  Stock filter — min win rate     : {min_win_rate}%")
        print(f"  Stock filter — min profit factor: {min_profit_factor}")
        print(f"  Stock filter — min trades req.  : {min_trades_required}")
        print(f"  Stock filter — max loss rate    : {max_loss_rate}%")
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
        high = candles['high']
        low = candles['low']
        close = candles['close']
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
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
        cutoff_end = pd.Timestamp(analysis_date) - pd.Timedelta(days=1)

        # Only dates in the lookback window, BEFORE analysis_date (no lookahead)
        historical = daily_candles[
            (daily_candles.index >= cutoff_start) &
            (daily_candles.index < pd.Timestamp(analysis_date))
            ].copy()

        if len(historical) < 2:
            return None

        historical['prev_close'] = historical['close'].shift(1)
        historical['prev_high'] = historical['high'].shift(1)
        historical['prev_low'] = historical['low'].shift(1)
        historical = historical.dropna(subset=['prev_close', 'prev_high', 'prev_low'])

        historical['gap_pct'] = (historical['open'] - historical['prev_close']) / historical['prev_close']

        # Classify each row by direction
        historical['gap_type'] = np.where(
            historical['gap_pct'] >= self.gap_threshold_pct, 'UP',
            np.where(historical['gap_pct'] <= -self.gap_threshold_pct, 'DOWN', 'NONE')
        )
        # Exclude extreme gaps
        historical.loc[historical['gap_pct'].abs() > self.max_gap_pct, 'gap_type'] = 'NONE'

        # Classify as Full Gap or Partial Gap
        #   Full Gap Up   : open > prev_high  (opens entirely above previous range)
        #   Partial Gap Up: open > prev_close AND open <= prev_high (within prev range)
        #   Full Gap Down  : open < prev_low   (opens entirely below previous range)
        #   Partial Gap Down: open < prev_close AND open >= prev_low (within prev range)
        is_full_up = (historical['gap_type'] == 'UP') & (historical['open'] > historical['prev_high'])
        is_partial_up = (historical['gap_type'] == 'UP') & (historical['open'] <= historical['prev_high'])
        is_full_down = (historical['gap_type'] == 'DOWN') & (historical['open'] < historical['prev_low'])
        is_partial_down = (historical['gap_type'] == 'DOWN') & (historical['open'] >= historical['prev_low'])

        historical['gap_subtype'] = 'NONE'
        historical.loc[is_full_up, 'gap_subtype'] = 'FULL'
        historical.loc[is_partial_up, 'gap_subtype'] = 'PARTIAL'
        historical.loc[is_full_down, 'gap_subtype'] = 'FULL'
        historical.loc[is_partial_down, 'gap_subtype'] = 'PARTIAL'

        # Apply gap_calculation_mode: suppress gaps that don't match the chosen mode
        if self.gap_calculation_mode == 'full':
            historical.loc[historical['gap_subtype'] != 'FULL', 'gap_type'] = 'NONE'
        elif self.gap_calculation_mode == 'partial':
            historical.loc[historical['gap_subtype'] != 'PARTIAL', 'gap_type'] = 'NONE'
        # 'both' → keep all qualifying gaps as-is

        # Behaviour: did price close above or below open?
        # continuation for gap-up  → close > open
        # continuation for gap-down → close < open
        historical['gap_up_continuation'] = (historical['gap_type'] == 'UP') & (historical['close'] > historical['open'])
        historical['gap_up_reversal'] = (historical['gap_type'] == 'UP') & (historical['close'] < historical['open'])
        historical['gap_down_continuation'] = (historical['gap_type'] == 'DOWN') & (historical['close'] < historical['open'])
        historical['gap_down_reversal'] = (historical['gap_type'] == 'DOWN') & (historical['close'] > historical['open'])

        gap_up_rows = historical[historical['gap_type'] == 'UP']
        gap_down_rows = historical[historical['gap_type'] == 'DOWN']

        def _rate(mask_series, total_df):
            if len(total_df) == 0:
                return 0.0
            return mask_series.sum() / len(total_df)

        result = {
            'gap_up_count': len(gap_up_rows),
            'gap_up_continuation_rate': _rate(gap_up_rows['gap_up_continuation'], gap_up_rows),
            'gap_up_reversal_rate': _rate(gap_up_rows['gap_up_reversal'], gap_up_rows),
            'gap_down_count': len(gap_down_rows),
            'gap_down_continuation_rate': _rate(gap_down_rows['gap_down_continuation'], gap_down_rows),
            'gap_down_reversal_rate': _rate(gap_down_rows['gap_down_reversal'], gap_down_rows),
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
            rev = stats['gap_up_reversal_rate']
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
            rev = stats['gap_down_reversal_rate']
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
        print(f"\n{'=' * 60}")
        print(f"  Backtesting: {symbol}")
        print(f"{'=' * 60}")

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

        # Add previous day's OHLC references to daily
        daily['prev_close'] = daily['close'].shift(1)
        daily['prev_high'] = daily['high'].shift(1)
        daily['prev_low'] = daily['low'].shift(1)
        daily = daily.dropna(subset=['prev_close', 'prev_high', 'prev_low'])

        # All unique trading dates in intraday candles — restrict to last N days
        trading_dates = sorted(set(intra.index.date))
        if self.backtest_days and len(trading_dates) > self.backtest_days:
            trading_dates = trading_dates[-self.backtest_days:]
        print(f"  Backtesting {len(trading_dates)} day(s): {trading_dates[0]} → {trading_dates[-1]}"
              if trading_dates else "  No trading dates found.")

        # ----------------------------------------------------------------
        # Back-test loop
        # ----------------------------------------------------------------
        cash = self.initial_capital
        trades = []
        portfolio = []
        position = None

        for tdate in trading_dates:
            ts_date = pd.Timestamp(tdate)

            # Daily row for this date
            if ts_date not in daily.index:
                continue
            day_row = daily.loc[ts_date]
            day_open = day_row['open']
            prev_close = day_row['prev_close']
            prev_high = day_row['prev_high']
            prev_low = day_row['prev_low']

            if (prev_close == 0
                    or np.isnan(prev_close) or np.isnan(day_open)
                    or np.isnan(prev_high) or np.isnan(prev_low)):
                continue

            # Detect today's gap
            gap_pct = (day_open - prev_close) / prev_close

            if abs(gap_pct) < self.gap_threshold_pct or abs(gap_pct) > self.max_gap_pct:
                continue  # No qualifying gap today

            gap_type = 'UP' if gap_pct > 0 else 'DOWN'

            # Classify as Full Gap or Partial Gap
            #   Full Gap Up   : open > prev_high  → strong momentum, opens beyond prior range
            #   Partial Gap Up: prev_close < open <= prev_high → within prior range
            #   Full Gap Down  : open < prev_low   → opens entirely below prior range
            #   Partial Gap Down: prev_low <= open < prev_close → within prior range
            if gap_type == 'UP':
                gap_subtype = 'FULL_UP' if day_open > prev_high else 'PARTIAL_UP'
            else:
                gap_subtype = 'FULL_DOWN' if day_open < prev_low else 'PARTIAL_DOWN'

            # Apply gap_calculation_mode filter
            if self.gap_calculation_mode == 'full' and gap_subtype not in ('FULL_UP', 'FULL_DOWN'):
                continue
            if self.gap_calculation_mode == 'partial' and gap_subtype not in ('PARTIAL_UP', 'PARTIAL_DOWN'):
                continue

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
            position = None  # None | dict
            entry_candle_count = 0

            for idx, row in day_candles.iterrows():
                candle_time = idx.time()
                is_sqoff = candle_time >= self.square_off_time

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
                        initial_stop = entry_price - self.initial_stop_atr_mult * atr_val
                        trailing_stop = entry_price - self.trailing_stop_atr_mult * atr_val
                        target = entry_price + self.target_atr_mult * atr_val
                    else:  # SHORT
                        initial_stop = entry_price + self.initial_stop_atr_mult * atr_val
                        trailing_stop = entry_price + self.trailing_stop_atr_mult * atr_val
                        target = entry_price - self.target_atr_mult * atr_val

                    shares = max(1, int(self.initial_capital / entry_price))
                    cost = shares * entry_price
                    cash -= cost

                    position = {
                        'symbol': symbol,
                        'date': tdate,
                        'direction': direction,
                        'entry_time': idx,
                        'entry_price': entry_price,
                        'shares': shares,
                        'cash_committed': cost,
                        'initial_stop': initial_stop,
                        'trailing_stop': trailing_stop,
                        'target': target,
                        'gap_type': gap_type,
                        'gap_subtype': gap_subtype,
                        'gap_pct': gap_pct,
                        'gap_stats': stats,
                    }
                    trades_today += 1


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
            'symbol': position['symbol'],
            'trade_date': str(position['date']),
            'direction': position['direction'],
            'gap_type': position['gap_type'],
            'gap_subtype': position['gap_subtype'],
            'gap_pct': round(position['gap_pct'] * 100, 3),
            'entry_time': position['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'exit_time': exit_time.strftime('%Y-%m-%d %H:%M:%S'),
            'entry_price': round(position['entry_price'], 2),
            'exit_price': round(exit_price, 2),
            'shares': position['shares'],
            'pnl': round(pnl, 2),
            'return_pct': round(pnl / position['cash_committed'] * 100, 3),
            'exit_reason': reason,
            'cash_after': round(cash_after, 2),
            'hist_cont_rate': round(
                position['gap_stats'].get(
                    f"gap_{position['gap_type'].lower()}_continuation_rate", 0) * 100, 1),
            'hist_rev_rate': round(
                position['gap_stats'].get(
                    f"gap_{position['gap_type'].lower()}_reversal_rate", 0) * 100, 1),
            'hist_gap_count': position['gap_stats'].get(
                f"gap_{position['gap_type'].lower()}_count", 0),
        }

    def _compute_summary(self, symbol, trades, final_cash):
        if not trades:
            return {
                'symbol': symbol, 'total_trades': 0, 'win_rate': 0,
                'net_pnl': 0, 'final_capital': final_cash,
            }

        pnl_list = [t['pnl'] for t in trades]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]
        net_pnl = sum(pnl_list)
        win_rate = len(wins) / len(trades) * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        pf = abs(sum(wins) / sum(losses)) if losses else float('inf')

        exit_counts = {}
        for t in trades:
            r = t['exit_reason']
            exit_counts[r] = exit_counts.get(r, 0) + 1

        long_trades = [t for t in trades if t['direction'] == 'LONG']
        short_trades = [t for t in trades if t['direction'] == 'SHORT']
        gapup_trades = [t for t in trades if t['gap_type'] == 'UP']
        gapdn_trades = [t for t in trades if t['gap_type'] == 'DOWN']
        full_gapup_trades = [t for t in trades if t.get('gap_subtype') == 'FULL_UP']
        full_gapdn_trades = [t for t in trades if t.get('gap_subtype') == 'FULL_DOWN']
        partial_gapup_trades = [t for t in trades if t.get('gap_subtype') == 'PARTIAL_UP']
        partial_gapdn_trades = [t for t in trades if t.get('gap_subtype') == 'PARTIAL_DOWN']

        return {
            'symbol': symbol,
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(win_rate, 2),
            'net_pnl': round(net_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(pf, 2),
            'best_trade': round(max(pnl_list), 2),
            'worst_trade': round(min(pnl_list), 2),
            'final_capital': round(final_cash, 2),
            'return_pct': round((final_cash - self.initial_capital) / self.initial_capital * 100, 2),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'gap_up_trades': len(gapup_trades),
            'gap_down_trades': len(gapdn_trades),
            'full_gap_up_trades': len(full_gapup_trades),
            'full_gap_down_trades': len(full_gapdn_trades),
            'partial_gap_up_trades': len(partial_gapup_trades),
            'partial_gap_down_trades': len(partial_gapdn_trades),
            'exit_breakdown': exit_counts,
            'trade_log': trades,
        }

    # ------------------------------------------------------------------
    # Monthly breakdown
    # ------------------------------------------------------------------

    def _compute_monthly_breakdown(self, trades):
        """
        Group a list of trade dicts by calendar month (YYYY-MM) and
        compute per-month statistics.

        Returns
        -------
        list of dicts, one per month, sorted chronologically:
            month, trades, wins, losses, win_rate, net_pnl,
            avg_win, avg_loss, profit_factor, best_trade, worst_trade
        """
        if not trades:
            return []

        monthly: dict = {}
        for t in trades:
            month_key = str(t['trade_date'])[:7]   # "YYYY-MM"
            monthly.setdefault(month_key, []).append(t['pnl'])

        rows = []
        for month in sorted(monthly.keys()):
            pnl_list = monthly[month]
            wins   = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p < 0]
            net_pnl = sum(pnl_list)
            total   = len(pnl_list)
            win_rate = len(wins) / total * 100 if total else 0.0
            avg_win  = float(np.mean(wins))   if wins   else 0.0
            avg_loss = float(np.mean(losses)) if losses else 0.0
            pf = abs(sum(wins) / sum(losses)) if losses else float('inf')
            rows.append({
                'month':         month,
                'trades':        total,
                'wins':          len(wins),
                'losses':        len(losses),
                'win_rate':      round(win_rate, 2),
                'net_pnl':       round(net_pnl, 2),
                'avg_win':       round(avg_win, 2),
                'avg_loss':      round(avg_loss, 2),
                'profit_factor': round(pf, 2) if pf != float('inf') else None,
                'best_trade':    round(max(pnl_list), 2),
                'worst_trade':   round(min(pnl_list), 2),
            })
        return rows

    def print_trades_table(self):
        """Print a clean per-symbol trades table with one row per trade."""
        if not self.results:
            return

        COL = 115
        print("\n")
        print("=" * COL)
        print("                         GAP STRATEGY — TRADES BY SYMBOL")
        print("=" * COL)

        HDR = (
            f"  {'Date':<12} {'Entry':>6} {'Exit':>6}  {'Dir':<6} {'Qty':>5}  "
            f"{'Entry ₹':>9}  {'Exit ₹':>9}  {'Gap Type':<14} {'Gap%':>6}  "
            f"{'Hist Cont':>9} {'Hist Rev':>8}  {'Exit Reason':<14}  {'P&L ₹':>11}"
        )
        SEP = "  " + "-" * (COL - 2)

        for sym in sorted(self.results.keys()):
            r = self.results[sym]
            trades = r.get('trade_log', [])

            print(f"\n  {'─' * (COL - 2)}")
            if not trades:
                print(f"  Symbol: {sym}  —  No trades")
                continue

            total_pnl = sum(t['pnl'] for t in trades)
            wins = sum(1 for t in trades if t['pnl'] > 0)
            win_rate = wins / len(trades) * 100

            pnl_label = f"+₹{total_pnl:,.2f}" if total_pnl >= 0 else f"₹{total_pnl:,.2f}"
            print(
                f"  Symbol : {sym}   |   {len(trades)} trade(s)   |   "
                f"Net P&L: {pnl_label}   |   "
                f"Win Rate: {wins}/{len(trades)} ({win_rate:.1f}%)"
            )
            print(SEP)
            print(HDR)
            print(SEP)

            for t in trades:
                pnl = t['pnl']
                entry_dt = t['entry_time']   # "YYYY-MM-DD HH:MM:SS"
                exit_dt  = t['exit_time']
                entry_date = entry_dt[:10]
                entry_time = entry_dt[11:16]
                exit_time  = exit_dt[11:16]
                pnl_str = f"+₹{pnl:>10,.2f}" if pnl >= 0 else f" ₹{pnl:>10,.2f}"
                print(
                    f"  {entry_date:<12} {entry_time:>6} {exit_time:>6}  "
                    f"{t['direction']:<6} {t['shares']:>5}  "
                    f"₹{t['entry_price']:>8.2f}  ₹{t['exit_price']:>8.2f}  "
                    f"{t['gap_subtype']:<14} {t['gap_pct']:>+6.2f}%  "
                    f"{t['hist_cont_rate']:>8.0f}% {t['hist_rev_rate']:>7.0f}%  "
                    f"{t['exit_reason']:<14}  {pnl_str}"
                )

            print(SEP)
            print(
                f"  {'':12} {'':6} {'':6}  {'':6} {'':5}  {'':9}  {'':9}  "
                f"{'':14} {'':6}  {'':9} {'':8}  "
                f"{'TOTAL':<14}  "
                f"{('+₹' if total_pnl >= 0 else ' ₹')}{total_pnl:>10,.2f}"
            )

        print("\n" + "=" * COL)

    def print_monthly_summary(self):
        """Print a month-wise breakdown table for every symbol that has trades."""
        if not self.results:
            print("\nNo backtest results for monthly summary.")
            return

        print("\n")
        print("=" * 100)
        print("                     GAP STRATEGY — MONTH-WISE BACKTEST RESULTS")
        print("=" * 100)

        grand_monthly: dict = {}   # month → list of PnLs (across all symbols)

        for sym, r in sorted(self.results.items()):
            trades = r.get('trade_log', [])
            if not trades:
                continue

            monthly_rows = self._compute_monthly_breakdown(trades)
            if not monthly_rows:
                continue

            print(f"\n  Symbol: {sym}")
            print(
                f"  {'Month':<10} {'Trades':>6} {'Wins':>5} {'Loss':>5} "
                f"{'WinRate':>8} {'NetPnL':>10} {'AvgWin':>9} {'AvgLoss':>9} "
                f"{'PF':>6} {'Best':>9} {'Worst':>9}"
            )
            print("  " + "-" * 93)
            for row in monthly_rows:
                pf_str = f"{row['profit_factor']:.2f}" if row['profit_factor'] is not None else "  ∞"
                print(
                    f"  {row['month']:<10} {row['trades']:>6} {row['wins']:>5} {row['losses']:>5} "
                    f"{row['win_rate']:>7.1f}% ₹{row['net_pnl']:>9,.0f} "
                    f"₹{row['avg_win']:>8,.0f} ₹{row['avg_loss']:>8,.0f} "
                    f"{pf_str:>6} ₹{row['best_trade']:>8,.0f} ₹{row['worst_trade']:>8,.0f}"
                )

            # Accumulate for grand total
            for t in trades:
                mk = str(t['trade_date'])[:7]
                grand_monthly.setdefault(mk, []).append(t['pnl'])

        # Grand total across all symbols, per month
        if grand_monthly:
            print("\n")
            print("  ALL SYMBOLS COMBINED — Monthly Totals")
            print(
                f"  {'Month':<10} {'Trades':>6} {'Wins':>5} {'Loss':>5} "
                f"{'WinRate':>8} {'NetPnL':>10} {'AvgWin':>9} {'AvgLoss':>9} "
                f"{'PF':>6} {'Best':>9} {'Worst':>9}"
            )
            print("  " + "-" * 93)
            for month in sorted(grand_monthly.keys()):
                pnl_list = grand_monthly[month]
                wins   = [p for p in pnl_list if p > 0]
                losses = [p for p in pnl_list if p < 0]
                net_pnl  = sum(pnl_list)
                total    = len(pnl_list)
                win_rate = len(wins) / total * 100 if total else 0.0
                avg_win  = float(np.mean(wins))   if wins   else 0.0
                avg_loss = float(np.mean(losses)) if losses else 0.0
                pf       = abs(sum(wins) / sum(losses)) if losses else float('inf')
                pf_str   = f"{pf:.2f}" if pf != float('inf') else "  ∞"
                print(
                    f"  {month:<10} {total:>6} {len(wins):>5} {len(losses):>5} "
                    f"{win_rate:>7.1f}% ₹{net_pnl:>9,.0f} "
                    f"₹{avg_win:>8,.0f} ₹{avg_loss:>8,.0f} "
                    f"{pf_str:>6} ₹{max(pnl_list):>8,.0f} ₹{min(pnl_list):>8,.0f}"
                )

        print("=" * 100)

    # ------------------------------------------------------------------
    # Stock performance filtering & ranking
    # ------------------------------------------------------------------

    def _score_symbol(self, result):
        """
        Compute a composite performance score for a symbol.

        Score components (all normalised to 0-1 range):
          - Win rate (40%): win_rate / 100
          - Profit factor (40%): capped at 5.0, normalised to [0, 1]
          - Return % (20%): capped at +20%, normalised to [0, 1]

        Returns a float in [0, 1].  Higher is better.
        """
        win_rate_score = result.get('win_rate', 0.0) / 100.0
        pf_capped = min(result.get('profit_factor', 0.0), 5.0)
        pf_score = pf_capped / 5.0
        ret_capped = max(0.0, min(result.get('return_pct', 0.0), 20.0))
        ret_score = ret_capped / 20.0
        return 0.40 * win_rate_score + 0.40 * pf_score + 0.20 * ret_score

    def filter_symbols_by_performance(self):
        """
        Evaluate every symbol in self.results against the configured
        thresholds and return a dict with:

            {
                'qualified': [(symbol, score, result), ...],   # sorted best→worst
                'excluded':  [(symbol, reason, result), ...],  # sorted worst→best
            }

        Pass criteria (ALL must hold):
          1. total_trades >= min_trades_required   (enough history)
          2. win_rate     >= min_win_rate           (enough winners)
          3. profit_factor >= min_profit_factor     (gross-win > gross-loss)
          4. loss_rate    <= max_loss_rate          (not too many losers)
          5. net_pnl      >  0                      (overall profitable)
        """
        qualified = []
        excluded = []

        for sym, result in self.results.items():
            total = result.get('total_trades', 0)

            # Not enough trades to evaluate reliably
            if total < self.min_trades_required:
                excluded.append((sym, f"too few trades ({total} < {self.min_trades_required})", result))
                continue

            win_rate = result.get('win_rate', 0.0)
            loss_rate = 100.0 - win_rate
            pf = result.get('profit_factor', 0.0)
            net_pnl = result.get('net_pnl', 0.0)

            failures = []
            if win_rate < self.min_win_rate:
                failures.append(f"win_rate {win_rate:.1f}% < {self.min_win_rate}%")
            if pf < self.min_profit_factor:
                failures.append(f"profit_factor {pf:.2f} < {self.min_profit_factor}")
            if loss_rate > self.max_loss_rate:
                failures.append(f"loss_rate {loss_rate:.1f}% > {self.max_loss_rate}%")
            if net_pnl <= 0:
                failures.append(f"net_pnl ₹{net_pnl:,.0f} ≤ 0")

            if failures:
                excluded.append((sym, "; ".join(failures), result))
            else:
                score = self._score_symbol(result)
                qualified.append((sym, score, result))

        # Sort qualified best → worst; excluded worst → best (score)
        qualified.sort(key=lambda x: x[1], reverse=True)
        excluded.sort(key=lambda x: self._score_symbol(x[2]))

        return {'qualified': qualified, 'excluded': excluded}

    def rank_symbols_by_performance(self):
        """
        Return ALL symbols sorted by composite score (best first),
        regardless of pass/fail status.  Useful for tiered views.
        """
        ranked = []
        for sym, result in self.results.items():
            if result.get('total_trades', 0) == 0:
                continue
            score = self._score_symbol(result)
            ranked.append((sym, score, result))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def get_qualified_symbols(self):
        """
        Return only the symbol strings that passed all performance filters.
        Ready to use as the `symbols` list for a live/paper session.
        """
        filtered = self.filter_symbols_by_performance()
        return [sym for sym, _score, _r in filtered['qualified']]

    def print_filter_report(self):
        """
        Print a ranked table showing which stocks qualify for live orders
        and which are blocked, with reasons.
        """
        filtered = self.filter_symbols_by_performance()
        qualified = filtered['qualified']
        excluded = filtered['excluded']

        print("\n")
        print("=" * 95)
        print("               STOCK PERFORMANCE FILTER REPORT  —  GAP STRATEGY")
        print("=" * 95)
        print(
            f"  Filters: win_rate≥{self.min_win_rate}%  |  "
            f"profit_factor≥{self.min_profit_factor}  |  "
            f"trades≥{self.min_trades_required}  |  "
            f"loss_rate≤{self.max_loss_rate}%  |  net_pnl>0"
        )
        print("=" * 95)

        # ---- QUALIFIED (ORDER ALLOWED) ----
        print(f"\n  QUALIFIED — {len(qualified)} symbol(s) eligible for order placement\n")
        print(f"  {'Rank':<5} {'Symbol':<25} {'Score':>6} {'Trades':>6} {'WinRate':>8} "
              f"{'PF':>6} {'NetPnL':>10} {'Return%':>8}")
        print("  " + "-" * 80)
        for rank, (sym, score, r) in enumerate(qualified, 1):
            print(
                f"  {rank:<5} {sym:<25} {score:>6.3f} {r['total_trades']:>6} "
                f"{r['win_rate']:>7.1f}% {r['profit_factor']:>6.2f} "
                f"₹{r['net_pnl']:>9,.0f} {r['return_pct']:>7.2f}%"
            )
        if not qualified:
            print("  (none)")

        # ---- EXCLUDED (ORDER BLOCKED) ----
        print(f"\n  EXCLUDED — {len(excluded)} symbol(s) blocked from order placement\n")
        print(f"  {'Symbol':<25} {'Trades':>6} {'WinRate':>8} {'PF':>6} {'NetPnL':>10}  Reason")
        print("  " + "-" * 90)
        for sym, reason, r in excluded:
            total = r.get('total_trades', 0)
            if total == 0:
                print(f"  {sym:<25} {'0':>6}  {'N/A':>8}  {'N/A':>6}  {'N/A':>10}  {reason}")
            else:
                print(
                    f"  {sym:<25} {total:>6} {r['win_rate']:>7.1f}% "
                    f"{r['profit_factor']:>6.2f} ₹{r['net_pnl']:>9,.0f}  {reason}"
                )
        if not excluded:
            print("  (none)")

        print("\n" + "=" * 95)
        print(f"  ORDER PLACEMENT SUMMARY: {len(qualified)} APPROVED  |  {len(excluded)} BLOCKED")
        print("=" * 95)

        return filtered

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

        self.print_trades_table()
        self.print_summary()
        self.print_monthly_summary()
        self.print_filter_report()
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

        total_pnl = 0
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
            total_pnl += r['net_pnl']
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
        print("  " + "-" * 90)
        print(f"  {'Symbol':<25}  {'GapUp':>6}  {'GapDn':>6}  "
              f"{'FullUp':>7}  {'FullDn':>7}  {'PartUp':>7}  {'PartDn':>7}  "
              f"{'Long':>5}  {'Short':>5}")
        print("  " + "-" * 90)
        for sym, r in sorted(self.results.items()):
            if r['total_trades'] == 0:
                continue
            print(
                f"  {sym:<25}  {r['gap_up_trades']:>6}  {r['gap_down_trades']:>6}  "
                f"{r['full_gap_up_trades']:>7}  {r['full_gap_down_trades']:>7}  "
                f"{r['partial_gap_up_trades']:>7}  {r['partial_gap_down_trades']:>7}  "
                f"{r['long_trades']:>5}  {r['short_trades']:>5}"
            )
        print("=" * 90)

        gap_mode_labels = {
            "full":    "Full Gap only (open beyond prev high/low)",
            "partial": "Partial Gap only (open beyond prev close, within prev range)",
            "both":    "Both Full & Partial gaps",
        }
        print(f"\n  Gap Calculation Mode: {gap_mode_labels.get(self.gap_calculation_mode, self.gap_calculation_mode)}")
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

        # Filter report CSV — which stocks are approved/blocked for live orders
        filtered = self.filter_symbols_by_performance()
        filter_rows = []
        for rank, (sym, score, r) in enumerate(filtered['qualified'], 1):
            filter_rows.append({
                'rank': rank,
                'symbol': sym,
                'status': 'QUALIFIED',
                'score': round(score, 4),
                'total_trades': r.get('total_trades', 0),
                'win_rate': r.get('win_rate', 0),
                'profit_factor': r.get('profit_factor', 0),
                'net_pnl': r.get('net_pnl', 0),
                'return_pct': r.get('return_pct', 0),
                'rejection_reason': '',
            })
        for sym, reason, r in filtered['excluded']:
            filter_rows.append({
                'rank': '',
                'symbol': sym,
                'status': 'EXCLUDED',
                'score': round(self._score_symbol(r), 4) if r.get('total_trades', 0) > 0 else 0,
                'total_trades': r.get('total_trades', 0),
                'win_rate': r.get('win_rate', 0),
                'profit_factor': r.get('profit_factor', 0),
                'net_pnl': r.get('net_pnl', 0),
                'return_pct': r.get('return_pct', 0),
                'rejection_reason': reason,
            })
        if filter_rows:
            filter_path = os.path.join("output", f"gap_strategy_filter_{ts}.csv")
            pd.DataFrame(filter_rows).to_csv(filter_path, index=False)
            print(f"  Filter report    → {filter_path}")

        # Monthly breakdown CSV — one row per (symbol, month)
        monthly_rows = []
        for sym, r in self.results.items():
            trades = r.get('trade_log', [])
            for row in self._compute_monthly_breakdown(trades):
                monthly_rows.append({'symbol': sym, **row})

        if monthly_rows:
            monthly_path = os.path.join("output", f"gap_strategy_monthly_{ts}.csv")
            pd.DataFrame(monthly_rows).to_csv(monthly_path, index=False)
            print(f"  Monthly results  → {monthly_path}")


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
        # "NSE:BANKNIFTY26FEB60000PE",

        "NSE:ONGC-EQ",
        "NSE:OIL-EQ",
        "NSE:GAIL-EQ",

        # Renewables — Structural Beneficiaries
        "NSE:ADANIGREEN-EQ",
        "NSE:TATAPOWER-EQ",
        "NSE:CESC-EQ",

        # City Gas / LNG Distribution
        "NSE:IGL-EQ",
        "NSE:MGL-EQ",
        "NSE:GUJGASLTD-EQ",
        "NSE:PETRONET-EQ",

        # Defence
        "NSE:HAL-EQ",
        "NSE:BEL-EQ",
        "NSE:MAZDOCK-EQ",
        "NSE:DATAPATTNS-EQ",

        # Sugar - Ethanol
        "NSE:EIDPARRY-EQ",
        "NSE:BALRAMCHIN-EQ",
        "NSE:TRIVENI-EQ",

        # Pharmaceuticals
        "NSE:SUNPHARMA-EQ",
        "NSE:DIVISLAB-EQ",
        "NSE:CIPLA-EQ",

        # Petroluem (Oil Marketing Companies)
        "NSE:IOC-EQ",
        "NSE:BPCL-EQ",
        "NSE:HINDPETRO-EQ",

        # Airlines
        "NSE:INDIGO-EQ",

        # Paints
        "NSE:ASIANPAINT-EQ",
        "NSE:BERGEPAINT-EQ",
        "NSE:KANSAINER-EQ",
        "NSE:INDIGOPNTS-EQ",

        # Tyres
        "NSE:CEATLTD-EQ",
        "NSE:MRF-EQ",
        "NSE:APOLLOTYRE-EQ",
        "NSE:JKTYRE-EQ",
        "NSE:BALKRISIND-EQ",

        # Autos(Nifty Auto)
        "NSE:MARUTI-EQ",
        "NSE:M&M-EQ",
        "NSE:BAJAJ-AUTO-EQ",
        "NSE:EICHERMOT-EQ",
        "NSE:TVSMOTOR-EQ",

        # IPO Stocks
        "NSE:VIKRAMSOLR-EQ",
        "NSE:ATLANTAELE-EQ",
        "NSE:SOLARWORLD-EQ",
        "NSE:RUBICON-EQ",
        "NSE:MIDWESTLTD-EQ",

        # Favourite Stocks
        "NSE:STLTECH-EQ",
        "NSE:SKYGOLD-EQ",
        "NSE:AXISCADES-EQ",
    ]

    backtester = GapStrategyBacktester(
        fyers_client_id=FYERS_CLIENT_ID,
        fyers_access_token=FYERS_ACCESS_TOKEN,
        symbols=SYMBOLS,

        # Gap detection
        gap_threshold_pct=1.0,  # minimum 0.3% gap to qualify
        max_gap_pct=5.0,  # ignore gaps larger than 5% (news/earnings)

        # Historical behaviour analysis
        behavior_lookback_days=90,
        min_gap_history=5,  # need at least 5 prior gap events to decide
        continuation_threshold=0.55,
        reversal_threshold=0.55,

        # Entry
        entry_candles=12,  # enter within first 3 × 5-min candles after open
        candle_interval="5S",
        last_entry_time="09:45",  # no new entries after 9:45 AM

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

        # Stock performance filter — controls which stocks get live orders
        # A stock must satisfy ALL four conditions to be QUALIFIED:
        #   win_rate >= min_win_rate
        #   profit_factor >= min_profit_factor
        #   loss_rate <= max_loss_rate
        #   net_pnl > 0
        min_win_rate=50.0,        # At least 50% of trades must be winners
        min_profit_factor=1.0,    # Gross wins must exceed gross losses
        min_trades_required=3,    # Need at least 3 trades to evaluate reliability
        max_loss_rate=60.0,       # No more than 60% losing trades allowed
    )

    results = backtester.run_backtest(max_workers=4)

    # ----------------------------------------------------------------
    # After the backtest, retrieve only the stocks that PASSED all
    # performance filters — these are safe to route live orders to.
    # ----------------------------------------------------------------
    qualified_symbols = backtester.get_qualified_symbols()

    print("\n" + "=" * 70)
    print("  LIVE ORDER ROUTING — APPROVED SYMBOLS")
    print("=" * 70)
    if qualified_symbols:
        print(f"  {len(qualified_symbols)} symbol(s) approved for live / paper orders:\n")
        for sym in qualified_symbols:
            print(f"    ✔  {sym}")
    else:
        print("  No symbols passed the performance filter.")
        print("  Consider relaxing min_win_rate / min_profit_factor thresholds.")
    print("=" * 70)
