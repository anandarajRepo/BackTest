import sqlite3
import pandas as pd
import numpy as np
import json
import glob
import os
from datetime import datetime, time, timedelta
import pytz
import warnings

warnings.filterwarnings('ignore')


class ADXCrossoverBacktester:
    """
    ADX Crossover Strategy — Buy Only
    ----------------------------------
    Entry : DI+ crosses above DI-  (for CE)  /  DI- crosses above DI+ (for PE)
    Exit  : Crossover reverses (direction flips from +ve to -ve)

    Uses the same dynamic Nifty ATM option contract selection as
    AdvancedHeikinAshi.py.
    """

    def __init__(self, data_folder="data", symbols=None,
                 adx_period=14, adx_threshold=20,
                 atr_period=14, atr_multiplier=1.5,
                 initial_capital=100000, square_off_time="15:20",
                 tick_interval=None, max_trades_per_day=5,
                 avoid_opening_mins=15, last_entry_time="14:45",
                 use_nifty_atm=False, nifty_strike_interval=50,
                 backtest_days=None):

        self.data_folder = data_folder
        self.db_files = self._find_database_files()
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.initial_capital = initial_capital
        self.square_off_time = self._parse_time(square_off_time)
        self.tick_interval = tick_interval
        self.max_trades_per_day = max_trades_per_day
        self.avoid_opening_mins = avoid_opening_mins
        self.last_entry_time = self._parse_time(last_entry_time)
        self.use_nifty_atm = use_nifty_atm
        self.nifty_strike_interval = nifty_strike_interval
        self.backtest_days = backtest_days

        self.results = {}
        self.combined_data = {}
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print(f"{'='*100}")
        print(f"ADX CROSSOVER STRATEGY — BUY ONLY")
        print(f"{'='*100}")
        print(f"  ADX Period: {self.adx_period} | Threshold: {self.adx_threshold}")
        print(f"  ATR Period: {self.atr_period} | Multiplier: {self.atr_multiplier}x")
        print(f"  Tick Interval: {self.tick_interval or 'Raw tick data'}")
        print(f"  Max Trades/Day: {self.max_trades_per_day}")
        print(f"  Last Entry: {last_entry_time} IST | Square-off: {square_off_time} IST")
        print(f"  Backtest Period: {'Last ' + str(backtest_days) + ' days' if backtest_days else 'All data'}")
        print(f"  Capital: Rs.{self.initial_capital:,}")
        print(f"  Nifty ATM: {'ON (interval={})'.format(nifty_strike_interval) if use_nifty_atm else 'OFF'}")
        print(f"{'='*100}")

        # Resolve symbols
        if symbols is not None:
            self.symbols = symbols
        elif self.use_nifty_atm:
            if self.backtest_days is not None and self.backtest_days >= 90:
                # Weekly expiry mode: one ATM contract per expiry, resolved at backtest time
                print("  Mode: Weekly Expiry Backtest — symbols resolved per-expiry during run")
                self.symbols = []
            else:
                ce_sym, pe_sym = self.fetch_nifty_atm_contracts()
                if ce_sym and pe_sym:
                    self.symbols = [ce_sym, pe_sym]
                else:
                    print("ATM fetch failed — no symbols to backtest.")
                    self.symbols = []
        else:
            self.symbols = []

        if self.symbols:
            print(f"\nSymbols to backtest: {self.symbols}")

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_time(time_str):
        try:
            h, m = map(int, time_str.split(':'))
            return time(h, m)
        except Exception:
            return time(15, 20)

    def _find_database_files(self):
        if not os.path.exists(self.data_folder):
            return []
        return sorted(glob.glob(os.path.join(self.data_folder, "*.db")))

    # ── Nifty ATM contract selection (same as AdvancedHeikinAshi.py) ─────

    def get_weekly_expiry_date(self, reference_date=None):
        if reference_date is None:
            reference_date = datetime.now(self.ist_tz).date()
        days_until_tuesday = (1 - reference_date.weekday()) % 7
        return reference_date + timedelta(days=days_until_tuesday)

    def is_last_tuesday_of_month(self, expiry_date):
        return (expiry_date + timedelta(days=7)).month != expiry_date.month

    def get_atm_strike(self, price):
        interval = self.nifty_strike_interval
        return int(round(price / interval) * interval)

    def build_nifty_option_symbol(self, expiry_date, strike, option_type):
        yy = expiry_date.strftime('%y')
        mon = str(expiry_date.month)
        day = str(expiry_date.day)
        opt = option_type.upper()
        if self.is_last_tuesday_of_month(expiry_date):
            return f"NSE:NIFTY{yy}{mon}{strike}{opt}"
        else:
            return f"NSE:NIFTY{yy}{mon}{day}{strike}{opt}"

    def _get_nifty_open_from_fyers(self):
        try:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass

            client_id = os.environ.get("FYERS_CLIENT_ID", "").strip()
            access_token = os.environ.get("FYERS_ACCESS_TOKEN", "").strip()

            if not client_id or not access_token:
                return None

            from fyers_apiv3 import fyersModel
            fyers = fyersModel.FyersModel(
                client_id=client_id,
                token=access_token,
                is_async=False,
                log_path=""
            )

            today = datetime.now(self.ist_tz).date()
            # Roll back to Friday if today is Saturday (5) or Sunday (6)
            weekday = today.weekday()
            if weekday == 5:        # Saturday → go back 1 day to Friday
                trading_day = today - timedelta(days=1)
            elif weekday == 6:      # Sunday → go back 2 days to Friday
                trading_day = today - timedelta(days=2)
            else:
                trading_day = today
            data = {
                "symbol": "NSE:NIFTY50-INDEX",
                "resolution": "1",
                "date_format": "1",
                "range_from": str(trading_day),
                "range_to": str(trading_day),
                "cont_flag": "1"
            }

            response = fyers.history(data=data)
            if response.get('s') == 'ok' and response.get('candles'):
                # candles[0] → [timestamp, open, high, low, close, volume]
                return float(response['candles'][0][1])

            return None

        except Exception as e:
            print(f"  Fyers API error while fetching Nifty open: {e}")
            return None

    def _get_nifty_open_from_db(self):
        nifty_candidates = [
            "NSE:NIFTY50-INDEX",
            "NIFTY50-INDEX",
            "NSE:NIFTY-INDEX",
            "NIFTY50",
            "NIFTY 50",
        ]

        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                for sym in nifty_candidates:
                    query = """
                    SELECT ltp
                    FROM market_data
                    WHERE symbol = ?
                    ORDER BY timestamp ASC
                    LIMIT 1
                    """
                    df = pd.read_sql_query(query, conn, params=(sym,))
                    if not df.empty:
                        conn.close()
                        return float(df.iloc[0]['ltp'])
                conn.close()
            except Exception:
                continue

        return None

    def fetch_nifty_atm_contracts(self, reference_date=None):
        print("\n" + "=" * 60)
        print("  NIFTY ATM CONTRACT SELECTION")
        print("=" * 60)

        # Step 1 — expiry date
        expiry_date = self.get_weekly_expiry_date(reference_date)
        expiry_label = "Monthly" if self.is_last_tuesday_of_month(expiry_date) else "Weekly"
        print(f"  Expiry Date  : {expiry_date.strftime('%d-%b-%Y')} ({expiry_label})")

        # Step 2 — Nifty 50 opening price
        print("  Fetching Nifty 50 opening price...")
        nifty_open = self._get_nifty_open_from_fyers()
        if nifty_open is not None:
            print(f"  Price Source : Fyers API")
        else:
            print("  Fyers API unavailable — trying local database...")
            nifty_open = self._get_nifty_open_from_db()
            if nifty_open is not None:
                print(f"  Price Source : Local Database")

        if nifty_open is None:
            print("  ERROR: Could not fetch Nifty 50 opening price from any source.")
            print("         Set FYERS_CLIENT_ID / FYERS_ACCESS_TOKEN in .env, or")
            print("         ensure Nifty index data exists in the database.")
            print("=" * 60)
            return None, None

        print(f"  Nifty Open   : ₹{nifty_open:,.2f}")

        # Step 3 — ATM strike
        atm_strike = self.get_atm_strike(nifty_open)
        print(f"  ATM Strike   : {atm_strike} "
              f"(nearest {self.nifty_strike_interval}-pt interval)")

        # Step 4 — Build symbols
        ce_symbol = self.build_nifty_option_symbol(expiry_date, atm_strike, 'CE')
        pe_symbol = self.build_nifty_option_symbol(expiry_date, atm_strike, 'PE')

        print(f"  CE Symbol    : {ce_symbol}")
        print(f"  PE Symbol    : {pe_symbol}")
        print("=" * 60 + "\n")

        return ce_symbol, pe_symbol

    # ── Data loading ─────────────────────────────────────────────────────

    def load_data_from_all_databases(self, symbol):
        combined_df = pd.DataFrame()
        for db_file in self.db_files:
            try:
                df = self._load_single_db(db_file, symbol)
                if df is not None and not df.empty:
                    combined_df = pd.concat([combined_df, df], ignore_index=False)
            except Exception:
                continue
        if combined_df.empty:
            return None
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        if self.tick_interval is not None:
            combined_df = self._resample(combined_df, self.tick_interval)
        return combined_df

    def _load_single_db(self, db_path, symbol):
        try:
            conn = sqlite3.connect(db_path)
            count = pd.read_sql_query(
                "SELECT COUNT(*) FROM market_data WHERE symbol=?",
                conn, params=(symbol,)).iloc[0, 0]
            if count == 0:
                conn.close()
                return None
            df = pd.read_sql_query(
                "SELECT timestamp, ltp, high_price, low_price, close_price, volume, raw_data "
                "FROM market_data WHERE symbol=? ORDER BY timestamp",
                conn, params=(symbol,))
            conn.close()
            if df.empty:
                return None

            def _parse_raw(raw_str):
                try:
                    if raw_str:
                        rd = json.loads(raw_str)
                        return pd.Series({
                            'high_raw': rd.get('high_price', np.nan),
                            'low_raw': rd.get('low_price', np.nan),
                            'volume_raw': rd.get('vol_traded_today', 0)})
                except Exception:
                    pass
                return pd.Series({'high_raw': np.nan, 'low_raw': np.nan, 'volume_raw': 0})

            raw_parsed = df['raw_data'].apply(_parse_raw)
            df = pd.concat([df, raw_parsed], axis=1)
            df['open'] = df['ltp']
            df['high'] = df['high_raw'].fillna(df['high_price']).fillna(df['ltp'])
            df['low'] = df['low_raw'].fillna(df['low_price']).fillna(df['ltp'])
            df['close'] = df['close_price'].fillna(df['ltp'])
            df['volume'] = df['volume_raw'].fillna(df['volume']).fillna(0)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.dropna(subset=['close', 'high', 'low'])
            return df[['open', 'high', 'low', 'close', 'volume']].copy()
        except Exception:
            return None

    def load_data_from_fyers(self, symbol):
        try:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            client_id = os.environ.get("FYERS_CLIENT_ID", "").strip()
            access_token = os.environ.get("FYERS_ACCESS_TOKEN", "").strip()
            if not client_id or not access_token:
                return None
            from fyers_apiv3 import fyersModel
            fyers = fyersModel.FyersModel(
                client_id=client_id, token=access_token,
                is_async=False, log_path="")
            fetch_days = (self.backtest_days or 30) + 5
            now = datetime.now(self.ist_tz)
            range_from = int((now - timedelta(days=fetch_days)).timestamp())
            range_to = int(now.timestamp())
            data = {
                "symbol": symbol, "resolution": "1", "date_format": "0",
                "range_from": str(range_from), "range_to": str(range_to),
                "cont_flag": "1"
            }
            response = fyers.history(data=data)
            if response.get('s') != 'ok' or 'candles' not in response:
                return None
            candles = response['candles']
            if not candles:
                return None
            df = pd.DataFrame(candles,
                              columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')
            df.index = df.index.tz_localize('UTC').tz_convert(self.ist_tz).tz_localize(None)
            df = df.between_time('09:00', '15:30')
            df = df.dropna(subset=['close', 'high', 'low'])
            if df.empty:
                return None
            print(f"  Loaded {len(df)} candles from Fyers ({df.index[0].date()} -> {df.index[-1].date()})")
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            if self.tick_interval:
                try:
                    target = pd.tseries.frequencies.to_offset(self.tick_interval)
                    if target >= pd.tseries.frequencies.to_offset('1T'):
                        df = self._resample(df, self.tick_interval)
                except Exception:
                    pass
            return df
        except Exception as e:
            print(f"  Fyers error for {symbol}: {e}")
            return None

    def _resample(self, df, interval):
        if df is None or df.empty:
            return df
        resampled = df.resample(interval).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'})
        resampled = resampled.dropna(subset=['close'])
        for col in ('open', 'high', 'low'):
            resampled[col] = resampled[col].fillna(resampled['close'])
        resampled['volume'] = resampled['volume'].fillna(0)
        return resampled

    def _resolve_trading_day_cutoff(self, df):
        trading_dates = sorted(set(df.index.date))
        if not trading_dates:
            return (datetime.now(self.ist_tz) - timedelta(days=self.backtest_days)).date()
        if self.backtest_days >= len(trading_dates):
            return trading_dates[0]
        return trading_dates[-self.backtest_days]

    # ── Indicators ───────────────────────────────────────────────────────

    def calculate_adx(self, df):
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        df['dm_plus'] = np.where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
            np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        df['dm_minus'] = np.where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
            np.maximum(df['low'].shift(1) - df['low'], 0), 0)

        alpha = 1 / self.adx_period
        df['tr_smooth'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
        df['dm_plus_smooth'] = df['dm_plus'].ewm(alpha=alpha, adjust=False).mean()
        df['dm_minus_smooth'] = df['dm_minus'].ewm(alpha=alpha, adjust=False).mean()

        df['di_plus'] = 100 * (df['dm_plus_smooth'] / df['tr_smooth'])
        df['di_minus'] = 100 * (df['dm_minus_smooth'] / df['tr_smooth'])
        df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
        df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()
        return df

    def calculate_atr(self, df):
        if 'tr' not in df.columns:
            df['prev_close'] = df['close'].shift(1)
            df['tr'] = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['prev_close']),
                abs(df['low'] - df['prev_close'])], axis=1).max(axis=1)
        df['atr'] = df['tr'].ewm(span=self.atr_period, adjust=False).mean()
        return df

    # ── Signal generation (ADX DI+/DI- crossover) ───────────────────────

    def _detect_symbol_type(self, symbol):
        """Return 'CE' or 'PE' based on the symbol suffix."""
        sym = symbol.upper().rstrip()
        if sym.endswith('CE'):
            return 'CE'
        elif sym.endswith('PE'):
            return 'PE'
        return 'CE'  # default to CE logic

    def is_valid_trading_time(self, timestamp):
        try:
            t = timestamp.time()
            if t < time(9, 15 + self.avoid_opening_mins):
                return False
            if t >= self.last_entry_time:
                return False
            return True
        except Exception:
            return False

    def generate_signals(self, df, symbol):
        """
        Generate ADX crossover buy/sell signals.

        CE contracts: buy when DI+ crosses above DI-  (bullish crossover)
                      exit when DI+ crosses below DI- (bearish crossover)
        PE contracts: buy when DI- crosses above DI+  (bearish crossover)
                      exit when DI- crosses below DI+ (bullish crossover)
        """
        sym_type = self._detect_symbol_type(symbol)

        # Direction flag: True when DI+ > DI-
        df['di_positive'] = df['di_plus'] > df['di_minus']
        df['di_positive_prev'] = df['di_positive'].shift(1)
        df['valid_time'] = df.index.map(self.is_valid_trading_time)

        # Crossovers
        crossover_to_positive = df['di_positive'] & ~df['di_positive_prev']
        crossover_to_negative = ~df['di_positive'] & df['di_positive_prev']

        if sym_type == 'CE':
            # CE: buy on bullish crossover (DI+ crosses above DI-)
            df['buy_signal'] = (
                crossover_to_positive &
                (df['adx'] >= self.adx_threshold) &
                df['valid_time'] &
                df['adx'].notna())
            df['sell_signal'] = crossover_to_negative
        else:
            # PE: buy on bearish crossover (DI- crosses above DI+)
            df['buy_signal'] = (
                crossover_to_negative &
                (df['adx'] >= self.adx_threshold) &
                df['valid_time'] &
                df['adx'].notna())
            df['sell_signal'] = crossover_to_positive

        return df

    # ── Backtest loop ────────────────────────────────────────────────────

    def backtest_single_symbol(self, symbol):
        print(f"\n{'='*100}")
        print(f"Backtesting: {symbol}  (type: {self._detect_symbol_type(symbol)})")
        print(f"{'='*100}")

        df = self.load_data_from_all_databases(symbol)
        min_pts = self.adx_period * 3
        if df is None or len(df) < min_pts:
            print(f"  DB insufficient — trying Fyers API...")
            df = self.load_data_from_fyers(symbol)
        if df is None or len(df) < min_pts:
            print(f"  Insufficient data for {symbol}. Skipping.")
            return None

        self.combined_data[symbol] = df.copy()

        # Calculate indicators and signals
        df = self.calculate_adx(df)
        df = self.calculate_atr(df)
        df = self.generate_signals(df, symbol)

        df['trading_day'] = df.index.date
        df['is_square_off'] = df.index.map(lambda ts: ts.time() >= self.square_off_time)

        # Apply backtest window
        if self.backtest_days is not None:
            cutoff = self._resolve_trading_day_cutoff(df)
            backtest_df = df[df.index.date >= cutoff].copy()
            if backtest_df.empty:
                print(f"  No data in last {self.backtest_days} days. Skipping.")
                return None
            actual_days = len(set(backtest_df.index.date))
            print(f"  Window: {backtest_df.index.date[0]} -> {backtest_df.index.date[-1]} ({actual_days} day(s))")
        else:
            backtest_df = df

        # Trading state
        cash = self.initial_capital
        position = 0
        entry_price = 0
        entry_time = None
        trades = []
        trade_number = 0
        trades_today = {}

        for i in range(len(backtest_df)):
            row = backtest_df.iloc[i]
            ts = backtest_df.index[i]
            price = row['close']
            is_sqoff = row['is_square_off']
            day = row['trading_day']

            if day not in trades_today:
                trades_today[day] = 0

            # Square off at EOD
            if position == 1 and is_sqoff:
                shares = int(cash / entry_price) if entry_price > 0 else 0
                pnl = shares * (price - entry_price)
                ret = ((price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                trade_number += 1
                dur = (ts - entry_time).total_seconds() / 60
                trades.append({
                    'trade_num': trade_number, 'entry_time': entry_time,
                    'exit_time': ts, 'entry_price': entry_price,
                    'exit_price': price, 'shares': shares,
                    'duration_minutes': dur, 'pnl': pnl,
                    'return_pct': ret, 'exit_reason': 'SQUARE_OFF',
                    'adx': row.get('adx', 0), 'di_plus': row.get('di_plus', 0),
                    'di_minus': row.get('di_minus', 0)})
                print(f"  [#{trade_number}] SQUARE_OFF {entry_time.strftime('%H:%M')}->{ts.strftime('%H:%M')} "
                      f"Rs.{entry_price:.2f}->Rs.{price:.2f}  P&L: Rs.{pnl:+.2f} ({ret:+.2f}%)")
                position = 0
                trades_today[day] += 1

            # Entry
            elif position == 0 and row['buy_signal'] and not is_sqoff:
                if trades_today[day] >= self.max_trades_per_day:
                    continue
                position = 1
                entry_price = price
                entry_time = ts

            # Exit on crossover reversal
            elif position == 1 and row['sell_signal']:
                shares = int(cash / entry_price) if entry_price > 0 else 0
                pnl = shares * (price - entry_price)
                ret = ((price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                trade_number += 1
                dur = (ts - entry_time).total_seconds() / 60
                trades.append({
                    'trade_num': trade_number, 'entry_time': entry_time,
                    'exit_time': ts, 'entry_price': entry_price,
                    'exit_price': price, 'shares': shares,
                    'duration_minutes': dur, 'pnl': pnl,
                    'return_pct': ret, 'exit_reason': 'ADX_CROSSOVER',
                    'adx': row.get('adx', 0), 'di_plus': row.get('di_plus', 0),
                    'di_minus': row.get('di_minus', 0)})
                print(f"  [#{trade_number}] ADX_CROSSOVER {entry_time.strftime('%H:%M')}->{ts.strftime('%H:%M')} "
                      f"Rs.{entry_price:.2f}->Rs.{price:.2f}  P&L: Rs.{pnl:+.2f} ({ret:+.2f}%)")
                position = 0
                trades_today[day] += 1

        metrics = self._calc_metrics(trades)
        return {'symbol': symbol, 'data': backtest_df, 'trades': trades, 'metrics': metrics}

    # ── Metrics ──────────────────────────────────────────────────────────

    def _calc_metrics(self, trades):
        empty = {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0,
            'total_return': 0, 'avg_return': 0, 'best_trade': 0,
            'worst_trade': 0, 'avg_duration': 0, 'avg_win': 0,
            'avg_loss': 0, 'profit_factor': 0, 'max_consecutive_losses': 0}
        if not trades:
            return empty
        n = len(trades)
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] <= 0]
        total_pnl = sum(t['pnl'] for t in trades)
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        max_consec = curr = 0
        for t in trades:
            if t['pnl'] <= 0:
                curr += 1
                max_consec = max(max_consec, curr)
            else:
                curr = 0
        return {
            'total_trades': n,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / n * 100,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / n,
            'total_return': sum(t['return_pct'] for t in trades),
            'avg_return': sum(t['return_pct'] for t in trades) / n,
            'best_trade': max(t['pnl'] for t in trades),
            'worst_trade': min(t['pnl'] for t in trades),
            'avg_duration': sum(t['duration_minutes'] for t in trades) / n,
            'avg_win': total_wins / len(wins) if wins else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else 0,
            'max_consecutive_losses': max_consec}

    # ── Weekly expiry backtest ───────────────────────────────────────────

    def get_all_weekly_expiries_in_window(self, start_date, end_date):
        """Return all weekly Nifty expiry (Tuesday) dates in [start_date, end_date]."""
        expiries = []
        days_to_tuesday = (1 - start_date.weekday()) % 7
        current = start_date + timedelta(days=days_to_tuesday)
        while current <= end_date:
            expiries.append(current)
            current += timedelta(days=7)
        return expiries

    def _get_historical_nifty_opens(self, start_date, end_date):
        """Fetch Nifty 50 daily open prices for the date range via Fyers API."""
        try:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            client_id = os.environ.get("FYERS_CLIENT_ID", "").strip()
            access_token = os.environ.get("FYERS_ACCESS_TOKEN", "").strip()
            if not client_id or not access_token:
                print("  Fyers credentials missing — cannot fetch historical Nifty opens.")
                return {}
            from fyers_apiv3 import fyersModel
            fyers = fyersModel.FyersModel(
                client_id=client_id, token=access_token, is_async=False, log_path="")
            data = {
                "symbol": "NSE:NIFTY50-INDEX",
                "resolution": "D",
                "date_format": "1",
                "range_from": str(start_date),
                "range_to": str(end_date),
                "cont_flag": "1"
            }
            response = fyers.history(data=data)
            if response.get('s') != 'ok' or 'candles' not in response:
                print(f"  Fyers historical Nifty fetch: {response.get('message', 'failed')}")
                return {}
            result = {}
            for candle in response['candles']:
                dt = datetime.fromtimestamp(int(candle[0]), tz=self.ist_tz).date()
                result[dt] = float(candle[1])   # open price
            print(f"  Nifty daily opens fetched: {len(result)} trading days")
            return result
        except Exception as e:
            print(f"  Error fetching historical Nifty opens: {e}")
            return {}

    def get_nifty_open_for_expiry_week(self, expiry_date, nifty_opens):
        """Return (open_price, price_date) for the Monday of the expiry week.

        Expiry is Tuesday, so Monday = expiry_date - 1 day.  Falls back to the
        nearest earlier trading day (up to 4 days back) then to the expiry day itself.
        """
        monday = expiry_date - timedelta(days=1)
        for delta in range(0, 5):
            check_date = monday - timedelta(days=delta)
            if check_date in nifty_opens:
                return nifty_opens[check_date], check_date
        if expiry_date in nifty_opens:
            return nifty_opens[expiry_date], expiry_date
        return None, None

    def load_data_from_fyers_for_date_range(self, symbol, from_date, to_date):
        """Load 1-min option data from Fyers for an explicit calendar date range."""
        try:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            client_id = os.environ.get("FYERS_CLIENT_ID", "").strip()
            access_token = os.environ.get("FYERS_ACCESS_TOKEN", "").strip()
            if not client_id or not access_token:
                return None
            from fyers_apiv3 import fyersModel
            fyers = fyersModel.FyersModel(
                client_id=client_id, token=access_token, is_async=False, log_path="")
            range_from_ts = int(datetime.combine(from_date, time(9, 0)).timestamp())
            range_to_ts = int(datetime.combine(to_date, time(23, 59)).timestamp())
            data = {
                "symbol": symbol, "resolution": "1", "date_format": "0",
                "range_from": str(range_from_ts), "range_to": str(range_to_ts),
                "cont_flag": "1"
            }
            response = fyers.history(data=data)
            if response.get('s') != 'ok' or 'candles' not in response:
                return None
            candles = response['candles']
            if not candles:
                return None
            df = pd.DataFrame(candles,
                              columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')
            df.index = df.index.tz_localize('UTC').tz_convert(self.ist_tz).tz_localize(None)
            df = df.between_time('09:00', '15:30')
            df = df.dropna(subset=['close', 'high', 'low'])
            if df.empty:
                return None
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            if self.tick_interval:
                try:
                    target = pd.tseries.frequencies.to_offset(self.tick_interval)
                    if target >= pd.tseries.frequencies.to_offset('1T'):
                        df = self._resample(df, self.tick_interval)
                except Exception:
                    pass
            return df
        except Exception as e:
            print(f"    Fyers error for {symbol}: {e}")
            return None

    def backtest_single_symbol_for_expiry(self, symbol, expiry_date, week_start, week_end):
        """Backtest one option symbol restricted to its expiry week."""
        min_pts = self.adx_period * 3

        # Try DB first, then Fyers with explicit date range
        df = self.load_data_from_all_databases(symbol)
        if df is None or len(df) < min_pts:
            df = self.load_data_from_fyers_for_date_range(symbol, week_start, week_end)
        if df is None or len(df) < min_pts:
            print(f"    Insufficient data for {symbol}. Skipping.")
            return None

        # Filter to expiry week only
        df = df[(df.index.date >= week_start) & (df.index.date <= week_end)].copy()
        if df.empty or len(df) < min_pts:
            print(f"    No usable data in {week_start} - {week_end}. Skipping.")
            return None

        print(f"    Data: {len(df)} rows | "
              f"{df.index[0].strftime('%d-%b %H:%M')} -> {df.index[-1].strftime('%d-%b %H:%M')}")

        df = self.calculate_adx(df)
        df = self.calculate_atr(df)
        df = self.generate_signals(df, symbol)
        df['trading_day'] = df.index.date
        df['is_square_off'] = df.index.map(lambda ts: ts.time() >= self.square_off_time)

        cash = self.initial_capital
        position = 0
        entry_price = 0
        entry_time = None
        trades = []
        trade_number = 0
        trades_today = {}

        for i in range(len(df)):
            row = df.iloc[i]
            ts = df.index[i]
            price = row['close']
            is_sqoff = row['is_square_off']
            day = row['trading_day']
            trades_today.setdefault(day, 0)

            if position == 1 and is_sqoff:
                shares = int(cash / entry_price) if entry_price > 0 else 0
                pnl = shares * (price - entry_price)
                ret = ((price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                trade_number += 1
                dur = (ts - entry_time).total_seconds() / 60
                trades.append({
                    'trade_num': trade_number, 'entry_time': entry_time,
                    'exit_time': ts, 'entry_price': entry_price,
                    'exit_price': price, 'shares': shares,
                    'duration_minutes': dur, 'pnl': pnl, 'return_pct': ret,
                    'exit_reason': 'SQUARE_OFF', 'adx': row.get('adx', 0),
                    'di_plus': row.get('di_plus', 0), 'di_minus': row.get('di_minus', 0)})
                print(f"    [#{trade_number}] SQUARE_OFF "
                      f"{entry_time.strftime('%H:%M')}->{ts.strftime('%H:%M')} "
                      f"Rs.{entry_price:.2f}->Rs.{price:.2f}  P&L: Rs.{pnl:+.2f} ({ret:+.2f}%)")
                position = 0
                trades_today[day] += 1

            elif position == 0 and row['buy_signal'] and not is_sqoff:
                if trades_today[day] >= self.max_trades_per_day:
                    continue
                position = 1
                entry_price = price
                entry_time = ts

            elif position == 1 and row['sell_signal']:
                shares = int(cash / entry_price) if entry_price > 0 else 0
                pnl = shares * (price - entry_price)
                ret = ((price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                trade_number += 1
                dur = (ts - entry_time).total_seconds() / 60
                trades.append({
                    'trade_num': trade_number, 'entry_time': entry_time,
                    'exit_time': ts, 'entry_price': entry_price,
                    'exit_price': price, 'shares': shares,
                    'duration_minutes': dur, 'pnl': pnl, 'return_pct': ret,
                    'exit_reason': 'ADX_CROSSOVER', 'adx': row.get('adx', 0),
                    'di_plus': row.get('di_plus', 0), 'di_minus': row.get('di_minus', 0)})
                print(f"    [#{trade_number}] ADX_CROSSOVER "
                      f"{entry_time.strftime('%H:%M')}->{ts.strftime('%H:%M')} "
                      f"Rs.{entry_price:.2f}->Rs.{price:.2f}  P&L: Rs.{pnl:+.2f} ({ret:+.2f}%)")
                position = 0
                trades_today[day] += 1

        metrics = self._calc_metrics(trades)
        return {
            'symbol': symbol, 'expiry_date': expiry_date,
            'week_start': week_start, 'week_end': week_end,
            'data': df, 'trades': trades, 'metrics': metrics
        }

    def _run_weekly_expiry_backtest(self):
        """Iterate over every weekly expiry in the backtest window and backtest each."""
        now = datetime.now(self.ist_tz).date()
        start_date = now - timedelta(days=self.backtest_days)
        end_date = now

        print(f"\n{'='*100}")
        print(f"WEEKLY EXPIRY BACKTEST MODE  ({self.backtest_days} days)")
        print(f"Period: {start_date} -> {end_date}")
        print(f"{'='*100}")

        expiries = self.get_all_weekly_expiries_in_window(start_date, end_date)
        print(f"\nWeekly expiries found ({len(expiries)}):")
        for exp in expiries:
            label = "Monthly" if self.is_last_tuesday_of_month(exp) else "Weekly"
            print(f"  {exp.strftime('%d-%b-%Y (%a)')}  [{label}]")

        print(f"\n  Fetching Nifty 50 daily opens ({start_date} -> {end_date}) ...")
        nifty_opens = self._get_historical_nifty_opens(start_date, end_date)
        if not nifty_opens:
            print("  WARNING: No Nifty historical data available — ATM strikes cannot be "
                  "determined.  Ensure FYERS_CLIENT_ID / FYERS_ACCESS_TOKEN are set.")

        for expiry_date in expiries:
            # Expiry week: Wednesday of previous week -> Tuesday (expiry day)
            week_start = expiry_date - timedelta(days=6)   # Wednesday
            week_end = expiry_date                          # Tuesday

            nifty_price, price_date = self.get_nifty_open_for_expiry_week(
                expiry_date, nifty_opens)
            if nifty_price is None:
                print(f"\n  [{expiry_date}] Skipping — no Nifty open found for this week.")
                continue

            atm_strike = self.get_atm_strike(nifty_price)
            ce_symbol = self.build_nifty_option_symbol(expiry_date, atm_strike, 'CE')
            pe_symbol = self.build_nifty_option_symbol(expiry_date, atm_strike, 'PE')

            label = "Monthly" if self.is_last_tuesday_of_month(expiry_date) else "Weekly"
            print(f"\n{'='*80}")
            print(f"  Expiry : {expiry_date.strftime('%d-%b-%Y')} [{label}]  |  "
                  f"Nifty open ({price_date}): {nifty_price:.0f}  |  ATM: {atm_strike}")
            print(f"  Week   : {week_start} -> {week_end}")
            print(f"  CE     : {ce_symbol}")
            print(f"  PE     : {pe_symbol}")

            for symbol in [ce_symbol, pe_symbol]:
                try:
                    result = self.backtest_single_symbol_for_expiry(
                        symbol, expiry_date, week_start, week_end)
                    if result:
                        self.results[symbol] = result
                except Exception as e:
                    print(f"    Error backtesting {symbol}: {e}")

        self.print_weekly_expiry_summary()

    def print_weekly_expiry_summary(self):
        if not self.results:
            print("\nNo results to display.")
            return
        print(f"\n{'='*100}")
        print("WEEKLY EXPIRY BACKTEST SUMMARY")
        print(f"{'='*100}")

        by_expiry = {}
        for result in self.results.values():
            exp = result['expiry_date']
            by_expiry.setdefault(exp, []).append(result)

        rows = []
        for exp in sorted(by_expiry.keys()):
            for result in by_expiry[exp]:
                m = result['metrics']
                sym = result['symbol']
                clean = sym.split(':')[1] if ':' in sym else sym
                rows.append({
                    'Expiry': exp.strftime('%d-%b-%Y'),
                    'Symbol': clean,
                    'Type': self._detect_symbol_type(sym),
                    'Trades': m['total_trades'],
                    'Win Rate': f"{m['win_rate']:.1f}%",
                    'Total P&L': f"Rs.{m['total_pnl']:+,.2f}",
                    'Avg P&L': f"Rs.{m['avg_pnl']:+,.2f}",
                    'Profit Factor': f"{m['profit_factor']:.2f}",
                })

        if rows:
            summary_df = pd.DataFrame(rows)
            print(summary_df.to_string(index=False))

            total_trades = sum(r['metrics']['total_trades'] for r in self.results.values())
            total_pnl = sum(r['metrics']['total_pnl'] for r in self.results.values())
            total_wins = sum(r['metrics']['winning_trades'] for r in self.results.values())

            print(f"\n{'='*100}")
            print("OVERALL STATISTICS")
            print(f"{'='*100}")
            print(f"  Expiries Tested  : {len(by_expiry)}")
            print(f"  Total Pairs      : {len(self.results)}")
            print(f"  Total Trades     : {total_trades}")
            print(f"  Total P&L        : Rs.{total_pnl:+,.2f}")
            if total_trades > 0:
                print(f"  Overall Win Rate : {total_wins / total_trades * 100:.1f}%")

            os.makedirs('output', exist_ok=True)
            out_file = f"output/adx_weekly_expiry_{self.backtest_days}days.csv"
            summary_df.to_csv(out_file, index=False)
            print(f"\n  Results saved to: {out_file}")
        print(f"{'='*100}")

    # ── Run & Report ─────────────────────────────────────────────────────

    def run_backtest(self):
        print(f"\n{'='*100}")
        print("STARTING ADX CROSSOVER BACKTEST")
        print(f"{'='*100}")

        # Weekly expiry mode: backtest_days >= 90 with Nifty ATM
        if (self.backtest_days is not None and self.backtest_days >= 90
                and self.use_nifty_atm):
            self._run_weekly_expiry_backtest()
            return

        for symbol in self.symbols:
            try:
                result = self.backtest_single_symbol(symbol)
                if result:
                    self.results[symbol] = result
            except Exception as e:
                print(f"Error backtesting {symbol}: {e}")
        self.print_summary()
        self.print_backtest_days_report()

    def print_summary(self):
        if not self.results:
            print("\nNo results to display.")
            return
        print(f"\n{'='*100}")
        print("ADX CROSSOVER RESULTS — ALL SYMBOLS")
        print(f"{'='*100}")
        rows = []
        for symbol, result in self.results.items():
            m = result['metrics']
            clean = symbol.split(':')[1].replace('-EQ', '') if ':' in symbol else symbol.replace('-EQ', '')
            rows.append({
                'Symbol': clean, 'Type': self._detect_symbol_type(symbol),
                'Trades': m['total_trades'],
                'Win Rate': f"{m['win_rate']:.1f}%",
                'Total P&L': f"Rs.{m['total_pnl']:.2f}",
                'Avg P&L': f"Rs.{m['avg_pnl']:.2f}",
                'Profit Factor': f"{m['profit_factor']:.2f}",
                'Max Consec Loss': m['max_consecutive_losses'],
                'Avg Duration': f"{m['avg_duration']:.1f}m"})
        summary_df = pd.DataFrame(rows)
        print(summary_df.to_string(index=False))

        total_trades = sum(r['metrics']['total_trades'] for r in self.results.values())
        total_pnl = sum(r['metrics']['total_pnl'] for r in self.results.values())
        total_wins = sum(r['metrics']['winning_trades'] for r in self.results.values())
        print(f"\n{'='*100}")
        print("OVERALL STATISTICS")
        print(f"{'='*100}")
        print(f"  Symbols Tested : {len(self.results)}")
        print(f"  Total Trades   : {total_trades}")
        print(f"  Total P&L      : Rs.{total_pnl:.2f}")
        if total_trades > 0:
            print(f"  Win Rate       : {total_wins / total_trades * 100:.1f}%")
        os.makedirs('output', exist_ok=True)
        summary_df.to_csv('output/adx_crossover_results.csv', index=False)
        print(f"\n  Results saved to: output/adx_crossover_results.csv")

    def print_backtest_days_report(self):
        if not self.results or self.backtest_days is None:
            return
        print(f"\n{'='*100}")
        print(f"BACKTEST REPORT — LAST {self.backtest_days} DAYS")
        print(f"{'='*100}")
        all_trades = []
        for symbol, result in self.results.items():
            clean = symbol.split(':')[1].replace('-EQ', '') if ':' in symbol else symbol.replace('-EQ', '')
            for t in result['trades']:
                all_trades.append({**t, 'symbol': clean})
        if not all_trades:
            print("  No trades in the specified period.")
            return
        trades_by_day = {}
        for t in all_trades:
            d = t['entry_time'].date()
            trades_by_day.setdefault(d, []).append(t)
        day_rows = []
        for d in sorted(trades_by_day):
            dt = trades_by_day[d]
            day_pnl = sum(t['pnl'] for t in dt)
            w = sum(1 for t in dt if t['pnl'] > 0)
            day_rows.append({
                'Date': d.strftime('%Y-%m-%d (%a)'),
                'Trades': len(dt), 'Wins': w, 'Losses': len(dt) - w,
                'Win Rate': f"{w/len(dt)*100:.1f}%",
                'Day P&L': f"Rs.{day_pnl:+,.2f}",
                'Status': 'PROFIT' if day_pnl > 0 else ('LOSS' if day_pnl < 0 else 'BREAKEVEN')})
        day_df = pd.DataFrame(day_rows)
        print(day_df.to_string(index=False))
        total_pnl = sum(t['pnl'] for t in all_trades)
        total_wins = sum(1 for t in all_trades if t['pnl'] > 0)
        profitable_days = sum(1 for ts in trades_by_day.values() if sum(t['pnl'] for t in ts) > 0)
        avg_daily = total_pnl / len(trades_by_day) if trades_by_day else 0
        print(f"\n  Trading Days : {len(trades_by_day)}")
        print(f"  Total Trades : {len(all_trades)}")
        print(f"  Total P&L    : Rs.{total_pnl:+,.2f}")
        print(f"  Avg Daily    : Rs.{avg_daily:+,.2f}")
        print(f"  Win Rate     : {total_wins/len(all_trades)*100:.1f}%")
        print(f"  Profitable Days : {profitable_days}/{len(trades_by_day)}")
        os.makedirs('output', exist_ok=True)
        day_df.to_csv(f"output/adx_crossover_{self.backtest_days}days_report.csv", index=False)
        print(f"  Report saved to: output/adx_crossover_{self.backtest_days}days_report.csv")
        print(f"{'='*100}")


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("ADX CROSSOVER STRATEGY — NIFTY ATM OPTIONS (DYNAMIC WEEKLY CONTRACTS)")
    print("=" * 100)

    BACKTEST_DAYS = 90

    bt = ADXCrossoverBacktester(
        data_folder="data/symbolupdate",
        symbols=None,

        # Nifty ATM
        use_nifty_atm=True,
        nifty_strike_interval=50,

        # ADX
        adx_period=14,
        adx_threshold=20,

        # ATR (for reference — no stop-loss in this strategy)
        atr_period=14,
        atr_multiplier=1.5,

        # Time filters
        avoid_opening_mins=15,
        last_entry_time="14:45",

        # Limits
        max_trades_per_day=5,
        initial_capital=100000,
        square_off_time="15:20",
        tick_interval='30s',

        backtest_days=BACKTEST_DAYS,
    )

    bt.run_backtest()
