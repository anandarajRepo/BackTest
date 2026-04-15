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


class ADXCrossoverMonthlyBacktester:
    """
    ADX Crossover Strategy for MONTHLY Options & Contracts
    -------------------------------------------------------
    Entry : DI+ crosses above DI-  (for CE)  /  DI- crosses above DI+ (for PE)
    Exit  : Crossover reverses (direction flips from +ve to -ve)

    Supports:
    - Monthly Options: BANKNIFTY, FINNIFTY, MIDCPNIFTY
    - Monthly Contracts: Gold, Silver, Oil, Natural Gas, etc.

    Each instrument has configurable ATM strike intervals and contract naming conventions.
    """

    # Instrument configurations: strike intervals, suffixes, and API symbols
    INSTRUMENTS = {
        'BANKNIFTY': {
            'strike_interval': 100,
            'type': 'index_option',
            'symbol_prefix': 'NSE:BANKNIFTY',
            'expiry_format': 'monthly',  # Last Monday (effective April 2025)
            'lot_size': 30  # Updated Dec 30, 2025 (previously 35)
        },
        'FINNIFTY': {
            'strike_interval': 50,
            'type': 'index_option',
            'symbol_prefix': 'NSE:FINNIFTY',
            'expiry_format': 'monthly',  # Last Monday (effective April 2025)
            'lot_size': 60  # Updated Dec 30, 2025 (previously 65)
        },
        'MIDCPNIFTY': {
            'strike_interval': 50,
            'type': 'index_option',
            'symbol_prefix': 'NSE:MIDCPNIFTY',
            'expiry_format': 'monthly',  # Last Monday (effective April 2025)
            'lot_size': 120  # Updated Dec 30, 2025 (previously 140)
        },
        'GOLD': {
            'strike_interval': 1,
            'type': 'commodity',
            'symbol_prefix': 'MCX:GOLDM',
            'expiry_format': 'monthly'
        },
        'SILVER': {
            'strike_interval': 1,
            'type': 'commodity',
            'symbol_prefix': 'MCX:SILVERM',
            'expiry_format': 'monthly'
        },
        'CRUDEOIL': {
            'strike_interval': 1,
            'type': 'commodity',
            'symbol_prefix': 'MCX:CRUDEOIL',
            'expiry_format': 'monthly'
        },
        'NATURALGAS': {
            'strike_interval': 1,
            'type': 'commodity',
            'symbol_prefix': 'MCX:NATURALGAS',
            'expiry_format': 'monthly'
        }
    }

    def __init__(self, data_folder="data", instruments=None,
                 adx_period=14, adx_threshold=20,
                 atr_period=14, atr_multiplier=1.5,
                 initial_capital=100000, square_off_time="15:20",
                 tick_interval=None, max_trades_per_day=5,
                 avoid_opening_mins=15, last_entry_time="14:45",
                 use_atm=False, backtest_days=None):

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
        self.use_atm = use_atm
        self.backtest_days = backtest_days

        self.results = {}
        self.combined_data = {}
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print(f"{'='*100}")
        print(f"ADX CROSSOVER STRATEGY — MONTHLY OPTIONS & CONTRACTS")
        print(f"{'='*100}")
        print(f"  ADX Period: {self.adx_period} | Threshold: {self.adx_threshold}")
        print(f"  ATR Period: {self.atr_period} | Multiplier: {self.atr_multiplier}x")
        print(f"  Tick Interval: {self.tick_interval or 'Raw tick data'}")
        print(f"  Max Trades/Day: {self.max_trades_per_day}")
        print(f"  Last Entry: {last_entry_time} IST | Square-off: {square_off_time} IST")
        print(f"  Backtest Period: {'Last ' + str(backtest_days) + ' days' if backtest_days else 'All data'}")
        print(f"  Capital: Rs.{self.initial_capital:,}")
        print(f"  ATM Mode: {'ON' if use_atm else 'OFF'}")
        print(f"{'='*100}")

        # Resolve instruments
        if instruments is not None:
            self.instruments = instruments
        else:
            self.instruments = ['BANKNIFTY', 'FINNIFTY']

        if self.instruments:
            print(f"\nInstruments to backtest: {self.instruments}")

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

    # ── Monthly Expiry Date Calculation ─────────────────────────────────────

    def get_monthly_expiry_date(self, reference_date=None):
        """
        Return the monthly expiry date (last Monday of the month).
        Note: Effective April 2025, NSE moved derivative expiry from Thursday to Monday.
        """
        if reference_date is None:
            reference_date = datetime.now(self.ist_tz).date()

        # Find last day of month
        if reference_date.month == 12:
            next_month = reference_date.replace(year=reference_date.year + 1, month=1, day=1)
        else:
            next_month = reference_date.replace(month=reference_date.month + 1, day=1)

        last_day = next_month - timedelta(days=1)

        # Find last Monday (weekday() returns 0 for Monday)
        days_back = (last_day.weekday() - 0) % 7  # 0 is Monday
        if days_back == 0 and last_day.weekday() != 0:
            days_back = 7
        return last_day - timedelta(days=days_back)

    def get_atm_strike(self, price, strike_interval):
        """Round price to nearest strike interval."""
        return int(round(price / strike_interval) * strike_interval)

    def build_monthly_option_symbol(self, instrument, expiry_date, strike, option_type):
        """
        Build symbol for monthly options/contracts.
        Formats vary by instrument type.
        """
        config = self.INSTRUMENTS.get(instrument, {})
        prefix = config.get('symbol_prefix', f'NSE:{instrument}')
        opt = option_type.upper()

        # Format: PREFIX{YY}{MM}{STRIKE}{CE/PE}
        yy = expiry_date.strftime('%y')
        mm = expiry_date.strftime('%m')

        if config.get('type') == 'commodity':
            # Commodities typically use SYMBOLMYYSTRIKE format
            return f"{prefix}{yy}{mm}{int(strike)}"
        else:
            # Index options: PREFIX{YY}{MM}{STRIKE}{CE/PE}
            return f"{prefix}{yy}{mm}{int(strike)}{opt}"

    def _get_instrument_open_from_fyers(self, instrument):
        """Fetch current price for an instrument from Fyers API."""
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

            config = self.INSTRUMENTS.get(instrument, {})
            symbol = config.get('symbol_prefix', f'NSE:{instrument}')

            today = datetime.now(self.ist_tz).date()
            weekday = today.weekday()
            if weekday == 5:
                trading_day = today - timedelta(days=1)
            elif weekday == 6:
                trading_day = today - timedelta(days=2)
            else:
                trading_day = today

            data = {
                "symbol": symbol,
                "resolution": "1",
                "date_format": "1",
                "range_from": str(trading_day),
                "range_to": str(trading_day),
                "cont_flag": "1"
            }

            response = fyers.history(data=data)
            if response.get('s') == 'ok' and response.get('candles'):
                return float(response['candles'][0][1])

            return None

        except Exception as e:
            print(f"  Fyers API error while fetching {instrument}: {e}")
            return None

    def _get_instrument_open_from_db(self, instrument):
        """Fetch instrument price from local database."""
        config = self.INSTRUMENTS.get(instrument, {})
        symbol = config.get('symbol_prefix', f'NSE:{instrument}')

        candidates = [symbol, instrument, f"NSE:{instrument}"]

        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                for sym in candidates:
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

    def fetch_atm_contracts(self, instruments, reference_date=None):
        """
        Fetch ATM strike for each instrument and build CE/PE symbols.
        Returns dict: {instrument: {'CE': symbol, 'PE': symbol, 'strike': price, 'expiry': date}}
        """
        print("\n" + "=" * 60)
        print("  MONTHLY ATM CONTRACT SELECTION")
        print("=" * 60)

        expiry_date = self.get_monthly_expiry_date(reference_date)
        print(f"  Monthly Expiry: {expiry_date.strftime('%d-%b-%Y (%A)')}\n")

        result = {}
        for instrument in instruments:
            print(f"  {instrument}:")
            config = self.INSTRUMENTS.get(instrument, {})
            strike_interval = config.get('strike_interval', 50)

            # Try Fyers first, then database
            price = self._get_instrument_open_from_fyers(instrument)
            source = "Fyers API"

            if price is None:
                price = self._get_instrument_open_from_db(instrument)
                source = "Local DB"

            if price is None:
                print(f"    WARNING: Could not fetch price for {instrument}")
                continue

            atm_strike = self.get_atm_strike(price, strike_interval)

            if config.get('type') == 'commodity':
                # Commodities don't have CE/PE
                symbol = self.build_monthly_option_symbol(instrument, expiry_date, atm_strike, '')
                result[instrument] = {
                    'symbol': symbol,
                    'strike': atm_strike,
                    'price': price,
                    'expiry': expiry_date,
                    'source': source
                }
                print(f"    Price: ₹{price:,.2f} ({source})")
                print(f"    ATM Strike: {atm_strike}")
                print(f"    Symbol: {symbol}")
            else:
                # Index options have CE/PE
                ce_symbol = self.build_monthly_option_symbol(instrument, expiry_date, atm_strike, 'CE')
                pe_symbol = self.build_monthly_option_symbol(instrument, expiry_date, atm_strike, 'PE')
                result[instrument] = {
                    'CE': ce_symbol,
                    'PE': pe_symbol,
                    'strike': atm_strike,
                    'price': price,
                    'expiry': expiry_date,
                    'source': source
                }
                print(f"    Price: ₹{price:,.2f} ({source})")
                print(f"    ATM Strike: {atm_strike}")
                print(f"    CE: {ce_symbol}")
                print(f"    PE: {pe_symbol}")

        print("=" * 60 + "\n")
        return result

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
        """Return 'CE', 'PE', or 'COMMODITY' based on the symbol."""
        sym = symbol.upper().rstrip()
        if sym.endswith('CE'):
            return 'CE'
        elif sym.endswith('PE'):
            return 'PE'
        else:
            return 'COMMODITY'

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

        CE contracts: buy when DI+ crosses above DI-
                      exit when DI+ crosses below DI-
        PE contracts: buy when DI- crosses above DI+
                      exit when DI- crosses below DI+
        COMMODITIES: Always buy on DI+ crossover (long bias)
        """
        sym_type = self._detect_symbol_type(symbol)

        df['di_positive'] = df['di_plus'] > df['di_minus']
        df['di_positive_prev'] = df['di_positive'].shift(1)
        df['valid_time'] = df.index.map(self.is_valid_trading_time)

        crossover_to_positive = df['di_positive'] & ~df['di_positive_prev']
        crossover_to_negative = ~df['di_positive'] & df['di_positive_prev']

        if sym_type == 'PE':
            df['buy_signal'] = (
                crossover_to_negative &
                (df['adx'] >= self.adx_threshold) &
                df['valid_time'] &
                df['adx'].notna())
            df['sell_signal'] = crossover_to_positive
        else:
            # CE or COMMODITY: buy on bullish crossover
            df['buy_signal'] = (
                crossover_to_positive &
                (df['adx'] >= self.adx_threshold) &
                df['valid_time'] &
                df['adx'].notna())
            df['sell_signal'] = crossover_to_negative

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

        df = self.calculate_adx(df)
        df = self.calculate_atr(df)
        df = self.generate_signals(df, symbol)

        df['trading_day'] = df.index.date
        df['is_square_off'] = df.index.map(lambda ts: ts.time() >= self.square_off_time)

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

    # ── Run & Report ─────────────────────────────────────────────────────

    def run_backtest(self, symbols_to_test=None):
        """
        Run backtest on specified symbols or ATM contracts.
        """
        print(f"\n{'='*100}")
        print("STARTING ADX CROSSOVER MONTHLY BACKTEST")
        print(f"{'='*100}")

        if symbols_to_test is None:
            if self.use_atm:
                atm_contracts = self.fetch_atm_contracts(self.instruments)
                symbols_to_test = []
                for inst, data in atm_contracts.items():
                    if data.get('type') == 'commodity':
                        symbols_to_test.append(data['symbol'])
                    else:
                        symbols_to_test.append(data['CE'])
                        symbols_to_test.append(data['PE'])
            else:
                symbols_to_test = []

        for symbol in symbols_to_test:
            try:
                result = self.backtest_single_symbol(symbol)
                if result:
                    self.results[symbol] = result
            except Exception as e:
                print(f"Error backtesting {symbol}: {e}")

        self.print_summary()
        if self.backtest_days:
            self.print_backtest_days_report()

    def print_summary(self):
        if not self.results:
            print("\nNo results to display.")
            return
        print(f"\n{'='*100}")
        print("ADX CROSSOVER MONTHLY RESULTS — ALL SYMBOLS")
        print(f"{'='*100}")
        rows = []
        for symbol, result in self.results.items():
            m = result['metrics']
            clean = symbol.split(':')[1] if ':' in symbol else symbol
            rows.append({
                'Symbol': clean, 'Type': self._detect_symbol_type(symbol),
                'Trades': m['total_trades'],
                'Win Rate': f"{m['win_rate']:.1f}%",
                'Total P&L': f"Rs.{m['total_pnl']:+,.2f}",
                'Avg P&L': f"Rs.{m['avg_pnl']:+,.2f}",
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
        print(f"  Total P&L      : Rs.{total_pnl:+,.2f}")
        if total_trades > 0:
            print(f"  Win Rate       : {total_wins / total_trades * 100:.1f}%")

        os.makedirs('output', exist_ok=True)
        summary_df.to_csv('output/adx_monthly_results.csv', index=False)
        print(f"\n  Results saved to: output/adx_monthly_results.csv")

    def print_backtest_days_report(self):
        if not self.results or self.backtest_days is None:
            return
        print(f"\n{'='*100}")
        print(f"BACKTEST REPORT — LAST {self.backtest_days} DAYS")
        print(f"{'='*100}")
        all_trades = []
        for symbol, result in self.results.items():
            clean = symbol.split(':')[1] if ':' in symbol else symbol
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
        day_df.to_csv(f"output/adx_monthly_{self.backtest_days}days_report.csv", index=False)
        print(f"  Report saved to: output/adx_monthly_{self.backtest_days}days_report.csv")
        print(f"{'='*100}")


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("ADX CROSSOVER STRATEGY — MONTHLY OPTIONS & CONTRACTS")
    print("=" * 100)

    BACKTEST_DAYS = 60

    # Example 1: Backtest with ATM contract selection
    bt = ADXCrossoverMonthlyBacktester(
        data_folder="data/symbolupdate",
        instruments=['BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'GOLD', 'SILVER'],

        # ADX settings
        adx_period=14,
        adx_threshold=20,

        # ATR (for reference)
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

        # Backtest window
        backtest_days=BACKTEST_DAYS,

        # ATM mode
        use_atm=True,
    )

    # Run backtest
    bt.run_backtest()

    # Example 2: Backtest specific symbols (if you have them in your database)
    # bt2 = ADXCrossoverMonthlyBacktester(
    #     data_folder="data/symbolupdate",
    #     symbols=['NSE:BANKNIFTY2601100CE', 'NSE:BANKNIFTY2601100PE'],
    #     backtest_days=30
    # )
    # bt2.run_backtest(['NSE:BANKNIFTY2601100CE', 'NSE:BANKNIFTY2601100PE'])
