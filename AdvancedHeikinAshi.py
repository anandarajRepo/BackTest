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


class ImprovedAdvancedHeikinAshiBacktester:
    def __init__(self, data_folder="data", symbols=None,
                 ha_smoothing=3, adx_period=14, adx_threshold=30,
                 volume_percentile=75, atr_period=14, atr_multiplier=1.5,
                 breakeven_profit_pct=0.5, consecutive_candles=3,
                 initial_capital=100000, square_off_time="15:20",
                 min_data_points=100, tick_interval=None,
                 max_loss_pct=0.75, min_risk_reward=2.0,
                 avoid_opening_mins=15, avoid_lunch_start="12:30",
                 avoid_lunch_end="13:30", last_entry_time="14:45",
                 min_ha_body_pct=0.15, max_trades_per_day=5,
                 use_nifty_atm=False, nifty_strike_interval=50,
                 backtest_days=None):
        """
        IMPROVED Advanced Heikin Ashi Strategy with Better Risk Management

        KEY IMPROVEMENTS:
        -----------------
        1. Stricter entry filters (reduces false signals)
        2. Tighter stop losses (limits losses)
        3. Faster breakeven (protects profits)
        4. Time-based filters (avoids choppy periods)
        5. Risk-reward validation (quality over quantity)
        6. Trade frequency limits (prevents overtrading)

        NEW PARAMETERS:
        ---------------
        - max_loss_pct: Maximum loss per trade (0.75% default)
        - min_risk_reward: Minimum risk-reward ratio (2.0 = 2:1)
        - avoid_opening_mins: Skip first N minutes (15 mins)
        - avoid_lunch_start/end: Lunch hour to avoid
        - last_entry_time: No new trades after this time
        - min_ha_body_pct: Minimum HA candle body size (% of price)
        - max_trades_per_day: Maximum trades per symbol per day
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()
        self.ha_smoothing = ha_smoothing
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.volume_percentile = volume_percentile
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.breakeven_profit_pct = breakeven_profit_pct / 100
        self.consecutive_candles = consecutive_candles
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.tick_interval = tick_interval

        # NEW: Enhanced risk management parameters
        self.max_loss_pct = max_loss_pct / 100
        self.min_risk_reward = min_risk_reward
        self.avoid_opening_mins = avoid_opening_mins
        self.avoid_lunch_start = self.parse_square_off_time(avoid_lunch_start)
        self.avoid_lunch_end = self.parse_square_off_time(avoid_lunch_end)
        self.last_entry_time = self.parse_square_off_time(last_entry_time)
        self.min_ha_body_pct = min_ha_body_pct / 100
        self.max_trades_per_day = max_trades_per_day

        # Backtest period filter
        self.backtest_days = backtest_days

        # Nifty ATM contract selection
        self.use_nifty_atm = use_nifty_atm
        self.nifty_strike_interval = nifty_strike_interval

        self.results = {}
        self.combined_data = {}

        # Set up IST timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print(f"{'='*100}")
        print(f"IMPROVED ADVANCED HEIKIN ASHI STRATEGY - Enhanced Risk Management")
        print(f"{'='*100}")
        print(f"Strategy Parameters:")
        print(f"  Tick Interval: {self.tick_interval if self.tick_interval else 'Raw tick data'}")
        print(f"  HA Smoothing: {self.ha_smoothing} | Consecutive Candles: {self.consecutive_candles}")
        print(f"  ADX Period: {self.adx_period} | Threshold: {self.adx_threshold} (STRICTER)")
        print(f"  Volume Percentile: {self.volume_percentile}% (HIGHER)")
        print(f"  ATR Period: {self.atr_period} | Multiplier: {self.atr_multiplier}x (TIGHTER)")
        print(f"  Breakeven Trigger: {breakeven_profit_pct}% (FASTER)")
        print(f"  Max Loss Per Trade: {max_loss_pct}% (NEW)")
        print(f"  Min Risk-Reward: {min_risk_reward}:1 (NEW)")
        print(f"  Min HA Body Size: {min_ha_body_pct}% (NEW)")
        print(f"  Max Trades/Day: {max_trades_per_day} (NEW)")
        print(f"  Backtest Period: {'Last ' + str(backtest_days) + ' days' if backtest_days else 'All available data'}")
        print(f"  Trading Hours: {avoid_opening_mins} mins after open, avoid {avoid_lunch_start}-{avoid_lunch_end}")
        print(f"  Last Entry: {last_entry_time} IST")
        print(f"  Square-off Time: {square_off_time} IST")
        print(f"  Initial Capital: ₹{self.initial_capital:,}")
        print(f"  Nifty ATM Mode: {'ENABLED (strike interval: {})'.format(nifty_strike_interval) if use_nifty_atm else 'DISABLED'}")
        print(f"{'='*100}")

        # Determine symbols to backtest
        if symbols is not None:
            self.symbols = symbols
        elif self.use_nifty_atm:
            ce_sym, pe_sym = self.fetch_nifty_atm_contracts()
            if ce_sym and pe_sym:
                self.symbols = [ce_sym, pe_sym]
            else:
                print("ATM fetch failed — no symbols to backtest.")
                self.symbols = []
        else:
            self.symbols = []

        print(f"\nSymbols to backtest: {len(self.symbols)}")

    def parse_square_off_time(self, time_str):
        """Parse time string to time object"""
        try:
            hour, minute = map(int, time_str.split(':'))
            return time(hour, minute)
        except:
            return time(15, 20)

    def find_database_files(self):
        """Find all database files"""
        if not os.path.exists(self.data_folder):
            return []
        db_files = glob.glob(os.path.join(self.data_folder, "*.db"))
        return sorted(db_files)

    # ------------------------------------------------------------------ #
    #  NIFTY ATM CONTRACT SELECTION                                        #
    # ------------------------------------------------------------------ #

    def get_weekly_expiry_date(self, reference_date=None):
        """
        Return the nearest upcoming Nifty weekly expiry date (Tuesday).

        - If reference_date is a Tuesday, that same date is returned
          (today's expiry).
        - If reference_date falls on Wednesday / Thursday / Friday / Saturday / Sunday, the
          *next* Tuesday is returned.

        Parameters
        ----------
        reference_date : datetime.date or None
            Date to compute the expiry from. Defaults to today (IST).

        Returns
        -------
        datetime.date
        """
        if reference_date is None:
            reference_date = datetime.now(self.ist_tz).date()

        # weekday() → Monday=0 … Tuesday=1 … Sunday=6
        days_until_tuesday = (1 - reference_date.weekday()) % 7
        return reference_date + timedelta(days=days_until_tuesday)

    def is_last_tuesday_of_month(self, expiry_date):
        """
        Return True when *expiry_date* is the last Tuesday of its month,
        which corresponds to the Nifty monthly (not weekly) expiry.

        The check is simple: if the Tuesday one week later falls in the
        next month, then the current Tuesday is the last one.

        Parameters
        ----------
        expiry_date : datetime.date

        Returns
        -------
        bool
        """
        return (expiry_date + timedelta(days=7)).month != expiry_date.month

    def get_atm_strike(self, price):
        """
        Round *price* to the nearest Nifty ATM strike.

        Nifty options use ``self.nifty_strike_interval``-point strike
        intervals (default 50).  For example, a Nifty open of 22,313
        rounds to 22,300.

        Parameters
        ----------
        price : float

        Returns
        -------
        int
        """
        interval = self.nifty_strike_interval
        return int(round(price / interval) * interval)

    def build_nifty_option_symbol(self, expiry_date, strike, option_type):
        """
        Build a Fyers NSE option symbol string for a Nifty contract.

        Symbol formats (Fyers convention):
        - Weekly expiry  : ``NSE:NIFTY{YY}{MON}{D}{STRIKE}{CE/PE}``
          (day without leading zero, e.g. ``3`` for the 3rd)
        - Monthly expiry : ``NSE:NIFTY{YY}{MON}{STRIKE}{CE/PE}``
          (last Tuesday of the month — no day component)

        Parameters
        ----------
        expiry_date : datetime.date
        strike      : int
        option_type : str  — ``'CE'`` or ``'PE'``

        Returns
        -------
        str  e.g. ``'NSE:NIFTY25APR1024500CE'``
        """
        yy = expiry_date.strftime('%y')           # '25'
        mon = expiry_date.strftime('%b').upper()   # 'APR'
        day = str(expiry_date.day)                 # '10'  (no leading zero)
        opt = option_type.upper()                  # 'CE' or 'PE'

        if self.is_last_tuesday_of_month(expiry_date):
            # Monthly expiry — no day in symbol
            return f"NSE:NIFTY{yy}{mon}{strike}{opt}"
        else:
            # Weekly expiry — day included
            return f"NSE:NIFTY{yy}{mon}{day}{strike}{opt}"

    def _get_nifty_open_from_fyers(self):
        """
        Fetch the Nifty 50 opening price (first 1-min candle open) for
        today via the Fyers API.

        Requires ``FYERS_CLIENT_ID`` and ``FYERS_ACCESS_TOKEN`` to be set
        in the environment (or a ``.env`` file).

        Returns
        -------
        float or None
            Opening price, or None if credentials are missing / API fails.
        """
        try:
            import os
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
        """
        Fetch the Nifty 50 opening price from the local SQLite database.

        Tries several common symbol names for the Nifty 50 index and
        returns the ``ltp`` of the earliest record found.

        Returns
        -------
        float or None
        """
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
        """
        Dynamically determine the Nifty ATM call and put option symbols
        for the current (or given) week, based on the Nifty 50 Index
        opening price.

        Resolution order
        ----------------
        1. Determine this week's weekly expiry date (nearest Tuesday).
        2. Obtain the Nifty 50 opening price:
           a. Fyers API (live/today)  — preferred.
           b. Local SQLite database   — fallback.
        3. Round to the nearest ``nifty_strike_interval``-point ATM strike.
        4. Build and return CE and PE Fyers option symbols.

        Parameters
        ----------
        reference_date : datetime.date or None
            Override the date used to find the weekly expiry. Defaults to
            today (IST).

        Returns
        -------
        tuple[str, str] or tuple[None, None]
            ``(ce_symbol, pe_symbol)`` on success, ``(None, None)`` if
            the Nifty opening price cannot be determined.

        Example
        -------
        >>> ce, pe = backtester.fetch_nifty_atm_contracts()
        >>> print(ce)   # e.g.  NSE:NIFTY25APR1024500CE
        >>> print(pe)   # e.g.  NSE:NIFTY25APR1024500PE
        """
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

    # ------------------------------------------------------------------ #

    def load_data_from_all_databases(self, symbol):
        """Load and combine data from all databases"""
        combined_df = pd.DataFrame()

        for db_file in self.db_files:
            try:
                df = self.load_data_from_single_db(db_file, symbol)
                if df is not None and not df.empty:
                    df['source_db'] = os.path.basename(db_file)
                    combined_df = pd.concat([combined_df, df], ignore_index=False)
            except:
                continue

        if combined_df.empty:
            return None

        # Sort and remove duplicates
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        # Resample tick data if needed
        if self.tick_interval is not None:
            combined_df = self.resample_tick_data(combined_df, self.tick_interval)

        return combined_df

    def load_data_from_single_db(self, db_path, symbol):
        """Load data from a single database"""
        try:
            conn = sqlite3.connect(db_path)

            check_query = "SELECT COUNT(*) FROM market_data WHERE symbol = ?"
            count_result = pd.read_sql_query(check_query, conn, params=(symbol,))

            if count_result.iloc[0, 0] == 0:
                conn.close()
                return None

            query = """
            SELECT timestamp, symbol, ltp, high_price, low_price, close_price,
                   volume, raw_data
            FROM market_data
            WHERE symbol = ?
            ORDER BY timestamp
            """

            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()

            if df.empty:
                return None

            # Parse raw_data
            def parse_raw_data(raw_data_str):
                try:
                    if raw_data_str:
                        raw_data = json.loads(raw_data_str)
                        return pd.Series({
                            'high_raw': raw_data.get('high_price', np.nan),
                            'low_raw': raw_data.get('low_price', np.nan),
                            'volume_raw': raw_data.get('vol_traded_today', 0)
                        })
                    else:
                        return pd.Series({'high_raw': np.nan, 'low_raw': np.nan, 'volume_raw': 0})
                except:
                    return pd.Series({'high_raw': np.nan, 'low_raw': np.nan, 'volume_raw': 0})

            raw_parsed = df['raw_data'].apply(parse_raw_data)
            df = pd.concat([df, raw_parsed], axis=1)

            # Create OHLC
            df['open'] = df['ltp']
            df['high'] = df['high_raw'].fillna(df['high_price']).fillna(df['ltp'])
            df['low'] = df['low_raw'].fillna(df['low_price']).fillna(df['ltp'])
            df['close'] = df['close_price'].fillna(df['ltp'])
            df['volume'] = df['volume_raw'].fillna(df['volume']).fillna(0)

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            # Remove missing data
            df = df.dropna(subset=['close', 'high', 'low'])

            return df[['open', 'high', 'low', 'close', 'volume']].copy()

        except Exception as e:
            return None

    def resolve_trading_day_cutoff(self, df):
        """
        Return the cutoff date so that only the last ``backtest_days``
        *actual trading days* (dates present in *df*) are included.

        Why: using calendar days fails on weekends / market holidays because
        ``datetime.now() - timedelta(days=N)`` may land on a day with no
        market data.  Instead we look at which dates really have data and
        count backwards N trading sessions from the most-recent one.

        Parameters
        ----------
        df : pd.DataFrame with a DatetimeIndex

        Returns
        -------
        datetime.date
            The earliest trading date to include (inclusive).
        """
        # Unique sorted trading dates present in the data
        trading_dates = sorted(set(df.index.date))

        if not trading_dates:
            # Fallback to calendar-day cutoff if somehow empty
            return (datetime.now(self.ist_tz) - timedelta(days=self.backtest_days)).date()

        n = self.backtest_days
        if n >= len(trading_dates):
            # Requested more days than available — use all data
            return trading_dates[0]

        # The Nth-from-last trading date is the cutoff
        return trading_dates[-n]

    def resample_tick_data(self, df, interval):
        """Resample tick data to specified time interval"""
        if df is None or df.empty:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        resampled = df.resample(interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        resampled = resampled.dropna(subset=['close'])
        resampled['open'] = resampled['open'].fillna(resampled['close'])
        resampled['high'] = resampled['high'].fillna(resampled['close'])
        resampled['low'] = resampled['low'].fillna(resampled['close'])
        resampled['volume'] = resampled['volume'].fillna(0)

        return resampled

    def calculate_heikin_ashi(self, df):
        """Calculate Heikin Ashi candles with smoothing"""
        ha_df = df.copy()

        # Standard Heikin Ashi
        ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_df['ha_open'] = (df['open'] + df['close']) / 2

        for i in range(1, len(ha_df)):
            ha_df.iloc[i, ha_df.columns.get_loc('ha_open')] = (
                ha_df.iloc[i-1]['ha_open'] + ha_df.iloc[i-1]['ha_close']
            ) / 2

        ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
        ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)

        # Apply smoothing
        if self.ha_smoothing > 1:
            ha_df['ha_open_smooth'] = ha_df['ha_open'].ewm(span=self.ha_smoothing, adjust=False).mean()
            ha_df['ha_close_smooth'] = ha_df['ha_close'].ewm(span=self.ha_smoothing, adjust=False).mean()
            ha_df['ha_high_smooth'] = ha_df['ha_high'].ewm(span=self.ha_smoothing, adjust=False).mean()
            ha_df['ha_low_smooth'] = ha_df['ha_low'].ewm(span=self.ha_smoothing, adjust=False).mean()
        else:
            ha_df['ha_open_smooth'] = ha_df['ha_open']
            ha_df['ha_close_smooth'] = ha_df['ha_close']
            ha_df['ha_high_smooth'] = ha_df['ha_high']
            ha_df['ha_low_smooth'] = ha_df['ha_low']

        # Candle properties
        ha_df['ha_bullish'] = ha_df['ha_close_smooth'] > ha_df['ha_open_smooth']
        ha_df['ha_bearish'] = ha_df['ha_close_smooth'] < ha_df['ha_open_smooth']
        ha_df['ha_body'] = abs(ha_df['ha_close_smooth'] - ha_df['ha_open_smooth'])

        # NEW: Calculate body size as percentage of price
        ha_df['ha_body_pct'] = (ha_df['ha_body'] / ha_df['close']) * 100

        return ha_df

    def calculate_adx(self, df):
        """Calculate ADX with directional indicators"""
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        df['dm_plus'] = np.where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
            np.maximum(df['high'] - df['high'].shift(1), 0),
            0
        )

        df['dm_minus'] = np.where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
            np.maximum(df['low'].shift(1) - df['low'], 0),
            0
        )

        alpha = 1 / self.adx_period
        df['tr_smooth'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
        df['dm_plus_smooth'] = df['dm_plus'].ewm(alpha=alpha, adjust=False).mean()
        df['dm_minus_smooth'] = df['dm_minus'].ewm(alpha=alpha, adjust=False).mean()

        df['di_plus'] = 100 * (df['dm_plus_smooth'] / df['tr_smooth'])
        df['di_minus'] = 100 * (df['dm_minus_smooth'] / df['tr_smooth'])

        df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
        df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()

        # NEW: ADX slope for trend strength validation
        df['adx_slope'] = df['adx'].diff()

        return df

    def calculate_atr(self, df):
        """Calculate ATR"""
        if 'tr' not in df.columns:
            df['prev_close'] = df['close'].shift(1)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['prev_close'])
            df['tr3'] = abs(df['low'] - df['prev_close'])
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        df['atr'] = df['tr'].ewm(span=self.atr_period, adjust=False).mean()

        return df

    def identify_good_volume(self, df):
        """Identify high volume periods"""
        volume_threshold = df['volume'].quantile(self.volume_percentile / 100)
        df['good_volume'] = df['volume'] > volume_threshold
        return df

    def is_valid_trading_time(self, timestamp):
        """NEW: Check if time is valid for trading"""
        try:
            current_time = timestamp.time()

            # Market opens at 9:15, avoid first N minutes
            market_open = time(9, 15)
            avoid_until = time(9, 15 + self.avoid_opening_mins)

            # Check all conditions
            if current_time < avoid_until:
                return False

            # Avoid lunch hour
            if self.avoid_lunch_start <= current_time <= self.avoid_lunch_end:
                return False

            # No new entries after last_entry_time
            if current_time >= self.last_entry_time:
                return False

            return True
        except:
            return False

    def generate_signals(self, df):
        """Generate trading signals with enhanced filters"""
        # Count consecutive bullish candles
        df['bullish_streak'] = 0
        streak = 0
        for i in range(len(df)):
            if df.iloc[i]['ha_bullish']:
                streak += 1
            else:
                streak = 0
            df.iloc[i, df.columns.get_loc('bullish_streak')] = streak

        # NEW: Add valid trading time flag
        df['valid_time'] = df.index.map(self.is_valid_trading_time)

        # IMPROVED BUY SIGNAL with stricter filters
        df['buy_signal'] = (
            (df['ha_bullish']) &
            (df['bullish_streak'] >= self.consecutive_candles) &
            (df['adx'] > self.adx_threshold) &
            (df['di_plus'] > df['di_minus']) &  # NEW: Confirm uptrend
            (df['adx_slope'] >= 0) &  # NEW: ADX should be rising or stable
            (df['ha_body_pct'] >= self.min_ha_body_pct) &  # NEW: Significant body
            (df['good_volume']) &
            (df['valid_time']) &  # NEW: Time filter
            (~df['adx'].isna()) &
            (~df['atr'].isna())
        )

        # SELL SIGNAL
        df['sell_signal'] = (
            (df['ha_bearish']) &
            (df['ha_bullish'].shift(1))
        )

        return df

    def calculate_risk_reward(self, entry_price, stop_price, atr, adx):
        """NEW: Calculate potential risk-reward ratio"""
        risk = entry_price - stop_price

        # Potential target based on ADX strength
        if adx > 40:
            target_multiplier = 3.0  # Strong trend
        elif adx > 30:
            target_multiplier = 2.5  # Good trend
        else:
            target_multiplier = 2.0  # Moderate trend

        potential_target = entry_price + (atr * target_multiplier)
        reward = potential_target - entry_price

        if risk <= 0:
            return 0

        return reward / risk

    def is_square_off_time(self, timestamp):
        """Check if square-off time"""
        try:
            current_time = timestamp.time()
            return current_time >= self.square_off_time
        except:
            return False

    def backtest_single_symbol(self, symbol):
        """Backtest with improved risk management"""
        print(f"\n{'='*100}")
        print(f"Backtesting: {symbol}")
        print(f"{'='*100}")

        df = self.load_data_from_all_databases(symbol)

        if df is None or len(df) < max(self.adx_period, self.atr_period, self.ha_smoothing) * 3:
            print(f"Insufficient data for {symbol}. Skipping.")
            return None

        self.combined_data[symbol] = df.copy()

        # Calculate indicators
        df = self.calculate_heikin_ashi(df)
        df = self.calculate_adx(df)
        df = self.calculate_atr(df)
        df = self.identify_good_volume(df)
        df = self.generate_signals(df)

        df['trading_day'] = df.index.date
        df['is_square_off'] = df.index.map(self.is_square_off_time)

        # ── Backtest-days window filter ────────────────────────────────────
        # Indicators are computed on the full history so that values at the
        # start of the window are accurate.  The trading loop only runs over
        # the requested window.
        if self.backtest_days is not None:
            # Use actual trading days present in the data instead of calendar
            # days so that weekends / market holidays do not shift the window.
            cutoff_date = self.resolve_trading_day_cutoff(df)
            backtest_df = df[df.index.date >= cutoff_date].copy()
            if backtest_df.empty:
                print(f"  No data found within the last {self.backtest_days} trading days for {symbol}. Skipping.")
                return None
            first_day = backtest_df.index.date[0]
            last_day  = backtest_df.index.date[-1]
            actual_days = len(set(backtest_df.index.date))
            print(f"  Backtest window : {first_day} → {last_day} ({actual_days} trading day(s), requested {self.backtest_days})")
        else:
            backtest_df = df
        # ──────────────────────────────────────────────────────────────────

        # Trading variables
        cash = self.initial_capital
        position = 0
        entry_price = 0
        trailing_stop = 0
        max_stop = 0  # NEW: Maximum stop based on % loss
        breakeven_triggered = False
        trades = []
        entry_time = None
        entry_atr = 0
        trade_number = 0
        trades_today = {}  # NEW: Track trades per day

        # Backtest loop (runs over backtest_df — the filtered window)
        for i in range(len(backtest_df)):
            current_time = backtest_df.index[i]
            current_price = backtest_df.iloc[i]['close']
            current_atr = backtest_df.iloc[i]['atr']
            current_adx = backtest_df.iloc[i]['adx']
            is_square_off = backtest_df.iloc[i]['is_square_off']
            current_day = backtest_df.iloc[i]['trading_day']

            # Initialize day counter
            if current_day not in trades_today:
                trades_today[current_day] = 0

            # Square off at EOD
            if position == 1 and is_square_off:
                shares = int(cash / entry_price) if entry_price > 0 else 0
                trade_pnl = shares * (current_price - entry_price)
                trade_return = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

                trade_number += 1
                duration = (current_time - entry_time).total_seconds() / 60

                print(f"\n[Trade #{trade_number}] SQUARE-OFF @ {current_time.strftime('%H:%M:%S')}")
                print(f"  Entry: {entry_time.strftime('%H:%M:%S')} @ ₹{entry_price:.2f}")
                print(f"  Exit: ₹{current_price:.2f} | P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
                print(f"  {'✅ PROFIT' if trade_pnl > 0 else '❌ LOSS'}")

                trades.append({
                    'trade_num': trade_number,
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': shares,
                    'duration_minutes': duration,
                    'pnl': trade_pnl,
                    'return_pct': trade_return,
                    'exit_reason': 'SQUARE_OFF',
                    'adx': current_adx,
                    'atr': current_atr,
                    'breakeven_triggered': breakeven_triggered
                })

                position = 0
                breakeven_triggered = False
                trades_today[current_day] += 1

            # Entry signal with enhanced validation
            elif position == 0 and backtest_df.iloc[i]['buy_signal'] and not is_square_off:
                # NEW: Check trade frequency limit
                if trades_today[current_day] >= self.max_trades_per_day:
                    continue

                # Calculate stops
                atr_stop = current_price - (current_atr * self.atr_multiplier)
                max_loss_stop = current_price * (1 - self.max_loss_pct)
                initial_stop = max(atr_stop, max_loss_stop)

                # NEW: Validate risk-reward ratio
                risk_reward = self.calculate_risk_reward(current_price, initial_stop, current_atr, current_adx)

                if risk_reward < self.min_risk_reward:
                    continue  # Skip trade with poor risk-reward

                position = 1
                entry_price = current_price
                entry_time = current_time
                entry_atr = current_atr
                trailing_stop = initial_stop
                max_stop = max_loss_stop
                breakeven_triggered = False

            # Exit management
            elif position == 1:
                # Update trailing stop
                if current_price > entry_price:
                    atr_stop = current_price - (current_atr * self.atr_multiplier)

                    # Breakeven protection (faster trigger)
                    if current_price >= entry_price * (1 + self.breakeven_profit_pct):
                        if not breakeven_triggered:
                            breakeven_triggered = True
                        trailing_stop = max(trailing_stop, entry_price, atr_stop)
                    else:
                        trailing_stop = max(trailing_stop, atr_stop)

                    # Ensure we don't violate max loss
                    trailing_stop = max(trailing_stop, max_stop)

                exit_signal = False
                exit_reason = ""

                # Check exits
                if current_price <= trailing_stop:
                    exit_signal = True
                    exit_reason = "STOP_LOSS" if current_price < entry_price else "TRAILING_STOP"
                elif backtest_df.iloc[i]['sell_signal']:
                    exit_signal = True
                    exit_reason = "HA_BEARISH"

                if exit_signal:
                    shares = int(cash / entry_price) if entry_price > 0 else 0
                    trade_pnl = shares * (current_price - entry_price)
                    trade_return = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

                    trade_number += 1
                    duration = (current_time - entry_time).total_seconds() / 60

                    print(f"\n[Trade #{trade_number}] {exit_reason}")
                    print(f"  Entry: {entry_time.strftime('%H:%M:%S')} @ ₹{entry_price:.2f}")
                    print(f"  Exit: {current_time.strftime('%H:%M:%S')} @ ₹{current_price:.2f}")
                    print(f"  Duration: {duration:.1f} min | P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
                    print(f"  ADX: {current_adx:.1f} | ATR: ₹{current_atr:.2f}")
                    print(f"  {'✅ PROFIT' if trade_pnl > 0 else '❌ LOSS'}")

                    trades.append({
                        'trade_num': trade_number,
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': shares,
                        'duration_minutes': duration,
                        'pnl': trade_pnl,
                        'return_pct': trade_return,
                        'exit_reason': exit_reason,
                        'adx': current_adx,
                        'atr': current_atr,
                        'breakeven_triggered': breakeven_triggered
                    })

                    position = 0
                    breakeven_triggered = False
                    trades_today[current_day] += 1

        metrics = self.calculate_metrics(trades)

        return {
            'symbol': symbol,
            'data': backtest_df,
            'trades': trades,
            'metrics': metrics
        }

    def calculate_metrics(self, trades):
        """Calculate performance metrics"""
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
                'max_consecutive_losses': 0
            }

        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        losing_trades = sum(1 for t in trades if t['pnl'] <= 0)

        total_pnl = sum(t['pnl'] for t in trades)
        avg_pnl = total_pnl / total_trades

        total_return = sum(t['return_pct'] for t in trades)
        avg_return = total_return / total_trades

        best_trade = max(t['pnl'] for t in trades)
        worst_trade = min(t['pnl'] for t in trades)

        avg_duration = sum(t['duration_minutes'] for t in trades) / total_trades

        # NEW: Additional metrics
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] <= 0]

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Calculate max consecutive losses
        max_consecutive = 0
        current_consecutive = 0
        for t in trades:
            if t['pnl'] <= 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
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
            'max_consecutive_losses': max_consecutive
        }

    def print_backtest_days_report(self):
        """Print a detailed per-day breakdown when backtest_days is set."""
        if not self.results or self.backtest_days is None:
            return

        print(f"\n{'='*100}")
        print(f"BACKTEST REPORT — LAST {self.backtest_days} DAYS")
        print(f"{'='*100}")

        # ── Collect all trades, tagged with their symbol ──────────────────
        all_trades = []
        for symbol, result in self.results.items():
            clean = (symbol.split(':')[1].replace('-EQ', '')
                     if ':' in symbol else symbol.replace('-EQ', ''))
            for t in result['trades']:
                all_trades.append({**t, 'symbol': clean})

        if not all_trades:
            print("No trades were executed in the specified period.")
            return

        # ── Per-day summary ───────────────────────────────────────────────
        trades_by_day = {}
        for t in all_trades:
            d = t['entry_time'].date()
            trades_by_day.setdefault(d, []).append(t)

        print(f"\n{'─'*100}")
        print("DAILY SUMMARY")
        print(f"{'─'*100}")

        day_rows = []
        for d in sorted(trades_by_day):
            day_trades = trades_by_day[d]
            day_pnl    = sum(t['pnl'] for t in day_trades)
            wins       = sum(1 for t in day_trades if t['pnl'] > 0)
            losses     = len(day_trades) - wins
            win_rate   = wins / len(day_trades) * 100
            status     = 'PROFIT' if day_pnl > 0 else ('LOSS' if day_pnl < 0 else 'BREAKEVEN')
            day_rows.append({
                'Date'      : d.strftime('%Y-%m-%d (%a)'),
                'Trades'    : len(day_trades),
                'Wins'      : wins,
                'Losses'    : losses,
                'Win Rate'  : f"{win_rate:.1f}%",
                'Day P&L'   : f"Rs.{day_pnl:+,.2f}",
                'Status'    : status,
            })

        day_df = pd.DataFrame(day_rows)
        print(day_df.to_string(index=False))

        # ── Per-symbol per-day breakdown ──────────────────────────────────
        print(f"\n{'─'*100}")
        print("PER-SYMBOL DAILY BREAKDOWN")
        print(f"{'─'*100}")

        sym_day_rows = []
        for symbol, result in self.results.items():
            clean = (symbol.split(':')[1].replace('-EQ', '')
                     if ':' in symbol else symbol.replace('-EQ', ''))
            sym_by_day = {}
            for t in result['trades']:
                d = t['entry_time'].date()
                sym_by_day.setdefault(d, []).append(t)
            for d in sorted(sym_by_day):
                ts   = sym_by_day[d]
                pnl  = sum(t['pnl'] for t in ts)
                wins = sum(1 for t in ts if t['pnl'] > 0)
                sym_day_rows.append({
                    'Date'   : d.strftime('%Y-%m-%d'),
                    'Symbol' : clean,
                    'Trades' : len(ts),
                    'Wins'   : wins,
                    'Losses' : len(ts) - wins,
                    'P&L'    : f"Rs.{pnl:+,.2f}",
                })

        if sym_day_rows:
            sym_day_df = pd.DataFrame(sym_day_rows)
            print(sym_day_df.to_string(index=False))

        # ── Period-level summary ──────────────────────────────────────────
        total_pnl        = sum(t['pnl'] for t in all_trades)
        total_wins       = sum(1 for t in all_trades if t['pnl'] > 0)
        profitable_days  = sum(1 for ts in trades_by_day.values()
                               if sum(t['pnl'] for t in ts) > 0)
        avg_daily_pnl    = total_pnl / len(trades_by_day) if trades_by_day else 0
        best_day_pnl     = max(sum(t['pnl'] for t in ts)
                               for ts in trades_by_day.values())
        worst_day_pnl    = min(sum(t['pnl'] for t in ts)
                               for ts in trades_by_day.values())

        print(f"\n{'─'*100}")
        print(f"PERIOD SUMMARY  (Last {self.backtest_days} days)")
        print(f"{'─'*100}")
        print(f"  Trading Days with Activity : {len(trades_by_day)}")
        print(f"  Total Trades               : {len(all_trades)}")
        print(f"  Total P&L                  : Rs.{total_pnl:+,.2f}")
        print(f"  Average Daily P&L          : Rs.{avg_daily_pnl:+,.2f}")
        print(f"  Best Day P&L               : Rs.{best_day_pnl:+,.2f}")
        print(f"  Worst Day P&L              : Rs.{worst_day_pnl:+,.2f}")
        print(f"  Overall Win Rate           : {(total_wins / len(all_trades) * 100):.1f}%")
        print(f"  Profitable Days            : {profitable_days}/{len(trades_by_day)}")

        # ── Save to CSV ───────────────────────────────────────────────────
        os.makedirs('output', exist_ok=True)
        report_path = f"output/backtest_{self.backtest_days}days_report.csv"
        day_df.to_csv(report_path, index=False)
        print(f"\n  Report saved to : {report_path}")

        if sym_day_rows:
            detail_path = f"output/backtest_{self.backtest_days}days_symbol_detail.csv"
            sym_day_df.to_csv(detail_path, index=False)
            print(f"  Detail saved to : {detail_path}")

        print(f"{'='*100}")

    def run_backtest(self):
        """Run backtest for all symbols"""
        print(f"\n{'='*100}")
        print("STARTING IMPROVED HEIKIN ASHI BACKTEST")
        print(f"{'='*100}")

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
        """Print enhanced summary"""
        if not self.results:
            print("\nNo results to display.")
            return

        print(f"\n{'='*100}")
        print("IMPROVED STRATEGY RESULTS - ALL SYMBOLS")
        print(f"{'='*100}")

        summary_data = []
        for symbol, result in self.results.items():
            metrics = result['metrics']
            clean_symbol = symbol.split(':')[1].replace('-EQ', '') if ':' in symbol else symbol.replace('-EQ', '')

            summary_data.append({
                'Symbol': clean_symbol,
                'Trades': metrics['total_trades'],
                'Win Rate': f"{metrics['win_rate']:.1f}%",
                'Total P&L': f"₹{metrics['total_pnl']:.2f}",
                'Avg P&L': f"₹{metrics['avg_pnl']:.2f}",
                'Profit Factor': f"{metrics['profit_factor']:.2f}",
                'Max Losses': metrics['max_consecutive_losses'],
                'Avg Duration': f"{metrics['avg_duration']:.1f}m"
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        # Overall stats
        print(f"\n{'='*100}")
        print("OVERALL STATISTICS")
        print(f"{'='*100}")

        total_trades = sum(r['metrics']['total_trades'] for r in self.results.values())
        total_pnl = sum(r['metrics']['total_pnl'] for r in self.results.values())
        winning_trades = sum(r['metrics']['winning_trades'] for r in self.results.values())
        profitable_symbols = sum(1 for r in self.results.values() if r['metrics']['total_pnl'] > 0)

        print(f"Symbols Tested: {len(self.results)}")
        print(f"Total Trades: {total_trades}")
        print(f"Total P&L: ₹{total_pnl:.2f}")
        print(f"Overall Win Rate: {(winning_trades / total_trades * 100):.1f}%")
        print(f"Profitable Symbols: {profitable_symbols}/{len(self.results)}")

        os.makedirs('output', exist_ok=True)
        summary_df.to_csv('output/improved_heikin_ashi_results.csv', index=False)
        print(f"\n✅ Results saved to: output/improved_heikin_ashi_results.csv")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # CONFIGURATION: Nifty ATM Options — dynamic weekly contract fetch
    #
    # Set use_nifty_atm=True so that the backtester automatically:
    #   1. Finds this week's Tuesday expiry date.
    #   2. Fetches the Nifty 50 Index opening price (Fyers API → DB).
    #   3. Rounds to the nearest 50-pt ATM strike.
    #   4. Uses the resulting CE and PE symbols as the backtest universe.
    #
    # Requires FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN in .env for live
    # price fetching; falls back to local database otherwise.
    #
    # backtest_days (optional)
    # ─────────────────────────
    # Set backtest_days=N to restrict the backtest to the last N calendar
    # days only.  Indicators (ADX, ATR, HA) are still computed on the full
    # history so warm-up is accurate; only the trade-execution loop runs
    # over the N-day window.
    #
    # Examples:
    #   backtest_days=15   → last 15 calendar days
    #   backtest_days=30   → last 30 calendar days
    #   backtest_days=None → use all available data  (default)
    #
    # A detailed per-day report is printed and saved to:
    #   output/backtest_<N>days_report.csv
    #   output/backtest_<N>days_symbol_detail.csv
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("RUNNING IMPROVED STRATEGY — NIFTY ATM OPTIONS (DYNAMIC WEEKLY CONTRACTS)")
    print("=" * 100)

    # ── Change backtest_days to the number of past days you want to test ──
    BACKTEST_DAYS = 15   # e.g. 15 → last 15 calendar days; None → all data

    atm_backtester = ImprovedAdvancedHeikinAshiBacktester(
        data_folder="data/symbolupdate",
        symbols=None,           # ATM symbols resolved automatically

        # Nifty ATM contract selection
        use_nifty_atm=True,     # Enable dynamic ATM contract fetching
        nifty_strike_interval=50,  # Nifty options trade in 50-pt steps

        # Stricter filters
        ha_smoothing=3,
        adx_period=14,
        adx_threshold=30,
        volume_percentile=75,
        consecutive_candles=3,

        # Tighter risk management
        atr_period=14,
        atr_multiplier=1.5,
        breakeven_profit_pct=0.5,
        max_loss_pct=0.75,
        min_risk_reward=2.0,

        # Time filters
        avoid_opening_mins=15,
        avoid_lunch_start="12:30",
        avoid_lunch_end="13:30",
        last_entry_time="14:45",

        # Quality filters
        min_ha_body_pct=0.15,
        max_trades_per_day=5,

        initial_capital=100000,
        square_off_time="15:20",
        tick_interval='30s',

        # ── Backtest window ──────────────────────────────────────────────
        backtest_days=BACKTEST_DAYS,  # None = all data; N = last N days
    )

    atm_backtester.run_backtest()
