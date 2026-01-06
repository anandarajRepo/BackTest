import sqlite3
import pandas as pd
import numpy as np
import json
import glob
import os
from datetime import datetime, time
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
                 min_ha_body_pct=0.15, max_trades_per_day=5):
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
        print(f"  Trading Hours: {avoid_opening_mins} mins after open, avoid {avoid_lunch_start}-{avoid_lunch_end}")
        print(f"  Last Entry: {last_entry_time} IST")
        print(f"  Square-off Time: {square_off_time} IST")
        print(f"  Initial Capital: ₹{self.initial_capital:,}")
        print(f"{'='*100}")

        # Auto-detect symbols if not provided
        if symbols is None:
            print("\nAuto-detecting symbols from databases...")
            self.symbols = self.auto_detect_symbols()
        else:
            self.symbols = symbols

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

    def auto_detect_symbols(self):
        """Auto-detect symbols from databases"""
        all_symbols = set()
        symbol_stats = {}

        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                query = """
                SELECT symbol, COUNT(*) as record_count,
                       MIN(timestamp) as first_record,
                       MAX(timestamp) as last_record
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
                        symbol_stats[symbol] = {
                            'total_records': 0,
                            'databases': []
                        }

                    symbol_stats[symbol]['total_records'] += row['record_count']
                    symbol_stats[symbol]['databases'].append(os.path.basename(db_file))
            except:
                continue

        # Filter and sort symbols
        filtered_symbols = [s for s, stats in symbol_stats.items()
                          if stats['total_records'] >= self.min_data_points]

        return sorted(filtered_symbols,
                     key=lambda s: symbol_stats[s]['total_records'],
                     reverse=True)[:20]

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

        # Backtest loop
        for i in range(len(df)):
            current_time = df.index[i]
            current_price = df.iloc[i]['close']
            current_atr = df.iloc[i]['atr']
            current_adx = df.iloc[i]['adx']
            is_square_off = df.iloc[i]['is_square_off']
            current_day = df.iloc[i]['trading_day']

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
            elif position == 0 and df.iloc[i]['buy_signal'] and not is_square_off:
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
                elif df.iloc[i]['sell_signal']:
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
            'data': df,
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

        summary_df.to_csv('improved_heikin_ashi_results.csv', index=False)
        print(f"\n✅ Results saved to: improved_heikin_ashi_results.csv")


if __name__ == "__main__":
    # CONFIGURATION 1: Conservative (Lower Risk, Higher Win Rate)
    print("\n" + "="*100)
    print("RUNNING IMPROVED STRATEGY - CONSERVATIVE CONFIGURATION")
    print("="*100)

    backtester = ImprovedAdvancedHeikinAshiBacktester(
        data_folder="data/symbolupdate",
        symbols=None,

        # Stricter filters
        ha_smoothing=3,
        adx_period=14,
        adx_threshold=30,  # Increased from 25
        volume_percentile=75,  # Increased from 60
        consecutive_candles=3,  # Increased from 2

        # Tighter risk management
        atr_period=14,
        atr_multiplier=1.5,  # Reduced from 2.0
        breakeven_profit_pct=0.5,  # Reduced from 1.0
        max_loss_pct=0.75,  # NEW: Maximum 0.75% loss per trade
        min_risk_reward=2.0,  # NEW: Minimum 2:1 risk-reward

        # Time filters
        avoid_opening_mins=15,  # NEW: Skip first 15 minutes
        avoid_lunch_start="12:30",  # NEW: Avoid lunch hour
        avoid_lunch_end="13:30",
        last_entry_time="14:45",  # NEW: No entries after 2:45 PM

        # Quality filters
        min_ha_body_pct=0.15,  # NEW: Minimum candle body size
        max_trades_per_day=5,  # NEW: Limit trades per symbol

        initial_capital=100000,
        square_off_time="15:20",
        tick_interval='30s'
    )

    backtester.run_backtest()

    print("\n\n" + "="*100)
    print("COMPARISON: Run original strategy to compare results")
    print("="*100)
