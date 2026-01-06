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


class HorizontalFadeBreakoutBacktester:
    def __init__(self, data_folder="data", symbols=None,
                 box_period=20, box_threshold_pct=1.5, min_box_candles=6,
                 breakout_confirmation_pct=0.3, volume_threshold_mult=1.5,
                 atr_period=14, stop_loss_atr_mult=2.0, target_atr_mult=3.0,
                 fade_rsi_period=14, fade_rsi_threshold=65,
                 initial_capital=100000, square_off_time="15:20",
                 min_data_points=100, tick_interval='1min',
                 avoid_opening_mins=15, last_entry_time="14:45",
                 max_trades_per_day=5, min_risk_reward=1.5,
                 false_breakout_threshold=3):
        """
        Horizontal Fade / Box Theory Breakout Strategy for Intraday Trading

        STRATEGY CONCEPT:
        ----------------
        This strategy identifies consolidation boxes (horizontal ranges) where price
        moves sideways, then trades breakouts from these zones. It also includes
        fade logic to trade against false breakouts in overbought/oversold conditions.

        KEY COMPONENTS:
        ---------------
        1. Box/Range Identification
           - Detect periods where price stays within a tight horizontal range
           - Minimum number of candles required in the box
           - Box defined by support (low) and resistance (high) levels

        2. Breakout Detection
           - Upward breakout: Price closes above resistance
           - Downward breakout: Price closes below support
           - Volume confirmation required
           - Breakout must be sustained (not immediately reversed)

        3. Fade Logic (Optional)
           - Trade against false breakouts when RSI is extreme
           - Mean reversion play when breakout lacks momentum

        4. Risk Management
           - ATR-based stop losses
           - ATR-based profit targets
           - Time-based filters
           - Trade frequency limits

        ENTRY SIGNALS:
        --------------
        BREAKOUT LONG:
        - Price in consolidation box for minimum period
        - Price breaks above resistance with volume
        - Breakout confirmed (sustained move)
        - Valid trading time

        BREAKOUT SHORT:
        - Price in consolidation box for minimum period
        - Price breaks below support with volume
        - Breakout confirmed (sustained move)
        - Valid trading time

        FADE LONG (Mean Reversion):
        - Price breaks below support
        - RSI oversold (< 35)
        - Weak volume (likely false breakout)

        FADE SHORT (Mean Reversion):
        - Price breaks above resistance
        - RSI overbought (> 65)
        - Weak volume (likely false breakout)

        EXIT SIGNALS:
        -------------
        - ATR-based stop loss hit
        - ATR-based profit target hit
        - End of day square-off

        PARAMETERS:
        -----------
        - box_period: Lookback period for box detection (default: 20)
        - box_threshold_pct: Max range % for box identification (default: 1.5%)
        - min_box_candles: Minimum candles in box (default: 6)
        - breakout_confirmation_pct: % move required for breakout (default: 0.3%)
        - volume_threshold_mult: Volume multiplier for confirmation (default: 1.5x)
        - atr_period: ATR calculation period (default: 14)
        - stop_loss_atr_mult: Stop loss in ATR multiples (default: 2.0)
        - target_atr_mult: Target in ATR multiples (default: 3.0)
        - fade_rsi_period: RSI period for fade signals (default: 14)
        - fade_rsi_threshold: RSI threshold for fade (default: 65)
        - avoid_opening_mins: Skip first N minutes (default: 15)
        - last_entry_time: Last entry time (default: "14:45")
        - max_trades_per_day: Maximum trades per day (default: 5)
        - min_risk_reward: Minimum risk-reward ratio (default: 1.5)
        - false_breakout_threshold: Candles to wait for breakout confirmation (default: 3)
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()

        # Box detection parameters
        self.box_period = box_period
        self.box_threshold_pct = box_threshold_pct / 100
        self.min_box_candles = min_box_candles

        # Breakout parameters
        self.breakout_confirmation_pct = breakout_confirmation_pct / 100
        self.volume_threshold_mult = volume_threshold_mult
        self.false_breakout_threshold = false_breakout_threshold

        # Fade parameters
        self.fade_rsi_period = fade_rsi_period
        self.fade_rsi_threshold = fade_rsi_threshold

        # ATR parameters
        self.atr_period = atr_period
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.target_atr_mult = target_atr_mult

        # Trading parameters
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.tick_interval = tick_interval

        # Risk management
        self.avoid_opening_mins = avoid_opening_mins
        self.last_entry_time = self.parse_square_off_time(last_entry_time)
        self.max_trades_per_day = max_trades_per_day
        self.min_risk_reward = min_risk_reward

        # Results storage
        self.results = {}
        self.combined_data = {}

        # Timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print(f"{'='*100}")
        print(f"HORIZONTAL FADE / BOX THEORY BREAKOUT STRATEGY - INTRADAY TRADING")
        print(f"{'='*100}")
        print(f"Strategy Parameters:")
        print(f"  Tick Interval: {self.tick_interval}")
        print(f"  Box Detection: {self.box_period} period, {box_threshold_pct}% range, {self.min_box_candles} min candles")
        print(f"  Breakout Confirmation: {breakout_confirmation_pct}% move, {self.volume_threshold_mult}x volume")
        print(f"  False Breakout Filter: {self.false_breakout_threshold} candles")
        print(f"  Fade Logic: RSI {self.fade_rsi_period} period, threshold {self.fade_rsi_threshold}")
        print(f"  ATR Period: {self.atr_period}")
        print(f"  Stop Loss: {self.stop_loss_atr_mult}x ATR")
        print(f"  Target: {self.target_atr_mult}x ATR")
        print(f"  Max Trades/Day: {self.max_trades_per_day}")
        print(f"  Min Risk-Reward: {self.min_risk_reward}:1")
        print(f"  Avoid Opening: {self.avoid_opening_mins} minutes")
        print(f"  Last Entry: {last_entry_time}")
        print(f"  Square-off: {square_off_time}")
        print(f"  Initial Capital: ₹{self.initial_capital:,}")
        print(f"{'='*100}")

        # Auto-detect symbols
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

        # Resample tick data
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

    def calculate_atr(self, df):
        """Calculate Average True Range"""
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        df['atr'] = df['tr'].ewm(span=self.atr_period, adjust=False).mean()

        return df

    def calculate_rsi(self, df):
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.fade_rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.fade_rsi_period).mean()

        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    def detect_boxes(self, df):
        """Detect horizontal consolidation boxes/ranges"""
        # Calculate rolling high and low over box_period
        df['box_high'] = df['high'].rolling(window=self.box_period).max()
        df['box_low'] = df['low'].rolling(window=self.box_period).min()
        df['box_range'] = df['box_high'] - df['box_low']
        df['box_midpoint'] = (df['box_high'] + df['box_low']) / 2

        # Calculate range as percentage of midpoint
        df['box_range_pct'] = (df['box_range'] / df['box_midpoint']) * 100

        # Box is active when range is tight
        df['in_box'] = df['box_range_pct'] < (self.box_threshold_pct * 100)

        # Count consecutive candles in box
        df['box_count'] = 0
        count = 0
        for i in range(len(df)):
            if df.iloc[i]['in_box']:
                count += 1
            else:
                count = 0
            df.iloc[i, df.columns.get_loc('box_count')] = count

        # Box is valid only after minimum candles
        df['valid_box'] = df['box_count'] >= self.min_box_candles

        return df

    def detect_breakouts(self, df):
        """Detect breakouts from consolidation boxes"""
        # Calculate average volume for comparison
        df['avg_volume'] = df['volume'].rolling(window=self.box_period).mean()
        df['volume_surge'] = df['volume'] > (df['avg_volume'] * self.volume_threshold_mult)

        # Previous candle was in valid box
        df['prev_valid_box'] = df['valid_box'].shift(1)

        # Breakout above resistance
        df['breakout_up'] = (
            (df['prev_valid_box']) &
            (df['close'] > df['box_high'].shift(1) * (1 + self.breakout_confirmation_pct)) &
            (df['volume_surge'])
        )

        # Breakout below support
        df['breakout_down'] = (
            (df['prev_valid_box']) &
            (df['close'] < df['box_low'].shift(1) * (1 - self.breakout_confirmation_pct)) &
            (df['volume_surge'])
        )

        # Fade signals (mean reversion on false breakouts)
        df['fade_long'] = (
            (df['prev_valid_box']) &
            (df['close'] < df['box_low'].shift(1)) &
            (df['rsi'] < (100 - self.fade_rsi_threshold)) &
            (~df['volume_surge'])  # Weak volume suggests false breakout
        )

        df['fade_short'] = (
            (df['prev_valid_box']) &
            (df['close'] > df['box_high'].shift(1)) &
            (df['rsi'] > self.fade_rsi_threshold) &
            (~df['volume_surge'])  # Weak volume suggests false breakout
        )

        return df

    def is_valid_trading_time(self, timestamp):
        """Check if time is valid for trading"""
        try:
            current_time = timestamp.time()

            # Market opens at 9:15, avoid first N minutes
            market_open = time(9, 15)
            avoid_until = time(9, 15 + self.avoid_opening_mins)

            # Check conditions
            if current_time < avoid_until:
                return False

            # No new entries after last_entry_time
            if current_time >= self.last_entry_time:
                return False

            return True
        except:
            return False

    def generate_signals(self, df):
        """Generate trading signals"""
        # Valid trading time
        df['valid_time'] = df.index.map(self.is_valid_trading_time)

        # LONG SIGNAL: Breakout above resistance with volume
        df['long_signal'] = (
            (df['breakout_up']) &
            (df['valid_time']) &
            (~df['atr'].isna()) &
            (~df['rsi'].isna())
        )

        # SHORT SIGNAL: Breakout below support with volume
        df['short_signal'] = (
            (df['breakout_down']) &
            (df['valid_time']) &
            (~df['atr'].isna()) &
            (~df['rsi'].isna())
        )

        # FADE LONG SIGNAL: Mean reversion on false breakdown
        df['fade_long_signal'] = (
            (df['fade_long']) &
            (df['valid_time']) &
            (~df['atr'].isna()) &
            (~df['rsi'].isna())
        )

        # FADE SHORT SIGNAL: Mean reversion on false breakout
        df['fade_short_signal'] = (
            (df['fade_short']) &
            (df['valid_time']) &
            (~df['atr'].isna()) &
            (~df['rsi'].isna())
        )

        return df

    def is_square_off_time(self, timestamp):
        """Check if square-off time"""
        try:
            current_time = timestamp.time()
            return current_time >= self.square_off_time
        except:
            return False

    def backtest_single_symbol(self, symbol):
        """Backtest horizontal fade breakout strategy for a symbol"""
        print(f"\n{'='*100}")
        print(f"Backtesting: {symbol}")
        print(f"{'='*100}")

        df = self.load_data_from_all_databases(symbol)

        if df is None or len(df) < max(self.box_period, self.atr_period, self.fade_rsi_period) * 3:
            print(f"Insufficient data for {symbol}. Skipping.")
            return None

        self.combined_data[symbol] = df.copy()

        # Calculate all indicators
        df = self.calculate_atr(df)
        df = self.calculate_rsi(df)
        df = self.detect_boxes(df)
        df = self.detect_breakouts(df)
        df = self.generate_signals(df)

        df['trading_day'] = df.index.date
        df['is_square_off'] = df.index.map(self.is_square_off_time)

        # Trading variables
        cash = self.initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        stop_loss = 0
        target = 0
        trades = []
        entry_time = None
        entry_atr = 0
        entry_type = ""  # "breakout" or "fade"
        trade_number = 0
        trades_today = {}

        # Breakout confirmation tracking
        breakout_pending = False
        breakout_direction = 0
        breakout_start_time = None
        breakout_candles_count = 0

        # Backtest loop
        for i in range(len(df)):
            current_time = df.index[i]
            current_price = df.iloc[i]['close']
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            current_atr = df.iloc[i]['atr']
            is_square_off = df.iloc[i]['is_square_off']
            current_day = df.iloc[i]['trading_day']

            # Initialize day counter
            if current_day not in trades_today:
                trades_today[current_day] = 0

            # Square off at EOD
            if position != 0 and is_square_off:
                shares = trades[-1]['shares']

                if position == 1:  # Long position
                    trade_pnl = shares * (current_price - entry_price)
                else:  # Short position
                    trade_pnl = shares * (entry_price - current_price)

                trade_return = (trade_pnl / (shares * entry_price)) * 100

                trade_number += 1
                duration = (current_time - entry_time).total_seconds() / 60

                print(f"\n[Trade #{trade_number}] SQUARE-OFF @ {current_time.strftime('%H:%M:%S')}")
                print(f"  Type: {entry_type.upper()} | Direction: {'LONG' if position == 1 else 'SHORT'}")
                print(f"  Entry: {entry_time.strftime('%H:%M:%S')} @ ₹{entry_price:.2f}")
                print(f"  Exit: ₹{current_price:.2f} | P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
                print(f"  {'✅ PROFIT' if trade_pnl > 0 else '❌ LOSS'}")

                trades[-1].update({
                    'exit_time': current_time,
                    'exit_price': current_price,
                    'pnl': trade_pnl,
                    'return_pct': trade_return,
                    'exit_reason': 'SQUARE_OFF',
                    'duration_minutes': duration
                })

                position = 0
                trades_today[current_day] += 1

            # Breakout confirmation logic
            elif breakout_pending:
                breakout_candles_count += 1

                # Check if breakout is sustained
                if breakout_direction == 1:  # Up breakout
                    if current_low < df.iloc[i - breakout_candles_count]['box_high']:
                        # Failed breakout
                        breakout_pending = False
                        breakout_candles_count = 0
                    elif breakout_candles_count >= self.false_breakout_threshold:
                        # Confirmed breakout - enter long
                        breakout_pending = False
                        breakout_candles_count = 0

                        if trades_today[current_day] < self.max_trades_per_day:
                            stop_loss_price = current_price - (current_atr * self.stop_loss_atr_mult)
                            target_price = current_price + (current_atr * self.target_atr_mult)

                            risk = current_price - stop_loss_price
                            reward = target_price - current_price
                            risk_reward = reward / risk if risk > 0 else 0

                            if risk_reward >= self.min_risk_reward:
                                position = 1
                                entry_price = current_price
                                entry_time = current_time
                                entry_atr = current_atr
                                entry_type = "breakout"
                                stop_loss = stop_loss_price
                                target = target_price

                                shares = int(cash / entry_price)

                                if shares > 0:
                                    trades.append({
                                        'trade_num': trade_number + 1,
                                        'direction': 'LONG',
                                        'entry_type': entry_type,
                                        'entry_time': entry_time,
                                        'entry_price': entry_price,
                                        'stop_loss': stop_loss,
                                        'target': target,
                                        'shares': shares,
                                        'entry_atr': entry_atr,
                                        'entry_rsi': df.iloc[i]['rsi'],
                                        'risk_reward': risk_reward
                                    })

                                    print(f"\n[BREAKOUT LONG ENTRY] {current_time.strftime('%H:%M:%S')}")
                                    print(f"  Price: ₹{entry_price:.2f} | Stop: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                                    print(f"  Shares: {shares} | R:R = 1:{risk_reward:.2f}")

                elif breakout_direction == -1:  # Down breakout
                    if current_high > df.iloc[i - breakout_candles_count]['box_low']:
                        # Failed breakout
                        breakout_pending = False
                        breakout_candles_count = 0
                    elif breakout_candles_count >= self.false_breakout_threshold:
                        # Confirmed breakout - enter short
                        breakout_pending = False
                        breakout_candles_count = 0

                        if trades_today[current_day] < self.max_trades_per_day:
                            stop_loss_price = current_price + (current_atr * self.stop_loss_atr_mult)
                            target_price = current_price - (current_atr * self.target_atr_mult)

                            risk = stop_loss_price - current_price
                            reward = current_price - target_price
                            risk_reward = reward / risk if risk > 0 else 0

                            if risk_reward >= self.min_risk_reward:
                                position = -1
                                entry_price = current_price
                                entry_time = current_time
                                entry_atr = current_atr
                                entry_type = "breakout"
                                stop_loss = stop_loss_price
                                target = target_price

                                shares = int(cash / entry_price)

                                if shares > 0:
                                    trades.append({
                                        'trade_num': trade_number + 1,
                                        'direction': 'SHORT',
                                        'entry_type': entry_type,
                                        'entry_time': entry_time,
                                        'entry_price': entry_price,
                                        'stop_loss': stop_loss,
                                        'target': target,
                                        'shares': shares,
                                        'entry_atr': entry_atr,
                                        'entry_rsi': df.iloc[i]['rsi'],
                                        'risk_reward': risk_reward
                                    })

                                    print(f"\n[BREAKOUT SHORT ENTRY] {current_time.strftime('%H:%M:%S')}")
                                    print(f"  Price: ₹{entry_price:.2f} | Stop: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                                    print(f"  Shares: {shares} | R:R = 1:{risk_reward:.2f}")

            # Entry signals
            elif position == 0 and not is_square_off:
                # Check trade frequency limit
                if trades_today[current_day] >= self.max_trades_per_day:
                    continue

                # BREAKOUT LONG (pending confirmation)
                if df.iloc[i]['long_signal']:
                    breakout_pending = True
                    breakout_direction = 1
                    breakout_start_time = current_time
                    breakout_candles_count = 0

                # BREAKOUT SHORT (pending confirmation)
                elif df.iloc[i]['short_signal']:
                    breakout_pending = True
                    breakout_direction = -1
                    breakout_start_time = current_time
                    breakout_candles_count = 0

                # FADE LONG (immediate entry - mean reversion)
                elif df.iloc[i]['fade_long_signal']:
                    stop_loss_price = current_price - (current_atr * self.stop_loss_atr_mult)
                    target_price = df.iloc[i]['box_midpoint']  # Target is box midpoint

                    risk = current_price - stop_loss_price
                    reward = target_price - current_price
                    risk_reward = reward / risk if risk > 0 else 0

                    if risk_reward >= self.min_risk_reward:
                        position = 1
                        entry_price = current_price
                        entry_time = current_time
                        entry_atr = current_atr
                        entry_type = "fade"
                        stop_loss = stop_loss_price
                        target = target_price

                        shares = int(cash / entry_price)

                        if shares > 0:
                            trades.append({
                                'trade_num': trade_number + 1,
                                'direction': 'LONG',
                                'entry_type': entry_type,
                                'entry_time': entry_time,
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'target': target,
                                'shares': shares,
                                'entry_atr': entry_atr,
                                'entry_rsi': df.iloc[i]['rsi'],
                                'risk_reward': risk_reward
                            })

                            print(f"\n[FADE LONG ENTRY] {current_time.strftime('%H:%M:%S')}")
                            print(f"  Price: ₹{entry_price:.2f} | Stop: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                            print(f"  Shares: {shares} | R:R = 1:{risk_reward:.2f} | RSI: {df.iloc[i]['rsi']:.1f}")

                # FADE SHORT (immediate entry - mean reversion)
                elif df.iloc[i]['fade_short_signal']:
                    stop_loss_price = current_price + (current_atr * self.stop_loss_atr_mult)
                    target_price = df.iloc[i]['box_midpoint']  # Target is box midpoint

                    risk = stop_loss_price - current_price
                    reward = current_price - target_price
                    risk_reward = reward / risk if risk > 0 else 0

                    if risk_reward >= self.min_risk_reward:
                        position = -1
                        entry_price = current_price
                        entry_time = current_time
                        entry_atr = current_atr
                        entry_type = "fade"
                        stop_loss = stop_loss_price
                        target = target_price

                        shares = int(cash / entry_price)

                        if shares > 0:
                            trades.append({
                                'trade_num': trade_number + 1,
                                'direction': 'SHORT',
                                'entry_type': entry_type,
                                'entry_time': entry_time,
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'target': target,
                                'shares': shares,
                                'entry_atr': entry_atr,
                                'entry_rsi': df.iloc[i]['rsi'],
                                'risk_reward': risk_reward
                            })

                            print(f"\n[FADE SHORT ENTRY] {current_time.strftime('%H:%M:%S')}")
                            print(f"  Price: ₹{entry_price:.2f} | Stop: ₹{stop_loss:.2f} | Target: ₹{target:.2f}")
                            print(f"  Shares: {shares} | R:R = 1:{risk_reward:.2f} | RSI: {df.iloc[i]['rsi']:.1f}")

            # Exit management
            elif position != 0 and not is_square_off:
                exit_signal = False
                exit_reason = ""
                exit_price = current_price

                if position == 1:  # Long position
                    if current_low <= stop_loss:
                        exit_signal = True
                        exit_reason = "STOP_LOSS"
                        exit_price = stop_loss
                    elif current_high >= target:
                        exit_signal = True
                        exit_reason = "TARGET_HIT"
                        exit_price = target

                elif position == -1:  # Short position
                    if current_high >= stop_loss:
                        exit_signal = True
                        exit_reason = "STOP_LOSS"
                        exit_price = stop_loss
                    elif current_low <= target:
                        exit_signal = True
                        exit_reason = "TARGET_HIT"
                        exit_price = target

                if exit_signal:
                    shares = trades[-1]['shares']

                    if position == 1:
                        trade_pnl = shares * (exit_price - entry_price)
                    else:
                        trade_pnl = shares * (entry_price - exit_price)

                    trade_return = (trade_pnl / (shares * entry_price)) * 100

                    trade_number += 1
                    duration = (current_time - entry_time).total_seconds() / 60

                    print(f"\n[Trade #{trade_number}] {exit_reason}")
                    print(f"  Type: {entry_type.upper()} | Direction: {'LONG' if position == 1 else 'SHORT'}")
                    print(f"  Entry: {entry_time.strftime('%H:%M:%S')} @ ₹{entry_price:.2f}")
                    print(f"  Exit: {current_time.strftime('%H:%M:%S')} @ ₹{exit_price:.2f}")
                    print(f"  Duration: {duration:.1f} min | P&L: ₹{trade_pnl:.2f} ({trade_return:+.2f}%)")
                    print(f"  {'✅ PROFIT' if trade_pnl > 0 else '❌ LOSS'}")

                    trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'pnl': trade_pnl,
                        'return_pct': trade_return,
                        'exit_reason': exit_reason,
                        'duration_minutes': duration
                    })

                    position = 0
                    trades_today[current_day] += 1

        # Calculate metrics
        completed_trades = [t for t in trades if 'exit_time' in t]
        metrics = self.calculate_metrics(completed_trades)

        return {
            'symbol': symbol,
            'data': df,
            'trades': completed_trades,
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
                'breakout_trades': 0,
                'fade_trades': 0,
                'breakout_win_rate': 0,
                'fade_win_rate': 0
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

        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] <= 0]

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Entry type analysis
        breakout_trades = [t for t in trades if t.get('entry_type') == 'breakout']
        fade_trades = [t for t in trades if t.get('entry_type') == 'fade']

        breakout_wins = sum(1 for t in breakout_trades if t['pnl'] > 0)
        fade_wins = sum(1 for t in fade_trades if t['pnl'] > 0)

        breakout_win_rate = (breakout_wins / len(breakout_trades) * 100) if breakout_trades else 0
        fade_win_rate = (fade_wins / len(fade_trades) * 100) if fade_trades else 0

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
            'breakout_trades': len(breakout_trades),
            'fade_trades': len(fade_trades),
            'breakout_win_rate': breakout_win_rate,
            'fade_win_rate': fade_win_rate
        }

    def run_backtest(self):
        """Run backtest for all symbols"""
        print(f"\n{'='*100}")
        print("STARTING HORIZONTAL FADE BREAKOUT BACKTEST")
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
        """Print summary of all results"""
        if not self.results:
            print("\nNo results to display.")
            return

        print(f"\n{'='*100}")
        print("HORIZONTAL FADE BREAKOUT STRATEGY RESULTS - ALL SYMBOLS")
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
                'Breakout': f"{metrics['breakout_trades']} ({metrics['breakout_win_rate']:.0f}%)",
                'Fade': f"{metrics['fade_trades']} ({metrics['fade_win_rate']:.0f}%)",
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

        total_breakout = sum(r['metrics']['breakout_trades'] for r in self.results.values())
        total_fade = sum(r['metrics']['fade_trades'] for r in self.results.values())

        print(f"Symbols Tested: {len(self.results)}")
        print(f"Total Trades: {total_trades}")
        print(f"Total P&L: ₹{total_pnl:.2f}")
        print(f"Overall Win Rate: {(winning_trades / total_trades * 100):.1f}%" if total_trades > 0 else "N/A")
        print(f"Profitable Symbols: {profitable_symbols}/{len(self.results)}")
        print(f"Breakout Trades: {total_breakout} | Fade Trades: {total_fade}")

        summary_df.to_csv('horizontal_fade_breakout_results.csv', index=False)
        print(f"\n✅ Results saved to: horizontal_fade_breakout_results.csv")


if __name__ == "__main__":
    print("\n" + "="*100)
    print("HORIZONTAL FADE / BOX THEORY BREAKOUT STRATEGY - INTRADAY BACKTEST")
    print("="*100)

    backtester = HorizontalFadeBreakoutBacktester(
        data_folder="data/symbolupdate",
        symbols=None,  # Auto-detect

        # Box detection parameters
        box_period=20,
        box_threshold_pct=1.5,  # 1.5% range for box
        min_box_candles=6,  # Minimum 6 candles in consolidation

        # Breakout parameters
        breakout_confirmation_pct=0.3,  # 0.3% move beyond box
        volume_threshold_mult=1.5,  # 1.5x average volume
        false_breakout_threshold=3,  # Wait 3 candles for confirmation

        # Fade parameters
        fade_rsi_period=14,
        fade_rsi_threshold=65,  # RSI > 65 for fade short, < 35 for fade long

        # Risk management
        atr_period=14,
        stop_loss_atr_mult=2.0,
        target_atr_mult=3.0,
        min_risk_reward=1.5,

        # Trading rules
        initial_capital=100000,
        square_off_time="15:20",
        tick_interval='1min',
        avoid_opening_mins=15,
        last_entry_time="14:45",
        max_trades_per_day=5
    )

    backtester.run_backtest()

    print("\n" + "="*100)
    print("BACKTEST COMPLETE")
    print("="*100)
