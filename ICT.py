import sqlite3
import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, time
import pytz
import warnings

warnings.filterwarnings('ignore')


class ICTStrategyBacktester:
    def __init__(self, data_folder="data", symbols=None,
                 initial_capital=100000, square_off_time="15:20",
                 min_data_points=100, candle_timeframe="5T",
                 fvg_threshold=0.001, order_block_lookback=20,
                 liquidity_sweep_tolerance=0.002, risk_reward_ratio=2.0):
        """
        ICT (Inner Circle Trader) Strategy Backtester

        Key ICT Concepts Implemented:
        1. Fair Value Gaps (FVG/Imbalance)
        2. Order Blocks (OB)
        3. Break of Structure (BOS)
        4. Change of Character (ChoCh)
        5. Liquidity Sweeps
        6. Market Structure Shifts

        Parameters:
        - data_folder: Folder containing database files
        - symbols: List of symbols (if None, auto-detect)
        - initial_capital: Starting capital per symbol
        - square_off_time: Daily square-off time
        - candle_timeframe: "1T", "5T", "15T" recommended for ICT
        - fvg_threshold: Minimum gap size for FVG (as % of price)
        - order_block_lookback: Candles to look back for order blocks
        - liquidity_sweep_tolerance: % tolerance for liquidity sweeps
        - risk_reward_ratio: Target profit vs stop loss ratio
        """
        self.data_folder = data_folder
        self.db_files = self.find_database_files()
        self.initial_capital = initial_capital
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.min_data_points = min_data_points
        self.candle_timeframe = candle_timeframe
        self.fvg_threshold = fvg_threshold
        self.order_block_lookback = order_block_lookback
        self.liquidity_sweep_tolerance = liquidity_sweep_tolerance
        self.risk_reward_ratio = risk_reward_ratio
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        # Timeframe descriptions
        timeframe_names = {
            "1T": "1 minute",
            "5T": "5 minutes",
            "15T": "15 minutes",
            "30T": "30 minutes",
            "1H": "1 hour"
        }
        self.timeframe_name = timeframe_names.get(candle_timeframe, candle_timeframe)

        print("=" * 80)
        print("ICT (INNER CIRCLE TRADER) STRATEGY BACKTESTER")
        print("=" * 80)
        print(f"Candle Timeframe: {self.timeframe_name}")
        print(f"Fair Value Gap Threshold: {fvg_threshold * 100}%")
        print(f"Order Block Lookback: {order_block_lookback} candles")
        print(f"Liquidity Sweep Tolerance: {liquidity_sweep_tolerance * 100}%")
        print(f"Risk:Reward Ratio: 1:{risk_reward_ratio}")
        print(f"Square-off Time: {square_off_time} IST")
        print(f"Initial Capital: ₹{initial_capital:,}")
        print("=" * 80)

        # Auto-detect symbols
        if symbols is None:
            self.symbols = self.auto_detect_symbols()
        else:
            self.symbols = symbols

        print(f"\nSymbols to backtest: {len(self.symbols)}")

    def parse_square_off_time(self, time_str):
        """Parse square-off time"""
        try:
            hour, minute = map(int, time_str.split(':'))
            return time(hour, minute)
        except:
            return time(15, 20)

    def find_database_files(self):
        """Find database files"""
        if not os.path.exists(self.data_folder):
            return []
        db_files = glob.glob(os.path.join(self.data_folder, "*.db"))
        return sorted(db_files)

    def auto_detect_symbols(self):
        """Auto-detect symbols from databases"""
        all_symbols = set()
        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                query = """
                SELECT symbol, COUNT(*) as count 
                FROM market_data 
                GROUP BY symbol 
                HAVING count >= ?
                ORDER BY count DESC
                LIMIT 10
                """
                symbols_df = pd.read_sql_query(query, conn, params=(self.min_data_points,))
                conn.close()
                for symbol in symbols_df['symbol']:
                    all_symbols.add(symbol)
            except:
                continue
        return list(all_symbols)

    def load_data(self, symbol):
        """Load tick data for a symbol"""
        combined_df = pd.DataFrame()

        for db_file in self.db_files:
            try:
                conn = sqlite3.connect(db_file)
                query = """
                SELECT timestamp, ltp, close_price, high_price, low_price, volume
                FROM market_data 
                WHERE symbol = ? 
                ORDER BY timestamp
                """
                df = pd.read_sql_query(query, conn, params=(symbol,))
                conn.close()

                if not df.empty:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
            except:
                continue

        if combined_df.empty:
            return None

        # Process tick data
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.set_index('timestamp')
        combined_df['close'] = combined_df['close_price'].fillna(combined_df['ltp'])
        combined_df['high'] = combined_df['high_price'].fillna(combined_df['ltp'])
        combined_df['low'] = combined_df['low_price'].fillna(combined_df['ltp'])
        combined_df = combined_df.dropna(subset=['close'])
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        return combined_df

    def convert_to_candles(self, tick_data):
        """Convert streaming tick data to OHLC candles"""
        print(f"  Converting {len(tick_data)} ticks to {self.timeframe_name} candles...")

        ohlc_dict = {
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume': 'sum'
        }

        candles = tick_data.resample(self.candle_timeframe).agg(ohlc_dict)
        candles['open'] = tick_data['close'].resample(self.candle_timeframe).first()
        candles = candles.dropna()
        candles = candles[['open', 'high', 'low', 'close', 'volume']]

        print(f"  Created {len(candles)} {self.timeframe_name} candles")
        return candles

    def identify_market_structure(self, df):
        """
        Identify Market Structure: Swing Highs and Swing Lows
        A swing high is a high that is higher than n candles before and after
        A swing low is a low that is lower than n candles before and after
        """
        lookback = 5

        df['swing_high'] = False
        df['swing_low'] = False

        for i in range(lookback, len(df) - lookback):
            # Swing High
            is_swing_high = all(df.iloc[i]['high'] > df.iloc[j]['high']
                                for j in range(i - lookback, i)) and \
                            all(df.iloc[i]['high'] > df.iloc[j]['high']
                                for j in range(i + 1, i + lookback + 1))
            df.iloc[i, df.columns.get_loc('swing_high')] = is_swing_high

            # Swing Low
            is_swing_low = all(df.iloc[i]['low'] < df.iloc[j]['low']
                               for j in range(i - lookback, i)) and \
                           all(df.iloc[i]['low'] < df.iloc[j]['low']
                               for j in range(i + 1, i + lookback + 1))
            df.iloc[i, df.columns.get_loc('swing_low')] = is_swing_low

        return df

    def identify_fair_value_gaps(self, df):
        """
        Identify Fair Value Gaps (FVG/Imbalance)
        Bullish FVG: Gap between candle[i-2].high and candle[i].low (candle[i-1] doesn't fill it)
        Bearish FVG: Gap between candle[i-2].low and candle[i].high (candle[i-1] doesn't fill it)
        """
        df['bullish_fvg'] = False
        df['bearish_fvg'] = False
        df['fvg_high'] = np.nan
        df['fvg_low'] = np.nan

        for i in range(2, len(df)):
            # Bullish FVG
            gap_low = df.iloc[i - 2]['high']
            gap_high = df.iloc[i]['low']

            if gap_high > gap_low:
                gap_size = (gap_high - gap_low) / gap_low
                if gap_size >= self.fvg_threshold:
                    # Check if middle candle doesn't fill the gap
                    if df.iloc[i - 1]['low'] > gap_low and df.iloc[i - 1]['high'] < gap_high:
                        df.iloc[i, df.columns.get_loc('bullish_fvg')] = True
                        df.iloc[i, df.columns.get_loc('fvg_low')] = gap_low
                        df.iloc[i, df.columns.get_loc('fvg_high')] = gap_high

            # Bearish FVG
            gap_high = df.iloc[i - 2]['low']
            gap_low = df.iloc[i]['high']

            if gap_high > gap_low:
                gap_size = (gap_high - gap_low) / gap_low
                if gap_size >= self.fvg_threshold:
                    # Check if middle candle doesn't fill the gap
                    if df.iloc[i - 1]['high'] < gap_high and df.iloc[i - 1]['low'] > gap_low:
                        df.iloc[i, df.columns.get_loc('bearish_fvg')] = True
                        df.iloc[i, df.columns.get_loc('fvg_low')] = gap_low
                        df.iloc[i, df.columns.get_loc('fvg_high')] = gap_high

        return df

    def identify_order_blocks(self, df):
        """
        Identify Order Blocks (OB)
        Bullish OB: Last down candle before strong up move
        Bearish OB: Last up candle before strong down move
        """
        df['bullish_ob'] = False
        df['bearish_ob'] = False
        df['ob_high'] = np.nan
        df['ob_low'] = np.nan

        for i in range(1, len(df)):
            # Bullish Order Block (last red candle before green momentum)
            if (df.iloc[i]['close'] < df.iloc[i]['open'] and  # Current is red
                    i + 1 < len(df) and
                    df.iloc[i + 1]['close'] > df.iloc[i + 1]['open'] and  # Next is green
                    df.iloc[i + 1]['close'] > df.iloc[i]['high']):  # Next closes above current high

                df.iloc[i, df.columns.get_loc('bullish_ob')] = True
                df.iloc[i, df.columns.get_loc('ob_high')] = df.iloc[i]['high']
                df.iloc[i, df.columns.get_loc('ob_low')] = df.iloc[i]['low']

            # Bearish Order Block (last green candle before red momentum)
            if (df.iloc[i]['close'] > df.iloc[i]['open'] and  # Current is green
                    i + 1 < len(df) and
                    df.iloc[i + 1]['close'] < df.iloc[i + 1]['open'] and  # Next is red
                    df.iloc[i + 1]['close'] < df.iloc[i]['low']):  # Next closes below current low

                df.iloc[i, df.columns.get_loc('bearish_ob')] = True
                df.iloc[i, df.columns.get_loc('ob_high')] = df.iloc[i]['high']
                df.iloc[i, df.columns.get_loc('ob_low')] = df.iloc[i]['low']

        return df

    def identify_bos_choch(self, df):
        """
        Identify Break of Structure (BOS) and Change of Character (ChoCh)
        BOS: Breaking previous swing high/low in direction of trend
        ChoCh: Breaking previous swing high/low against trend (trend reversal signal)
        """
        df['bos_bullish'] = False
        df['bos_bearish'] = False
        df['choch_bullish'] = False
        df['choch_bearish'] = False

        # Track recent swing points
        recent_highs = []
        recent_lows = []
        trend = 'neutral'  # 'bullish', 'bearish', 'neutral'

        for i in range(len(df)):
            if df.iloc[i]['swing_high']:
                recent_highs.append(df.iloc[i]['high'])
                if len(recent_highs) > 3:
                    recent_highs.pop(0)

            if df.iloc[i]['swing_low']:
                recent_lows.append(df.iloc[i]['low'])
                if len(recent_lows) > 3:
                    recent_lows.pop(0)

            # Check for BOS and ChoCh
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                prev_high = max(recent_highs[:-1]) if len(recent_highs) > 1 else recent_highs[0]
                prev_low = min(recent_lows[:-1]) if len(recent_lows) > 1 else recent_lows[0]

                # Bullish BOS: Break above previous high in uptrend
                if df.iloc[i]['close'] > prev_high:
                    if trend == 'bullish' or trend == 'neutral':
                        df.iloc[i, df.columns.get_loc('bos_bullish')] = True
                        trend = 'bullish'
                    else:  # Was bearish, now breaking up = ChoCh
                        df.iloc[i, df.columns.get_loc('choch_bullish')] = True
                        trend = 'bullish'

                # Bearish BOS: Break below previous low in downtrend
                if df.iloc[i]['close'] < prev_low:
                    if trend == 'bearish' or trend == 'neutral':
                        df.iloc[i, df.columns.get_loc('bos_bearish')] = True
                        trend = 'bearish'
                    else:  # Was bullish, now breaking down = ChoCh
                        df.iloc[i, df.columns.get_loc('choch_bearish')] = True
                        trend = 'bearish'

        return df

    def identify_liquidity_sweeps(self, df):
        """
        Identify Liquidity Sweeps
        When price briefly breaks a recent high/low to grab liquidity, then reverses
        """
        df['liquidity_sweep_high'] = False
        df['liquidity_sweep_low'] = False

        lookback = 10

        for i in range(lookback, len(df)):
            recent_high = df.iloc[i - lookback:i]['high'].max()
            recent_low = df.iloc[i - lookback:i]['low'].min()

            # Sweep above high then reject (bearish)
            if (df.iloc[i]['high'] > recent_high * (1 + self.liquidity_sweep_tolerance) and
                    df.iloc[i]['close'] < df.iloc[i]['open'] and
                    df.iloc[i]['close'] < recent_high):
                df.iloc[i, df.columns.get_loc('liquidity_sweep_high')] = True

            # Sweep below low then reject (bullish)
            if (df.iloc[i]['low'] < recent_low * (1 - self.liquidity_sweep_tolerance) and
                    df.iloc[i]['close'] > df.iloc[i]['open'] and
                    df.iloc[i]['close'] > recent_low):
                df.iloc[i, df.columns.get_loc('liquidity_sweep_low')] = True

        return df

    def generate_ict_signals(self, df):
        """
        Generate ICT-based trading signals

        BUY Signals:
        1. Bullish ChoCh + Price in Bullish FVG or Bullish OB
        2. Liquidity Sweep Low + Bullish Order Block
        3. BOS Bullish + Price retraces to FVG

        SELL Signals:
        1. Bearish ChoCh + Price in Bearish FVG or Bearish OB
        2. Liquidity Sweep High + Bearish Order Block
        3. BOS Bearish + Price retraces to FVG
        """
        df['buy_signal'] = False
        df['sell_signal'] = False
        df['signal_reason'] = ''

        for i in range(self.order_block_lookback, len(df)):
            current_price = df.iloc[i]['close']

            # Look for recent bullish setups
            recent_df = df.iloc[max(0, i - self.order_block_lookback):i]

            # BUY Signal 1: Bullish ChoCh + in FVG/OB zone
            if df.iloc[i - 1]['choch_bullish']:
                # Check if price is in recent bullish FVG
                bullish_fvgs = recent_df[recent_df['bullish_fvg']]
                for idx in bullish_fvgs.index:
                    fvg_low = df.loc[idx, 'fvg_low']
                    fvg_high = df.loc[idx, 'fvg_high']
                    if fvg_low <= current_price <= fvg_high:
                        df.iloc[i, df.columns.get_loc('buy_signal')] = True
                        df.iloc[i, df.columns.get_loc('signal_reason')] = 'ChoCh_Bullish+FVG'
                        break

                # Check if price is in recent bullish OB
                bullish_obs = recent_df[recent_df['bullish_ob']]
                for idx in bullish_obs.index:
                    ob_low = df.loc[idx, 'ob_low']
                    ob_high = df.loc[idx, 'ob_high']
                    if ob_low <= current_price <= ob_high:
                        df.iloc[i, df.columns.get_loc('buy_signal')] = True
                        df.iloc[i, df.columns.get_loc('signal_reason')] = 'ChoCh_Bullish+OB'
                        break

            # BUY Signal 2: Liquidity Sweep Low + recovery
            if df.iloc[i]['liquidity_sweep_low']:
                df.iloc[i, df.columns.get_loc('buy_signal')] = True
                df.iloc[i, df.columns.get_loc('signal_reason')] = 'Liquidity_Sweep_Low'

            # BUY Signal 3: BOS Bullish + retracement to OB
            if df.iloc[i - 1]['bos_bullish']:
                bullish_obs = recent_df[recent_df['bullish_ob']]
                for idx in bullish_obs.index:
                    ob_low = df.loc[idx, 'ob_low']
                    ob_high = df.loc[idx, 'ob_high']
                    if ob_low <= current_price <= ob_high:
                        df.iloc[i, df.columns.get_loc('buy_signal')] = True
                        df.iloc[i, df.columns.get_loc('signal_reason')] = 'BOS_Bullish+OB_Retest'
                        break

            # SELL Signal 1: Bearish ChoCh + in FVG/OB zone
            if df.iloc[i - 1]['choch_bearish']:
                # Check if price is in recent bearish FVG
                bearish_fvgs = recent_df[recent_df['bearish_fvg']]
                for idx in bearish_fvgs.index:
                    fvg_low = df.loc[idx, 'fvg_low']
                    fvg_high = df.loc[idx, 'fvg_high']
                    if fvg_low <= current_price <= fvg_high:
                        df.iloc[i, df.columns.get_loc('sell_signal')] = True
                        df.iloc[i, df.columns.get_loc('signal_reason')] = 'ChoCh_Bearish+FVG'
                        break

                # Check if price is in recent bearish OB
                bearish_obs = recent_df[recent_df['bearish_ob']]
                for idx in bearish_obs.index:
                    ob_low = df.loc[idx, 'ob_low']
                    ob_high = df.loc[idx, 'ob_high']
                    if ob_low <= current_price <= ob_high:
                        df.iloc[i, df.columns.get_loc('sell_signal')] = True
                        df.iloc[i, df.columns.get_loc('signal_reason')] = 'ChoCh_Bearish+OB'
                        break

            # SELL Signal 2: Liquidity Sweep High + rejection
            if df.iloc[i]['liquidity_sweep_high']:
                df.iloc[i, df.columns.get_loc('sell_signal')] = True
                df.iloc[i, df.columns.get_loc('signal_reason')] = 'Liquidity_Sweep_High'

            # SELL Signal 3: BOS Bearish + retracement to OB
            if df.iloc[i - 1]['bos_bearish']:
                bearish_obs = recent_df[recent_df['bearish_ob']]
                for idx in bearish_obs.index:
                    ob_low = df.loc[idx, 'ob_low']
                    ob_high = df.loc[idx, 'ob_high']
                    if ob_low <= current_price <= ob_high:
                        df.iloc[i, df.columns.get_loc('sell_signal')] = True
                        df.iloc[i, df.columns.get_loc('signal_reason')] = 'BOS_Bearish+OB_Retest'
                        break

        return df

    def is_square_off_time(self, timestamp):
        """Check if square-off time"""
        try:
            return timestamp.time() >= self.square_off_time
        except:
            return False

    def backtest_symbol(self, symbol):
        """Backtest ICT strategy for a symbol"""
        print(f"\n{'=' * 80}")
        print(f"BACKTESTING: {symbol}")
        print(f"{'=' * 80}")

        # Load and convert data
        tick_data = self.load_data(symbol)
        if tick_data is None or len(tick_data) < 200:
            print(f"❌ Insufficient tick data for {symbol}")
            return None

        print(f"  Loaded {len(tick_data)} tick records")

        candles = self.convert_to_candles(tick_data)
        if len(candles) < 100:
            print(f"❌ Insufficient candles for {symbol}")
            return None

        # Apply ICT analysis
        print(f"  Applying ICT analysis...")
        candles = self.identify_market_structure(candles)
        candles = self.identify_fair_value_gaps(candles)
        candles = self.identify_order_blocks(candles)
        candles = self.identify_bos_choch(candles)
        candles = self.identify_liquidity_sweeps(candles)
        candles = self.generate_ict_signals(candles)

        candles['trading_day'] = candles.index.date
        candles['is_square_off'] = candles.index.map(self.is_square_off_time)

        # Backtest loop
        cash = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        trades = []
        entry_day = None

        for i in range(len(candles)):
            current_time = candles.index[i]
            current_price = candles.iloc[i]['close']
            current_day = candles.iloc[i]['trading_day']
            is_square_off = candles.iloc[i]['is_square_off']

            # Square-off existing positions
            if position != 0 and is_square_off:
                shares = trades[-1]['shares']
                proceeds = shares * current_price
                cash += proceeds

                if position == 1:
                    pnl = shares * (current_price - entry_price)
                else:  # Short position
                    pnl = shares * (entry_price - current_price)

                pnl_pct = ((current_price - entry_price) / entry_price) * 100 * position
                position = 0

                trades[-1].update({
                    'exit_time': current_time,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'SQUARE_OFF_3:20PM'
                })

                self.print_trade(trades[-1])
                entry_day = None

            # Entry signals
            elif position == 0 and not is_square_off:
                # Long entry
                if candles.iloc[i]['buy_signal']:
                    shares = int(cash / current_price)
                    if shares > 0:
                        position = 1
                        entry_price = current_price
                        entry_day = current_day

                        # Set stop loss and take profit
                        atr = candles.iloc[max(0, i - 14):i]['high'].max() - candles.iloc[max(0, i - 14):i]['low'].min()
                        stop_loss = entry_price - (atr * 0.5)
                        take_profit = entry_price + (atr * 0.5 * self.risk_reward_ratio)

                        cost = shares * current_price
                        cash -= cost

                        trades.append({
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'shares': shares,
                            'position_type': 'LONG',
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'signal_reason': candles.iloc[i]['signal_reason']
                        })

                        print(f"\n🟢 LONG ENTRY at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"   Price: ₹{entry_price:.2f} | Shares: {shares}")
                        print(f"   Signal: {candles.iloc[i]['signal_reason']}")
                        print(f"   SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f}")

                # Short entry
                elif candles.iloc[i]['sell_signal']:
                    shares = int(cash / current_price)
                    if shares > 0:
                        position = -1
                        entry_price = current_price
                        entry_day = current_day

                        # Set stop loss and take profit
                        atr = candles.iloc[max(0, i - 14):i]['high'].max() - candles.iloc[max(0, i - 14):i]['low'].min()
                        stop_loss = entry_price + (atr * 0.5)
                        take_profit = entry_price - (atr * 0.5 * self.risk_reward_ratio)

                        proceeds = shares * current_price
                        cash += proceeds

                        trades.append({
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'shares': shares,
                            'position_type': 'SHORT',
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'signal_reason': candles.iloc[i]['signal_reason']
                        })

                        print(f"\n🔴 SHORT ENTRY at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"   Price: ₹{entry_price:.2f} | Shares: {shares}")
                        print(f"   Signal: {candles.iloc[i]['signal_reason']}")
                        print(f"   SL: ₹{stop_loss:.2f} | TP: ₹{take_profit:.2f}")

            # Exit management
            elif position != 0 and entry_day == current_day and not is_square_off:
                exit_triggered = False
                exit_reason = ""

                # Long position exits
                if position == 1:
                    if current_price <= stop_loss:
                        exit_triggered = True
                        exit_reason = "STOP_LOSS"
                    elif current_price >= take_profit:
                        exit_triggered = True
                        exit_reason = "TAKE_PROFIT"
                    elif candles.iloc[i]['sell_signal']:
                        exit_triggered = True
                        exit_reason = "OPPOSITE_SIGNAL"

                # Short position exits
                elif position == -1:
                    if current_price >= stop_loss:
                        exit_triggered = True
                        exit_reason = "STOP_LOSS"
                    elif current_price <= take_profit:
                        exit_triggered = True
                        exit_reason = "TAKE_PROFIT"
                    elif candles.iloc[i]['buy_signal']:
                        exit_triggered = True
                        exit_reason = "OPPOSITE_SIGNAL"

                if exit_triggered:
                    shares = trades[-1]['shares']

                    if position == 1:
                        proceeds = shares * current_price
                        cash += proceeds
                        pnl = shares * (current_price - entry_price)
                    else:  # Short
                        cost = shares * current_price
                        cash -= cost
                        pnl = shares * (entry_price - current_price)

                    pnl_pct = ((current_price - entry_price) / entry_price) * 100 * position
                    position = 0

                    trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason
                    })

                    self.print_trade(trades[-1])
                    entry_day = None

        # Summary
        self.print_summary(symbol, trades, cash, candles)

        return {
            'symbol': symbol,
            'trades': [t for t in trades if 'exit_time' in t],
            'final_capital': cash,
            'candles': candles
        }

    def print_trade(self, trade):
        """Print individual trade details"""
        duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60

        print(f"\n📊 TRADE EXECUTED")
        print(f"   Type: {trade['position_type']}")
        print(f"   Signal: {trade['signal_reason']}")
        print(f"   Entry:  {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S')} @ ₹{trade['entry_price']:.2f}")
        print(f"   Exit:   {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S')} @ ₹{trade['exit_price']:.2f}")
        print(f"   SL: ₹{trade['stop_loss']:.2f} | TP: ₹{trade['take_profit']:.2f}")
        print(f"   Shares: {trade['shares']}")
        print(f"   Duration: {duration:.1f} minutes")
        print(f"   P&L: ₹{trade['pnl']:.2f} ({trade['pnl_pct']:+.2f}%)")
        print(f"   Exit Reason: {trade['exit_reason']}")

        if trade['pnl'] > 0:
            print(f"   Result: ✅ PROFIT")
        else:
            print(f"   Result: ❌ LOSS")

    def print_summary(self, symbol, trades, final_cash, candles):
        """Print strategy summary"""
        completed_trades = [t for t in trades if 'exit_time' in t]

        if not completed_trades:
            print(f"\n⚠️  No completed trades for {symbol}")
            return

        total_pnl = sum(t['pnl'] for t in completed_trades)
        total_return = ((final_cash - self.initial_capital) / self.initial_capital) * 100
        winning_trades = [t for t in completed_trades if t['pnl'] > 0]
        losing_trades = [t for t in completed_trades if t['pnl'] <= 0]
        win_rate = (len(winning_trades) / len(completed_trades)) * 100

        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0

        # ICT-specific metrics
        long_trades = [t for t in completed_trades if t['position_type'] == 'LONG']
        short_trades = [t for t in completed_trades if t['position_type'] == 'SHORT']

        # Count ICT patterns
        fvg_trades = [t for t in completed_trades if 'FVG' in t['signal_reason']]
        ob_trades = [t for t in completed_trades if 'OB' in t['signal_reason']]
        choch_trades = [t for t in completed_trades if 'ChoCh' in t['signal_reason']]
        bos_trades = [t for t in completed_trades if 'BOS' in t['signal_reason']]
        liquidity_trades = [t for t in completed_trades if 'Liquidity' in t['signal_reason']]

        print(f"\n{'=' * 80}")
        print(f"SUMMARY: {symbol}")
        print(f"{'=' * 80}")
        print(f"Strategy:           ICT (Inner Circle Trader)")
        print(f"Candle Timeframe:   {self.timeframe_name}")
        print(f"Total Candles:      {len(candles)}")
        print(f"")
        print(f"Initial Capital:    ₹{self.initial_capital:,.2f}")
        print(f"Final Capital:      ₹{final_cash:,.2f}")
        print(f"Total P&L:          ₹{total_pnl:,.2f}")
        print(f"Total Return:       {total_return:+.2f}%")
        print(f"")
        print(f"Total Trades:       {len(completed_trades)}")
        print(f"Long Trades:        {len(long_trades)} ({len(long_trades) / len(completed_trades) * 100:.1f}%)")
        print(f"Short Trades:       {len(short_trades)} ({len(short_trades) / len(completed_trades) * 100:.1f}%)")
        print(f"Winning Trades:     {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losing Trades:      {len(losing_trades)} ({100 - win_rate:.1f}%)")
        print(f"")
        print(f"Average Win:        ₹{avg_win:,.2f}")
        print(f"Average Loss:       ₹{avg_loss:,.2f}")
        if avg_loss != 0:
            print(f"Profit Factor:      {abs(avg_win / avg_loss):.2f}")
        print(f"Best Trade:         ₹{max(t['pnl'] for t in completed_trades):,.2f}")
        print(f"Worst Trade:        ₹{min(t['pnl'] for t in completed_trades):,.2f}")
        print(f"{'=' * 80}")

        # ICT Pattern Performance
        print(f"\nICT PATTERN BREAKDOWN:")
        print(f"Fair Value Gap (FVG):     {len(fvg_trades)} trades")
        print(f"Order Block (OB):         {len(ob_trades)} trades")
        print(f"Change of Character:      {len(choch_trades)} trades")
        print(f"Break of Structure:       {len(bos_trades)} trades")
        print(f"Liquidity Sweeps:         {len(liquidity_trades)} trades")

        # Exit reasons breakdown
        print(f"\nEXIT REASONS:")
        exit_reasons = {}
        for trade in completed_trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        for reason, count in exit_reasons.items():
            pct = (count / len(completed_trades)) * 100
            print(f"  {reason:25} {count:3d} trades ({pct:.1f}%)")

    def run_backtest(self):
        """Run backtest for all symbols"""
        print(f"\n🚀 Starting ICT Strategy Backtest")
        print(f"Candle Timeframe: {self.timeframe_name}")
        print(f"Key ICT Concepts: FVG, Order Blocks, BOS, ChoCh, Liquidity Sweeps\n")

        all_results = []

        for symbol in self.symbols:
            try:
                result = self.backtest_symbol(symbol)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"❌ Error with {symbol}: {e}")
                import traceback
                traceback.print_exc()

        # Overall summary
        if all_results:
            self.print_overall_summary(all_results)

        return all_results

    def print_overall_summary(self, results):
        """Print overall strategy performance"""
        print(f"\n{'=' * 80}")
        print(f"OVERALL ICT STRATEGY PERFORMANCE")
        print(f"{'=' * 80}")
        print(f"Strategy:              ICT (Inner Circle Trader)")
        print(f"Candle Timeframe:      {self.timeframe_name}")
        print(f"")

        total_trades = sum(len(r['trades']) for r in results)
        total_pnl = sum(r['final_capital'] - self.initial_capital for r in results)
        profitable_symbols = sum(1 for r in results if r['final_capital'] > self.initial_capital)

        # Aggregate all trades
        all_trades = []
        for r in results:
            all_trades.extend(r['trades'])

        winning_trades = [t for t in all_trades if t['pnl'] > 0]
        overall_win_rate = (len(winning_trades) / len(all_trades) * 100) if all_trades else 0

        print(f"Symbols Tested:        {len(results)}")
        print(f"Profitable Symbols:    {profitable_symbols} ({profitable_symbols / len(results) * 100:.1f}%)")
        print(f"Total Trades:          {total_trades}")
        print(f"Overall Win Rate:      {overall_win_rate:.1f}%")
        print(f"Total P&L:             ₹{total_pnl:,.2f}")
        print(f"Average Return/Symbol: {(total_pnl / (self.initial_capital * len(results))) * 100:+.2f}%")
        print(f"{'=' * 80}")

        # Best and worst performers
        if results:
            best = max(results, key=lambda x: x['final_capital'])
            worst = min(results, key=lambda x: x['final_capital'])

            print(f"\n🏆 Best Performer:  {best['symbol']}")
            print(f"   Final Capital: ₹{best['final_capital']:,.2f}")
            print(f"   Return: {((best['final_capital'] - self.initial_capital) / self.initial_capital) * 100:+.2f}%")
            print(f"   Trades: {len(best['trades'])}")

            print(f"\n📉 Worst Performer: {worst['symbol']}")
            print(f"   Final Capital: ₹{worst['final_capital']:,.2f}")
            print(f"   Return: {((worst['final_capital'] - self.initial_capital) / self.initial_capital) * 100:+.2f}%")
            print(f"   Trades: {len(worst['trades'])}")

        # ICT pattern statistics across all symbols
        all_fvg = sum(1 for t in all_trades if 'FVG' in t['signal_reason'])
        all_ob = sum(1 for t in all_trades if 'OB' in t['signal_reason'])
        all_choch = sum(1 for t in all_trades if 'ChoCh' in t['signal_reason'])
        all_bos = sum(1 for t in all_trades if 'BOS' in t['signal_reason'])
        all_liq = sum(1 for t in all_trades if 'Liquidity' in t['signal_reason'])

        print(f"\n📊 ICT PATTERN USAGE (ALL SYMBOLS):")
        print(f"Fair Value Gaps:       {all_fvg} trades ({all_fvg / len(all_trades) * 100:.1f}%)")
        print(f"Order Blocks:          {all_ob} trades ({all_ob / len(all_trades) * 100:.1f}%)")
        print(f"Change of Character:   {all_choch} trades ({all_choch / len(all_trades) * 100:.1f}%)")
        print(f"Break of Structure:    {all_bos} trades ({all_bos / len(all_trades) * 100:.1f}%)")
        print(f"Liquidity Sweeps:      {all_liq} trades ({all_liq / len(all_trades) * 100:.1f}%)")


# Main execution
if __name__ == "__main__":
    """
    ICT STRATEGY BACKTESTER

    This strategy implements key Inner Circle Trader (ICT) concepts:

    1. FAIR VALUE GAPS (FVG): Price imbalances that act as magnets
    2. ORDER BLOCKS (OB): Institutional order zones (last opposite candle before move)
    3. BREAK OF STRUCTURE (BOS): Continuation pattern in trend direction
    4. CHANGE OF CHARACTER (ChoCh): Reversal pattern against trend
    5. LIQUIDITY SWEEPS: Stop hunts above/below key levels

    RECOMMENDED TIMEFRAMES:
    - "5T"  = 5 minutes (scalping)
    - "15T" = 15 minutes (day trading)
    - "1H"  = 1 hour (swing trading)

    SIGNAL LOGIC:
    BUY when:
    - Bullish ChoCh + price in Bullish FVG/OB
    - Liquidity sweep below + rejection
    - BOS Bullish + retest of Order Block

    SELL when:
    - Bearish ChoCh + price in Bearish FVG/OB
    - Liquidity sweep above + rejection
    - BOS Bearish + retest of Order Block
    """

    # Initialize ICT backtester
    backtester = ICTStrategyBacktester(
        data_folder="data/symbolupdate",  # Update with your path
        symbols=None,  # Auto-detect symbols
        initial_capital=100000,
        square_off_time="15:20",
        candle_timeframe="5S",  # 5-minute candles (recommended for ICT)
        fvg_threshold=0.001,  # 0.1% minimum gap for FVG
        order_block_lookback=20,  # Look back 20 candles for OB
        liquidity_sweep_tolerance=0.002,  # 0.2% tolerance for sweeps
        risk_reward_ratio=2.0  # 1:2 risk-reward
    )

    # Run backtest
    results = backtester.run_backtest()

    print(f"\n✅ ICT Backtest Complete!")
    print(f"Tested {len(results)} symbols using ICT methodology")
    print(f"Timeframe: {backtester.timeframe_name}")

    # Tips for optimization
    print(f"\n💡 OPTIMIZATION TIPS:")
    print(f"1. Adjust 'fvg_threshold' to filter noise (0.0005-0.002)")
    print(f"2. Try different timeframes: 5T for scalping, 15T for day trading")
    print(f"3. Modify 'risk_reward_ratio' (1.5-3.0) based on your risk appetite")
    print(f"4. Combine with session times (London/NY open) for better results")
    print(f"5. Filter trades during high-impact news events")