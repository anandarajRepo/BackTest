import pandas as pd
import numpy as np
import os
from datetime import datetime, time, timedelta
import pytz
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

warnings.filterwarnings('ignore')


@dataclass
class ArbitrageTrade:
    """Data class to store arbitrage trade information"""
    entry_time: datetime
    exit_time: datetime
    symbol_spot: str
    symbol_futures: str
    position_type: str  # 'long_spot_short_futures' or 'short_spot_long_futures'
    entry_spot_price: float
    entry_futures_price: float
    exit_spot_price: float
    exit_futures_price: float
    entry_spread_pct: float
    exit_spread_pct: float
    entry_mispricing: float
    exit_mispricing: float
    entry_fair_value: float
    exit_fair_value: float
    entry_confidence: float
    entry_volume_ratio: float
    entry_z_score: float
    exit_z_score: float
    days_to_expiry_entry: int
    days_to_expiry_exit: int
    quantity: int
    spot_pnl: float
    futures_pnl: float
    total_pnl: float
    total_pnl_pct: float
    exit_reason: str
    holding_period_minutes: float
    target_spread_pct: float
    stop_loss_spread_pct: float


class LiveStrategyBacktester:
    """
    Backtest implementation of the LIVE FyersArbitrage strategy logic
    Implements exact logic from arbitrage_strategy.py
    Uses Fyers API for historical data instead of database files
    """

    def __init__(self, symbols, access_token=None,
                 # Data parameters
                 lookback_days=30,
                 timeframe="5min",
                 # Live strategy parameters
                 portfolio_value=100000,
                 risk_per_trade_pct=2.0,
                 max_positions=5,
                 min_spread_threshold=0.5,
                 max_spread_threshold=3.0,
                 target_profit_pct=0.3,
                 stop_loss_pct=0.5,
                 min_confidence=0.7,
                 min_liquidity_ratio=1.5,
                 require_convergence_trend=False,
                 square_off_time="15:15",
                 basis_lookback=50,
                 lot_size=1,
                 risk_free_rate=0.05):
        """
        Live Strategy Backtester using Fyers API

        Parameters:
        - symbols: List of dicts with 'spot', 'futures', 'base_name' keys
        - access_token: Fyers API access token (or set FYERS_ACCESS_TOKEN env var)
        - lookback_days: Number of days of historical data to fetch (default 30)
        - timeframe: Candle timeframe - "1", "5", "15", "30", "60", "D" (default "5min")

        Live Strategy Parameters (from ArbitrageStrategyConfig):
        - portfolio_value: Total capital (default: ₹1 lakh)
        - risk_per_trade_pct: Risk per trade (default: 2%)
        - max_positions: Maximum concurrent positions (default: 5)
        - min_spread_threshold: Minimum spread to enter (default: 0.5%)
        - max_spread_threshold: Maximum spread to consider (default: 3.0%)
        - target_profit_pct: Target profit on convergence (default: 0.3%)
        - stop_loss_pct: Stop loss if spread widens (default: 0.5%)
        - min_confidence: Minimum signal confidence (default: 0.7)
        - min_liquidity_ratio: Min volume ratio for entry (default: 1.5)
        - require_convergence_trend: Require converging spread (default: False)
        """

        # Fyers API setup
        if access_token is None:
            access_token = os.getenv('FYERS_ACCESS_TOKEN')
            if not access_token:
                raise ValueError("Access token required. Set FYERS_ACCESS_TOKEN env var or pass access_token parameter")

        self.fyers = fyersModel.FyersModel(client_id="", token=access_token, log_path="")

        self.symbol_pairs = symbols
        self.lookback_days = lookback_days
        self.timeframe = timeframe

        # Live strategy parameters
        self.portfolio_value = portfolio_value
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_positions = max_positions
        self.min_spread_threshold = min_spread_threshold
        self.max_spread_threshold = max_spread_threshold
        self.target_profit_pct = target_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.min_confidence = min_confidence
        self.min_liquidity_ratio = min_liquidity_ratio
        self.require_convergence_trend = require_convergence_trend
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.risk_free_rate = risk_free_rate

        # Supporting parameters
        self.basis_lookback = basis_lookback
        self.lot_size = lot_size
        self.results = {}

        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print("=" * 100)
        print("LIVE STRATEGY BACKTEST - EXACT LOGIC FROM FyersArbitrage (FYERS API)")
        print("=" * 100)
        print(f"Strategy: Fair Value Mispricing + Multi-Factor Confidence")
        print(f"Data Source: Fyers API")
        print(f"Lookback Period: {lookback_days} days")
        print(f"Timeframe: {timeframe}")
        print(f"")
        print(f"ENTRY LOGIC:")
        print(f"  - Fair Value Premium: (Risk-free rate × Days to expiry) / 365")
        print(f"  - Mispricing: Actual Spread - Fair Value Premium")
        print(f"  - Entry Threshold: ±{min_spread_threshold}% mispricing")
        print(f"  - Spread Range: {min_spread_threshold}% to {max_spread_threshold}%")
        print(f"  - Volume Ratio: >= {min_liquidity_ratio}x")
        print(f"  - Min Confidence: {min_confidence * 100}%")
        print(f"  - Convergence Required: {require_convergence_trend}")
        print(f"")
        print(f"EXIT LOGIC:")
        print(f"  - Target Profit: {target_profit_pct}% on spread")
        print(f"  - Stop Loss: {stop_loss_pct}% on spread")
        print(f"  - Near Expiry: <= 3 days")
        print(f"  - Square-off Time: {square_off_time} IST")
        print(f"")
        print(f"RISK MANAGEMENT:")
        print(f"  - Portfolio Value: ₹{portfolio_value:,}")
        print(f"  - Risk per Trade: {risk_per_trade_pct}%")
        print(f"  - Max Positions: {max_positions}")
        print(f"  - Position Sizing: Risk-based (not fixed %)")
        print("=" * 100)

        print(f"\nSpot-Futures Pairs to backtest: {len(self.symbol_pairs)}")
        for pair in self.symbol_pairs:
            print(f"  - {pair['spot']} <-> {pair['futures']}")

    def parse_square_off_time(self, time_str):
        """Parse square-off time"""
        try:
            hour, minute = map(int, time_str.split(':'))
            return time(hour, minute)
        except:
            return time(15, 15)

    def fetch_historical_data(self, symbol, days=None):
        """
        Fetch historical data from Fyers API

        Args:
            symbol: Fyers symbol (e.g., "NSE:RELIANCE-EQ", "NSE:RELIANCE25DECFUT")
            days: Number of days to fetch (overrides self.lookback_days)

        Returns:
            DataFrame with OHLCV data
        """
        if days is None:
            days = self.lookback_days

        print(f"    Fetching {days} days of {self.timeframe} data for {symbol}...")

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Convert to date strings
        range_from = start_date.strftime("%Y-%m-%d")
        range_to = end_date.strftime("%Y-%m-%d")

        # Map timeframe to Fyers format
        timeframe_map = {
            "1": "1",  # 1 minute
            "5": "5",  # 5 minutes
            "15": "15",  # 15 minutes
            "30": "30",  # 30 minutes
            "60": "60",  # 1 hour
            "D": "D",  # Daily
            "1min": "1",
            "5min": "5",
            "15min": "15",
            "30min": "30",
            "1H": "60",
            "1D": "D"
        }

        fyers_timeframe = timeframe_map.get(self.timeframe, "5")

        try:
            data = {
                "symbol": symbol,
                "resolution": fyers_timeframe,
                "date_format": "1",  # Unix timestamp
                "range_from": range_from,
                "range_to": range_to,
                "cont_flag": "1"
            }

            response = self.fyers.history(data=data)

            if response['s'] != 'ok':
                print(f"      Error fetching data: {response.get('message', 'Unknown error')}")
                return None

            # Convert to DataFrame
            candles = response.get('candles', [])
            if not candles:
                print(f"      No data returned for {symbol}")
                return None

            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(self.ist_tz)
            df = df.set_index('timestamp')

            print(f"      Fetched {len(df)} candles from {df.index[0]} to {df.index[-1]}")

            return df[['close', 'volume']]

        except Exception as e:
            print(f"      Error fetching data for {symbol}: {e}")
            return None

    def load_spot_futures_data(self, spot_symbol, futures_symbol):
        """Load data for both spot and futures from Fyers API"""
        print(f"\n  Loading data from Fyers API...")
        print(f"  Spot: {spot_symbol}")

        spot_data = self.fetch_historical_data(spot_symbol)

        if spot_data is None:
            print(f"  Failed to load spot data for {spot_symbol}")
            return None, None

        print(f"  Futures: {futures_symbol}")
        futures_data = self.fetch_historical_data(futures_symbol)

        if futures_data is None:
            print(f"  Failed to load futures data for {futures_symbol}")
            return None, None

        print(f"  Spot: {len(spot_data)} records | Futures: {len(futures_data)} records")

        return spot_data, futures_data

    def align_spot_futures_data(self, spot_data, futures_data):
        """Align spot and futures data on common timestamps"""
        aligned = pd.merge(
            spot_data, futures_data,
            left_index=True, right_index=True,
            suffixes=('_spot', '_futures'),
            how='inner'
        )

        print(f"  Aligned data: {len(aligned)} common timestamps")

        return aligned

    def calculate_days_to_expiry(self, timestamp):
        """
        Calculate days to expiry (simplified)
        In live strategy, this comes from FuturesContract
        """
        # Simplified: assume 25 days average to expiry
        # In reality, would calculate from actual contract expiry date
        day_of_month = timestamp.day

        # Assume expiry on last Thursday of month
        # Rough approximation for backtesting
        if day_of_month <= 7:
            return 21  # ~3 weeks
        elif day_of_month <= 14:
            return 14  # ~2 weeks
        elif day_of_month <= 21:
            return 7  # ~1 week
        else:
            return 3  # Near expiry

    def calculate_fair_value_premium(self, days_to_expiry):
        """
        Calculate fair value premium based on cost of carry
        From live strategy: (Risk-free rate × Days to expiry) / 365
        """
        premium_pct = (self.risk_free_rate * days_to_expiry) / 365 * 100
        return premium_pct

    def calculate_spread_statistics(self, df):
        """
        Calculate spread statistics including fair value and mispricing
        Implements live strategy logic from arbitrage_strategy.py
        """
        # Basic spread
        df['spread'] = df['close_futures'] - df['close_spot']
        df['spread_pct'] = (df['spread'] / df['close_spot']) * 100

        # Days to expiry
        df['days_to_expiry'] = df.index.map(self.calculate_days_to_expiry)

        # Fair value premium (cost of carry)
        df['fair_value_premium'] = df['days_to_expiry'].map(self.calculate_fair_value_premium)

        # Mispricing = actual spread - fair value
        df['mispricing'] = df['spread_pct'] - df['fair_value_premium']

        # Z-score for confidence calculation
        df['spread_mean'] = df['spread_pct'].rolling(window=self.basis_lookback).mean()
        df['spread_std'] = df['spread_pct'].rolling(window=self.basis_lookback).std()
        df['z_score'] = (df['spread_pct'] - df['spread_mean']) / df['spread_std']

        # Volume ratio
        df['volume_ratio'] = df['volume_futures'] / df['volume_spot'].replace(0, 1)

        # Convergence detection (simplified)
        df['spread_pct_change'] = df['spread_pct'].diff()
        df['is_converging'] = (df['spread_pct_change'].rolling(3).mean() < 0)

        return df

    def calculate_confidence_score(self, row):
        """
        Calculate multi-factor confidence score
        Exact implementation from live strategy (_calculate_signal_confidence)

        Components:
        1. Z-score strength (30%)
        2. Volume ratio (25%)
        3. Convergence trend (25%)
        4. Days to expiry (20%)
        """
        confidence = 0.0

        # 1. Z-score component (30%)
        z_score = abs(row['z_score']) if not pd.isna(row['z_score']) else 0
        z_score_normalized = min(z_score / 3.0, 1.0)
        confidence += z_score_normalized * 0.30

        # 2. Volume component (25%)
        volume_score = min(row['volume_ratio'] / 2.0, 1.0) if row['volume_ratio'] > 0 else 0.0
        confidence += volume_score * 0.25

        # 3. Convergence component (25%)
        convergence_score = 0.7 if row['is_converging'] else 0.3
        confidence += convergence_score * 0.25

        # 4. Days to expiry component (20%)
        expiry_score = min(row['days_to_expiry'] / 20, 1.0)
        confidence += expiry_score * 0.20

        return min(max(confidence, 0.0), 1.0)

    def generate_arbitrage_signals(self, df):
        """
        Generate arbitrage signals using LIVE STRATEGY logic
        From evaluate_arbitrage_opportunity() in arbitrage_strategy.py
        """
        # Calculate confidence for all rows
        df['confidence'] = df.apply(self.calculate_confidence_score, axis=1)

        # ENTRY SIGNAL: Long Spot + Short Futures
        # When futures are OVERPRICED (mispricing > threshold)
        df['long_spot_signal'] = (
                (df['mispricing'] > self.min_spread_threshold) &  # Futures overpriced
                (df['spread_pct'].abs() >= self.min_spread_threshold) &  # Min spread
                (df['spread_pct'].abs() <= self.max_spread_threshold) &  # Max spread
                (df['volume_ratio'] >= self.min_liquidity_ratio) &  # Liquidity
                (df['confidence'] >= self.min_confidence) &  # Confidence
                (~df['z_score'].isna()) &
                (df['days_to_expiry'] > 3)  # Not near expiry
        )

        # If convergence required, add filter
        if self.require_convergence_trend:
            df['long_spot_signal'] = df['long_spot_signal'] & df['is_converging']

        # ENTRY SIGNAL: Short Spot + Long Futures
        # When futures are UNDERPRICED (mispricing < -threshold)
        df['short_spot_signal'] = (
                (df['mispricing'] < -self.min_spread_threshold) &  # Futures underpriced
                (df['spread_pct'].abs() >= self.min_spread_threshold) &
                (df['spread_pct'].abs() <= self.max_spread_threshold) &
                (df['volume_ratio'] >= self.min_liquidity_ratio) &
                (df['confidence'] >= self.min_confidence) &
                (~df['z_score'].isna()) &
                (df['days_to_expiry'] > 3)
        )

        if self.require_convergence_trend:
            df['short_spot_signal'] = df['short_spot_signal'] & df['is_converging']

        return df

    def is_square_off_time(self, timestamp):
        """Check if square-off time"""
        try:
            return timestamp.time() >= self.square_off_time
        except:
            return False

    def calculate_position_size(self, entry_spot_price, entry_futures_price, stop_loss_pct):
        """
        Calculate position size based on risk
        From live strategy: risk-based position sizing
        """
        # Risk per trade in rupees
        risk_amount = self.portfolio_value * (self.risk_per_trade_pct / 100)

        # Stop loss in rupees per unit
        stop_loss_per_unit = entry_spot_price * (stop_loss_pct / 100)

        # Quantity based on risk
        if stop_loss_per_unit > 0:
            quantity = int(risk_amount / stop_loss_per_unit)
        else:
            # Fallback: use 50% of portfolio
            quantity = int((self.portfolio_value * 0.5) / entry_spot_price)

        # Ensure minimum quantity
        quantity = max(quantity, 1)

        return quantity

    def backtest_pair(self, pair):
        """Backtest LIVE strategy for a spot-futures pair"""
        spot_symbol = pair['spot']
        futures_symbol = pair['futures']
        base_name = pair['base_name']

        print(f"\n{'=' * 100}")
        print(f"BACKTESTING LIVE STRATEGY: {base_name}")
        print(f"Spot: {spot_symbol} | Futures: {futures_symbol}")
        print(f"{'=' * 100}")

        # Load data from Fyers API
        spot_data, futures_data = self.load_spot_futures_data(spot_symbol, futures_symbol)

        if spot_data is None or futures_data is None:
            print(f"✗ Insufficient data for {base_name}")
            return None

        # Align timestamps
        df = self.align_spot_futures_data(spot_data, futures_data)

        if len(df) < self.basis_lookback * 2:
            print(f"✗ Insufficient aligned data for {base_name}")
            return None

        # Calculate spread statistics and signals
        df = self.calculate_spread_statistics(df)
        df = self.generate_arbitrage_signals(df)

        # Add trading info
        df['trading_day'] = df.index.date
        df['is_square_off'] = df.index.map(self.is_square_off_time)

        # Initialize trading
        cash = self.portfolio_value
        active_positions = []  # List of active positions (max 5)
        trades: List[ArbitrageTrade] = []

        # Backtest loop
        for i in range(self.basis_lookback, len(df)):
            current_time = df.index[i]
            current_data = df.iloc[i]

            spot_price = current_data['close_spot']
            futures_price = current_data['close_futures']
            spread_pct = current_data['spread_pct']
            mispricing = current_data['mispricing']
            is_square_off = current_data['is_square_off']

            # Skip if invalid data
            if pd.isna(current_data['z_score']) or pd.isna(mispricing):
                continue

            # Force square-off at designated time
            if active_positions and is_square_off:
                for pos in active_positions[:]:  # Copy list to modify during iteration
                    quantity = pos['quantity']

                    if pos['position_type'] == 'long_spot_short_futures':
                        spot_pnl = quantity * (spot_price - pos['entry_spot_price'])
                        futures_pnl = quantity * (pos['entry_futures_price'] - futures_price) * self.lot_size
                    else:  # short_spot_long_futures
                        spot_pnl = quantity * (pos['entry_spot_price'] - spot_price)
                        futures_pnl = quantity * (futures_price - pos['entry_futures_price']) * self.lot_size

                    total_pnl = spot_pnl + futures_pnl
                    capital_invested = quantity * pos['entry_spot_price'] * 2
                    total_pnl_pct = (total_pnl / capital_invested) * 100 if capital_invested > 0 else 0

                    trade = ArbitrageTrade(
                        entry_time=pos['entry_time'],
                        exit_time=current_time,
                        symbol_spot=spot_symbol,
                        symbol_futures=futures_symbol,
                        position_type=pos['position_type'],
                        entry_spot_price=pos['entry_spot_price'],
                        entry_futures_price=pos['entry_futures_price'],
                        exit_spot_price=spot_price,
                        exit_futures_price=futures_price,
                        entry_spread_pct=pos['entry_spread_pct'],
                        exit_spread_pct=spread_pct,
                        entry_mispricing=pos['entry_mispricing'],
                        exit_mispricing=mispricing,
                        entry_fair_value=pos['entry_fair_value'],
                        exit_fair_value=current_data['fair_value_premium'],
                        entry_confidence=pos['entry_confidence'],
                        entry_volume_ratio=pos['entry_volume_ratio'],
                        entry_z_score=pos['entry_z_score'],
                        exit_z_score=current_data['z_score'],
                        days_to_expiry_entry=pos['days_to_expiry'],
                        days_to_expiry_exit=current_data['days_to_expiry'],
                        quantity=quantity,
                        spot_pnl=spot_pnl,
                        futures_pnl=futures_pnl,
                        total_pnl=total_pnl,
                        total_pnl_pct=total_pnl_pct,
                        exit_reason='SQUARE_OFF_TIME',
                        holding_period_minutes=(current_time - pos['entry_time']).total_seconds() / 60,
                        target_spread_pct=pos['target_spread_pct'],
                        stop_loss_spread_pct=pos['stop_loss_spread_pct']
                    )

                    trades.append(trade)
                    cash += total_pnl
                    active_positions.remove(pos)

                    self.print_trade(trade)

            # Check for new entry signals (if space available)
            elif len(active_positions) < self.max_positions and not is_square_off:

                # Long Spot + Short Futures (futures overpriced)
                if current_data['long_spot_signal']:
                    # Calculate targets and stops
                    target_spread = spread_pct - self.target_profit_pct
                    stop_loss_spread = spread_pct + self.stop_loss_pct

                    # Calculate position size
                    quantity = self.calculate_position_size(
                        spot_price, futures_price, self.stop_loss_pct
                    )

                    position = {
                        'position_type': 'long_spot_short_futures',
                        'entry_time': current_time,
                        'entry_spot_price': spot_price,
                        'entry_futures_price': futures_price,
                        'entry_spread_pct': spread_pct,
                        'entry_mispricing': mispricing,
                        'entry_fair_value': current_data['fair_value_premium'],
                        'entry_confidence': current_data['confidence'],
                        'entry_volume_ratio': current_data['volume_ratio'],
                        'entry_z_score': current_data['z_score'],
                        'days_to_expiry': current_data['days_to_expiry'],
                        'quantity': quantity,
                        'target_spread_pct': target_spread,
                        'stop_loss_spread_pct': stop_loss_spread
                    }

                    active_positions.append(position)

                    print(f"\n✓ LONG SPOT + SHORT FUTURES")
                    print(f"   Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   Spot: ₹{spot_price:.2f} | Futures: ₹{futures_price:.2f}")
                    print(f"   Spread: {spread_pct:.3f}% | Fair Value: {current_data['fair_value_premium']:.3f}%")
                    print(f"   Mispricing: {mispricing:+.3f}% (Futures OVERPRICED)")
                    print(f"   Confidence: {current_data['confidence']:.2f} | Volume Ratio: {current_data['volume_ratio']:.2f}")
                    print(f"   Target: {target_spread:.3f}% | Stop: {stop_loss_spread:.3f}%")
                    print(f"   Quantity: {quantity} | Days to Expiry: {current_data['days_to_expiry']}")

                # Short Spot + Long Futures (futures underpriced)
                elif current_data['short_spot_signal']:
                    target_spread = spread_pct + self.target_profit_pct
                    stop_loss_spread = spread_pct - self.stop_loss_pct

                    quantity = self.calculate_position_size(
                        spot_price, futures_price, self.stop_loss_pct
                    )

                    position = {
                        'position_type': 'short_spot_long_futures',
                        'entry_time': current_time,
                        'entry_spot_price': spot_price,
                        'entry_futures_price': futures_price,
                        'entry_spread_pct': spread_pct,
                        'entry_mispricing': mispricing,
                        'entry_fair_value': current_data['fair_value_premium'],
                        'entry_confidence': current_data['confidence'],
                        'entry_volume_ratio': current_data['volume_ratio'],
                        'entry_z_score': current_data['z_score'],
                        'days_to_expiry': current_data['days_to_expiry'],
                        'quantity': quantity,
                        'target_spread_pct': target_spread,
                        'stop_loss_spread_pct': stop_loss_spread
                    }

                    active_positions.append(position)

                    print(f"\n✓ SHORT SPOT + LONG FUTURES")
                    print(f"   Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   Spot: ₹{spot_price:.2f} | Futures: ₹{futures_price:.2f}")
                    print(f"   Spread: {spread_pct:.3f}% | Fair Value: {current_data['fair_value_premium']:.3f}%")
                    print(f"   Mispricing: {mispricing:+.3f}% (Futures UNDERPRICED)")
                    print(f"   Confidence: {current_data['confidence']:.2f} | Volume Ratio: {current_data['volume_ratio']:.2f}")
                    print(f"   Target: {target_spread:.3f}% | Stop: {stop_loss_spread:.3f}%")
                    print(f"   Quantity: {quantity} | Days to Expiry: {current_data['days_to_expiry']}")

            # Monitor active positions for exit
            elif active_positions and not is_square_off:
                for pos in active_positions[:]:
                    should_exit = False
                    exit_reason = ""

                    quantity = pos['quantity']

                    # Calculate current P&L
                    if pos['position_type'] == 'long_spot_short_futures':
                        spot_pnl = quantity * (spot_price - pos['entry_spot_price'])
                        futures_pnl = quantity * (pos['entry_futures_price'] - futures_price) * self.lot_size
                    else:
                        spot_pnl = quantity * (pos['entry_spot_price'] - spot_price)
                        futures_pnl = quantity * (futures_price - pos['entry_futures_price']) * self.lot_size

                    total_pnl = spot_pnl + futures_pnl

                    # Check TARGET
                    if pos['position_type'] == 'long_spot_short_futures':
                        if spread_pct <= pos['target_spread_pct']:
                            should_exit = True
                            exit_reason = 'TARGET_PROFIT'
                    else:
                        if spread_pct >= pos['target_spread_pct']:
                            should_exit = True
                            exit_reason = 'TARGET_PROFIT'

                    # Check STOP LOSS
                    if not should_exit:
                        if pos['position_type'] == 'long_spot_short_futures':
                            if spread_pct >= pos['stop_loss_spread_pct']:
                                should_exit = True
                                exit_reason = 'STOP_LOSS'
                        else:
                            if spread_pct <= pos['stop_loss_spread_pct']:
                                should_exit = True
                                exit_reason = 'STOP_LOSS'

                    # Check NEAR EXPIRY
                    if not should_exit and current_data['days_to_expiry'] <= 3:
                        should_exit = True
                        exit_reason = 'NEAR_EXPIRY'

                    if should_exit:
                        capital_invested = quantity * pos['entry_spot_price'] * 2
                        total_pnl_pct = (total_pnl / capital_invested) * 100 if capital_invested > 0 else 0

                        trade = ArbitrageTrade(
                            entry_time=pos['entry_time'],
                            exit_time=current_time,
                            symbol_spot=spot_symbol,
                            symbol_futures=futures_symbol,
                            position_type=pos['position_type'],
                            entry_spot_price=pos['entry_spot_price'],
                            entry_futures_price=pos['entry_futures_price'],
                            exit_spot_price=spot_price,
                            exit_futures_price=futures_price,
                            entry_spread_pct=pos['entry_spread_pct'],
                            exit_spread_pct=spread_pct,
                            entry_mispricing=pos['entry_mispricing'],
                            exit_mispricing=mispricing,
                            entry_fair_value=pos['entry_fair_value'],
                            exit_fair_value=current_data['fair_value_premium'],
                            entry_confidence=pos['entry_confidence'],
                            entry_volume_ratio=pos['entry_volume_ratio'],
                            entry_z_score=pos['entry_z_score'],
                            exit_z_score=current_data['z_score'],
                            days_to_expiry_entry=pos['days_to_expiry'],
                            days_to_expiry_exit=current_data['days_to_expiry'],
                            quantity=quantity,
                            spot_pnl=spot_pnl,
                            futures_pnl=futures_pnl,
                            total_pnl=total_pnl,
                            total_pnl_pct=total_pnl_pct,
                            exit_reason=exit_reason,
                            holding_period_minutes=(current_time - pos['entry_time']).total_seconds() / 60,
                            target_spread_pct=pos['target_spread_pct'],
                            stop_loss_spread_pct=pos['stop_loss_spread_pct']
                        )

                        trades.append(trade)
                        cash += total_pnl
                        active_positions.remove(pos)

                        self.print_trade(trade)

        # Calculate metrics
        metrics = self.calculate_metrics(trades, df)

        return {
            'pair': pair,
            'trades': trades,
            'metrics': metrics,
            'final_capital': cash,
            'data': df
        }

    def print_trade(self, trade: ArbitrageTrade):
        """Print trade details"""
        print(f"\n✓ ARBITRAGE TRADE CLOSED - LIVE STRATEGY")
        print(f"   Position: {trade.position_type.replace('_', ' ').upper()}")
        print(f"   Entry: {trade.entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Exit:  {trade.exit_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Duration: {trade.holding_period_minutes:.1f} minutes")
        print(f"")
        print(f"   Entry: Spot ₹{trade.entry_spot_price:.2f} | Futures ₹{trade.entry_futures_price:.2f}")
        print(f"   Exit:  Spot ₹{trade.exit_spot_price:.2f} | Futures ₹{trade.exit_futures_price:.2f}")
        print(f"")
        print(f"   Entry Spread:      {trade.entry_spread_pct:.3f}%")
        print(f"   Exit Spread:       {trade.exit_spread_pct:.3f}%")
        print(f"   Spread Change:     {trade.exit_spread_pct - trade.entry_spread_pct:+.3f}%")
        print(f"")
        print(f"   Entry Fair Value:  {trade.entry_fair_value:.3f}%")
        print(f"   Entry Mispricing:  {trade.entry_mispricing:+.3f}%")
        print(f"   Exit Mispricing:   {trade.exit_mispricing:+.3f}%")
        print(f"   Entry Confidence:  {trade.entry_confidence:.2f}")
        print(f"")
        print(f"   Target Spread:     {trade.target_spread_pct:.3f}%")
        print(f"   Stop Loss Spread:  {trade.stop_loss_spread_pct:.3f}%")
        print(f"")
        print(f"   Spot Leg P&L:      ₹{trade.spot_pnl:,.2f}")
        print(f"   Futures Leg P&L:   ₹{trade.futures_pnl:,.2f}")
        print(f"   Total P&L:         ₹{trade.total_pnl:,.2f} ({trade.total_pnl_pct:+.2f}%)")
        print(f"   Exit Reason:       {trade.exit_reason}")

        if trade.total_pnl > 0:
            print(f"   Result: ✓ PROFIT")
        else:
            print(f"   Result: ✗ LOSS")

    def calculate_metrics(self, trades: List[ArbitrageTrade], df):
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl_per_trade': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_holding_period': 0,
                'avg_confidence': 0,
                'avg_entry_mispricing': 0,
                'avg_exit_mispricing': 0,
                'target_exits': 0,
                'stop_loss_exits': 0,
                'expiry_exits': 0,
                'square_off_exits': 0,
                'long_spot_trades': 0,
                'short_spot_trades': 0
            }

        winning_trades = [t for t in trades if t.total_pnl > 0]
        losing_trades = [t for t in trades if t.total_pnl <= 0]

        target_exits = [t for t in trades if t.exit_reason == 'TARGET_PROFIT']
        stop_loss_exits = [t for t in trades if t.exit_reason == 'STOP_LOSS']
        expiry_exits = [t for t in trades if t.exit_reason == 'NEAR_EXPIRY']
        square_off_exits = [t for t in trades if t.exit_reason == 'SQUARE_OFF_TIME']

        long_spot = [t for t in trades if t.position_type == 'long_spot_short_futures']
        short_spot = [t for t in trades if t.position_type == 'short_spot_long_futures']

        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(trades)) * 100,
            'total_pnl': sum(t.total_pnl for t in trades),
            'avg_pnl_per_trade': np.mean([t.total_pnl for t in trades]),
            'best_trade': max(t.total_pnl for t in trades),
            'worst_trade': min(t.total_pnl for t in trades),
            'avg_holding_period': np.mean([t.holding_period_minutes for t in trades]),
            'avg_confidence': np.mean([t.entry_confidence for t in trades]),
            'avg_entry_mispricing': np.mean([abs(t.entry_mispricing) for t in trades]),
            'avg_exit_mispricing': np.mean([abs(t.exit_mispricing) for t in trades]),
            'target_exits': len(target_exits),
            'stop_loss_exits': len(stop_loss_exits),
            'expiry_exits': len(expiry_exits),
            'square_off_exits': len(square_off_exits),
            'long_spot_trades': len(long_spot),
            'short_spot_trades': len(short_spot),
            'long_spot_pnl': sum(t.total_pnl for t in long_spot),
            'short_spot_pnl': sum(t.total_pnl for t in short_spot)
        }

    def run_backtest(self):
        """Run backtest for all pairs"""
        print(f"\n{'=' * 100}")
        print(f"STARTING LIVE STRATEGY BACKTEST")
        print(f"{'=' * 100}")
        print(f"Pairs: {len(self.symbol_pairs)}")
        print(f"Method: Fair Value Mispricing + Multi-Factor Confidence")
        print(f"Entry: ±{self.min_spread_threshold}% mispricing")
        print(f"Exit: {self.target_profit_pct}% target | {self.stop_loss_pct}% stop")
        print(f"{'=' * 100}\n")

        for pair in self.symbol_pairs:
            try:
                result = self.backtest_pair(pair)
                if result:
                    self.results[pair['base_name']] = result
            except Exception as e:
                print(f"✗ Error with {pair['base_name']}: {e}")
                import traceback
                traceback.print_exc()

        if self.results:
            self.print_overall_summary()

        return self.results

    def print_overall_summary(self):
        """Print overall performance summary"""
        print(f"\n{'=' * 100}")
        print("OVERALL LIVE STRATEGY PERFORMANCE")
        print(f"{'=' * 100}")

        total_pairs = len(self.results)
        total_trades = sum(r['metrics']['total_trades'] for r in self.results.values())
        total_pnl = sum(r['final_capital'] - self.portfolio_value for r in self.results.values())
        avg_return = (total_pnl / (self.portfolio_value * total_pairs)) * 100

        profitable_pairs = sum(1 for r in self.results.values() if r['final_capital'] > self.portfolio_value)

        print(f"Pairs Tested:          {total_pairs}")
        print(f"Profitable Pairs:      {profitable_pairs} ({(profitable_pairs / total_pairs) * 100:.1f}%)")
        print(f"Total Trades:          {total_trades}")
        print(f"Total P&L:             ₹{total_pnl:,.2f}")
        print(f"Average Return/Pair:   {avg_return:+.2f}%")

        if total_trades > 0:
            all_trades = []
            for result in self.results.values():
                all_trades.extend(result['trades'])

            winning = [t for t in all_trades if t.total_pnl > 0]
            overall_win_rate = (len(winning) / len(all_trades)) * 100
            avg_holding = np.mean([t.holding_period_minutes for t in all_trades])
            avg_confidence = np.mean([t.entry_confidence for t in all_trades])
            avg_mispricing = np.mean([abs(t.entry_mispricing) for t in all_trades])

            print(f"Overall Win Rate:      {overall_win_rate:.1f}%")
            print(f"Avg Holding Period:    {avg_holding:.1f} minutes")
            print(f"Avg Entry Confidence:  {avg_confidence:.2f}")
            print(f"Avg Entry Mispricing:  {avg_mispricing:.3f}%")

            # Exit reason breakdown
            print(f"\nExit Reason Breakdown:")
            target = sum(r['metrics']['target_exits'] for r in self.results.values())
            stop_loss = sum(r['metrics']['stop_loss_exits'] for r in self.results.values())
            expiry = sum(r['metrics']['expiry_exits'] for r in self.results.values())
            square_off = sum(r['metrics']['square_off_exits'] for r in self.results.values())

            print(f"  Target Profit:  {target} ({(target / total_trades) * 100:.1f}%)")
            print(f"  Stop Loss:      {stop_loss} ({(stop_loss / total_trades) * 100:.1f}%)")
            print(f"  Near Expiry:    {expiry} ({(expiry / total_trades) * 100:.1f}%)")
            print(f"  Square-off:     {square_off} ({(square_off / total_trades) * 100:.1f}%)")

            # Position type breakdown
            print(f"\nPosition Type Breakdown:")
            long_spot = [t for t in all_trades if t.position_type == 'long_spot_short_futures']
            short_spot = [t for t in all_trades if t.position_type == 'short_spot_long_futures']

            if long_spot:
                long_spot_pnl = sum(t.total_pnl for t in long_spot)
                long_spot_win = len([t for t in long_spot if t.total_pnl > 0])
                print(f"  Long Spot:   {len(long_spot)} trades | P&L: ₹{long_spot_pnl:,.2f} | Win: {(long_spot_win / len(long_spot)) * 100:.1f}%")

            if short_spot:
                short_spot_pnl = sum(t.total_pnl for t in short_spot)
                short_spot_win = len([t for t in short_spot if t.total_pnl > 0])
                print(f"  Short Spot:  {len(short_spot)} trades | P&L: ₹{short_spot_pnl:,.2f} | Win: {(short_spot_win / len(short_spot)) * 100:.1f}%")

        # Best and worst pairs
        best_pair = max(self.results.items(), key=lambda x: x[1]['final_capital'])
        worst_pair = min(self.results.items(), key=lambda x: x[1]['final_capital'])

        print(f"\n✓ Best Pair:  {best_pair[0]}")
        print(f"   Return: {((best_pair[1]['final_capital'] - self.portfolio_value) / self.portfolio_value) * 100:+.2f}%")
        print(f"   Trades: {best_pair[1]['metrics']['total_trades']}")

        print(f"\n✗ Worst Pair: {worst_pair[0]}")
        print(f"   Return: {((worst_pair[1]['final_capital'] - self.portfolio_value) / self.portfolio_value) * 100:+.2f}%")
        print(f"   Trades: {worst_pair[1]['metrics']['total_trades']}")

        print(f"{'=' * 100}")

    def export_results(self, filename_prefix="live_strategy_fyers"):
        """Export results to CSV"""
        if not self.results:
            print("No results to export.")
            return

        # Collect all trades
        all_trades = []
        for pair_name, result in self.results.items():
            for trade in result['trades']:
                all_trades.append({
                    'pair': pair_name,
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'position_type': trade.position_type,
                    'entry_spot_price': trade.entry_spot_price,
                    'entry_futures_price': trade.entry_futures_price,
                    'exit_spot_price': trade.exit_spot_price,
                    'exit_futures_price': trade.exit_futures_price,
                    'entry_spread_pct': trade.entry_spread_pct,
                    'exit_spread_pct': trade.exit_spread_pct,
                    'entry_mispricing': trade.entry_mispricing,
                    'exit_mispricing': trade.exit_mispricing,
                    'entry_fair_value': trade.entry_fair_value,
                    'exit_fair_value': trade.exit_fair_value,
                    'entry_confidence': trade.entry_confidence,
                    'entry_volume_ratio': trade.entry_volume_ratio,
                    'entry_z_score': trade.entry_z_score,
                    'exit_z_score': trade.exit_z_score,
                    'days_to_expiry_entry': trade.days_to_expiry_entry,
                    'days_to_expiry_exit': trade.days_to_expiry_exit,
                    'quantity': trade.quantity,
                    'spot_pnl': trade.spot_pnl,
                    'futures_pnl': trade.futures_pnl,
                    'total_pnl': trade.total_pnl,
                    'total_pnl_pct': trade.total_pnl_pct,
                    'exit_reason': trade.exit_reason,
                    'holding_period_minutes': trade.holding_period_minutes,
                    'target_spread_pct': trade.target_spread_pct,
                    'stop_loss_spread_pct': trade.stop_loss_spread_pct
                })

        trades_df = pd.DataFrame(all_trades)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_trades_{timestamp}.csv"
        trades_df.to_csv(filename, index=False)
        print(f"\n✓ Trades exported to: {filename}")

        # Summary by pair
        summary_data = []
        for pair_name, result in self.results.items():
            metrics = result['metrics']
            returns = ((result['final_capital'] - self.portfolio_value) / self.portfolio_value) * 100

            summary_data.append({
                'pair': pair_name,
                'total_trades': metrics['total_trades'],
                'winning_trades': metrics['winning_trades'],
                'losing_trades': metrics['losing_trades'],
                'win_rate': metrics['win_rate'],
                'total_pnl': metrics['total_pnl'],
                'avg_pnl_per_trade': metrics['avg_pnl_per_trade'],
                'best_trade': metrics['best_trade'],
                'worst_trade': metrics['worst_trade'],
                'return_pct': returns,
                'final_capital': result['final_capital'],
                'avg_holding_period': metrics['avg_holding_period'],
                'avg_confidence': metrics['avg_confidence'],
                'avg_entry_mispricing': metrics['avg_entry_mispricing'],
                'target_exits': metrics['target_exits'],
                'stop_loss_exits': metrics['stop_loss_exits'],
                'expiry_exits': metrics['expiry_exits'],
                'square_off_exits': metrics['square_off_exits'],
                'long_spot_trades': metrics['long_spot_trades'],
                'short_spot_trades': metrics['short_spot_trades']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"{filename_prefix}_summary_{timestamp}.csv"
        summary_df.to_csv(summary_filename, index=False)
        print(f"✓ Summary exported to: {summary_filename}")


# Main execution
if __name__ == "__main__":
    """
    LIVE STRATEGY BACKTEST WITH FYERS API

    Tests the EXACT logic from FyersArbitrage live strategy:
    - Fair value premium calculation (cost of carry)
    - Mispricing-based entry signals
    - Multi-factor confidence scoring
    - Risk-based position sizing
    - Fixed profit target (0.3%) and stop loss (0.5%)

    Now fetches data from Fyers API instead of database files
    """

    # Example symbol pairs (adjust based on available Fyers symbols)
    symbol_pairs = [
        {
            'spot': 'NSE:RELIANCE-EQ',
            'futures': 'NSE:RELIANCE25DECFUT',  # Adjust month as needed
            'base_name': 'RELIANCE'
        },
        {
            'spot': 'NSE:HDFCBANK-EQ',
            'futures': 'NSE:HDFCBANK25DECFUT',
            'base_name': 'HDFCBANK'
        },
        {
            'spot': 'NSE:INFY-EQ',
            'futures': 'NSE:INFY25DECFUT',
            'base_name': 'INFY'
        },
        {
            'spot': 'NSE:TCS-EQ',
            'futures': 'NSE:TCS25DECFUT',
            'base_name': 'TCS'
        }
    ]

    backtester = LiveStrategyBacktester(
        symbols=symbol_pairs,
        access_token=None,  # Will use FYERS_ACCESS_TOKEN env var

        # Data parameters
        lookback_days=60,  # 60 days of historical data
        timeframe="5min",  # 5-minute candles

        # Live strategy parameters (from config/settings.py)
        portfolio_value=100000,
        risk_per_trade_pct=2.0,
        max_positions=5,

        # Entry thresholds
        min_spread_threshold=0.5,  # 0.5% minimum mispricing
        max_spread_threshold=3.0,  # 3.0% maximum spread
        min_confidence=0.7,  # 70% minimum confidence
        min_liquidity_ratio=1.5,  # 1.5x volume ratio

        # Exit targets
        target_profit_pct=0.3,  # 0.3% profit target
        stop_loss_pct=0.5,  # 0.5% stop loss

        # Other settings
        require_convergence_trend=False,  # Set True to require convergence
        square_off_time="15:15",  # 3:15 PM square-off
        basis_lookback=50,  # For z-score in confidence
        lot_size=1,
        risk_free_rate=0.05  # 5% annual
    )

    # Run backtest
    results = backtester.run_backtest()

    # Export results
    if results:
        backtester.export_results()
        print(f"\n✓ Live Strategy Backtest Complete!")
        print(f"Tested {len(results)} spot-futures pairs with LIVE strategy logic using Fyers API")