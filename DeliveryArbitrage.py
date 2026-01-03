import pandas as pd
import numpy as np
import json
from datetime import datetime, time, timedelta
import pytz
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

warnings.filterwarnings('ignore')


@dataclass
class DeliveryArbitrageTrade:
    """Data class to store delivery arbitrage trade information"""
    entry_time: datetime
    exit_time: datetime
    symbol_spot: str
    symbol_futures: str
    position_type: str  # 'long_futures_short_spot' or 'short_futures_long_spot'
    entry_spot_price: float
    entry_futures_price: float
    exit_spot_price: float
    exit_futures_price: float
    entry_basis: float
    exit_basis: float
    entry_basis_pct: float
    exit_basis_pct: float
    entry_z_score: float
    exit_z_score: float
    quantity: int
    spot_pnl: float
    futures_pnl: float
    funding_cost: float  # Overnight holding cost
    total_pnl: float
    total_pnl_pct: float
    exit_reason: str
    holding_period_days: float
    num_overnight_holds: int


class DeliveryArbitrageBacktester:
    def __init__(self, symbols, access_token=None,
                 lookback_days=90, timeframe="D",
                 basis_lookback=50, entry_zscore_threshold=2.0,
                 exit_zscore_threshold=0.5, stop_loss_pct=2.0,
                 max_holding_days=25, funding_rate_daily=0.02,
                 initial_capital=200000, lot_size=1):
        """
        Delivery Spot-Futures Arbitrage Strategy Backtester using Fyers API

        Strategy Logic:
        - Calculate basis = futures_price - spot_price
        - Calculate basis percentage = (basis / spot_price) * 100
        - Calculate z-score of basis percentage
        - ENTRY LONG FUTURES + SHORT SPOT: z-score < -entry_threshold (futures underpriced)
        - ENTRY SHORT FUTURES + LONG SPOT: z-score > +entry_threshold (futures overpriced)
        - EXIT: |z-score| < exit_threshold (basis converged to mean)
        - NO INTRADAY SQUARE-OFF: Positions held across days
        - LOWER BROKERAGE: Delivery trades have lower transaction costs

        Parameters:
        - symbols: List of dicts with 'spot', 'futures', 'base_name', 'expiry_date' keys
        - access_token: Fyers API access token (or set FYERS_ACCESS_TOKEN env var)
        - lookback_days: Number of days of historical data to fetch (default 90)
        - timeframe: Candle timeframe - "D" for daily (default), "60" for hourly
        - basis_lookback: Period for calculating basis mean and std dev (default 50)
        - entry_zscore_threshold: Z-score threshold for entry (default 2.0)
        - exit_zscore_threshold: Z-score threshold for exit (default 0.5)
        - stop_loss_pct: Stop loss on total position value (default 2%)
        - max_holding_days: Maximum days to hold position (default 25 days before expiry)
        - funding_rate_daily: Daily funding/carry cost as % (default 0.02%)
        - lot_size: Futures lot size multiplier (default 1)
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
        self.basis_lookback = basis_lookback
        self.entry_zscore_threshold = entry_zscore_threshold
        self.exit_zscore_threshold = exit_zscore_threshold
        self.stop_loss_pct = stop_loss_pct / 100
        self.max_holding_days = max_holding_days
        self.funding_rate_daily = funding_rate_daily / 100
        self.initial_capital = initial_capital
        self.lot_size = lot_size
        self.results = {}

        self.ist_tz = pytz.timezone('Asia/Kolkata')

        print("=" * 100)
        print("DELIVERY SPOT-FUTURES ARBITRAGE STRATEGY BACKTESTER (FYERS API)")
        print("=" * 100)
        print(f"Strategy: Multi-Day Market-Neutral Basis Trading")
        print(f"Trade Type: DELIVERY (Lower Brokerage)")
        print(f"Data Source: Fyers API")
        print(f"Lookback Period: {lookback_days} days")
        print(f"Timeframe: {timeframe}")
        print(f"Basis Lookback: {basis_lookback} periods")
        print(f"Entry Z-Score Threshold: ±{entry_zscore_threshold}")
        print(f"Exit Z-Score Threshold: ±{exit_zscore_threshold}")
        print(f"Stop Loss: {stop_loss_pct}% on total position")
        print(f"Max Holding Period: {max_holding_days} days")
        print(f"Daily Funding Cost: {funding_rate_daily}%")
        print(f"Lot Size: {lot_size}")
        print("=" * 100)

        print(f"\nSpot-Futures Pairs to backtest: {len(self.symbol_pairs)}")
        for pair in self.symbol_pairs:
            print(f"  - {pair['spot']} <-> {pair['futures']} (Expiry: {pair.get('expiry_date', 'Not specified')})")

    def fetch_historical_data(self, symbol, days=None):
        """
        Fetch historical data from Fyers API

        Args:
            symbol: Fyers symbol (e.g., "NSE:RELIANCE-EQ", "NSE:RELIANCE25JANFUT")
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

        # Convert to Unix timestamps
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

        fyers_timeframe = timeframe_map.get(self.timeframe, "D")

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

            return df[['close']]

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
        # Merge on index (timestamp)
        aligned = pd.merge(
            spot_data, futures_data,
            left_index=True, right_index=True,
            suffixes=('_spot', '_futures'),
            how='inner'
        )

        print(f"  Aligned data: {len(aligned)} common timestamps")

        return aligned

    def calculate_basis_statistics(self, df):
        """Calculate basis and z-score"""
        # Basis = Futures - Spot
        df['basis'] = df['close_futures'] - df['close_spot']

        # Basis percentage = (basis / spot) * 100
        df['basis_pct'] = (df['basis'] / df['close_spot']) * 100

        # Rolling statistics
        df['basis_mean'] = df['basis_pct'].rolling(window=self.basis_lookback).mean()
        df['basis_std'] = df['basis_pct'].rolling(window=self.basis_lookback).std()

        # Z-score = (current_basis - mean) / std
        df['z_score'] = (df['basis_pct'] - df['basis_mean']) / df['basis_std']

        # Fair value = spot + (spot * basis_mean / 100)
        df['fair_value'] = df['close_spot'] + (df['close_spot'] * df['basis_mean'] / 100)

        # Mispricing = (futures - fair_value) / fair_value
        df['mispricing_pct'] = ((df['close_futures'] - df['fair_value']) / df['fair_value']) * 100

        return df

    def generate_arbitrage_signals(self, df):
        """Generate arbitrage entry and exit signals"""
        # Entry signals
        # Long Futures + Short Spot: When futures are underpriced (negative z-score)
        df['long_futures_signal'] = (
                (df['z_score'] < -self.entry_zscore_threshold) &
                (~df['z_score'].isna()) &
                (df['basis_std'] > 0)
        )

        # Short Futures + Long Spot: When futures are overpriced (positive z-score)
        df['short_futures_signal'] = (
                (df['z_score'] > self.entry_zscore_threshold) &
                (~df['z_score'].isna()) &
                (df['basis_std'] > 0)
        )

        # Exit signals: Basis converged (z-score near zero)
        df['exit_signal'] = (
                (abs(df['z_score']) < self.exit_zscore_threshold) &
                (~df['z_score'].isna())
        )

        return df

    def calculate_funding_cost(self, position_value, num_days):
        """Calculate overnight funding/carry cost"""
        return position_value * self.funding_rate_daily * num_days

    def days_to_expiry(self, current_date, expiry_date):
        """Calculate days remaining to futures expiry"""
        if expiry_date is None:
            return 999  # Large number if no expiry specified

        try:
            if isinstance(expiry_date, str):
                expiry = datetime.strptime(expiry_date, "%Y-%m-%d").date()
            else:
                expiry = expiry_date

            if isinstance(current_date, datetime):
                current = current_date.date()
            else:
                current = current_date

            return (expiry - current).days
        except:
            return 999

    def backtest_pair(self, pair):
        """Backtest delivery arbitrage strategy for a spot-futures pair"""
        spot_symbol = pair['spot']
        futures_symbol = pair['futures']
        base_name = pair['base_name']
        expiry_date = pair.get('expiry_date', None)

        print(f"\n{'=' * 100}")
        print(f"BACKTESTING: {base_name} (DELIVERY)")
        print(f"Spot: {spot_symbol} | Futures: {futures_symbol}")
        if expiry_date:
            print(f"Futures Expiry: {expiry_date}")
        print(f"{'=' * 100}")

        # Load data
        spot_data, futures_data = self.load_spot_futures_data(spot_symbol, futures_symbol)

        if spot_data is None or futures_data is None:
            print(f"  Insufficient data for {base_name}")
            return None

        # Align timestamps
        df = self.align_spot_futures_data(spot_data, futures_data)

        if len(df) < self.basis_lookback * 2:
            print(f"  Insufficient aligned data for {base_name}")
            return None

        # Calculate basis statistics
        df = self.calculate_basis_statistics(df)
        df = self.generate_arbitrage_signals(df)

        # Initialize trading
        cash = self.initial_capital
        position = None  # 'long_futures' or 'short_futures' or None
        entry_price_spot = 0
        entry_price_futures = 0
        entry_basis = 0
        entry_z_score = 0
        entry_time = None
        entry_index = 0
        trades: List[DeliveryArbitrageTrade] = []

        # Backtest loop
        for i in range(self.basis_lookback, len(df)):
            current_time = df.index[i]
            spot_price = df.iloc[i]['close_spot']
            futures_price = df.iloc[i]['close_futures']
            current_basis = df.iloc[i]['basis']
            current_z_score = df.iloc[i]['z_score']

            # Skip if z-score is NaN
            if pd.isna(current_z_score):
                continue

            # Check days to expiry
            days_left = self.days_to_expiry(current_time, expiry_date)

            # Force exit if approaching expiry (3 days before)
            if position and days_left <= 3:
                holding_days = (current_time - entry_time).days
                quantity = int((cash * 0.5) / entry_price_spot)  # 50% per leg
                position_value = quantity * entry_price_spot * 2

                # Calculate funding cost
                num_overnight = max(0, holding_days)
                funding_cost = self.calculate_funding_cost(position_value, num_overnight)

                # Calculate P&L for both legs
                if position == 'long_futures':
                    spot_pnl = quantity * (entry_price_spot - spot_price)  # Short spot
                    futures_pnl = quantity * (futures_price - entry_price_futures) * self.lot_size  # Long futures
                else:  # short_futures
                    spot_pnl = quantity * (spot_price - entry_price_spot)  # Long spot
                    futures_pnl = quantity * (entry_price_futures - futures_price) * self.lot_size  # Short futures

                total_pnl = spot_pnl + futures_pnl - funding_cost
                total_pnl_pct = (total_pnl / position_value) * 100

                trade = DeliveryArbitrageTrade(
                    entry_time=entry_time,
                    exit_time=current_time,
                    symbol_spot=spot_symbol,
                    symbol_futures=futures_symbol,
                    position_type=position,
                    entry_spot_price=entry_price_spot,
                    entry_futures_price=entry_price_futures,
                    exit_spot_price=spot_price,
                    exit_futures_price=futures_price,
                    entry_basis=entry_basis,
                    exit_basis=current_basis,
                    entry_basis_pct=(entry_basis / entry_price_spot) * 100,
                    exit_basis_pct=(current_basis / spot_price) * 100,
                    entry_z_score=entry_z_score,
                    exit_z_score=current_z_score,
                    quantity=quantity,
                    spot_pnl=spot_pnl,
                    futures_pnl=futures_pnl,
                    funding_cost=funding_cost,
                    total_pnl=total_pnl,
                    total_pnl_pct=total_pnl_pct,
                    exit_reason='APPROACHING_EXPIRY',
                    holding_period_days=holding_days,
                    num_overnight_holds=num_overnight
                )

                trades.append(trade)
                cash += total_pnl
                position = None

                self.print_trade(trade)

            # Entry signals (only if no position and sufficient time to expiry)
            elif not position and days_left > self.max_holding_days:
                # Long Futures + Short Spot (futures underpriced)
                if df.iloc[i]['long_futures_signal']:
                    position = 'long_futures'
                    entry_price_spot = spot_price
                    entry_price_futures = futures_price
                    entry_basis = current_basis
                    entry_z_score = current_z_score
                    entry_time = current_time
                    entry_index = i

                    print(f"\n  ENTRY: LONG FUTURES + SHORT SPOT (DELIVERY)")
                    print(f"   Time: {current_time.strftime('%Y-%m-%d')}")
                    print(f"   Spot: ₹{spot_price:.2f} | Futures: ₹{futures_price:.2f}")
                    print(f"   Basis: ₹{current_basis:.2f} ({(current_basis / spot_price) * 100:.3f}%)")
                    print(f"   Z-Score: {current_z_score:.2f} (Futures UNDERPRICED)")
                    print(f"   Days to Expiry: {days_left}")

                # Short Futures + Long Spot (futures overpriced)
                elif df.iloc[i]['short_futures_signal']:
                    position = 'short_futures'
                    entry_price_spot = spot_price
                    entry_price_futures = futures_price
                    entry_basis = current_basis
                    entry_z_score = current_z_score
                    entry_time = current_time
                    entry_index = i

                    print(f"\n  ENTRY: SHORT FUTURES + LONG SPOT (DELIVERY)")
                    print(f"   Time: {current_time.strftime('%Y-%m-%d')}")
                    print(f"   Spot: ₹{spot_price:.2f} | Futures: ₹{futures_price:.2f}")
                    print(f"   Basis: ₹{current_basis:.2f} ({(current_basis / spot_price) * 100:.3f}%)")
                    print(f"   Z-Score: {current_z_score:.2f} (Futures OVERPRICED)")
                    print(f"   Days to Expiry: {days_left}")

            # Exit signals
            elif position:
                should_exit = False
                exit_reason = ""
                holding_days = (current_time - entry_time).days

                # Exit on basis convergence
                if df.iloc[i]['exit_signal']:
                    should_exit = True
                    exit_reason = "BASIS_CONVERGENCE"

                # Max holding period reached
                elif holding_days >= self.max_holding_days:
                    should_exit = True
                    exit_reason = "MAX_HOLDING_PERIOD"

                # Stop loss check
                quantity = int((cash * 0.5) / entry_price_spot)
                position_value = quantity * entry_price_spot * 2
                num_overnight = max(0, holding_days)
                funding_cost = self.calculate_funding_cost(position_value, num_overnight)

                if position == 'long_futures':
                    spot_pnl = quantity * (entry_price_spot - spot_price)
                    futures_pnl = quantity * (futures_price - entry_price_futures) * self.lot_size
                else:
                    spot_pnl = quantity * (spot_price - entry_price_spot)
                    futures_pnl = quantity * (entry_price_futures - futures_price) * self.lot_size

                total_pnl = spot_pnl + futures_pnl - funding_cost
                loss_pct = abs(total_pnl / position_value)

                if total_pnl < 0 and loss_pct >= self.stop_loss_pct:
                    should_exit = True
                    exit_reason = "STOP_LOSS"

                if should_exit:
                    total_pnl_pct = (total_pnl / position_value) * 100

                    trade = DeliveryArbitrageTrade(
                        entry_time=entry_time,
                        exit_time=current_time,
                        symbol_spot=spot_symbol,
                        symbol_futures=futures_symbol,
                        position_type=position,
                        entry_spot_price=entry_price_spot,
                        entry_futures_price=entry_price_futures,
                        exit_spot_price=spot_price,
                        exit_futures_price=futures_price,
                        entry_basis=entry_basis,
                        exit_basis=current_basis,
                        entry_basis_pct=(entry_basis / entry_price_spot) * 100,
                        exit_basis_pct=(current_basis / spot_price) * 100,
                        entry_z_score=entry_z_score,
                        exit_z_score=current_z_score,
                        quantity=quantity,
                        spot_pnl=spot_pnl,
                        futures_pnl=futures_pnl,
                        funding_cost=funding_cost,
                        total_pnl=total_pnl,
                        total_pnl_pct=total_pnl_pct,
                        exit_reason=exit_reason,
                        holding_period_days=holding_days,
                        num_overnight_holds=num_overnight
                    )

                    trades.append(trade)
                    cash += total_pnl
                    position = None

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

    def print_trade(self, trade: DeliveryArbitrageTrade):
        """Print trade details"""
        print(f"\n  DELIVERY ARBITRAGE TRADE CLOSED")
        print(f"   Position: {trade.position_type.replace('_', ' ').upper()}")
        print(f"   Entry:  {trade.entry_time.strftime('%Y-%m-%d')}")
        print(f"   Exit:   {trade.exit_time.strftime('%Y-%m-%d')}")
        print(f"   Duration: {trade.holding_period_days} days ({trade.num_overnight_holds} overnight holds)")
        print(f"")
        print(f"   Entry Prices  - Spot: ₹{trade.entry_spot_price:.2f} | Futures: ₹{trade.entry_futures_price:.2f}")
        print(f"   Exit Prices   - Spot: ₹{trade.exit_spot_price:.2f} | Futures: ₹{trade.exit_futures_price:.2f}")
        print(f"   Entry Basis:  ₹{trade.entry_basis:.2f} ({trade.entry_basis_pct:.3f}%) | Z-Score: {trade.entry_z_score:.2f}")
        print(f"   Exit Basis:   ₹{trade.exit_basis:.2f} ({trade.exit_basis_pct:.3f}%) | Z-Score: {trade.exit_z_score:.2f}")
        print(f"   Basis Change: {trade.exit_basis - trade.entry_basis:.2f} ({trade.exit_basis_pct - trade.entry_basis_pct:.3f}%)")
        print(f"")
        print(f"   Spot Leg P&L:      ₹{trade.spot_pnl:,.2f}")
        print(f"   Futures Leg P&L:   ₹{trade.futures_pnl:,.2f}")
        print(f"   Funding Cost:      ₹{trade.funding_cost:,.2f}")
        print(f"   Net P&L:           ₹{trade.total_pnl:,.2f} ({trade.total_pnl_pct:+.2f}%)")
        print(f"   Exit Reason:       {trade.exit_reason}")

        if trade.total_pnl > 0:
            print(f"   Result: ✓ PROFIT")
        else:
            print(f"   Result: ✗ LOSS")

    def calculate_metrics(self, trades: List[DeliveryArbitrageTrade], df):
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_funding_cost': 0,
                'avg_pnl_per_trade': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_holding_period': 0,
                'avg_overnight_holds': 0,
                'avg_entry_z_score': 0,
                'avg_exit_z_score': 0,
                'avg_basis_convergence': 0,
                'basis_convergence_trades': 0,
                'stop_loss_trades': 0,
                'max_holding_trades': 0,
                'expiry_trades': 0,
                'long_futures_trades': 0,
                'short_futures_trades': 0,
                'avg_spot_pnl': 0,
                'avg_futures_pnl': 0
            }

        winning_trades = [t for t in trades if t.total_pnl > 0]
        losing_trades = [t for t in trades if t.total_pnl <= 0]

        basis_convergence = [t for t in trades if t.exit_reason == 'BASIS_CONVERGENCE']
        stop_loss = [t for t in trades if t.exit_reason == 'STOP_LOSS']
        max_holding = [t for t in trades if t.exit_reason == 'MAX_HOLDING_PERIOD']
        expiry = [t for t in trades if t.exit_reason == 'APPROACHING_EXPIRY']

        long_futures = [t for t in trades if t.position_type == 'long_futures']
        short_futures = [t for t in trades if t.position_type == 'short_futures']

        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(trades)) * 100,
            'total_pnl': sum(t.total_pnl for t in trades),
            'total_funding_cost': sum(t.funding_cost for t in trades),
            'avg_pnl_per_trade': np.mean([t.total_pnl for t in trades]),
            'best_trade': max(t.total_pnl for t in trades),
            'worst_trade': min(t.total_pnl for t in trades),
            'avg_holding_period': np.mean([t.holding_period_days for t in trades]),
            'avg_overnight_holds': np.mean([t.num_overnight_holds for t in trades]),
            'avg_entry_z_score': np.mean([abs(t.entry_z_score) for t in trades]),
            'avg_exit_z_score': np.mean([abs(t.exit_z_score) for t in trades]),
            'avg_basis_convergence': np.mean([abs(t.entry_basis_pct - t.exit_basis_pct) for t in trades]),
            'basis_convergence_trades': len(basis_convergence),
            'stop_loss_trades': len(stop_loss),
            'max_holding_trades': len(max_holding),
            'expiry_trades': len(expiry),
            'long_futures_trades': len(long_futures),
            'short_futures_trades': len(short_futures),
            'avg_spot_pnl': np.mean([t.spot_pnl for t in trades]),
            'avg_futures_pnl': np.mean([t.futures_pnl for t in trades]),
            'long_futures_pnl': sum(t.total_pnl for t in long_futures),
            'short_futures_pnl': sum(t.total_pnl for t in short_futures)
        }

    def run_backtest(self):
        """Run backtest for all pairs"""
        print(f"\nStarting Delivery Spot-Futures Arbitrage Backtest")
        print(f"Pairs: {len(self.symbol_pairs)}")
        print(f"Entry Threshold: Z-score ±{self.entry_zscore_threshold}")
        print(f"Exit Threshold: Z-score ±{self.exit_zscore_threshold}")
        print(f"Max Holding: {self.max_holding_days} days\n")

        for pair in self.symbol_pairs:
            try:
                result = self.backtest_pair(pair)
                if result:
                    self.results[pair['base_name']] = result
            except Exception as e:
                print(f"\n✗ Error with {pair['base_name']}: {e}")
                import traceback
                traceback.print_exc()

        if self.results:
            self.print_overall_summary()

        return self.results

    def print_overall_summary(self):
        """Print overall performance summary"""
        print(f"\n{'=' * 100}")
        print("OVERALL DELIVERY ARBITRAGE PERFORMANCE")
        print(f"{'=' * 100}")

        total_pairs = len(self.results)
        total_trades = sum(r['metrics']['total_trades'] for r in self.results.values())
        total_pnl = sum(r['final_capital'] - self.initial_capital for r in self.results.values())
        total_funding = sum(r['metrics']['total_funding_cost'] for r in self.results.values())
        avg_return = (total_pnl / (self.initial_capital * total_pairs)) * 100

        profitable_pairs = sum(1 for r in self.results.values() if r['final_capital'] > self.initial_capital)

        print(f"Pairs Tested:          {total_pairs}")
        print(f"Profitable Pairs:      {profitable_pairs} ({(profitable_pairs / total_pairs) * 100:.1f}%)")
        print(f"Total Trades:          {total_trades}")
        print(f"Total P&L:             ₹{total_pnl:,.2f}")
        print(f"Total Funding Cost:    ₹{total_funding:,.2f}")
        print(f"Net P&L:               ₹{total_pnl:,.2f}")
        print(f"Average Return/Pair:   {avg_return:+.2f}%")

        if total_trades > 0:
            all_trades = []
            for result in self.results.values():
                all_trades.extend(result['trades'])

            winning = [t for t in all_trades if t.total_pnl > 0]
            overall_win_rate = (len(winning) / len(all_trades)) * 100
            avg_holding = np.mean([t.holding_period_days for t in all_trades])
            avg_overnight = np.mean([t.num_overnight_holds for t in all_trades])

            print(f"Overall Win Rate:      {overall_win_rate:.1f}%")
            print(f"Avg Holding Period:    {avg_holding:.1f} days ({avg_overnight:.1f} overnight holds)")

            # Position type breakdown
            long_fut = [t for t in all_trades if t.position_type == 'long_futures']
            short_fut = [t for t in all_trades if t.position_type == 'short_futures']

            print(f"\nPosition Type Breakdown:")
            if long_fut:
                long_fut_pnl = sum(t.total_pnl for t in long_fut)
                long_fut_win = len([t for t in long_fut if t.total_pnl > 0])
                print(f"  Long Futures:  {len(long_fut)} trades | P&L: ₹{long_fut_pnl:,.2f} | Win Rate: {(long_fut_win / len(long_fut)) * 100:.1f}%")

            if short_fut:
                short_fut_pnl = sum(t.total_pnl for t in short_fut)
                short_fut_win = len([t for t in short_fut if t.total_pnl > 0])
                print(f"  Short Futures: {len(short_fut)} trades | P&L: ₹{short_fut_pnl:,.2f} | Win Rate: {(short_fut_win / len(short_fut)) * 100:.1f}%")

            # Exit reason breakdown
            print(f"\nExit Reason Breakdown:")
            convergence = sum(r['metrics']['basis_convergence_trades'] for r in self.results.values())
            stop_loss = sum(r['metrics']['stop_loss_trades'] for r in self.results.values())
            max_holding = sum(r['metrics']['max_holding_trades'] for r in self.results.values())
            expiry = sum(r['metrics']['expiry_trades'] for r in self.results.values())

            print(f"  Basis Convergence:    {convergence} ({(convergence / total_trades) * 100:.1f}%)")
            print(f"  Stop Loss:            {stop_loss} ({(stop_loss / total_trades) * 100:.1f}%)")
            print(f"  Max Holding Period:   {max_holding} ({(max_holding / total_trades) * 100:.1f}%)")
            print(f"  Approaching Expiry:   {expiry} ({(expiry / total_trades) * 100:.1f}%)")

        # Best and worst pairs
        if self.results:
            best_pair = max(self.results.items(), key=lambda x: x[1]['final_capital'])
            worst_pair = min(self.results.items(), key=lambda x: x[1]['final_capital'])

            print(f"\n Best Pair:  {best_pair[0]}")
            print(f"   Return: {((best_pair[1]['final_capital'] - self.initial_capital) / self.initial_capital) * 100:+.2f}%")
            print(f"   Trades: {best_pair[1]['metrics']['total_trades']}")

            print(f"\n Worst Pair: {worst_pair[0]}")
            print(f"   Return: {((worst_pair[1]['final_capital'] - self.initial_capital) / self.initial_capital) * 100:+.2f}%")
            print(f"   Trades: {worst_pair[1]['metrics']['total_trades']}")

        print(f"{'=' * 100}")

    def export_results(self, filename_prefix="delivery_arbitrage"):
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
                    'entry_basis': trade.entry_basis,
                    'exit_basis': trade.exit_basis,
                    'entry_basis_pct': trade.entry_basis_pct,
                    'exit_basis_pct': trade.exit_basis_pct,
                    'entry_z_score': trade.entry_z_score,
                    'exit_z_score': trade.exit_z_score,
                    'quantity': trade.quantity,
                    'spot_pnl': trade.spot_pnl,
                    'futures_pnl': trade.futures_pnl,
                    'funding_cost': trade.funding_cost,
                    'total_pnl': trade.total_pnl,
                    'total_pnl_pct': trade.total_pnl_pct,
                    'exit_reason': trade.exit_reason,
                    'holding_period_days': trade.holding_period_days,
                    'num_overnight_holds': trade.num_overnight_holds
                })

        trades_df = pd.DataFrame(all_trades)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_trades_{timestamp}.csv"
        trades_df.to_csv(filename, index=False)
        print(f"\nTrades exported to: {filename}")

        # Summary by pair
        summary_data = []
        for pair_name, result in self.results.items():
            metrics = result['metrics']
            returns = ((result['final_capital'] - self.initial_capital) / self.initial_capital) * 100

            summary_data.append({
                'pair': pair_name,
                'total_trades': metrics['total_trades'],
                'winning_trades': metrics['winning_trades'],
                'losing_trades': metrics['losing_trades'],
                'win_rate': metrics['win_rate'],
                'total_pnl': metrics['total_pnl'],
                'total_funding_cost': metrics['total_funding_cost'],
                'avg_pnl_per_trade': metrics['avg_pnl_per_trade'],
                'best_trade': metrics['best_trade'],
                'worst_trade': metrics['worst_trade'],
                'return_pct': returns,
                'final_capital': result['final_capital'],
                'avg_holding_period_days': metrics['avg_holding_period'],
                'avg_overnight_holds': metrics['avg_overnight_holds'],
                'basis_convergence_trades': metrics['basis_convergence_trades'],
                'stop_loss_trades': metrics['stop_loss_trades'],
                'max_holding_trades': metrics['max_holding_trades'],
                'expiry_trades': metrics['expiry_trades'],
                'long_futures_trades': metrics['long_futures_trades'],
                'short_futures_trades': metrics['short_futures_trades']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"{filename_prefix}_summary_{timestamp}.csv"
        summary_df.to_csv(summary_filename, index=False)
        print(f"Summary exported to: {summary_filename}")


# Main execution
if __name__ == "__main__":
    """
    DELIVERY SPOT-FUTURES ARBITRAGE STRATEGY (FYERS API)

    Market-Neutral Strategy with Multi-Day Holding (Lower Brokerage)

    Key Advantages Over Intraday:
    - Lower transaction costs (delivery vs intraday brokerage)
    - No forced 3:20 PM square-off
    - Allows basis to converge naturally over days
    - Better risk-reward for patient traders

    Entry Logic:
    - Long Futures + Short Spot: When basis z-score < -2.0 (futures underpriced)
    - Short Futures + Long Spot: When basis z-score > +2.0 (futures overpriced)

    Exit Logic:
    - Basis Convergence: |z-score| < 0.5
    - Stop Loss: 2% on total position value
    - Max Holding: 25 days (or before futures expiry)
    - Approaching Expiry: 3 days before expiry date

    Costs Considered:
    - Daily funding/carry cost: 0.02% per day
    - Reflects overnight holding costs

    Risk Profile:
    - Market-neutral (hedged position)
    - Lower risk than directional strategies
    - Profits from basis convergence over multiple days

    Setup:
    - Set FYERS_ACCESS_TOKEN environment variable
    - Or pass access_token parameter to backtester
    - Specify futures expiry dates for each pair
    """

    # Example symbol pairs with expiry dates
    symbol_pairs = [
        {
            'spot': 'NSE:RELIANCE-EQ',
            'futures': 'NSE:RELIANCE25JANFUT',
            'base_name': 'RELIANCE',
            'expiry_date': '2025-01-30'  # Last Thursday of January
        },
        {
            'spot': 'NSE:HDFCBANK-EQ',
            'futures': 'NSE:HDFCBANK25JANFUT',
            'base_name': 'HDFCBANK',
            'expiry_date': '2025-01-30'
        },
        {
            'spot': 'NSE:INFY-EQ',
            'futures': 'NSE:INFY25JANFUT',
            'base_name': 'INFY',
            'expiry_date': '2025-01-30'
        },
        {
            'spot': 'NSE:TCS-EQ',
            'futures': 'NSE:TCS25JANFUT',
            'base_name': 'TCS',
            'expiry_date': '2025-01-30'
        }
    ]

    # Initialize backtester
    backtester = DeliveryArbitrageBacktester(
        symbols=symbol_pairs,
        access_token=None,  # Will use FYERS_ACCESS_TOKEN env var
        lookback_days=30,  # 90 days of historical data
        timeframe="D",  # Daily candles (can use "60" for hourly if needed)
        basis_lookback=50,  # 50-period lookback for basis statistics
        entry_zscore_threshold=2.0,  # Enter when z-score > 2 or < -2
        exit_zscore_threshold=0.5,  # Exit when |z-score| < 0.5
        stop_loss_pct=2.0,  # 2% stop loss on total position
        max_holding_days=25,  # Maximum 25 days holding
        funding_rate_daily=0.02,  # 0.02% daily funding cost
        initial_capital=200000,
        lot_size=1  # Adjust for actual futures lot size
    )

    # Run backtest
    results = backtester.run_backtest()

    # Export results
    if results:
        backtester.export_results()
        print(f"\nDelivery Arbitrage Backtest Complete!")
        print(f"Tested {len(results)} spot-futures pairs")