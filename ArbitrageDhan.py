"""
Dhan Broker Arbitrage Trading Backtester
==========================================

This backtester identifies and trades arbitrage opportunities using Dhan broker's
historical data API. Supports multiple arbitrage strategies:

1. Spot-Futures Arbitrage (Cash & Carry)
2. Exchange Arbitrage (NSE vs BSE)
3. Calendar Spread Arbitrage (Near vs Far month futures)
4. Synthetic Arbitrage (Futures vs Spot with dividends)

Author: AJ's Trading System
API Reference: https://dhanhq.co/docs/v2/historical-data/
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import pytz
import json
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time as time_module

warnings.filterwarnings('ignore')


@dataclass
class ArbitrageTrade:
    """Data class for arbitrage trade"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    strategy: str
    leg1_entry: float
    leg2_entry: float
    leg1_exit: float
    leg2_exit: float
    spread_entry: float
    spread_exit: float
    quantity: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    transaction_cost: float


class DhanArbitrageBacktester:
    """
    Comprehensive Arbitrage Trading Backtester using Dhan Historical Data API
    """

    def __init__(
            self,
            access_token: str,
            strategy_type: str = "SPOT_FUTURES",
            initial_capital: float = 1000000,
            min_spread_threshold: float = 0.3,  # Minimum spread % for entry
            max_spread_threshold: float = 0.1,  # Maximum spread % for exit
            position_size_pct: float = 20,
            transaction_cost_pct: float = 0.1,  # Total transaction cost %
            square_off_time: str = "15:20",
            symbols: List[str] = None,
            from_date: str = None,
            to_date: str = None
    ):
        """
        Initialize Dhan Arbitrage Backtester

        Parameters:
        -----------
        access_token : str
            Dhan API access token (JWT)
        strategy_type : str
            "SPOT_FUTURES", "EXCHANGE_ARB", "CALENDAR_SPREAD", "ALL"
        initial_capital : float
            Starting capital for backtesting
        min_spread_threshold : float
            Minimum spread % to enter arbitrage (default 0.3%)
        max_spread_threshold : float
            Maximum spread % to exit arbitrage (default 0.1%)
        position_size_pct : float
            Position size as % of capital per trade
        transaction_cost_pct : float
            Total transaction cost including brokerage, taxes, slippage
        square_off_time : str
            Mandatory square-off time in HH:MM format
        symbols : List[str]
            List of symbols to backtest
        from_date : str
            Start date for backtest (YYYY-MM-DD)
        to_date : str
            End date for backtest (YYYY-MM-DD)
        """
        self.access_token = access_token
        self.strategy_type = strategy_type.upper()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.min_spread_threshold = min_spread_threshold / 100
        self.max_spread_threshold = max_spread_threshold / 100
        self.position_size_pct = position_size_pct / 100
        self.transaction_cost_pct = transaction_cost_pct / 100
        self.square_off_time = self.parse_square_off_time(square_off_time)
        self.symbols = symbols or ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
        self.from_date = from_date or (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        self.to_date = to_date or datetime.now().strftime("%Y-%m-%d")

        # IST timezone
        self.ist_tz = pytz.timezone('Asia/Kolkata')

        # Dhan API base URL
        self.base_url = "https://api.dhan.co/v2"

        # Results storage
        self.trades = []
        self.daily_pnl = {}
        self.symbol_data = {}

        print("=" * 100)
        print("DHAN BROKER ARBITRAGE TRADING BACKTESTER")
        print("=" * 100)
        print(f"Strategy Type: {self.strategy_type}")
        print(f"Initial Capital: ₹{self.initial_capital:,.0f}")
        print(f"Min Spread Threshold: {min_spread_threshold}%")
        print(f"Max Spread Threshold: {max_spread_threshold}%")
        print(f"Position Size: {position_size_pct}% of capital")
        print(f"Transaction Cost: {transaction_cost_pct}%")
        print(f"Backtest Period: {self.from_date} to {self.to_date}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print("=" * 100)

    def parse_square_off_time(self, time_str: str) -> time:
        """Parse square-off time"""
        try:
            hour, minute = map(int, time_str.split(':'))
            return time(hour, minute)
        except:
            return time(15, 20)

    def fetch_intraday_data(
            self,
            security_id: str,
            exchange_segment: str,
            instrument: str,
            interval: int = 1,
            from_date: str = None,
            to_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch intraday data from Dhan API

        Parameters:
        -----------
        security_id : str
            Dhan security ID
        exchange_segment : str
            "NSE_EQ", "NSE_FNO", "BSE_EQ", etc.
        instrument : str
            "EQUITY", "FUTURES", "OPTIONS", etc.
        interval : int
            1, 5, 15, 25, or 60 minutes
        """
        url = f"{self.base_url}/charts/intraday"

        headers = {
            "Content-Type": "application/json",
            "access-token": self.access_token
        }

        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": str(interval),
            "oi": True if instrument in ["FUTURES", "OPTIONS"] else False,
            "fromDate": from_date or self.from_date + " 09:15:00",
            "toDate": to_date or self.to_date + " 15:30:00"
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # Convert to DataFrame
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['timestamp'], unit='s'),
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume'],
                'open_interest': data.get('open_interest', [0] * len(data['timestamp']))
            })

            df.set_index('timestamp', inplace=True)

            print(f"Fetched {len(df)} candles for {security_id} ({exchange_segment}) - {interval}min")

            return df

        except Exception as e:
            print(f"✗ Error fetching data for {security_id}: {e}")
            return pd.DataFrame()

    def fetch_daily_data(
            self,
            security_id: str,
            exchange_segment: str,
            instrument: str,
            from_date: str = None,
            to_date: str = None
    ) -> pd.DataFrame:
        """Fetch daily historical data from Dhan API"""
        url = f"{self.base_url}/charts/historical"

        headers = {
            "Content-Type": "application/json",
            "access-token": self.access_token
        }

        payload = {
            "securityId": security_id,
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "expiryCode": 0,
            "oi": True if instrument in ["FUTURES", "OPTIONS"] else False,
            "fromDate": from_date or self.from_date,
            "toDate": to_date or self.to_date
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # Convert to DataFrame
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['timestamp'], unit='s'),
                'open': data['open'],
                'high': data['high'],
                'low': data['low'],
                'close': data['close'],
                'volume': data['volume'],
                'open_interest': data.get('open_interest', [0] * len(data['timestamp']))
            })

            df.set_index('timestamp', inplace=True)

            print(f"✓ Fetched {len(df)} daily candles for {security_id} ({exchange_segment})")

            return df

        except Exception as e:
            print(f"✗ Error fetching daily data for {security_id}: {e}")
            return pd.DataFrame()

    def calculate_spread(self, price1: float, price2: float, spread_type: str = "percentage") -> float:
        """
        Calculate spread between two prices

        Parameters:
        -----------
        price1 : float
            Price of leg 1 (usually spot/near month)
        price2 : float
            Price of leg 2 (usually futures/far month)
        spread_type : str
            "percentage" or "absolute"
        """
        if spread_type == "percentage":
            return ((price2 - price1) / price1) * 100 if price1 > 0 else 0
        else:
            return price2 - price1

    def backtest_spot_futures_arbitrage(self, symbol: str) -> List[ArbitrageTrade]:
        """
        Backtest Cash & Carry Arbitrage (Spot vs Futures)

        Strategy:
        - Buy spot when futures are trading at premium > threshold
        - Sell futures simultaneously
        - Exit when spread converges or at expiry
        """
        print(f"\n{'=' * 100}")
        print(f"BACKTESTING SPOT-FUTURES ARBITRAGE: {symbol}")
        print(f"{'=' * 100}")

        # Note: You'll need to replace these with actual security IDs from Dhan
        # This is a placeholder - actual implementation requires instrument master
        spot_security_id = self.get_security_id(symbol, "NSE_EQ", "EQUITY")
        futures_security_id = self.get_security_id(symbol, "NSE_FNO", "FUTURES")

        if not spot_security_id or not futures_security_id:
            print(f"✗ Could not find security IDs for {symbol}")
            return []

        # Fetch spot data
        spot_data = self.fetch_intraday_data(
            spot_security_id,
            "NSE_EQ",
            "EQUITY",
            interval=1
        )

        # Fetch futures data
        futures_data = self.fetch_intraday_data(
            futures_security_id,
            "NSE_FNO",
            "FUTURES",
            interval=1
        )

        if spot_data.empty or futures_data.empty:
            print(f"✗ No data available for {symbol}")
            return []

        # Align data
        combined_data = pd.DataFrame({
            'spot_close': spot_data['close'],
            'futures_close': futures_data['close'],
            'futures_oi': futures_data['open_interest']
        }).dropna()

        if combined_data.empty:
            print(f"✗ No aligned data for {symbol}")
            return []

        # Calculate spread
        combined_data['spread'] = combined_data.apply(
            lambda row: self.calculate_spread(row['spot_close'], row['futures_close']),
            axis=1
        )

        combined_data['spread_pct'] = combined_data['spread'] / 100  # Convert to decimal

        # Add trading day and square-off indicator
        combined_data['trading_day'] = combined_data.index.date
        combined_data['is_square_off'] = combined_data.index.map(
            lambda x: x.time() >= self.square_off_time
        )

        # Backtest logic
        trades = []
        position = 0  # 0 = no position, 1 = arbitrage position open
        entry_spot = 0
        entry_futures = 0
        entry_spread = 0
        entry_time = None
        entry_day = None

        for i in range(len(combined_data)):
            current_time = combined_data.index[i]
            spot_price = combined_data.iloc[i]['spot_close']
            futures_price = combined_data.iloc[i]['futures_close']
            spread = combined_data.iloc[i]['spread_pct']
            current_day = combined_data.iloc[i]['trading_day']
            is_square_off = combined_data.iloc[i]['is_square_off']

            # Force square-off at 3:20 PM
            if position == 1 and is_square_off:
                exit_reason = "SQUARE_OFF"
                quantity = trades[-1]['quantity']

                # Calculate P&L
                spot_pnl = quantity * (spot_price - entry_spot)
                futures_pnl = quantity * (entry_futures - futures_price)
                gross_pnl = spot_pnl + futures_pnl

                # Transaction costs
                transaction_cost = (
                        quantity * entry_spot * self.transaction_cost_pct +  # Buy spot cost
                        quantity * entry_futures * self.transaction_cost_pct +  # Sell futures cost
                        quantity * spot_price * self.transaction_cost_pct +  # Sell spot cost
                        quantity * futures_price * self.transaction_cost_pct  # Buy futures cost
                )

                net_pnl = gross_pnl - transaction_cost
                pnl_pct = (net_pnl / (quantity * entry_spot)) * 100

                # Update capital
                self.current_capital += net_pnl

                trade = ArbitrageTrade(
                    entry_time=entry_time,
                    exit_time=current_time,
                    symbol=symbol,
                    strategy="SPOT_FUTURES",
                    leg1_entry=entry_spot,
                    leg2_entry=entry_futures,
                    leg1_exit=spot_price,
                    leg2_exit=futures_price,
                    spread_entry=entry_spread,
                    spread_exit=spread,
                    quantity=quantity,
                    pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=exit_reason,
                    transaction_cost=transaction_cost
                )

                trades.append(trade)
                position = 0
                entry_day = None

            # Entry: Spread > threshold
            elif position == 0 and spread >= self.min_spread_threshold and not is_square_off:
                # Calculate position size
                position_value = self.current_capital * self.position_size_pct
                quantity = int(position_value / spot_price)

                if quantity > 0:
                    position = 1
                    entry_spot = spot_price
                    entry_futures = futures_price
                    entry_spread = spread
                    entry_time = current_time
                    entry_day = current_day

                    trades.append({
                        'entry_time': current_time,
                        'symbol': symbol,
                        'strategy': 'SPOT_FUTURES',
                        'spot_entry': entry_spot,
                        'futures_entry': entry_futures,
                        'spread_entry': spread,
                        'quantity': quantity
                    })

                    print(f"✓ ENTER: {current_time.strftime('%Y-%m-%d %H:%M')} | "
                          f"Spot: ₹{entry_spot:.2f} | Futures: ₹{entry_futures:.2f} | "
                          f"Spread: {spread * 100:.3f}% | Qty: {quantity}")

            # Exit: Spread convergence
            elif position == 1 and entry_day == current_day and not is_square_off:
                should_exit = False
                exit_reason = ""

                # Exit when spread narrows below threshold
                if spread <= self.max_spread_threshold:
                    should_exit = True
                    exit_reason = "SPREAD_CONVERGENCE"

                # Stop loss: spread widens excessively (unusual market condition)
                elif spread > entry_spread * 2:
                    should_exit = True
                    exit_reason = "STOP_LOSS_SPREAD_WIDENING"

                if should_exit:
                    quantity = trades[-1]['quantity']

                    # Calculate P&L
                    spot_pnl = quantity * (spot_price - entry_spot)
                    futures_pnl = quantity * (entry_futures - futures_price)
                    gross_pnl = spot_pnl + futures_pnl

                    # Transaction costs
                    transaction_cost = (
                            quantity * entry_spot * self.transaction_cost_pct +
                            quantity * entry_futures * self.transaction_cost_pct +
                            quantity * spot_price * self.transaction_cost_pct +
                            quantity * futures_price * self.transaction_cost_pct
                    )

                    net_pnl = gross_pnl - transaction_cost
                    pnl_pct = (net_pnl / (quantity * entry_spot)) * 100

                    # Update capital
                    self.current_capital += net_pnl

                    trade = ArbitrageTrade(
                        entry_time=entry_time,
                        exit_time=current_time,
                        symbol=symbol,
                        strategy="SPOT_FUTURES",
                        leg1_entry=entry_spot,
                        leg2_entry=entry_futures,
                        leg1_exit=spot_price,
                        leg2_exit=futures_price,
                        spread_entry=entry_spread,
                        spread_exit=spread,
                        quantity=quantity,
                        pnl=net_pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        transaction_cost=transaction_cost
                    )

                    trades.append(trade)

                    print(f"✓ EXIT: {current_time.strftime('%Y-%m-%d %H:%M')} | "
                          f"Spot: ₹{spot_price:.2f} | Futures: ₹{futures_price:.2f} | "
                          f"Spread: {spread * 100:.3f}% | P&L: ₹{net_pnl:,.2f} ({pnl_pct:+.2f}%) | "
                          f"Reason: {exit_reason}")

                    position = 0
                    entry_day = None

        # Filter only completed trades (ArbitrageTrade objects)
        completed_trades = [t for t in trades if isinstance(t, ArbitrageTrade)]

        print(f"\n{'=' * 100}")
        print(f"COMPLETED: {len(completed_trades)} arbitrage trades for {symbol}")
        print(f"{'=' * 100}")

        return completed_trades

    def backtest_exchange_arbitrage(self, symbol: str) -> List[ArbitrageTrade]:
        """
        Backtest Exchange Arbitrage (NSE vs BSE)

        Strategy:
        - Buy on cheaper exchange, sell on expensive exchange
        - Exit when price differential narrows
        """
        print(f"\n{'=' * 100}")
        print(f"BACKTESTING EXCHANGE ARBITRAGE: {symbol}")
        print(f"{'=' * 100}")

        # Get security IDs for both exchanges
        nse_security_id = self.get_security_id(symbol, "NSE_EQ", "EQUITY")
        bse_security_id = self.get_security_id(symbol, "BSE_EQ", "EQUITY")

        if not nse_security_id or not bse_security_id:
            print(f"✗ Could not find security IDs for {symbol} on both exchanges")
            return []

        # Fetch NSE data
        nse_data = self.fetch_intraday_data(
            nse_security_id,
            "NSE_EQ",
            "EQUITY",
            interval=1
        )

        # Fetch BSE data
        bse_data = self.fetch_intraday_data(
            bse_security_id,
            "BSE_EQ",
            "EQUITY",
            interval=1
        )

        if nse_data.empty or bse_data.empty:
            print(f"✗ No data available for {symbol}")
            return []

        # Align data
        combined_data = pd.DataFrame({
            'nse_close': nse_data['close'],
            'bse_close': bse_data['close']
        }).dropna()

        if combined_data.empty:
            print(f"✗ No aligned data for {symbol}")
            return []

        # Calculate spread (NSE - BSE)
        combined_data['spread'] = combined_data['nse_close'] - combined_data['bse_close']
        combined_data['spread_pct'] = (combined_data['spread'] / combined_data['bse_close']) * 100
        combined_data['spread_pct_decimal'] = combined_data['spread_pct'] / 100

        # Add trading day and square-off indicator
        combined_data['trading_day'] = combined_data.index.date
        combined_data['is_square_off'] = combined_data.index.map(
            lambda x: x.time() >= self.square_off_time
        )

        # Backtest logic (similar to spot-futures)
        trades = []
        position = 0
        entry_nse = 0
        entry_bse = 0
        entry_spread = 0
        entry_time = None
        entry_day = None
        buy_exchange = ""  # Which exchange we bought on

        for i in range(len(combined_data)):
            current_time = combined_data.index[i]
            nse_price = combined_data.iloc[i]['nse_close']
            bse_price = combined_data.iloc[i]['bse_close']
            spread = combined_data.iloc[i]['spread_pct_decimal']
            current_day = combined_data.iloc[i]['trading_day']
            is_square_off = combined_data.iloc[i]['is_square_off']

            # Force square-off
            if position == 1 and is_square_off:
                exit_reason = "SQUARE_OFF"
                quantity = trades[-1]['quantity']

                # Calculate P&L based on which exchange we bought/sold
                if buy_exchange == "BSE":
                    gross_pnl = quantity * (nse_price - entry_nse) - quantity * (bse_price - entry_bse)
                else:  # buy_exchange == "NSE"
                    gross_pnl = quantity * (bse_price - entry_bse) - quantity * (nse_price - entry_nse)

                transaction_cost = (
                        quantity * entry_nse * self.transaction_cost_pct +
                        quantity * entry_bse * self.transaction_cost_pct +
                        quantity * nse_price * self.transaction_cost_pct +
                        quantity * bse_price * self.transaction_cost_pct
                )

                net_pnl = gross_pnl - transaction_cost
                pnl_pct = (net_pnl / (quantity * min(entry_nse, entry_bse))) * 100

                self.current_capital += net_pnl

                trade = ArbitrageTrade(
                    entry_time=entry_time,
                    exit_time=current_time,
                    symbol=symbol,
                    strategy="EXCHANGE_ARB",
                    leg1_entry=entry_nse,
                    leg2_entry=entry_bse,
                    leg1_exit=nse_price,
                    leg2_exit=bse_price,
                    spread_entry=entry_spread,
                    spread_exit=spread,
                    quantity=quantity,
                    pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=exit_reason,
                    transaction_cost=transaction_cost
                )

                trades.append(trade)
                position = 0
                entry_day = None

            # Entry: Significant spread exists
            elif position == 0 and abs(spread) >= self.min_spread_threshold and not is_square_off:
                position_value = self.current_capital * self.position_size_pct

                # Determine which exchange to buy/sell
                if spread > 0:  # NSE > BSE, buy BSE, sell NSE
                    buy_exchange = "BSE"
                    quantity = int(position_value / bse_price)
                else:  # BSE > NSE, buy NSE, sell BSE
                    buy_exchange = "NSE"
                    quantity = int(position_value / nse_price)

                if quantity > 0:
                    position = 1
                    entry_nse = nse_price
                    entry_bse = bse_price
                    entry_spread = spread
                    entry_time = current_time
                    entry_day = current_day

                    trades.append({
                        'entry_time': current_time,
                        'symbol': symbol,
                        'strategy': 'EXCHANGE_ARB',
                        'nse_entry': entry_nse,
                        'bse_entry': entry_bse,
                        'spread_entry': spread,
                        'quantity': quantity,
                        'buy_exchange': buy_exchange
                    })

                    print(f"✓ ENTER: {current_time.strftime('%Y-%m-%d %H:%M')} | "
                          f"NSE: ₹{entry_nse:.2f} | BSE: ₹{entry_bse:.2f} | "
                          f"Spread: {spread * 100:.3f}% | Buy: {buy_exchange} | Qty: {quantity}")

            # Exit: Spread convergence
            elif position == 1 and entry_day == current_day and not is_square_off:
                should_exit = False
                exit_reason = ""

                # Exit when spread narrows
                if abs(spread) <= self.max_spread_threshold:
                    should_exit = True
                    exit_reason = "SPREAD_CONVERGENCE"

                # Stop loss: spread reverses direction
                elif (entry_spread > 0 and spread < 0) or (entry_spread < 0 and spread > 0):
                    should_exit = True
                    exit_reason = "SPREAD_REVERSAL"

                if should_exit:
                    quantity = trades[-1]['quantity']

                    if buy_exchange == "BSE":
                        gross_pnl = quantity * (nse_price - entry_nse) - quantity * (bse_price - entry_bse)
                    else:
                        gross_pnl = quantity * (bse_price - entry_bse) - quantity * (nse_price - entry_nse)

                    transaction_cost = (
                            quantity * entry_nse * self.transaction_cost_pct +
                            quantity * entry_bse * self.transaction_cost_pct +
                            quantity * nse_price * self.transaction_cost_pct +
                            quantity * bse_price * self.transaction_cost_pct
                    )

                    net_pnl = gross_pnl - transaction_cost
                    pnl_pct = (net_pnl / (quantity * min(entry_nse, entry_bse))) * 100

                    self.current_capital += net_pnl

                    trade = ArbitrageTrade(
                        entry_time=entry_time,
                        exit_time=current_time,
                        symbol=symbol,
                        strategy="EXCHANGE_ARB",
                        leg1_entry=entry_nse,
                        leg2_entry=entry_bse,
                        leg1_exit=nse_price,
                        leg2_exit=bse_price,
                        spread_entry=entry_spread,
                        spread_exit=spread,
                        quantity=quantity,
                        pnl=net_pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        transaction_cost=transaction_cost
                    )

                    trades.append(trade)

                    print(f"✓ EXIT: {current_time.strftime('%Y-%m-%d %H:%M')} | "
                          f"NSE: ₹{nse_price:.2f} | BSE: ₹{bse_price:.2f} | "
                          f"Spread: {spread * 100:.3f}% | P&L: ₹{net_pnl:,.2f} ({pnl_pct:+.2f}%) | "
                          f"Reason: {exit_reason}")

                    position = 0
                    entry_day = None

        completed_trades = [t for t in trades if isinstance(t, ArbitrageTrade)]

        print(f"\n{'=' * 100}")
        print(f"COMPLETED: {len(completed_trades)} exchange arbitrage trades for {symbol}")
        print(f"{'=' * 100}")

        return completed_trades

    def get_security_id(self, symbol: str, exchange_segment: str, instrument: str) -> str:
        """
        Get Dhan security ID for a symbol

        Note: In production, you should download and parse the Dhan instrument master file
        from: https://images.dhan.co/api-data/api-scrip-master.csv

        This is a placeholder that returns mock IDs for demonstration
        """
        # Mock implementation - replace with actual instrument master lookup
        mock_ids = {
            ("RELIANCE", "NSE_EQ", "EQUITY"): "1333",
            ("RELIANCE", "NSE_FNO", "FUTURES"): "40813",
            ("RELIANCE", "BSE_EQ", "EQUITY"): "500325",
            ("TCS", "NSE_EQ", "EQUITY"): "11536",
            ("TCS", "NSE_FNO", "FUTURES"): "48953",
            ("TCS", "BSE_EQ", "EQUITY"): "532540",
            ("INFY", "NSE_EQ", "EQUITY"): "1594",
            ("INFY", "NSE_FNO", "FUTURES"): "41661",
            ("INFY", "BSE_EQ", "EQUITY"): "500209",
            ("HDFCBANK", "NSE_EQ", "EQUITY"): "1333",
            ("HDFCBANK", "NSE_FNO", "FUTURES"): "40813",
            ("ICICIBANK", "NSE_EQ", "EQUITY"): "4963",
            ("ICICIBANK", "NSE_FNO", "FUTURES"): "40813",
        }

        key = (symbol, exchange_segment, instrument)
        return mock_ids.get(key)

    def run_backtest(self) -> Dict:
        """Run the complete arbitrage backtest"""
        print(f"\n{'=' * 100}")
        print("STARTING ARBITRAGE BACKTEST")
        print(f"{'=' * 100}")

        all_trades = []

        for symbol in self.symbols:
            try:
                if self.strategy_type in ["SPOT_FUTURES", "ALL"]:
                    spot_futures_trades = self.backtest_spot_futures_arbitrage(symbol)
                    all_trades.extend(spot_futures_trades)

                    # Small delay to avoid API rate limits
                    time_module.sleep(1)

                if self.strategy_type in ["EXCHANGE_ARB", "ALL"]:
                    exchange_trades = self.backtest_exchange_arbitrage(symbol)
                    all_trades.extend(exchange_trades)

                    time_module.sleep(1)

            except Exception as e:
                print(f"✗ Error backtesting {symbol}: {e}")
                import traceback
                traceback.print_exc()

        self.trades = all_trades

        # Calculate metrics
        results = self.calculate_metrics()

        return results

    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            print("\n⚠️  No trades executed")
            return {}

        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in self.trades)
        total_return = (total_pnl / self.initial_capital) * 100

        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')

        avg_holding_period = np.mean([(t.exit_time - t.entry_time).total_seconds() / 3600 for t in self.trades])

        total_transaction_costs = sum(t.transaction_cost for t in self.trades)

        # Strategy breakdown
        strategy_breakdown = {}
        for trade in self.trades:
            if trade.strategy not in strategy_breakdown:
                strategy_breakdown[trade.strategy] = {
                    'trades': 0,
                    'pnl': 0,
                    'wins': 0
                }
            strategy_breakdown[trade.strategy]['trades'] += 1
            strategy_breakdown[trade.strategy]['pnl'] += trade.pnl
            if trade.pnl > 0:
                strategy_breakdown[trade.strategy]['wins'] += 1

        metrics = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': max(t.pnl for t in self.trades),
            'worst_trade': min(t.pnl for t in self.trades),
            'profit_factor': profit_factor,
            'avg_holding_hours': avg_holding_period,
            'total_transaction_costs': total_transaction_costs,
            'net_return_after_costs': total_return,
            'strategy_breakdown': strategy_breakdown,
            'final_capital': self.current_capital
        }

        # Print results
        self.print_results(metrics)

        return metrics

    def print_results(self, metrics: Dict):
        """Print backtest results"""
        print(f"\n{'=' * 100}")
        print("ARBITRAGE BACKTEST RESULTS")
        print(f"{'=' * 100}")
        print(f"Initial Capital:       ₹{self.initial_capital:,.0f}")
        print(f"Final Capital:         ₹{metrics['final_capital']:,.0f}")
        print(f"Total P&L:             ₹{metrics['total_pnl']:,.0f}")
        print(f"Total Return:          {metrics['total_return_pct']:+.2f}%")
        print(f"")
        print(f"Total Trades:          {metrics['total_trades']}")
        print(f"Winning Trades:        {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)")
        print(f"Losing Trades:         {metrics['losing_trades']} ({100 - metrics['win_rate']:.1f}%)")
        print(f"")
        print(f"Average Win:           ₹{metrics['avg_win']:,.0f}")
        print(f"Average Loss:          ₹{metrics['avg_loss']:,.0f}")
        print(f"Best Trade:            ₹{metrics['best_trade']:,.0f}")
        print(f"Worst Trade:           ₹{metrics['worst_trade']:,.0f}")
        print(f"Profit Factor:         {metrics['profit_factor']:.2f}")
        print(f"")
        print(f"Avg Holding Period:    {metrics['avg_holding_hours']:.2f} hours")
        print(f"Total Transaction Cost: ₹{metrics['total_transaction_costs']:,.0f}")

        print(f"\n{'=' * 100}")
        print("STRATEGY BREAKDOWN")
        print(f"{'=' * 100}")

        for strategy, stats in metrics['strategy_breakdown'].items():
            win_rate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
            print(f"{strategy:20} | Trades: {stats['trades']:3d} | "
                  f"P&L: ₹{stats['pnl']:10,.0f} | Win Rate: {win_rate:5.1f}%")

        print(f"{'=' * 100}")

    def export_results(self, filename_prefix: str = "dhan_arbitrage"):
        """Export results to CSV"""
        if not self.trades:
            print("No trades to export")
            return

        # Convert trades to DataFrame
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'Entry Time': trade.entry_time,
                'Exit Time': trade.exit_time,
                'Symbol': trade.symbol,
                'Strategy': trade.strategy,
                'Leg 1 Entry': trade.leg1_entry,
                'Leg 2 Entry': trade.leg2_entry,
                'Leg 1 Exit': trade.leg1_exit,
                'Leg 2 Exit': trade.leg2_exit,
                'Spread Entry (%)': trade.spread_entry * 100,
                'Spread Exit (%)': trade.spread_exit * 100,
                'Quantity': trade.quantity,
                'P&L': trade.pnl,
                'P&L (%)': trade.pnl_pct,
                'Exit Reason': trade.exit_reason,
                'Transaction Cost': trade.transaction_cost,
                'Holding Hours': (trade.exit_time - trade.entry_time).total_seconds() / 3600
            })

        trades_df = pd.DataFrame(trades_data)

        # Export
        filename = f"{filename_prefix}_{self.strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(filename, index=False)

        print(f"\n✓ Results exported to: {filename}")

        return filename


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def demo_backtest():
    """
    Demo backtest with mock data

    NOTE: Replace 'YOUR_DHAN_ACCESS_TOKEN' with your actual Dhan API token
    Get your token from: https://dhanhq.co/docs/v2/authentication/
    """

    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║              DHAN ARBITRAGE TRADING BACKTESTER - DEMO MODE                ║
    ║                                                                           ║
    ║  This is a demonstration of the Dhan Arbitrage Backtester framework.     ║
    ║  To use with real data, you need:                                        ║
    ║                                                                           ║
    ║  1. Dhan Trading Account                                                 ║
    ║  2. API Access Token (JWT) from Dhan                                     ║
    ║  3. Download instrument master: https://images.dhan.co/api-data/...      ║
    ║  4. Update get_security_id() method with actual instrument IDs           ║
    ║                                                                           ║
    ║  API Documentation: https://dhanhq.co/docs/v2/historical-data/           ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    # Initialize backtester
    backtester = DhanArbitrageBacktester(
        access_token="YOUR_DHAN_ACCESS_TOKEN",  # Replace with actual token
        strategy_type="SPOT_FUTURES",  # Options: SPOT_FUTURES, EXCHANGE_ARB, ALL
        initial_capital=1000000,
        min_spread_threshold=0.3,  # 0.3% minimum spread
        max_spread_threshold=0.1,  # 0.1% exit threshold
        position_size_pct=20,
        transaction_cost_pct=0.1,
        square_off_time="15:20",
        symbols=["RELIANCE", "TCS"],
        from_date="2024-01-01",
        to_date="2024-03-31"
    )

    # Run backtest
    results = backtester.run_backtest()

    # Export results
    if results:
        backtester.export_results("dhan_arbitrage_backtest")


if __name__ == "__main__":
    """
    Main execution

    IMPORTANT SETUP STEPS:
    ======================

    1. Get Dhan API Access Token:
       - Login to Dhan: https://dhanhq.co
       - Go to API section
       - Generate access token (JWT)
       - Replace 'YOUR_DHAN_ACCESS_TOKEN' below

    2. Download Instrument Master:
       - URL: https://images.dhan.co/api-data/api-scrip-master.csv
       - Parse this CSV to get correct security IDs
       - Update get_security_id() method with mapping

    3. Configure Strategy:
       - SPOT_FUTURES: Cash & Carry arbitrage
       - EXCHANGE_ARB: NSE vs BSE arbitrage
       - CALENDAR_SPREAD: Near vs Far month futures (TODO)
       - ALL: Run all strategies

    4. Set Parameters:
       - min_spread_threshold: Minimum spread to enter (e.g., 0.3%)
       - max_spread_threshold: Maximum spread to exit (e.g., 0.1%)
       - transaction_cost_pct: Include all costs (brokerage, taxes, slippage)

    5. Run Backtest:
       - Choose date range
       - Select symbols
       - Execute and analyze results
    """

    # Example 1: Spot-Futures Arbitrage
    print("\n" + "=" * 100)
    print("EXAMPLE 1: SPOT-FUTURES ARBITRAGE (CASH & CARRY)")
    print("=" * 100)

    backtester_sf = DhanArbitrageBacktester(
        access_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzY1MTI5NTc0LCJpYXQiOjE3NjUwNDMxNzQsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMDAwMDM5NTk0In0.v89tmTeCa6H4ZysegHRKqug_8r_ltzZwdq_dig-dlDnii64Bu-LMhEp6b18hGZ1NfXr-a12gHX234zer5nD99w",
        strategy_type="SPOT_FUTURES",
        initial_capital=1000000,
        min_spread_threshold=0.5,  # Enter when futures premium > 0.5%
        max_spread_threshold=0.2,  # Exit when premium < 0.2%
        position_size_pct=25,
        transaction_cost_pct=0.15,
        symbols=["RELIANCE", "TCS", "INFY"]
    )

    results_sf = backtester_sf.run_backtest()
    backtester_sf.export_results()

    # Example 2: Exchange Arbitrage
    print("\n" + "=" * 100)
    print("EXAMPLE 2: EXCHANGE ARBITRAGE (NSE vs BSE)")
    print("=" * 100)

    backtester_ex = DhanArbitrageBacktester(
        access_token="YOUR_DHAN_ACCESS_TOKEN",
        strategy_type="EXCHANGE_ARB",
        initial_capital=500000,
        min_spread_threshold=0.3,  # Enter when price diff > 0.3%
        max_spread_threshold=0.1,  # Exit when price diff < 0.1%
        position_size_pct=30,
        transaction_cost_pct=0.2,  # Higher for exchange arb
        symbols=["RELIANCE", "TCS"]
    )

    # results_ex = backtester_ex.run_backtest()
    # backtester_ex.export_results()

    # Example 3: Run All Strategies
    print("\n" + "=" * 100)
    print("EXAMPLE 3: ALL ARBITRAGE STRATEGIES")
    print("=" * 100)

    backtester_all = DhanArbitrageBacktester(
        access_token="YOUR_DHAN_ACCESS_TOKEN",
        strategy_type="ALL",
        initial_capital=2000000,
        min_spread_threshold=0.4,
        max_spread_threshold=0.15,
        position_size_pct=15,
        transaction_cost_pct=0.15,
        symbols=["RELIANCE", "TCS", "INFY", "HDFCBANK"]
    )

    # results_all = backtester_all.run_backtest()
    # backtester_all.export_results()

    print("\n" + "=" * 100)
    print("SETUP INSTRUCTIONS")
    print("=" * 100)
    print("""
    To run this backtester with real data:

    1. Get your Dhan API access token:
       https://dhanhq.co/docs/v2/authentication/

    2. Download instrument master CSV:
       https://images.dhan.co/api-data/api-scrip-master.csv

    3. Update get_security_id() method with actual security IDs

    4. Replace 'YOUR_DHAN_ACCESS_TOKEN' with your actual token

    5. Uncomment the .run_backtest() lines above

    6. Run: python DhanArbitrageBacktester.py
    """)
    print("=" * 100)