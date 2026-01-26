"""
Momentum Score Calculator for Stock Analysis

This module provides comprehensive momentum scoring functionality for stocks,
calculating various momentum metrics over n number of days to support
trading strategy decisions.

Momentum Score Components:
1. Price Rate of Change (ROC) - Price percentage change over period
2. Relative Strength (RS) - Performance vs benchmark/average
3. Volume Momentum - Volume trend strength
4. Multi-timeframe Momentum - Combined scores across different periods
5. Momentum Acceleration - Rate of change of momentum
6. Trend Consistency - Directional consistency of price movement

Author: Claude
Date: 2026-01-26
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import warnings
from typing import Optional, Dict, List, Tuple, Union

warnings.filterwarnings('ignore')


class MomentumScoreCalculator:
    """
    A comprehensive momentum score calculator for stock analysis.

    This class calculates various momentum metrics and combines them into
    a single composite momentum score that can be used for stock screening
    and trading decisions.

    Attributes:
        lookback_days (int): Default number of days for momentum calculation
        short_period (int): Short-term momentum period (default: 5 days)
        medium_period (int): Medium-term momentum period (default: 10 days)
        long_period (int): Long-term momentum period (default: 20 days)
    """

    def __init__(self,
                 lookback_days: int = 20,
                 short_period: int = 5,
                 medium_period: int = 10,
                 long_period: int = 20,
                 volume_period: int = 20,
                 smoothing_period: int = 3):
        """
        Initialize the MomentumScoreCalculator.

        Args:
            lookback_days: Default lookback period for calculations
            short_period: Short-term period for momentum (days)
            medium_period: Medium-term period for momentum (days)
            long_period: Long-term period for momentum (days)
            volume_period: Period for volume analysis
            smoothing_period: Period for smoothing momentum values
        """
        self.lookback_days = lookback_days
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period
        self.volume_period = volume_period
        self.smoothing_period = smoothing_period

        # Score weights for composite calculation
        self.weights = {
            'roc': 0.25,           # Rate of Change weight
            'rsi_momentum': 0.15,  # RSI-based momentum weight
            'volume_momentum': 0.15,  # Volume momentum weight
            'trend_consistency': 0.15,  # Trend consistency weight
            'acceleration': 0.15,  # Momentum acceleration weight
            'multi_timeframe': 0.15  # Multi-timeframe score weight
        }

    def calculate_price_roc(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Price Rate of Change (ROC).

        ROC measures the percentage change in price over a specified period.
        Formula: ROC = ((Current Price - Price N periods ago) / Price N periods ago) * 100

        Args:
            df: DataFrame with 'close' column
            period: Lookback period (uses default if None)

        Returns:
            pd.Series: ROC values as percentages
        """
        period = period or self.lookback_days
        roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        return roc

    def calculate_momentum(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate basic momentum (price difference).

        Momentum = Current Price - Price N periods ago

        Args:
            df: DataFrame with 'close' column
            period: Lookback period

        Returns:
            pd.Series: Momentum values
        """
        period = period or self.lookback_days
        return df['close'] - df['close'].shift(period)

    def calculate_momentum_percentage(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate momentum as a percentage of price.

        Args:
            df: DataFrame with 'close' column
            period: Lookback period

        Returns:
            pd.Series: Momentum percentage values
        """
        period = period or self.lookback_days
        momentum = self.calculate_momentum(df, period)
        return (momentum / df['close'].shift(period)) * 100

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI is a momentum oscillator measuring speed and magnitude of price changes.
        Formula: RSI = 100 - (100 / (1 + RS))
        Where RS = Average Gain / Average Loss

        Args:
            df: DataFrame with 'close' column
            period: RSI calculation period

        Returns:
            pd.Series: RSI values (0-100)
        """
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_rsi_momentum(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate RSI-based momentum score.

        Converts RSI to a momentum score:
        - RSI > 70: Strong bullish momentum (positive score)
        - RSI < 30: Strong bearish momentum (negative score)
        - RSI 50: Neutral (score = 0)

        Args:
            df: DataFrame with 'close' column
            period: RSI period

        Returns:
            pd.Series: RSI momentum score (-100 to +100)
        """
        rsi = self.calculate_rsi(df, period)
        # Convert RSI (0-100) to momentum score (-100 to +100)
        # RSI 50 = 0, RSI 100 = +100, RSI 0 = -100
        rsi_momentum = (rsi - 50) * 2
        return rsi_momentum

    def calculate_volume_momentum(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate volume momentum score.

        Measures the trend in volume relative to its average.
        Positive values indicate increasing volume (confirming moves).

        Args:
            df: DataFrame with 'volume' column
            period: Lookback period for volume analysis

        Returns:
            pd.Series: Volume momentum score
        """
        period = period or self.volume_period

        if 'volume' not in df.columns:
            return pd.Series(0, index=df.index)

        # Calculate average volume
        avg_volume = df['volume'].rolling(window=period).mean()

        # Volume ratio (current vs average)
        volume_ratio = df['volume'] / avg_volume

        # Calculate volume trend (is volume increasing?)
        volume_roc = ((df['volume'] - df['volume'].shift(period)) /
                     df['volume'].shift(period)) * 100

        # Combine: high volume with increasing trend = strong momentum
        # Normalize to -100 to +100 scale
        volume_momentum = np.clip(volume_roc, -100, 100)

        # Multiply by price direction to get directional volume momentum
        price_direction = np.sign(df['close'] - df['close'].shift(1))
        directional_volume_momentum = volume_momentum * price_direction.rolling(
            window=self.smoothing_period).mean()

        return directional_volume_momentum

    def calculate_trend_consistency(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate trend consistency score.

        Measures how consistently price has moved in one direction.
        Score of 100 = all periods positive
        Score of -100 = all periods negative

        Args:
            df: DataFrame with 'close' column
            period: Lookback period

        Returns:
            pd.Series: Trend consistency score (-100 to +100)
        """
        period = period or self.lookback_days

        # Calculate daily returns
        daily_returns = df['close'].pct_change()

        # Calculate percentage of positive days
        positive_days = (daily_returns > 0).astype(int)
        consistency = positive_days.rolling(window=period).mean()

        # Convert to -100 to +100 scale (0.5 = 0, 1.0 = +100, 0.0 = -100)
        trend_consistency = (consistency - 0.5) * 200

        return trend_consistency

    def calculate_momentum_acceleration(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate momentum acceleration (rate of change of momentum).

        Positive acceleration = momentum is increasing
        Negative acceleration = momentum is decreasing

        Args:
            df: DataFrame with 'close' column
            period: Period for momentum calculation

        Returns:
            pd.Series: Momentum acceleration score
        """
        period = period or self.short_period

        # Calculate momentum
        momentum = self.calculate_momentum_percentage(df, period)

        # Calculate acceleration (change in momentum)
        acceleration = momentum - momentum.shift(period)

        # Normalize to -100 to +100 scale
        acceleration_normalized = np.clip(acceleration * 10, -100, 100)

        return acceleration_normalized

    def calculate_multi_timeframe_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate multi-timeframe momentum score.

        Combines short, medium, and long-term momentum for a comprehensive view.
        Stronger signals when all timeframes align.

        Args:
            df: DataFrame with 'close' column

        Returns:
            pd.Series: Multi-timeframe momentum score
        """
        # Calculate ROC for different timeframes
        short_roc = self.calculate_price_roc(df, self.short_period)
        medium_roc = self.calculate_price_roc(df, self.medium_period)
        long_roc = self.calculate_price_roc(df, self.long_period)

        # Normalize each to -100 to +100 scale
        short_score = np.clip(short_roc * 5, -100, 100)  # More sensitive
        medium_score = np.clip(medium_roc * 3, -100, 100)
        long_score = np.clip(long_roc * 2, -100, 100)  # Less sensitive

        # Weighted average (more weight to shorter timeframes for responsiveness)
        multi_tf_score = (short_score * 0.4 + medium_score * 0.35 + long_score * 0.25)

        # Boost score when all timeframes agree
        alignment_multiplier = 1.0
        if short_score.notna().any():
            same_direction = (
                (np.sign(short_score) == np.sign(medium_score)) &
                (np.sign(medium_score) == np.sign(long_score))
            )
            alignment_multiplier = np.where(same_direction, 1.2, 1.0)

        multi_tf_score = multi_tf_score * alignment_multiplier

        return np.clip(multi_tf_score, -100, 100)

    def calculate_relative_strength(self,
                                    df: pd.DataFrame,
                                    benchmark_df: pd.DataFrame = None,
                                    period: int = None) -> pd.Series:
        """
        Calculate relative strength vs benchmark.

        Measures how a stock performs relative to a benchmark or the market.

        Args:
            df: Stock DataFrame with 'close' column
            benchmark_df: Benchmark DataFrame with 'close' column (optional)
            period: Lookback period

        Returns:
            pd.Series: Relative strength score
        """
        period = period or self.lookback_days

        # Stock performance
        stock_roc = self.calculate_price_roc(df, period)

        if benchmark_df is not None and 'close' in benchmark_df.columns:
            # Benchmark performance
            bench_roc = self.calculate_price_roc(benchmark_df, period)

            # Relative strength = stock ROC - benchmark ROC
            relative_strength = stock_roc - bench_roc
        else:
            # Without benchmark, just use absolute momentum
            relative_strength = stock_roc

        return relative_strength

    def calculate_composite_momentum_score(self,
                                          df: pd.DataFrame,
                                          benchmark_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate the composite momentum score combining all factors.

        This is the main method that combines all momentum metrics into
        a single comprehensive score.

        Args:
            df: DataFrame with OHLCV data
            benchmark_df: Optional benchmark for relative strength

        Returns:
            pd.DataFrame: Original df with all momentum scores added
        """
        result_df = df.copy()

        # Calculate individual components
        result_df['mom_roc'] = self.calculate_price_roc(df, self.lookback_days)
        result_df['mom_roc_short'] = self.calculate_price_roc(df, self.short_period)
        result_df['mom_roc_medium'] = self.calculate_price_roc(df, self.medium_period)
        result_df['mom_roc_long'] = self.calculate_price_roc(df, self.long_period)

        result_df['mom_rsi'] = self.calculate_rsi(df)
        result_df['mom_rsi_score'] = self.calculate_rsi_momentum(df)

        result_df['mom_volume'] = self.calculate_volume_momentum(df)
        result_df['mom_trend_consistency'] = self.calculate_trend_consistency(df)
        result_df['mom_acceleration'] = self.calculate_momentum_acceleration(df)
        result_df['mom_multi_tf'] = self.calculate_multi_timeframe_score(df)

        # Calculate relative strength if benchmark provided
        result_df['mom_relative_strength'] = self.calculate_relative_strength(
            df, benchmark_df)

        # Normalize ROC to score (-100 to +100)
        roc_score = np.clip(result_df['mom_roc'] * 3, -100, 100)

        # Calculate composite score using weights
        result_df['momentum_score'] = (
            roc_score * self.weights['roc'] +
            result_df['mom_rsi_score'] * self.weights['rsi_momentum'] +
            result_df['mom_volume'].fillna(0) * self.weights['volume_momentum'] +
            result_df['mom_trend_consistency'] * self.weights['trend_consistency'] +
            result_df['mom_acceleration'] * self.weights['acceleration'] +
            result_df['mom_multi_tf'] * self.weights['multi_timeframe']
        )

        # Smooth the final score
        result_df['momentum_score_smoothed'] = result_df['momentum_score'].rolling(
            window=self.smoothing_period, min_periods=1).mean()

        # Categorize momentum
        result_df['momentum_category'] = pd.cut(
            result_df['momentum_score'],
            bins=[-float('inf'), -50, -20, 20, 50, float('inf')],
            labels=['Strong Bearish', 'Bearish', 'Neutral', 'Bullish', 'Strong Bullish']
        )

        return result_df

    def calculate_momentum_score_for_n_days(self,
                                            df: pd.DataFrame,
                                            n_days: int,
                                            benchmark_df: pd.DataFrame = None) -> Dict:
        """
        Calculate momentum score summary for the last n days.

        This method provides a summary of momentum metrics for a specified
        number of days, useful for stock screening and ranking.

        Args:
            df: DataFrame with OHLCV data
            n_days: Number of days to analyze
            benchmark_df: Optional benchmark for relative strength

        Returns:
            Dict: Summary of momentum metrics
        """
        # Ensure we have enough data
        if len(df) < n_days:
            return {
                'error': f'Insufficient data. Need {n_days} days, have {len(df)}',
                'momentum_score': None
            }

        # Get last n days of data
        df_subset = df.tail(n_days).copy()

        # Calculate all momentum metrics
        result_df = self.calculate_composite_momentum_score(df_subset, benchmark_df)

        # Get the latest values
        latest = result_df.iloc[-1]

        # Calculate summary statistics
        summary = {
            # Current scores
            'momentum_score': round(latest.get('momentum_score', 0), 2),
            'momentum_score_smoothed': round(latest.get('momentum_score_smoothed', 0), 2),
            'momentum_category': latest.get('momentum_category', 'Unknown'),

            # Individual components (latest)
            'roc_score': round(latest.get('mom_roc', 0), 2),
            'roc_short': round(latest.get('mom_roc_short', 0), 2),
            'roc_medium': round(latest.get('mom_roc_medium', 0), 2),
            'roc_long': round(latest.get('mom_roc_long', 0), 2),
            'rsi': round(latest.get('mom_rsi', 50), 2),
            'rsi_score': round(latest.get('mom_rsi_score', 0), 2),
            'volume_momentum': round(latest.get('mom_volume', 0), 2),
            'trend_consistency': round(latest.get('mom_trend_consistency', 0), 2),
            'acceleration': round(latest.get('mom_acceleration', 0), 2),
            'multi_timeframe_score': round(latest.get('mom_multi_tf', 0), 2),
            'relative_strength': round(latest.get('mom_relative_strength', 0), 2),

            # Period statistics
            'n_days': n_days,
            'avg_momentum_score': round(result_df['momentum_score'].mean(), 2),
            'max_momentum_score': round(result_df['momentum_score'].max(), 2),
            'min_momentum_score': round(result_df['momentum_score'].min(), 2),

            # Price statistics
            'price_start': round(df_subset['close'].iloc[0], 2),
            'price_end': round(df_subset['close'].iloc[-1], 2),
            'price_change_pct': round(
                ((df_subset['close'].iloc[-1] - df_subset['close'].iloc[0]) /
                 df_subset['close'].iloc[0]) * 100, 2),

            # Trend analysis
            'bullish_days': int((result_df['momentum_score'] > 20).sum()),
            'bearish_days': int((result_df['momentum_score'] < -20).sum()),
            'neutral_days': int(((result_df['momentum_score'] >= -20) &
                                (result_df['momentum_score'] <= 20)).sum()),
        }

        # Calculate momentum trend (is momentum increasing or decreasing?)
        if len(result_df) >= 5:
            recent_momentum = result_df['momentum_score'].tail(5).mean()
            older_momentum = result_df['momentum_score'].head(5).mean()
            summary['momentum_trend'] = 'Increasing' if recent_momentum > older_momentum else 'Decreasing'
        else:
            summary['momentum_trend'] = 'Unknown'

        return summary

    def rank_stocks_by_momentum(self,
                                stocks_data: Dict[str, pd.DataFrame],
                                n_days: int = 20,
                                benchmark_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Rank multiple stocks by their momentum scores.

        Args:
            stocks_data: Dictionary of {symbol: DataFrame}
            n_days: Number of days for momentum calculation
            benchmark_df: Optional benchmark for relative strength

        Returns:
            pd.DataFrame: Ranked stocks with momentum metrics
        """
        rankings = []

        for symbol, df in stocks_data.items():
            try:
                score_summary = self.calculate_momentum_score_for_n_days(
                    df, n_days, benchmark_df)

                if 'error' not in score_summary:
                    score_summary['symbol'] = symbol
                    rankings.append(score_summary)
            except Exception as e:
                print(f"Error calculating momentum for {symbol}: {e}")
                continue

        if not rankings:
            return pd.DataFrame()

        # Create DataFrame and sort by momentum score
        rankings_df = pd.DataFrame(rankings)
        rankings_df = rankings_df.sort_values('momentum_score', ascending=False)
        rankings_df['rank'] = range(1, len(rankings_df) + 1)

        # Reorder columns
        cols = ['rank', 'symbol', 'momentum_score', 'momentum_category',
                'momentum_trend', 'rsi', 'price_change_pct', 'trend_consistency',
                'relative_strength', 'volume_momentum']
        available_cols = [c for c in cols if c in rankings_df.columns]
        other_cols = [c for c in rankings_df.columns if c not in cols]
        rankings_df = rankings_df[available_cols + other_cols]

        return rankings_df


def calculate_momentum_score(df: pd.DataFrame,
                            n_days: int = 20,
                            short_period: int = 5,
                            medium_period: int = 10,
                            long_period: int = 20) -> Dict:
    """
    Convenience function to calculate momentum score for a stock.

    This is a simple wrapper around MomentumScoreCalculator for quick usage.

    Args:
        df: DataFrame with OHLCV data (must have 'close' column)
        n_days: Number of days for analysis
        short_period: Short-term momentum period
        medium_period: Medium-term momentum period
        long_period: Long-term momentum period

    Returns:
        Dict: Momentum score summary

    Example:
        >>> df = pd.DataFrame({'close': [100, 102, 105, 103, 108, ...]})
        >>> score = calculate_momentum_score(df, n_days=20)
        >>> print(f"Momentum Score: {score['momentum_score']}")
    """
    calculator = MomentumScoreCalculator(
        lookback_days=n_days,
        short_period=short_period,
        medium_period=medium_period,
        long_period=long_period
    )
    return calculator.calculate_momentum_score_for_n_days(df, n_days)


def add_momentum_score_to_df(df: pd.DataFrame,
                             lookback_days: int = 20,
                             short_period: int = 5,
                             medium_period: int = 10,
                             long_period: int = 20) -> pd.DataFrame:
    """
    Add momentum score columns to a DataFrame.

    This function adds all momentum-related columns to the input DataFrame
    for use in backtesting strategies.

    Args:
        df: DataFrame with OHLCV data
        lookback_days: Main lookback period
        short_period: Short-term period
        medium_period: Medium-term period
        long_period: Long-term period

    Returns:
        pd.DataFrame: DataFrame with momentum columns added

    Added columns:
        - momentum_score: Composite momentum score (-100 to +100)
        - momentum_score_smoothed: Smoothed version of momentum score
        - momentum_category: Categorical label (Strong Bearish to Strong Bullish)
        - mom_roc: Rate of Change (percentage)
        - mom_rsi: RSI value (0-100)
        - mom_rsi_score: RSI converted to momentum score
        - mom_volume: Volume momentum score
        - mom_trend_consistency: Trend consistency score
        - mom_acceleration: Momentum acceleration
        - mom_multi_tf: Multi-timeframe score
    """
    calculator = MomentumScoreCalculator(
        lookback_days=lookback_days,
        short_period=short_period,
        medium_period=medium_period,
        long_period=long_period
    )
    return calculator.calculate_composite_momentum_score(df)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("MOMENTUM SCORE CALCULATOR - TEST")
    print("=" * 80)

    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')

    # Simulate price data with an upward trend
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 100)  # Mean positive return
    prices = base_price * np.cumprod(1 + returns)
    volumes = np.random.randint(100000, 1000000, 100)

    sample_df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
        'high': prices * (1 + np.random.uniform(0, 0.02, 100)),
        'low': prices * (1 - np.random.uniform(0, 0.02, 100)),
        'close': prices,
        'volume': volumes
    }, index=dates)

    print("\nSample Data (last 5 rows):")
    print(sample_df.tail())

    # Test 1: Calculate momentum score for n days
    print("\n" + "=" * 80)
    print("TEST 1: Calculate Momentum Score for 20 days")
    print("=" * 80)

    score = calculate_momentum_score(sample_df, n_days=20)

    print(f"\nMomentum Score: {score['momentum_score']}")
    print(f"Momentum Category: {score['momentum_category']}")
    print(f"Momentum Trend: {score['momentum_trend']}")
    print(f"\nComponent Scores:")
    print(f"  ROC (20-day): {score['roc_score']}%")
    print(f"  RSI: {score['rsi']}")
    print(f"  RSI Score: {score['rsi_score']}")
    print(f"  Volume Momentum: {score['volume_momentum']}")
    print(f"  Trend Consistency: {score['trend_consistency']}")
    print(f"  Acceleration: {score['acceleration']}")
    print(f"  Multi-timeframe: {score['multi_timeframe_score']}")
    print(f"\nPrice Change over {score['n_days']} days: {score['price_change_pct']}%")
    print(f"Bullish Days: {score['bullish_days']}, Bearish Days: {score['bearish_days']}, Neutral Days: {score['neutral_days']}")

    # Test 2: Add momentum columns to DataFrame
    print("\n" + "=" * 80)
    print("TEST 2: Add Momentum Score Columns to DataFrame")
    print("=" * 80)

    df_with_momentum = add_momentum_score_to_df(sample_df, lookback_days=20)

    print("\nDataFrame with momentum columns (last 5 rows):")
    momentum_cols = ['close', 'momentum_score', 'momentum_category', 'mom_roc', 'mom_rsi']
    print(df_with_momentum[momentum_cols].tail())

    # Test 3: Different periods
    print("\n" + "=" * 80)
    print("TEST 3: Momentum Scores for Different Periods")
    print("=" * 80)

    for n_days in [5, 10, 20, 30]:
        if len(sample_df) >= n_days:
            score = calculate_momentum_score(sample_df, n_days=n_days)
            print(f"  {n_days}-day Momentum Score: {score['momentum_score']:+.2f} ({score['momentum_category']})")

    # Test 4: Rank multiple stocks
    print("\n" + "=" * 80)
    print("TEST 4: Rank Multiple Stocks by Momentum")
    print("=" * 80)

    # Create additional sample stocks
    stocks_data = {}
    for symbol in ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D', 'STOCK_E']:
        np.random.seed(hash(symbol) % 1000)
        returns = np.random.normal(np.random.uniform(-0.002, 0.003), 0.02, 100)
        prices = base_price * np.cumprod(1 + returns)
        stocks_data[symbol] = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)

    calculator = MomentumScoreCalculator()
    rankings = calculator.rank_stocks_by_momentum(stocks_data, n_days=20)

    print("\nStock Rankings by Momentum:")
    print(rankings[['rank', 'symbol', 'momentum_score', 'momentum_category',
                   'momentum_trend', 'price_change_pct']].to_string(index=False))

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
