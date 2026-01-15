# Supertrend Trailing Stop Strategy

## Overview

This strategy uses the **Supertrend indicator line itself as a dynamic trailing stop loss**. Unlike traditional ATR-based trailing stops, this approach leverages the Supertrend's built-in trend-following characteristics to manage positions.

## Key Concept

The Supertrend indicator calculates a dynamic support/resistance line based on:
- Average True Range (ATR) for volatility measurement
- High-Low average for price positioning
- Trend direction detection

When in a bullish trend, the Supertrend line acts as a rising support level that:
- **Moves up** as the trend strengthens (but never down)
- **Automatically adjusts** to market volatility via ATR
- **Locks in profits** while giving room for trend continuation
- **Provides clear exit signals** when price crosses below the line

## Strategy Logic

### Entry Conditions (ALL must be true)
1. **Supertrend turns bullish** - Price crosses above the Supertrend line (trend reversal)
2. **Valid indicators** - Supertrend and ATR values are calculated
3. **Time filter** - Entry before last entry time (default: 2:30 PM)
4. **Trade limit** - Daily trade limit not exceeded (default: 5 trades/day)

### Exit Conditions (ANY triggers exit)
1. **Trailing Stop Hit** - Price closes below the Supertrend line
2. **Square-off Time** - Mandatory end-of-day exit (default: 3:20 PM)

## Advantages of Using Supertrend as Trailing Stop

### 1. Volatility-Adjusted
The Supertrend uses ATR in its calculation, so it automatically widens in volatile markets and tightens in calm markets. No need for manual adjustment.

### 2. Trend-Following
The Supertrend line only moves up during uptrends, never down. This means:
- Profits are progressively locked in
- You stay in winning trades longer
- Quick exit when trend reverses

### 3. Simplicity
One indicator serves two purposes:
- Entry signal (trend reversal)
- Exit signal (trailing stop)

### 4. Clear Signals
No subjective decisions - if price closes below Supertrend, exit immediately. Black and white rule.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `supertrend_period` | 10 | ATR period for Supertrend calculation |
| `supertrend_multiplier` | 3.0 | Multiplier for ATR bands (higher = wider stops) |
| `initial_capital` | 100,000 | Starting capital per symbol (₹) |
| `square_off_time` | 15:20 | Mandatory exit time (3:20 PM IST) |
| `last_entry_time` | 14:30 | No new entries after this time (2:30 PM IST) |
| `max_trades_per_day` | 5 | Maximum trades per symbol per day |
| `tick_interval` | 30s | Data resampling interval |

## How It Differs from SupertrendHeikinAshi Strategy

| Aspect | SupertrendTrailingStop | SupertrendHeikinAshi |
|--------|------------------------|----------------------|
| **Entry** | Supertrend buy signal only | Supertrend + Heikin Ashi confirmation |
| **Exit** | Price below Supertrend line | Multiple: Supertrend, HA bearish, ATR trailing stop |
| **Trailing Stop** | Supertrend line itself | Separate ATR-based calculation |
| **Complexity** | Simple, pure trend-following | More complex, dual confirmation |
| **Filters** | Minimal (time-based only) | Multiple (HA candle significance, etc.) |
| **Trade Frequency** | Potentially higher | Lower (more filters) |

## Visual Example

```
Price Movement:

    ┌─→ Exit: Price crosses below Supertrend
    │
  120 ──┐     ╱╲
        │    ╱  ╲
  115   │   ╱    ╲
        │  ╱      ╲
  110   └─╱        ╲___
       ╱              ╲
  105 ╱                ╲
    Entry: Supertrend     Supertrend line
    turns bullish         (rising support)

Timeline: ────────────────────────────>

The Supertrend line (shown as a rising line) acts as:
1. Entry trigger when price crosses above
2. Dynamic trailing stop that moves up with price
3. Exit trigger when price breaks below
```

## Usage

### Basic Usage
```python
from SupertrendTrailingStop import SupertrendTrailingStopBacktester

backtester = SupertrendTrailingStopBacktester(
    data_folder="data/symbolupdate",
    symbols=None,  # Auto-detect from databases
    supertrend_period=10,
    supertrend_multiplier=3.0,
    tick_interval='30s'
)

backtester.run_backtest()
```

### Custom Symbols
```python
backtester = SupertrendTrailingStopBacktester(
    data_folder="data/symbolupdate",
    symbols=['NSE:RELIANCE-EQ', 'NSE:TCS-EQ', 'NSE:INFY-EQ'],
    supertrend_period=10,
    supertrend_multiplier=3.0
)

backtester.run_backtest()
```

### Adjust Parameters for Different Trading Styles

#### Aggressive (Tighter Stops)
```python
backtester = SupertrendTrailingStopBacktester(
    supertrend_period=7,       # Shorter period = more responsive
    supertrend_multiplier=2.0,  # Lower multiplier = tighter stops
    max_trades_per_day=10       # More trades allowed
)
```

#### Conservative (Wider Stops)
```python
backtester = SupertrendTrailingStopBacktester(
    supertrend_period=14,       # Longer period = smoother
    supertrend_multiplier=4.0,  # Higher multiplier = wider stops
    max_trades_per_day=3        # Fewer trades
)
```

## Output

### Console Output
The backtest prints detailed information for each trade:

```
[Trade #1] SUPERTREND_TRAILING_STOP
  Entry:  2024-12-15 09:45:30 @ ₹1,234.50
  Exit:   2024-12-15 11:23:15 @ ₹1,256.20
  Entry ST: ₹1,220.30 → Exit ST: ₹1,245.80 (moved +25.50)
  Duration: 97.8 min | P&L: ₹2,145.00 (+1.76%)
  ✅ PROFIT
```

### CSV Export
Results are exported to `supertrend_trailing_stop_results.csv`:

| Symbol | Trades | Win Rate | Total P&L | Avg P&L | Avg Return | Avg Duration | ST Exits | Square-off | Avg ST Move |
|--------|--------|----------|-----------|---------|------------|--------------|----------|------------|-------------|
| RELIANCE | 12 | 58.3% | ₹4,523.50 | ₹376.96 | 0.42% | 85.3 min | 9 | 3 | ₹15.20 |
| TCS | 8 | 62.5% | ₹3,210.20 | ₹401.28 | 0.38% | 112.5 min | 6 | 2 | ₹22.45 |

### Summary Statistics
- Total trades across all symbols
- Overall win rate
- Total P&L
- Profitable symbols count
- Exit reason breakdown

## Performance Metrics Tracked

1. **Total Trades** - Number of completed trades
2. **Win Rate** - Percentage of profitable trades
3. **Total P&L** - Sum of all trade profits/losses
4. **Average P&L** - Mean profit/loss per trade
5. **Average Return** - Mean percentage return per trade
6. **Best/Worst Trade** - Largest profit and loss
7. **Average Duration** - Mean time in trades
8. **Exit Reasons** - Breakdown of how trades exited
9. **Supertrend Movement** - How much the stop moved up on average

## Strategy Strengths

1. **Simplicity** - Easy to understand and implement
2. **Objectivity** - No subjective decisions
3. **Trend Capture** - Stays in strong trends
4. **Risk Management** - Dynamic stop loss management
5. **Volatility Adjustment** - Adapts to market conditions automatically

## Strategy Limitations

1. **Whipsaws** - Can get stopped out in choppy, ranging markets
2. **Late Entries** - Enters after trend has already started
3. **No Confirmation** - Single indicator (no additional filters)
4. **Gap Risk** - Doesn't protect against overnight gaps
5. **Parameter Sensitivity** - Results vary with period/multiplier settings

## Optimization Tips

1. **Test Different Periods** - Try 7, 10, 14, 21 for different timeframes
2. **Adjust Multiplier** - Lower for day trading (2-2.5), higher for swing (3-4)
3. **Add Volume Filter** - Only enter on above-average volume
4. **Add Time Filter** - Avoid first 15 minutes (high volatility)
5. **Combine with Trend** - Only trade in direction of higher timeframe trend

## Example Modifications

### Add Volume Filter
```python
# In generate_signals method
df['buy_signal'] = (
    (df['st_buy_signal']) &
    (~df['supertrend'].isna()) &
    (~df['atr'].isna()) &
    (df['volume'] > df['volume'].rolling(20).mean())  # Add this line
)
```

### Add Minimum Trend Strength
```python
# Require price to be significantly above Supertrend at entry
df['buy_signal'] = (
    (df['st_buy_signal']) &
    (~df['supertrend'].isna()) &
    (~df['atr'].isna()) &
    (df['close'] > df['supertrend'] * 1.005)  # 0.5% above ST line
)
```

## Files

- `SupertrendTrailingStop.py` - Main strategy implementation
- `SUPERTREND_TRAILING_STOP_STRATEGY.md` - This documentation
- `supertrend_trailing_stop_results.csv` - Output results (generated after run)

## Related Strategies

- `SupertrendHeikinAshi.py` - Supertrend with Heikin Ashi confirmation
- `AdvancedHeikinAshi.py` - Heikin Ashi with ADX filter
- `VolatilityContraction.py` - Bollinger Band based strategy

## References

- Supertrend Indicator: Combines ATR and price to identify trends
- ATR (Average True Range): Volatility measurement
- Trailing Stop Loss: Risk management technique that locks in profits

---

**Created**: 2026-01-15
**Version**: 1.0
**Author**: Backtest Framework
**License**: For educational and research purposes
