# IndicatorEngine Module

## Purpose
The `IndicatorEngine` is a stateless module responsible for calculating technical indicators and market metadata from raw OHLCV data. It ensures that features used for strategy logic and ML training are calculated consistently across all environments.

## Responsibilities
- **EMA Calculation**: Exponential Moving Averages for multiple periods.
- **Volatility**: ATR (Average True Range) calculation using Wilder's method.
- **Trend Detection**: Calculating the slope of long-term EMAs (600, 800).
- **Metadata**: Calculating candle body ratios, shadow sizes, and direction.
- **Relative Features**: Calculating price distance from EMAs and crossovers.

## Public API

### `IndicatorEngine(ema_periods, atr_period, slope_period)`
**Constructor**
- **ema_periods** (list[int]): List of EMA windows to calculate (e.g., `[50, 600]`).
- **atr_period** (int): Window for ATR calculation (default: 14).
- **slope_period** (int): Lookback for calculating EMA slopes (default: 32).

### `calculate(df: pd.DataFrame) -> pd.DataFrame`
Receives a DataFrame in standard schema and returns a new DataFrame with all indicators and metadata columns appended.
- **Side Effects**: None. Does not modify the input DataFrame.
- **Exceptions**: Logs a warning if the input DataFrame has fewer rows than the longest EMA period.

## Usage Example

```python
from Collecting_Data.indicators import IndicatorEngine

engine = IndicatorEngine(ema_periods=[50, 600])
df_with_indicators = engine.calculate(raw_ohlcv_df)

print(df_with_indicators['ema_600'].iloc[-1])
print(df_with_indicators['ema_slope_600'].iloc[-1])
```

## Interaction with Other Modules
- **MMStrategy**: Uses the returned indicators to evaluate signal entry conditions.
- **Machine Learning**: The output of this engine provides the feature vector for model training.

## Common Mistakes
- **Warmup**: Forgetting that EMA 600 requires at least 600+ bars of history to be accurate. Always provide a "pre-roll" of data.
- **Lookahead Bias**: Ensure you only look at `iloc[-2]` (last closed bar) for entry signals to avoid using information from a candle that hasn't finished forming.
