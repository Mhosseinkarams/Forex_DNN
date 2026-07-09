# Market Structure Engine

## Purpose
The `MarketStructureEngine` extracts objective Smart Money market structure from OHLC data. It identifies key structural points and transitions without making trading decisions.

## Algorithms

### Swing High/Low Detection
Detects local peaks and troughs based on a configurable lookback period.
- **Swing High**: A candle high that is higher than the `n` candles before it and `n` candles after it.
- **Swing Low**: A candle low that is lower than the `n` candles before it and `n` candles after it.

```ascii
Lookback (n) = 2

      [H]           <-- Swing High
     /   \
  [ ]     [ ]
 /           \
[ ]           [ ]
```

### Break of Structure (BOS)
A BOS occurs when the price continues the current trend by breaking the last confirmed swing point in the trend direction.
- **Bullish BOS**: Price closes above the last confirmed Swing High during a bullish trend.
- **Bearish BOS**: Price closes below the last confirmed Swing Low during a bearish trend.

```ascii
Bullish Trend
      [SH2]  <-- BOS (Price breaks SH1)
      /  \
    /     \
 [SH1]     \
 /  \       \
/    \      [SL2]
     [SL1]
```

### Change of Character (CHOCH)
A CHOCH signals a potential trend reversal by breaking the last swing point of the *opposite* direction.
- **Bullish CHOCH**: Price closes above the last confirmed Swing High while the trend was bearish.
- **Bearish CHOCH**: Price closes below the last confirmed Swing Low while the trend was bullish.

```ascii
Bearish to Bullish CHOCH
[SH1]
  \      [SH2] <-- CHOCH (Price breaks SH1)
   \     /
   [SL1]/
```

## Columns Added to DataFrame
- `trend`: Current trend direction (1: Bullish, -1: Bearish, 0: Neutral).
- `bos`: BOS event (1: Bullish BOS, -1: Bearish BOS, 0: None).
- `choch`: CHOCH event (1: Bullish CHOCH, -1: Bearish CHOCH, 0: None).
- `bars_since_bos`: Number of bars since the last BOS.
- `bars_since_choch`: Number of bars since the last CHOCH.

## Market Structure Summary Dictionary
The `get_summary()` method returns:
- `trend`
- `structure_state`
- `bos_count`
- `choch_count`
- `bars_since_bos`
- `bars_since_choch`
- `last_bos_direction`
- `last_choch_direction`
- `swing_high`
- `swing_low`

## Usage Example
```python
from MarketStructure.market_structure import MarketStructureEngine

engine = MarketStructureEngine(lookback=3)
df_enriched = engine.process(df)
summary = engine.get_summary(df_enriched)
```
