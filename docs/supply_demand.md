# Supply and Demand Engine

## Purpose
The `SupplyDemandEngine` identifies institutional Supply and Demand zones based on impulsive market moves. These zones represent areas where significant buying or selling interest was previously concentrated.

## Algorithm

### Zone Detection
Zones are created at the base of "impulsive moves". An impulsive move is defined as a candle whose body size exceeds a multiplier (default: 2.0) of the recent ATR.
- **Demand Zone**: Created at the base of a strong bullish impulsive move.
- **Supply Zone**: Created at the base of a strong bearish impulsive move.

```ascii
Demand Zone Creation
      [ ]  <-- Impulsive Bullish Candle
      [ ]
      [ ]
[BASE]     <-- Demand Zone (Upper: base close/open, Lower: base low)
```

### Zone Lifecycle
- **Fresh**: A newly created zone that has not been touched.
- **Mitigated**: A zone that has been touched by price (price entered the zone).
- **Broken**: A zone is considered broken and removed from active consideration if price *closes* past the zone's boundary.

### Strength Scoring
Strength is determined by:
1. **Impulse Size**: How large the initial move was relative to ATR.
2. **Departure Speed**: How quickly price continued to move away from the zone in subsequent candles.
3. **Retests**: Each touch reduces the strength score as liquidity is consumed.

## Columns Added to DataFrame
- `nearest_supply_distance`: Distance to the nearest active supply zone above price.
- `nearest_demand_distance`: Distance to the nearest active demand zone below price.
- `inside_supply`: Boolean (1/0) indicating if price is currently inside a supply zone.
- `inside_demand`: Boolean (1/0) indicating if price is currently inside a demand zone.
- `supply_strength`: Strength score of the nearest supply zone.
- `demand_strength`: Strength score of the nearest demand zone.
- `bars_since_supply`: Bars since the nearest supply zone was created.
- `bars_since_demand`: Bars since the nearest demand zone was created.

## Usage Example
```python
from MarketStructure.supply_demand import SupplyDemandEngine

engine = SupplyDemandEngine(atr_period=14, impulse_threshold=2.5)
df_enriched = engine.process(df)
```
