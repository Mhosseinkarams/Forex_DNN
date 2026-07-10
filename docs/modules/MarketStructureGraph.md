# MarketStructureGraph

## Purpose
The `MarketStructureGraph` acts as the shared, centralized representation of the market. It organizes all detected physical structures—such as swings, trend breaks, and institutional order blocks—into an object-oriented schema rather than scattered indicator columns.

## Key Dataclasses

### StructureLevel
- `price`: Floating-point price of the level.
- `index`: Historical index of the candle.
- `timestamp`: Datetime of creation.
- `level_type`: Type designation (`SwingHigh`, `SwingLow`, `ProtectedHigh`, `ProtectedLow`).

### Zone
- `upper` & `lower`: Physical price boundaries of the zone.
- `type`: Category (`Supply` or `Demand`).
- `freshness`: Boolean indicating if the zone has not been mitigated.
- `touch_count`: Number of times price has retested the zone.
- `broken`: Boolean indicating if a candle close has broken past the zone.

### MarketStructureGraph
- `swing_highs` & `swing_lows`: List of historical `StructureLevel` objects.
- `supply_zones` & `demand_zones`: List of current active/inactive institutional supply and demand zones.
- `bos` & `choch`: History of structure breaks and trend changes.
- `trend_direction`: Overall direction (`Bull`, `Bear`, `Neutral`).
- `ema_distance_atr`: Relative distance between fast and slow moving averages.

## Example Helper Methods
- `get_nearest_demand(price)`: Retrieves the closest active demand zone below the specified price.
- `get_nearest_supply(price)`: Retrieves the closest active supply zone above the specified price.
