# RiskSizing (PositionSizer) Module

## Purpose
The `PositionSizer` module calculates the appropriate lot size for a trade based on account risk parameters and broker constraints. It ensures that the dollar amount at risk remains consistent with the user's risk management policy.

## Responsibilities
- **Lot Calculation**: Computing `volume = (Balance * Risk%) / (SL_Distance * Contract_Size)`.
- **Broker Constraints**: Adjusting calculated lots to fit the broker's `volume_min`, `volume_max`, and `volume_step`.
- **Rounding**: Applying precise rounding to avoid "Invalid Volume" errors from MT5.
- **Verification**: Calculating the *actual* dollar risk and percentage after rounding/capping.

## Public API

### `calculate_lot_size(symbol, entry_price, sl_price, risk_pct, account_balance) -> dict`
**Main Calculation Method**
- **risk_pct**: Float representing risk (e.g., `0.01` for 1%).
- **Returns**: A dictionary with:
    - `success` (bool)
    - `lot_size` (float)
    - `risk_dollars` (float)
    - `risk_pct_actual` (float)
    - `capped_at_max` (bool)
    - `error` (str | None)

## Interaction with Other Modules
- **SendOrder**: Calls this module before every trade entry.
- **MMStrategy**: The strategy provides the `risk_pct_default` which is then passed to this module.

## Example Usage

```python
sizer = PositionSizer()
res = sizer.calculate_lot_size("EURUSD_o", 1.1000, 1.0950, 0.01, 10000.0)

if res["success"]:
    print(f"Trade {res['lot_size']} lots for ${res['risk_dollars']} risk.")
```

## Best Practices
- **Pip Distance**: Always calculate distance in price points, not pips, to maintain precision.
- **Check Capping**: Monitor the `capped_at_max` flag. If a trade is consistently capped, the account risk settings or leverage might need adjustment.
