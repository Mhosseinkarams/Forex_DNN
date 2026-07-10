# TradeLocationEngine

## Purpose
The `TradeLocationEngine` determines entry coordinates, Stop Loss (SL) boundaries, and Take Profit (TP) targets using purely structural information from the `MarketStructureGraph`.

## Responsibilities
- Locates candidate stop-loss coordinates behind nearest protected levels or supply/demand boundaries.
- Replaces raw ATR multipliers or arbitrary pip-offsets with physical market levels.
- Calculates point-in-time Risk/Reward (R:R) targets.

## Method Reference

### `get_trade_levels(...)`
- **Inputs**:
  - `msg`: `MarketStructureGraph` instance.
  - `direction`: Buy (`1`) or Sell (`-1`).
  - `entry_price`: Current market entry candidate.
  - `exit_profile`: Applied trade management profile.
- **Returns**: A dictionary containing `entry_price`, `sl_price`, `tp_price`, `invalidation_level`, and `rr_ratio`.
