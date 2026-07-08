# PositionManager Module

## Purpose
The `PositionManager` is the low-level execution engine of the framework. It handles the actual communication with the broker (live or simulated) to open, close, and modify positions.

## Responsibilities
- **Order Execution**: Sending `ORDER_TYPE_BUY` and `ORDER_TYPE_SELL` requests.
- **Position Closure**: Closing full or partial volumes of open tickets.
- **Modification**: Updating Stop-Loss (SL) and Take-Profit (TP) levels of existing positions.
- **Magic Number Management**: Ensuring trades are tagged with the correct ID for tracking.

## Public API

### `PositionManager(magic_unity, magic_mm, deviation, filling_mode)`
**Constructor**
- **magic_unity** (int): Magic number for unity strategy.
- **magic_mm** (int): Magic number for MM strategy.
- **deviation** (int): Allowed price slippage in points.

### `open_position(symbol, direction, lot_size, sl_price, tp_price, strategy, comment) -> dict`
Submits a market order. Returns a standardized result dictionary with `success`, `ticket`, and `entry_price`.

### `close_position(ticket, volume=None) -> dict`
Closes a position. If `volume` is `None`, the entire position is closed.

### `modify_position(ticket, sl_price=None, tp_price=None) -> dict`
Modifies the SL or TP of an active ticket.

## Interaction with Other Modules
- **SendOrder**: Calls `open_position` after validation.
- **ExitManager**: Calls `close_position` and `modify_position` to manage the trade lifecycle.
- **PositionTracker**: Indirectly related; `PositionManager` creates the positions that the tracker monitors.

## Best Practices
- **Error Handling**: Always check the `success` key in the return dictionary. If `False`, the `comment` or `retcode` will contain the reason for failure.
- **Volume Step**: Ensure `lot_size` adheres to the broker's `volume_step` (handled upstream in `PositionSizer`).
