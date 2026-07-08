# TradingJournal Module

## Purpose
The `TradingJournal` provides a unified logging interface for all trading activity. It implements a two-layer architecture designed for both real-time monitoring and high-fidelity post-trade analysis.

## Responsibilities
- **Layer 1 (Event Journal)**: Real-time logging of individual actions (signals, entries, partials, closures).
- **Layer 2 (Position Summary)**: Construction and storage of the `PositionLifecycle` canonical summary.
- **File Management**: Managing directory structures for `live`, `backtest`, and `training` modes.
- **Dynamic Schema**: Automatically handling new metadata fields without breaking existing CSV files.
- **Thread Safety**: Using per-file locks to ensure data integrity in multi-threaded environments.

## Public API

### `TradingJournal(journal_root, mode)`
**Constructor**
- **journal_root** (str): Root directory for logs.
- **mode** (str): `live`, `backtest`, `validation`, or `training`.

### `log_signal(...) -> str`
Logs a technical setup. Returns a `signal_id` (UUID) used to link all future events.

### `log_order_open(signal_id, ticket, actual_entry, ...)`
Logs the successful execution of an order.

### `log_position_closed(signal_id, ticket, exit_price, reason, ...)`
Logs the final closure of a position.

### `log_lifecycle(lifecycle: PositionLifecycle)`
Logs a completed Layer 2 summary as both CSV and JSONL.

## Interaction with Other Modules
- **MMStrategy**: Logs signals.
- **SendOrder**: Logs order entry success/failure.
- **ExitManager**: Logs partial closes and final position summaries.
- **TradeAuditor**: Consumes journal data to reconstruct histories.

## Best Practices
- **UUID**: Always use the `signal_id` returned by `log_signal` for all subsequent calls related to that trade.
- **Atomic Renames**: The journal uses atomic renames for CSV header updates to prevent data loss.
