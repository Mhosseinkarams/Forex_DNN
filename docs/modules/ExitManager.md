# ExitManager Module

## Purpose
The `ExitManager` handles the "active management" of a trade after it has been opened. It implements complex exit strategies like multi-staged take-profits and breakeven moves.

## Responsibilities
- **Registration**: Onboarding new tickets into the management system.
- **TP Ladder Calculation**: Calculating the price levels for TP1, TP2, etc., based on the entry and SL.
- **Staged Execution**: Executing partial closes and moving SL to entry (breakeven) or previous TP levels.
- **Closure Detection**: Detecting when a trade hits SL or TP and triggering final journaling.
- **Forensic Reconstruction**: Building the `PositionLifecycle` summary after trade closure.

## Public API

### `ExitManager(position_tracker, position_manager, poll_interval_seconds, state_file, trading_journal)`
**Constructor**
- **position_tracker**: Instance of `PositionTracker`.
- **position_manager**: Instance of `PositionManager`.
- **trading_journal**: Instance of `TradingJournal`.

### `register_position(ticket, entry_price, sl_price, direction, exit_profile, signal_id)`
Registers a new position.
- **exit_profile**: `standard` (TP1->BE->TP2) or `single` (TP1->Close).

### `start()` / `stop()`
Starts/stops the background polling thread.

## Interaction with Other Modules
- **SendOrder**: Calls `register_position` immediately after a successful trade entry.
- **PositionTracker**: Used to monitor current prices of active tickets.
- **PositionManager**: Used to execute partial closes and SL modifications.
- **TradingJournal**: Receives lifecycle events (partial closes, modifications) and the final `PositionLifecycle` object.

## Typical Workflow
1.  **Register**: A trade is opened; `ExitManager` calculates the TP ladder.
2.  **Monitor**: Price reaches TP1.
3.  **Execute**: `ExitManager` calls `PositionManager.close_position(volume=50%)` and `modify_position(sl=entry)`.
4.  **Finalize**: Price reaches TP2. `ExitManager` closes the trade and triggers `PositionLifecycle` generation.

## Best Practices
- **Atomic State**: The `ExitManager` depends heavily on `exit_manager_state.json`. Never edit this file manually while the bot is running.
