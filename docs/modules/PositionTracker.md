# PositionTracker Module

## Purpose
The `PositionTracker` maintains a real-time, local snapshot of all active trades matching the framework's magic numbers. It serves as the "Source of Truth" for other modules.

## Responsibilities
- **Polling**: Continuously querying the broker for open positions.
- **Metric Calculation**: Calculating current risk, reward, and floating PnL for each position.
- **State Persistence**: Saving the list of active tickets to `position_tracker_state.json` for recovery.
- **Anomaly Detection**: Identifying if a position was closed externally (e.g., manual closure in MT5).

## Public API

### `PositionTracker(magic_numbers, poll_interval_seconds, state_file)`
**Constructor**
- **magic_numbers** (list[int]): Only track positions with these magic numbers.
- **poll_interval_seconds** (int): How often to poll the broker.

### `start()` / `stop()`
Starts/stops the background polling thread.

### `get_open_positions() -> list[dict]`
Returns a list of snapshots for all currently tracked positions.

### `get_open_risk() -> float`
Returns the total dollar value of all active risk (distance from entry to SL).

## Usage Example

```python
tracker = PositionTracker(magic_numbers=[100001, 100002])
tracker.start()

# Later...
risk = tracker.get_open_risk()
if risk > 500:
    print("Warning: High exposure!")
```

## Interaction with Other Modules
- **DrawdownManager**: Consumes `get_open_risk()` to calculate the remaining daily loss limit.
- **ExitManager**: Reconciles its tracked tickets against the tracker's live list to detect closures.
- **SendOrder**: Checks the tracker for conflicting positions before opening new ones.

## Common Mistakes
- **Poll Interval**: Setting the poll interval too high (>30s) can lead to stale risk calculations and delayed recovery.
