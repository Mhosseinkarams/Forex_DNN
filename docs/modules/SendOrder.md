# SendOrder Module

## Purpose
`SendOrder` is the primary orchestrator for trade entry. It acts as a safety layer between strategy signals and the broker, ensuring that all trades comply with risk limits and do not conflict with existing positions.

## Responsibilities
- **Drawdown Validation**: Checking if the `DrawdownManager` allows new trades.
- **Conflict Checking**: Implementing rules (Rule 1-3) to prevent duplicate or contradictory trades on the same symbol.
- **Risk Allocation**: Applying per-signal risk caps (e.g., 0.5% for high-risk signals).
- **Orchestration**: Coordinating `PositionSizer`, `PositionManager`, and `ExitManager` to execute a trade.
- **State Metadata**: Persisting the "Signal Category" of each ticket for conflict resolution.

## Public API

### `execute(symbol, direction, entry_price, sl_price, exit_profile, strategy, signal_category, signal_id) -> dict`
**The main entry point for strategy signals.**
- **signal_category**: `standard`, `high_risk`, or `reversal`.
- **exit_profile**: `standard` or `single`.
- **Returns**: A dictionary indicating success/failure and the reason (e.g., `conflict_blocked`, `drawdown_blocked`).

## Conflict Rules
1.  **Rule 1**: Same category + same direction -> Block (prevents overexposure).
2.  **Rule 2**: Different category + same direction -> Allowed only if new SL is "safer" (higher for buys, lower for sells).
3.  **Rule 3**: Opposite direction -> Blocked if TP crosses existing SL (prevents "wash trading" unless the existing trade is a `reversal`).

## Interaction with Other Modules
- **MMStrategy**: Calls `execute()` when a signal is detected.
- **DrawdownManager**: Consulted for trading permission and max risk %.
- **PositionTracker**: Queried for existing positions to check conflicts.
- **PositionSizer**: Used to calculate the lot size.
- **PositionManager**: Used to send the order to MT5.
- **ExitManager**: Used to register the position for management after successful entry.

## Best Practices
- **Signal ID**: Always generate a unique UUID for the `signal_id` before calling `execute`. This links all subsequent events in the journal.
