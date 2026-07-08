# State Management & Recovery Guide

This document explains how the framework maintains persistence and recovers from system interruptions.

## Why State Exists

Automated trading is a continuous process, but software is subject to restarts (updates, crashes, server reboots). To manage trades that last for hours or days, the framework must remember:
- Which positions it is currently tracking.
- What stage of management each position is in (e.g., has TP1 already been hit?).
- What the daily drawdown status is.
- Which signal category a ticket belongs to (to prevent conflicts).

## State Files

All state is stored in the `State/` directory as JSON files. These files use an **atomic write pattern** (write to `.tmp`, then rename) to prevent corruption during power failures.

| File | Module | Key Information |
| :--- | :--- | :--- |
| `position_tracker_state.json` | `PositionTracker` | List of tickets currently being monitored. |
| `exit_manager_state.json` | `ExitManager` | TP ladder, current stage, breakeven status for each ticket. |
| `drawdown_state.json` | `DrawdownManager` | Start-of-day balance and current day snapshot date. |
| `send_order_state.json` | `SendOrder` | Mapping of ticket IDs to signal categories (standard/hr/rev). |
| `mm_strategy_state.json` | `MMStrategy` | Last processed bar time per symbol/timeframe. |

## Persistence Logic

1.  **Polling Updates**: Modules like `PositionTracker` and `ExitManager` update their state files every poll cycle.
2.  **On-Action Updates**: `SendOrder` updates its state immediately after a successful trade entry.
3.  **Atomic Writes**: The framework ensures that a crash during a write doesn't leave an empty or partial JSON file.

## Recovery Workflow

Upon startup (e.g., when `main.py` is executed), the following recovery sequence occurs:

1.  **Drawdown Restoration**: `DrawdownManager` loads the last `start_of_day_balance`. If the current date matches the `snapshot_date`, it resumes. If it's a new day, it snapshots the current balance.
2.  **Position Discovery**: `PositionTracker` queries MT5 for all open positions matching its magic numbers.
3.  **Exit Re-registration**: `ExitManager` loads its state. Any ticket found in the broker's active list that is also in the state file is resumed exactly where it left off.
4.  **Strategy Sync**: `MMStrategy` loads `last_bar_time`. It will ignore signals for the current bar if it has already processed it, preventing duplicate trades on restart.

## Common Recovery Scenarios

### Scenario 1: Server Reboot
The system reboots while 3 trades are open.
- **Recovery**: Upon restart, the framework sees the 3 tickets in MT5. It loads `exit_manager_state.json` and sees that Ticket A had already hit TP1. It continues managing Ticket A starting from the TP2 stage.

### Scenario 2: Manual Trade Closure
A user manually closes a trade in MT5 while the bot is offline.
- **Recovery**: Upon restart, `PositionTracker` sees the ticket is gone. `ExitManager` attempts to reconcile, finds no active ticket, and determines the closure reason from broker history. It logs the final `PositionLifecycle` as a `manual_client` closure.

### Scenario 3: Missing State File
If `exit_manager_state.json` is deleted but positions are still open.
- **Recovery**: The framework will see the open positions but won't know their management history (TP1 status, etc.). It will re-register them as "New" positions and might attempt to hit TP1 again if the price is still favorable. **Warning**: Avoid deleting state files while positions are active.

## Best Practices

- **Monitoring**: Regularly back up the `State/` directory.
- **Clock Sync**: Ensure the server clock is accurate, as state files contain ISO8601 timestamps used for day rollover detection.
- **Shutdown**: Always attempt a graceful shutdown (`Ctrl+C`) to ensure final state flushes, though the system is designed to handle hard crashes.
