# TradeAuditor Guide

The `TradeAuditor` is a forensic tool designed to reconstruct the complete history of a trade by aggregating data from multiple fragmented sources.

## Purpose

Automated trading involves multiple asynchronous layers:
- Strategy signals (logged locally).
- Orders and deals (stored on the broker's server).
- Exit management logic (state stored in local JSON files).

When a trade goes wrong (e.g., unexpected closure, slippage), it is difficult to see the "Big Picture" from one file. The `TradeAuditor` solves this by building a canonical `PositionLifecycle` object from all available evidence.

## Architecture

The auditor works by:
1.  **Scanning Journals**: Loading all event CSVs to find the `signal_id` related to a ticket.
2.  **Querying the Broker**: Connecting to MT5 to fetch historical deals and orders.
3.  **Inspecting State**: Reading `exit_manager_state.json` and `position_tracker_state.json` to see the framework's intent at the time.
4.  **Synthesizing**: Using the `PositionLifecycleBuilder` to merge these into a structured report.

## How Audits Work

### Consistency Checks
The auditor automatically performs anomaly detection:
- **Price Match**: Compares the journaled `actual_entry` price with the broker's `deal` price.
- **Journal Completeness**: Verifies if `signal`, `order_open`, and `position_closed` events are all present.
- **Latency Analysis**: Calculates the time taken from signal detection to broker execution.
- **State Integrity**: Checks if the position was correctly registered in `ExitManager`.

### Timeline Reconstruction
The auditor generates a chronological ASCII timeline of the trade:
```text
2023-10-27T10:00:01Z
  signal: MMStrategy - BUY EURUSD_o
    │
    ▼
2023-10-27T10:00:02Z
  order_open: Ticket 12345678 (Entry: 1.1000)
    │
    ▼
2023-10-27T10:15:30Z
  partial_close: TP1 reached (Closed: 0.05 lot)
    │
    ▼
2023-10-27T10:45:00Z
  position_closed: Reason: stop_loss (Exit: 1.1000 - BREAKEVEN)
```

## Usage

### Interactive Latest Trades
The easiest way to audit recent trades:
```bash
python trade_auditor.py --mode live --latest
```

### Audit by Ticket
If you have a specific MT5 ticket number:
```bash
python trade_auditor.py --mode live --ticket 12345678
```

### Audit by Signal ID
If you have a UUID from the journal:
```bash
python trade_auditor.py --mode backtest --signal-id "550e8400-e29b-41d4-a716-446655440000"
```

## Example Report

The auditor can generate Markdown or JSON reports. A typical report includes:
- **Summary**: Ticket, Symbol, Strategy, Final Result.
- **Signal**: Indicators snapshot (e.g., EMA 600 slope at entry).
- **Execution**: Slippage, latency, and broker deal IDs.
- **Management**: A list of every TP hit and SL modification.
- **Outcome**: Realized profit, duration, and R-multiple.

## Investigating Failed Trades

Common scenarios for using the auditor:
1.  **Unexpected Closure**: Audit the trade to see if it was closed by the broker (SL/TP) or by a manual intervention (Reason: `manual_client`).
2.  **High Slippage**: Check the difference between `requested_entry` and `actual_entry`.
3.  **Missing Stages**: Verify if `ExitManager` correctly registered the position by checking the `management` section of the report.
4.  **Drawdown Block**: If a signal wasn't taken, find the signal in the journal and use the auditor to check if `blocked_by_drawdown` was set to `True` in the extra fields.
