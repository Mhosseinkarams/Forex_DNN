# Trade Auditor

## Purpose
The Trade Auditor is a developer forensic tool designed to reconstruct the complete lifecycle of a trade within the Forex Trading Framework. It aggregates data from multiple sources to provide a clear, chronological report of what happened during a trade, without requiring manual log inspection.

The auditor is **READ-ONLY**:
- It NEVER sends orders.
- It NEVER modifies state files or journals.
- It NEVER changes broker data.

## Architecture
The tool is built as a standalone utility that interfaces with:
- **TradingJournal CSV files**: Event history (signal, order_open, partial_close, outcome).
- **Framework State Files**: `State/*.json` (PositionTracker, ExitManager, SendOrder).
- **MetaTrader 5 API**: Historical deals, orders, and current positions.

### Data Sources
1. `Journals/<mode>/*.csv`
2. `State/*.json`
3. MT5 History (`history_deals_get`, `history_orders_get`)

## Search and Reconstruction
The auditor can find a trade using:
- **MT5 Ticket Number** (`--ticket`)
- **Signal ID** (`--signal-id`)
- **Latest Trades** (`--latest`): Displays a list of recent trades for selection.

### Reconstruction Logic
1. **Identify**: Locates the `signal_id` and `ticket` by searching through the TradingJournal.
2. **Aggregate**: Gathers all matching events from the journal.
3. **Fetch State**: Reads relevant snapshots from framework state files.
4. **Broker Sync**: Queries MT5 for the actual broker-side execution details and PnL.
5. **Timeline**: Builds a unified chronological timeline of all framework and broker events.

## Consistency Checks & Anomaly Detection
Every audit performs automated checks to ensure system integrity:
- **Journal Completeness**: Verifies all lifecycle events (Signal -> Order -> Outcome) are present.
- **Broker Verification**: Compares Journal entry/exit prices and volumes against MT5 records.
- **State Restoration**: Checks if the trade was correctly persisted and restored in framework modules.
- **Anomaly Detection**:
    - Duplicate signals or orders.
    - Missing journal entries.
    - Significant price mismatches (Journal vs. Broker).
    - State/Journal desynchronization.

## Usage

### CLI Examples
```bash
# Audit by ticket
python trade_auditor.py --ticket 12345678

# Audit by signal ID
python trade_auditor.py --signal-id 6fd8...

# Select from latest trades
python trade_auditor.py --latest

# Output to Markdown file
python trade_auditor.py --ticket 12345678 --format markdown

# Specify trading mode
python trade_auditor.py --ticket 12345678 --mode validation
```

### Reports
Generated reports are saved in `AuditReports/`:
- `audit_ticket_<ticket>.md`
- `audit_ticket_<ticket>.json`

## Status Indicators
- **PASS**: Data is consistent and complete.
- **FAIL**: Critical mismatch or missing data detected.
- **WARNING**: Non-critical issue or missing optional data (e.g., no broker history found because the position is still open).
