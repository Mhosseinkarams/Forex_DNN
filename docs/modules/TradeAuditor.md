# TradeAuditor Module

## Purpose
The `TradeAuditor` is a forensic tool used to reconstruct a trade's complete timeline by aggregating data from local journals, state files, and broker history. It is essential for debugging, performance review, and compliance.

## Responsibilities
- **Data Aggregation**: Merging disparate data points (CSV, JSON, MT5 API).
- **History Reconstruction**: Rebuilding the timeline of a trade from signal to closure.
- **Consistency Verification**: Identifying discrepancies between local records and broker truth.
- **Reporting**: Generating human-readable Markdown and machine-readable JSON reports.

## Public API

### `TradeAuditor(journal_root, state_dir, mode)`
**Constructor**
- **journal_root** (str): Path to journals.
- **state_dir** (str): Path to framework state files.

### `reconstruct_trade_lifecycle(ticket, signal_id) -> PositionLifecycle`
The primary method for rebuilding a trade.
- **ticket** (int): Optional MT5 ticket number.
- **signal_id** (str): Optional Framework signal ID.

### `get_latest_trades(limit=10) -> list[dict]`
Returns a list of recent unique trades from the journals for interactive selection.

### `save_reports(lifecycle, output_dir="AuditReports")`
Writes the audited data to Markdown and JSON files.

## Interaction with Other Modules
- **PositionLifecycle**: Uses the `PositionLifecycleBuilder` to perform the actual reconstruction.
- **TradingJournal**: Acts as the primary source for technical features and local events.
- **MetaTrader 5 API**: Acts as the source for actual execution prices, commissions, and swap.

## Usage Example

```bash
# Auditing the latest trade in live mode
python trade_auditor.py --mode live --latest
```

## Best Practices
- **History Depth**: Ensure your MT5 account history is set to "All History" to allow the auditor to find old deals.
