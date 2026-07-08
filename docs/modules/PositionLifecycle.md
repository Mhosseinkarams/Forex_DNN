# PositionLifecycle Module

## Purpose
The `PositionLifecycle` module defines the immutable domain objects that represent a trade's entire history. It acts as the canonical data model for post-trade analysis, machine learning training, and performance reporting.

## Responsibilities
- **Data Definition**: Defining `SignalInfo`, `ExecutionInfo`, `ManagementInfo`, and `OutcomeInfo` dataclasses.
- **Serialization**: Converting lifecycle objects to/from JSON and flattened CSV rows.
- **Reconstruction**: Providing the `PositionLifecycleBuilder` to aggregate data from multiple sources into a single object.

## Core Dataclasses

### `SignalInfo`
- Context: `signal_id`, `strategy`, `symbol`, `direction`.
- Features: `indicator_snapshot` (all indicators at entry).

### `ExecutionInfo`
- Entry: `ticket`, `actual_entry`, `initial_volume`, `slippage`.
- Risk: `risk_percent`, `risk_amount`.

### `ManagementInfo`
- Timeline: `partial_closes`, `breakeven_events`, `management_events`.
- Metrics: `maximum_favorable_excursion`, `maximum_adverse_excursion`.

### `OutcomeInfo`
- Finality: `realized_profit`, `result` (WIN/LOSS/BE), `r_multiple`, `duration`.

## Public API

### `PositionLifecycleBuilder.build_from_data(...) -> PositionLifecycle`
**Static Method**
Aggregates journal DataFrames, broker deals, and state file dictionaries to produce a full object.

### `to_csv_row() -> dict`
Flattens the entire object into a single-level dictionary for CSV logging.

### `to_markdown() -> str`
Generates a human-readable audit report.

## Interaction with Other Modules
- **ExitManager**: Triggers the builder when a position closes.
- **TradeAuditor**: Uses the builder to reconstruct history for forensic review.
- **StatisticsEngine**: Consumes a list of these objects to calculate strategy performance.

## Best Practices
- **Immutability**: Treat the `PositionLifecycle` object as read-only once generated.
- **Serialization**: Use `to_json()` for storage to maintain the full fidelity of nested lists and dictionaries.
