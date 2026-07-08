# Journal Schema & Position Lifecycle

The framework uses a two-layer journaling system to ensure both real-time auditability and high-fidelity historical analysis.

## Layer 1: Event Journal (Real-Time)

Located in `Journals/{mode}/events/`, these files are append-only logs of every significant event in the system.

### CSV Files
- `*_signals.csv`: Every technical setup detected by the strategy.
- `*_order_open.csv`: Results of order execution (tickets, actual entry prices).
- `*_partial_close.csv`: Records of TP1/TP2 hits and volume reduction.
- `*_position_closed.csv`: Final closure event (reason, exit price).
- `*_enrichment.csv`: Additional metadata added to a signal for ML (indicators, sentiment).

### Core Fields (Common to all Layer 1 events)
| Field | Description |
| :--- | :--- |
| `event_id` | Unique UUID for the event. |
| `signal_id` | Shared UUID linking all events related to a single trade setup. |
| `event_type` | `signal`, `order_open`, `partial_close`, `position_closed`, etc. |
| `system_timestamp` | ISO8601 time when the event occurred on the local machine. |
| `bar_timestamp` | Datetime of the bar that triggered the signal. |
| `strategy` | Name of the strategy (e.g., `mm`). |
| `symbol` | Trading symbol (e.g., `EURUSD_o`). |
| `timeframe` | `M5`, `M15`, etc. |
| `direction` | `1` for BUY, `-1` for SELL. |

## Layer 2: Position Summary (Completed Positions)

Located in `Journals/{mode}/positions/`, these files represent the canonical "Truth" of a completed trade. They are generated only after a position is fully closed.

### Formats
- **CSV**: Flattened one-row-per-trade format, ideal for Pandas/Excel analysis.
- **JSONL**: Full nested structure containing every detail of the `PositionLifecycle` object.

### The PositionLifecycle Object
The lifecycle is divided into four sections:

#### 1. Signal Info
- `strategy`, `signal_category`, `exit_profile`.
- `indicator_snapshot`: Dictionary of all technical indicator values at entry.

#### 2. Execution Info
- `ticket`, `magic_number`.
- `actual_entry`, `initial_volume`.
- `risk_percent`, `risk_amount` (dollars).
- `execution_latency`: Time between signal detection and broker confirmation.

#### 3. Management Info
- `partial_closes`: List of timestamps, volumes, and prices.
- `breakeven_events`: List of SL modifications.
- `maximum_favorable_excursion (MFE)`: The highest profit the trade reached.
- `maximum_adverse_excursion (MAE)`: The lowest equity point during the trade.

#### 4. Outcome Info
- `realized_profit`: Final dollar profit/loss.
- `result`: `WIN`, `LOSS`, or `BREAKEVEN`.
- `r_multiple`: Profit normalized by initial risk.
- `duration`: Time in seconds from entry to final close.

## Typical Event Sequences

### Successful Standard Trade (WIN)
1.  `signal`: Buy signal detected at M5 close.
2.  `order_open`: Ticket 123456 opened at 1.1000.
3.  `partial_close`: TP1 hit, 50% closed at 1.1010.
4.  `sl_modified`: SL moved to 1.1000 (breakeven).
5.  `position_closed`: TP2 hit, remaining volume closed at 1.1020.
6.  *Lifecycle Construction*: Final summary written to `positions/`.

### Stop Loss Hit (LOSS)
1.  `signal`: Sell signal detected.
2.  `order_open`: Ticket 123457 opened.
3.  `position_closed`: Reason `stop_loss`, exit at SL price.
4.  *Lifecycle Construction*: Final summary written.

## Field Definitions (Summary CSV)

| CSV Column Prefix | Description |
| :--- | :--- |
| `signal_*` | Context from the strategy (e.g., `signal_indicator_snapshot_ema_600`). |
| `execution_*` | Trade entry details (e.g., `execution_slippage`, `execution_latency`). |
| `management_*` | Evolution of the trade (e.g., `management_current_stage`). |
| `outcome_*` | Final performance (e.g., `outcome_r_multiple`, `outcome_realized_profit`). |
