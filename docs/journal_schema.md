# Journal Schema

## Layer 1: Event Journal
The Event Journal records raw events as they occur.

### Base Schema (All Events)
| Field | Type | Description |
|---|---|---|
| event_id | UUID | Unique event identifier |
| signal_id | UUID | Links related events |
| event_type | string | e.g., signal, order_open, sl_modified |
| system_timestamp | ISO8601 | UTC recording time |
| bar_timestamp | ISO8601 | Chart bar time |
| strategy | string | Strategy name |
| symbol | string | Symbol traded |
| timeframe | string | Chart timeframe |
| direction | int | 1 (Buy), -1 (Sell) |

### Event-Specific Fields
- **signal**: `entry_price`, `sl_price`, `tp_level`, `stage`, `signal_category`
- **order_open**: `ticket`, `actual_entry`, `actual_sl`, `actual_tp`, `lot_size`, `risk_pct`
- **sl_modified**: `ticket`, `new_sl`, `reason`
- **position_closed**: `ticket`, `exit_price`, `reason`

---

## Layer 2: Completed Position Summary
The Position Summary contains the finalized `PositionLifecycle` record.

### Schema
| Field | Type | Description |
|---|---|---|
| signal_signal_id | UUID | Unique signal identifier |
| execution_ticket | int | Broker ticket number |
| signal_strategy | string | Strategy name |
| outcome_result | string | WIN, LOSS, BREAKEVEN |
| outcome_realized_profit | float | Final PnL in currency |
| outcome_duration | float | Total duration in seconds |
| outcome_strategy_reason | string | Reason for exit |
| outcome_exit_timestamp | ISO8601 | Time of final exit |
| execution_actual_entry | float | Fill price |
| outcome_average_exit_price | float | Weighted exit price |
| execution_initial_volume | float | Opening lot size |
| outcome_deal_count | int | Total number of deals |
| management_current_stage | int | Final TP stage reached |
| ... | ... | See `PositionLifecycle` for full fields |

**Location:** `Journals/{mode}/positions/{strategy}_{symbol}_{timeframe}_positions.csv`
