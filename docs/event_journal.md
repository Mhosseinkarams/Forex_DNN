# Event Journal (Layer 1)

## Purpose
The Event Journal is a real-time, append-only chronological log of all significant events that occur during the operation of the trading framework. It provides a complete audit trail of "what happened" as it happened.

## Architecture
- **Location:** `Journals/{mode}/events/`
- **Format:** CSV
- **Naming Convention:** `{strategy}_{symbol}_{timeframe}_events.csv`
- **Philosophy:** Append-only, immutable, no calculated final outcomes (PnL, Duration) while trade is active.

## Supported Events
- `signal`: Generated when a strategy identifies a potential trade.
- `order_request`: When a request is sent to the broker.
- `order_open`: Confirmed fill from the broker.
- `order_failure`: When an order request is rejected or fails.
- `sl_modified`: Stop-loss price update.
- `tp_modified`: Take-profit price update.
- `partial_close`: Partial volume closure at a staged TP.
- `breakeven`: Stop-loss moved to entry price.
- `trailing_start`: Activation of trailing stop logic.
- `trailing_update`: Update of trailing stop price.
- `position_closed`: Confirmed final closure of the position.
- `enrichment`: Additional metadata added to a signal/trade.

## Required Fields
All events share a common base:
- `event_id`: Unique UUID for the event.
- `signal_id`: UUID linking all events related to a single trade signal.
- `event_type`: Type of event (see above).
- `system_timestamp`: ISO 8601 UTC time of event recording.
- `bar_timestamp`: Time of the chart bar associated with the event.
- `strategy`, `symbol`, `timeframe`, `direction`: Contextual metadata.

## Relationship to PositionLifecycle
The Event Journal is the primary source of truth for reconstructing the trade timeline. Once a `position_closed` event is recorded, the framework aggregates all related events from the journal and cross-references them with broker history to build the final `PositionLifecycle` summary.
