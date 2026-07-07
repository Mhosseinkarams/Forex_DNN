# MMStrategy Module

## Purpose
`MMStrategy` implements the core intraday trading logic of the framework. It monitors market conditions, identifies technical setups, and orchestrates the submission of trades via the execution layer.

## Responsibilities
- **Polling**: Driving the main strategy loop based on a configured interval.
- **Signal Evaluation**: Implementing logic for `Standard`, `High-Risk`, and `Reversal` signals.
- **Stop Loss Calculation**: Determining SL prices based on swing highs/lows and pip caps.
- **Feature Capture**: Snapshotting indicator values at the time of signal for journaling and ML.
- **State Persistence**: Remembering the last processed bar to prevent duplicate trades on restart.

## Public API

### `MMStrategy(data_feed, send_order, drawdown_manager, symbols, ...)`
**Constructor**
- **data_feed**: Instance of `MT5DataFeed`.
- **send_order**: Instance of `SendOrder`.
- **drawdown_manager**: Instance of `DrawdownManager`.
- **symbols** (list[str]): Symbols to monitor.

### `start()`
Starts the background polling thread.

### `stop()`
Gracefully stops the background thread.

### `_poll_cycle()`
The internal loop that iterates over symbols and timeframes, retrieves data, and checks for signals.

## Typical Workflow
1.  Check for a "New Bar" on M5 or M15.
2.  Calculate indicators using `IndicatorEngine`.
3.  Check conditions in order of priority: High-Risk > Standard > Reversal.
4.  If a signal is found, calculate the SL and entry price.
5.  Call `send_order.execute()`.

## Interaction with Other Modules
- **SendOrder**: Receives the signal details for validation and execution.
- **DrawdownManager**: Consulted before every trade to ensure loss limits aren't exceeded.
- **TradingJournal**: Receives signal events for logging.

## Best Practices
- **Priority**: High-risk signals are evaluated first because they often represent sharp momentum shifts that invalidate standard trend continuation setups.
- **Timeframes**: Standard trend signals are evaluated on both M5 and M15 to capture multi-timeframe alignment.
