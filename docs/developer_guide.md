# Developer Guide: Extending the Framework

This guide explains how to extend and customize the framework with new strategies, indicators, and management rules.

## How to Create a New Strategy

1.  **Inherit or Implement**: Create a new class. It needs access to `DataFeed`, `SendOrder`, `DrawdownManager`, and `TradingJournal`.
2.  **Define Signal Logic**: Implement a `_poll_cycle` method that checks for signals.
3.  **Execute**: Use `SendOrder.execute()` to submit signals.

### Example Strategy Template
```python
class MyNewStrategy:
    def __init__(self, data_feed, send_order, ...):
        self.df = data_feed
        self.so = send_order

    def _poll_cycle(self):
        # 1. Get Data
        df = self.df.get_ohlcv("EURUSD_o", "M5")

        # 2. Logic (Example: RSI Cross)
        if df['rsi'].iloc[-2] < 30 and df['rsi'].iloc[-1] > 30:
            # 3. Submit
            self.so.execute(
                symbol="EURUSD_o",
                direction=1,
                entry_price=0.0,
                sl_price=df['Low'].min(),
                exit_profile="standard",
                strategy="my_new_strat",
                signal_category="standard",
                signal_id=str(uuid.uuid4())
            )
```

## How to Create a New Indicator

Indicators are calculated in `Collecting_Data/indicators.py`.

1.  Add parameters to `IndicatorEngine.__init__`.
2.  Add calculation logic to `IndicatorEngine.calculate`.
3.  Ensure you work on a copy of the DataFrame (`df.copy()`).

### Example
```python
# indicators.py
def calculate(self, df):
    df = df.copy()
    # Add RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    return df
```

## How to Create a New Exit Profile

Exit profiles define how `ExitManager` handles a position after entry.

1.  **Define Constant**: In `Collecting_Data/position_lifecycle.py`:
    ```python
    EXIT_PROFILE_TRAILING = "trailing"
    ```
2.  **Map Logic**: In `PositionManager/exit_manager.py`, update `register_position` to handle the new profile:
    ```python
    if exit_profile == EXIT_PROFILE_TRAILING:
        stage = "trailing"
        final_tp = 5 # arbitrary
    ```
3.  **Implement Behavior**: Create a `_handle_trailing_stage` method in `ExitManager` and call it from `_poll_cycle`.

## How to Create a New Risk Model

Risk models are encapsulated in `PositionManager/risk_sizing.py`.

1.  Modify `calculate_lot_size` or add a new method to `PositionSizer`.
2.  Example: Implement a "Fixed Lot" model instead of "Percentage Risk".

```python
def calculate_fixed_lot(self, symbol, lot_size=0.1):
    # Validation against symbol_info (min/max/step)
    info = mt5.symbol_info(symbol)
    # ... rounding logic ...
    return {"success": True, "lot_size": 0.1, ...}
```

## How to Add a New Journal Field

The `TradingJournal` uses dynamic column detection. To add a field:
1.  Pass the new field in the `extra_fields` dictionary when calling `log_signal` or `log_order_open`.
2.  The journal will automatically detect the new key, update the CSV header, and backfill previous rows with empty strings.

### Example
```python
extra = {"sentiment_score": 0.85, "news_event": "NFP"}
journal.log_signal(..., extra_fields=extra)
```

## How to Add a New PositionLifecycle Metric

Metrics are defined in `Collecting_Data/position_lifecycle.py`.

1.  **Update Dataclass**: Add the field to `ManagementInfo` or `OutcomeInfo`.
2.  **Update Builder**: In `PositionLifecycleBuilder.build_from_data`, implement the logic to calculate the new metric from broker deals or journal events.

### Example: Calculating "Max Consecutive Wins"
1.  Add `max_cons_wins` to a new `StatisticsInfo` dataclass.
2.  Update `StatisticsEngine.calculate_metrics` to compute it from the `df['outcome_result']` series.
