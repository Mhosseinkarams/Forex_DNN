# LabelEngine Operator Guide

The `LabelEngine` module is the centralized, deterministic, rule-based labeling system in the `Forex_DNN` quantitative trading framework. It is responsible for generating high-quality machine learning target classes strictly based on predefined strategy rules, ensuring look-ahead protection and full experiment traceability.

---

## Architecture Overview

Instead of training machine learning models to discover rules from raw data (which is prone to overfitting and noise), `Forex_DNN` utilizes a decoupled paradigm:
1. **Analytical Engines** (`MarketStructureEngine`, `SupplyDemandEngine`) compute objective mathematical states (swings, breaks, zones).
2. **LabelEngine** processes these structural states over configurable sliding windows.
3. **Rule-Based Labelers** (e.g., `MarketStateLabeler`) determine whether a window belongs to a target state or should be removed.
4. **DatasetValidator** ensures that no corrupt, duplicate, or inconsistent samples enter the training datasets.

```
[Raw Candles] -> [MarketStructureEngine] -> [SupplyDemandEngine] -> [MarketStructureGraph]
                                                                             |
[Sliding Window Slices] <----------------------------------------------------+
       |
[MarketStateLabeler] -> [Label Rules Match?] -> YES -> [FeaturePipeline] -> [Save Row]
                               |
                              NO
                               |
                        [Discard Row] (Logged & Manifested)
```

---

## Configuration Hyper-parameters

The `LabelEngine` sliding window behavior is highly customizable to allow quantitative research into optimal context lengths:

- **`window_size`** (Default: `35`): Number of candles included in the context window. Experiments in `Forex_DNN` show `35` balances lookback context and reactivity. Future experiments can test `20`, `25`, `35`, `50`, and `75`.
- **`window_stride`** (Default: `1`): Number of candles between consecutive window starts.
- **`labeler`**: An implementation of `BaseLabeler` providing target logic.
- **`registry`**: Feature registry serving as the source of truth for features.

---

## Deterministic Market State Rules

The default labeler, `MarketStateLabeler`, implements three target classes:

### 1. `TREND`
- **Criteria**:
  - fast/slow EMA separation is greater than `ema_separation_trend` (Default: `1.5` ATR).
  - AND at least one `BOS` break occurs inside the window.
  - AND no opposing `CHOCH` has occurred inside the window (reversal protection).
- **Confidence**: Dynamically scaled based on ATR separation and BOS frequency.

### 2. `RANGE`
- **Criteria**:
  - fast/slow EMA separation is less than `ema_separation_range` (Default: `0.8` ATR).
  - OR the window contains at least `min_rejections_range` (Default: `2`) supply/demand zone touches or inner-candle crossings.
  - AND no `BOS` breaks occur inside the window.
- **Confidence**: Scaled by retests count and zone touch density.

### 3. `TRANSITION`
- **Criteria**:
  - EMAs crossed within the window.
  - OR a `CHOCH` trend change occurred within the window.
  - OR EMA separation is rapidly shrinking (falling from >1.2 to <0.9 in the last 10 candles).
- **Confidence**: Scaled by EMA cross signals and CHOCH strength.

### Unlabeled Sample Quality Control
If a window sample does not meet the strict deterministic criteria for `TREND`, `RANGE`, or `TRANSITION`, the labeler returns `None`. **Ambiguous samples are discarded from the training set.** There are NO fallback classes like `UNKNOWN` or `OTHER`, preventing model confusion on boundary conditions. Discard reasons are logged and recorded in the manifest.

---

## Dataset Validation Checks

The `DatasetValidator` verifies generated datasets before serialization:
- **Missing Values**: Any missing values in critical columns (`target`, `confidence`, `symbol`, `datetime`) trigger a failure.
- **Duplicate Rows**: Identifies duplicate rows or timestamps for a given symbol/timeframe.
- **Timestamp Monotonicity**: Verifies that candle timestamps strictly increase chronologically.
- **Window Consistency**: Assures that `window_end - window_start + 1 == window_size` for every row.
- **Class Distribution**: Identifies class imbalance (e.g., if a class constitutes < 1% of the dataset).

---

## Reproducibility Manifest

For every generated dataset, a JSON manifest is saved atomically alongside the CSV file. This records all hyperparameters and execution statistics, allowing complete traceability of datasets:

```json
{
    "window_size": 35,
    "window_stride": 1,
    "label_version": "1.0.0",
    "feature_registry_version": "a1b2c3d4...",
    "market_structure_engine_version": "1.0.0",
    "supply_demand_engine_version": "1.0.0",
    "symbols": ["EURUSD"],
    "timeframes": ["M15"],
    "date_range": {
        "start": "2026-01-01T00:00:00",
        "end": "2026-06-30T23:45:00"
    },
    "total_windows_generated": 14000,
    "samples_removed_due_to_missing_labels": 3100,
    "removal_reasons_distribution": {
        "unlabeled_ambiguous": 3100
    },
    "final_class_distribution": {
        "TREND": 4900,
        "TRANSITION": 3200,
        "RANGE": 2800
    },
    "dataset_rows": 10900
}
```

---

## Extending with Future Labelers

The `LabelEngine` can be seamlessly extended for other ML training objectives by subclassing `BaseLabeler` in `ML/market_state_labeler.py`:

```python
from ML.market_state_labeler import BaseLabeler

class LevelBreakLabeler(BaseLabeler):
    @property
    def label_version(self) -> str:
        return "1.0.0"

    def label_window(self, df, msg, window_start, window_end):
        # Implement custom deterministic logic for breaks vs rejections
        # Return label ("BREAK", "REJECT", or None), confidence, and info dict
        pass
```
