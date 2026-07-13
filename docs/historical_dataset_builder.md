# Production-Grade Centralized ML Data Pipeline

`HistoricalDatasetBuilder` has been refactored into the **ONLY official, production-grade dataset generation pipeline** inside Forex_DNN. It is the centralized data engineering backbone of the entire repository. Every future ML model must obtain its training data from this builder. No model is allowed to generate its own dataset.

---

## 1. Pipeline Architecture

Instead of hardcoding a sequential series of executions, the pipeline is fully decoupled and registry-driven:

```
Historical Data
      ↓
   Pipeline()
      ↓
.register(IndicatorEngine())
      ↓
.register(MarketStructureEngine())
      ↓
.register(SupplyDemandEngine())
      ↓
.register(LiquidityEngine()) [Plug-In Ready]
      ↓
.register(OrderBlockEngine()) [Plug-In Ready]
      ↓
.register(FeatureRegistry())
      ↓
.register(LabelEngine())
      ↓
Unified ML Dataset
```

The pipeline executes these registered stages sequentially, transforming the raw data and producing strongly-typed result containers:

1. **IndicatorEngine**: Computes EMA lines, wilders-based ATR, slopes, distances, and shadow ratios.
2. **MarketStructureEngine**: Detects swing highs/lows, protected levels, BOS, and CHOCH.
3. **SupplyDemandEngine**: Computes ATR-normalized supply and demand zones.
4. **FeatureRegistry / FeaturePipeline**: Performs completely registry-driven feature extraction, utilizing zero feature name knowledge in the builder.
5. **LabelEngine**: Assigns deterministic labels (`TREND`, `RANGE`, `TRANSITION`) based on strategy rules.

---

## 2. Execution Flow

At each sliding window end index, the builder coordinates extracting the features and labels to package them into strongly-typed output models:

```
   SlidingWindow (start_idx, end_idx)
                ↓
      MarketStructureResult
                ↓
       SupplyDemandResult
                ↓
         FeatureVector
                ↓
          LabelResult
                ↓
         DatasetSample (with deterministic sample_id)
```

### Deterministic Sample IDs
Every sample receives a deterministic `sample_id` derived from:
`{symbol}_{timeframe}_{window_end_datetime}`
Example: `EURUSD_M5_2019-07-15T13:35`

This guarantees exact reproducibility and makes error tracing or auditing highly straightforward.

---

## 3. Directory Structure

Versioned outputs and snapshots are organized under the `datasets/` directory:

```
datasets/
├── v001/
│   ├── dataset.parquet
│   ├── dataset.csv
│   ├── metadata.json
│   ├── feature_registry.json
│   ├── engine_versions.json
│   ├── label_config.json
│   ├── statistics.json
│   └── manifest.json
├── v002/
│   └── ...
```

---

## 4. Dataset Version Manager

The `DatasetVersionManager` coordinates:
1. **Auto-Increment**: Finds the next available version (e.g. `v001` -> `v002`) and initializes the folder.
2. **Never Overwrite**: Prevents overwriting old datasets to ensure model reproducibility.
3. **Snapshot Files**: Saves all configurations, configurations of the FeatureRegistry (`feature_registry.json`), and rules of the LabelEngine (`label_config.json`) alongside dataset outputs.

---

## 5. Cache & Resume System

Intermediate results are cached per symbol in the `cache/` directory:
- Saved as `cache/{symbol}_{timeframe}_{version}_cache.parquet`
- **Automatic Resume**: If the builder gets interrupted at 80% completion, the next execution reads completed symbols directly from the cache and continues from where it left off, avoiding hours of recomputation.

---

## 6. Symbol-Level Parallelization

To maintain chronological ordering and prevent race conditions:
- We parallelize at the **symbol level**, not the window level.
- Fresh thread-safe engine instances are cloned per worker thread, preventing reentrancy crashes like `ValueError: list modified during sort`.
- We use a thread/process pool to run symbol tasks concurrently.

---

## 7. Progress Monitor

We replace basic print logs with a premium, real-time `tqdm` monitor that displays:
- **Current Symbol**
- **Progress %**
- **Windows/sec / symbols/sec**
- **ETA**
- **Memory Usage (via `psutil`)**
- **Current Dataset Size (in samples)**

---

## 8. Data Validation & Quality Reports

The expanded `DatasetValidator` performs deep analytical tests:
- **Null / Inf values**
- **Duplicate rows / IDs / timestamps**
- **Missing labels**
- **Constant or low-variance columns**
- **Class / Symbol imbalance**
- **Feature correlation (Pearson r > 0.98)**
- **Outlier detection (Z-score > 5.0)**
- **Lookahead/Leakage detection**

It automatically generates a complete `statistics.json` quality report inside each version directory, covering:
```json
{
    "Rows": 150000,
    "Columns": 76,
    "Memory_MB": 12.45,
    "Features": 50,
    "Classes": 3,
    "Label_Distribution": {
        "RANGE": 85000,
        "TRANSITION": 45000,
        "TREND": 20000
    },
    "Missing_Percentage": 0.0,
    "Generation_Time_Sec": 154.21,
    "Average_Windows_Per_Sec": 972.1,
    "Dataset_Size_Bytes": 13054800,
    "Largest_Symbols": [["XAUUSD", 75000]],
    "Smallest_Symbols": [["YM", 75000]]
}
```

---

## 9. Future Extension Points & Plug-In Support

Adding a new technical or Smart Money Concept engine to the Forex_DNN framework requires **zero modifications** to `HistoricalDatasetBuilder`.

To add an engine, simply register it to the builder's pipeline:
```python
builder = HistoricalDatasetBuilder(...)

# Plug-In Ready
builder.register_engine(LiquidityEngine())
builder.register_engine(OrderBlockEngine())
builder.register_engine(FairValueGapEngine())

# Execute
df_final, metadata = builder.build_dataset()
```

The builder automatically executes registered engines sequentially during DataFrame transformation and forwards their computed output columns to `FeatureRegistry` and `FeaturePipeline` for extraction.
