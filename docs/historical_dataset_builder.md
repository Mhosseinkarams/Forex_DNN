# Production-Grade Centralized ML Data Pipeline

`HistoricalDatasetBuilder` has been refactored into the **ONLY official, production-grade dataset generation pipeline** inside Forex_DNN. It is the centralized data engineering backbone of the entire repository. Every future ML model must obtain its training data from this builder. No model is allowed to generate its own dataset.

---

## 1. Pipeline Architecture

Instead of hardcoding a sequential series of executions, the pipeline is fully decoupled and registry-driven:

```
Historical Data
      │
      ▼
   Pipeline()
      │
      ├── .register(IndicatorEngine())
      ├── .register(MarketStructureEngine())
      ├── .register(SupplyDemandEngine())
      ├── .register(LiquidityEngine()) [Plug-In Ready]
      └── .register(OrderBlockEngine()) [Plug-In Ready]
      │
      ▼
   FeatureRegistry Snapshots
      │
      ▼
   LabelEngine Classification
      │
      ▼
   Dataset Versioning & Quality Reports
      │
      ▼
   Trained Models Registry (with traceability)
      │
      ▼
   Backtesting & Execution
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

## 3. Directory Layout Structure

The project directory structure is formalized to cleanly isolate raw data, processed outputs, cache, models, reports, and experiments:

```
Forex_DNN/
├── raw_data/                 # Raw data feeds
├── processed_data/           # Cleaned/processed bar sequences
├── cache/                    # Intermediate cache Parquet files
├── datasets/                 # Labeled training dataset versions
│   └── v001/
│       ├── dataset.parquet
│       ├── dataset.csv
│       ├── metadata.json
│       ├── feature_registry.json
│       ├── engine_versions.json
│       ├── label_config.json
│       ├── statistics.json
│       ├── manifest.json
│       └── dataset_quality_report.html
├── models/                   # Registry for trained model wrappers
│   ├── MarketState/
│   │   └── v001/
│   │       ├── model.joblib
│   │       └── reproducibility.json
│   └── LevelBreak/
│       └── v001/
├── experiments/              # Research workbenches and temporary notes
├── training_runs/            # Artifacts from training script executions
├── reports/                  # Validation reports and feature registry maps
└── backtests/                # Performance logs of historical tests
```

---

## 4. Dataset Fingerprinting

To guarantee complete dataset traceability and safety, every dataset version contains a unique **fingerprint** inside `metadata.json`, `manifest.json`, and `statistics.json`:

- `dataset_hash`: Deterministic SHA-256 hash of compiled DataFrame content.
- `feature_hash`: Unique hash representing the registered/enabled FeatureRegistry configuration.
- `engine_hash`: Deterministic hash of all pipeline engine version combinations.
- `git_commit`: Traceable git commit hash at compilation time.
- `creation_time`: ISO-8601 UTC timestamp.

---

## 5. Model Reproducibility and Registry

When training models using scripts like `train_market_state.py`, the registry saves companion metadata `reproducibility.json` alongside model wrappers (e.g. `model.joblib`), registering:

- `trained_from_dataset`: Version of dataset directory used (e.g. `v001`).
- `dataset_hash`: Fingerprint of the exact training dataset.
- `git_commit`: Active code version at training time.
- `training_script_version`: training run script tracking identifier.

This guarantees that any trained model can be traced back directly through the pipeline to its exact training dataset version and codebase commit.

---

## 6. Premium Interactive HTML Quality Reports

During generation, the builder automatically compiles `dataset_quality_report.html` under the dataset version directory. It features responsive styling via Tailwind CSS and interactive animations via Chart.js, providing:

- **Metadata Profile**: Version, Row/Column/Feature counts, and dataset hashes.
- **Label Distributions**: Interactive doughnut charts representing classes.
- **Symbol Distributions**: Interactive bar charts of symbol sample sizes.
- **Temporal Horizon Range**: Datetime start/end dates and chronological monotonic checks.
- **Duplicate & Missing Values Profile**: Missing cells count and duplicates percentage.
- **Multi-collinearity Warning Check**: Identifies and reports highly correlated features (Pearson $r > 0.98$).
- **Data Leakage Scans**: Automatically alerts of future leakage columns.
- **Feature Importance Preview**: Horizontal bar charts displaying a quick RandomForest-based top 10 feature importance preview.

---

## 7. Dataset Version Manager

The `DatasetVersionManager` coordinates:
1. **Auto-Increment**: Finds the next available version (e.g. `v001` -> `v002`) and initializes the folder.
2. **Never Overwrite**: Prevents overwriting old datasets to ensure model reproducibility.
3. **Snapshot Files**: Saves all configurations, configurations of the FeatureRegistry (`feature_registry.json`), and rules of the LabelEngine (`label_config.json`) alongside dataset outputs.

---

## 8. Cache & Resume System

Intermediate results are cached per symbol in the `cache/` directory:
- Saved as `cache/{symbol}_{timeframe}_{version}_cache.parquet`
- **Automatic Resume**: If the builder gets interrupted at 80% completion, the next execution reads completed symbols directly from the cache and continues from where it left off, avoiding hours of recomputation.

---

## 9. Symbol-Level Parallelization

To maintain chronological ordering and prevent race conditions:
- We parallelize at the **symbol level**, not the window level.
- Fresh thread-safe engine instances are cloned per worker thread, preventing reentrancy crashes like `ValueError: list modified during sort`.
- We use a thread/process pool to run symbol tasks concurrently.

---

## 10. Future Extension Points & Plug-In Support

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
