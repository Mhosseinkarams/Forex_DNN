# Production-Grade Large-Scale Training Pipeline Guide

This guide details the design, architecture, API, and usage of the Forex_DNN Production-Grade Large-Scale Training Pipeline (`train_production_pipeline.py`).

The training pipeline is engineered as a highly robust, scalable, memory-efficient, and transparent system for processing multiple historical symbol datasets, performing rigorous pre-training data diagnostics, executing chronological splits, training state-of-the-art LightGBM/RandomForest models, and generating detailed diagnostic reports.

---

## 1. System Architecture

The end-to-end workflow decouples data collection, dataset construction, data cleaning, model training, evaluation, and experiment archiving.

```text
HistoricalData/
  ├── EURUSD/M5.parquet
  ├── GBPUSD/M5.parquet
  └── XAUUSD/M5.parquet
         │
         ▼
[1. Data Discovery] ──► Automatically finds CSV / Parquet files per symbol and timeframe
         │
         ▼
[2. Parallel/Serial Loop] ──► Processes one symbol at a time to minimize memory (RAM) usage
         │
         ├──► [Indicators, SMC Structure, Supply/Demand Engines] ──► Enrich price data
         ├──► [MarketStructureGraph] ──► Builds spatial, point-in-time state
         ├──► [Sliding Window Labeling] ──► LabelEngine (MarketState) or DatasetBuilder (LevelBreak)
         ├──► [Data Cleaner] ──► Drops duplicates, fills NaNs, fixes negative spreads/ATR
         ├──► [Local Disk Caching] ──► Saves symbol parquet to prevent redundant computation
         └──► [Symbol Statistics Report] ──► Saves granular BOS/CHOCH and zone counts
         │
         ▼
[3. Dataset Consolidation] ──► Combines symbol datasets and sorts chronologically
         │
         ▼
[4. Dataset Diagnostics] ──► DatasetValidator runs rigorous pre-training health checks
         │
         ├───► PASS ──► Continue to model training
         └───► FAIL ──► Abort training to prevent look-ahead bias, leakage, or bad data
         │
         ▼
[5. Chronological Split] ──► Splits consolidated master dataset: 70% Train, 15% Val, 15% Test
         │
         ▼
[6. Model Training & Calibration] ──► Fits LightGBM / RF; calibrates probabilities using Platt scaling
         │
         ▼
[7. Multi-Market Generalization Evaluation]
         │
         ├───► Global Test Split Evaluation (Accuracy, Precision, Recall, F1, ROC, PR, Calibration curves)
         └───► Per-Symbol Test Evaluation (measures cross-market model generalization)
         │
         ▼
[8. Experiment Archiving] ──► Saves model.joblib, reports, logs, and plots into a timestamped directory
```

---

## 2. API Reference & CLI Options

The production training script is executed via `train_production_pipeline.py`. It includes a complete command-line interface.

### Options:
* `--model-name`: Name of the model wrapper to train. Supported:
  * `MarketStateClassifier` (Default): multi-class regime mapping (`TREND`=0, `RANGE`=1, `TRANSITION`=2).
  * `LevelBreakProbabilityModel`: binary support level break forecasting (`REJECT`=0, `BREAK`=1).
* `--model-type`: Backend machine learning model. Supported: `lightgbm` (Default), `randomforest`.
* `--data-dir`: The root folder containing historical symbol subdirectories. Defaults to `HistoricalData`.
* `--symbol`: Specify exact symbols to process as a comma-separated list (e.g. `EURUSD,GBPUSD,XAUUSD`), or pass `all` to automatically discover and process all symbols. Defaults to `all`.
* `--timeframe`: Timeframe file name to discover (e.g., `M5`, `M15`, `H1`). Defaults to `M5`.
* `--window-size`: Sliding window size (number of candles). Defaults to `35`.
* `--window-stride`: Sliding window stride. To compute full datasets, use `1` (Default). For fast development runs, use higher strides (e.g., `50`).
* `--cache-dir`: Directory where intermediate symbol datasets are stored as Parquet files. Defaults to `cache`.
* `--output-dir`: Main output folder. Defaults to `output`.
* `--seed`: Random seed for model training reproducibility. Defaults to `42`.
* `--force`: Force model training even if critical diagnostics warnings exist.

### Usage Examples:

```bash
# 1. Train production MarketStateClassifier on all discovered M5 symbols using LightGBM
python train_production_pipeline.py --model-name MarketStateClassifier --timeframe M5

# 2. Train LevelBreakProbabilityModel using RandomForest on EURUSD and GBPUSD with M15 files
python train_production_pipeline.py --model-name LevelBreakProbabilityModel --symbol EURUSD,GBPUSD --timeframe M15 --model-type randomforest

# 3. Quick test run with high stride (50) to verify everything runs instantly
python train_production_pipeline.py --model-name MarketStateClassifier --window-stride 50 --force
```

---

## 3. Advanced Features

### 3.1 Memory-Efficient Caching & Resuming
Large-scale sliding window operations can consume significant RAM and time when processing multiple years of data across many symbols.
* **Symbol Isolation**: The pipeline processes only **one symbol at a time**.
* **Disk Caching**: Cleaned datasets are saved as Parquet files under `cache/` (e.g., `processed_EURUSD_M5_w35_s1_MarketStateClassifier.parquet`).
* **Interrupted Runs**: If processing is interrupted, running the command again will instantly reload previously completed symbols from the cache, resuming exactly where it left off.

### 3.2 Pre-Training Diagnostics (Dataset Diagnostics)
Before running fitting routines, the `DatasetValidator` performs rigorous checks on the consolidated master dataset:
1. **Label Balance**: Verifies that no class represents less than 1% of the dataset.
2. **Missing Labels/Features**: Checks for NaN or infinite cells in features or target.
3. **Constant Columns**: Finds columns with 0 variance which can cause model bloat or training degradation.
4. **Collinearity**: Flag features with a Pearson correlation coefficient `r > 0.98`.
5. **Duplicate Samples**: Detects identical rows across the dataset.
6. **Future Leakage**: Scans features for lookahead markers or names containing future patterns.
7. **Chronological Ordering**: Enforces that datetimes monotonically increase within symbols to prevent temporal leakage.

*If any check fails critical rules, the script prints error logs and **aborts training** to protect you from deploying a model trained on contaminated data.*

### 3.3 Multi-Market Cross-Market Generalization
A major pitfall of quantitative trading models is overfitting to a single currency pair's unique characteristics. To combat this:
* After fitting the global model, the script splits the **chronological test set** by symbol.
* It evaluates metrics (Accuracy, F1-Score, and Confusion Matrix) for each symbol individually.
* This allows quantitative researchers to directly measure and verify **cross-market generalization** and detect if a model performs poorly on specific currency pairs.

---

## 4. Experiment Tracking Directory Schema

Every successful run creates a separate, non-overlapping timestamped folder under `output/experiments/YYYYMMDD_HHMMSS/` containing:

```text
output/experiments/YYYYMMDD_HHMMSS/
  ├── config.json                 # Model and feature metadata configuration
  ├── training.log                # Copy of the console logs for absolute transparency
  ├── metrics.json                # Global and per-symbol evaluation metrics (Accuracy, F1, ROC)
  ├── dataset_summary.json        # Global dataset statistics, distributions, and footprints
  ├── per_symbol_statistics.json  # Consolidated per-symbol BOS/CHOCH and S/D zone counts
  ├── feature_importance.csv       # Relative feature importance ranking
  ├── classification_report.txt   # Plain-text global classification metrics report
  ├── confusion_matrix.png        # Plot of predictions vs actuals
  ├── roc_curve.png               # ROC-AUC curve (binary or multiclass per-class ROC)
  ├── precision_recall.png        # Precision-Recall curve
  ├── calibration_curve.png       # Calibration curve (Platt Scaling probability audit)
  ├── shap_summary.png            # SHAP value feature importance summary plot (if shap is installed)
  ├── model.joblib                # Fully serialized calibrated BaseTradingModel wrapper
  ├── model_metadata.json         # Direct metadata companion JSON file
  ├── <SYMBOL>_statistics.json    # Individual per-symbol statistics JSON reports
  ├── label_distribution.png      # Global label counts bar chart
  └── symbol_distribution.png     # Samples per symbol counts bar chart
```

---

## 5. Troubleshooting & Metrics Interpretation

### 5.1 Diagnostics Failed and Training Aborted
* **Symptom**: Console logs show `Diagnostics Audit status: FAIL` and script exits.
* **Reason**: Critical problems such as missing target labels, duplicate sample IDs, or future leakage were detected.
* **Resolution**: Inspect the terminal output to identify the failing check. If you are doing quick testing and want to bypass this check, pass the `--force` flag.

### 5.2 Missing SHAP Plots
* **Symptom**: `shap_summary.png` contains a placeholder.
* **Reason**: The `shap` package is not installed in the environment.
* **Resolution**: Install SHAP using `pip install shap`. The pipeline will automatically detect it and generate beautiful SHAP summary plots on the next run.

### 5.3 Missing Volatility or Session breakdowns
* **Symptom**: `dataset_summary.json` has empty dictionaries for session or volatility.
* **Reason**: Columns like `hour` or `atr_percentile` are missing from the dataset.
* **Resolution**: Ensure that indicators are fully calculated by the default `IndicatorEngine`. They are built by default, but if you modified the registry or feature configs, make sure these base features are enabled.
