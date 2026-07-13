# Machine Learning Workbench Research & Validation Guide

This document describes the design, architecture, and usage of the Machine Learning Research & Validation Workbench (`notebooks/ml_pipeline_research.ipynb`).

The ML Research Workbench is a unified development laboratory designed to prototype, debug, inspect, and evaluate the end-to-end Forex_DNN machine learning workflow on a subset of data (e.g., EURUSD M5) with fast iteration cycles before code is integrated into production training pipelines.

---

## 1. Objectives & Purpose

Before writing production-grade ML pipeline components, it is critical to have an interactive and highly transparent playground where developers can:
- **Inspect** data loads, timestamp integrity, and missing values.
- **Trace** how raw prices are sequential-transformed by the `IndicatorEngine`, `MarketStructureEngine`, and `SupplyDemandEngine` into a `MarketStructureGraph`.
- **Analyze** rule-based label assignments (from `LabelEngine` and `DatasetBuilder`) and observe drop reasons for ambiguous windows.
- **Audit** feature vectors for correctness against the `FeatureRegistry`.
- **Visualize** dataset distributions, correlation structures, and suspicious anomalies (colinearity, zero variance).
- **Evaluate** classification performance and confidence calibration.
- **Perform Error Analysis** via an interactive **Random Sample Explorer** that visualizes historical price windows with overlaid structures, supply/demand zones, predictions, and confidence.

---

## 2. Workbench Architecture & Sections

The workbench is organized into 13 distinct sections, each representing a logical boundary of the machine learning pipeline:

| Section | Phase | Description |
| :--- | :--- | :--- |
| **1** | **Configuration** | Setup directories, target symbol, timeframe, limits, random seeds, and model selection (`MarketStateClassifier` or `LevelBreakProbabilityModel`). |
| **2** | **Load Historical Data** | Load parquet/csv files, output metrics (range, gaps), and raise abort check-guards if data is corrupt. |
| **3** | **HistoricalDatasetBuilder** | Run the existing pipeline stages to compute technical features, swings, and supply/demand zones. Mapped into a shared `MarketStructureGraph`. |
| **4** | **LabelEngine** | Generate rule-based target labels (`TREND`/`RANGE`/`TRANSITION` or binary level breaks) and report class distribution, removals, and plot bar frequencies. |
| **5** | **FeaturePipeline** | Query enabled features dynamically from the `FeatureRegistry` and verify alignment with the generated dataset. |
| **6** | **DatasetCleaner** | Run standard cleaners to drop duplicates, manage NaNs, encode categoricals, and remove constant columns. |
| **7** | **Dataset Inspection** | Generate correlation heatmaps, plot KDE/histogram distributions, and run automated anomaly detection. |
| **8** | **Train/Val/Test Split** | Split data chronologically (70% train, 15% validation, 15% test) to respect temporal causality and verify zero overlap. |
| **9** | **Feature Analysis** | Fit a RandomForest model to extract gini feature importances, listing the top 20 most influential features. |
| **10** | **Train Model** | Fit the target wrapper (`MarketStateClassifier` or `LevelBreakProbabilityModel`) using a LightGBM/RandomForest backend. Track elapsed training time. |
| **11** | **Evaluate Model** | Calculate metrics (Accuracy, Precision, Recall, F1) and plot confusion heatmaps, binary/multiclass ROC-AUC curves, PR curves, and Calibration curves. |
| **12** | **Error Analysis** | Analyze failure profiles, plot prediction confidence, and run the **Random Sample Explorer** for custom candlestick chart overlays. |
| **13** | **Save Outputs** | Package all trained models, configurations, metrics, and plots into a timestamped directory under `output/research_experiments/`. |

---

## 3. The Random Sample Explorer

One of the premium components of the workbench is the **Random Sample Explorer** in Section 12.
To validate model behavior and ensure features/labels make logical sense, the explorer:
1. Picks a random sample index from the chronological test set.
2. Extracts the ISO-format datetime of the sample's window endpoint.
3. Maps that datetime back to the raw historical candle series index, adjusting for timezone offsets or rounding anomalies safely.
4. Slices out the exact preceding 35-candle lookback window.
5. Draws a beautiful candlestick chart of the 35 candles.
6. Overlays **active supply/demand zones** from the `MarketStructureGraph` as shaded transparent bands.
7. Draws dashed horizontal reference lines for **BOS (Break of Structure)** and **CHOCH (Change of Character)** events.
8. Plots the slow/fast **EMA lines** (50, 600, 800) calculated during the pipeline run.
9. Displays the true label, model prediction, and confidence in the title, alongside printing a table of the top 8 feature values.

This visual audit ensures the data pipeline has zero look-ahead bias and that indicators align correctly with trade setups.

---

## 4. Relationship to Production Pipelines & Migration Guide

The ML Research Workbench is the upstream blueprint for future production pipelines. Once code is refined and validated in the notebook, it should be migrated into clean, structured Python modules.

### Future Production Module Target Layout
```
ML/
├── configs/
│   └── default_config.yaml         <-- Migrated from Section 1 (hyperparameters)
├── dataset_builder.py              <-- Migrated from Sections 2, 3, 5, 6
├── label_engine.py                 <-- Migrated from Section 4
├── data_cleaner.py                 <-- Migrated from Section 6
├── trainer.py                      <-- Migrated from Section 10 (fit orchestrator)
├── evaluator.py                    <-- Migrated from Section 11 (metrics & curves generation)
└── analyzer.py                     <-- Migrated from Section 12 (error diagnostics)
```

### Migration Workflow Steps
1. **Config Extraction**: Move hyperparameters from Section 1 to a unified YAML/JSON config file inside `ML/configs/` or `Core/config.py`.
2. **Modular Functions**: Extract helper functions from the notebook (e.g. data validation, index lookups) and place them in utility classes.
3. **Pipeline Orchestrator**: Migrate training loops from Section 10 and 11 into a production-grade script/class (e.g. `ML/trainer.py`), accepting configuration arguments and running headless via CLI.
4. **Automation of Visual Metrics**: Migrate curve plots to headless exporters that save reports automatically (e.g., using `matplotlib` without interactive UI displays).
5. **Continuous Verification**: Keep the research notebook updated so that anytime a developer proposes a new feature group or labeling rule, it is validated in the workbench first, and then compiled into production.
