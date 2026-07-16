# Forex_DNN Repository Clean-Up and Architectural Audit Report

This report documents the architectural clean-up of the Forex_DNN repository. All deletions, moves, and reorganizations are recorded with strong justification to preserve the integrity of the production system.

## 1. Identified Obsolete and Duplicated Files

| File/Directory | Reason | Replacement | Confidence |
| :--- | :--- | :--- | :--- |
| `MarketStructure/` | Outdated/duplicated implementations of Smart Money Concepts (SMC) structure detection. | `Market_Data_Pipeline/structure_engine.py` and `Market_Data_Pipeline/supply_demand_engine.py` | 100% |
| `archive/` | Leftover folder containing legacy, unused preprocessing files (`preproc_single_inout.py`, etc.). | Centralized dynamic features in `ML/feature_pipeline.py`. | 100% |
| `DNN/` | Obsolete and abandoned Deep Neural Network (Keras/TensorFlow/LSTM) prototypes. These are not part of the active, production-grade hybrid architecture. | The LightGBM and RandomForest models under the `ML/` package. | 100% |
| `RL_Approach/` | Unfinished, abandoned experimental Reinforcement Learning approach, not integrated into the active strategy execution flow. | Standard strategy-agnostic `MLDecisionEngine` and rule-based strategy policy integration. | 100% |
| `train_production_pipeline.py` | Obsolete monolithic CLI script. Refactored into unified orchestrators. | Unified orchestrator `train.py` coupled with `Pipeline/training_pipeline.py` and `Training/` scripts. | 100% |
| `tf` | Empty file at the root. | None | 100% |
| `test_data.csv` | Unused raw CSV mock test data file at the root. | Test fixture generation in individual unit tests. | 100% |
| `executed_val.ipynb` | Leftover executed notebook at the root directory. | None (already documented in standard examples). | 100% |
| `validate_label_engine.ipynb` | Scratchpad research notebook at the root. | Verified test suite in `tests/test_ml_decision_engine.py` and `ML/test_label_engine.py`. | 100% |
| `validation_market_structure.ipynb` | Temporary evaluation notebook at the root. | Unit tests in `Market_Data_Pipeline/test_engines.py` and validation notebooks in `examples/`. | 100% |

---

## 2. Unification Directory Tree Design

Following the Lead Software Architect's directives, we are unifying the file/data layout into a scalable, stage-specific structure under the `Data/` directory, while completely separating run state journals and logs.

```text
Forex_DNN/
├── Configs/            # All YAML configuration parameters
├── Data/               # Standardized data directories
│   ├── Historical/     # Multi-timeframe raw historical CSV and Parquet files
│   ├── ML/             # Machine Learning pipeline data artifacts
│   │   ├── Processed/  # Cleaned, structured historical datasets
│   │   ├── Features/   # Intermediate feature maps and serialized dataframes
│   │   ├── Labels/     # Computed label datasets
│   │   ├── Datasets/   # Standardized versioned training Parquets and manifests
│   │   ├── Models/     # Model weight checkpoints and registry metadata
│   │   └── Reports/    # Model evaluation dashboards and reports
│   ├── Cache/          # Dynamic calculation feature caches
│   └── Temporary/      # Transient files (e.g., test scratchpads)
├── Journals/           # Chronological trading journal files (Layer 1 & 2)
├── Logs/               # Operational logs (errors, runtime debug logs)
├── Docs/               # Core framework documentation and guides
├── Scripts/            # Operations scripts
├── Examples/           # Step-by-step notebooks demonstrating core flows
└── Validation/         # Pre-flight and production-readiness validation suites
```

---

## 3. Path Manager Integration Plan

To implement this structural change cleanly and maintain 100% backward compatibility, we introduce `Configs/path_manager.py` (with the helper class `PathManager`). This serves as the single source of truth:
- All core analytical engines, training pipeline scripts, and data downloaders query `PathManager.get_path()` or `PathManager.get_relative_path()`.
- Hardcoded `"output/"`, `"HistoricalData/"`, `"cache/"`, `"models/"`, `"datasets/"` values are eliminated across the system.
