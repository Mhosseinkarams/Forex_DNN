# Forex_DNN: Decentralized, Market-Driven Algorithmic Research & Trading Framework

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](#)
[![MT5 Integration](https://img.shields.io/badge/MetaTrader5-Compatible-green.svg)](#)

---

## 1. Project Overview

**Forex_DNN** is not just a trading bot or a simple script that executes buy and sell signals. It is a **production-grade, complete quantitative research and automated trading platform** designed for professional algorithmic traders, quantitative developers, and machine learning researchers.

Most retail trading bots are built as rigid, single-threaded scripts where a single module downloads data, calculates indicators, executes logic, and handles orders. If the connection fails or the logic changes, the entire system crashes, leaving open positions untracked.

**Forex_DNN** solves this by decoupling the entire trading lifecycle into a **centralized, deterministic, market-driven pipeline**. It is designed with the same design philosophies as industry-standard libraries like TensorFlow, Backtrader, or QuantConnect:

- **Centralized Market Structure Graph**: No individual strategy or module directly calculates trends, swings, or key zones. Instead, high-performance, stateless analytical engines build a spatial object model—the `MarketStructureGraph`—which serves as the single source of truth for the entire framework.
- **Smart Money Concepts (SMC)**: Automated detection of Market Structure Shifts, Breaks of Structure (BOS), Change of Character (CHOCH), and Protected Highs/Lows.
- **Supply & Demand Detection**: Institutional order blocks and supply/demand zones are automatically identified using ATR-normalized impulsive moves and tracked through their entire lifecycle (fresh, mitigated, and broken).
- **Centralized Feature Registry**: A schema-driven ML pipeline that manages feature extraction, normalization, look-ahead protection, and configuration hashing, preventing data leakage during training and production.
- **Hybrid Rule-Based + Machine Learning Architecture**: Advanced classifiers predict market regimes and level break probabilities to filter and confirm high-probability structural setups.
- **Decoupled Execution & Risk Systems**: Centralized position trackers, exit managers, and profile-driven drawdown systems guarantee risk preservation and support multi-stage trailing exits.
- **Forensic Post-Trade Auditing**: Reconstructs the complete historical lifecycle of any trade from raw broker events, journals, and state files to guarantee data integrity and eliminate slippage.
- **Dual Visualizers**: High-fidelity chart visualization in Jupyter notebooks using Matplotlib and asynchronous passive chart overlays in real-time MetaTrader 5 using the `.mq5` renderer indicator.

---

## 2. Repository Architecture

The repository is highly modular and strictly organized to separate data ingestion, analytical feature generation, execution management, simulation, and post-trade auditing.

```text
Forex_DNN/
├── Collecting_Data/        # Data ingestion, technical indicators, authentication, and logging
│   ├── auth.py             # Broker credentials loading and MT5 authorization
│   ├── data_feed.py        # Unified interface for live data ingestion and polling
│   ├── indicators.py       # Optimized IndicatorEngine (EMA, ATR, slopes, deduplication)
│   ├── logging_config.py   # Hierarchical logging configuration
│   ├── position_lifecycle.py# Centralized dataclasses representing a trade lifecycle
│   ├── trading_journal.py  # Layer 1 (Chronological Events) & Layer 2 (Summaries) logger
│   └── utils.py            # Atomic file IO and mathematical helpers
│
├── Market_Data_Pipeline/   # Deterministic Market Intelligence & Spatial Modeling Layer
│   ├── structure_graph.py  # MarketStructureGraph object-oriented spatial model
│   ├── structure_engine.py # Smarts Money Concepts swing, BOS, CHOCH, and protected levels detector
│   ├── supply_demand_engine.py # Impulsive ATR-normalized supply/demand zone tracker
│   ├── state_engine.py     # MarketStateEngine for identifying Trend, Range, and Transition
│   └── feature_pipeline.py # Point-in-time lookup and feature vector generation
│
├── ML/                     # Machine Learning Subsystem
│   ├── feature_registry.py # Centralized registry for feature schemas, versioning, and hashing
│   ├── feature_definition.py# Base abstract definition class for features
│   ├── feature_groups.py   # Implementations of technical, structural, and state features
│   ├── dataset_builder.py  # Constructs clean Market State and Level Break datasets
│   └── data_cleaner.py     # Normalizes values, handles missing rows, generates quality reports
│
├── Strategies/             # Quant Strategy Implementations
│   └── mm_strategy.py      # MMStrategy (Standard, High-Risk, Reversal intraday Forex signals)
│
├── Trade_Execution/        # Decoupled Execution, Tracking, and Risk Systems
│   ├── drawdown.py         # DrawdownManager enforcing daily/total absolute risk budgets
│   ├── exit_manager.py     # Centralized multi-stage ExitManager (TP1, TP2, Trailing SL)
│   ├── location_engine.py  # TradeLocationEngine resolving structural entry, SL, and TP levels
│   ├── position_manager.py # Environment-agnostic broker orders (Open, Close, Modify)
│   ├── position_tracker.py # Real-time position monitor and risk calculator
│   ├── risk_sizing.py      # PositionSizer utilizing tick-based risk formulas
│   ├── send_order.py       # Pre-trade conflict resolution and risk constraint verification
│   └── trade_auditor.py    # Forensic batch reconstruction of completed trades
│
├── simulation/             # High-Performance Backtesting & Simulation Package
│   ├── historical_data_feed.py# Multi-timeframe synchronized historical CSV reader
│   ├── simulation_account.py  # Tracks virtual cash, balance, equity, margin, and leverage
│   ├── simulation_broker.py   # Emulates MT5 fill behavior, spread, commission, and active SL/TP hits
│   ├── simulation_clock.py    # Global chronological backtest event clock
│   ├── simulation_environment.py# Singleton library hijacking sys.modules['MetaTrader5']
│   ├── simulation_order_engine.py# Translates deal executions into simulated broker actions
│   ├── simulation_runner.py   # Orchestrator of historical tests with stale state clearing
│   └── statistics_engine.py   # Performance tracker (Win Rate, Profit Factor, Expectancy)
│
├── Visualization/          # Multi-Platform Debugging & Visualization Engine
│   ├── chart_annotator.py  # ChartAnnotationEngine constructing DrawInstruction layers
│   ├── draw_instruction_writer.py# Atomic, independent multi-layer CSV writer for MT5 polling
│   ├── render_types.py     # Strongly-typed models for visual annotations (line, zone, signal, etc.)
│   ├── debug_config.py     # Interactive layer activation and color map configs
│   ├── FX_DNN_Chart_Renderer.mq5# MT5 Expert Advisor/Indicator rendering layers asynchronously
│   └── visualization_examples.ipynb# Matplotlib research visualizer
│
├── Validation/             # Local automated validation and test suites
│   └── validation_report.txt# Test execution and environment status report
│
├── Models/                 # Directory to store finalized trained ML model weights (*.joblib, *.h5)
│
├── State/                  # Atomic persistence files for fail-safe crash and restart recovery
│
├── Journals/               # File outputs of Layer 1 (Events) and Layer 2 (Summaries) logs
│
├── Data/                   # Directory to store imported historical CSV files
│
├── Examples/               # Step-by-step notebooks demonstrating core flows
│   ├── backtest_eurusd_last_year.ipynb # Historical simulation example
│   └── ml_research_workbench.ipynb     # Interactive ML engineering and modeling notebook
│
├── main.py                 # Core CLI entrypoint for backtests, demo, and live execution
├── live_validation.py      # Live pre-flight sanity check and production readiness suite
├── integration_validation.py# End-to-end simulation environment integration verification test
├── train_market_state.py   # Command-line training pipeline for MarketStateClassifier
├── train_level_break.py    # Command-line training pipeline for LevelBreakProbabilityModel
├── debug_config.json       # Visual overlay toggles config
├── requirements.txt        # Full framework python packages
└── requirements_no_mt5.txt # Package list optimized for non-Windows/CI/Mac environments
```

---

## 3. Complete Workflow

To utilize the framework's full capabilities, operations must follow a strict chronological sequence. This workflow enforces time-determinism, look-ahead protection, and strict risk guidelines.

```text
    +--------------------------------------------------------------+
    |                     HISTORICAL DATA INGESTION                 |
    +--------------------------------------------------------------+
                                   |
                                   v  (EURUSD_M5.csv, EURUSD_M15.csv)
    +--------------------------------------------------------------+
    |                     MARKET INTELLIGENCE PIPELINE              |
    +--------------------------------------------------------------+
                                   |
         +-------------------------+-------------------------+
         |                                                   |
         v                                                   v
  [MarketStructureEngine]                             [SupplyDemandEngine]
  - SMC Swing Identification                          - Impulsive Move Detection
  - CHOCH & BOS Isolation                            - Zone Mitigation & Breaches
         |                                                   |
         +-------------------------+-------------------------+
                                   |
                                   v  (Centralized MarketStructureGraph)
    +--------------------------------------------------------------+
    |                    centralized ML pipeline                   |
    +--------------------------------------------------------------+
                                   |
                                   v  (Registry-Driven Feature Vectors)
                     [FeatureRegistry & FeaturePipeline]
                                   |
         +-------------------------+-------------------------+
         |                                                   |
         v (Market State Labels)                             v (Lookahead Zone Breaks)
    [DatasetBuilder]                                    [DatasetBuilder]
  - build_market_state_dataset                        - build_level_break_dataset
         |                                                   |
         v (clean, normalize)                                v (clean, normalize)
  [DataCleaner]                                       [DataCleaner]
         |                                                   |
         v (market_state_dataset.csv)                        v (level_break_dataset.csv)
    +--------------------------------------------------------------+
    |                      MACHINE LEARNING MODELS                 |
    +--------------------------------------------------------------+
         |                                                   |
         v (python train_market_state.py)                    v (python train_level_break.py)
  [MarketStateClassifier]                             [LevelBreakProbabilityModel]
  - Regime Classifier (LightGBM)                      - Probability Estimator (LightGBM)
  - Saves: market_state_classifier.joblib             - Saves: level_break_probability.joblib
         |                                                   |
         +-------------------------+-------------------------+
                                   |
                                   v (Model Weights Loaded)
    +--------------------------------------------------------------+
    |                   BACKTESTING, OPTIMIZATION & RUN            |
    +--------------------------------------------------------------+
                                   |
                                   v (SimulationRunner with historical CSV data)
                           [SimulationRunner]
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
          [SimulationBroker]               [MMStrategy & Models]
          - emulates slippage, commission  - passive prediction overlays
          - tracks active SL & TP hits     - structural SL/TP trade execution
                    |                             |
                    +--------------+--------------+
                                   |
                                   v (Appends chronological files)
                           [TradingJournal]
                                   |
         +-------------------------+-------------------------+
         |                                                   |
         v (reconstructs positions)                         v (calculates performance)
   [TradeAuditor]                                      [StatisticsEngine]
 - PositionLifecycle reconstruction                  - Wins, Losses, Drawdowns
 - Signal consistency validation                     - Profit Factor, expectancy
         |                                                   |
         +-------------------------+-------------------------+
                                   |
                                   v (Parameter Fine-tuning)
    +--------------------------------------------------------------+
    |                     VALIDATION, DEMO & LIVE                  |
    +--------------------------------------------------------------+
                                   |
         +-------------------------+-------------------------+
         |                                                   |
         v (Local/CI check)                                  v (Credentials configuration)
  [integration_validation.py]                        [live_validation.py]
 - Verifies mock interfaces                          - Verifies real broker connection
 - End-to-end sanity checking                        - Checks account balance and spread
         |                                                   |
         +-------------------------+-------------------------+
                                   |
                                   v (TRADING_MODE=demo main.py)
                           [Demo Trading]
                           - Safety limit enforcement
                           - Virtual state recovery validation
                                   |
                                   v (TRADING_MODE=live main.py)
                           [Live Production]
                           - Real execution, sub-second polling
                           - Multi-stage trailing exit monitoring
                                   |
                                   v
                           [Trade Auditor]
                           - Slippage auditing & discrepancy check
                                   |
                                   v
                     [Continuous Model Improvement]
```

---

## 4. First Time Setup

Follow these precise steps to establish a clean execution environment.

### 4.1 Step 1: Install Python
Ensure **Python 3.12** is installed. Do not use Python 3.13 or newer, as standard dependencies such as TensorFlow or LightGBM may not have pre-compiled wheels.
Verify your installation:
```bash
python --version
```

### 4.2 Step 2: Establish Virtual Environment
Create and activate an isolated Python environment to prevent dependency version collisions.

**On Linux/macOS:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4.3 Step 3: Install Required Libraries
For environments with a functional MetaTrader 5 terminal (typically Windows):
```bash
pip install -r requirements.txt
```

For environments without MT5 (macOS, Linux, or CI runners where you only perform backtesting, ML dataset building, and model training):
```bash
pip install -r requirements_no_mt5.txt
```

### 4.4 Step 4: MT5 Installation & Terminal Prep (Windows Only)
1. Download and install your preferred broker's MetaTrader 5 terminal.
2. Open MetaTrader 5.
3. Click **Tools -> Options** from the top menu, go to the **Expert Advisors** tab:
   - Check **Allow algorithmic trading**.
   - (Optional for external integrations) Check **Allow WebRequest for listed URLs**.
4. Log into your Demo or Live account inside the terminal. Keep the terminal running.

### 4.5 Step 5: Configure Credentials
Create a file named `credentials.json` directly in the root directory. This contains sensitive auth credentials (do not commit this file to git).
```json
{
  "mt5": {
    "login": 12345678,
    "password": "YourSecurePassword Here",
    "server": "Broker-DemoServerName"
  }
}
```

### 4.6 Step 6: Verify MT5 and Environment Connection
Run the validation script to verify module structures, environment configurations, and mock integrity:
```bash
python integration_validation.py
```
Expected Output:
```text
========================================
   FRAMEWORK INTEGRATION VALIDATION REPORT
========================================
MT5 Connection            : PASS
Authentication            : PASS
Module Initialization     : PASS
...
OVERALL STATUS: PASS
========================================
```

---

## 5. Collect Historical Data

The framework features a professional, highly robust historical data downloader located at `Collecting_Data/historical_data_collector.py`. It is the official utility for downloading historical bar data from multiple sources with high performance, failure resistance, and automatic formatting.

### 5.1 Architecture & Features
The downloader is designed with a provider-based abstraction layer, isolating downstream analytical and machine learning pipelines from raw data providers:
- **Provider Abstraction**: Supports MT5, Dukascopy, OANDA, CSV import, and Mock providers. Downstream systems do not know where data originates, allowing seamless switching of backends.
- **Fail-Safe Chunked Downloading**: Downloads large datasets in configurable chunk-increments (e.g., 180 days) instead of requesting massive ranges in a single API call, reducing terminal lag and connection drops.
- **Interruption Resume**: Automatically writes download checkpoints to `HistoricalData/download_state.json`. If a download is interrupted, it will resume exactly where it left off on subsequent executions.
- **On-the-fly Validation**: Validates every downloaded chunk for duplicate rows, missing intervals, invalid OHLC values (High < Low, etc.), negative volumes, and timezone alignment, logging detailed diagnostics.
- **Parallel Execution**: Uses a high-performance multithreading pool (configurable, default 4 workers) to download and process multiple symbols concurrently.
- **Dual Serialization**: Saves outputs as high-performance **Parquet** files (for rapid ML processing) and standard **CSV** files.
- **Metadata & Reporting**: Produces a `metadata.json` for every symbol containing broker/timezone info, bar count, duplicate/invalid count, and estimated missing candles, as well as a global `download_report.csv` summary.

### 5.2 Command-Line Usage
Run the collector from the repository root:

```bash
# Download M5 data for EURUSD, GBPUSD, and XAUUSD from MT5
python Collecting_Data/historical_data_collector.py \
    --timeframe M5 \
    --start 2010-06-01 \
    --end auto \
    --symbols EURUSD,GBPUSD,XAUUSD \
    --format parquet \
    --workers 3

# Run a mock download offline to test pipeline connectivity (works on Linux/macOS without MT5)
python Collecting_Data/historical_data_collector.py \
    --provider mock \
    --symbols EURUSD,GBPUSD,XAUUSD \
    --start 2026-06-01 \
    --end 2026-07-01 \
    --format both \
    --workers 3
```

### 5.3 CLI Parameters
| CLI Flag | Default Value | Options / Description |
| :--- | :--- | :--- |
| `--timeframe` | `M5` | `M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1` |
| `--start` | `2010-06-01` | Start date in `YYYY-MM-DD` |
| `--end` | `auto` | End date in `YYYY-MM-DD` or `auto` (current candle) |
| `--symbols` | `all` | Specific symbols (e.g., `EURUSD,GBPJPY`) or filters (`forex`, `metals`, `indices`, `crypto`, `all`) |
| `--provider` | `mt5` | `mt5`, `dukascopy`, `oanda`, `csv`, `mock` |
| `--format` | `parquet` | `parquet`, `csv`, `both` |
| `--output-dir` | `HistoricalData` | Output storage folder |
| `--chunk-size-days` | `180` | Int. Chunk query span in days |
| `--workers` | `4` | Concurrency thread count |

### 5.4 Storage Structure
All downloaded history is saved in a structured dataset immediately readable by the `FeaturePipeline`, `LabelEngine`, and backtesters:
```text
HistoricalData/
├── EURUSD/
│   ├── M5.parquet       # High-speed ML binary format
│   ├── M5.csv           # Human-readable format
│   └── metadata.json    # Symbol specific execution & continuity report
├── GBPUSD/
├── download_state.json  # Checkpoint resume JSON file
└── download_report.csv  # Global downloader execution logs
```

### 5.5 Guidelines & Recommendations
- **Timeframe Configurations**: Make sure to download corresponding symbols on **both** M5 and M15 timeframes. `MMStrategy` evaluates both M15 (for high-level swing structure) and M5 (for entry candle indicators and micro trends).

---

## 6. Build Market Structure

Before feeding raw market prices to models or strategy rules, the framework parses prices into a deterministic spatial object model using decoupled engines.

```text
               Raw Candle Price DataFrame
                           |
                           v
              [MarketStructureEngine]
         - Isolates Swing Highs and Swing Lows
         - Identifies Structure Breaches (BOS)
         - Identifies Character Shifts (CHOCH)
                           |
                           +------------------------+
                           |                        |
                           v                        v
                [SupplyDemandEngine]          [MarketStateEngine]
           - Identifies impulsive candles    - Classifies current regime
           - Tracks mitigation events         - TREND, RANGE, TRANSITION
           - Isolates zone breakages                |
                           |                        |
                           +------------------------+
                           |
                           v
              Shared MarketStructureGraph
```

### 6.1 The Analytical Engines

#### MarketStructureEngine
Calculates market structural nodes point-by-point.
- **Swing Points**: Identifies swing highs/lows using configurable lookback windows (e.g., standard left-strength=5, right-strength=5).
- **Protected Levels**: Marks the highest high or lowest low of an established swing leg as a "Protected High" or "Protected Low."
- **BOS & CHOCH**: Evaluates when a candle close breaches the previous swing point. A breach in the direction of the dominant trend is marked as a **Break of Structure (BOS)**; a trend-reversal breach is marked as a **Change of Character (CHOCH)**.

#### SupplyDemandEngine
Tracks institutional order blocks.
- **Zone Creation**: An impulsive, ATR-normalized move (where body size is at least $X \times \text{ATR}$) forms a Supply zone (top of the move) or a Demand zone (bottom of the move).
- **Mitigation**: Tracks when subsequent prices touch the boundary of the zone.
- **Breakage**: When a candle close breaches the opposite boundary, the zone is flagged as broken.

### 6.2 Running and Visualizing Structure
You can interactively visualize detected levels, zones, and swings using the Jupyter Research Workbench.

Run Jupyter:
```bash
jupyter notebook
```
Open **`examples/research_workbench.ipynb`**. Execute the cells to inspect the step-by-step structural output and render matplotlib plots.

---

## 7. Build ML Features

To train robust classifiers, raw indicators and structures are processed through a lookahead-protected feature generation system.

### 7.1 Schema Definitions: FeatureRegistry & FeaturePipeline
- **FeatureRegistry**: Functions as the central schema manager (`ML/feature_registry.py`). It defines exactly what features are enabled, their version, category, and dependencies.
- **FeaturePipeline**: Point-by-point feature generator. It consumes the `MarketStructureGraph` and the indicator DataFrame to assemble unified vectors. Because features are resolved strictly at index $t$, **look-ahead bias is mathematically impossible**.

### 7.2 Building and Cleaning Training Datasets
The primary entrypoint and single source of truth for constructing machine learning datasets is the **`HistoricalDatasetBuilder`** (`Market_Data_Pipeline/historical_dataset_builder.py`). This production-grade offline pipeline converts raw historical OHLCV data into versioned, fully validated ML datasets across symbols in parallel, serving as the only authorized data generator for all models.

For every bar, the builder:
1. Generates a sliding window (default: 35 candles).
2. Executes technical indicator calculations (`IndicatorEngine`).
3. Runs structural engines (`MarketStructureEngine` and `SupplyDemandEngine`).
4. Queries the central `FeatureRegistry` for all active features.
5. Invokes the `LabelEngine` to determine the objective training label (e.g., `TREND`, `RANGE`, `TRANSITION` from `MarketStateLabeler`), dropping any indeterminate/unlabeled samples.
6. Assembles one unified, scale-invariant feature vector along with raw OHLCV and EMAs inside metadata columns for research & plotting.

#### Key Features
- **Parallel Multi-Symbol Processing**: Built on `concurrent.futures`, enabling high-performance parallel generation across 30+ symbols.
- **Robust Versioning Pattern**: Automatically names files sequentially (e.g., `dataset_v001.parquet`, `dataset_v001.csv`, `dataset_v001_metadata.json`) to prevent overwrites and guarantee 100% reproducibility.
- **Data Quality and Integrity Auditing**: Detects and reports missing values, duplicate rows, duplicate timestamps, class/symbol imbalances, and constant features with zero variance.

#### Command-Line and Programmatic Usage
Verify the configurations of the Feature Registry first:
```bash
# Verify the configuration of the Feature Registry
python -m unittest ML/test_feature_registry.py
```

To run the unified dataset generation, invoke the `HistoricalDatasetBuilder` programmatically or write a simple script:

```python
from Market_Data_Pipeline.historical_dataset_builder import HistoricalDatasetBuilder

builder = HistoricalDatasetBuilder(
    input_dir="HistoricalData",
    output_dir="output",
    window_size=35,
    window_stride=1,
    timeframe="M5"
)
df, metadata = builder.build_dataset(max_workers=4)
```

This generates and saves optimized training artifacts under `output/`:
- `output/dataset_v001.parquet` (compressed training samples)
- `output/dataset_v001.csv` (CSV copy for manual inspection)
- `output/dataset_v001_metadata.json` (comprehensive manifest containing class distribution, feature variance, constant columns, elapsed time, and and model config hash verification)

---

## 8. Train ML Models

We provide two production-ready baseline machine learning models designed to filter and qualify trading signals.

```text
Raw CSV / Graph -> [DatasetBuilder] -> Cleaned CSV -> [train_market_state.py] -> market_state_classifier.joblib
                                                   -> [train_level_break.py]  -> level_break_probability.joblib
```

---

### 8.1 MarketStateClassifier

- **Purpose**: Identifies the current market regime (`TREND`, `RANGE`, or `TRANSITION`). This ensures trend strategies are disabled during range periods, and range strategies are active only when appropriate.
- **Inputs**: 49 features including EMA slopes, ATR ratios, distance from swing points, and trend age.
- **Outputs**: Multi-class probability vectors and predictions.

To execute training:
```bash
python train_market_state.py --dataset output/market_state_dataset.csv --model output/market_state_classifier.joblib
```

**Expected Console Output:**
```text
==================================================
      TRAINING MARKET STATE CLASSIFIER MODEL
==================================================
Loaded dataset containing 1000 samples.
Train samples: 800 | Test samples: 200

--- Model Performance Evaluation ---
Accuracy: 0.8145
F1-Score (Weighted): 0.8112

--- Classification Report ---
              precision    recall  f1-score   support
       TREND       0.85      0.82      0.83        72
       RANGE       0.79      0.84      0.81        68
  TRANSITION       0.80      0.78      0.79        60

Top 10 Feature Importances:
ema_slope_50                       : 0.1843
dist_ema_50                        : 0.1412
atr_14                             : 0.1101
...
Model saved successfully to output/market_state_classifier.joblib
```

---

### 8.2 LevelBreakProbabilityModel

- **Purpose**: Calculates the probability that a supply or demand level will break rather than hold (rejection). This prevents entering a buy order directly in front of a supply zone that is highly likely to fail.
- **Inputs**: Zone strength, freshness, number of mitigations, candle volume, and micro momentum.
- **Outputs**: Binary classification (0: REJECT, 1: BREAK).

To execute training:
```bash
python train_level_break.py --dataset output/level_break_dataset.csv --model output/level_break_probability.joblib
```

**Expected Console Output:**
```text
==================================================
    TRAINING LEVEL BREAK PROBABILITY MODEL
==================================================
Loaded dataset containing 1000 samples.
Train samples: 800 | Test samples: 200

--- Model Performance Evaluation ---
Accuracy: 0.7650
F1-Score: 0.7420
ROC-AUC:  0.8214

--- Classification Report ---
              precision    recall  f1-score   support
      REJECT       0.78      0.81      0.79       108
       BREAK       0.75      0.72      0.73        92

Model saved successfully to output/level_break_probability.joblib
```

---

### 8.3 Machine Learning Research Workbench

We provide an interactive development laboratory and prototype sandbox in `notebooks/ml_pipeline_research.ipynb`. This notebook is designed to inspect, debug, and validate every stage of the machine learning training workflow prior to compiling it into production modules.

**Features of the Research Workbench:**
- **Configuration**: Dynamic switching between `MarketStateClassifier` and `LevelBreakProbabilityModel` workflows.
- **Visual Validation**: An interactive **Random Sample Explorer** that visualizes historical candlestick sequences of 35 candles, overlaid with indicators, BOS/CHOCH events, and active supply/demand zones.
- **Data Quality Safeguards**: Automatic checks for missing timestamps, duplicate datetimes, and NaN cells, plus auto-flagging of suspicious anomalies (low variance, collinearity).
- **Advanced Evaluation**: Confusion matrix heatmaps, binary/multiclass ROC-AUC curves, Precision-Recall curves, and Calibration curves.
- **Deterministic Archiving**: Packages and archives trained model weights, configurations, performance metrics, and plots into a timestamped directory under `output/research_experiments/`.

For complete details on using the workbench and migrating notebook prototypes to production modules, see the [ML Pipeline Research Guide](docs/ml_pipeline_research.md).

---

### 8.4 Production-Grade Large-Scale Training Pipeline

For large-scale, enterprise-ready model development, we provide a centralized, highly-optimized production training pipeline in `train_production_pipeline.py`.

This pipeline processes multiple historical symbol datasets individually to minimize memory (RAM) usage, aggregates them into a consolidated master dataset, executes thorough pre-training diagnostics, performs chronological splits (70% Train, 15% Val, 15% Test), trains calibrated LightGBM/RandomForest models, evaluates multi-market cross-generalization, and packages all logs, statistics, metrics, and figures into timestamped experiment directories.

#### Features & Diagnostics:
- **Memory-Efficient Caching**: Processes symbols one-by-one and caches intermediate results as Parquet files under `cache/` for instant resuming and crash-recovery.
- **Thorough Diagnostics**: Scans for label balance, constant features, collinearity, duplicate samples, temporal ordering, and look-ahead leakage. It automatically aborts training if critical errors are found.
- **Cross-Market Generalization**: Evaluates model performance separately on each symbol's test set to guarantee that models learn genuine, cross-market rules rather than overfitting a single currency pair.
- **Experiment Archiving**: Creates timestamped folders under `output/experiments/` containing configurations, logs, metric reports, classification metrics, feature importances, and standard diagnostic curves (Confusion Matrix, ROC, Precision-Recall, Calibration, SHAP).

#### Usage Examples:
```bash
# Train MarketStateClassifier on all available M5 symbols using LightGBM
python train_production_pipeline.py --model-name MarketStateClassifier --timeframe M5

# Train LevelBreakProbabilityModel using RandomForest on EURUSD and GBPUSD on M15 data
python train_production_pipeline.py --model-name LevelBreakProbabilityModel --symbol EURUSD,GBPUSD --timeframe M15 --model-type randomforest

# Run a quick high-stride test execution to verify pipeline end-to-end
python train_production_pipeline.py --model-name MarketStateClassifier --window-stride 50 --force
```

For complete details on the production training workflow, metrics interpretation, and troubleshooting guidelines, see the [Production Training Pipeline Guide](docs/production_training_pipeline.md).

---

## 9. Run Backtesting

The framework contains a high-fidelity, event-driven backtesting engine (`simulation/`) that mocks MT5 tick data and execution behavior.

```text
                  [SimulationRunner]
                          |
             +------------+------------+
             |                         |
             v                         v
     [SimulationClock]        [SimulationBroker]
     - Global Timeline        - Emulates real broker
     - Timestep ticks         - Checks SL/TP hits
             |                         |
             +------------+------------+
                          |
                          v
         [MMStrategy] & Centralized Engines
```

### 9.1 Backtest Execution
Execute a chronological, multi-symbol backtest using the runner script:
```bash
# SimulationRunner clears its journal State/ directory to ensure time determinism.
# See docs/backtesting.md for a complete runnable example with CSV data paths.
```
*(Alternatively, you can run and modify custom backtest parameters using the Jupyter notebook `examples/backtest_eurusd_last_year.ipynb`).*

### 9.2 Key Backtest Outputs
When a backtest completes, files are generated inside `output/` and `Journals/`:
- **`Test_Backtest_Journals/backtest_report.txt`**: Consolidated execution report.
- **`Journals/backtest/positions/`**: Active JSONL logs containing reconstructed `PositionLifecycle` states.
- **Equity Curve Plot**: Saved as a visualization showing balance development.

---

## 10. Optimize Parameters

Finding the optimal parameters (such as Left/Right swing strength, ATR multipliers, or EMA lengths) is critical before deploying live.

The `simulation` package allows running search scripts to fine-tune strategy performance:
- **Grid Search**: Evaluates discrete parameter matrices.
- **Walk-Forward Analysis**: Optimizes on sliding training windows and evaluates on adjacent test windows to prevent overfitting.

Create a script `optimize.py` inside the root to perform a grid search:
```python
import numpy as np
from simulation.simulation_runner import SimulationRunner

runner = SimulationRunner(
    symbols=["EURUSD_o"],
    timeframes=["M5", "M15"],
    data_files={
        ("EURUSD_o", "M5"): "Data/EURUSD_M5.csv",
        ("EURUSD_o", "M15"): "Data/EURUSD_M15.csv",
    },
)

best_f1 = 0
best_params = {}

# Simple search space
for swing_strength in [3, 5, 8]:
    for atr_mult in [1.5, 2.0, 2.5]:
        print(f"Testing Swing Strength: {swing_strength} | ATR Mult: {atr_mult}")
        # Inject configurations, run backtest
        runner.run()
        # Read the generated backtest report before comparing parameter sets.
        if stats.profit_factor > best_f1:
            best_f1 = stats.profit_factor
            best_params = {"swing_strength": swing_strength, "atr_mult": atr_mult}

print(f"Optimization Complete. Best Parameters: {best_params}")
```

Save optimized JSON configurations directly in your project state directory for strategies to load on initialization.

---

## 11. Validation

To ensure safe production deployment, the framework provides three distinct validation layers:

| Validation Script | Purpose | When to Use | Output |
| :--- | :--- | :--- | :--- |
| **`integration_validation.py`** | Verifies mock-broker connection, core loops, position tracking, state recovery, and error management. | Run in CI/CD pipeline or after making core changes. | PASS/FAIL console report. |
| **`live_validation.py`** | Performs active pre-flight checks, authenticates real credentials, checks account balance and spread. | Execute immediately before launching a live bot. | `Validation/validation_report.txt` |
| **`validation.py`** | General unit test runner validating math modules and registries. | Run during feature development. | Test runner outputs. |

```bash
# Run integration check
python integration_validation.py

# Run live pre-flight check
python live_validation.py
```

---

## 12. Demo Trading

Before trading with real capital, run in **Demo Trading** mode. This connects the live pipeline to your broker's demo account.

### 12.1 Activation
Set the system environment variable:
```bash
# On Linux/macOS
export TRADING_MODE=demo
python main.py

# On Windows (PowerShell)
$env:TRADING_MODE="demo"
python main.py
```

### 12.2 Expected Demo Behavior
- Execution commands are forwarded to the real MT5 demo server.
- Stale states are recovered automatically from files in the `State/` folder.
- All logs, journal events, and errors are written to the live directory folders.

---

## 13. Live Trading

Once fully validated, transition to production live execution.

```bash
# On Windows
$env:TRADING_MODE="live"
python main.py
```

### 13.1 Engines Initialized Sequentially
When main is launched, it boots the following modules:
1. **Hierarchical Logging**: Active output is sent to stdout and log files.
2. **MT5 Authentication**: Validates credentials via `credentials.json`.
3. **TradingJournal**: Opens append-only event files.
4. **PositionTracker & Sizer**: Restores tracking state and checks balance.
5. **DrawdownManager**: Enforces hard stop thresholds.
6. **ExitManager**: Mounts trailing multi-stage target monitors.
7. **ChartAnnotationEngine**: Opens visual instructions file outputs.
8. **MMStrategy**: Boots live polling loops evaluating entry rules.

---

## 14. Visualization System

A high-frequency trading platform requires clear visual verification to debug indicator calculations, entries, and exits.

```text
       Core Engine (e.g., SupplyDemandEngine)
                        |
                        v
     [ChartAnnotationEngine] (Visualization/chart_annotator.py)
                        |
                        v (Builds layered DrawInstruction objects)
                        |
            [DrawInstructionWriter]
                        |
     +------------------+------------------+
     |                                     |
     v (Saves layers atomically to CSV)     v (Matplotlib plot engine)
  State/levels/zones CSV layers       [visualization_examples.ipynb]
     |
     v (Polled asynchronously by MT5)
  [FX_DNN_Chart_Renderer.mq5]
```

---

### 14.1 Visual Layers Explained
You can toggle independent visual layers via `debug_config.json`:
- **`swings`**: Renders swing highs/lows with colored dots.
- **`structure`**: Draws BOS and CHOCH break lines.
- **`zones`**: Overlays active Supply (red) and Demand (blue) boxes.
- **`levels`**: Displays calculated structural entry, SL, and TP lines.
- **`signals`**: Highlights entry signals with green (buy) or red (sell) arrows.
- **`ml`**: Overlays probability values from active LightGBM classifiers.

---

### 14.2 MetaTrader 5 Chart Renderer Integration
To overlay framework calculations directly onto live MT5 charts:
1. Copy `Visualization/FX_DNN_Chart_Renderer.mq5` into your MT5 terminal's indicator directory (typically `MQL5/Indicators/` or `MQL5/Experts/`).
2. Compile the file inside the MT5 MetaEditor.
3. Drag and drop the indicator onto your active chart (e.g. EURUSD).
4. The renderer asynchronously polls the independent visual CSV layers from the `State/` folder and dynamically draws native MT5 objects on screen without lagging the terminal.

---

## 15. Trade Auditing

To maintain strategy efficacy and debug execution slippage, the framework includes a forensic **Trade Auditor** module (`trade_auditor.py`).

```text
   MetaTrader 5 Deals     TradingJournal CSV      State Files
           |                     |                     |
           v                     v                     v
    +-------------------------------------------------------+
    |                     [TradeAuditor]                    |
    +-------------------------------------------------------+
                               |
                               v (Forensic Reconstruction)
             Reconstructed PositionLifecycle Objects
                               |
         +---------------------+---------------------+
         |                                           |
         v (Consistency Auditing)                    v (Diagnostic Checking)
  - Execution slippage analysis               - Rejects unauthorized sizes
  - Missing journal logs                      - Flags order modifications
```

The Trade Auditor performs the following checks:
- **`PositionLifecycle` Re-assembly**: Reconstructs a comprehensive timeline for every transaction.
- **Execution Slippage Analysis**: Compares requested entry, SL, and TP prices against actual broker filled levels.
- **Consistency Verification**: Identifies discrepancies between local strategy signals and executed broker deals.

To run a batch audit on all current journal entries:
```bash
python trade_auditor.py --journal-dir Journals/live/
```

---

## 16. Journals

**TradingJournal** operates a fail-safe, append-only logger writing structured, chronological logs across two primary layers:

### 16.1 Layer 1: Chronological Events
Saved as chronological CSV logs inside `Journals/{mode}/events/` (e.g. `Journals/live/events/mm_EURUSD_o_M5_events.csv`).
Tracks individual events:
- `signal`: Trigger parameters, candidate levels, and indicators.
- `order_open`: Lot size, actual entry fills, and broker tickets.
- `order_modify`: SL and TP coordinate updates.
- `position_closed`: Close fills, realized profit, and close reason.

### 16.2 Layer 2: Summaries
Consolidates completed execution records inside `Journals/{mode}/positions/`:
- **`.csv`**: Tabular format for spreadsheet analysis.
- **`.jsonl`**: Complete JSON-serialized `PositionLifecycle` data (contains full chronological events, indicators, and executions).

---

## 17. State Files

The `State/` directory stores active runtime variables to allow the system to recover gracefully from broker disconnects, hardware failures, or application restarts.

- **`position_tracker_state.json`**: Active tickets, lot sizes, entry prices, and current risk metrics.
- **`drawdown_manager_state.json`**: Current balance snapshot, daily high-water mark, and accumulated risk.
- **`exit_manager_state.json`**: Active trailing stop targets and multi-stage exit parameters.
- **`mm_strategy_state.json`**: Last processed bar timestamps to prevent signal duplication.

### 17.1 Fail-Safe Recovery Pattern
The framework uses an atomic write pattern (`safe_file_replace` in `Collecting_Data/utils.py`) to prevent state file corruption during power failures or unexpected crashes:
1. Write temporary changes to `State/filename.tmp`.
2. Perform a file system flush.
3. Rename the temporary file to the final destination name (`State/filename.json`), ensuring consistent, atomic updates on Windows and Linux.

---

## 18. Logging

Logs are managed hierarchically using standard handlers. All runtime outputs are written to files inside the `Logs/` directory.

- **`Logs/trading.log`**: Standard operational log file containing general process flows and warnings.
- **`Logs/errors.log`**: Hard exceptions and stack traces.

### 18.1 Adjusting Log Verbosity
To modify log levels, edit the `setup_logging` call in `main.py` or configuration files:
- **`logging.INFO`**: Recommended for demo and live production trading.
- **`logging.DEBUG`**: Highly verbose, capturing raw MT5 network responses.

---

## 19. Machine Learning Roadmap

The framework is designed to evolve into a fully automated, machine learning driven platform.

```text
[Current Classifiers]                       [Planned Modules]
 - MarketStateClassifier (Regime)            - TradeQualityModel (Signal Filtering)
 - LevelBreakProbabilityModel (Break probability) - SL/TP Optimization (ATR-Adjuster)
                                             - Multi-Strategy Portfolio Manager
```

### Current Status
- **MarketStateClassifier**: Fully operational. Classifies `TREND`, `RANGE`, and `TRANSITION` states.
- **LevelBreakProbabilityModel**: Fully operational. Evaluates whether a supply/demand level is likely to break.

### Future Modules
- **TradeQualityModel**: Predicts expected trade returns before order execution.
- **SL/TP Optimizer**: Dynamically adjusts trailing exits based on real-time volatility.
- **Position Scaling Model**: Scales position sizing based on model prediction confidence.
- **Execution Timing Model**: Optimizes trade entry to minimize slippage.

---

## 20. Strategies

Strategies are decoupled from execution and order management. They receive structured state from the engines and return execution signal candidates.

### 20.1 MM Strategy
- **Purpose**: High-probability trend-following strategy designed for intraday forex trading (M5/M15).
- **Core Rules**:
  - Entry trigger: Candle close crosses above/below the **EMA50**.
  - Trend filter: Overall trend determined by the **EMA600/800** direction.
  - Sizing: centralized exit profiles define standard risk sizing.
- **Current Status**: **Fully Operational** in backtesting, demo, and live modes.

### 20.2 UniT Strategy
- **Purpose**: Trend-continuation strategy designed for strong trending environments.
- **Core Rules**: Evaluates minor structural pullbacks and enters positions on trend-continuation breaks.
- **Current Status**: **Planned**.

### 20.3 SM Strategy
- **Purpose**: Range-bound mean reversion strategy.
- **Core Rules**: Identifies range boundaries, structural highs/lows, and supply/demand zone rejections using ML-confirmed entry signals.
- **Current Status**: **Under Development**.

---

## 21. Complete Development Pipeline

This pipeline details the continuous cycle of development, testing, and system optimization.

```text
    [Configure Strategy & Risk Rules]
                   |
                   v
         [Download CSV Data]
                   |
                   v
       [Generate Structural MSG]
                   |
                   v
        [Registry Feature Build]
                   |
                   v
          [Train ML Models]
                   |
                   v
         [Run Backtest (main)]
                   |
                   v
        [Optimize Parameters]
                   |
                   v
        [Integration Validation]
                   |
                   v
          [Demo Sandbox Run]
                   |
                   v
        [Live Account Deploy]
                   |
                   v
         [Trade Auditor Check]
                   |
                   v
      [Improve Model Features] -> (Repeat Loop)
```

---

## 22. Typical User Workflows

---

### Workflow 1: New User (Getting Started Quickly)

1. **Verify Setup**: Install packages and run validation:
   ```bash
   pip install -r requirements_no_mt5.txt
   python integration_validation.py
   ```
2. **Download Historical Data**: Run data collection to get EURUSD CSV data:
   ```bash
   python collect_history.py
   ```
3. **Build ML Datasets**: Run the ML dataset builder via the research notebook or train scripts:
   ```bash
   # Generates default synthetic datasets if CSVs are missing
   python train_market_state.py
   python train_level_break.py
   ```
4. **Execute Backtest**: Run a local historical simulation:
   ```bash
   python main.py
   ```
5. **Inspect Performance**: Review execution statistics in `Test_Backtest_Journals/backtest_report.txt`.

---

### Workflow 2: Quant Researcher (Feature Engineering)

1. **Create Feature**: Add a new feature definition in `ML/feature_groups.py`.
2. **Register Feature**: Define the feature and its metadata in `ML/feature_registry.py`.
3. **Rebuild Dataset**: Run the DatasetBuilder to regenerate training parquet files.
4. **Evaluate Feature**: Train ML models and analyze feature importances.
5. **Backtest Integration**: Run backtests to verify if the new feature reduces drawdown or improves win rate.

---

### Workflow 3: Strategy Developer (Creating a New Strategy)

1. **Implement Logic**: Create a new strategy file in the `Strategies/` directory.
2. **Register Strategy**: Mount the strategy inside the main execution manager (`main.py`).
3. **Run Simulation**: Run historical backtests to evaluate strategy performance.
4. **Optimize Levels**: Fine-tune parameters using grid search optimization.
5. **Deploy**: Push changes to Demo and Live trading environments.

---

## 23. Troubleshooting

### MT5 Connection Errors
- **Error**: `Terminal initialization failed`.
- **Solution**: Ensure your MetaTrader 5 application is open and running on your system. Verify that the credentials in `credentials.json` are correct and match your active broker account.

### Missing Data/NaN Errors in Features
- **Error**: `KeyError` or feature columns containing NaN values.
- **Solution**: Ensure you have downloaded historical data for **both** M5 and M15 timeframes. The feature pipeline requires both timeframes to compute multi-timeframe indicators and swing points.

### Indicator Rendering Issues
- **Error**: Indicator does not display overlay on MT5 chart.
- **Solution**: Open the Experts tab inside MT5 and verify that the `.mq5` renderer is compiled correctly. Check `debug_config.json` to ensure the required visual layers are enabled.

---

## 24. Future Roadmap

### Near-Term Vision
- Complete implementation of the **SM Mean-Reversion Strategy** for range-bound markets.
- Integrate the **MarketStateClassifier** directly into the strategy execution pipeline.
- Upgrade the **LevelBreakProbabilityModel** to act as a hard entry filter for institutional zones.
- Improve the visual overlay system inside MT5 with custom dashboard components.

### Long-Term Vision
- Multi-strategy portfolio allocation using modern portfolio theory.
- Auto-regressive regime switching models to dynamically select active trading strategies.
- Deep reinforcement learning for order execution optimization.
- Distributed hyperparameter optimization across cloud networks.
- Dedicated web-based monitoring dashboard.

---

*Forex_DNN is designed and maintained by professional quantitative engineers. Use responsibly, manage risk, and test thoroughly before allocating real trading capital.*
