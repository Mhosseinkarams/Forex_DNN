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
                                   v (python main.py --mode backtest)
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

The framework uses clean, uniform CSV historical bar feeds for research, ML dataset generation, and historical simulation.

### 5.1 Historical Collection Commands
To collect historical M5 and M15 bars directly from your connected MT5 terminal, create a simple script (e.g. `collect_history.py` inside a user workflow or run via terminal utility) or utilize `MT5DataFeed` programmatically.

Here is an example utility script you can create as `collect_history.py`:
```python
import os
import pandas as pd
from Collecting_Data.data_feed import MT5DataFeed
from simulation.simulation_environment import env as mt5
from Collecting_Data.auth import load_credentials

creds = load_credentials("credentials.json")
mt5.initialize(login=creds["login"], password=creds["password"], server=creds["server"])

feed = MT5DataFeed()
feed.connect()

symbols = ["EURUSD_o", "GBPUSD_o", "GBPJPY_o", "XAUUSD_o"]
timeframes = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}

os.makedirs("Data", exist_ok=True)

for sym in symbols:
    for tf_name, tf_val in timeframes.items():
        print(f"Downloading 100,000 candles of {sym} on {tf_name}...")
        bars = mt5.copy_rates_from_pos(sym, tf_val, 0, 100000)
        if bars is not None and len(bars) > 0:
            df = pd.DataFrame(bars)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            # Rename columns to standard framework format
            df = df.rename(columns={
                "time": "Datetime", "open": "Open", "high": "High",
                "low": "Low", "close": "Close", "tick_volume": "TickVolume", "spread": "Spread"
            })
            # Drop unnecessary columns
            df = df[["Datetime", "Open", "High", "Low", "Close", "TickVolume", "Spread"]]
            output_path = f"Data/{sym}_{tf_name}.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} bars to {output_path}")
        else:
            print(f"Failed to copy rates for {sym} {tf_name}")

mt5.shutdown()
```

Run the script:
```bash
python collect_history.py
```

### 5.2 Expected CSV Data Format
The generated historical files inside `Data/` will have this exact layout:
```csv
Datetime,Open,High,Low,Close,TickVolume,Spread
2024-01-01 08:00:00,1.09124,1.09156,1.09101,1.09142,425,1
2024-01-01 08:05:00,1.09141,1.09210,1.09138,1.09198,512,1
...
```

### 5.3 Guidelines & Recommendations
- **Candles to download**: Standardizing on **100,000 candles** provides enough chronological data for modeling (approx. 1 year of M5 bars).
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
Run the dataset generator via the command line to construct machine learning ready files.

```bash
# Verify the configuration of the Feature Registry
python -m ML.test_feature_registry
```

Once verified, the registry-driven builders will parse your downloaded CSV bars and structural graphs to construct structured training tables, normalize inputs, replace NaN values, and write optimized parquet/CSV files.

The generation scripts are executed in the modeling step detailed below, producing:
- `output/market_state_dataset.csv`
- `output/level_break_dataset.csv`

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
# In backtest mode, SimulationRunner clears State/ files to ensure time-determinism
python main.py
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

runner = SimulationRunner(symbols=["EURUSD_o"], start_date="2024-01-01", end_date="2024-06-01")

best_f1 = 0
best_params = {}

# Simple search space
for swing_strength in [3, 5, 8]:
    for atr_mult in [1.5, 2.0, 2.5]:
        print(f"Testing Swing Strength: {swing_strength} | ATR Mult: {atr_mult}")
        # Inject configurations, run backtest
        stats = runner.run(config_overrides={"swing_strength": swing_strength, "atr_mult": atr_mult})
        print(f"Profit Factor: {stats.profit_factor:.2f}")
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
