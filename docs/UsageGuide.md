# Forex_DNN Usage & Operations Guide

Welcome to the production operator guide for the **Forex_DNN Quantitative Trading Framework**.
This guide provides the complete end-to-end 10-step quantitative workflow, module responsibilities, architecture, and configuration details.

---

## 1. Modular Architecture Overview

The Forex_DNN repository is structured into isolated, highly modular packages, separating data processing, model training, and trading execution runtimes:

```
├── Configs/                   # Centralized runtime YAML configuration parameters
├── Collecting_Data/           # Ingestion, MT5 adapters, and historical collectors
├── Market_Data_Pipeline/      # Analytical engines: structure (BOS/CHOCH), S&D zones, spikes
├── ML/                        # Central ML feature pipelines, registries, and models wrappers
├── Training/                  # Model-specific training scripts and trainers
├── Pipeline/                  # Training & Trading pipeline runtime orchestrators
├── Strategies/                # Trend-following (MM), Ranging mean-reversion (SM), and UniT
├── Trade_Execution/           # Position managers, drawdown limits, and order execution
├── Simulation/                # High-fidelity event-driven chronological backtesting
├── Validation/                # Post-trade audits and system dry-runs
└── Visualization/             # Analytical Matplotlib and MT5 CSV chart annotation overlays
```

---

## 2. Comprehensive 10-Step Operational Workflow

Follow this standardized chronological pipeline to run data processing, ML model retraining, backtesting, and live trading.

```
[1. Collect Data] -> [2. Run process_data.py] -> [3. Inspect Datasets]
                             ↓
[6. Run Validation] <- [5. Review Reports] <- [4. Train ML (train.py)]
        ↓
[7. Run Backtests] -> [8. Optimize Parameters] -> [9. Paper Trading] -> [10. Live Trading]
```

### Step 1: Collect Historical Data
- Use the unified `Collecting_Data/historical_data_collector.py` to download high-fidelity ticks or candles from your broker or MetaTrader 5 terminal.
- Ensure files are stored in Parquet or CSV under `HistoricalData/<SYMBOL>/`.

### Step 2: Run process_data.py
- Prepares ALL historical data for machine learning by running deduplication, chronological sorting, Indicator Calculation, SMC structure detection, zone mapping, and candle strength/rejection evaluations.
- **Command**:
  ```bash
  python3 process_data.py --symbol ALL --timeframe M5
  ```

### Step 3: Inspect Generated Datasets
- Check the output datasets saved under `output/datasets/`:
  - `market_state_dataset.parquet` (three-class regime labels: TREND, RANGE, TRANSITION)
  - `level_break_dataset.parquet` (binary approach outcome labels: REJECT, BREAK)
  - `future_rl_dataset.parquet` (Reinforcement Learning state vectors with reward/done annotations)
  - `future_trade_quality_dataset.parquet` (candidate feature vectors annotated with trade quality statistics)
- Inspect the generated metadata and schema log at `output/datasets/metadata.json`.

### Step 4: Train ML Models Using train.py
- Use the unified orchestrator `train.py` to discover datasets and trainers and execute model retraining in chronological dependency order.
- **Command**:
  ```bash
  python3 train.py --all
  ```

### Step 5: Review Reports
- Retraining runs automatically output comprehensive diagnostic performance figures, confusion heatmaps, feature importances, and interactive HTML dashboards under `output/reports/`.
- Review the `output/reports/market_state_evaluation_report.html` and `output/reports/level_break_evaluation_report.html` in your browser.

### Step 6: Run Validation
- Execute the end-to-end dry-run integration check to verify all pipeline and trading bootstrap modules are perfectly safe:
  ```bash
  python3 Validation/validate_all.py
  ```

### Step 7: Run Backtests
- Run high-fidelity historical simulations against virtual brokers and simulated account clocks using the unified `trade.py` entry point:
  ```bash
  python3 trade.py --mode backtest --symbols EURUSD --strategy mm_strategy
  ```

### Step 8: Optimize Parameters
- Tweak strategy variables and indicators inside `Configs/strategy_config.yaml` to improve historical metrics (win rates, drawdowns, expectancy). No strategy code modifications are ever needed.

### Step 9: Run Paper Trading
- Execute the pipeline in Demo mode against your broker's simulated MetaTrader 5 terminal to verify real-time fills and execution latency:
  ```bash
  python3 trade.py --mode demo --symbols EURUSD
  ```

### Step 10: Run Live Trading
- Launch full live operations once all paper test cycles pass.
- **Command**:
  ```bash
  python3 trade.py --mode live --symbols EURUSD
  ```

---

## 3. Configuration instructions

All system variables are isolated under `Configs/` as clear YAML fields:
- **`process_config.yaml`**: Data cleaning, time sorting, window size/stride, and parallel worker counts.
- **`training_config.yaml`**: Hyperparameters, seeds, train/validation split configurations, and dataset targets.
- **`trading_config.yaml`**: Selection of live vs simulated run modes, risk ceilings, drawdown limits, and strategy flags.
- **`strategy_config.yaml`**: Custom indicators periods, entry filters, and SL/TP placement ratios.
- **`visualization_config.yaml`**: Active overlay annotation layers and color codes for Matplotlib/MT5 charts.
