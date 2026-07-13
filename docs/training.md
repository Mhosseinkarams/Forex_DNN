# Model Training Guide: Production Machine Learning Workflow

This document describes the unified, production-grade machine learning pipeline for the Forex Trading Framework, spanning feature definition, dataset building, model training, calibration, and real-time execution.

---

## 1. Machine Learning Subsystem Architecture

The ML subsystem is structured around modular, reusable components under `ML/` that support every future ML model:

```
  +-------------------------+
  |     FeatureRegistry     |  <-- Single source of truth for features
  +------------+------------+
               |
               v
  +-------------------------+
  |    BaseTradingModel     |  <-- Core parent class (save/load, calibrate, predict)
  +------------+------------+
               |
         +-----+-----+
         |           |
         v           v
  +------------+  +------------+
  |MarketState |  | LevelBreak |  <-- Child models defining custom architectures
  | Classifier |  | Probability|
  +------------+  +------------+
         |           |
         +-----+-----+
               |
               v
  +------------+------------+
  |         Trainer         |  <-- Chronological splits, splits calibration, fits
  +------------+------------+
               |
               v
  +------------+------------+
  |        Evaluator        |  <-- Visualizes and reports stats (HTML & Markdown)
  +------------+------------+
               |
               v
  +------------+------------+
  |      ModelRegistry      |  <-- Tracks versions and loads production weights
  +-------------------------+
```

---

## 2. Standardized Step-by-Step Pipeline

The machine learning workflow follows a strict sequential pipeline:

```
Step 1: Download History -> Step 2: Build Dataset -> Step 3: Run Trainer -> Step 4: Model Registry -> Step 5: Inference
```

### Step 1: Ingest High-Quality Historical Data
Use the unified historical data downloader to gather parquet-formatted bar data:
```bash
python Collecting_Data/historical_data_collector.py --timeframe M5 --symbols EURUSD --format parquet
```

### Step 2: Build ML Training Datasets
Use the `HistoricalDatasetBuilder` to build scale-invariant, point-in-time training datasets:
```python
from Market_Data_Pipeline.historical_dataset_builder import HistoricalDatasetBuilder

builder = HistoricalDatasetBuilder(input_dir="HistoricalData", output_dir="output", timeframe="M5")
df, metadata = builder.build_dataset()
```
This writes `output/dataset_v001.parquet` along with cryptographic verification metadata.

### Step 3: Standardized Training (`Trainer`)
Launch training using `Trainer` with config parameter files:
```bash
# Trains Market State Classifier
python train_market_state.py --dataset output/market_state_dataset.csv --model output/market_state_classifier.joblib

# Trains Level Break Probability
python train_level_break.py --dataset output/level_break_dataset.csv --model output/level_break_probability.joblib
```

### Step 4: Model Registry Logging (`ModelRegistry`)
Every saved model is registered automatically inside `models/model_registry.json`, capturing:
- Metrics (Accuracy, F1, Confusion Matrix, Precision, Recall)
- Dataset Hash & Registry Version
- Git Commit SHA
- Calibration method (sigmoid or isotonic)

---

## 3. Probability Calibration

In quantitative trading, raw classifier probabilities are often poorly calibrated. To solve this, `BaseTradingModel` incorporates built-in probability calibration:
- **Platt Scaling (`sigmoid`)**: Fits a logistic regression model on top of predictions.
- **Isotonic Regression (`isotonic`)**: A non-parametric calibration approach.

```python
model = MarketStateClassifier()
# Fits on validation data
model.calibrate(X_val, y_val, method="sigmoid")
```

---

## 4. Real-time Inference API

Strategies should never call scikit-learn or LightGBM directly. Instead, they interact with the unified inference API of the loaded wrapper model:

```python
# Query latest production model from Registry
from ML.model_registry import ModelRegistry
registry = ModelRegistry()
model = registry.load_latest_production("MarketStateClassifier")

# Inference for single snapshot
snapshot = {"ema50_slope": 0.05, "atr": 12.5, "volume": 1200}
prediction = model.predict(snapshot)

# Returns structured, typed prediction dataclass
print(prediction.regime)               # "TREND"
print(prediction.trend_probability)   # 0.84
print(prediction.confidence)          # 0.84
```

---

## 5. Adding a Future Model

To implement a new model (e.g. `TradeQualityModel`):
```python
import numpy as np
from typing import Dict, List, Any
from lightgbm import LGBMClassifier
from ML.base_model import BaseTradingModel, LevelBreakPrediction

class TradeQualityModel(BaseTradingModel):
    def build_model(self):
        self.model = LGBMClassifier(
            random_state=self.random_state,
            n_estimators=self.hyperparameters.get("n_estimators", 100),
            learning_rate=self.hyperparameters.get("learning_rate", 0.05)
        )

    def prediction_schema(self, probas: np.ndarray, raw_pred: np.ndarray) -> LevelBreakPrediction:
        prob = probas[0]
        return LevelBreakPrediction(
            break_probability=float(prob[1]),
            reject_probability=float(prob[0]),
            confidence=float(max(prob))
        )

    def required_feature_groups(self) -> List[str]:
        return ["SMC_Structural", "Volatility"]

    def evaluation_metrics(self) -> List[str]:
        return ["accuracy", "f1_binary"]

    def default_hyperparameters(self) -> Dict[str, Any]:
        return {"n_estimators": 100, "learning_rate": 0.05}
```
No redundant fit, save/load, metadata logging, or calibration code needed!
