# Production Machine Learning Framework (ML Core Core)

The **Forex_DNN Production ML Framework** is a unified, highly scalable quantitative research and production-grade model management system. It decouples core trading logic from underlying machine learning libraries (such as LightGBM, RandomForest, and others), standardizing feature extraction, hyperparameter tuning, training workflows, verification validation, calibration, and real-time model loading.

---

## 1. Architectural Overview

The framework transitions Forex_DNN from thin wrapper scripts into a robust, object-oriented production system. It is composed of five core modules:

```
                  +-------------------------+
                  |     FeatureRegistry     |
                  +------------+------------+
                               |
                               v
                  +-------------------------+
                  |    BaseTradingModel     |
                  +------------+------------+
                               |
         +---------------------+---------------------+
         |                                           |
         v                                           v
+------------------------+                  +------------------------+
| MarketStateClassifier  |                  |LevelBreakProbability...|
+------------------------+                  +------------------------+
         |                                           |
         +---------------------+---------------------+
                               |
                               v
                  +-------------------------+
                  |  Structured Predictions |
                  +------------+------------+
                               |
         +---------------------+---------------------+
         v                                           v
+-----------------+                         +-----------------+
|     Trainer     |                         |    Evaluator    |
+-----------------+                         +-----------------+
         |                                           |
         v                                           v
+-----------------+                         +-----------------+
|  ModelRegistry  |                         |  HTML/MD Audit  |
+-----------------+                         +-----------------+
```

---

## 2. BaseTradingModel (`ML/base_model.py`)

`BaseTradingModel` is the abstract base class (ABC) and parent class for every current and future ML model in Forex_DNN. It is strictly strategy-agnostic and manages:
- **Initialization**: Standardized parameters (`model_type`, `config_path`, `hyperparameters`, `random_state`).
- **YAML Config Binding**: Automatically loads external YAML configurations (e.g. `configs/market_state.yaml`) and merges them with defaults.
- **Unified Feature Registry Querying**: Dynamically maps requirements to registered feature definitions.
- **Model Persistability**: Uses joblib to serialize the complete object wrapper alongside metadata, feature names, and calibrated estimators.
- **Probability Calibration**: Built-in Platt Scaling (`sigmoid`) and Isotonic Regression (`isotonic`) for robust confidence calibration (safely handling scikit-learn version constraints).
- **Comprehensive Metadata Logging**: Captures training timestamp, samples count, dataset version/hash, Feature Registry hashes, git commit, metrics, and hyperparameter snapshots.

### Implementation Checklist for Future Models
To add a new model (e.g. `TradeQualityModel`, `RiskEstimator`):
1. Inherit from `BaseTradingModel`.
2. Implement:
   - `build_model()`: Initialize classifier backends.
   - `prediction_schema(probas, raw_pred)`: Build typed prediction results.
   - `required_feature_groups()`: List enabled groups (e.g. `["SMC_Structural", "Volatility"]`).
   - `evaluation_metrics()`: Metrics to compute.
   - `default_hyperparameters()`: Hyperparameters dictionary.

---

## 3. Standardized Structured Prediction Objects

No strategy should ever handle raw numpy probability arrays directly. The framework wraps inference into typed dataclasses to prevent indexing errors and guarantee extensibility:

### MarketStatePrediction
```python
@dataclass
class MarketStatePrediction:
    regime: str                    # "TREND", "RANGE", "TRANSITION"
    trend_probability: float
    range_probability: float
    transition_probability: float
    confidence: float
    trend_strength: float = 0.0
    expected_volatility: float = 0.0
    expected_persistence: float = 0.0
```

### LevelBreakPrediction
```python
@dataclass
class LevelBreakPrediction:
    break_probability: float
    reject_probability: float
    confidence: float
    expected_move: float = 0.0
    expected_time_to_break: float = 0.0
```

*All predictions expose `.to_dict()` and serialize cleanly.*

---

## 4. Centralized Model Registry (`ML/model_registry.py`)

The Model Registry maintains a structured JSON-based registry (`models/model_registry.json`) detailing all trained checkpoints. It allows easy, programmatic retrieval of `"latest production model"` without hardcoded paths in backtesters or live execution engines.

### Querying the Latest Production Model
```python
from ML.model_registry import ModelRegistry

registry = ModelRegistry()
# Automatically infers the class type and loads the production model
model = registry.load_latest_production("MarketStateClassifier")
```

---

## 5. Unified Trainer & Evaluator (`ML/trainer.py`, `ML/evaluator.py`)

### Training Harness (`Trainer`)
Executes chronological (zero-lookahead) data splits or shuffled splits, fits the estimator, performs post-training probability calibration on validation datasets, evaluates statistical metrics, and automatically logs training metadata with direct Model Registry serialization.

```python
from ML.models.market_state_classifier import MarketStateClassifier
from ML.trainer import Trainer

model = MarketStateClassifier(config_path="configs/market_state.yaml")
trainer = Trainer()
results = trainer.train_model(
    model=model,
    df=dataset_df,
    target_col="label",
    model_save_path="models/MarketState/lgbm_model.joblib",
    is_production=True
)
```

### Automatic Verification Auditing (`Evaluator`)
Calculates accuracy, weighted F1, precision, recall, and ROC-AUC metrics. Generates a premium static compilation figure, an autogenerated Markdown audit, and a beautiful standalone interactive HTML dashboard utilizing Tailwind CSS and Chart.js.

---

## 6. Usage and Operations Guide

### Command Line Interfaces (CLI)

```bash
# Train Market State Classifier
python train_market_state.py --dataset output/market_state_dataset.csv --model output/market_state_classifier.joblib

# Train Level Break Probability Model
python train_level_break.py --dataset output/level_break_dataset.csv --model output/level_break_probability.joblib
```

### Common Operator Mistakes & Troubleshooting
1. **ValueError: invalid literal for int() in Target Column**: The CSV contains string labels. The `Trainer` automatically encodes string/object targets if they are of string type. Ensure target columns are designated in the `target_col` argument.
2. **Missing columns in Inference**: If the DataFrame passed to `predict()` does not contain all training features, the model automatically references the `FeatureRegistry` to fill missing features with default values instead of throwing a KeyError.

---

## 7. Model Extension Template

Here is how you can write any future ML model in less than 50 lines of code without touching standard infrastructure:

```python
import numpy as np
from typing import Dict, List, Any
from lightgbm import LGBMClassifier
from ML.base_model import BaseTradingModel, LevelBreakPrediction # or write your own typed dataclass

class TradeQualityModel(BaseTradingModel):
    def build_model(self):
        self.model = LGBMClassifier(
            random_state=self.random_state,
            n_estimators=self.hyperparameters.get("n_estimators", 100),
            learning_rate=self.hyperparameters.get("learning_rate", 0.05)
        )

    def prediction_schema(self, probas: np.ndarray, raw_pred: np.ndarray) -> LevelBreakPrediction:
        # Map class predictions to a typed object
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
