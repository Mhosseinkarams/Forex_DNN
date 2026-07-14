# Machine Learning Pipeline & Inference Flow (Module 16)

This document outlines the end-to-end data processing, feature verification, model inference, and confidence calibration pipeline within the Centralized ML Inference Layer.

---

## 1. Inference Pipeline Workflow

The machine learning pipeline inside `MLDecisionEngine` executes as a deterministic 5-step transaction:

```
+--------------------------------------------------+
|               1. Raw Input Vector                |
|  (Dictionary, pandas DataFrame, or numpy array)  |
+                        |                         |
                         v
+--------------------------------------------------+
|           2. Feature Schema Alignment            |
|  - Compares input against FeatureRegistry schema  |
|  - Coerces types (e.g. string to float/int)       |
|  - Fills missing values with feature defaults   |
+                        |                         |
                         v
+--------------------------------------------------+
|             3. Multi-Model Inference             |
|  - Lazy-loads production model checkpoints      |
|  - Resolves features to trained sequence order   |
|  - Outputs raw class probability distributions  |
+                        |                         |
                         v
+--------------------------------------------------+
|            4. Probability Calibration             |
|  - Platt Scaling / Isotonic Piecewise Regression |
|  - Standardizes raw confidence boundaries        |
+                        |                         |
                         v
+--------------------------------------------------+
|            5. Immutable Aggregation              |
|  - Compiles final metrics into DecisionContext   |
+--------------------------------------------------+
```

---

## 2. Robust Feature Schema Alignment

To prevent KeyErrors or dimension mismatch exceptions during production live trading or historical backtesting, `MLDecisionEngine` implements strict schema validation via `FeatureRegistry`:

- **Missing Features**: If any of the registry's enabled features are omitted from the strategy's input dictionary, the engine automatically registers a diagnostic warning and fills the missing feature with the defined `default_value` from the registry.
- **Type Coercion**: Features of string format (e.g. `"0.85"`) are automatically coerced into target data types specified in `FeatureDefinition.dtype` (such as `float` or `int`).
- **Numpy Arrays**: Raw numpy vectors are validated against the expected length of enabled features. Mismatches are flagged, and features are safely sequenced with padding/truncating defaults.

---

## 3. Decoupled Confidence Calibration

Raw ML model probabilities from classifiers like LightGBM or Random Forests are frequently uncalibrated (e.g. skewed toward extreme scores). Module 16 introduces three decoupled confidence calibration strategies:

### 3.1 Calibration Options

| Calibrator | Methodology | Use Case |
| :--- | :--- | :--- |
| **IdentityCalibrator** | No modification. Passes raw probability directly. | Well-calibrated base models or exploratory research. |
| **PlattCalibrator** | Platt scaling logic using logistic mapping: $P = \frac{1}{1 + e^{A \cdot f(x) + B}}$ | Correcting sigmoid distortion/overconfidence. |
| **IsotonicCalibrator** | Piecewise linear isotonic regression interpolation. | Highly non-parametric calibration adjustments. |

### 3.2 Configuration Interface

Calibration is configured at engine startup, allowing strategies or operators to swap mapping layers seamlessly without altering the underlying model class files:

```python
from ML.confidence_calibrator import IsotonicCalibrator, PlattCalibrator
from ML.ml_decision_engine import MLDecisionEngine

engine = MLDecisionEngine(
    calibrators={
        "MarketStateClassifier": PlattCalibrator(A=-1.2, B=0.1),
        "LevelBreakProbabilityModel": IsotonicCalibrator(
            thresholds=[0.0, 0.4, 0.7, 1.0],
            targets=[0.0, 0.3, 0.8, 1.0]
        )
    }
)

---

## 4. Milestone 5: Runtime ML Integration Layers

Milestone 5 extends the centralized ML pipeline into the active trading runtimes (live trading, simulation, and backtesting notebooks), ensuring 100% equivalence in feature calculation, schema tracking, and transaction safety.

### 4.1 Runtime Feature Pipeline (Module 17)
The `FeaturePipeline` (`ML/feature_pipeline.py`) performs the crucial task of generating the exact same features used during training.
- **Strict Verification**: Every runtime feature vector is checked for missing attributes, NaN values, infinites, and wrong types. Anomalies are logged as warnings and replaced with registry-defined default values.
- **Cache Acceleration**: In-memory caching avoids redundant calculations (such as rolling ATR ratio or linear regressions) within the same index loop.

### 4.2 Trade Feature Recorder (Module 19)
The `TradeFeatureRecorder` (`ML/trade_feature_recorder.py`) handles continuous dataset collection for future retraining cycles.
- **Step 1: record_candidate**: When a signal is checked by the strategy, the recorder flattens all metrics—including metadata, all registry-driven features, and ML DecisionContext predictions—into a single tabular CSV/Parquet row.
- **Step 2: record_outcome**: When the position is eventually closed, the `TradingJournal` automatically invokes the recorder using the unique `signal_id`. The recorder updates the matched row with full outcome metrics (pips, realized profit, maximum drawdown, duration, exit reason).

### 4.3 Signal Evaluator (Module 20)
The `SignalEvaluator` (`Strategies/signal_evaluator.py`) decouples the strategy's core logic from ML internals.
- **Agnostic Interface**: Strategies submit signal candidates, the computed feature dictionary, and active `DecisionContext` to the evaluator.
- **Policy Decoupling**: In Shadow Mode, the evaluator ensures that technical rules drive executions, but logs full ML metrics and policy recommended multi-target adjustments under custom loggers:
  - `Logs/runtime_features.log`
  - `Logs/decision_engine.log`
  - `Logs/shadow_mode.log`
  - `Logs/signal_evaluator.log`
```
