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
```
