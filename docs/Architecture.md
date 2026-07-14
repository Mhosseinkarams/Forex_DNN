# Centralized ML Decision Architecture (Module 16)

This document describes the high-performance, thread-safe Centralized Machine Learning Inference and Decision layer (`MLDecisionEngine`) introduced in Module 16 of the Forex_DNN trading framework.

---

## 1. Core Philosophy and Decoupled Design

To support future reinforcement learning, imitation learning, and Bayesian policies without disrupting standard execution loops or hardcoding strategies, Module 16 implements a fully decoupled **RL-Ready Architecture**:

- **Decoupled Inference**: Strategy files and MetaTrader 5 order execution are completely unaware of model backends, training data schemas, or raw probability arrays.
- **centralized Layer**: `MLDecisionEngine` acts as a single, thread-safe inference hub. It aggregates models, runs raw/calibrated predictions, evaluates a target policy, and emits an immutable snapshot.
- **Zero-Action Principle**: `MLDecisionEngine` never executes orders, manages risk margins, or modifies open positions. It is strictly passive and strategy-agnostic. Strategies consume recommendations and make the final transaction decisions.

---

## 2. Component Hierarchy & Flow

```
+------------------------------------------------------------+
|                       Trading Strategy                     |
|  (consumes DecisionContext and executes trades via broker)  |
+-----------------------------+------------------------------+
                              |
                              | 1. evaluate(...)
                              v
+-----------------------------+------------------------------+
|                     MLDecisionEngine                       |
|   (orchestrates validation, inference, and policy evaluation)|
+--+--------------------+---------------------+------------+-+
   |                    |                     |            |
   | 2. Align Features  | 3. Query            | 4. Calibrate| 5. Policy
   v                    v                     v            v
+--+-------------+ +----+-------------+ +-----+-----+ +----+--------+
| FeatureRegistry| |  ModelRegistry   | |Calibrators| | RulePolicy  |
| (Type Coercion)| |  (Lazy / Cache)  | |(Platt/Iso)| | (Action, RR)|
+----------------+ +------------------+ +-----------+ +-------------+
                              |
                              | 6. Return immutable DecisionContext
                              v
+-----------------------------+------------------------------+
|                      DecisionContext                       |
|  (Strongly-typed, thread-safe, immutable snapshot)         |
+------------------------------------------------------------+
```

---

## 3. Class Specifications

### 3.1 DecisionContext & PolicyRecommendation
An immutable dataclass (`@dataclass(frozen=True)`) containing:
- **Metadata**: `symbol`, `timeframe`, `timestamp`.
- **Market State Classifier**: `predicted_state`, `state_probabilities`, `state_confidence`.
- **Level Break Model**: `break_probability`, `rejection_probability`.
- **Trade Quality Model**: `trade_quality_score`, `confidence_score`.
- **Policy Recommendation**: `allow_trade`, `suggested_risk_multiplier`, `suggested_position_scale`, `suggested_tp_mode`, `suggested_sl_adjustment`.
- **Diagnostics**: `model_versions`, `inference_time_ms`, `missing_features`, `warnings`.

### 3.2 ModelRegistry Extensions
- **Lazy Loading**: Loaded models are only deserialized from disk on the first evaluate request.
- **Caching**: Fully thread-safe model caching using internal threading locks ensures subsequent queries incur zero serialization overhead.
- **Tolerance**: Missing optional models log warnings into the registry but do not crash inference.

---

## 4. Execution Sequence Diagram

```
Strategy                MLDecisionEngine          ModelRegistry         Calibrators         RulePolicy
   |                            |                       |                    |                  |
   |---- evaluate(vector) ----->|                       |                    |                  |
   |                            |-- validate features ->|                    |                  |
   |                            |                       |                    |                  |
   |                            |-- load models (lazy) ->|                   |                  |
   |                            |<-- return instances --|                    |                  |
   |                            |                       |                    |                  |
   |                            |--------- run raw inference --------------->|                  |
   |                            |                                            |                  |
   |                            |--------- calibrate raw probabilities ----->|                  |
   |                            |<-------- return calibrated scores ---------|                  |
   |                            |                                                               |
   |                            |------------------------- evaluate policy -------------------->|
   |                            |<------------------------ return actions ----------------------|
   |                            |
   |<- return DecisionContext --|
```

---

## 5. Caching and Thread-Safety

To support live high-frequency streams and thousands of evaluations per hour across dozens of concurrent pairs:
- **Threading Locks**: Caching registers and load mechanisms utilize python `threading.Lock` coordinates to guarantee thread-safe memory allocations.
- **Pre-mapped Features**: Enabled feature listings are cached inside the decision engine at startup to avoid re-evaluating the registry map on every candle tick.
