# Operations and Integration Guide (Module 16)

This document provides complete instructions for integrating, operating, and extending the Module 16 ML Decision Engine inside backtests, live strategies, and future reinforcement learning workflows.

---

## 1. Quick Start Integration Example

Every strategy consumes predictions and policy suggestions via the standardized `.evaluate()` API.

```python
from ML.ml_decision_engine import MLDecisionEngine

# 1. Initialize the decision engine at strategy startup
decision_engine = MLDecisionEngine(registry_path="models/model_registry.json")

# 2. Within the strategy's bar-by-bar execution loop:
# Collect point-in-time features (from Indicators, Structural Graph, and S&D levels)
feature_snapshot = {
    "ema_50_distance": 1.25,
    "atr_ratio": 0.95,
    "volume_ratio": 1.1,
    "protected_high_distance": -2.3,
    "zone_touch_count": 2,
    # ... any other active features
}

# 3. Query the unified ML Decision Layer
context = decision_engine.evaluate(
    symbol="EURUSD",
    timeframe="M15",
    feature_vector=feature_snapshot,
    strategy_name="MMStrategy"
)

# 4. Consume policy recommendations to drive trading entries
if context.policy_recommendation.allow_trade:
    # Resolve stop-loss and take-profit structures
    tp_mode = context.policy_recommendation.suggested_tp_mode
    sl_offset = context.policy_recommendation.suggested_sl_adjustment

    # Scale risk dynamically using model suggestion
    risk_multiplier = context.policy_recommendation.suggested_risk_multiplier
    position_scale = context.policy_recommendation.suggested_position_scale

    print(f"Executing trade on EURUSD: TP Mode = {tp_mode}, Sizing Scale = {position_scale}")
else:
    print(f"Trade suppressed. Reason: predicted regime {context.predicted_state}, quality {context.trade_quality_score:.2f}")
```

---

## 2. API Reference

### `MLDecisionEngine`
```python
def __init__(
    self,
    registry_path: str = "models/model_registry.json",
    policy: Optional[BasePolicy] = None,
    calibrators: Optional[Dict[str, BaseCalibrator]] = None
)
```
- **`registry_path`**: Path to the central production model registry.
- **`policy`**: Active policy instance (defaults to `RuleBasedPolicy`).
- **`calibrators`**: Dictionary mapping model class names to `BaseCalibrator` instances.

```python
def evaluate(
    self,
    symbol: str,
    timeframe: str,
    feature_vector: Union[Dict[str, Any], pd.DataFrame, np.ndarray],
    strategy_name: str,
    timestamp: Optional[str] = None
) -> DecisionContext
```
- Orchestrates feature alignment, lazy model retrieval, confidence calibration, and policy evaluation. Returns an immutable `DecisionContext`.

---

## 3. Extension Guide: Adding Future Models

To integrate a new model into the centralized decision layer:

1. **Create the Model File**: Create your wrapper (e.g. `ML/models/risk_estimator.py`) inheriting from `BaseTradingModel`.
2. **Define predictions**: Create a typed prediction class (e.g. `RiskPrediction` in `ML/base_model.py`).
3. **Register mapping**: Update `ML/model_registry.py` inside `load_latest_production` to map your new model string to its corresponding wrapper class.
4. **Update DecisionContext**: Add corresponding fields (such as `risk_probability`) to the `DecisionContext` dataclass in `ML/decision_context.py`.
5. **Integrate inside MLDecisionEngine**: Query your model inside `ML/ml_decision_engine.py` and pass the prediction to the Policy evaluator.

---

## 4. Future Path: Reinforcement Learning Integration

The fundamental value of Module 16 is that **Reinforcement Learning can be introduced in the future without changing a single line of Strategy or Execution code**.

Here is the exact blueprint of how this transition is achieved:

### Step 1: Implement an RLPolicy Wrapper
Create your RL policy network inside `ML/policy.py` by completing the `RLPolicy` skeleton class:

```python
from ML.policy import BasePolicy
from ML.decision_context import PolicyRecommendation
import numpy as np

class RLPolicy(BasePolicy):
    def __init__(self, agent_path: str):
        import stable_baselines3
        # Load the pre-trained RL model (PPO, SAC, DQN)
        self.agent = stable_baselines3.PPO.load(agent_path)

    def evaluate(self, state_dict: dict) -> PolicyRecommendation:
        # 1. Map input metrics into a normalized RL observation vector
        observation = np.array([
            state_dict["state_confidence"],
            state_dict["break_probability"],
            state_dict["rejection_probability"],
            state_dict["trade_quality_score"]
        ], dtype=np.float32)

        # 2. Run the policy actor network
        action, _states = self.agent.predict(observation, deterministic=True)

        # Action space mapping: [allow_trade (0/1), risk_multiplier (continuous), tp_mode (0,1,2)]
        allow_trade = bool(action[0] > 0.5)
        risk_scale = float(action[1])
        tp_modes = ["REJECTION_TARGET", "STRUCTURE_TARGET", "BREAKOUT_TARGET"]
        tp_mode = tp_modes[int(np.clip(action[2], 0, 2))]

        # 3. Output the standard immutable recommendation
        return PolicyRecommendation(
            allow_trade=allow_trade,
            suggested_risk_multiplier=risk_scale,
            suggested_position_scale=risk_scale,
            suggested_tp_mode=tp_mode,
            suggested_sl_adjustment=0.0
        )
```

### Step 2: Swap the Policy inside the Strategy Startup
In your strategy execution code or configuration file, instantiate the engine with the RL policy instead of the rule-based policy:

```python
from ML.ml_decision_engine import MLDecisionEngine
from ML.policy import RLPolicy

# Standard Strategy initializes MLDecisionEngine with RL policy
self.decision_engine = MLDecisionEngine(
    registry_path="models/model_registry.json",
    policy=RLPolicy(agent_path="models/RL/ppo_agent.zip")
)
```

**Because the Strategy only consumes the resulting standard `DecisionContext`, the entire backend can be updated from deterministic rules to advanced neural reinforcement policies with zero downstream code changes.**

---

## 5. Runtime ML Integration and Shadow Mode (Milestone 5)

Milestone 5 introduces the runtime ML integration layer, linking analytical engines, ML decisions, and strategy rules under a single, unified execution flow.

### Feature Pipeline Execution (`extract_runtime`)
To generate equivalent feature vectors during live trading, backtesting, or notebooks:
```python
from ML.feature_pipeline import FeaturePipeline
from ML.feature_registry import FeatureRegistry

# 1. Initialize Registry and Pipeline
registry = FeatureRegistry(load_defaults=True)
pipeline = FeaturePipeline(registry)

# 2. Extract features relative to the point-in-time MarketStructureGraph
feature_vector = pipeline.extract_runtime(
    df=df_final,
    msg=graph,
    idx=-2,  # Closed bar to protect against look-ahead bias
    account_session_context={"session": "Asian", "spread": 2.0},
    strategy_context={"signal_direction": 1, "signal_type": "standard"}
)

# feature_vector is a FeatureVector object containing:
# - feature_vector.features: Dict[str, Any]
# - feature_vector.vector: 1D numpy array
# - feature_vector.metadata: dict of extraction statistics and failures
```

### Signal Evaluation (`SignalEvaluator`)
To decouple strategies from ML diagnostics, use the unified `SignalEvaluator` interface:
```python
from Strategies.signal_evaluator import SignalEvaluator

# Initialize the evaluator (Shadow Mode is True, ML Filtering is False)
evaluator = SignalEvaluator(shadow_mode=True, ml_filtering=False)

candidate_signal = {
    "symbol": "EURUSD",
    "timeframe": "M5",
    "direction": 1,
    "signal_type": "standard",
    "technical_rules_satisfied": True
}

evaluation = evaluator.evaluate(
    strategy_name="MMStrategy",
    signal_candidate=candidate_signal,
    feature_vector=feature_vector.features,
    decision_context=decision_context,
    market_structure=graph,
    risk_state={"trading_allowed": True}
)

if evaluation.accepted:
    print(f"Trade Approved. Priority: {evaluation.priority}. Reasons: {evaluation.reasons}")
```

### Tabular Trade Feature Recording (`TradeFeatureRecorder`)
To write training datasets dynamically during operations:
```python
from ML.trade_feature_recorder import TradeFeatureRecorder

# Initialize the recorder (writes daily rolling CSV and Parquet files)
recorder = TradeFeatureRecorder(storage_dir="ML/recorded_features", file_format="both")

# 1. Log Candidate Signal
recorder.record_candidate(
    signal_id=signal_id,
    timestamp="2026-07-10T12:00:00Z",
    strategy="mm",
    symbol="EURUSD",
    timeframe="M5",
    direction="BUY",
    features=feature_vector.features,
    decision_context=decision_context,
    accepted=evaluation.accepted,
    reason="Candidate met technical criteria"
)

# 2. Log Trade Outcome (called automatically inside TradingJournal.log_lifecycle)
# recorder.record_outcome(signal_id=signal_id, lifecycle=lifecycle_instance)
```
