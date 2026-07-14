from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass(frozen=True)
class PolicyRecommendation:
    """
    Immutable representation of policy actions recommended by the policy layer.
    """
    allow_trade: bool
    suggested_risk_multiplier: float
    suggested_position_scale: float
    suggested_tp_mode: str
    suggested_sl_adjustment: float


@dataclass(frozen=True)
class DecisionContext:
    """
    Strongly typed, immutable dataclass containing the integrated market state,
    level break probabilities, aggregated trade metrics, and policy suggestions.
    """
    symbol: str
    timeframe: str
    timestamp: str

    # Market state
    predicted_state: str
    state_probabilities: Dict[str, float]
    state_confidence: float

    # Level analysis
    break_probability: float
    rejection_probability: float

    # Trade evaluation
    trade_quality_score: float
    confidence_score: float

    # Policy recommendations
    policy_recommendation: PolicyRecommendation

    # Diagnostics
    model_versions: Dict[str, str] = field(default_factory=dict)
    inference_time_ms: float = 0.0
    missing_features: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
