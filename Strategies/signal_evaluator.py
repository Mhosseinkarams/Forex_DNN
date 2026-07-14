import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

from ML.decision_context import DecisionContext
from Market_Data_Pipeline.structure_graph import MarketStructureGraph

logger = logging.getLogger("SignalEvaluator")

@dataclass
class SignalEvaluation:
    """
    Immutable container representing the result of a unified signal evaluation.
    """
    accepted: bool
    rejected: bool
    confidence: float
    priority: str
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ml_diagnostics: Dict[str, Any] = field(default_factory=dict)
    risk_diagnostics: Dict[str, Any] = field(default_factory=dict)

class SignalEvaluator:
    """
    Purpose:
        Provide a single, unified, strategy-agnostic interface for evaluating
        every strategy candidate signal against ML predictions, rule-based policies,
        and overall risk states.

    Design Principle:
        Decouples strategies completely from ML internal details.
        Strategies call `SignalEvaluator.evaluate(...)` to determine final execution.
    """
    def __init__(self, shadow_mode: bool = True, ml_filtering: bool = False):
        self.shadow_mode = shadow_mode
        self.ml_filtering = ml_filtering
        logger.info(f"SignalEvaluator initialized (SHADOW_MODE={self.shadow_mode}, ML_FILTERING={self.ml_filtering})")

    def evaluate(
        self,
        strategy_name: str,
        signal_candidate: Dict[str, Any],
        feature_vector: Any,
        decision_context: Optional[DecisionContext],
        market_structure: MarketStructureGraph,
        supply_demand: Optional[Any] = None,
        risk_state: Optional[Dict[str, Any]] = None
    ) -> SignalEvaluation:
        """
        Evaluate a candidate signal using rule-based strategies, risk constraints, and ML guidance.
        Under Version 1, ML is used in Shadow Mode only and never rejects a trade on its own.
        """
        reasons = []
        warnings = []
        ml_diagnostics = {}
        risk_diagnostics = {}

        # 1. Parse signal candidate properties
        symbol = signal_candidate.get("symbol", market_structure.symbol)
        timeframe = signal_candidate.get("timeframe", market_structure.timeframe)
        direction = signal_candidate.get("direction", 1)
        signal_type = signal_candidate.get("signal_type", "standard")

        dir_text = "BUY" if direction == 1 else "SELL"

        # 2. Collect ML Diagnostics
        ml_allow = True
        predicted_state = "UNKNOWN"
        state_confidence = 0.0

        if decision_context:
            predicted_state = decision_context.predicted_state
            state_confidence = decision_context.state_confidence
            ml_allow = decision_context.policy_recommendation.allow_trade

            ml_diagnostics.update({
                "predicted_state": predicted_state,
                "state_confidence": state_confidence,
                "break_probability": decision_context.break_probability,
                "rejection_probability": decision_context.rejection_probability,
                "trade_quality_score": decision_context.trade_quality_score,
                "policy_allow": ml_allow,
                "inference_time_ms": decision_context.inference_time_ms,
                "warnings": decision_context.warnings
            })
            for w in decision_context.warnings:
                warnings.append(f"ML Warning: {w}")
        else:
            ml_diagnostics["status"] = "No ML DecisionContext provided"

        # 3. Collect Risk Diagnostics
        risk_allow = True
        if risk_state:
            risk_allow = risk_state.get("trading_allowed", True)
            risk_diagnostics.update(risk_state)
            if not risk_allow:
                reasons.append("Blocked by risk constraints (e.g. Drawdown limit reached)")
        else:
            risk_diagnostics["status"] = "No risk state details provided"

        # 4. Resolve technical rule fulfillment
        # Since the strategy created the candidate, the technical conditions are satisfied.
        tech_satisfied = signal_candidate.get("technical_rules_satisfied", True)
        if not tech_satisfied:
            reasons.append("Technical conditions not satisfied (e.g., trend alignment or candle body ratio failed)")

        # 5. Determine final accepted / rejected status (Version 1 rules)
        # Never reject because of ML in shadow mode or when filtering is off.
        final_accepted = tech_satisfied and risk_allow

        if final_accepted:
            reasons.append(f"{strategy_name} conditions satisfied")
            if decision_context:
                reasons.append(f"ML predicts {predicted_state} ({state_confidence:.2f})")
            if self.shadow_mode:
                reasons.append("Shadow Mode only (No ML filtering applied)")
        else:
            reasons.append(f"Signal rejected due to technical/risk rules")

        # 6. Priority resolution
        # Higher quality score from ML or standard technical priority
        priority = "NORMAL"
        if decision_context and decision_context.trade_quality_score > 0.8:
            priority = "HIGH"
        elif signal_type == "high_risk":
            priority = "MEDIUM"

        confidence = float(state_confidence if state_confidence > 0 else 0.5)

        evaluation = SignalEvaluation(
            accepted=final_accepted,
            rejected=not final_accepted,
            confidence=confidence,
            priority=priority,
            reasons=reasons,
            warnings=warnings,
            ml_diagnostics=ml_diagnostics,
            risk_diagnostics=risk_diagnostics
        )

        logger.info(
            f"Evaluator completed for {symbol} {timeframe} {dir_text} | "
            f"Accepted: {evaluation.accepted} | Priority: {evaluation.priority} | "
            f"Reasons: {', '.join(reasons)}"
        )

        return evaluation
