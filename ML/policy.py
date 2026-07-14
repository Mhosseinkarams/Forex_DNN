from abc import ABC, abstractmethod
from ML.decision_context import PolicyRecommendation


class BasePolicy(ABC):
    """
    Abstract Policy Interface for determining trading action and sizing
    from an aggregated DecisionContext snapshot.
    """
    @abstractmethod
    def evaluate(self, state_dict: dict) -> PolicyRecommendation:
        """
        Evaluate predictive scores and return an immutable PolicyRecommendation.

        Args:
            state_dict: Dictionary containing predicted_state, break_probability,
                        trade_quality_score, and related confidence metrics.
        """
        pass


class RuleBasedPolicy(BasePolicy):
    """
    Standard deterministic policy execution based on configurable thresholds.
    """
    def __init__(
        self,
        min_state_confidence: float = 0.5,
        min_trade_quality: float = 0.5,
        default_tp_mode: str = "STRUCTURE_TARGET"
    ):
        self.min_state_confidence = min_state_confidence
        self.min_trade_quality = min_trade_quality
        self.default_tp_mode = default_tp_mode

    def evaluate(self, state_dict: dict) -> PolicyRecommendation:
        """
        Evaluate raw dictionary of metrics to produce a recommendation.
        """
        predicted_state = state_dict.get("predicted_state", "TRANSITION")
        state_confidence = state_dict.get("state_confidence", 0.0)
        trade_quality_score = state_dict.get("trade_quality_score", 0.0)
        break_probability = state_dict.get("break_probability", 0.0)

        # Baseline decision checks
        is_trend_or_range = predicted_state in ["TREND", "RANGE"]
        high_confidence_state = state_confidence >= self.min_state_confidence
        good_quality = trade_quality_score >= self.min_trade_quality

        # Policy allows trading if regime is favorable, high confidence state, and trade is quality
        allow_trade = is_trend_or_range and high_confidence_state and good_quality

        # Compute scaling parameters dynamically based on predictive inputs
        suggested_risk_multiplier = 1.0
        suggested_position_scale = 1.0

        if allow_trade:
            # Scale up on high quality, trend-based setups
            if predicted_state == "TREND" and trade_quality_score > 0.8:
                suggested_risk_multiplier = 1.5
                suggested_position_scale = 1.2
            elif predicted_state == "RANGE":
                # Slightly scale down in ranges
                suggested_risk_multiplier = 0.8
                suggested_position_scale = 0.8

        # TP mode recommendation based on level analysis / break probability
        suggested_tp_mode = self.default_tp_mode
        if break_probability > 0.7:
            suggested_tp_mode = "BREAKOUT_TARGET"
        elif break_probability < 0.3:
            suggested_tp_mode = "REJECTION_TARGET"

        # Suggested SL adjustments based on break probability / uncertainty
        suggested_sl_adjustment = 0.0
        if break_probability > 0.6:
            suggested_sl_adjustment = 0.5  # wider stop for breakouts
        elif break_probability < 0.4:
            suggested_sl_adjustment = -0.2  # tighter stop for rejections

        return PolicyRecommendation(
            allow_trade=allow_trade,
            suggested_risk_multiplier=float(suggested_risk_multiplier),
            suggested_position_scale=float(suggested_position_scale),
            suggested_tp_mode=suggested_tp_mode,
            suggested_sl_adjustment=float(suggested_sl_adjustment)
        )


class RLPolicy(BasePolicy):
    """
    Placeholder/Hook architecture for future Reinforcement Learning Policies.
    Integrates directly with trading environment or pre-trained policy networks (PPO, DQN, SAC).
    """
    def __init__(self, agent_path: str = None):
        self.agent_path = agent_path
        # e.g., self.model = stable_baselines3.PPO.load(agent_path)

    def evaluate(self, state_dict: dict) -> PolicyRecommendation:
        # Future RL policy implementation will call agent predict on normalized observation vector
        # observation = np.array([state_dict[f] for f in self.feature_list])
        # action, _ = self.model.predict(observation)
        return PolicyRecommendation(
            allow_trade=True,
            suggested_risk_multiplier=1.0,
            suggested_position_scale=1.0,
            suggested_tp_mode="RL_TARGET",
            suggested_sl_adjustment=0.0
        )


class ImitationLearningPolicy(BasePolicy):
    """
    Placeholder/Hook architecture for future Imitation Learning / Behavioral Cloning Policies.
    """
    def evaluate(self, state_dict: dict) -> PolicyRecommendation:
        return PolicyRecommendation(
            allow_trade=True,
            suggested_risk_multiplier=1.0,
            suggested_position_scale=1.0,
            suggested_tp_mode="IMITATION_TARGET",
            suggested_sl_adjustment=0.0
        )


class BayesianPolicy(BasePolicy):
    """
    Placeholder/Hook architecture for future Bayesian Inference decision layers.
    """
    def evaluate(self, state_dict: dict) -> PolicyRecommendation:
        return PolicyRecommendation(
            allow_trade=True,
            suggested_risk_multiplier=1.0,
            suggested_position_scale=1.0,
            suggested_tp_mode="BAYESIAN_TARGET",
            suggested_sl_adjustment=0.0
        )
