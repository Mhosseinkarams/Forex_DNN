import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union

from ML.feature_registry import FeatureRegistry
from ML.model_registry import ModelRegistry
from ML.decision_context import DecisionContext, PolicyRecommendation
from ML.confidence_calibrator import BaseCalibrator, IdentityCalibrator, PlattCalibrator, IsotonicCalibrator
from ML.policy import BasePolicy, RuleBasedPolicy

# Dedicated decision engine logger
logger = logging.getLogger("MLDecisionEngine")


class MLDecisionEngine:
    """
    Centralized ML inference and decision layer that aggregates outputs of multiple models
    and provides a unified immutable DecisionContext to every strategy.

    This class is thread-safe, high-performance, and completely strategy-agnostic.
    """
    def __init__(
        self,
        registry_path: str = "models/model_registry.json",
        policy: Optional[BasePolicy] = None,
        calibrators: Optional[Dict[str, BaseCalibrator]] = None
    ):
        self.model_registry = ModelRegistry(registry_path=registry_path)
        self.feature_registry = FeatureRegistry(load_defaults=True)
        self.policy = policy or RuleBasedPolicy()

        # Configure confidence calibrators per model type
        self.calibrators = calibrators or {
            "MarketStateClassifier": IdentityCalibrator(),
            "LevelBreakProbabilityModel": IdentityCalibrator(),
            "TradeQualityModel": IdentityCalibrator()
        }

        # Fast caching of feature registry enabled names and length
        self._enabled_features = [f.name for f in self.feature_registry.list_enabled()]
        self._enabled_feature_count = len(self._enabled_features)

        logger.info(f"MLDecisionEngine initialized. {self._enabled_feature_count} enabled features tracked.")

    def evaluate(
        self,
        symbol: str,
        timeframe: str,
        feature_vector: Union[Dict[str, Any], pd.DataFrame, np.ndarray],
        strategy_name: str,
        timestamp: Optional[str] = None
    ) -> DecisionContext:
        """
        Runs inference on all active ML models, calibrates raw probabilities,
        evaluates the active Policy, and returns a compiled, immutable DecisionContext.
        """
        start_time = time.perf_counter()
        timestamp = timestamp or pd.Timestamp.now().isoformat()

        warnings: List[str] = []
        missing_features: List[str] = []
        model_versions: Dict[str, str] = {}

        # 1. Feature Vector Validation and Normalization
        processed_df = self._validate_and_align_features(feature_vector, missing_features, warnings)

        # 2. Lazy load / Query all models via ModelRegistry
        state_regime = "TRANSITION"
        state_probabilities = {"TREND": 0.33, "RANGE": 0.33, "TRANSITION": 0.34}
        state_confidence = 0.0

        break_probability = 0.5
        rejection_probability = 0.5

        trade_quality_score = 0.5
        trade_confidence_score = 0.0

        # Predict - Market State Classifier
        market_state_model = self.model_registry.load_latest_production("MarketStateClassifier")
        if market_state_model is not None:
            try:
                pred = market_state_model.predict(processed_df)

                # Calibrate probabilities
                calibrator = self.calibrators.get("MarketStateClassifier", IdentityCalibrator())
                cal_trend = calibrator.calibrate(pred.trend_probability)
                cal_range = calibrator.calibrate(pred.range_probability)
                cal_transition = calibrator.calibrate(pred.transition_probability)

                # Re-normalize to sum to 1.0
                total_prob = cal_trend + cal_range + cal_transition
                if total_prob > 0.0:
                    cal_trend /= total_prob
                    cal_range /= total_prob
                    cal_transition /= total_prob

                state_probabilities = {
                    "TREND": cal_trend,
                    "RANGE": cal_range,
                    "TRANSITION": cal_transition
                }

                # Map regime to the highest probability class
                regime_idx = int(np.argmax([cal_trend, cal_range, cal_transition]))
                state_regime = ["TREND", "RANGE", "TRANSITION"][regime_idx]
                state_confidence = float(np.max([cal_trend, cal_range, cal_transition]))

                # Update diagnostics
                meta = self.model_registry.get_model_metadata("MarketStateClassifier")
                model_versions["MarketStateClassifier"] = meta.get("version", "unknown") if meta else "unknown"
            except Exception as e:
                err_msg = f"MarketStateClassifier prediction failed: {e}"
                logger.error(err_msg)
                warnings.append(err_msg)
        else:
            warnings.append("MarketStateClassifier model is missing/unavailable.")

        # Predict - Level Break Probability Model
        level_break_model = self.model_registry.load_latest_production("LevelBreakProbabilityModel")
        if level_break_model is not None:
            try:
                pred = level_break_model.predict(processed_df)

                # Calibrate probabilities
                calibrator = self.calibrators.get("LevelBreakProbabilityModel", IdentityCalibrator())
                break_probability = calibrator.calibrate(pred.break_probability)
                rejection_probability = calibrator.calibrate(pred.reject_probability)

                # Update diagnostics
                meta = self.model_registry.get_model_metadata("LevelBreakProbabilityModel")
                model_versions["LevelBreakProbabilityModel"] = meta.get("version", "unknown") if meta else "unknown"
            except Exception as e:
                err_msg = f"LevelBreakProbabilityModel prediction failed: {e}"
                logger.error(err_msg)
                warnings.append(err_msg)
        else:
            warnings.append("LevelBreakProbabilityModel model is missing/unavailable.")

        # Predict - Trade Quality Model
        trade_quality_model = self.model_registry.load_latest_production("TradeQualityModel")
        if trade_quality_model is not None:
            try:
                pred = trade_quality_model.predict(processed_df)

                # Calibrate probabilities
                calibrator = self.calibrators.get("TradeQualityModel", IdentityCalibrator())
                trade_quality_score = calibrator.calibrate(pred.quality_score)
                trade_confidence_score = calibrator.calibrate(pred.confidence)

                # Update diagnostics
                meta = self.model_registry.get_model_metadata("TradeQualityModel")
                model_versions["TradeQualityModel"] = meta.get("version", "unknown") if meta else "unknown"
            except Exception as e:
                err_msg = f"TradeQualityModel prediction failed: {e}"
                logger.error(err_msg)
                warnings.append(err_msg)
        else:
            warnings.append("TradeQualityModel model is missing/unavailable.")

        # 3. Compile context dictionary for Policy Layer
        state_dict = {
            "predicted_state": state_regime,
            "state_confidence": state_confidence,
            "break_probability": break_probability,
            "rejection_probability": rejection_probability,
            "trade_quality_score": trade_quality_score,
            "trade_confidence_score": trade_confidence_score
        }

        # 4. Evaluate Policy
        try:
            policy_rec = self.policy.evaluate(state_dict)
        except Exception as e:
            logger.error(f"Policy evaluation failed: {e}. Falling back to default denial policy.")
            warnings.append(f"Policy error: {e}")
            policy_rec = PolicyRecommendation(
                allow_trade=False,
                suggested_risk_multiplier=1.0,
                suggested_position_scale=1.0,
                suggested_tp_mode="STRUCTURE_TARGET",
                suggested_sl_adjustment=0.0
            )

        # Calculate inference duration
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000.0

        # Log details
        logger.info(
            f"Symbol: {symbol} | Timeframe: {timeframe} | Latency: {inference_time_ms:.2f}ms | "
            f"State: {state_regime} ({state_confidence:.2f}) | Quality: {trade_quality_score:.2f} | "
            f"Allow Trade: {policy_rec.allow_trade}"
        )

        # 5. Return Immutable DecisionContext
        return DecisionContext(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            predicted_state=state_regime,
            state_probabilities=state_probabilities,
            state_confidence=state_confidence,
            break_probability=break_probability,
            rejection_probability=rejection_probability,
            trade_quality_score=trade_quality_score,
            confidence_score=trade_confidence_score,
            policy_recommendation=policy_rec,
            model_versions=model_versions,
            inference_time_ms=inference_time_ms,
            missing_features=missing_features,
            warnings=warnings
        )

    def _validate_and_align_features(
        self,
        feature_vector: Union[Dict[str, Any], pd.DataFrame, np.ndarray],
        missing_features: List[str],
        warnings: List[str]
    ) -> pd.DataFrame:
        """
        Validates the input feature vector against the FeatureRegistry schema.
        Ensures strict matching of sizes, types, and sequence, filling missing values
        with feature defaults. Returns a standardized 1-row pandas DataFrame.
        """
        # If dictionary snapshot
        if isinstance(feature_vector, dict):
            aligned_dict = {}
            for name in self._enabled_features:
                if name in feature_vector:
                    # Type checking/coercion
                    feat_def = self.feature_registry.get(name)
                    raw_val = feature_vector[name]
                    try:
                        if feat_def.dtype in [float, "float"]:
                            aligned_dict[name] = float(raw_val)
                        elif feat_def.dtype in [int, "int"]:
                            aligned_dict[name] = int(raw_val)
                        else:
                            aligned_dict[name] = str(raw_val)
                    except (ValueError, TypeError):
                        warnings.append(f"Type mismatch/coercion failed for feature {name}. Value: {raw_val}")
                        aligned_dict[name] = feat_def.default_value
                else:
                    missing_features.append(name)
                    aligned_dict[name] = self.feature_registry.get(name).default_value

            if missing_features:
                logger.warning(f"Feature vector validation warning: {len(missing_features)} features missing.")
            return pd.DataFrame([aligned_dict])

        # If pandas DataFrame
        elif isinstance(feature_vector, pd.DataFrame):
            # Align features using FeatureRegistry
            # Select the first row of input DataFrame for snapshot evaluation
            row_dict = feature_vector.iloc[0].to_dict() if len(feature_vector) > 0 else {}
            return self._validate_and_align_features(row_dict, missing_features, warnings)

        # If raw numpy array
        elif isinstance(feature_vector, np.ndarray):
            flat_vector = feature_vector.flatten()
            if len(flat_vector) != self._enabled_feature_count:
                err_msg = f"Input feature vector shape {flat_vector.shape} does not match active FeatureRegistry count {self._enabled_feature_count}."
                logger.error(err_msg)
                warnings.append(err_msg)
                # Map raw features onto enabled names by sequence, truncating/padding with defaults
                aligned_dict = {}
                for idx, name in enumerate(self._enabled_features):
                    if idx < len(flat_vector):
                        aligned_dict[name] = flat_vector[idx]
                    else:
                        missing_features.append(name)
                        aligned_dict[name] = self.feature_registry.get(name).default_value
                return pd.DataFrame([aligned_dict])
            else:
                # Direct sequence mapping
                aligned_dict = {name: flat_vector[idx] for idx, name in enumerate(self._enabled_features)}
                return pd.DataFrame([aligned_dict])

        else:
            raise TypeError("Unsupported feature_vector type. Must be dict, pandas.DataFrame, or numpy.ndarray.")
