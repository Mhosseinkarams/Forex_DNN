import os
import tempfile
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from ML.feature_registry import FeatureRegistry
from ML.model_registry import ModelRegistry
from ML.decision_context import DecisionContext, PolicyRecommendation
from ML.confidence_calibrator import IdentityCalibrator, PlattCalibrator, IsotonicCalibrator
from ML.policy import RuleBasedPolicy, RLPolicy
from ML.ml_decision_engine import MLDecisionEngine
from ML.models.market_state_classifier import MarketStateClassifier
from ML.models.level_break_probability import LevelBreakProbabilityModel
from ML.models.trade_quality_model import TradeQualityModel


def test_confidence_calibrators():
    # Identity Calibrator
    ident = IdentityCalibrator()
    assert ident.calibrate(0.7) == 0.7
    assert ident.calibrate(1.5) == 1.0
    assert ident.calibrate(-0.2) == 0.0

    # Platt Calibrator (Sigmoid mapper)
    platt = PlattCalibrator(A=-1.0, B=0.0)
    # 0.5 probability should yield 0.5 (logit is 0, exp(0) is 1, 1/(1+1) is 0.5)
    assert pytest.approx(platt.calibrate(0.5), rel=1e-3) == 0.5
    assert platt.calibrate(0.9) > 0.5
    assert platt.calibrate(0.1) < 0.5

    # Isotonic Calibrator (Piecewise linear mapper)
    iso = IsotonicCalibrator(thresholds=[0.0, 0.5, 1.0], targets=[0.0, 0.4, 1.0])
    assert pytest.approx(iso.calibrate(0.5), rel=1e-3) == 0.4
    assert pytest.approx(iso.calibrate(0.25), rel=1e-3) == 0.2
    assert pytest.approx(iso.calibrate(0.75), rel=1e-3) == 0.7


def test_rule_based_policy():
    policy = RuleBasedPolicy(min_state_confidence=0.6, min_trade_quality=0.6)

    # Favorable, high confidence, good quality -> True
    state_dict_ok = {
        "predicted_state": "TREND",
        "state_confidence": 0.8,
        "break_probability": 0.2,
        "rejection_probability": 0.8,
        "trade_quality_score": 0.7,
        "trade_confidence_score": 0.8
    }
    rec_ok = policy.evaluate(state_dict_ok)
    assert rec_ok.allow_trade is True
    assert rec_ok.suggested_tp_mode == "REJECTION_TARGET"
    assert rec_ok.suggested_sl_adjustment == -0.2

    # Breakout setup
    state_dict_bo = {
        "predicted_state": "TREND",
        "state_confidence": 0.8,
        "break_probability": 0.8,
        "rejection_probability": 0.2,
        "trade_quality_score": 0.9,
        "trade_confidence_score": 0.8
    }
    rec_bo = policy.evaluate(state_dict_bo)
    assert rec_bo.allow_trade is True
    assert rec_bo.suggested_tp_mode == "BREAKOUT_TARGET"
    assert rec_bo.suggested_risk_multiplier == 1.5
    assert rec_bo.suggested_position_scale == 1.2
    assert rec_bo.suggested_sl_adjustment == 0.5

    # Low state confidence -> False
    state_dict_low_conf = {
        "predicted_state": "TREND",
        "state_confidence": 0.4,
        "break_probability": 0.2,
        "rejection_probability": 0.8,
        "trade_quality_score": 0.7,
        "trade_confidence_score": 0.8
    }
    rec_lc = policy.evaluate(state_dict_low_conf)
    assert rec_lc.allow_trade is False

    # Low trade quality -> False
    state_dict_low_qual = {
        "predicted_state": "TREND",
        "state_confidence": 0.8,
        "break_probability": 0.2,
        "rejection_probability": 0.8,
        "trade_quality_score": 0.3,
        "trade_confidence_score": 0.8
    }
    rec_lq = policy.evaluate(state_dict_low_qual)
    assert rec_lq.allow_trade is False


def test_ml_decision_engine_missing_models():
    # Write empty model registry json file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump({"models": []}, tmp)
        tmp_name = tmp.name

    try:
        engine = MLDecisionEngine(registry_path=tmp_name)
        # Empty registry should log warnings and return default DecisionContext values
        feature_registry = FeatureRegistry(load_defaults=True)
        enabled_count = len(feature_registry.list_enabled())
        dummy_vector = np.zeros(enabled_count)

        context = engine.evaluate("EURUSD", "M15", dummy_vector, "TestStrategy")

        assert isinstance(context, DecisionContext)
        assert context.predicted_state == "TRANSITION"
        assert context.policy_recommendation.allow_trade is False
        assert len(context.warnings) > 0
        assert "MarketStateClassifier model is missing/unavailable." in context.warnings
        assert "LevelBreakProbabilityModel model is missing/unavailable." in context.warnings
        assert "TradeQualityModel model is missing/unavailable." in context.warnings
    finally:
        os.remove(tmp_name)


def test_feature_vector_validation_and_coercion():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump({"models": []}, tmp)
        tmp_name = tmp.name

    try:
        engine = MLDecisionEngine(registry_path=tmp_name)

        # Test input dict with partial/missing features
        partial_dict = {
            "ema_50_distance": 1.5,
            "atr_ratio": "0.85"  # String representation to test type coercion
        }

        # This shouldn't raise any exception but fill with defaults and coerce atr_ratio to float
        context = engine.evaluate("GBPUSD", "H1", partial_dict, "TestStrategy")
        assert len(context.missing_features) > 0
        assert "ema_50_distance" not in context.missing_features
        assert "atr_ratio" not in context.missing_features

        # Test 1D numpy array with incorrect length
        incorrect_vector = np.array([1.0, 2.0])
        context_np = engine.evaluate("GBPUSD", "H1", incorrect_vector, "TestStrategy")
        assert len(context_np.warnings) > 0
        assert any("does not match active FeatureRegistry count" in w for w in context_np.warnings)
    finally:
        os.remove(tmp_name)


def test_model_registry_lazy_loading_and_caching():
    # Setup temporary registry file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump({
            "models": [
                {
                    "model_name": "MarketStateClassifier",
                    "version": "1.0.2",
                    "model_path": "models/dummy_market_state.joblib",
                    "metrics": {"accuracy": 0.75},
                    "dataset_version": "v001",
                    "dataset_hash": "dummy_hash",
                    "feature_registry_version": "dummy_feat_hash",
                    "model_type": "lightgbm",
                    "is_production": True,
                    "registered_at": "2026-07-10T12:00:00"
                }
            ]
        }, tmp)
        tmp_name = tmp.name

    try:
        registry = ModelRegistry(registry_path=tmp_name)

        # If the file doesn't exist, loading should log a warning and return None
        model = registry.load_latest_production("MarketStateClassifier")
        assert model is None

        # Mock load method of MarketStateClassifier
        mock_model = MagicMock()
        mock_model.predict.return_value = MagicMock()

        with patch.object(MarketStateClassifier, "load", return_value=mock_model) as mock_load:
            with patch("os.path.exists", return_value=True):
                # First load should call load
                model1 = registry.load_latest_production("MarketStateClassifier")
                assert model1 is mock_model
                mock_load.assert_called_once_with("models/dummy_market_state.joblib")

                # Second load should retrieve from cache (mock_load called count remains 1)
                model2 = registry.load_latest_production("MarketStateClassifier")
                assert model2 is mock_model
                assert mock_load.call_count == 1
    finally:
        os.remove(tmp_name)


def test_complete_ml_decision_engine_inference_pipeline():
    # Fully mock out the models so we can run a complete simulated prediction loop
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump({
            "models": [
                {
                    "model_name": "MarketStateClassifier",
                    "version": "2.1.0",
                    "model_path": "models/m_state.joblib",
                    "metrics": {"accuracy": 0.8},
                    "dataset_version": "v002",
                    "dataset_hash": "hash1",
                    "feature_registry_version": "feat_hash",
                    "model_type": "lightgbm",
                    "is_production": True
                },
                {
                    "model_name": "LevelBreakProbabilityModel",
                    "version": "1.4.5",
                    "model_path": "models/l_break.joblib",
                    "metrics": {"accuracy": 0.72},
                    "dataset_version": "v002",
                    "dataset_hash": "hash2",
                    "feature_registry_version": "feat_hash",
                    "model_type": "lightgbm",
                    "is_production": True
                },
                {
                    "model_name": "TradeQualityModel",
                    "version": "1.0.0",
                    "model_path": "models/t_quality.joblib",
                    "metrics": {"accuracy": 0.85},
                    "dataset_version": "v002",
                    "dataset_hash": "hash3",
                    "feature_registry_version": "feat_hash",
                    "model_type": "lightgbm",
                    "is_production": True
                }
            ]
        }, tmp)
        tmp_name = tmp.name

    try:
        # Create mock predictions
        mock_mstate_pred = MagicMock()
        mock_mstate_pred.trend_probability = 0.7
        mock_mstate_pred.range_probability = 0.2
        mock_mstate_pred.transition_probability = 0.1

        mock_lbreak_pred = MagicMock()
        mock_lbreak_pred.break_probability = 0.85
        mock_lbreak_pred.reject_probability = 0.15

        mock_tquality_pred = MagicMock()
        mock_tquality_pred.quality_score = 0.75
        mock_tquality_pred.confidence = 0.8

        # Mock class instances
        mstate_instance = MagicMock()
        mstate_instance.predict.return_value = mock_mstate_pred

        lbreak_instance = MagicMock()
        lbreak_instance.predict.return_value = mock_lbreak_pred

        tquality_instance = MagicMock()
        tquality_instance.predict.return_value = mock_tquality_pred

        with patch("os.path.exists", return_value=True):
            with patch.object(MarketStateClassifier, "load", return_value=mstate_instance):
                with patch.object(LevelBreakProbabilityModel, "load", return_value=lbreak_instance):
                    with patch.object(TradeQualityModel, "load", return_value=tquality_instance):

                        engine = MLDecisionEngine(
                            registry_path=tmp_name,
                            policy=RuleBasedPolicy(min_state_confidence=0.5, min_trade_quality=0.5),
                            calibrators={
                                "MarketStateClassifier": PlattCalibrator(A=-1.0, B=0.0),
                                "LevelBreakProbabilityModel": IdentityCalibrator(),
                                "TradeQualityModel": IsotonicCalibrator()
                            }
                        )

                        # Process evaluation snapshot
                        feature_registry = FeatureRegistry(load_defaults=True)
                        enabled_count = len(feature_registry.list_enabled())
                        dummy_vector = np.zeros(enabled_count)

                        context = engine.evaluate("EURUSD", "M5", dummy_vector, "MMStrategy")

                        # Validate DecisionContext compiles correctly
                        assert isinstance(context, DecisionContext)
                        assert context.symbol == "EURUSD"
                        assert context.timeframe == "M5"

                        # Calibrated state confidence check (should map TREND to highest probability)
                        assert context.predicted_state == "TREND"
                        assert context.state_confidence > 0.5
                        assert context.break_probability == 0.85
                        assert context.rejection_probability == 0.15
                        assert context.trade_quality_score > 0.0

                        # Check policy evaluations
                        assert isinstance(context.policy_recommendation, PolicyRecommendation)
                        assert context.policy_recommendation.allow_trade is True
                        assert context.policy_recommendation.suggested_tp_mode == "BREAKOUT_TARGET"
                        assert context.policy_recommendation.suggested_sl_adjustment == 0.5

                        # Check version diagnostics
                        assert context.model_versions["MarketStateClassifier"] == "2.1.0"
                        assert context.model_versions["LevelBreakProbabilityModel"] == "1.4.5"
                        assert context.model_versions["TradeQualityModel"] == "1.0.0"
                        assert context.inference_time_ms > 0.0
                        assert len(context.warnings) == 0
    finally:
        os.remove(tmp_name)
