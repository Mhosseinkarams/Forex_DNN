import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone

from ML.feature_registry import FeatureRegistry
from ML.feature_pipeline import FeaturePipeline
from ML.trade_feature_recorder import TradeFeatureRecorder
from Strategies.signal_evaluator import SignalEvaluator, SignalEvaluation
from Market_Data_Pipeline.structure_graph import MarketStructureGraph
from Collecting_Data.position_lifecycle import PositionLifecycle, SignalInfo, ExecutionInfo, ManagementInfo, OutcomeInfo

def test_feature_pipeline_runtime_extraction():
    registry = FeatureRegistry(load_defaults=True)
    pipeline = FeaturePipeline(registry)

    # Create simple dummy dataframe with necessary columns for indicators
    df = pd.DataFrame({
        "Datetime": pd.date_range("2026-07-10 12:00:00", periods=50, freq="5min"),
        "Open": np.linspace(1.1200, 1.1250, 50),
        "High": np.linspace(1.1210, 1.1260, 50),
        "Low": np.linspace(1.1190, 1.1240, 50),
        "Close": np.linspace(1.1205, 1.1255, 50),
        "TickVolume": np.ones(50) * 100.0,
        "Spread": np.ones(50) * 1.5,
        "ema_50": np.linspace(1.1200, 1.1240, 50),
        "ema_600": np.linspace(1.1150, 1.1170, 50),
        "ema_800": np.linspace(1.1140, 1.1160, 50),
        "atr_14": np.ones(50) * 0.0010,
        "body_pct": np.ones(50) * 0.75,
        "body_vs_avg": np.ones(50) * 1.2
    })

    # Create MarketStructureGraph
    msg = MarketStructureGraph(
        symbol="EURUSD",
        timeframe="M5",
        timestamp=datetime(2026, 7, 10, 23, 45, tzinfo=timezone.utc),
        swing_highs=[],
        swing_lows=[],
        protected_high=None,
        protected_low=None,
        bos=[],
        choch=[],
        supply_zones=[],
        demand_zones=[],
        trend_direction="Bull",
        atr=0.0010,
        volatility=10.0
    )

    # Extract runtime features
    fv = pipeline.extract_runtime(df=df, msg=msg, idx=-1, return_df=False)

    # Basic assertions
    assert fv.metadata["symbol"] == "EURUSD"
    assert fv.metadata["timeframe"] == "M5"
    assert len(fv.features) == len(registry.list_enabled())

    # Verify NaN/Inf handling
    # Let's inject a NaN close price to trigger coercion and default fallback
    df_with_nan = df.copy()
    df_with_nan.loc[49, "body_pct"] = np.nan

    fv_nan = pipeline.extract_runtime(df=df_with_nan, msg=msg, idx=-1)
    # The nan feature should fallback to its registry default_value
    assert fv_nan.features["candle_body"] == registry.get("candle_body").default_value

def test_signal_evaluator():
    evaluator = SignalEvaluator(shadow_mode=True, ml_filtering=False)

    # Setup simulated candidate and decision context
    candidate = {
        "symbol": "GBPUSD",
        "timeframe": "M15",
        "direction": -1,
        "signal_type": "reversal",
        "technical_rules_satisfied": True
    }

    from ML.decision_context import DecisionContext, PolicyRecommendation
    mock_policy = PolicyRecommendation(
        allow_trade=True,
        suggested_risk_multiplier=1.0,
        suggested_position_scale=1.0,
        suggested_tp_mode="STRUCTURE_TARGET",
        suggested_sl_adjustment=0.0
    )

    decision_ctx = DecisionContext(
        symbol="GBPUSD",
        timeframe="M15",
        timestamp="2026-07-10T12:00:00Z",
        predicted_state="RANGE",
        state_probabilities={"TREND": 0.1, "RANGE": 0.8, "TRANSITION": 0.1},
        state_confidence=0.8,
        break_probability=0.2,
        rejection_probability=0.8,
        trade_quality_score=0.75,
        confidence_score=0.8,
        policy_recommendation=mock_policy,
        model_versions={},
        inference_time_ms=5.0,
        missing_features=[],
        warnings=[]
    )

    msg = MarketStructureGraph(
        symbol="GBPUSD",
        timeframe="M15",
        timestamp=datetime.now(timezone.utc),
        swing_highs=[],
        swing_lows=[],
        protected_high=None,
        protected_low=None,
        bos=[],
        choch=[],
        supply_zones=[],
        demand_zones=[],
        trend_direction="Neutral",
        atr=0.0012,
        volatility=12.0
    )

    evaluation = evaluator.evaluate(
        strategy_name="MMStrategy",
        signal_candidate=candidate,
        feature_vector={},
        decision_context=decision_ctx,
        market_structure=msg,
        risk_state={"trading_allowed": True}
    )

    assert evaluation.accepted is True
    assert evaluation.rejected is False
    assert evaluation.priority == "NORMAL"
    assert any("ML predicts RANGE" in r for r in evaluation.reasons)
    assert any("Shadow Mode only" in r for r in evaluation.reasons)

def test_trade_feature_recorder_csv_parquet():
    temp_dir = tempfile.mkdtemp()
    try:
        recorder = TradeFeatureRecorder(storage_dir=temp_dir, file_format="both")

        signal_id = "test-signal-1"
        timestamp = "2026-07-10T14:35:00Z"
        features = {"ema50_slope": 0.5, "atr": 15.0}

        # 1. Record candidate
        recorder.record_candidate(
            signal_id=signal_id,
            timestamp=timestamp,
            strategy="mm",
            symbol="EURUSD",
            timeframe="M5",
            direction="BUY",
            features=features,
            decision_context=None,
            accepted=True,
            reason="Technical setup matches"
        )

        csv_file = os.path.join(temp_dir, "recorded_features_20260710.csv")
        parquet_file = os.path.join(temp_dir, "recorded_features_20260710.parquet")

        assert os.path.exists(csv_file)
        assert os.path.exists(parquet_file)

        # Verify candidate values
        df_csv = pd.read_csv(csv_file)
        assert len(df_csv) == 1
        assert df_csv.iloc[0]["signal_id"] == signal_id
        assert df_csv.iloc[0]["feature_ema50_slope"] == 0.5
        assert pd.isna(df_csv.iloc[0]["trade_profit"]) # Unfilled outcome

        # 2. Record outcome
        sig_info = SignalInfo(signal_id=signal_id, strategy="mm", signal_category="standard", exit_profile="standard",
                              symbol="EURUSD", timeframe="M5", direction=1, signal_timestamp=timestamp, bar_timestamp=timestamp)
        exec_info = ExecutionInfo(ticket=123, magic_number=999, requested_entry=1.1200, actual_entry=1.1200, average_entry=1.1200,
                                  initial_volume=0.1, remaining_volume=0.0, risk_percent=1.0, risk_amount=100.0,
                                  initial_stop_loss=1.1180, initial_take_profit=1.1240, spread=1.5, slippage=0.0, execution_latency=0.0)
        mgt_info = ManagementInfo()
        out_info = OutcomeInfo(exit_timestamp="2026-07-10T15:00:00Z", average_exit_price=1.1240, close_price=1.1240,
                               realized_profit=400.0, profit_points=40.0, profit_pips=40.0, profit_percent=4.0,
                               r_multiple=2.0, result="WIN", strategy_reason="take_profit", broker_reason="tp",
                               deal_count=2, partial_close_count=0, duration=1500.0, status="completed")

        lifecycle = PositionLifecycle(signal=sig_info, execution=exec_info, management=mgt_info, outcome=out_info)

        recorder.record_outcome(signal_id=signal_id, lifecycle=lifecycle)

        # Reload and check outcome appending
        df_csv_updated = pd.read_csv(csv_file)
        assert len(df_csv_updated) == 1
        assert df_csv_updated.iloc[0]["trade_outcome"] == "WIN"
        assert df_csv_updated.iloc[0]["trade_profit"] == 400.0
        assert df_csv_updated.iloc[0]["trade_exit_reason"] == "take_profit"

    finally:
        shutil.rmtree(temp_dir)
