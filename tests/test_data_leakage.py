import pytest
import pandas as pd
import numpy as np
from ML.feature_registry import FeatureRegistry, TARGET_COLUMNS
from ML.feature_pipeline import FeaturePipeline
from ML.dataset_builder import DatasetBuilder
from ML.trainer import Trainer
from Market_Data_Pipeline.structure_engine import MarketStructureEngine
from Market_Data_Pipeline.supply_demand_engine import SupplyDemandEngine
from Market_Data_Pipeline.structure_graph import MarketStructureGraph, StructureLevel, Zone


def test_target_feature_isolation():
    """
    Verifies that no target column is registered in FeatureRegistry.
    """
    reg = FeatureRegistry(load_defaults=True)
    enabled_names = {f.name for f in reg.list_enabled()}

    overlap = enabled_names.intersection(TARGET_COLUMNS)
    assert len(overlap) == 0, f"Target leakage error: Found target columns in FeatureRegistry: {overlap}"


def test_future_mutation_leakage():
    """
    Verifies that mutating price data at candle t + 5 does NOT alter feature vector extracted at candle t.
    """
    reg = FeatureRegistry(load_defaults=True)
    pipeline = FeaturePipeline(registry=reg)

    n_bars = 100
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="5min")

    df_original = pd.DataFrame({
        "Datetime": dates,
        "Open": np.linspace(1.1000, 1.1100, n_bars),
        "High": np.linspace(1.1010, 1.1110, n_bars),
        "Low": np.linspace(1.0990, 1.1090, n_bars),
        "Close": np.linspace(1.1005, 1.1105, n_bars),
        "TickVolume": [100.0] * n_bars,
        "ema_50": np.linspace(1.1000, 1.1100, n_bars),
        "ema_600": [1.1000] * n_bars,
        "atr_14": [0.0010] * n_bars
    })

    ms_engine = MarketStructureEngine(lookback=3)
    df_struct = ms_engine.process(df_original)

    msg1 = MarketStructureGraph(
        symbol="EURUSD",
        timeframe="M5",
        swing_highs=[s for s in ms_engine.swings if s.level_type == "SwingHigh"],
        swing_lows=[s for s in ms_engine.swings if s.level_type == "SwingLow"],
        bos=list(ms_engine.bos_list),
        choch=list(ms_engine.choch_list)
    )

    anchor_t = 50
    feats_before = pipeline.extract_all(df_struct, msg1, idx=anchor_t)

    # Mutate future candles > anchor_t (candles 55..99)
    df_mutated = df_struct.copy()
    df_mutated.loc[55:, "High"] += 10.0
    df_mutated.loc[55:, "Close"] += 10.0

    pipeline.clear_cache()
    feats_after = pipeline.extract_all(df_mutated, msg1, idx=anchor_t)

    for k in feats_before.keys():
        val_before = feats_before[k]
        val_after = feats_after[k]
        if isinstance(val_before, (int, float, np.number)):
            assert val_before == pytest.approx(val_after, abs=1e-6), f"Leakage detected in feature {k} at anchor {anchor_t} after future mutation!"


def test_structure_causality_point_in_time():
    """
    Verifies that a swing level confirmed at candle i + k is NOT returned by point-in-time query at t < i + k.
    """
    msg = MarketStructureGraph(symbol="EURUSD", timeframe="M5")

    # Swing high at candle 30, confirmed at candle 33
    swing_high = StructureLevel(
        price=1.1050,
        index=30,
        confirmation_candle=33,
        level_type="SwingHigh"
    )
    msg.swing_highs.append(swing_high)

    # Query at candle 32 (before confirmation) -> must return empty
    swings_at_32 = msg.get_confirmed_swings_high(idx=32)
    assert len(swings_at_32) == 0

    # Query at candle 33 (at confirmation) -> must return swing
    swings_at_33 = msg.get_confirmed_swings_high(idx=33)
    assert len(swings_at_33) == 1
    assert swings_at_33[0].price == 1.1050


def test_purged_chronological_split():
    """
    Verifies that Trainer's purged chronological split leaves a purge gap between train and validation sets.
    """
    n_samples = 1000
    df = pd.DataFrame({
        "feature_1": np.random.randn(n_samples),
        "target": np.random.choice([0, 1], size=n_samples)
    })

    from ML.models.market_state_classifier import MarketStateClassifier
    clf = MarketStateClassifier()

    trainer = Trainer(random_seed=42)
    res = trainer.train_model(
        model=clf,
        df=df,
        target_col="target",
        feature_cols=["feature_1"],
        test_size=0.2,
        chronological=True,
        purge_window=50
    )

    X_train = res["X_train"]
    X_val = res["X_val"]

    last_train_idx = X_train.index[-1]
    first_val_idx = X_val.index[0]

    purged_count = len(df) - (len(X_train) + len(X_val))
    assert purged_count == 50, f"Expected purge gap of 50 samples, got {purged_count}"
