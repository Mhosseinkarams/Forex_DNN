import pytest
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from ML.label_engine import LabelEngine
from ML.market_state_labeler import MarketStateLabeler
from ML.dataset_validator import DatasetValidator
from ML.generate_dataset import generate_synthetic_candles


def test_market_state_labeler_rules():
    """
    Tests that the MarketStateLabeler correctly determines TREND, RANGE, and TRANSITION states
    given custom, predefined dataframes and structural indicators.
    """
    labeler = MarketStateLabeler(
        ema_separation_trend=1.5,
        ema_separation_range=0.8,
        min_bos_trend=1,
        min_rejections_range=2
    )

    # 1. Mock a TREND scenario: wide separation, 1 BOS, no CHOCH
    df_trend = pd.DataFrame({
        "Close": [1.1000] * 35,
        "ema_50": [1.1020] * 35,
        "ema_600": [1.1000] * 35,
        "atr_14": [0.0010] * 35,  # 20 pips. Separation is 20 pips / 10 pips = 2.0 ATR
    })

    # Create mock structure graph with 1 BOS
    from Market_Data_Pipeline.structure_graph import MarketStructureGraph, BOS, CHOCH
    msg_trend = MarketStructureGraph(
        symbol="EURUSD",
        timeframe="M15",
        bos=[BOS(index=15, direction=1, broken_level=1.1010)],
        choch=[]
    )

    lbl, conf, info = labeler.label_window(df_trend, msg_trend, 0, 34)
    assert lbl == "TREND"
    assert conf >= 0.5
    assert info["rule_fired"] == "trend_ema_sep_and_bos"

    # 2. Mock a RANGE scenario: converged EMAs, no BOS
    df_range = pd.DataFrame({
        "Close": [1.1000] * 35,
        "ema_50": [1.1005] * 35,
        "ema_600": [1.1000] * 35,
        "atr_14": [0.0010] * 35,  # 10 pips. Separation is 5 pips / 10 pips = 0.5 ATR (converged < 0.8)
    })
    msg_range = MarketStructureGraph(
        symbol="EURUSD",
        timeframe="M15",
        bos=[],
        choch=[]
    )

    lbl, conf, info = labeler.label_window(df_range, msg_range, 0, 34)
    assert lbl == "RANGE"
    assert info["rule_fired"] == "range_converged_or_retests"

    # 3. Mock a TRANSITION scenario: crossed EMAs or a CHOCH
    df_trans = pd.DataFrame({
        "Close": [1.1000] * 35,
        # EMA cross from 1.0990 (fast < slow) to 1.1010 (fast > slow)
        "ema_50": np.linspace(1.0990, 1.1010, 35),
        "ema_600": [1.1000] * 35,
        "atr_14": [0.0010] * 35,
    })
    msg_trans = MarketStructureGraph(
        symbol="EURUSD",
        timeframe="M15",
        bos=[],
        choch=[CHOCH(index=20, previous_trend=-1, new_trend=1)]
    )

    lbl, conf, info = labeler.label_window(df_trans, msg_trans, 0, 34)
    assert lbl == "TRANSITION"
    assert info["rule_fired"] == "transition_cross_or_choch_or_shrink"


def test_label_engine_sliding_window():
    """
    Verifies sliding window generation with stride, removal of unlabeled/indeterminate samples,
    and manifest serialization.
    """
    df_raw = generate_synthetic_candles(num_bars=100, seed=42)

    # Initialize with default configurations
    engine = LabelEngine(window_size=35, window_stride=5)

    df_labeled = engine.generate_dataset(
        data_inputs=df_raw,
        symbol="GBPUSD",
        timeframe="M5"
    )

    # Total windows: (100 - 35)/5 + 1 = 14 windows
    assert engine.total_windows_processed == 14
    assert len(df_labeled) + engine.removed_samples_count == 14

    # Save manifest and verify structure
    out_csv = "output/test_label_engine_out.csv"
    out_manifest = "output/test_label_engine_manifest.json"

    engine.save_dataset_and_manifest(df_labeled, out_csv, out_manifest)

    assert os.path.exists(out_csv)
    assert os.path.exists(out_manifest)

    import json
    with open(out_manifest, "r") as f:
        manifest = json.load(f)

    assert manifest["window_size"] == 35
    assert manifest["window_stride"] == 5
    assert manifest["symbols"] == ["GBPUSD"]
    assert manifest["timeframes"] == ["M5"]
    assert "total_windows_generated" in manifest
    assert "samples_removed_due_to_missing_labels" in manifest
    assert "final_class_distribution" in manifest


def test_dataset_validator():
    """
    Tests that DatasetValidator catches missing columns, duplicates, monotonic errors, and incorrect sizes.
    """
    # Build a perfectly valid dataframe
    df_valid = pd.DataFrame({
        "symbol": ["EURUSD"] * 5,
        "timeframe": ["M15"] * 5,
        "datetime": ["2026-01-01T00:00:00", "2026-01-01T00:15:00", "2026-01-01T00:30:00", "2026-01-01T00:45:00", "2026-01-01T01:00:00"],
        "window_start": [0, 1, 2, 3, 4],
        "window_end": [34, 35, 36, 37, 38],
        "target": ["TREND", "RANGE", "TREND", "TRANSITION", "RANGE"],
        "confidence": [0.8, 0.9, 0.7, 0.6, 0.9],
        "ema50_slope": [0.01, 0.02, 0.01, -0.01, 0.0],
        "atr": [10.0, 11.0, 10.5, 9.8, 10.2]
    })

    validator = DatasetValidator(critical_missing_threshold=0.0)

    # 1. Check valid
    report = validator.validate(df_valid, expected_window_size=35)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0

    # 2. Check missing value in target (critical)
    df_invalid_missing = df_valid.copy()
    df_invalid_missing.loc[2, "target"] = None
    report_missing = validator.validate(df_invalid_missing, expected_window_size=35)
    assert report_missing["is_valid"] is False
    assert any("target" in err for err in report_missing["errors"])

    # 3. Check inconsistent window sizes
    df_invalid_window = df_valid.copy()
    df_invalid_window.loc[2, "window_end"] = 40  # 40 - 2 + 1 = 39 != 35
    report_window = validator.validate(df_invalid_window, expected_window_size=35)
    assert report_window["is_valid"] is False
    assert any("window size" in err for err in report_window["errors"])

    # 4. Check duplicate timestamps
    df_invalid_dup = df_valid.copy()
    df_invalid_dup.loc[2, "datetime"] = "2026-01-01T00:15:00"  # Duplicate of index 1
    report_dup = validator.validate(df_invalid_dup, expected_window_size=35)
    assert report_dup["is_valid"] is False
    assert any("duplicate timestamp" in err.lower() for err in report_dup["errors"])
