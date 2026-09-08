import pytest
import pandas as pd
import numpy as np
from ML.level_event_labeler import LevelEventLabeler
from Market_Data_Pipeline.structure_graph import MarketStructureGraph, Zone


def test_level_event_labeler_scenarios():
    labeler = LevelEventLabeler(
        proximity_atr=0.5,
        break_buffer_atr=0.2,
        rejection_distance_atr=1.0,
        future_horizon=10
    )

    supply_zone = Zone(
        upper=1.1020,
        lower=1.1000,
        type="Supply",
        created_idx=0
    )

    msg = MarketStructureGraph(symbol="EURUSD", timeframe="M5")

    # 1. Break scenario: Price closes decisively above supply.upper (1.1020 + buffer)
    df_break = pd.DataFrame({
        "Open": [1.0990] * 12,
        "High": [1.0995] + [1.1030] * 11,
        "Low": [1.0985] + [1.0990] * 11,
        "Close": [1.0990] + [1.1025] * 11, # 1.1025 > 1.1020 + 0.2*0.0010 (1.1022)
        "atr_14": [0.0010] * 12
    })

    res_break = labeler.evaluate_level_event(df_break, msg, anchor_idx=0, zone=supply_zone)
    assert res_break.event_type == "BREAK"
    assert res_break.break_probability_target == 1
    assert res_break.touch_detected is True

    # 2. Rejection scenario: Price touches supply, then drops below zone.lower - 1.0*ATR
    df_reject = pd.DataFrame({
        "Open": [1.0990] * 12,
        "High": [1.0990] + [1.1005] + [1.0990] * 10,
        "Low": [1.0985] + [1.0985] + [1.0985] * 10, # fl = 1.0985 < 1.1000 - 1.0*0.0010 (1.0990)
        "Close": [1.0990] + [1.0995] + [1.0988] * 10,
        "atr_14": [0.0010] * 12
    })

    res_reject = labeler.evaluate_level_event(df_reject, msg, anchor_idx=0, zone=supply_zone)
    assert res_reject.event_type in ["REJECTION", "SWEEP_REJECTION"]
    assert res_reject.break_probability_target == 0

    # 3. No interaction scenario: Price stays far away
    df_no_touch = pd.DataFrame({
        "Open": [1.0800] * 12,
        "High": [1.0805] * 12,
        "Low": [1.0795] * 12,
        "Close": [1.0800] * 12,
        "atr_14": [0.0010] * 12
    })

    res_no_touch = labeler.evaluate_level_event(df_no_touch, msg, anchor_idx=0, zone=supply_zone)
    assert res_no_touch.event_type == "NO_INTERACTION"
    assert res_no_touch.break_probability_target is None
