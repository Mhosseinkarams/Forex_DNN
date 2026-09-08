import pytest
import pandas as pd
import numpy as np
from ML.strategy_outcome_evaluator import StrategyOutcomeEvaluator


def test_strategy_outcome_evaluator_scenarios():
    evaluator = StrategyOutcomeEvaluator(future_horizon=10)

    # 1. WIN scenario (BUY trade hitting TP first)
    df_win = pd.DataFrame({
        "Open": [1.1000] * 11,
        "High": [1.1005] + [1.1025] * 10,  # Hits TP 1.1020
        "Low": [1.0995] + [1.0998] * 10,   # Does NOT hit SL 1.0990
        "Close": [1.1000] + [1.1022] * 10
    })

    res_win = evaluator.evaluate_outcome(
        df=df_win,
        anchor_idx=0,
        direction=1,
        entry_price=1.1000,
        stop_loss=1.0990,
        take_profit=1.1020
    )

    assert res_win.outcome == "WIN"
    assert res_win.r_multiple == pytest.approx(2.0, abs=0.01)
    assert res_win.first_hit == "TP"

    # 2. LOSS scenario (BUY trade hitting SL first)
    df_loss = pd.DataFrame({
        "Open": [1.1000] * 11,
        "High": [1.1005] + [1.1005] * 10,
        "Low": [1.0995] + [1.0988] * 10,   # Hits SL 1.0990
        "Close": [1.1000] + [1.0992] * 10
    })

    res_loss = evaluator.evaluate_outcome(
        df=df_loss,
        anchor_idx=0,
        direction=1,
        entry_price=1.1000,
        stop_loss=1.0990,
        take_profit=1.1020
    )

    assert res_loss.outcome == "LOSS"
    assert res_loss.r_multiple == -1.0
    assert res_loss.first_hit == "SL"

    # 3. TIMEOUT scenario
    df_timeout = pd.DataFrame({
        "Open": [1.1000] * 11,
        "High": [1.1005] * 11,
        "Low": [1.0995] * 11,
        "Close": [1.1000] + [1.1005] * 10
    })

    res_timeout = evaluator.evaluate_outcome(
        df=df_timeout,
        anchor_idx=0,
        direction=1,
        entry_price=1.1000,
        stop_loss=1.0990,
        take_profit=1.1020
    )

    assert res_timeout.outcome == "TIMEOUT"
    assert res_timeout.first_hit == "TIMEOUT"
