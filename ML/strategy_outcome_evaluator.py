import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger("StrategyOutcomeEvaluator")


@dataclass
class StrategyOutcomeResult:
    outcome: str  # 'WIN', 'LOSS', 'TIMEOUT', 'AMBIGUOUS'
    r_multiple: float
    mae_risk_ratio: float
    mfe_risk_ratio: float
    first_hit: str  # 'TP', 'SL', 'TIMEOUT', 'BOTH'
    bars_to_resolution: int
    exit_price: float
    exit_reason: str
    confidence: float


class StrategyOutcomeEvaluator:
    """
    Evaluates future price action for hypothetical trade setups (BUY / SELL)
    over a configurable horizon [t + 1 ... t + strategy_horizon].
    Calculates exact outcomes (WIN, LOSS, TIMEOUT, AMBIGUOUS) and continuous
    r_multiple, MAE, and MFE metrics for ML and RL retraining.
    """

    def __init__(self, future_horizon: int = 50, label_version: str = "2.0.0-causal"):
        self.future_horizon = future_horizon
        self.label_version = label_version

    def evaluate_outcome(
        self,
        df: pd.DataFrame,
        anchor_idx: int,
        direction: int,  # 1 for Long/BUY, -1 for Short/SELL
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        strategy_name: str = "Generic",
        signal_type: str = "Candidate"
    ) -> StrategyOutcomeResult:
        """
        Evaluates the outcome of a trade setup on future candles > anchor_idx.
        """
        n_bars = len(df)
        future_end_idx = min(n_bars - 1, anchor_idx + self.future_horizon)
        actual_horizon = future_end_idx - anchor_idx

        risk_dist = abs(entry_price - stop_loss)
        if risk_dist <= 1e-9:
            risk_dist = 0.0001

        reward_dist = abs(take_profit - entry_price)

        if actual_horizon < 1:
            return StrategyOutcomeResult(
                outcome="AMBIGUOUS",
                r_multiple=0.0,
                mae_risk_ratio=0.0,
                mfe_risk_ratio=0.0,
                first_hit="TIMEOUT",
                bars_to_resolution=0,
                exit_price=entry_price,
                exit_reason="insufficient_future_data",
                confidence=0.0
            )

        future_highs = df["High"].values[anchor_idx + 1: future_end_idx + 1]
        future_lows = df["Low"].values[anchor_idx + 1: future_end_idx + 1]
        future_closes = df["Close"].values[anchor_idx + 1: future_end_idx + 1]

        max_favorable = 0.0
        max_adverse = 0.0

        for step, (fh, fl, fc) in enumerate(zip(future_highs, future_lows, future_closes), start=1):
            if direction == 1:
                # Long (BUY)
                favorable = max(0.0, fh - entry_price)
                adverse = max(0.0, entry_price - fl)

                if favorable > max_favorable:
                    max_favorable = favorable
                if adverse > max_adverse:
                    max_adverse = adverse

                tp_hit = (fh >= take_profit)
                sl_hit = (fl <= stop_loss)

                if tp_hit and sl_hit:
                    # Same candle conflict
                    return StrategyOutcomeResult(
                        outcome="AMBIGUOUS",
                        r_multiple=0.0,
                        mae_risk_ratio=max_adverse / risk_dist,
                        mfe_risk_ratio=max_favorable / risk_dist,
                        first_hit="BOTH",
                        bars_to_resolution=step,
                        exit_price=stop_loss,
                        exit_reason="same_candle_tp_sl_conflict",
                        confidence=0.0
                    )
                elif tp_hit:
                    r_mult = reward_dist / risk_dist
                    return StrategyOutcomeResult(
                        outcome="WIN",
                        r_multiple=r_mult,
                        mae_risk_ratio=max_adverse / risk_dist,
                        mfe_risk_ratio=max_favorable / risk_dist,
                        first_hit="TP",
                        bars_to_resolution=step,
                        exit_price=take_profit,
                        exit_reason="take_profit_hit",
                        confidence=1.0
                    )
                elif sl_hit:
                    return StrategyOutcomeResult(
                        outcome="LOSS",
                        r_multiple=-1.0,
                        mae_risk_ratio=max_adverse / risk_dist,
                        mfe_risk_ratio=max_favorable / risk_dist,
                        first_hit="SL",
                        bars_to_resolution=step,
                        exit_price=stop_loss,
                        exit_reason="stop_loss_hit",
                        confidence=1.0
                    )

            else:
                # Short (SELL)
                favorable = max(0.0, entry_price - fl)
                adverse = max(0.0, fh - entry_price)

                if favorable > max_favorable:
                    max_favorable = favorable
                if adverse > max_adverse:
                    max_adverse = adverse

                tp_hit = (fl <= take_profit)
                sl_hit = (fh >= stop_loss)

                if tp_hit and sl_hit:
                    return StrategyOutcomeResult(
                        outcome="AMBIGUOUS",
                        r_multiple=0.0,
                        mae_risk_ratio=max_adverse / risk_dist,
                        mfe_risk_ratio=max_favorable / risk_dist,
                        first_hit="BOTH",
                        bars_to_resolution=step,
                        exit_price=stop_loss,
                        exit_reason="same_candle_tp_sl_conflict",
                        confidence=0.0
                    )
                elif tp_hit:
                    r_mult = reward_dist / risk_dist
                    return StrategyOutcomeResult(
                        outcome="WIN",
                        r_multiple=r_mult,
                        mae_risk_ratio=max_adverse / risk_dist,
                        mfe_risk_ratio=max_favorable / risk_dist,
                        first_hit="TP",
                        bars_to_resolution=step,
                        exit_price=take_profit,
                        exit_reason="take_profit_hit",
                        confidence=1.0
                    )
                elif sl_hit:
                    return StrategyOutcomeResult(
                        outcome="LOSS",
                        r_multiple=-1.0,
                        mae_risk_ratio=max_adverse / risk_dist,
                        mfe_risk_ratio=max_favorable / risk_dist,
                        first_hit="SL",
                        bars_to_resolution=step,
                        exit_price=stop_loss,
                        exit_reason="stop_loss_hit",
                        confidence=1.0
                    )

        # Timeout scenario (neither hit within horizon)
        final_close = future_closes[-1]
        final_pnl = (final_close - entry_price) * direction
        unrealized_r = final_pnl / risk_dist

        return StrategyOutcomeResult(
            outcome="TIMEOUT",
            r_multiple=unrealized_r,
            mae_risk_ratio=max_adverse / risk_dist,
            mfe_risk_ratio=max_favorable / risk_dist,
            first_hit="TIMEOUT",
            bars_to_resolution=actual_horizon,
            exit_price=final_close,
            exit_reason="horizon_exceeded",
            confidence=0.8
        )
