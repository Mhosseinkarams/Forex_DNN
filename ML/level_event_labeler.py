import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np

from Market_Data_Pipeline.structure_graph import MarketStructureGraph, Zone

logger = logging.getLogger("LevelEventLabeler")


@dataclass
class LevelEventResult:
    event_type: str  # 'NO_INTERACTION', 'REJECTION', 'SWEEP_REJECTION', 'BREAK', 'AMBIGUOUS'
    break_probability_target: Optional[int]  # 1 for BREAK, 0 for REJECTION / SWEEP_REJECTION, None for NO_INTERACTION / AMBIGUOUS
    confidence: float
    bars_to_resolution: int
    mae: float
    mfe: float
    penetration_depth: float
    close_distance: float
    touch_detected: bool
    ambiguity_reason: str


class LevelEventLabeler:
    """
    ATR-normalized Level Event Labeler for Smart Money supply/demand and swing levels.
    Evaluates future price action strictly over horizon [t + 1 ... t + level_event_horizon].

    Distinguishes:
      - NO_INTERACTION: Price never approached within proximity_atr * ATR.
      - REJECTION: Touches/enters zone and bounces away without closing beyond.
      - SWEEP_REJECTION: Wick exceeds zone boundary, but close remains inside/below, followed by bounce.
      - BREAK: Decisive close beyond zone boundary + break_buffer_atr * ATR.
      - AMBIGUOUS: Resolution unresolved or conflicting within future horizon.
    """

    def __init__(
        self,
        proximity_atr: float = 0.5,
        break_buffer_atr: float = 0.2,
        rejection_distance_atr: float = 1.0,
        future_horizon: int = 20,
        label_version: str = "2.0.0-causal"
    ):
        self.proximity_atr = proximity_atr
        self.break_buffer_atr = break_buffer_atr
        self.rejection_distance_atr = rejection_distance_atr
        self.future_horizon = future_horizon
        self.label_version = label_version

    def evaluate_level_event(
        self,
        df: pd.DataFrame,
        msg: MarketStructureGraph,
        anchor_idx: int,
        zone: Zone
    ) -> LevelEventResult:
        """
        Evaluates the future interaction between price and a level/zone strictly after anchor_idx.
        """
        n_bars = len(df)
        future_end_idx = min(n_bars - 1, anchor_idx + self.future_horizon)
        actual_horizon = future_end_idx - anchor_idx

        if actual_horizon < 1:
            return LevelEventResult(
                event_type="AMBIGUOUS",
                break_probability_target=None,
                confidence=0.0,
                bars_to_resolution=0,
                mae=0.0,
                mfe=0.0,
                penetration_depth=0.0,
                close_distance=0.0,
                touch_detected=False,
                ambiguity_reason="insufficient_future_bars"
            )

        atr_col = "atr_14" if "atr_14" in df.columns else "atr"
        atr = df.at[anchor_idx, atr_col] if atr_col in df.columns else 0.0001
        if atr <= 0:
            atr = 0.0001

        anchor_close = df.at[anchor_idx, "Close"]
        is_supply = (zone.type == "Supply")

        # Proximity check at anchor
        if is_supply:
            dist_at_anchor = zone.lower - anchor_close
        else:
            dist_at_anchor = anchor_close - zone.upper

        proximity_threshold = self.proximity_atr * atr

        # Future slice
        future_highs = df["High"].values[anchor_idx + 1: future_end_idx + 1]
        future_lows = df["Low"].values[anchor_idx + 1: future_end_idx + 1]
        future_closes = df["Close"].values[anchor_idx + 1: future_end_idx + 1]

        touch_detected = False
        max_penetration = 0.0
        mae = 0.0
        mfe = 0.0

        for step, (fh, fl, fc) in enumerate(zip(future_highs, future_lows, future_closes), start=1):
            if is_supply:
                # Supply Zone (Resistance)
                # Touch: High enters zone (High >= zone.lower) or approaches within proximity
                if fh >= zone.lower - proximity_threshold:
                    touch_detected = True

                penetration = max(0.0, fh - zone.lower)
                if penetration > max_penetration:
                    max_penetration = penetration

                # MAE is adverse movement upwards through resistance
                current_mae = max(0.0, fh - anchor_close)
                if current_mae > mae:
                    mae = current_mae

                # MFE is favorable move downwards away from supply
                current_mfe = max(0.0, anchor_close - fl)
                if current_mfe > mfe:
                    mfe = current_mfe

                # Check Break condition: Decisive CLOSE above zone.upper + break_buffer
                if fc > zone.upper + self.break_buffer_atr * atr:
                    close_dist = (fc - zone.upper) / atr
                    return LevelEventResult(
                        event_type="BREAK",
                        break_probability_target=1,
                        confidence=0.9,
                        bars_to_resolution=step,
                        mae=mae / atr,
                        mfe=mfe / atr,
                        penetration_depth=max_penetration / atr,
                        close_distance=close_dist,
                        touch_detected=True,
                        ambiguity_reason="none"
                    )

                # Check Rejection / Sweep Rejection condition: Requires touch_detected first and price bouncing away
                if touch_detected and fl < zone.lower - self.rejection_distance_atr * atr:
                    is_sweep = (max_penetration > zone.width)  # Wick surpassed upper, but close stayed below
                    event_type = "SWEEP_REJECTION" if is_sweep else "REJECTION"
                    return LevelEventResult(
                        event_type=event_type,
                        break_probability_target=0,
                        confidence=0.85 if is_sweep else 0.9,
                        bars_to_resolution=step,
                        mae=mae / atr,
                        mfe=mfe / atr,
                        penetration_depth=max_penetration / atr,
                        close_distance=(zone.lower - fc) / atr,
                        touch_detected=True,
                        ambiguity_reason="none"
                    )

            else:
                # Demand Zone (Support)
                if fl <= zone.upper + proximity_threshold:
                    touch_detected = True

                penetration = max(0.0, zone.upper - fl)
                if penetration > max_penetration:
                    max_penetration = penetration

                # MAE is adverse movement downwards through support
                current_mae = max(0.0, anchor_close - fl)
                if current_mae > mae:
                    mae = current_mae

                # MFE is favorable move upwards away from demand
                current_mfe = max(0.0, fh - anchor_close)
                if current_mfe > mfe:
                    mfe = current_mfe

                # Check Break condition: Decisive CLOSE below zone.lower - break_buffer
                if fc < zone.lower - self.break_buffer_atr * atr:
                    close_dist = (zone.lower - fc) / atr
                    return LevelEventResult(
                        event_type="BREAK",
                        break_probability_target=1,
                        confidence=0.9,
                        bars_to_resolution=step,
                        mae=mae / atr,
                        mfe=mfe / atr,
                        penetration_depth=max_penetration / atr,
                        close_distance=close_dist,
                        touch_detected=True,
                        ambiguity_reason="none"
                    )

                # Check Rejection / Sweep Rejection condition: Requires touch_detected first and price bouncing away
                if touch_detected and fh > zone.upper + self.rejection_distance_atr * atr:
                    is_sweep = (max_penetration > zone.width)
                    event_type = "SWEEP_REJECTION" if is_sweep else "REJECTION"
                    return LevelEventResult(
                        event_type=event_type,
                        break_probability_target=0,
                        confidence=0.85 if is_sweep else 0.9,
                        bars_to_resolution=step,
                        mae=mae / atr,
                        mfe=mfe / atr,
                        penetration_depth=max_penetration / atr,
                        close_distance=(fc - zone.upper) / atr,
                        touch_detected=True,
                        ambiguity_reason="none"
                    )

        if not touch_detected:
            return LevelEventResult(
                event_type="NO_INTERACTION",
                break_probability_target=None,
                confidence=1.0,
                bars_to_resolution=actual_horizon,
                mae=mae / atr,
                mfe=mfe / atr,
                penetration_depth=0.0,
                close_distance=dist_at_anchor / atr,
                touch_detected=False,
                ambiguity_reason="level_never_approached"
            )

        # Unresolved within future horizon
        return LevelEventResult(
            event_type="AMBIGUOUS",
            break_probability_target=None,
            confidence=0.5,
            bars_to_resolution=actual_horizon,
            mae=mae / atr,
            mfe=mfe / atr,
            penetration_depth=max_penetration / atr,
            close_distance=0.0,
            touch_detected=True,
            ambiguity_reason="unresolved_within_horizon"
        )
