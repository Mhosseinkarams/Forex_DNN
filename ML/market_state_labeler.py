import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from ML.label_engine import BaseLabeler
from Market_Data_Pipeline.structure_graph import MarketStructureGraph

logger = logging.getLogger("MarketStateLabeler")


class MarketStateLabeler(BaseLabeler):
    """
    Deterministic, rule-based market state labeler.
    Generates rule-based targets (TREND, RANGE, TRANSITION) from technical and structural logic
    using EMA relationships, S/D zone rejections, BOS and CHOCH frequencies.
    """
    def __init__(
        self,
        label_version: str = "1.0",
        engine_version: str = "1.0",
        min_confidence: float = 0.4,
        trend_ema_separation: float = 1.5,
        range_ema_separation: float = 0.8,
    ):
        self.label_version = label_version
        self.engine_version = engine_version
        self.min_confidence = min_confidence
        self.trend_ema_separation = trend_ema_separation
        self.range_ema_separation = range_ema_separation

    def label_window(
        self,
        df: pd.DataFrame,
        msg: MarketStructureGraph,
        window_start: int,
        window_end: int
    ) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Evaluate market state rule-based labeling at the window_end index.

        Rules:
        - TREND: Strong EMA separation and presence of BOS.
        - RANGE: Absence of BOS combined with either high CHOCH frequency, multiple zone rejections,
                 or low/flat EMA separation.
        - TRANSITION: Structural transition defined by a combination of CHOCHs, moderate separation,
                      or initial breakouts that do not meet TREND confidence.
        """
        row = df.iloc[window_end]

        # 1. Warmup check: slow EMAs (600 or 800) require substantial bars to be stable
        # If we are before the stable index of the slow EMA, return None
        slow_period = 600 if "ema_600" in df.columns else (800 if "ema_800" in df.columns else 100)
        if window_end < slow_period:
            return None, 0.0, "warmup_period"

        atr = row.get("atr_14", 0.0001)
        if np.isnan(atr) or atr <= 0:
            return None, 0.0, "missing_atr"

        fast_ema = row.get("ema_50")
        slow_ema = row.get("ema_600", row.get("ema_800"))
        if fast_ema is None or slow_ema is None or np.isnan(fast_ema) or np.isnan(slow_ema):
            return None, 0.0, "missing_emas"

        # 2. Extract technical metrics
        ema_separation_atr = abs(fast_ema - slow_ema) / (atr + 1e-9)

        # 3. Extract structural metrics from graph within current sliding window
        bos_count = sum(1 for b in msg.bos if window_start <= b.index <= window_end)
        choch_count = sum(1 for c in msg.choch if window_start <= c.index <= window_end)

        # Count active zone mitigations/retests occurring within the window
        rejections = 0
        all_zones = msg.supply_zones + msg.demand_zones
        for z in all_zones:
            if z.mitigated and z.mitigated_idx is not None and window_start <= z.mitigated_idx <= window_end:
                rejections += 1

        # Check trend direction of the candle compared to the EMA trend to verify persistence
        is_trend_persistent = False
        ema_trend = 1 if fast_ema > slow_ema else -1
        candle_trend = row.get("candle_direction", 0)
        if ema_trend == candle_trend:
            is_trend_persistent = True

        # 4. Evaluation of rules

        # Rule A: TREND
        # Requirements: Strong EMA separation, at least 1 BOS inside the window, and matching trend direction
        if ema_separation_atr >= self.trend_ema_separation and bos_count >= 1:
            persistence_bonus = 0.05 if is_trend_persistent else 0.0
            confidence = min(1.0, 0.5 + (ema_separation_atr / 10.0) + (bos_count * 0.08) + persistence_bonus)

            if confidence >= self.min_confidence:
                return "TREND", float(confidence), "strong_ema_separation_and_bos"
            else:
                return None, float(confidence), "trend_low_confidence"

        # Rule B: RANGE
        # Requirements: Low EMA separation, OR no BOS combined with multiple rejections or CHOCHs.
        if (bos_count == 0 and choch_count >= 1) or rejections >= 2 or ema_separation_atr < self.range_ema_separation:
            confidence = min(1.0, 0.55 + (choch_count * 0.05) + (rejections * 0.08) - (ema_separation_atr * 0.05))

            if confidence >= self.min_confidence:
                return "RANGE", float(confidence), "low_ema_separation_or_no_bos_rejections"
            else:
                return None, float(confidence), "range_low_confidence"

        # Rule C: TRANSITION
        # Requirements: Moderate EMA separation, or active breakouts/chochs that suggest structure change,
        # but not a fully formed trend or a clear range.
        if (0.8 <= ema_separation_atr < 1.5) or (choch_count >= 1) or (bos_count >= 1):
            confidence = min(0.85, 0.45 + (choch_count * 0.1) + (bos_count * 0.05))

            if confidence >= self.min_confidence:
                return "TRANSITION", float(confidence), "structure_transition_state"
            else:
                return None, float(confidence), "transition_low_confidence"

        # 5. Default fallback to unlabeled to prevent garbage labels
        return None, 0.0, "ambiguous_or_unclassifiable"
