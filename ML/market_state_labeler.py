import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np

from Market_Data_Pipeline.structure_graph import MarketStructureGraph

logger = logging.getLogger("MarketStateLabeler")


class BaseLabeler(ABC):
    """
    Abstract Base Class for all rule-based labelers in the Forex_DNN framework.
    Provides a standardized interface for deterministic, rule-based labeling
    of machine learning datasets over sliding windows.
    """

    @property
    @abstractmethod
    def label_version(self) -> str:
        """
        Returns the version of this labeler's logic.
        """
        pass

    @abstractmethod
    def label_window(
        self,
        df: pd.DataFrame,
        msg: MarketStructureGraph,
        window_start: int,
        window_end: int
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Determines a deterministic label for the given window of candles.

        Args:
            df: The full enriched DataFrame (indicators included).
            msg: The associated MarketStructureGraph object model.
            window_start: The starting index of the window (inclusive).
            window_end: The ending index of the window (inclusive).

        Returns:
            label: The generated label (e.g., 'TREND', 'RANGE', 'TRANSITION'), or None if unlabeled.
            confidence: A confidence score between 0.0 and 1.0 indicating rule alignment.
            info: A dictionary of diagnostic metrics and rule-firing reasons.
        """
        pass


class MarketStateLabeler(BaseLabeler):
    """
    Deterministic rule-based labeler for Market State categorization.
    Generates high-quality labels ('TREND', 'RANGE', 'TRANSITION') based on
    predefined strategy rules using EMAs, BOS/CHOCH breaks, supply/demand zone
    interactions, and ATR-normalized volatility.
    """

    def __init__(
        self,
        ema_separation_trend: float = 1.5,
        ema_separation_range: float = 0.8,
        min_bos_trend: int = 1,
        min_rejections_range: int = 2,
        atr_period: int = 14,
        label_version: str = "1.0.0"
    ):
        """
        Args:
            ema_separation_trend: ATR-normalized separation between EMA50 and EMA600/800 above which is TREND candidate.
            ema_separation_range: ATR-normalized separation below which is RANGE candidate.
            min_bos_trend: Minimum number of BOS in the window to qualify for TREND.
            min_rejections_range: Minimum number of zone retest/touches in the window to qualify for RANGE.
            atr_period: The period used for ATR calculations.
            label_version: The version string for tracking experiments.
        """
        self.ema_sep_trend = ema_separation_trend
        self.ema_sep_range = ema_separation_range
        self.min_bos_trend = min_bos_trend
        self.min_rejections_range = min_rejections_range
        self.atr_period = atr_period
        self._label_version = label_version

    @property
    def label_version(self) -> str:
        return self._label_version

    def label_window(
        self,
        df: pd.DataFrame,
        msg: MarketStructureGraph,
        window_start: int,
        window_end: int
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Applies deterministic rules to classify the sliding window into TREND, RANGE, or TRANSITION.
        Samples not meeting strict thresholds are returned as None (for removal).
        """
        # Ensure we do not look beyond the window bounds for strict point-in-time rules
        window_df = df.iloc[window_start:window_end + 1]
        last_row = window_df.iloc[-1]

        atr = last_row.get(f"atr_{self.atr_period}", last_row.get("atr_14", 0.0001))
        if atr <= 0:
            atr = 0.0001

        # 1. EMA Separation calculation at window_end
        fast_ema = last_row.get("ema_50")
        slow_ema = last_row.get("ema_600", last_row.get("ema_800"))

        ema_separation_atr = 0.0
        if fast_ema is not None and slow_ema is not None:
            ema_separation_atr = abs(fast_ema - slow_ema) / atr

        # Check if EMA fast/slow crossed in this window
        ema_crossed = False
        if "ema_50" in window_df.columns:
            fast_emas = window_df["ema_50"].values
            slow_emas = window_df.get("ema_600", window_df.get("ema_800", window_df["ema_50"])).values
            diffs = fast_emas - slow_emas
            # Check if there is any sign change in difference across the window
            if len(diffs) > 1:
                has_pos = np.any(diffs > 0)
                has_neg = np.any(diffs < 0)
                if has_pos and has_neg:
                    ema_crossed = True

        # 2. Count BOS in window
        # Verify indices of BOS fall strictly within [window_start, window_end]
        bos_events = [b for b in msg.bos if window_start <= b.index <= window_end]
        bos_count = len(bos_events)

        # 3. Count CHOCH in window
        choch_events = [c for c in msg.choch if window_start <= c.index <= window_end]
        choch_count = len(choch_events)

        # 4. Supply & Demand retest/touches inside the window
        # We can extract interactions from df columns like `inside_supply`, `inside_demand`, etc.
        # Or from zone touch counts. Let's inspect the window_df for inside zone touches.
        inside_supply_count = int(window_df.get("inside_supply", pd.Series(0, index=window_df.index)).sum())
        inside_demand_count = int(window_df.get("inside_demand", pd.Series(0, index=window_df.index)).sum())
        total_zone_touches = inside_supply_count + inside_demand_count

        # Also let's look at the actual zone objects confirmed up to window_end
        zone_retests = 0
        all_zones = msg.supply_zones + msg.demand_zones
        for z in all_zones:
            # Did a touch event occur strictly inside this window?
            if z.mitigated and z.mitigated_idx is not None and window_start <= z.mitigated_idx <= window_end:
                zone_retests += 1
            if z.touch_count > 0:
                # Count touches that happened in the window
                # If z.created_idx is inside window, count its touches
                if window_start <= z.created_idx <= window_end:
                    zone_retests += z.touch_count

        # 5. Volatility & ATR Ratio
        # Is the volatility expanding or contracting relative to historical average?
        atr_ratio = last_row.get("atr_ratio", 1.0)

        info = {
            "ema_separation_atr": float(ema_separation_atr),
            "ema_crossed": bool(ema_crossed),
            "bos_count": int(bos_count),
            "choch_count": int(choch_count),
            "zone_retests": int(zone_retests),
            "total_zone_touches": int(total_zone_touches),
            "atr_ratio": float(atr_ratio),
            "rule_fired": "none"
        }

        # --- Rule 1: TREND ---
        # Strong EMA alignment and separation + at least 1 BOS break in trend direction,
        # and no opposing structural trend changes (no CHOCH).
        if ema_separation_atr >= self.ema_sep_trend and bos_count >= self.min_bos_trend:
            # Check trend persistence - did we have an opposing CHOCH?
            has_opposing_choch = False
            if choch_count > 0:
                # If CHOCH direction opposes the current trend bias
                last_trend = last_row.get("trend", 0)
                for c in choch_events:
                    if c.new_trend != last_trend:
                        has_opposing_choch = True
                        break

            if not has_opposing_choch:
                confidence = min(1.0, 0.5 + (ema_separation_atr / 10.0) + (bos_count * 0.1))
                info["rule_fired"] = "trend_ema_sep_and_bos"
                return "TREND", float(confidence), info

        # --- Rule 2: RANGE ---
        # Converged EMAs (low separation) OR prolonged zone interactions without structure breaks (BOS)
        is_converged = ema_separation_atr < self.ema_sep_range
        is_retesting_zones = (zone_retests >= self.min_rejections_range or total_zone_touches >= self.min_rejections_range)

        if (is_converged or is_retesting_zones) and bos_count == 0:
            # Extra confidence if both conditions hold
            base_conf = 0.6
            if is_converged and is_retesting_zones:
                base_conf += 0.15
            confidence = min(1.0, base_conf + (choch_count * 0.05) + (zone_retests * 0.05))
            info["rule_fired"] = "range_converged_or_retests"
            return "RANGE", float(confidence), info

        # --- Rule 3: TRANSITION ---
        # Clear signs of regime change: EMAs crossed within window, or CHOCH occurring,
        # or EMAs are rapidly shrinking in separation from a trending state.
        is_ema_shrinking = False
        if len(window_df) >= 10:
            # Check if EMA separation 10 bars ago was trending (> 1.5) and now has shrunk significantly (< 1.0)
            ema_col = "ema_50"
            if ema_col in window_df.columns:
                prev_row = window_df.iloc[-10]
                prev_fast = prev_row.get("ema_50")
                prev_slow = prev_row.get("ema_600", prev_row.get("ema_800"))
                prev_atr = prev_row.get(f"atr_{self.atr_period}", prev_row.get("atr_14", 0.0001))
                if prev_fast is not None and prev_slow is not None and prev_atr > 0:
                    prev_sep = abs(prev_fast - prev_slow) / prev_atr
                    if prev_sep >= 1.2 and ema_separation_atr < 0.9:
                        is_ema_shrinking = True

        if ema_crossed or choch_count >= 1 or is_ema_shrinking:
            # We must be careful not to label standard trend-continuation pullbacks as transition.
            # If we have a CHOCH or EMA cross, it's a transition.
            confidence = min(1.0, 0.5 + (choch_count * 0.1) + (0.2 if ema_crossed else 0.0))
            info["rule_fired"] = "transition_cross_or_choch_or_shrink"
            return "TRANSITION", float(confidence), info

        # --- Unlabeled samples ---
        # If a sample is ambiguous or does not meet strict rules, return None to remove it.
        info["rule_fired"] = "unlabeled_ambiguous"
        return None, 0.0, info
