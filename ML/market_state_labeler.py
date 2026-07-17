import logging
import bisect
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
        # Ensure fast lookup of the row at window_end
        atr_col = f"atr_{self.atr_period}" if f"atr_{self.atr_period}" in df.columns else "atr_14"
        atr = df.at[window_end, atr_col] if atr_col in df.columns else 0.0001
        if atr <= 0:
            atr = 0.0001

        # 1. EMA Separation calculation at window_end
        fast_ema = df.at[window_end, "ema_50"] if "ema_50" in df.columns else None
        slow_col = "ema_600" if "ema_600" in df.columns else ("ema_800" if "ema_800" in df.columns else None)
        slow_ema = df.at[window_end, slow_col] if (slow_col and slow_col in df.columns) else None

        ema_separation_atr = 0.0
        if fast_ema is not None and slow_ema is not None:
            ema_separation_atr = abs(fast_ema - slow_ema) / atr

        # Check if EMA fast/slow crossed in this window using fast numpy array slicing
        ema_crossed = False
        if "ema_50" in df.columns and slow_col and slow_col in df.columns:
            fast_arr = df["ema_50"].values[window_start:window_end + 1]
            slow_arr = df[slow_col].values[window_start:window_end + 1]
            diffs = fast_arr - slow_arr
            if len(diffs) > 1:
                has_pos = np.any(diffs > 0)
                has_neg = np.any(diffs < 0)
                if has_pos and has_neg:
                    ema_crossed = True

        # 2. Count BOS in window (O(log K) binary search instead of O(K) linear lookup)
        if not hasattr(msg, '_bos_indices_for_labeler'):
            msg._bos_indices_for_labeler = [b.index for b in msg.bos]
        bos_pos_left = bisect.bisect_left(msg._bos_indices_for_labeler, window_start)
        bos_pos_right = bisect.bisect_right(msg._bos_indices_for_labeler, window_end)
        bos_count = bos_pos_right - bos_pos_left

        # 3. Count CHOCH in window (O(log K) binary search)
        if not hasattr(msg, '_choch_indices_for_labeler'):
            msg._choch_indices_for_labeler = [c.index for c in msg.choch]
        choch_pos_left = bisect.bisect_left(msg._choch_indices_for_labeler, window_start)
        choch_pos_right = bisect.bisect_right(msg._choch_indices_for_labeler, window_end)
        choch_count = choch_pos_right - choch_pos_left

        # 4. Supply & Demand retest/touches inside the window
        # Access precomputed rolling sums directly
        if "inside_supply_rollsum" in df.columns:
            inside_supply_count = int(df.at[window_end, "inside_supply_rollsum"])
            inside_demand_count = int(df.at[window_end, "inside_demand_rollsum"])
        else:
            # Fallback
            window_df = df.iloc[window_start:window_end + 1]
            inside_supply_count = int(window_df.get("inside_supply", pd.Series(0, index=window_df.index)).sum())
            inside_demand_count = int(window_df.get("inside_demand", pd.Series(0, index=window_df.index)).sum())

        total_zone_touches = inside_supply_count + inside_demand_count
        zone_retests = total_zone_touches

        # 5. Volatility & ATR Ratio
        atr_ratio = df.at[window_end, "atr_ratio"] if "atr_ratio" in df.columns else 1.0

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
        if ema_separation_atr >= self.ema_sep_trend and bos_count >= self.min_bos_trend:
            has_opposing_choch = False
            if choch_count > 0:
                last_trend = df.at[window_end, "trend"] if "trend" in df.columns else 0
                # Slice the choch list using binary search positions for O(1) retrieval
                choch_events = msg.choch[choch_pos_left:choch_pos_right]
                for c in choch_events:
                    if c.new_trend != last_trend:
                        has_opposing_choch = True
                        break

            if not has_opposing_choch:
                confidence = min(1.0, 0.5 + (ema_separation_atr / 10.0) + (bos_count * 0.1))
                info["rule_fired"] = "trend_ema_sep_and_bos"
                return "TREND", float(confidence), info

        # --- Rule 2: RANGE ---
        is_converged = ema_separation_atr < self.ema_sep_range
        is_retesting_zones = (zone_retests >= self.min_rejections_range or total_zone_touches >= self.min_rejections_range)

        if (is_converged or is_retesting_zones) and bos_count == 0:
            base_conf = 0.6
            if is_converged and is_retesting_zones:
                base_conf += 0.15
            confidence = min(1.0, base_conf + (choch_count * 0.05) + (zone_retests * 0.05))
            info["rule_fired"] = "range_converged_or_retests"
            return "RANGE", float(confidence), info

        # --- Rule 3: TRANSITION ---
        is_ema_shrinking = False
        if (window_end - window_start + 1) >= 10:
            if "ema_50" in df.columns:
                prev_idx = window_end - 9
                prev_fast = df.at[prev_idx, "ema_50"]
                prev_slow = df.at[prev_idx, slow_col] if slow_col else None
                prev_atr = df.at[prev_idx, atr_col] if atr_col in df.columns else 0.0001
                if prev_fast is not None and prev_slow is not None and prev_atr > 0:
                    prev_sep = abs(prev_fast - prev_slow) / prev_atr
                    if prev_sep >= 1.2 and ema_separation_atr < 0.9:
                        is_ema_shrinking = True

        if ema_crossed or choch_count >= 1 or is_ema_shrinking:
            confidence = min(1.0, 0.5 + (choch_count * 0.1) + (0.2 if ema_crossed else 0.0))
            info["rule_fired"] = "transition_cross_or_choch_or_shrink"
            return "TRANSITION", float(confidence), info

        # --- Unlabeled samples ---
        info["rule_fired"] = "unlabeled_ambiguous"
        return None, 0.0, info
