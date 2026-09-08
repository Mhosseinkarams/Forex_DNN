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
    Supports strict causal separation:
      1. evaluate_causal_current_state: classifies input window [t-window_size+1 ... t].
      2. label_window: classifies FUTURE market state target over [t+1 ... t+future_horizon].
    Generates high-quality labels ('TREND', 'RANGE', 'TRANSITION') and explicit 'NO_LABEL'/None
    for ambiguous or unclassifiable future states.
    """

    def __init__(
        self,
        ema_separation_trend: float = 1.5,
        ema_separation_range: float = 0.8,
        min_bos_trend: int = 1,
        min_rejections_range: int = 2,
        future_horizon: int = 20,
        atr_period: int = 14,
        label_version: str = "2.0.0-causal"
    ):
        """
        Args:
            ema_separation_trend: ATR-normalized separation between EMA50 and EMA600/800 above which is TREND candidate.
            ema_separation_range: ATR-normalized separation below which is RANGE candidate.
            min_bos_trend: Minimum number of BOS in the window to qualify for TREND.
            min_rejections_range: Minimum number of zone retest/touches in the window to qualify for RANGE.
            future_horizon: Number of future candles (> t) to evaluate for the prediction target.
            atr_period: The period used for ATR calculations.
            label_version: The version string for tracking experiments.
        """
        self.ema_sep_trend = ema_separation_trend
        self.ema_sep_range = ema_separation_range
        self.min_bos_trend = min_bos_trend
        self.min_rejections_range = min_rejections_range
        self.future_horizon = future_horizon
        self.atr_period = atr_period
        self._label_version = label_version

    @property
    def label_version(self) -> str:
        return self._label_version

    def evaluate_causal_current_state(
        self,
        df: pd.DataFrame,
        msg: MarketStructureGraph,
        window_start: int,
        window_end: int
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Evaluates the causal current market state of the input window <= t.
        This represents descriptive information available at anchor t.
        """
        atr_col = f"atr_{self.atr_period}" if f"atr_{self.atr_period}" in df.columns else "atr_14"
        atr = df.at[window_end, atr_col] if atr_col in df.columns else 0.0001
        if atr <= 0:
            atr = 0.0001

        fast_ema = df.at[window_end, "ema_50"] if "ema_50" in df.columns else None
        slow_col = "ema_600" if "ema_600" in df.columns else ("ema_800" if "ema_800" in df.columns else None)
        slow_ema = df.at[window_end, slow_col] if (slow_col and slow_col in df.columns) else None

        ema_separation_atr = 0.0
        if fast_ema is not None and slow_ema is not None:
            ema_separation_atr = abs(fast_ema - slow_ema) / atr

        ema_crossed = False
        if "ema_50" in df.columns and slow_col and slow_col in df.columns:
            fast_arr = df["ema_50"].values[window_start:window_end + 1]
            slow_arr = df[slow_col].values[window_start:window_end + 1]
            diffs = fast_arr - slow_arr
            if len(diffs) > 1 and np.any(diffs > 0) and np.any(diffs < 0):
                ema_crossed = True

        if not hasattr(msg, '_bos_indices_for_labeler'):
            msg._bos_indices_for_labeler = [b.index for b in msg.bos]
        bos_pos_left = bisect.bisect_left(msg._bos_indices_for_labeler, window_start)
        bos_pos_right = bisect.bisect_right(msg._bos_indices_for_labeler, window_end)
        bos_count = bos_pos_right - bos_pos_left

        if not hasattr(msg, '_choch_indices_for_labeler'):
            msg._choch_indices_for_labeler = [c.index for c in msg.choch]
        choch_pos_left = bisect.bisect_left(msg._choch_indices_for_labeler, window_start)
        choch_pos_right = bisect.bisect_right(msg._choch_indices_for_labeler, window_end)
        choch_count = choch_pos_right - choch_pos_left

        if "inside_supply_rollsum" in df.columns:
            inside_supply_count = int(df.at[window_end, "inside_supply_rollsum"])
            inside_demand_count = int(df.at[window_end, "inside_demand_rollsum"])
        else:
            window_df = df.iloc[window_start:window_end + 1]
            inside_supply_count = int(window_df.get("inside_supply", pd.Series(0, index=window_df.index)).sum())
            inside_demand_count = int(window_df.get("inside_demand", pd.Series(0, index=window_df.index)).sum())

        total_zone_touches = inside_supply_count + inside_demand_count

        info = {
            "ema_separation_atr": float(ema_separation_atr),
            "ema_crossed": bool(ema_crossed),
            "bos_count": int(bos_count),
            "choch_count": int(choch_count),
            "zone_retests": int(total_zone_touches),
            "rule_fired": "none"
        }

        if ema_separation_atr >= self.ema_sep_trend and bos_count >= self.min_bos_trend:
            confidence = min(1.0, 0.5 + (ema_separation_atr / 10.0) + (bos_count * 0.1))
            info["rule_fired"] = "trend_ema_sep_and_bos"
            return "TREND", float(confidence), info

        is_converged = ema_separation_atr < self.ema_sep_range
        is_retesting = total_zone_touches >= self.min_rejections_range
        if (is_converged or is_retesting) and bos_count == 0:
            confidence = min(1.0, 0.6 + (choch_count * 0.05) + (total_zone_touches * 0.05))
            info["rule_fired"] = "range_converged_or_retests"
            return "RANGE", float(confidence), info

        if ema_crossed or choch_count >= 1:
            confidence = min(1.0, 0.5 + (choch_count * 0.1) + (0.2 if ema_crossed else 0.0))
            info["rule_fired"] = "transition_cross_or_choch_or_shrink"
            return "TRANSITION", float(confidence), info

        info["rule_fired"] = "unlabeled_ambiguous"
        return None, 0.0, info

    def label_window(
        self,
        df: pd.DataFrame,
        msg: MarketStructureGraph,
        window_start: int,
        window_end: int
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """
        Determines the FUTURE market state target strictly using candles > window_end.
        Evaluates structural behavior in [window_end + 1 ... window_end + future_horizon].
        """
        n_bars = len(df)
        anchor_idx = window_end

        # If df ends at anchor or horizon=0, fallback to current window state for backward compatibility
        if anchor_idx + 1 >= n_bars or self.future_horizon == 0:
            return self.evaluate_causal_current_state(df, msg, window_start, window_end)

        future_end_idx = min(n_bars - 1, anchor_idx + self.future_horizon)
        actual_horizon = future_end_idx - anchor_idx

        if actual_horizon < 5:
            info = {"rule_fired": "insufficient_future_data", "actual_horizon": actual_horizon}
            return None, 0.0, info

        atr_col = f"atr_{self.atr_period}" if f"atr_{self.atr_period}" in df.columns else "atr_14"
        atr_anchor = df.at[anchor_idx, atr_col] if atr_col in df.columns else 0.0001
        if atr_anchor <= 0:
            atr_anchor = 0.0001

        anchor_close = df.at[anchor_idx, "Close"]
        future_closes = df["Close"].values[anchor_idx + 1: future_end_idx + 1]
        future_highs = df["High"].values[anchor_idx + 1: future_end_idx + 1]
        future_lows = df["Low"].values[anchor_idx + 1: future_end_idx + 1]

        final_future_close = future_closes[-1]
        net_displacement_atr = abs(final_future_close - anchor_close) / atr_anchor

        future_max_high = np.max(future_highs)
        future_min_low = np.min(future_lows)
        future_range_height_atr = (future_max_high - future_min_low) / atr_anchor

        # Binary search future BOS and CHOCH strictly inside (anchor_idx, future_end_idx]
        if not hasattr(msg, '_bos_indices_for_labeler'):
            msg._bos_indices_for_labeler = [b.index for b in msg.bos]
        bos_pos_left = bisect.bisect_left(msg._bos_indices_for_labeler, anchor_idx + 1)
        bos_pos_right = bisect.bisect_right(msg._bos_indices_for_labeler, future_end_idx)
        future_bos_count = bos_pos_right - bos_pos_left

        if not hasattr(msg, '_choch_indices_for_labeler'):
            msg._choch_indices_for_labeler = [c.index for c in msg.choch]
        choch_pos_left = bisect.bisect_left(msg._choch_indices_for_labeler, anchor_idx + 1)
        choch_pos_right = bisect.bisect_right(msg._choch_indices_for_labeler, future_end_idx)
        future_choch_count = choch_pos_right - choch_pos_left

        info = {
            "future_horizon": actual_horizon,
            "net_displacement_atr": float(net_displacement_atr),
            "future_range_height_atr": float(future_range_height_atr),
            "future_bos_count": int(future_bos_count),
            "future_choch_count": int(future_choch_count),
            "rule_fired": "none"
        }

        # Current causal state as metadata
        causal_lbl, causal_conf, causal_info = self.evaluate_causal_current_state(df, msg, window_start, window_end)
        info["causal_current_state"] = causal_lbl or "AMBIGUOUS"

        # --- Rule 1: FUTURE TREND ---
        if (net_displacement_atr >= 1.2 or future_bos_count >= 1) and future_choch_count == 0:
            confidence = min(1.0, 0.5 + (net_displacement_atr / 10.0) + (future_bos_count * 0.15))
            info["rule_fired"] = "future_trend_continuation"
            return "TREND", float(confidence), info

        # --- Rule 2: FUTURE RANGE ---
        if future_range_height_atr <= 2.5 and future_bos_count == 0 and net_displacement_atr <= 0.8:
            confidence = min(1.0, 0.6 + (2.5 - future_range_height_atr) * 0.15)
            info["rule_fired"] = "future_range_oscillation"
            return "RANGE", float(confidence), info

        # --- Rule 3: FUTURE TRANSITION ---
        if future_choch_count >= 1 or (future_bos_count >= 1 and net_displacement_atr < 0.5):
            confidence = min(1.0, 0.5 + (future_choch_count * 0.2))
            info["rule_fired"] = "future_transition_reversal"
            return "TRANSITION", float(confidence), info

        info["rule_fired"] = "future_unlabeled_ambiguous"
        return None, 0.0, info
