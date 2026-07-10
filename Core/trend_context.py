import logging
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger("TrendContext")

@dataclass
class TrendContext:
    symbol: str
    timeframe: str
    timestamp: datetime
    trend_direction: str  # 'Bull' or 'Bear'
    ema_fast: float
    ema_slow: float
    ema_slope: float
    ema_distance: float
    ema_distance_atr: float
    trend_strength: str   # 'Very Strong', 'Strong', 'Normal', 'Weak'
    is_strong_trend: bool
    is_weak_trend: bool
    bars_since_cross: int
    bars_since_trend_change: int


class TrendContextBuilder:
    """
    Purpose:
        Builds a TrendContext object from an indicator DataFrame.
        Completely decouples market trend analysis from strategy/trading logic.
    """
    def __init__(
        self,
        slope_threshold: float = 0.1,
        strong_sep_atr: float = 1.5,
        very_strong_sep_atr: float = 4.0,
        weak_sep_atr: float = 0.5,
    ):
        self.slope_threshold = slope_threshold
        self.strong_sep_atr = strong_sep_atr
        self.very_strong_sep_atr = very_strong_sep_atr
        self.weak_sep_atr = weak_sep_atr

    def build(self, symbol: str, timeframe: str, df: pd.DataFrame, idx: int = -1) -> TrendContext:
        """
        Purpose:
            Extracts data from the given DataFrame at `idx` and computes trend context.
        """
        if df is None or len(df) == 0:
            raise ValueError("DataFrame cannot be empty or None")

        # Normalize negative index to positive index
        if idx < 0:
            idx = len(df) + idx

        if idx < 0 or idx >= len(df):
            raise IndexError(f"Index {idx} out of range for DataFrame of length {len(df)}")

        row = df.iloc[idx]

        # Datetime handling
        timestamp_raw = row.get("Datetime", row.get("Datetime", None))
        if isinstance(timestamp_raw, str):
            try:
                timestamp = datetime.fromisoformat(timestamp_raw)
            except Exception:
                timestamp = timestamp_raw
        else:
            timestamp = timestamp_raw

        # Detect EMAs based on timeframe
        # M5: EMA50 & EMA600, M15: EMA50 & EMA800
        fast_p = 50
        slow_p = 600 if timeframe == "M5" else 800

        ema_fast_col = f"ema_{fast_p}"
        ema_slow_col = f"ema_{slow_p}"
        slope_col = f"ema_slope_{slow_p}"
        atr_col = "atr_14"

        # Safe lookups
        ema_fast = float(row.get(ema_fast_col, 0.0))
        ema_slow = float(row.get(ema_slow_col, 0.0))
        ema_slope = float(row.get(slope_col, 0.0))
        atr_val = float(row.get(atr_col, 0.0))

        # 1. Trend Direction
        if ema_fast > ema_slow:
            trend_direction = "Bull"
        elif ema_fast < ema_slow:
            trend_direction = "Bear"
        else:
            trend_direction = "Bull"  # Default fallback if exactly equal

        # 2. EMA Separation & ATR Normalized Separation
        ema_distance = abs(ema_fast - ema_slow)
        ema_distance_atr = ema_distance / (atr_val + 1e-9)

        # 3. Trend Strength Classification
        # We classify strength using slope and distance ATR.
        if ema_slope >= self.slope_threshold:
            if ema_distance_atr >= self.very_strong_sep_atr:
                trend_strength = "Very Strong"
            elif ema_distance_atr >= self.strong_sep_atr:
                trend_strength = "Strong"
            else:
                trend_strength = "Normal"
        else:
            if ema_distance_atr < self.weak_sep_atr:
                trend_strength = "Weak"
            else:
                trend_strength = "Normal"

        is_strong_trend = trend_strength in ("Very Strong", "Strong")
        is_weak_trend = trend_strength == "Weak"

        # 4. Bars Since EMA Cross (Price crossing EMA50)
        bars_since_cross = -1
        cross_col = f"cross_ema_{fast_p}"
        if cross_col in df.columns:
            # Search backward from idx to find where cross_ema_50 != 0
            for i in range(idx, -1, -1):
                if df.iloc[i][cross_col] != 0:
                    bars_since_cross = idx - i
                    break

        # 5. Bars Since Trend Change (EMA50 crossing EMA600/800)
        bars_since_trend_change = -1
        if ema_fast_col in df.columns and ema_slow_col in df.columns:
            # We determine the trend state at j: True if fast > slow else False
            # Find the most recent transition
            current_state = df.iloc[idx][ema_fast_col] > df.iloc[idx][ema_slow_col]
            for i in range(idx - 1, -1, -1):
                prev_state = df.iloc[i][ema_fast_col] > df.iloc[i][ema_slow_col]
                if prev_state != current_state:
                    bars_since_trend_change = idx - (i + 1)
                    break
            # If no transition is found, then the trend has been the same for the entire dataframe history
            if bars_since_trend_change == -1:
                bars_since_trend_change = idx

        return TrendContext(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp,
            trend_direction=trend_direction,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_slope=ema_slope,
            ema_distance=ema_distance,
            ema_distance_atr=ema_distance_atr,
            trend_strength=trend_strength,
            is_strong_trend=is_strong_trend,
            is_weak_trend=is_weak_trend,
            bars_since_cross=bars_since_cross,
            bars_since_trend_change=bars_since_trend_change,
        )
