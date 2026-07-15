import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("StrongCandleEngine")

@dataclass
class StrongCandle:
    """
    Strongly-typed output of the StrongCandleEngine.
    Follows the common scoring convention:
    - confidence: float [0.0, 1.0]
    - quality_score: int [0, 100]
    """
    bullish: bool
    bearish: bool
    score: float             # Legacy/general score (0-100)
    quality_score: int       # Common scoring convention (0-100)
    confidence: float        # Common scoring convention (0.0-1.0)
    classification: str      # VERY_STRONG, STRONG, MEDIUM, WEAK, INDECISION, DOJI, EXPANSION, CLIMAX, EXHAUSTION, etc.
    metrics: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


class StrongCandleEngine:
    """
    Purpose:
        Analyze every candle's geometrical and contextual properties to classify its strength.
        This engine is strategy-agnostic, has zero trading signal logic, and is fully configurable.
    """
    def __init__(
        self,
        lookback_period: int = 20,
        doji_threshold: float = 0.10,
        exhaustion_wick_ratio: float = 0.60,
        very_strong_body_ratio: float = 0.75,
        strong_body_ratio: float = 0.60,
        medium_body_ratio: float = 0.40,
        weak_body_ratio: float = 0.20,
        climax_range_mult: float = 2.0,
        expansion_range_mult: float = 1.5,
        expansion_body_ratio: float = 0.65,
        very_strong_range_mult: float = 1.0,
        volume_spike_mult: float = 1.5
    ):
        # Configuration thresholds
        self.lookback_period = lookback_period
        self.doji_threshold = doji_threshold
        self.exhaustion_wick_ratio = exhaustion_wick_ratio
        self.very_strong_body_ratio = very_strong_body_ratio
        self.strong_body_ratio = strong_body_ratio
        self.medium_body_ratio = medium_body_ratio
        self.weak_body_ratio = weak_body_ratio
        self.climax_range_mult = climax_range_mult
        self.expansion_range_mult = expansion_range_mult
        self.expansion_body_ratio = expansion_body_ratio
        self.very_strong_range_mult = very_strong_range_mult
        self.volume_spike_mult = volume_spike_mult

    def evaluate(
        self,
        df: pd.DataFrame,
        idx: int,
        msg: Optional[Any] = None
    ) -> StrongCandle:
        """
        Evaluate candle properties and classify its strength at a specific index.
        """
        if idx < 0:
            idx = len(df) + idx

        if idx < 0 or idx >= len(df):
            return StrongCandle(
                bullish=False, bearish=False, score=0.0, quality_score=0, confidence=0.0,
                classification="INVALID", metrics={}, reasons=["Index out of bounds"]
            )

        row = df.iloc[idx]

        # 1. Base Geometry
        open_p = float(row["Open"])
        high_p = float(row["High"])
        low_p = float(row["Low"])
        close_p = float(row["Close"])
        volume = float(row.get("TickVolume", 0.0))

        candle_range = high_p - low_p
        if candle_range <= 0:
            return StrongCandle(
                bullish=False, bearish=False, score=0.0, quality_score=0, confidence=0.0,
                classification="INVALID", metrics={}, reasons=["Zero candle range"]
            )

        body_size = abs(close_p - open_p)
        body_ratio = body_size / candle_range
        upper_wick = high_p - max(open_p, close_p)
        lower_wick = min(open_p, close_p) - low_p
        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range

        bullish = close_p > open_p
        bearish = close_p < open_p

        # Close position inside candle (0.0 at low, 1.0 at high)
        close_pos = (close_p - low_p) / candle_range

        # 2. Contextual lookbacks (using idx to avoid lookahead bias)
        start_idx = max(0, idx - self.lookback_period)
        lookback_slice = df.iloc[start_idx:idx]

        avg_range = 1e-9
        avg_volume = 1e-9
        if not lookback_slice.empty:
            avg_range = float((lookback_slice["High"] - lookback_slice["Low"]).mean())
            avg_volume = float(lookback_slice.get("TickVolume", 0.0).mean())
            if avg_range <= 0:
                avg_range = 1e-9
            if avg_volume <= 0:
                avg_volume = 1e-9

        range_vs_avg = candle_range / avg_range
        volume_vs_avg = volume / avg_volume if avg_volume > 0 else 1.0

        # ATR size
        atr = float(row.get("atr_14", candle_range))
        if atr <= 0:
            atr = 1e-9
        atr_normalized_size = candle_range / atr

        # EMA details
        ema_dist = 0.0
        ema_dir = 0
        if "ema_50" in row:
            ema_val = float(row["ema_50"])
            ema_dist = (close_p - ema_val) / atr
            if idx > 0 and "ema_50" in df.columns:
                prev_ema = float(df.iloc[idx - 1]["ema_50"])
                ema_dir = 1 if ema_val > prev_ema else (-1 if ema_val < prev_ema else 0)

        # Volatility regime (ATR percentile/ratio)
        vol_regime = "NORMAL"
        atr_ratio = float(row.get("atr_ratio", 1.0))
        if atr_ratio >= 1.5:
            vol_regime = "HIGH"
        elif atr_ratio <= 0.7:
            vol_regime = "LOW"

        # Session
        session_val = "Asian"
        if "session" in row:
            session_val = str(row["session"])
        else:
            dt_val = pd.to_datetime(row["Datetime"]) if "Datetime" in df.columns else None
            if dt_val:
                h = dt_val.hour
                if 8 <= h < 13: session_val = "London"
                elif 13 <= h < 17: session_val = "London/NY"
                elif 17 <= h < 22: session_val = "NewYork"

        # 3. Strength & Classification Logic
        classification = "MEDIUM"
        reasons = []

        if body_ratio <= self.doji_threshold:
            classification = "DOJI"
            reasons.append(f"Body to range ratio {body_ratio:.2f} <= doji threshold {self.doji_threshold}")
        elif upper_wick_ratio >= self.exhaustion_wick_ratio and close_pos <= 0.3:
            classification = "EXHAUSTION"
            reasons.append(f"Large upper wick {upper_wick_ratio:.2f} with lower close")
        elif lower_wick_ratio >= self.exhaustion_wick_ratio and close_pos >= 0.7:
            classification = "EXHAUSTION"
            reasons.append(f"Large lower wick {lower_wick_ratio:.2f} with higher close")
        elif range_vs_avg >= self.climax_range_mult:
            classification = "CLIMAX"
            reasons.append(f"Range vs average {range_vs_avg:.2f} >= climax mult {self.climax_range_mult}")
        elif range_vs_avg >= self.expansion_range_mult and body_ratio >= self.expansion_body_ratio:
            classification = "EXPANSION"
            reasons.append(f"Strong body and large range expansion ({range_vs_avg:.2f}x)")
        elif body_ratio >= self.very_strong_body_ratio and range_vs_avg >= self.very_strong_range_mult:
            classification = "VERY_STRONG"
            reasons.append(f"Very strong body/range ratio {body_ratio:.2f} with good range")
        elif body_ratio >= self.strong_body_ratio:
            classification = "STRONG"
            reasons.append(f"Solid body/range ratio {body_ratio:.2f}")
        elif body_ratio >= self.medium_body_ratio:
            classification = "MEDIUM"
            reasons.append(f"Moderate body/range ratio {body_ratio:.2f}")
        elif body_ratio >= self.weak_body_ratio:
            classification = "WEAK"
            reasons.append(f"Weak body/range ratio {body_ratio:.2f}")
        else:
            classification = "INDECISION"
            reasons.append(f"Extremely small body/range {body_ratio:.2f} with large wicks")

        # 4. Scoring Logic (0-100)
        # Combine geometry, relative range, and volume support
        base_score = body_ratio * 60.0  # Up to 60 points from body ratio
        range_bonus = min(25.0, (range_vs_avg - 0.5) * 15.0)  # Up to 25 points from range size
        volume_bonus = min(15.0, (volume_vs_avg - 0.5) * 5.0) if volume_vs_avg > 1.0 else 0.0

        score_calc = max(0.0, min(100.0, base_score + range_bonus + volume_bonus))

        # Doji has naturally low trend-strength score, but high doji-quality score.
        # Let's override score if DOJI/EXHAUSTION for specific meanings, but general score indicates overall momentum/strength.
        if classification == "DOJI":
            score_calc = 100.0 * (1.0 - body_ratio) # High score means "very perfect doji"

        quality_score = int(round(score_calc))

        # Confidence (0.0 to 1.0)
        # Higher volume and range vs average increases confidence.
        confidence_calc = min(1.0, max(0.0, (body_ratio * 0.5) + (min(2.0, range_vs_avg) * 0.2) + (min(2.0, volume_vs_avg) * 0.1)))
        if classification == "DOJI":
            confidence_calc = 1.0 - body_ratio

        metrics = {
            "body_ratio": body_ratio,
            "upper_wick_ratio": upper_wick_ratio,
            "lower_wick_ratio": lower_wick_ratio,
            "close_pos": close_pos,
            "range_vs_avg": range_vs_avg,
            "volume_vs_avg": volume_vs_avg,
            "atr_normalized_size": atr_normalized_size,
            "ema_dist": ema_dist,
            "ema_dir": ema_dir,
            "vol_regime": vol_regime,
            "session": session_val
        }

        return StrongCandle(
            bullish=bullish,
            bearish=bearish,
            score=round(score_calc, 2),
            quality_score=quality_score,
            confidence=round(float(confidence_calc), 4),
            classification=classification,
            metrics=metrics,
            reasons=reasons
        )
