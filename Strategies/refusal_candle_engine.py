import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from Market_Data_Pipeline.structure_graph import MarketStructureGraph, Zone

logger = logging.getLogger("RefusalCandleEngine")

@dataclass
class RefusalResult:
    """
    Strongly-typed output from the RefusalCandleEngine.
    """
    score: float             # Calculated score (e.g. 0.0 to 100.0)
    confidence: float        # Confidence level (0.0 to 1.0)
    bullish: bool            # True if rejecting a Demand level (potential BUY)
    bearish: bool            # True if rejecting a Supply level (potential SELL)
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class RefusalCandleEngine:
    """
    Purpose:
        Evaluate whether the current price bar is refusing/rejecting a structural S/D level.
        Avoids simplistic hardcoded pin-bar logic, instead utilizing a configurable,
        multi-factor refusal scoring model.
    """
    def __init__(
        self,
        min_wick_body_ratio: float = 1.5,
        min_wick_pct: float = 0.50,
        atr_period: int = 14,
        volume_spike_mult: float = 1.5,
        weight_wick_ratio: float = 15.0,
        weight_close_pos: float = 20.0,
        weight_penetration: float = 15.0,
        weight_close_outside: float = 15.0,
        weight_prev_momentum: float = 10.0,
        weight_distance_center: float = 10.0,
        weight_protected_dist: float = 10.0,
        weight_volume_spike: float = 5.0
    ):
        # Threshold and tuning parameters
        self.min_wick_body_ratio = min_wick_body_ratio
        self.min_wick_pct = min_wick_pct
        self.atr_period = atr_period
        self.volume_spike_mult = volume_spike_mult

        # Weights summing up to 100.0 for scoring
        self.weight_wick_ratio = weight_wick_ratio
        self.weight_close_pos = weight_close_pos
        self.weight_penetration = weight_penetration
        self.weight_close_outside = weight_close_outside
        self.weight_prev_momentum = weight_prev_momentum
        self.weight_distance_center = weight_distance_center
        self.weight_protected_dist = weight_protected_dist
        self.weight_volume_spike = weight_volume_spike

    def evaluate_rejection(
        self,
        df: pd.DataFrame,
        idx: int,
        zone: Zone,
        msg: MarketStructureGraph
    ) -> RefusalResult:
        """
        Evaluate candle rejection at the given index against a specific Supply or Demand zone.
        """
        if idx < 1:
            return RefusalResult(0.0, 0.0, False, False, ["Index out of bounds"], {})

        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]

        # Basic candle metrics
        open_p = float(row["Open"])
        high_p = float(row["High"])
        low_p = float(row["Low"])
        close_p = float(row["Close"])
        volume = float(row.get("TickVolume", 0.0))

        prev_open = float(prev_row["Open"])
        prev_high = float(prev_row["High"])
        prev_low = float(prev_row["Low"])
        prev_close = float(prev_row["Close"])
        prev_volume = float(prev_row.get("TickVolume", 0.0))

        candle_range = high_p - low_p
        if candle_range <= 0:
            return RefusalResult(0.0, 0.0, False, False, ["Candle range is zero"], {})

        body_size = abs(close_p - open_p)
        upper_wick = high_p - max(open_p, close_p)
        lower_wick = min(open_p, close_p) - low_p

        # Direction detection based on zone type
        is_supply = (zone.type == "Supply")
        bullish = not is_supply   # Rejection of demand zone implies a bullish reversal
        bearish = is_supply      # Rejection of supply zone implies a bearish reversal

        reasons = []
        metrics = {}
        score = 0.0

        # --- FACTOR 1: Wick/Body & Wick Ratios (Max Weight: weight_wick_ratio) ---
        wick_body_ratio = (upper_wick if bearish else lower_wick) / (body_size + 1e-9)
        wick_pct = (upper_wick if bearish else lower_wick) / candle_range

        wick_score = 0.0
        if wick_pct >= self.min_wick_pct:
            wick_score += 50.0
        if wick_body_ratio >= self.min_wick_body_ratio:
            wick_score += 50.0

        score += (wick_score / 100.0) * self.weight_wick_ratio
        metrics["wick_body_ratio"] = wick_body_ratio
        metrics["wick_pct"] = wick_pct

        # --- FACTOR 2: Close Position within Candle (Max Weight: weight_close_pos) ---
        # For bearish: close should be near the bottom (0.0 = bottom, 1.0 = top)
        # For bullish: close should be near the top
        close_pos_pct = (close_p - low_p) / candle_range
        close_pos_score = 0.0
        if bearish:
            # lower position is better (rejection from above)
            close_pos_score = max(0.0, (1.0 - close_pos_pct) * 100.0)
        else:
            # higher position is better (rejection from below)
            close_pos_score = max(0.0, close_pos_pct * 100.0)

        score += (close_pos_score / 100.0) * self.weight_close_pos
        metrics["close_pos_pct"] = close_pos_pct

        # --- FACTOR 3: Zone Penetration Depth (Max Weight: weight_penetration) ---
        # How deep into the zone did price reach compared to the zone width?
        zone_width = zone.upper - zone.lower
        penetration_depth = 0.0
        if is_supply:
            # Penetration of supply: High of candle minus lower boundary of supply
            if high_p >= zone.lower:
                penetration_depth = (min(high_p, zone.upper) - zone.lower) / (zone_width + 1e-9)
        else:
            # Penetration of demand: Upper boundary of demand minus Low of candle
            if low_p <= zone.upper:
                penetration_depth = (zone.upper - max(low_p, zone.lower)) / (zone_width + 1e-9)

        # Deep penetration but not breaking is high rejection signal
        penetration_score = min(1.0, penetration_depth) * 100.0
        score += (penetration_score / 100.0) * self.weight_penetration
        metrics["penetration_depth"] = penetration_depth

        # --- FACTOR 4: Close Outside Zone (Max Weight: weight_close_outside) ---
        # Price rejects the zone if it closes outside the zone boundaries
        close_outside = False
        if is_supply:
            close_outside = (close_p < zone.lower)
        else:
            close_outside = (close_p > zone.upper)

        close_outside_score = 100.0 if close_outside else 0.0
        score += (close_outside_score / 100.0) * self.weight_close_outside
        metrics["close_outside"] = close_outside

        # --- FACTOR 5: Previous Candle Direction & Momentum (Max Weight: weight_prev_momentum) ---
        # For Supply rejection (bearish), we want previous candles to be bullish/incoming, showing a stop/reversal
        # For Demand rejection (bullish), we want previous candles to be bearish/incoming.
        prev_direction = 1 if prev_close > prev_open else (-1 if prev_close < prev_open else 0)
        prev_momentum_atr = abs(prev_close - prev_open) / (row.get("atr_14", 0.0001) + 1e-9)

        momentum_score = 0.0
        if bearish and prev_direction == 1:
            # Came in strong bullish, now rejecting
            momentum_score = min(1.0, prev_momentum_atr) * 100.0
        elif bullish and prev_direction == -1:
            # Came in strong bearish, now rejecting
            momentum_score = min(1.0, prev_momentum_atr) * 100.0

        score += (momentum_score / 100.0) * self.weight_prev_momentum
        metrics["prev_direction"] = prev_direction
        metrics["prev_momentum_atr"] = prev_momentum_atr

        # --- FACTOR 6: Distance from Zone Center (Max Weight: weight_distance_center) ---
        # Rejections are strongest when price penetrates close to the zone center
        zone_center = zone.mid
        dist_from_center = abs((high_p if bearish else low_p) - zone_center) / (zone_width / 2.0 + 1e-9)
        # Closer to center/beyond center is better (dist_from_center near 0 or negative is best, let's invert)
        dist_center_score = max(0.0, (1.0 - min(1.0, dist_from_center)) * 100.0)
        score += (dist_center_score / 100.0) * self.weight_distance_center
        metrics["dist_from_center"] = dist_from_center

        # --- FACTOR 7: Distance from Protected High/Low (Max Weight: weight_protected_dist) ---
        # Is there a nearby protected level that is holding?
        protected_score = 0.0
        if bearish and msg.protected_high:
            dist_prot = abs(msg.protected_high.price - high_p) / (row.get("atr_14", 0.0001) + 1e-9)
            protected_score = max(0.0, (1.0 - min(1.5, dist_prot)) * 100.0)
        elif bullish and msg.protected_low:
            dist_prot = abs(low_p - msg.protected_low.price) / (row.get("atr_14", 0.0001) + 1e-9)
            protected_score = max(0.0, (1.0 - min(1.5, dist_prot)) * 100.0)

        score += (protected_score / 100.0) * self.weight_protected_dist
        metrics["protected_score"] = protected_score

        # --- FACTOR 8: Volume Spike (Max Weight: weight_volume_spike) ---
        volume_score = 0.0
        if idx >= 10:
            avg_volume = df.iloc[idx-10:idx]["TickVolume"].mean()
            if avg_volume > 0:
                vol_ratio = volume / avg_volume
                metrics["volume_ratio"] = vol_ratio
                if vol_ratio >= self.volume_spike_mult:
                    volume_score = min(100.0, (vol_ratio / self.volume_spike_mult) * 50.0 + 50.0)
            else:
                metrics["volume_ratio"] = 1.0
        else:
            metrics["volume_ratio"] = 1.0

        score += (volume_score / 100.0) * self.weight_volume_spike

        # ATR normalization context
        atr_val = row.get("atr_14", 0.0001)
        metrics["range_to_atr"] = candle_range / (atr_val + 1e-9)

        # Context details
        metrics["ema_distance_atr"] = getattr(msg, "ema_distance_atr", 0.0)

        # Reasons logging
        if wick_pct >= self.min_wick_pct:
            reasons.append(f"Strong wick ratio: {wick_pct:.2f}")
        if close_outside:
            reasons.append("Closed outside structural zone boundaries")
        if penetration_depth > 0.5:
            reasons.append(f"Deep zone penetration: {penetration_depth:.2f}")
        if volume_score > 50.0:
            reasons.append("Volume spike confirmed on rejection bar")

        # Confidence is a product of wick strength and close position
        confidence = float(min(1.0, max(0.0, (wick_pct * 0.6 + (1.0 - abs(close_pos_pct - (1.0 if bullish else 0.0))) * 0.4))))

        return RefusalResult(
            score=round(float(score), 2),
            confidence=round(confidence, 4),
            bullish=bullish,
            bearish=bearish,
            reasons=reasons,
            metrics=metrics
        )
