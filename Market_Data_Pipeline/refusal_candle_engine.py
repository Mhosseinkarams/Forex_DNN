import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from Market_Data_Pipeline.structure_graph import MarketStructureGraph, Zone

logger = logging.getLogger("RefusalCandleEngine")

@dataclass
class RefusalSignal:
    """
    Strongly-typed output of the RefusalCandleEngine.
    Follows the common scoring convention:
    - confidence: float [0.0, 1.0]
    - quality_score: int [0, 100]
    """
    bullish: bool            # True if rejecting a Demand level (potential BUY)
    bearish: bool            # True if rejecting a Supply level (potential SELL)
    score: float             # Legacy/general score (0-100)
    quality_score: int       # Common scoring convention (0-100)
    confidence: float        # Common scoring convention (0.0-1.0)
    classification: str      # PERFECT, HIGH, MEDIUM, LOW, INVALID
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class RefusalCandleEngine:
    """
    Purpose:
        Evaluate whether the current price bar is refusing/rejecting a structural S/D level,
        protected high/low, or swing level. Combines candle geometry (shape) and structural context.
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
        # Configurable parameters
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
        zone: Optional[Zone] = None,
        msg: Optional[MarketStructureGraph] = None
    ) -> RefusalSignal:
        """
        Evaluate candle rejection at the given index using shape and structural context.
        If zone is None, dynamically detects interaction with the nearest active zone.
        """
        if idx < 0:
            idx = len(df) + idx

        if idx < 1 or idx >= len(df):
            return RefusalSignal(
                bullish=False, bearish=False, score=0.0, quality_score=0, confidence=0.0,
                classification="INVALID", reasons=["Index out of bounds"], metrics={}
            )

        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]

        # Geometry extraction
        open_p = float(row["Open"])
        high_p = float(row["High"])
        low_p = float(row["Low"])
        close_p = float(row["Close"])
        volume = float(row.get("TickVolume", 0.0))

        prev_open = float(prev_row["Open"])
        prev_close = float(prev_row["Close"])

        candle_range = high_p - low_p
        if candle_range <= 0:
            return RefusalSignal(
                bullish=False, bearish=False, score=0.0, quality_score=0, confidence=0.0,
                classification="INVALID", reasons=["Candle range is zero"], metrics={}
            )

        body_size = abs(close_p - open_p)
        body_pct = body_size / candle_range
        upper_wick = high_p - max(open_p, close_p)
        lower_wick = min(open_p, close_p) - low_p
        upper_wick_pct = upper_wick / candle_range
        lower_wick_pct = lower_wick / candle_range

        atr_val = float(row.get("atr_14", candle_range))
        if atr_val <= 0:
            atr_val = 1e-9

        # Try to resolve zone dynamically if not provided
        if zone is None and msg is not None:
            # Look for active demand or supply zone that price is currently interacting with
            # High of candle touches or exceeds supply zone lower edge, OR low of candle touches or goes below demand zone upper edge.
            eligible_supplies = [z for z in msg.supply_zones if not z.broken and not z.invalidated and high_p >= z.lower]
            eligible_demands = [z for z in msg.demand_zones if not z.broken and not z.invalidated and low_p <= z.upper]

            # Select nearest / deepest interaction
            interacted_zones = []
            for sz in eligible_supplies:
                interacted_zones.append((sz, high_p - sz.lower, "Supply"))
            for dz in eligible_demands:
                interacted_zones.append((dz, dz.upper - low_p, "Demand"))

            if interacted_zones:
                # pick the zone with the deepest overlap/interaction
                interacted_zones.sort(key=lambda x: x[1], reverse=True)
                zone = interacted_zones[0][0]

        # If still no zone, resolve direction based on larger wick
        is_supply = False
        if zone is not None:
            is_supply = (zone.type == "Supply")
            bullish = not is_supply   # Rejection of demand zone -> bullish potential BUY
            bearish = is_supply      # Rejection of supply zone -> bearish potential SELL
        else:
            # Fallback to pure shape direction if no zone is found/provided
            bullish = lower_wick > upper_wick
            bearish = upper_wick > lower_wick

        reasons = []
        metrics = {}
        score = 0.0

        target_wick = upper_wick if bearish else lower_wick
        target_wick_pct = upper_wick_pct if bearish else lower_wick_pct

        # --- FACTOR 1: Wick/Body & Wick Ratios (Max Weight: weight_wick_ratio) ---
        wick_body_ratio = target_wick / (body_size + 1e-9)
        wick_score = 0.0
        if target_wick_pct >= self.min_wick_pct:
            wick_score += 50.0
        if wick_body_ratio >= self.min_wick_body_ratio:
            wick_score += 50.0

        score += (wick_score / 100.0) * self.weight_wick_ratio
        metrics["wick_body_ratio"] = wick_body_ratio
        metrics["wick_pct"] = target_wick_pct

        # --- FACTOR 2: Close Position within Candle (Max Weight: weight_close_pos) ---
        close_pos_pct = (close_p - low_p) / candle_range
        close_pos_score = 0.0
        if bearish:
            # lower close position is better (rejection from above)
            close_pos_score = max(0.0, (1.0 - close_pos_pct) * 100.0)
        else:
            # higher close position is better (rejection from below)
            close_pos_score = max(0.0, close_pos_pct * 100.0)

        score += (close_pos_score / 100.0) * self.weight_close_pos
        metrics["close_pos_pct"] = close_pos_pct

        # --- FACTOR 3: Zone Penetration Depth (Max Weight: weight_penetration) ---
        penetration_depth = 0.0
        if zone is not None:
            zone_width = zone.upper - zone.lower
            if zone_width <= 0:
                zone_width = 1e-9
            if is_supply:
                if high_p >= zone.lower:
                    penetration_depth = (min(high_p, zone.upper) - zone.lower) / zone_width
            else:
                if low_p <= zone.upper:
                    penetration_depth = (zone.upper - max(low_p, zone.lower)) / zone_width

        penetration_score = min(1.0, penetration_depth) * 100.0
        score += (penetration_score / 100.0) * self.weight_penetration
        metrics["penetration_depth"] = penetration_depth

        # --- FACTOR 4: Close Outside Zone (Max Weight: weight_close_outside) ---
        close_outside = False
        if zone is not None:
            if is_supply:
                close_outside = (close_p < zone.lower)
            else:
                close_outside = (close_p > zone.upper)
        else:
            close_outside = True # Fallback if no zone context

        close_outside_score = 100.0 if close_outside else 0.0
        score += (close_outside_score / 100.0) * self.weight_close_outside
        metrics["close_outside"] = close_outside

        # --- FACTOR 5: Previous Candle Direction & Momentum (Max Weight: weight_prev_momentum) ---
        prev_direction = 1 if prev_close > prev_open else (-1 if prev_close < prev_open else 0)
        prev_momentum_atr = abs(prev_close - prev_open) / atr_val

        momentum_score = 0.0
        if bearish and prev_direction == 1:
            momentum_score = min(1.0, prev_momentum_atr) * 100.0
        elif bullish and prev_direction == -1:
            momentum_score = min(1.0, prev_momentum_atr) * 100.0

        score += (momentum_score / 100.0) * self.weight_prev_momentum
        metrics["prev_direction"] = prev_direction
        metrics["prev_momentum_atr"] = prev_momentum_atr

        # --- FACTOR 6: Distance from Zone Center (Max Weight: weight_distance_center) ---
        dist_from_center = 0.0
        if zone is not None:
            zone_center = zone.mid
            zone_width = zone.upper - zone.lower
            if zone_width <= 0:
                zone_width = 1e-9
            dist_from_center = abs((high_p if bearish else low_p) - zone_center) / (zone_width / 2.0)
            dist_center_score = max(0.0, (1.0 - min(1.0, dist_from_center)) * 100.0)
        else:
            dist_center_score = 50.0 # moderate neutral fallback

        score += (dist_center_score / 100.0) * self.weight_distance_center
        metrics["dist_from_center"] = dist_from_center

        # --- FACTOR 7: Distance from Protected High/Low (Max Weight: weight_protected_dist) ---
        protected_score = 0.0
        if msg is not None:
            if bearish and msg.protected_high:
                dist_prot = abs(msg.protected_high.price - high_p) / atr_val
                protected_score = max(0.0, (1.0 - min(1.5, dist_prot)) * 100.0)
            elif bullish and msg.protected_low:
                dist_prot = abs(low_p - msg.protected_low.price) / atr_val
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

        # Candle Classification (PERFECT, HIGH, MEDIUM, LOW, INVALID)
        classification = "INVALID"

        # Geometry checks for refusal
        has_refusal_wick = target_wick_pct >= 0.40
        has_small_body = body_pct <= 0.40
        closes_in_direction = (close_p > open_p if bullish else close_p < open_p)

        if has_refusal_wick and has_small_body:
            if target_wick_pct >= 0.60 and closes_in_direction and (zone is not None) and close_outside:
                classification = "PERFECT"
                reasons.append("Perfect rejection: very long wick, small body, closed outside of zone in correct direction.")
            elif target_wick_pct >= 0.50 and (zone is not None or protected_score > 50):
                classification = "HIGH"
                reasons.append("High confidence: strong wick inside key structural level.")
            elif target_wick_pct >= 0.45:
                classification = "MEDIUM"
                reasons.append("Medium confidence rejection shape.")
            else:
                classification = "LOW"
                reasons.append("Weak refusal shape.")
        elif target_wick_pct >= 0.35:
            classification = "LOW"
            reasons.append("Low confidence rejection wick.")
        else:
            classification = "INVALID"
            reasons.append("Rejection candle does not satisfy basic wick requirements.")

        if zone is None:
            reasons.append("No supply/demand zone context found for this bar.")

        # Confidence is a product of wick strength and close position
        confidence = float(min(1.0, max(0.0, (target_wick_pct * 0.6 + (1.0 - abs(close_pos_pct - (1.0 if bullish else 0.0))) * 0.4))))

        # Refined reasons list
        if target_wick_pct >= self.min_wick_pct:
            reasons.append(f"Strong wick ratio: {target_wick_pct:.2f}")
        if close_outside and zone is not None:
            reasons.append("Closed outside structural zone boundaries")
        if penetration_depth > 0.5:
            reasons.append(f"Deep zone penetration: {penetration_depth:.2f}")
        if volume_score > 50.0:
            reasons.append("Volume spike confirmed on rejection bar")

        metrics["range_to_atr"] = candle_range / atr_val
        metrics["ema_distance_atr"] = getattr(msg, "ema_distance_atr", 0.0) if msg else 0.0

        return RefusalSignal(
            bullish=bullish,
            bearish=bearish,
            score=round(float(score), 2),
            quality_score=int(round(float(score))),
            confidence=round(confidence, 4),
            classification=classification,
            reasons=reasons,
            metrics=metrics
        )
