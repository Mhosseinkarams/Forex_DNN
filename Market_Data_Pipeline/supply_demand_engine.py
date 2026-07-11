import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from Market_Data_Pipeline.structure_graph import Zone

logger = logging.getLogger('SupplyDemandEngine')

class SupplyDemandEngine:
    """
    Purpose:
        Version 1.0 Production-Ready Supply and Demand Engine.
        Detects institutional Supply and Demand zones based on impulsive moves.
        Manages comprehensive zone lifecycles: mitigation, retesting, breakage,
        invalidation, and nested containment.
    """
    def __init__(self, atr_period: int = 14, impulse_threshold: float = 2.0):
        self.atr_period = atr_period
        self.impulse_threshold = impulse_threshold
        self.zones: List[Zone] = []

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self.zones = [] # Reset stateless call

        closes = df['Close'].values
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        times = df['Datetime'].values if 'Datetime' in df.columns else [None] * len(df)

        # Pre-calculate ATR
        prev_close = df['Close'].shift(1)
        tr = pd.concat([df['High'] - df['Low'], (df['High'] - prev_close).abs(), (df['Low'] - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean().values

        # Output arrays
        ns_dist = np.full(len(df), np.nan)
        nd_dist = np.full(len(df), np.nan)
        in_s = np.zeros(len(df))
        in_d = np.zeros(len(df))
        s_st = np.zeros(len(df))
        d_st = np.zeros(len(df))
        bs_s = np.full(len(df), -1)
        bs_d = np.full(len(df), -1)

        active_zones: List[Zone] = []

        for i in range(1, len(df)):
            curr_atr = atr[i] if (not np.isnan(atr[i]) and atr[i] > 0) else 0.0001
            dt_i = pd.to_datetime(times[i]) if times[i] is not None else None

            # 1. Zone detection (look back at impulsive moves at i)
            move = closes[i] - opens[i]
            if abs(move) > curr_atr * self.impulse_threshold:
                base_idx = i - 1
                dt_base = pd.to_datetime(times[base_idx]) if times[base_idx] is not None else None

                new_zone = None
                if move > 0:
                    # Demand: base high to base low
                    upper_p = float(max(opens[base_idx], closes[base_idx]))
                    lower_p = float(lows[base_idx])
                    new_zone = Zone(
                        upper=upper_p,
                        lower=lower_p,
                        type='Demand',
                        created_time=dt_base,
                        created_idx=base_idx,
                        creation_candle=base_idx,
                        origin_candle=i,
                        freshness=True,
                        touch_count=0,
                        broken=False,
                        mitigated=False,
                        strength_score=float(abs(move) / curr_atr),
                        freshness_score=1.0,
                        number_of_reactions=0,
                        average_rejection=0.0,
                        average_penetration=0.0,
                        invalidated=False,
                        active=True,
                        why_detected=f"Impulsive move up {move:.5f} (> {self.impulse_threshold} * ATR) detected at bar {i}.",
                        rule_fired="impulsive_demand_creation",
                        thresholds_satisfied=[f"move_size_{abs(move):.5f} > atr_threshold_{curr_atr * self.impulse_threshold:.5f}"],
                        confidence_score=min(1.0, abs(move) / (curr_atr * 4.0))
                    )
                else:
                    # Supply: base high to base low
                    upper_p = float(highs[base_idx])
                    lower_p = float(min(opens[base_idx], closes[base_idx]))
                    new_zone = Zone(
                        upper=upper_p,
                        lower=lower_p,
                        type='Supply',
                        created_time=dt_base,
                        created_idx=base_idx,
                        creation_candle=base_idx,
                        origin_candle=i,
                        freshness=True,
                        touch_count=0,
                        broken=False,
                        mitigated=False,
                        strength_score=float(abs(move) / curr_atr),
                        freshness_score=1.0,
                        number_of_reactions=0,
                        average_rejection=0.0,
                        average_penetration=0.0,
                        invalidated=False,
                        active=True,
                        why_detected=f"Impulsive move down {move:.5f} (> {self.impulse_threshold} * ATR) detected at bar {i}.",
                        rule_fired="impulsive_supply_creation",
                        thresholds_satisfied=[f"move_size_{abs(move):.5f} > atr_threshold_{curr_atr * self.impulse_threshold:.5f}"],
                        confidence_score=min(1.0, abs(move) / (curr_atr * 4.0))
                    )

                if new_zone:
                    # Check nested zone containment
                    for old_z in active_zones:
                        if old_z.type == new_zone.type:
                            if new_zone.lower >= old_z.lower and new_zone.upper <= old_z.upper:
                                new_zone.nested_inside_idx = old_z.created_idx
                                break

                    active_zones.append(new_zone)
                    self.zones.append(new_zone)

            # 2. Update active zones lifecycles (mitigation, retests, breakage, reactions)
            curr_low = lows[i]
            curr_high = highs[i]
            curr_close = closes[i]
            still_active = []

            for z in active_zones:
                # Calculate Departure Speed (first 3 bars)
                if 0 < i - z.origin_candle <= 3:
                    move_away = closes[i] - closes[i-1]
                    if (z.type == 'Demand' and move_away > 0) or (z.type == 'Supply' and move_away < 0):
                        z.strength_score += float(abs(move_away) / curr_atr)

                # Process interaction based on zone type
                if z.type == 'Demand':
                    # Touch/Retest/Mitigation check
                    if curr_low < z.upper:
                        if not z.mitigated:
                            z.mitigated = True
                            z.mitigated_idx = i
                            z.freshness = False

                        # Retest details tracking (only count distinct reactions if previous bar was outside)
                        if lows[i-1] >= z.upper:
                            z.touch_count += 1
                            z.number_of_reactions += 1
                            # Penalize freshness score
                            z.freshness_score = float(max(0.0, z.freshness_score - 0.25))
                            z.strength_score = float(max(0.0, z.strength_score - 0.5))

                            # Penetration depth
                            penetration = float((z.upper - curr_low) / z.width) if z.width > 0 else 0.0
                            z.average_penetration = (z.average_penetration * (z.touch_count - 1) + penetration) / z.touch_count

                            # Rejection reaction size (from low to close)
                            rejection = float((curr_close - curr_low) / curr_atr)
                            z.average_rejection = (z.average_rejection * (z.touch_count - 1) + rejection) / z.touch_count

                    # Breakage / Invalidation Check
                    if curr_close < z.lower:
                        z.broken = True
                        z.broken_idx = i
                        z.invalidated = True
                        z.active = False
                        z.freshness = False
                    else:
                        still_active.append(z)

                else: # Supply
                    # Touch/Retest/Mitigation check
                    if curr_high > z.lower:
                        if not z.mitigated:
                            z.mitigated = True
                            z.mitigated_idx = i
                            z.freshness = False

                        if highs[i-1] <= z.lower:
                            z.touch_count += 1
                            z.number_of_reactions += 1
                            z.freshness_score = float(max(0.0, z.freshness_score - 0.25))
                            z.strength_score = float(max(0.0, z.strength_score - 0.5))

                            penetration = float((curr_high - z.lower) / z.width) if z.width > 0 else 0.0
                            z.average_penetration = (z.average_penetration * (z.touch_count - 1) + penetration) / z.touch_count

                            rejection = float((curr_high - curr_close) / curr_atr)
                            z.average_rejection = (z.average_rejection * (z.touch_count - 1) + rejection) / z.touch_count

                    # Breakage / Invalidation Check
                    if curr_close > z.upper:
                        z.broken = True
                        z.broken_idx = i
                        z.invalidated = True
                        z.active = False
                        z.freshness = False
                    else:
                        still_active.append(z)

            active_zones = still_active

            # 3. Fill results for current bar based on nearest supply/demand
            supplies = [z for z in active_zones if z.type == 'Supply' and z.lower > curr_close]
            demands = [z for z in active_zones if z.type == 'Demand' and z.upper < curr_close]

            if supplies:
                nz = min(supplies, key=lambda z: z.lower)
                ns_dist[i] = nz.lower - curr_close
                s_st[i] = nz.strength_score
                bs_s[i] = i - nz.created_idx
            if any(curr_high >= z.lower and curr_low <= z.upper for z in active_zones if z.type == 'Supply'):
                in_s[i] = 1

            if demands:
                nz = max(demands, key=lambda z: z.upper)
                nd_dist[i] = curr_close - nz.upper
                d_st[i] = nz.strength_score
                bs_d[i] = i - nz.created_idx
            if any(curr_low <= z.upper and curr_high >= z.lower for z in active_zones if z.type == 'Demand'):
                in_d[i] = 1

        df['nearest_supply_distance'] = ns_dist
        df['nearest_demand_distance'] = nd_dist
        df['inside_supply'] = in_s
        df['inside_demand'] = in_d
        df['supply_strength'] = s_st
        df['demand_strength'] = d_st
        df['bars_since_supply'] = bs_s
        df['bars_since_demand'] = bs_d

        return df.loc[:, ~df.columns.duplicated()]
