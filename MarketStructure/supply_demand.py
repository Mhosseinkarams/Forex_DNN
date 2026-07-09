import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Configure logger
logger = logging.getLogger('SupplyDemandEngine')

@dataclass
class Zone:
    upper: float
    lower: float
    type: str  # 'Supply' or 'Demand'
    created_time: datetime
    created_idx: int
    freshness: bool = True
    touch_count: int = 0
    broken: bool = False
    broken_idx: Optional[int] = None
    mitigated: bool = False
    mitigated_idx: Optional[int] = None
    strength_score: float = 0.0

    @property
    def mid(self) -> float:
        return (self.upper + self.lower) / 2

    @property
    def width(self) -> float:
        return self.upper - self.lower

class SupplyDemandEngine:
    """
    Purpose:
        Detect institutional Supply and Demand zones.
        Stateless incremental processing to avoid look-ahead bias and improve performance.
    """
    def __init__(self, atr_period: int = 14, impulse_threshold: float = 2.0):
        self.atr_period = atr_period
        self.impulse_threshold = impulse_threshold
        self.zones: List[Zone] = []

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self.zones = [] # Stateless for the call

        closes = df['Close'].values; opens = df['Open'].values
        highs = df['High'].values; lows = df['Low'].values
        times = df['Datetime'].values if 'Datetime' in df.columns else [None] * len(df)

        # Pre-calculate ATR
        prev_close = df['Close'].shift(1)
        tr = pd.concat([df['High'] - df['Low'], (df['High'] - prev_close).abs(), (df['Low'] - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean().values

        # Output arrays
        ns_dist = np.full(len(df), np.nan); nd_dist = np.full(len(df), np.nan)
        in_s = np.zeros(len(df)); in_d = np.zeros(len(df))
        s_st = np.zeros(len(df)); d_st = np.zeros(len(df))
        bs_s = np.full(len(df), -1); bs_d = np.full(len(df), -1)

        active_zones = [] # Zones not yet broken

        for i in range(1, len(df)):
            # 1. Zone detection (look back at i-1)
            if not np.isnan(atr[i]):
                move = closes[i] - opens[i]
                if abs(move) > atr[i] * self.impulse_threshold:
                    base_idx = i - 1
                    if move > 0:
                        new_zone = Zone(upper=max(opens[base_idx], closes[base_idx]), lower=lows[base_idx], type='Demand', created_time=times[i], created_idx=i)
                    else:
                        new_zone = Zone(upper=highs[base_idx], lower=min(opens[base_idx], closes[base_idx]), type='Supply', created_time=times[i], created_idx=i)

                    # Initial strength (no look-ahead)
                    new_zone.strength_score = abs(move) / (atr[i] if atr[i] > 0 else 1e-9)
                    active_zones.append(new_zone)

            # 2. Update active zones (mitigation, breakage, incremental strength)
            curr_low = lows[i]; curr_high = highs[i]; curr_close = closes[i]
            still_active = []

            for z in active_zones:
                # Incremental Strength (Departure speed within first 3 bars after creation)
                if 0 < i - z.created_idx <= 3:
                    move_away = closes[i] - closes[i-1]
                    if (z.type == 'Demand' and move_away > 0) or (z.type == 'Supply' and move_away < 0):
                        z.strength_score += abs(move_away) / (atr[i] if atr[i] > 0 else 1e-9)

                if z.type == 'Demand':
                    if curr_low < z.upper:
                        if not z.mitigated:
                            z.mitigated = True; z.mitigated_idx = i; z.freshness = False
                        if lows[i-1] >= z.upper:
                            z.touch_count += 1
                            z.strength_score = max(0, z.strength_score - 0.5)
                    if curr_close < z.lower:
                        z.broken = True; z.broken_idx = i
                    else:
                        still_active.append(z)
                else: # Supply
                    if curr_high > z.lower:
                        if not z.mitigated:
                            z.mitigated = True; z.mitigated_idx = i; z.freshness = False
                        if highs[i-1] <= z.lower:
                            z.touch_count += 1
                            z.strength_score = max(0, z.strength_score - 0.5)
                    if curr_close > z.upper:
                        z.broken = True; z.broken_idx = i
                    else:
                        still_active.append(z)
            active_zones = still_active

            # 3. Fill results for current bar
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

        df['nearest_supply_distance'] = ns_dist; df['nearest_demand_distance'] = nd_dist
        df['inside_supply'] = in_s; df['inside_demand'] = in_d
        df['supply_strength'] = s_st; df['demand_strength'] = d_st
        df['bars_since_supply'] = bs_s; df['bars_since_demand'] = bs_d

        return df.loc[:, ~df.columns.duplicated()]
