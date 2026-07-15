import pandas as pd
import numpy as np
import logging
import math
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
    created_time: Optional[datetime] = None
    created_idx: int = 0
    freshness: bool = True
    touch_count: int = 0
    broken: bool = False
    broken_idx: Optional[int] = None
    mitigated: bool = False
    mitigated_idx: Optional[int] = None
    strength_score: float = 0.0

    # Version 1.0 additions
    creation_candle: int = 0
    origin_candle: int = 0
    freshness_score: float = 1.0
    number_of_reactions: int = 0
    average_rejection: float = 0.0
    average_penetration: float = 0.0
    invalidated: bool = False
    active: bool = True
    nested_inside_idx: Optional[int] = None

    # Debugging and Support fields
    why_detected: str = ""
    rule_fired: str = ""
    thresholds_satisfied: List[str] = field(default_factory=list)
    thresholds_failed: List[str] = field(default_factory=list)
    confidence_score: float = 1.0

    @property
    def mid(self) -> float:
        return (self.upper + self.lower) / 2

    @property
    def width(self) -> float:
        return self.upper - self.lower

@dataclass
class FVG:
    upper: float
    lower: float
    bias: int  # 1: Bullish, -1: Bearish
    created_idx: int
    created_time: Optional[datetime] = None
    active: bool = True
    mitigated_idx: Optional[int] = None

@dataclass
class EQHLPattern:
    index: int
    pattern_type: str  # 'EQH' or 'EQL'
    price: float
    timestamp: Optional[datetime] = None

class SupplyDemandEngine:
    """
    SMC Order Block and Fair Value Gap (FVG) Engine.
    Detects supply and demand zones as Order Blocks (OBs) formed by structure breaks.
    Supports mitigation, FVG confluence, and EQH/EQL tracking.
    """
    def __init__(
        self,
        atr_period: int = 14,
        impulse_threshold: float = 2.0,
        back_limit: int = 5000,
        zone_show_weak: bool = True,
        zone_show_untested: bool = True,
        zone_show_turncoat: bool = True,
        zone_fuzzfactor: float = 0.75,
        fractal_fast_factor: float = 3.0,
        fractal_slow_factor: float = 6.0,
        zone_merge: bool = True,
        zone_extend: bool = True,
        use_fractal: bool = True
    ):
        self.atr_period = atr_period
        self.impulse_threshold = impulse_threshold
        self.back_limit = back_limit
        self.zone_show_weak = zone_show_weak
        self.zone_show_untested = zone_show_untested
        self.zone_show_turncoat = zone_show_turncoat
        self.zone_fuzzfactor = zone_fuzzfactor
        self.fractal_fast_factor = fractal_fast_factor
        self.fractal_slow_factor = fractal_slow_factor
        self.zone_merge = zone_merge
        self.zone_extend = zone_extend
        self.use_fractal = use_fractal

        # State / Outputs
        self.zones: List[Zone] = []
        self.fvgs: List[FVG] = []
        self.eqhl_patterns: List[EQHLPattern] = []

    def _fill_empty_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        n_bars = len(df)
        df['nearest_supply_distance'] = np.full(n_bars, np.nan)
        df['nearest_demand_distance'] = np.full(n_bars, np.nan)
        df['inside_supply'] = np.zeros(n_bars)
        df['inside_demand'] = np.zeros(n_bars)
        df['supply_strength'] = np.zeros(n_bars)
        df['demand_strength'] = np.zeros(n_bars)
        df['supply_freshness'] = np.zeros(n_bars)
        df['demand_freshness'] = np.zeros(n_bars)
        df['supply_touch_count'] = np.zeros(n_bars)
        df['demand_touch_count'] = np.zeros(n_bars)
        df['bars_since_supply'] = np.full(n_bars, -1)
        df['bars_since_demand'] = np.full(n_bars, -1)
        return df.loc[:, ~df.columns.duplicated()]

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.use_fractal:
            return self._process_impulse(df)

        df = df.copy()
        n_bars = len(df)
        self.zones = []
        self.fvgs = []
        self.eqhl_patterns = []

        if n_bars < 25:
            return self._fill_empty_columns(df)

        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        opens = df['Open'].values
        times = df['Datetime'].values if 'Datetime' in df.columns else [None] * n_bars

        # Precompute ATR
        prev_close = df['Close'].shift(1)
        tr = pd.concat([df['High'] - df['Low'], (df['High'] - prev_close).abs(), (df['Low'] - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean().fillna(0.0010).values

        # Config parameters matching MQL5
        lookback = 10  # Swing Length (InpSwingLength)
        inp_max_ob = 5  # InpMaxOB

        # Precompute Pivot Highs & Pivot Lows
        piv_highs = np.zeros(n_bars, dtype=bool)
        piv_lows = np.zeros(n_bars, dtype=bool)

        for i in range(lookback, n_bars - lookback):
            p_high = highs[i]
            is_ph = True
            for k in range(1, lookback + 1):
                if highs[i - k] >= p_high or highs[i + k] >= p_high:
                    is_ph = False
                    break
            piv_highs[i] = is_ph

            p_low = lows[i]
            is_pl = True
            for k in range(1, lookback + 1):
                if lows[i - k] <= p_low or lows[i + k] <= p_low:
                    is_pl = False
                    break
            piv_lows[i] = is_pl

        # Equal Highs / Lows tracker
        eqh_pivs = np.zeros(n_bars, dtype=bool)
        eql_pivs = np.zeros(n_bars, dtype=bool)
        eqh_len = 5
        for i in range(eqh_len, n_bars - eqh_len):
            p_high = highs[i]
            is_ph = True
            for k in range(1, eqh_len + 1):
                if highs[i - k] >= p_high or highs[i + k] >= p_high:
                    is_ph = False
                    break
            eqh_pivs[i] = is_ph

            p_low = lows[i]
            is_pl = True
            for k in range(1, eqh_len + 1):
                if lows[i - k] <= p_low or lows[i + k] <= p_low:
                    is_pl = False
                    break
            eql_pivs[i] = is_pl

        # Output arrays
        ns_dist = np.full(n_bars, np.nan)
        nd_dist = np.full(n_bars, np.nan)
        in_s = np.zeros(n_bars)
        in_d = np.zeros(n_bars)
        s_st = np.zeros(n_bars)
        d_st = np.zeros(n_bars)
        s_fresh = np.zeros(n_bars)
        d_fresh = np.zeros(n_bars)
        s_touches = np.zeros(n_bars)
        d_touches = np.zeros(n_bars)
        bs_s = np.full(n_bars, -1)
        bs_d = np.full(n_bars, -1)

        # running state for breakout tracking to align OB creation exactly
        g_phLevel = 0.0
        g_plLevel = 0.0
        g_trend = 0

        # EQH running state
        last_eqh_level = 0.0
        last_eql_level = 0.0

        # OB lists
        bull_OBs: List[Zone] = []
        bear_OBs: List[Zone] = []

        # Sequential bar-by-bar simulation
        for t in range(n_bars):
            idx = t - lookback
            # 1. Structure Breakout check for OB creation
            if idx >= lookback:
                if piv_highs[idx]:
                    pivPrice = highs[idx]
                    pivTime = times[idx] if isinstance(times[idx], datetime) else (pd.to_datetime(times[idx]) if times[idx] is not None else None)

                    if g_phLevel > 0 and pivPrice > g_phLevel:
                        g_trend = 1
                        # Create Bullish Order Block (Demand Zone)
                        found = -1
                        for k in range(idx - 1, max(0, idx - lookback) - 1, -1):
                            if closes[k] < opens[k]:
                                found = k
                                break
                        if found != -1:
                            bodyTop = max(opens[found], closes[found])
                            bodyBot = min(opens[found], closes[found])
                            bodyMid = (bodyTop + bodyBot) / 2.0
                            obTop = bodyMid
                            obBottom = bodyBot

                            # Duplication check
                            duplicate = False
                            for ob in bull_OBs:
                                if ob.active and abs(ob.upper - obTop) < atr[t] * 0.3:
                                    duplicate = True
                                    break

                            if obTop - obBottom >= 1e-6 and not duplicate:
                                # Keep under limit
                                active_bull = [o for o in bull_OBs if o.active]
                                if len(active_bull) >= inp_max_ob:
                                    active_bull[0].active = False
                                    active_bull[0].broken = True
                                    active_bull[0].broken_idx = t

                                new_zone = Zone(
                                    upper=float(obTop),
                                    lower=float(obBottom),
                                    type='Demand',
                                    created_idx=found,
                                    created_time=times[found],
                                    freshness=True,
                                    touch_count=0,
                                    strength_score=3.0,
                                    creation_candle=found,
                                    origin_candle=idx,
                                    active=True
                                )
                                bull_OBs.append(new_zone)
                                self.zones.append(new_zone)

                    g_phLevel = pivPrice

                if piv_lows[idx]:
                    pivPrice = lows[idx]
                    pivTime = times[idx] if isinstance(times[idx], datetime) else (pd.to_datetime(times[idx]) if times[idx] is not None else None)

                    if g_plLevel > 0 and pivPrice < g_plLevel:
                        g_trend = -1
                        # Create Bearish Order Block (Supply Zone)
                        found = -1
                        for k in range(idx - 1, max(0, idx - lookback) - 1, -1):
                            if closes[k] > opens[k]:
                                found = k
                                break
                        if found != -1:
                            bodyTop = max(opens[found], closes[found])
                            bodyBot = min(opens[found], closes[found])
                            bodyMid = (bodyTop + bodyBot) / 2.0
                            obTop = bodyTop
                            obBottom = bodyMid

                            # Duplication check
                            duplicate = False
                            for ob in bear_OBs:
                                if ob.active and abs(ob.upper - obTop) < atr[t] * 0.3:
                                    duplicate = True
                                    break

                            if obTop - obBottom >= 1e-6 and not duplicate:
                                active_bear = [o for o in bear_OBs if o.active]
                                if len(active_bear) >= inp_max_ob:
                                    active_bear[0].active = False
                                    active_bear[0].broken = True
                                    active_bear[0].broken_idx = t

                                new_zone = Zone(
                                    upper=float(obTop),
                                    lower=float(obBottom),
                                    type='Supply',
                                    created_idx=found,
                                    created_time=times[found],
                                    freshness=True,
                                    touch_count=0,
                                    strength_score=3.0,
                                    creation_candle=found,
                                    origin_candle=idx,
                                    active=True
                                )
                                bear_OBs.append(new_zone)
                                self.zones.append(new_zone)

                    g_plLevel = pivPrice

            # 2. EQH / EQL detection (Pivot length 5)
            idx_eq = t - eqh_len
            if idx_eq >= eqh_len:
                if eqh_pivs[idx_eq]:
                    pivPrice = highs[idx_eq]
                    if last_eqh_level > 0:
                        diff = abs(pivPrice - last_eqh_level)
                        if diff > 0 and diff < 0.5 * atr[t]:
                            self.eqhl_patterns.append(EQHLPattern(index=idx_eq, pattern_type='EQH', price=float(pivPrice), timestamp=times[idx_eq]))
                    last_eqh_level = pivPrice

                if eql_pivs[idx_eq]:
                    pivPrice = lows[idx_eq]
                    if last_eql_level > 0:
                        diff = abs(pivPrice - last_eql_level)
                        if diff > 0 and diff < 0.5 * atr[t]:
                            self.eqhl_patterns.append(EQHLPattern(index=idx_eq, pattern_type='EQL', price=float(pivPrice), timestamp=times[idx_eq]))
                    last_eql_level = pivPrice

            # 3. FVG detection
            if t >= 2:
                # Bullish FVG
                if lows[t] > highs[t - 2]:
                    fTop = lows[t]
                    fBot = highs[t - 2]
                    if fTop - fBot >= 0.15 * atr[t]:
                        duplicate = False
                        for f in self.fvgs:
                            if f.active and f.bias == 1 and abs(f.upper - fTop) < atr[t] * 0.2:
                                duplicate = True
                                break
                        if not duplicate:
                            self.fvgs.append(FVG(upper=float(fTop), lower=float(fBot), bias=1, created_idx=t-1, created_time=times[t-1], active=True))
                # Bearish FVG
                if highs[t] < lows[t - 2]:
                    fTop = lows[t - 2]
                    fBot = highs[t]
                    if fTop - fBot >= 0.15 * atr[t]:
                        duplicate = False
                        for f in self.fvgs:
                            if f.active and f.bias == -1 and abs(f.upper - fTop) < atr[t] * 0.2:
                                duplicate = True
                                break
                        if not duplicate:
                            self.fvgs.append(FVG(upper=float(fTop), lower=float(fBot), bias=-1, created_idx=t-1, created_time=times[t-1], active=True))

            # 4. Mitigation checks & touch calculations sequentially
            curr_close = closes[t]
            curr_low = lows[t]
            curr_high = highs[t]

            # FVG mitigation
            for f in self.fvgs:
                if f.active:
                    if f.bias == 1:
                        if curr_low <= f.upper and curr_low >= f.lower:
                            f.active = False
                            f.mitigated_idx = t
                    else:
                        if curr_high >= f.lower and curr_high <= f.upper:
                            f.active = False
                            f.mitigated_idx = t

            # OB Mitigation & Touch
            for ob in self.zones:
                if ob.active:
                    if ob.type == 'Demand':
                        # Touch logic
                        if curr_low <= ob.upper:
                            if not ob.mitigated:
                                ob.mitigated = True
                                ob.mitigated_idx = t
                                ob.freshness = False
                            ob.touch_count += 1
                            ob.number_of_reactions += 1
                        # Break check
                        if curr_close < ob.lower:
                            ob.active = False
                            ob.broken = True
                            ob.broken_idx = t
                    else:
                        # Touch logic
                        if curr_high >= ob.lower:
                            if not ob.mitigated:
                                ob.mitigated = True
                                ob.mitigated_idx = t
                                ob.freshness = False
                            ob.touch_count += 1
                            ob.number_of_reactions += 1
                        # Break check
                        if curr_close > ob.upper:
                            ob.active = False
                            ob.broken = True
                            ob.broken_idx = t

            # 5. Populate point-in-time metrics
            active_supplies = [ob for ob in bear_OBs if ob.active and ob.lower > curr_close]
            active_demands = [ob for ob in bull_OBs if ob.active and ob.upper < curr_close]

            if active_supplies:
                nz = min(active_supplies, key=lambda o: o.lower)
                ns_dist[t] = nz.lower - curr_close
                s_st[t] = nz.strength_score
                s_fresh[t] = float(nz.freshness)
                s_touches[t] = nz.touch_count
                bs_s[t] = t - nz.created_idx

            if any(curr_high >= ob.lower and curr_low <= ob.upper for ob in bear_OBs if ob.active):
                in_s[t] = 1

            if active_demands:
                nz = max(active_demands, key=lambda o: o.upper)
                nd_dist[t] = curr_close - nz.upper
                d_st[t] = nz.strength_score
                d_fresh[t] = float(nz.freshness)
                d_touches[t] = nz.touch_count
                bs_d[t] = t - nz.created_idx

            if any(curr_low <= ob.upper and curr_high >= ob.lower for ob in bull_OBs if ob.active):
                in_d[t] = 1

        df['nearest_supply_distance'] = ns_dist
        df['nearest_demand_distance'] = nd_dist
        df['inside_supply'] = in_s
        df['inside_demand'] = in_d
        df['supply_strength'] = s_st
        df['demand_strength'] = d_st
        df['supply_freshness'] = s_fresh
        df['demand_freshness'] = d_fresh
        df['supply_touch_count'] = s_touches
        df['demand_touch_count'] = d_touches
        df['bars_since_supply'] = bs_s
        df['bars_since_demand'] = bs_d

        return df.loc[:, ~df.columns.duplicated()]

    def _process_impulse(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback to previous impulsive move detection for backward compatibility."""
        df = df.copy()
        self.zones = []  # Reset stateless call

        closes = df['Close'].values
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        times = df['Datetime'].values if 'Datetime' in df.columns else [None] * len(df)

        # Pre-calculate ATR
        prev_close = df['Close'].shift(1)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - prev_close).abs(),
            (df['Low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean().values

        # Output arrays
        ns_dist = np.full(len(df), np.nan)
        nd_dist = np.full(len(df), np.nan)
        in_s = np.zeros(len(df))
        in_d = np.zeros(len(df))
        s_st = np.zeros(len(df))
        d_st = np.zeros(len(df))
        s_fresh = np.zeros(len(df))
        d_fresh = np.zeros(len(df))
        s_touches = np.zeros(len(df))
        d_touches = np.zeros(len(df))
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
                    for old_z in active_zones:
                        if old_z.type == new_zone.type:
                            if new_zone.lower >= old_z.lower and new_zone.upper <= old_z.upper:
                                new_zone.nested_inside_idx = old_z.created_idx
                                break

                    active_zones.append(new_zone)
                    self.zones.append(new_zone)

            curr_low = lows[i]
            curr_high = highs[i]
            curr_close = closes[i]
            still_active = []

            for z in active_zones:
                if i <= z.origin_candle:
                    still_active.append(z)
                    continue

                if 0 < i - z.origin_candle <= 3:
                    move_away = closes[i] - closes[i-1]
                    if (z.type == 'Demand' and move_away > 0) or (z.type == 'Supply' and move_away < 0):
                        z.strength_score += float(abs(move_away) / curr_atr)

                if z.type == 'Demand':
                    if curr_low < z.upper:
                        if not z.mitigated:
                            z.mitigated = True
                            z.mitigated_idx = i
                            z.freshness = False

                        if lows[i-1] >= z.upper:
                            z.touch_count += 1
                            z.number_of_reactions += 1
                            z.freshness_score = float(max(0.0, z.freshness_score - 0.25))
                            z.strength_score = float(max(0.0, z.strength_score - 0.5))

                            penetration = float((z.upper - curr_low) / z.width) if z.width > 0 else 0.0
                            z.average_penetration = (z.average_penetration * (z.touch_count - 1) + penetration) / z.touch_count

                            rejection = float((curr_close - curr_low) / curr_atr)
                            z.average_rejection = (z.average_rejection * (z.touch_count - 1) + rejection) / z.touch_count

                    if curr_close < z.lower:
                        z.broken = True
                        z.broken_idx = i
                        z.invalidated = True
                        z.active = False
                        z.freshness = False
                    else:
                        still_active.append(z)

                else:  # Supply
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

                    if curr_close > z.upper:
                        z.broken = True
                        z.broken_idx = i
                        z.invalidated = True
                        z.active = False
                        z.freshness = False
                    else:
                        still_active.append(z)

            active_zones = still_active

            supplies = [z for z in active_zones if z.type == 'Supply' and z.lower > curr_close]
            demands = [z for z in active_zones if z.type == 'Demand' and z.upper < curr_close]

            if supplies:
                nz = min(supplies, key=lambda z: z.lower)
                ns_dist[i] = nz.lower - curr_close
                s_st[i] = nz.strength_score
                s_fresh[i] = float(nz.freshness)
                s_touches[i] = nz.touch_count
                bs_s[i] = i - nz.created_idx
            if any(curr_high >= z.lower and curr_low <= z.upper for z in active_zones if z.type == 'Supply'):
                in_s[i] = 1

            if demands:
                nz = max(demands, key=lambda z: z.upper)
                nd_dist[i] = curr_close - nz.upper
                d_st[i] = nz.strength_score
                d_fresh[i] = float(nz.freshness)
                d_touches[i] = nz.touch_count
                bs_d[i] = i - nz.created_idx
            if any(curr_low <= z.upper and curr_high >= z.lower for z in active_zones if z.type == 'Demand'):
                in_d[i] = 1

        df['nearest_supply_distance'] = ns_dist
        df['nearest_demand_distance'] = nd_dist
        df['inside_supply'] = in_s
        df['inside_demand'] = in_d
        df['supply_strength'] = s_st
        df['demand_strength'] = d_st
        df['supply_freshness'] = s_fresh
        df['demand_freshness'] = d_fresh
        df['supply_touch_count'] = s_touches
        df['demand_touch_count'] = d_touches
        df['bars_since_supply'] = bs_s
        df['bars_since_demand'] = bs_d

        return df.loc[:, ~df.columns.duplicated()]
