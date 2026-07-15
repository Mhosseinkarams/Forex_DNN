import pandas as pd
import numpy as np
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from Market_Data_Pipeline.structure_graph import Zone

logger = logging.getLogger('SupplyDemandEngine')

# Constants matching MQL5 strength rankings
ZONE_WEAK = 0
ZONE_TURNCOAT = 1
ZONE_UNTESTED = 2
ZONE_VERIFIED = 3
ZONE_PROVEN = 4

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
    SMC Supply & Demand Engine that supports BOTH:
    1. shved_supply_and_demand_v1.5.mq5 (Fast/Slow Fractals, touch validation, turncoat, zone merging).
    2. SmartMoneyConcepts.mq5 Order Blocks (breakout-based zones), Fair Value Gaps (FVG), and EQH/EQL patterns.
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
        atr = tr.rolling(window=self.atr_period).mean().fillna(0.0010).values

        # -------------------------------------------------------------
        # PART 1: SHVED SUPPLY AND DEMAND (Fractal-based detection)
        # -------------------------------------------------------------
        # Calculate fractal radiuses based on factors
        P1 = int(self.fractal_fast_factor * 2 + math.ceil(self.fractal_fast_factor / 2))
        P2 = int(self.fractal_slow_factor * 2 + math.ceil(self.fractal_slow_factor / 2))

        # Arrays to hold fast/slow fractal high/low points
        fast_up = np.zeros(n_bars)
        fast_dn = np.zeros(n_bars)
        slow_up = np.zeros(n_bars)
        slow_dn = np.zeros(n_bars)

        # Precompute Shved fractals
        for idx in range(P1, n_bars - P1):
            is_peak = True
            for j in range(1, P1 + 1):
                if highs[idx - j] > highs[idx] or highs[idx + j] >= highs[idx]:
                    is_peak = False
                    break
            if is_peak:
                fast_up[idx] = highs[idx]

            is_trough = True
            for j in range(1, P1 + 1):
                if lows[idx - j] < lows[idx] or lows[idx + j] <= lows[idx]:
                    is_trough = False
                    break
            if is_trough:
                fast_dn[idx] = lows[idx]

        for idx in range(P2, n_bars - P2):
            is_peak = True
            for j in range(1, P2 + 1):
                if highs[idx - j] > highs[idx] or highs[idx + j] >= highs[idx]:
                    is_peak = False
                    break
            if is_peak:
                slow_up[idx] = highs[idx]

            is_trough = True
            for j in range(1, P2 + 1):
                if lows[idx - j] < lows[idx] or lows[idx + j] <= lows[idx]:
                    is_trough = False
                    break
            if is_trough:
                slow_dn[idx] = lows[idx]

        start_idx = max(0, n_bars - self.back_limit)
        raw_zones = []

        # Find Shved Zones
        for ii in range(start_idx, n_bars - 5):
            atr_ii = atr[ii] if (ii < len(atr) and not np.isnan(atr[ii]) and atr[ii] > 0) else 0.0001
            fu = (atr_ii / 2) * self.zone_fuzzfactor

            if fast_up[ii] > 0.001:
                is_weak = True
                if slow_up[ii] > 0.001:
                    is_weak = False

                hival = highs[ii]
                if self.zone_extend:
                    hival += fu

                loval = max(min(closes[ii], highs[ii] - fu), highs[ii] - fu * 2)

                raw_zones.append({
                    "start": ii,
                    "is_peak": True,
                    "hi": hival,
                    "lo": loval,
                    "is_weak": is_weak,
                    "hits": 0,
                    "turn": False
                })

            elif fast_dn[ii] > 0.001:
                is_weak = True
                if slow_dn[ii] > 0.001:
                    is_weak = False

                loval = lows[ii]
                if self.zone_extend:
                    loval -= fu

                hival = min(max(closes[ii], lows[ii] + fu), lows[ii] + fu * 2)

                raw_zones.append({
                    "start": ii,
                    "is_peak": False,
                    "hi": hival,
                    "lo": loval,
                    "is_weak": is_weak,
                    "hits": 0,
                    "turn": False
                })

        # Shved Touch & Mitigation Simulation
        for rz in raw_zones:
            ii = rz["start"]
            is_peak_zone = rz["is_peak"]
            hival = rz["hi"]
            loval = rz["lo"]
            is_weak = rz["is_weak"]

            turned = False
            hasturned = False
            is_bust = False
            bustcount = 0
            testcount = 0

            for i in range(ii + 1, n_bars):
                is_touch = False
                if is_peak_zone:
                    if not turned:
                        if fast_up[i] >= loval and fast_up[i] <= hival:
                            is_touch = True
                    else:
                        if fast_dn[i] <= hival and fast_dn[i] >= loval:
                            is_touch = True
                else:
                    if turned:
                        if fast_up[i] >= loval and fast_up[i] <= hival:
                            is_touch = True
                    else:
                        if fast_dn[i] <= hival and fast_dn[i] >= loval:
                            is_touch = True

                if is_touch:
                    touch_ok = True
                    for j in range(max(ii + 1, i - 10), i):
                        if is_peak_zone:
                            if not turned:
                                if fast_up[j] >= loval and fast_up[j] <= hival:
                                    touch_ok = False
                                    break
                            else:
                                if fast_dn[j] <= hival and fast_dn[j] >= loval:
                                    touch_ok = False
                                    break
                        else:
                            if turned:
                                if fast_up[j] >= loval and fast_up[j] <= hival:
                                    touch_ok = False
                                    break
                            else:
                                if fast_dn[j] <= hival and fast_dn[j] >= loval:
                                    touch_ok = False
                                    break

                    if touch_ok:
                        bustcount = 0
                        testcount += 1

                is_breaker = False
                if is_peak_zone:
                    if not turned:
                        if highs[i] > hival:
                            is_breaker = True
                    else:
                        if lows[i] < loval:
                            is_breaker = True
                else:
                    if turned:
                        if highs[i] > hival:
                            is_breaker = True
                    else:
                        if lows[i] < loval:
                            is_breaker = True

                if is_breaker:
                    bustcount += 1
                    if bustcount > 1 or is_weak:
                        is_bust = True
                        break

                    turned = not turned
                    hasturned = True
                    testcount = 0

            if is_bust:
                rz["hits"] = -1
            else:
                rz["hits"] = testcount
                rz["turn"] = hasturned

                if testcount > 3:
                    rz["strength"] = ZONE_PROVEN
                elif testcount > 0:
                    rz["strength"] = ZONE_VERIFIED
                elif hasturned:
                    rz["strength"] = ZONE_TURNCOAT
                elif not is_weak:
                    rz["strength"] = ZONE_UNTESTED
                else:
                    rz["strength"] = ZONE_WEAK

        # Shved Merge overlap
        temp_zones = [rz for rz in raw_zones if rz["hits"] >= 0]
        if self.zone_merge:
            merge_count = 1
            iterations = 0
            while merge_count > 0 and iterations < 3:
                merge_count = 0
                iterations += 1

                temp_merge = [False] * len(temp_zones)

                for i in range(len(temp_zones) - 1):
                    if temp_zones[i]["hits"] == -1 or temp_merge[i]:
                        continue

                    for j in range(i + 1, len(temp_zones)):
                        if temp_zones[j]["hits"] == -1 or temp_merge[j]:
                            continue

                        zi = temp_zones[i]
                        zj = temp_zones[j]

                        if ((zi["hi"] >= zj["lo"] and zi["hi"] <= zj["hi"]) or
                            (zi["lo"] <= zj["hi"] and zi["lo"] >= zj["lo"]) or
                            (zj["hi"] >= zi["lo"] and zj["hi"] <= zi["hi"]) or
                            (zj["lo"] <= zi["hi"] and zj["lo"] >= zi["lo"])):

                            zi["hi"] = max(zi["hi"], zj["hi"])
                            zi["lo"] = min(zi["lo"], zj["lo"])
                            zi["hits"] += zj["hits"]
                            zi["start"] = max(zi["start"], zj["start"])
                            zi["strength"] = max(zi["strength"], zj["strength"])

                            if zi["hits"] > 3:
                                zi["strength"] = ZONE_PROVEN

                            if zi["hits"] == 0 and not zi["turn"]:
                                zi["hits"] = 1
                                if zi["strength"] < ZONE_VERIFIED:
                                    zi["strength"] = ZONE_VERIFIED

                            if not zi["turn"] or not zj["turn"]:
                                zi["turn"] = False

                            if zi["turn"]:
                                zi["hits"] = 0

                            zj["hits"] = -1
                            temp_merge[i] = True
                            temp_merge[j] = True
                            merge_count += 1

            temp_zones = [z for z in temp_zones if z["hits"] >= 0]

        # Convert Shved Zones to official Zone models
        for rz in temp_zones:
            ii = rz["start"]
            hival = rz["hi"]
            loval = rz["lo"]

            # Determine type Support/Resistance
            ref_close = closes[n_bars - 5] if n_bars >= 5 else closes[-1]
            if hival < ref_close:
                final_type = 'Demand'
            elif loval > ref_close:
                final_type = 'Supply'
            else:
                final_type = 'Demand'
                for j in range(n_bars - 5, start_idx - 1, -1):
                    if j < 0 or j >= n_bars:
                        continue
                    if closes[j] < loval:
                        final_type = 'Supply'
                        break
                    elif closes[j] > hival:
                        final_type = 'Demand'
                        break

            strength_score_map = {
                ZONE_WEAK: 0.0,
                ZONE_TURNCOAT: 1.0,
                ZONE_UNTESTED: 2.0,
                ZONE_VERIFIED: 3.0,
                ZONE_PROVEN: 4.0
            }
            final_strength_score = strength_score_map.get(rz["strength"], 2.0)

            # Build Zone
            zone_obj = Zone(
                upper=float(hival),
                lower=float(loval),
                type=final_type,
                created_time=times[ii] if isinstance(times[ii], datetime) else (pd.to_datetime(times[ii]) if times[ii] is not None else None),
                created_idx=ii,
                freshness=(rz["hits"] == 0 and not rz["turn"]),
                touch_count=rz["hits"],
                broken=False,
                active=True,
                strength_score=final_strength_score,
                creation_candle=ii,
                origin_candle=ii
            )
            self.zones.append(zone_obj)

        # -------------------------------------------------------------
        # PART 2: SMC ORDER BLOCKS OVERLAY (from SmartMoneyConcepts.mq5)
        # -------------------------------------------------------------
        lookback = 10  # SMC Swing Length
        inp_max_ob = 5

        # Precompute SMC pivots
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

        # EQH pivots
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

        g_phLevel = 0.0
        g_plLevel = 0.0
        last_eqh_level = 0.0
        last_eql_level = 0.0

        # OB lists
        bull_OBs: List[Zone] = []
        bear_OBs: List[Zone] = []

        for t in range(n_bars):
            idx = t - lookback
            if idx >= lookback:
                # Pivot High structure break
                if piv_highs[idx]:
                    pivPrice = highs[idx]
                    if g_phLevel > 0 and pivPrice > g_phLevel:
                        # Bullish OB (Demand Zone)
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

                            duplicate = False
                            for ob in bull_OBs:
                                if ob.active and abs(ob.upper - obTop) < atr[t] * 0.3:
                                    duplicate = True
                                    break

                            if obTop - obBottom >= 1e-6 and not duplicate:
                                active_bull = [o for o in bull_OBs if o.active]
                                if len(active_bull) >= inp_max_ob:
                                    active_bull[0].active = False
                                    active_bull[0].broken = True
                                    active_bull[0].broken_idx = t

                                new_ob = Zone(
                                    upper=float(obTop),
                                    lower=float(obBottom),
                                    type='Demand',
                                    created_idx=found,
                                    created_time=times[found],
                                    freshness=True,
                                    touch_count=0,
                                    strength_score=3.0,  # Verified Order Block
                                    creation_candle=found,
                                    origin_candle=idx,
                                    active=True,
                                    why_detected="SMC Order Block breakout"
                                )
                                bull_OBs.append(new_ob)
                                self.zones.append(new_ob)

                    g_phLevel = pivPrice

                # Pivot Low structure break
                if piv_lows[idx]:
                    pivPrice = lows[idx]
                    if g_plLevel > 0 and pivPrice < g_plLevel:
                        # Bearish OB (Supply Zone)
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

                                new_ob = Zone(
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
                                    active=True,
                                    why_detected="SMC Order Block breakdown"
                                )
                                bear_OBs.append(new_ob)
                                self.zones.append(new_ob)

                    g_plLevel = pivPrice

            # EQH / EQL detection (Pivot 5)
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

            # FVG detection
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

        # -------------------------------------------------------------
        # PART 3: JOINT SIMULATION (Mitigations & distance generation)
        # -------------------------------------------------------------
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

        for t in range(n_bars):
            curr_close = closes[t]
            curr_low = lows[t]
            curr_high = highs[t]

            # FVG mitigation
            for f in self.fvgs:
                if f.active and f.created_idx < t:
                    if f.bias == 1:
                        if curr_low <= f.upper and curr_low >= f.lower:
                            f.active = False
                            f.mitigated_idx = t
                    else:
                        if curr_high >= f.lower and curr_high <= f.upper:
                            f.active = False
                            f.mitigated_idx = t

            # Joint zone mitigation & retests
            for z in self.zones:
                if z.active and z.created_idx < t:
                    if z.type == 'Demand':
                        if curr_low <= z.upper:
                            if not z.mitigated:
                                z.mitigated = True
                                z.mitigated_idx = t
                                z.freshness = False
                            z.touch_count += 1
                            z.number_of_reactions += 1
                        if curr_close < z.lower:
                            z.active = False
                            z.broken = True
                            z.broken_idx = t
                    else:
                        if curr_high >= z.lower:
                            if not z.mitigated:
                                z.mitigated = True
                                z.mitigated_idx = t
                                z.freshness = False
                            z.touch_count += 1
                            z.number_of_reactions += 1
                        if curr_close > z.upper:
                            z.active = False
                            z.broken = True
                            z.broken_idx = t

            # Populate metrics for current bar t
            active_supplies = [z for z in self.zones if z.active and z.created_idx < t and z.type == 'Supply' and z.lower > curr_close]
            active_demands = [z for z in self.zones if z.active and z.created_idx < t and z.type == 'Demand' and z.upper < curr_close]

            if active_supplies:
                nz = min(active_supplies, key=lambda o: o.lower)
                ns_dist[t] = nz.lower - curr_close
                s_st[t] = nz.strength_score
                s_fresh[t] = float(nz.freshness)
                s_touches[t] = nz.touch_count
                bs_s[t] = t - nz.created_idx

            if any(curr_high >= ob.lower and curr_low <= ob.upper for ob in self.zones if ob.active and ob.created_idx < t and ob.type == 'Supply'):
                in_s[t] = 1

            if active_demands:
                nz = max(active_demands, key=lambda o: o.upper)
                nd_dist[t] = curr_close - nz.upper
                d_st[t] = nz.strength_score
                d_fresh[t] = float(nz.freshness)
                d_touches[t] = nz.touch_count
                bs_d[t] = t - nz.created_idx

            if any(curr_low <= ob.upper and curr_high >= ob.lower for ob in self.zones if ob.active and ob.created_idx < t and ob.type == 'Demand'):
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
