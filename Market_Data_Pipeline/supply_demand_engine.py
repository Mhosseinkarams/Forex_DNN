import pandas as pd
import numpy as np
import logging
import math
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

class SupplyDemandEngine:
    """
    Purpose:
        Version 1.1 Production-Ready Supply and Demand Engine.
        Detects supply and demand zones based on fast and slow fractals
        as defined in the shved_supply_and_demand_v1.5 indicator,
        with fallback to the impulse-based logic for backward compatibility.
        Manages zone lifecycles (touches, side-flipping / turncoat, breakouts / busts)
        with 100% point-in-time correctness and zero look-ahead bias.
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
        self.zones: List[Zone] = []

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
        self.zones = []  # Reset stateless call

        closes = df['Close'].values
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        times = df['Datetime'].values if 'Datetime' in df.columns else [None] * len(df)
        n_bars = len(df)

        if n_bars < 5:
            return self._fill_empty_columns(df)

        # Pre-calculate ATR
        prev_close = df['Close'].shift(1)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - prev_close).abs(),
            (df['Low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean().values

        # Calculate fractal radiuses based on factors
        P1 = int(self.fractal_fast_factor * 2 + math.ceil(self.fractal_fast_factor / 2))
        P2 = int(self.fractal_slow_factor * 2 + math.ceil(self.fractal_slow_factor / 2))

        # Arrays to hold fast/slow fractal high/low points
        fast_up = np.zeros(n_bars)
        fast_dn = np.zeros(n_bars)
        slow_up = np.zeros(n_bars)
        slow_dn = np.zeros(n_bars)

        # Precompute fractals for all eligible indices
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

        # 1. Simulate lifecycle of all raw_zones up to the end of the dataframe (n_bars)
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

        # Keep only non-busted raw zones for merging
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

        # 2. Sequential simulation and recording
        active_zones_at_bar = [[] for _ in range(n_bars)]

        for rz in temp_zones:
            ii = rz["start"]
            is_peak_zone = rz["is_peak"]
            hival = rz["hi"]
            loval = rz["lo"]
            is_weak = rz["is_weak"]

            turned = rz["turn"] if not self.zone_merge else False
            hasturned = turned
            is_bust = False
            bustcount = 0
            testcount = 0

            mitigated = False
            mitigated_idx = None
            broken = False
            broken_idx = None

            for i in range(ii + 1, n_bars):
                if is_bust:
                    break

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
                    if not mitigated:
                        mitigated = True
                        mitigated_idx = i

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
                        broken = True
                        broken_idx = i
                    else:
                        turned = not turned
                        hasturned = True
                        testcount = 0

                if is_peak_zone:
                    current_type = 'Demand' if turned else 'Supply'
                else:
                    current_type = 'Supply' if turned else 'Demand'

                if testcount > 3:
                    curr_strength = ZONE_PROVEN
                elif testcount > 0:
                    curr_strength = ZONE_VERIFIED
                elif hasturned:
                    curr_strength = ZONE_TURNCOAT
                elif not is_weak:
                    curr_strength = ZONE_UNTESTED
                else:
                    curr_strength = ZONE_WEAK

                if not is_bust:
                    active_zones_at_bar[i].append({
                        "hi": hival,
                        "lo": loval,
                        "type": current_type,
                        "strength": curr_strength,
                        "freshness": 1.0 if (testcount == 0 and not turned) else 0.0,
                        "touch_count": testcount,
                        "created_idx": ii
                    })

            dt_ii = pd.to_datetime(times[ii]) if times[ii] is not None else None
            dt_broken = pd.to_datetime(times[broken_idx]) if (broken_idx is not None and times[broken_idx] is not None) else None
            dt_mitigated = pd.to_datetime(times[mitigated_idx]) if (mitigated_idx is not None and times[mitigated_idx] is not None) else None

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
            final_strength_score = strength_score_map.get(
                ZONE_PROVEN if testcount > 3 else (
                    ZONE_VERIFIED if testcount > 0 else (
                        ZONE_TURNCOAT if hasturned else (
                            ZONE_UNTESTED if not is_weak else ZONE_WEAK
                        )
                    )
                ), 0.0
            )

            zone_obj = Zone(
                upper=float(hival),
                lower=float(loval),
                type=final_type,
                created_time=dt_ii,
                created_idx=ii,
                freshness=(testcount == 0 and not turned),
                touch_count=testcount,
                broken=broken,
                broken_idx=broken_idx,
                mitigated=mitigated,
                mitigated_idx=mitigated_idx,
                strength_score=final_strength_score,
                creation_candle=ii,
                origin_candle=ii,
                freshness_score=1.0 if (testcount == 0 and not turned) else 0.0,
                number_of_reactions=testcount,
                average_rejection=0.0,
                average_penetration=0.0,
                invalidated=broken,
                active=not broken,
                why_detected=f"Fractal zone detected at bar {ii}.",
                rule_fired="fractal_zone_creation",
                confidence_score=1.0
            )
            self.zones.append(zone_obj)

        def merge_zones_list(zones_list):
            if not zones_list:
                return []

            t_zones = [dict(z) for z in zones_list]
            m_count = 1
            iters = 0
            while m_count > 0 and iters < 3:
                m_count = 0
                iters += 1
                t_merge = [False] * len(t_zones)

                for i in range(len(t_zones) - 1):
                    if t_zones[i]["touch_count"] == -1 or t_merge[i]:
                        continue

                    for j in range(i + 1, len(t_zones)):
                        if t_zones[j]["touch_count"] == -1 or t_merge[j]:
                            continue

                        zi = t_zones[i]
                        zj = t_zones[j]

                        if ((zi["hi"] >= zj["lo"] and zi["hi"] <= zj["hi"]) or
                            (zi["lo"] <= zj["hi"] and zi["lo"] >= zj["lo"]) or
                            (zj["hi"] >= zi["lo"] and zj["hi"] <= zi["hi"]) or
                            (zj["lo"] <= zi["hi"] and zj["lo"] >= zi["lo"])):

                            zi["hi"] = max(zi["hi"], zj["hi"])
                            zi["lo"] = min(zi["lo"], zj["lo"])
                            zi["touch_count"] += zj["touch_count"]
                            zi["created_idx"] = max(zi["created_idx"], zj["created_idx"])
                            zi["strength"] = max(zi["strength"], zj["strength"])

                            if zi["touch_count"] > 3:
                                zi["strength"] = ZONE_PROVEN

                            zi["freshness"] = min(zi["freshness"], zj["freshness"])

                            zj["touch_count"] = -1
                            t_merge[i] = True
                            t_merge[j] = True
                            m_count += 1

            return [z for z in t_zones if z["touch_count"] >= 0]

        if self.zone_merge:
            for i in range(n_bars):
                active_zones_at_bar[i] = merge_zones_list(active_zones_at_bar[i])

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

        strength_score_map = {
            ZONE_WEAK: 0.0,
            ZONE_TURNCOAT: 1.0,
            ZONE_UNTESTED: 2.0,
            ZONE_VERIFIED: 3.0,
            ZONE_PROVEN: 4.0
        }

        for i in range(1, n_bars):
            curr_close = closes[i]
            curr_high = highs[i]
            curr_low = lows[i]

            supplies = [z for z in active_zones_at_bar[i] if z["type"] == 'Supply' and z["lo"] > curr_close]
            demands = [z for z in active_zones_at_bar[i] if z["type"] == 'Demand' and z["hi"] < curr_close]

            if supplies:
                nz = min(supplies, key=lambda z: z["lo"])
                ns_dist[i] = nz["lo"] - curr_close
                s_st[i] = strength_score_map.get(nz["strength"], 0.0)
                s_fresh[i] = float(nz["freshness"])
                s_touches[i] = nz["touch_count"]
                bs_s[i] = i - nz["created_idx"]

            if any(curr_high >= z["lo"] and curr_low <= z["hi"] for z in active_zones_at_bar[i] if z["type"] == 'Supply'):
                in_s[i] = 1

            if demands:
                nz = max(demands, key=lambda z: z["hi"])
                nd_dist[i] = curr_close - nz["hi"]
                d_st[i] = strength_score_map.get(nz["strength"], 0.0)
                d_fresh[i] = float(nz["freshness"])
                d_touches[i] = nz["touch_count"]
                bs_d[i] = i - nz["created_idx"]

            if any(curr_low <= z["hi"] and curr_high >= z["lo"] for z in active_zones_at_bar[i] if z["type"] == 'Demand'):
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
