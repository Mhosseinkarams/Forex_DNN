import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any

from Market_Data_Pipeline.structure_graph import StructureLevel, BOS, CHOCH

logger = logging.getLogger('MarketStructureEngine')

class MarketStructureEngine:
    """
    Purpose:
        Version 1.0 Production-Ready Smart Money Market Structure Engine.
        Extracts objective SMC market structure from OHLC data.
        Detects Major, Minor, and Internal Swing points, Protected Levels,
        BOS, CHOCH, and nested swings with look-ahead protection.
    """
    def __init__(self, lookback: int = 3, lookback_major: int = 10, lookback_internal: int = 1):
        self.lookback = lookback  # Acts as minor lookback
        self.lookback_minor = lookback
        self.lookback_major = lookback_major
        self.lookback_internal = lookback_internal

        # State
        self.swings: List[StructureLevel] = []
        self.bos_list: List[BOS] = []
        self.choch_list: List[CHOCH] = []

        self.current_trend: int = 0  # 1: Bullish, -1: Bearish, 0: Neutral
        self.last_swing_high: Optional[StructureLevel] = None
        self.last_swing_low: Optional[StructureLevel] = None
        self.protected_high: Optional[StructureLevel] = None
        self.protected_low: Optional[StructureLevel] = None

        self.bos_count: int = 0
        self.choch_count: int = 0
        self.last_bos_idx: int = -1
        self.last_choch_idx: int = -1

    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> np.ndarray:
        """
        Calculate Average True Range (ATR) dynamically based on True Range rolling mean.
        """
        prev_close = df['Close'].shift(1)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - prev_close).abs(),
            (df['Low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(window=window).mean().values

    def _detect_swings(self, df: pd.DataFrame, atr: np.ndarray):
        """
        Detect Swing Highs and Swing Lows based on Major, Minor, and Internal lookbacks.
        Ensures nested structure tagging and full metadata tracking.
        """
        highs = df['High'].values
        lows = df['Low'].values
        times = df['Datetime'].values if 'Datetime' in df.columns else [None] * len(df)

        n_bars = len(df)

        # Helper to check peak dominance
        for i in range(1, n_bars - 1):
            dt = pd.to_datetime(times[i]) if times[i] is not None else None
            curr_atr = atr[i] if (i < len(atr) and not np.isnan(atr[i]) and atr[i] > 0) else 0.0001

            # Identify which lookback window is satisfied
            for stype, l_bk in [("Major", self.lookback_major), ("Minor", self.lookback_minor), ("Internal", self.lookback_internal)]:
                if i < l_bk or i >= n_bars - l_bk:
                    continue

                # Check Swing High
                is_sh = True
                for j in range(1, l_bk + 1):
                    if highs[i] < highs[i - j] or highs[i] < highs[i + j]:
                        is_sh = False
                        break

                if is_sh:
                    # Check duplicates
                    if not any(s.index == i and s.level_type == 'SwingHigh' and s.structure_type == stype for s in self.swings):
                        # Calculate strength score based on local variance / ATR
                        surrounding_highs = highs[max(0, i - l_bk):min(n_bars, i + l_bk + 1)]
                        mean_highs = np.mean(surrounding_highs)
                        strength_score = float((highs[i] - mean_highs) / curr_atr) if curr_atr > 0 else 1.0

                        swing = StructureLevel(
                            price=float(highs[i]),
                            index=i,
                            timestamp=dt,
                            strength=l_bk,
                            level_type='SwingHigh',
                            strength_score=strength_score,
                            confirmation_candle=i + l_bk,
                            confirmation_delay=l_bk,
                            bars_since_confirmation=-1,
                            is_valid=True,
                            broken=False,
                            reason="active",
                            structure_type=stype,
                            why_detected=f"Candle high {highs[i]:.5f} is a local peak with lookback {l_bk}.",
                            rule_fired="swing_high_detection",
                            thresholds_satisfied=[f"high_index_{i} > surrounding_{l_bk}_bars"],
                            confidence_score=min(1.0, l_bk / 10.0)
                        )
                        self.swings.append(swing)

                # Check Swing Low
                is_sl = True
                for j in range(1, l_bk + 1):
                    if lows[i] > lows[i - j] or lows[i] > lows[i + j]:
                        is_sl = False
                        break

                if is_sl:
                    if not any(s.index == i and s.level_type == 'SwingLow' and s.structure_type == stype for s in self.swings):
                        surrounding_lows = lows[max(0, i - l_bk):min(n_bars, i + l_bk + 1)]
                        mean_lows = np.mean(surrounding_lows)
                        strength_score = float((mean_lows - lows[i]) / curr_atr) if curr_atr > 0 else 1.0

                        swing = StructureLevel(
                            price=float(lows[i]),
                            index=i,
                            timestamp=dt,
                            strength=l_bk,
                            level_type='SwingLow',
                            strength_score=strength_score,
                            confirmation_candle=i + l_bk,
                            confirmation_delay=l_bk,
                            bars_since_confirmation=-1,
                            is_valid=True,
                            broken=False,
                            reason="active",
                            structure_type=stype,
                            why_detected=f"Candle low {lows[i]:.5f} is a local trough with lookback {l_bk}.",
                            rule_fired="swing_low_detection",
                            thresholds_satisfied=[f"low_index_{i} < surrounding_{l_bk}_bars"],
                            confidence_score=min(1.0, l_bk / 10.0)
                        )
                        self.swings.append(swing)

        # Build nest linkages (linking internal to minor, and minor to major)
        # Sort swings by index
        self.swings.sort(key=lambda s: (s.index, s.level_type))

        # Track last swing high and last swing low for summary compatibility
        sh_majors = [s for s in self.swings if s.level_type == 'SwingHigh' and s.structure_type == 'Major']
        sl_majors = [s for s in self.swings if s.level_type == 'SwingLow' and s.structure_type == 'Major']

        all_swing_highs = [s for s in self.swings if s.level_type == 'SwingHigh']
        all_swing_lows = [s for s in self.swings if s.level_type == 'SwingLow']

        if sh_majors:
            self.last_swing_high = sh_majors[-1]
        elif all_swing_highs:
            self.last_swing_high = all_swing_highs[-1]

        if sl_majors:
            self.last_swing_low = sl_majors[-1]
        elif all_swing_lows:
            self.last_swing_low = all_swing_lows[-1]

    def _detect_structure_breaks(self, df: pd.DataFrame, atr: np.ndarray):
        """
        Detect BOS and CHOCH precisely using confirmed swings.
        Determines and tracks protected swing levels dynamically.
        """
        if not self.swings:
            return

        closes = df['Close'].values
        volumes = df['TickVolume'].values if 'TickVolume' in df.columns else np.zeros(len(df))
        times = df['Datetime'].values if 'Datetime' in df.columns else [None] * len(df)

        # Calculate ATR
        atr_period = 14
        prev_close = df['Close'].shift(1)
        tr = pd.concat([df['High'] - df['Low'], (df['High'] - prev_close).abs(), (df['Low'] - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean().values

        broken_swings = set()

        for i in range(1, len(df)):
            curr_atr = atr[i] if (not np.isnan(atr[i]) and atr[i] > 0) else 0.0001
            dt_i = pd.to_datetime(times[i]) if times[i] is not None else None

            # Get swings confirmed by index i (index + confirmation_delay <= i)
            confirmed_swings = [s for s in self.swings if s.index + s.confirmation_delay <= i]
            if not confirmed_swings:
                continue

            # We use Major/Minor swings to define the market structure boundary
            latest_highs = [s for s in confirmed_swings if s.level_type == 'SwingHigh' and s.structure_type in ('Major', 'Minor')]
            latest_lows = [s for s in confirmed_swings if s.level_type == 'SwingLow' and s.structure_type in ('Major', 'Minor')]
            if not latest_highs or not latest_lows:
                continue

            curr_high_swing = latest_highs[-1]
            curr_low_swing = latest_lows[-1]

            # Removed redundant O(N) loop here. bars_since_confirmation will be vectorized/calculated below.

            # Check Bullish BOS / Bullish CHOCH
            if closes[i] > curr_high_swing.price and curr_high_swing.index not in broken_swings:
                broken_swings.add(curr_high_swing.index)
                curr_high_swing.broken = True
                curr_high_swing.reason = f"Broken by Close at bar {i}"

                distance = float(closes[i] - curr_high_swing.price)
                norm_dist = distance / curr_atr
                vol = float(volumes[i])

                # Impulse size from previous swing low
                prev_lows_before = [s for s in confirmed_swings if s.level_type == 'SwingLow' and s.index < curr_high_swing.index]
                impulse_base = prev_lows_before[-1].price if prev_lows_before else curr_high_swing.price - curr_atr
                imp_size = float(closes[i] - impulse_base)
                norm_imp = imp_size / curr_atr

                # Break strength is ratio of close-open or break candle body to ATR
                break_strength_val = float(abs(closes[i] - df['Open'].values[i]) / curr_atr)

                if self.current_trend >= 0:
                    # Bullish BOS
                    bos_obj = BOS(
                        index=i,
                        direction=1,
                        broken_level=curr_high_swing.price,
                        timestamp=dt_i,
                        strength=curr_high_swing.strength,
                        distance=distance,
                        atr_normalized_distance=norm_dist,
                        break_candle=i,
                        impulse_size=imp_size,
                        atr_normalized_impulse=norm_imp,
                        volume=vol,
                        break_strength=break_strength_val,
                        why_detected=f"Close {closes[i]:.5f} broke SwingHigh {curr_high_swing.price:.5f} in trend direction.",
                        rule_fired="bullish_bos_break",
                        thresholds_satisfied=[f"close_{closes[i]:.5f} > swing_high_{curr_high_swing.price:.5f}"],
                        confidence_score=min(1.0, break_strength_val / 2.0)
                    )
                    self.bos_list.append(bos_obj)
                    self.bos_count += 1
                    self.last_bos_idx = i
                else:
                    # Bullish CHOCH
                    choch_obj = CHOCH(
                        index=i,
                        previous_trend=self.current_trend,
                        new_trend=1,
                        timestamp=dt_i,
                        price=closes[i],
                        strength=curr_high_swing.strength,
                        confirmation_score=float(min(1.0, break_strength_val / 1.5)),
                        why_detected=f"Close {closes[i]:.5f} broke SwingHigh {curr_high_swing.price:.5f} against bearish trend.",
                        rule_fired="bullish_choch_break",
                        thresholds_satisfied=[f"close_{closes[i]:.5f} > swing_high_{curr_high_swing.price:.5f}"],
                        confidence_score=min(1.0, break_strength_val / 1.5)
                    )
                    self.choch_list.append(choch_obj)
                    self.choch_count += 1
                    self.last_choch_idx = i

                self.current_trend = 1
                self.protected_low = curr_low_swing
                if self.protected_low:
                    self.protected_low.level_type = "ProtectedLow"

            # Check Bearish BOS / Bearish CHOCH
            elif closes[i] < curr_low_swing.price and curr_low_swing.index not in broken_swings:
                broken_swings.add(curr_low_swing.index)
                curr_low_swing.broken = True
                curr_low_swing.reason = f"Broken by Close at bar {i}"

                distance = float(curr_low_swing.price - closes[i])
                norm_dist = distance / curr_atr
                vol = float(volumes[i])

                # Impulse size from previous swing high
                prev_highs_before = [s for s in confirmed_swings if s.level_type == 'SwingHigh' and s.index < curr_low_swing.index]
                impulse_base = prev_highs_before[-1].price if prev_highs_before else curr_low_swing.price + curr_atr
                imp_size = float(impulse_base - closes[i])
                norm_imp = imp_size / curr_atr

                break_strength_val = float(abs(closes[i] - df['Open'].values[i]) / curr_atr)

                if self.current_trend <= 0:
                    # Bearish BOS
                    bos_obj = BOS(
                        index=i,
                        direction=-1,
                        broken_level=curr_low_swing.price,
                        timestamp=dt_i,
                        strength=curr_low_swing.strength,
                        distance=distance,
                        atr_normalized_distance=norm_dist,
                        break_candle=i,
                        impulse_size=imp_size,
                        atr_normalized_impulse=norm_imp,
                        volume=vol,
                        break_strength=break_strength_val,
                        why_detected=f"Close {closes[i]:.5f} broke SwingLow {curr_low_swing.price:.5f} in trend direction.",
                        rule_fired="bearish_bos_break",
                        thresholds_satisfied=[f"close_{closes[i]:.5f} < swing_low_{curr_low_swing.price:.5f}"],
                        confidence_score=min(1.0, break_strength_val / 2.0)
                    )
                    self.bos_list.append(bos_obj)
                    self.bos_count += 1
                    self.last_bos_idx = i
                else:
                    # Bearish CHOCH
                    choch_obj = CHOCH(
                        index=i,
                        previous_trend=self.current_trend,
                        new_trend=-1,
                        timestamp=dt_i,
                        price=closes[i],
                        strength=curr_low_swing.strength,
                        confirmation_score=float(min(1.0, break_strength_val / 1.5)),
                        why_detected=f"Close {closes[i]:.5f} broke SwingLow {curr_low_swing.price:.5f} against bullish trend.",
                        rule_fired="bearish_choch_break",
                        thresholds_satisfied=[f"close_{closes[i]:.5f} < swing_low_{curr_low_swing.price:.5f}"],
                        confidence_score=min(1.0, break_strength_val / 1.5)
                    )
                    self.choch_list.append(choch_obj)
                    self.choch_count += 1
                    self.last_choch_idx = i

                self.current_trend = -1
                self.protected_high = curr_high_swing
                if self.protected_high:
                    self.protected_high.level_type = "ProtectedHigh"

    def get_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        bars_since_bos = len(df) - 1 - self.last_bos_idx if self.last_bos_idx != -1 else -1
        bars_since_choch = len(df) - 1 - self.last_choch_idx if self.last_choch_idx != -1 else -1

        return {
            "trend": self.current_trend,
            "structure_state": "Trending" if self.current_trend != 0 else "Neutral",
            "bos_count": self.bos_count,
            "choch_count": self.choch_count,
            "bars_since_bos": bars_since_bos,
            "bars_since_choch": bars_since_choch,
            "last_bos_direction": self.bos_list[-1].direction if self.bos_list else 0,
            "last_choch_direction": self.choch_list[-1].new_trend if self.choch_list else 0,
            "swing_high": self.last_swing_high.price if self.last_swing_high else None,
            "swing_low": self.last_swing_low.price if self.last_swing_low else None,
        }

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Reset state
        self.swings = []
        self.bos_list = []
        self.choch_list = []
        self.bos_count = 0
        self.choch_count = 0
        self.current_trend = 0
        self.last_bos_idx = -1
        self.last_choch_idx = -1
        self.last_swing_high = None
        self.last_swing_low = None
        self.protected_high = None
        self.protected_low = None

        # Calculate ATR once for reuse
        atr = self._calculate_atr(df, window=14)

        self._detect_swings(df, atr)
        self._detect_structure_breaks(df, atr)

        # Vectorized / single-pass calculation of bars_since_confirmation for all swings at end of process
        n_bars = len(df)
        for s in self.swings:
            confirmation_idx = s.index + s.confirmation_delay
            if confirmation_idx < n_bars:
                s.bars_since_confirmation = n_bars - 1 - confirmation_idx
            else:
                s.bars_since_confirmation = -1

        # Columns mapping for DataFrame enrichment (SMC outputs)
        trend_arr = np.zeros(len(df))
        bos_arr = np.zeros(len(df))
        choch_arr = np.zeros(len(df))
        bos_cnt_arr = np.zeros(len(df))
        choch_cnt_arr = np.zeros(len(df))
        last_bos_dir_arr = np.zeros(len(df))
        last_choch_dir_arr = np.zeros(len(df))
        sh_arr = np.full(len(df), np.nan)
        sl_arr = np.full(len(df), np.nan)

        # Swings
        for s in self.swings:
            if s.level_type == 'SwingHigh':
                sh_arr[s.index + s.confirmation_delay:] = s.price
            else:
                sl_arr[s.index + s.confirmation_delay:] = s.price

        # BOS
        curr_bos_cnt = 0
        last_bos_dir = 0
        for b in self.bos_list:
            bos_arr[b.index] = b.direction
            curr_bos_cnt += 1
            last_bos_dir = b.direction
            bos_cnt_arr[b.index:] = curr_bos_cnt
            last_bos_dir_arr[b.index:] = last_bos_dir

        # CHOCH
        curr_choch_cnt = 0
        last_choch_dir = 0
        for c in self.choch_list:
            choch_arr[c.index] = c.new_trend
            curr_choch_cnt += 1
            last_choch_dir = c.new_trend
            choch_cnt_arr[c.index:] = curr_choch_cnt
            last_choch_dir_arr[c.index:] = last_choch_dir

        # Trend and Bars Since
        curr_tr = 0
        last_b = -1
        last_c = -1
        bs_b = np.full(len(df), -1)
        bs_c = np.full(len(df), -1)

        for i in range(len(df)):
            if bos_arr[i] != 0:
                curr_tr = bos_arr[i]
                last_b = i
            if choch_arr[i] != 0:
                curr_tr = choch_arr[i]
                last_c = i
            trend_arr[i] = curr_tr
            if last_b != -1: bs_b[i] = i - last_b
            if last_c != -1: bs_c[i] = i - last_c

        df['trend'] = trend_arr
        df['bos'] = bos_arr
        df['choch'] = choch_arr
        df['bos_count'] = bos_cnt_arr
        df['choch_count'] = choch_cnt_arr
        df['bars_since_bos'] = bs_b
        df['bars_since_choch'] = bs_c
        df['last_bos_direction'] = last_bos_dir_arr
        df['last_choch_direction'] = last_choch_dir_arr
        df['swing_high'] = sh_arr
        df['swing_low'] = sl_arr

        return df.loc[:, ~df.columns.duplicated()]
