import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from Market_Data_Pipeline.structure_graph import StructureLevel, BOS, CHOCH

logger = logging.getLogger('MarketStructureEngine')

class MarketStructureEngine:
    """
    SMC Market Structure Engine based on SmartMoneyConcepts.mq5.
    Extracts objective Smart Money market structure from OHLC data.
    """
    def __init__(self, lookback: int = 3, lookback_major: int = 10, lookback_internal: int = 1):
        self.lookback = lookback  # Acts as the Swing Length (InpSwingLength)
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
        prev_close = df['Close'].shift(1)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - prev_close).abs(),
            (df['Low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(window=window).mean().fillna(0.0010).values

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n_bars = len(df)

        # Reset State
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

        # Prepare outputs
        trend_arr = np.zeros(n_bars)
        bos_arr = np.zeros(n_bars)
        choch_arr = np.zeros(n_bars)
        bos_cnt_arr = np.zeros(n_bars)
        choch_cnt_arr = np.zeros(n_bars)
        last_bos_dir_arr = np.zeros(n_bars)
        last_choch_dir_arr = np.zeros(n_bars)
        sh_arr = np.full(n_bars, np.nan)
        sl_arr = np.full(n_bars, np.nan)

        if n_bars < self.lookback * 2 + 1:
            df['trend'] = trend_arr
            df['bos'] = bos_arr
            df['choch'] = choch_arr
            df['bos_count'] = bos_cnt_arr
            df['choch_count'] = choch_cnt_arr
            df['bars_since_bos'] = np.full(n_bars, -1)
            df['bars_since_choch'] = np.full(n_bars, -1)
            df['last_bos_direction'] = last_bos_dir_arr
            df['last_choch_direction'] = last_choch_dir_arr
            df['swing_high'] = sh_arr
            df['swing_low'] = sl_arr
            return df

        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        opens = df['Open'].values
        times = df['Datetime'].values if 'Datetime' in df.columns else [None] * n_bars

        # Precompute Pivot Highs & Pivot Lows
        piv_highs = np.zeros(n_bars, dtype=bool)
        piv_lows = np.zeros(n_bars, dtype=bool)

        for i in range(self.lookback, n_bars - self.lookback):
            p_high = highs[i]
            is_ph = True
            for k in range(1, self.lookback + 1):
                if highs[i - k] >= p_high or highs[i + k] >= p_high:
                    is_ph = False
                    break
            piv_highs[i] = is_ph

            p_low = lows[i]
            is_pl = True
            for k in range(1, self.lookback + 1):
                if lows[i - k] <= p_low or lows[i + k] <= p_low:
                    is_pl = False
                    break
            piv_lows[i] = is_pl

        # Running SMC variables
        g_phLevel = 0.0
        g_plLevel = 0.0
        g_phIdx = -1
        g_plIdx = -1
        g_trend = 0  # 1: Bull, -1: Bear, 0: Neutral

        atr_vals = self._calculate_atr(df, window=14)

        # Sequential point-in-time calculation
        for t in range(n_bars):
            idx = t - self.lookback
            if idx >= self.lookback:
                # Process Pivot High Confirmation
                if piv_highs[idx]:
                    pivPrice = highs[idx]
                    pivTime = times[idx] if isinstance(times[idx], datetime) else (pd.to_datetime(times[idx]) if times[idx] is not None else None)

                    # Create StructureLevel Swing High
                    swing = StructureLevel(
                        price=float(pivPrice),
                        index=idx,
                        timestamp=pivTime,
                        strength=self.lookback,
                        level_type='SwingHigh',
                        strength_score=1.0,
                        confirmation_candle=t,
                        confirmation_delay=self.lookback,
                        bars_since_confirmation=t - idx,
                        is_valid=True,
                        broken=False,
                        structure_type='Minor' if self.lookback < 8 else 'Major'
                    )
                    self.swings.append(swing)
                    self.last_swing_high = swing

                    # Establish newly confirmed swing high level as active
                    g_phLevel = pivPrice
                    g_phIdx = idx

                # Process Pivot Low Confirmation
                if piv_lows[idx]:
                    pivPrice = lows[idx]
                    pivTime = times[idx] if isinstance(times[idx], datetime) else (pd.to_datetime(times[idx]) if times[idx] is not None else None)

                    # Create StructureLevel Swing Low
                    swing = StructureLevel(
                        price=float(pivPrice),
                        index=idx,
                        timestamp=pivTime,
                        strength=self.lookback,
                        level_type='SwingLow',
                        strength_score=1.0,
                        confirmation_candle=t,
                        confirmation_delay=self.lookback,
                        bars_since_confirmation=t - idx,
                        is_valid=True,
                        broken=False,
                        structure_type='Minor' if self.lookback < 8 else 'Major'
                    )
                    self.swings.append(swing)
                    self.last_swing_low = swing

                    # Establish newly confirmed swing low level as active
                    g_plLevel = pivPrice
                    g_plIdx = idx

            # Sequential Point-in-time Breakout / Breakdown Evaluation on every bar t
            if g_phLevel > 0 and highs[t] > g_phLevel:
                # Bullish Breakout
                is_choch = (g_trend == -1)
                tag = "CHoCH" if is_choch else "BOS"
                g_trend = 1

                distance = float(highs[t] - g_phLevel)
                norm_dist = float(distance / atr_vals[t]) if atr_vals[t] > 0 else 0.0

                if not is_choch:
                    bos_obj = BOS(
                        index=t,
                        direction=1,
                        broken_level=float(g_phLevel),
                        timestamp=times[t] if times[t] is not None else None,
                        strength=self.lookback,
                        distance=distance,
                        atr_normalized_distance=norm_dist,
                        break_candle=t,
                        impulse_size=distance,
                        atr_normalized_impulse=norm_dist,
                        break_strength=1.0
                    )
                    self.bos_list.append(bos_obj)
                    self.bos_count += 1
                    self.last_bos_idx = t
                    bos_arr[t] = 1
                else:
                    choch_obj = CHOCH(
                        index=t,
                        previous_trend=-1,
                        new_trend=1,
                        timestamp=times[t] if times[t] is not None else None,
                        price=float(highs[t]),
                        strength=self.lookback,
                        confirmation_score=1.0
                    )
                    self.choch_list.append(choch_obj)
                    self.choch_count += 1
                    self.last_choch_idx = t
                    choch_arr[t] = 1

                # Deactivate broken level (mitigation)
                g_phLevel = 0.0

                # Mark swing high level as broken
                if self.last_swing_high:
                    self.last_swing_high.broken = True

                # Protected Levels updates
                if self.last_swing_low:
                    self.protected_low = self.last_swing_low
                    self.protected_low.level_type = "ProtectedLow"

            if g_plLevel > 0 and lows[t] < g_plLevel:
                # Bearish Breakdown
                is_choch = (g_trend == 1)
                tag = "CHoCH" if is_choch else "BOS"
                g_trend = -1

                distance = float(g_plLevel - lows[t])
                norm_dist = float(distance / atr_vals[t]) if atr_vals[t] > 0 else 0.0

                if not is_choch:
                    bos_obj = BOS(
                        index=t,
                        direction=-1,
                        broken_level=float(g_plLevel),
                        timestamp=times[t] if times[t] is not None else None,
                        strength=self.lookback,
                        distance=distance,
                        atr_normalized_distance=norm_dist,
                        break_candle=t,
                        impulse_size=distance,
                        atr_normalized_impulse=norm_dist,
                        break_strength=1.0
                    )
                    self.bos_list.append(bos_obj)
                    self.bos_count += 1
                    self.last_bos_idx = t
                    bos_arr[t] = -1
                else:
                    choch_obj = CHOCH(
                        index=t,
                        previous_trend=1,
                        new_trend=-1,
                        timestamp=times[t] if times[t] is not None else None,
                        price=float(lows[t]),
                        strength=self.lookback,
                        confirmation_score=1.0
                    )
                    self.choch_list.append(choch_obj)
                    self.choch_count += 1
                    self.last_choch_idx = t
                    choch_arr[t] = -1

                # Deactivate broken level (mitigation)
                g_plLevel = 0.0

                # Mark swing low level as broken
                if self.last_swing_low:
                    self.last_swing_low.broken = True

                # Protected Levels updates
                if self.last_swing_high:
                    self.protected_high = self.last_swing_high
                    self.protected_high.level_type = "ProtectedHigh"

            # Populate point-in-time state arrays
            trend_arr[t] = g_trend
            bos_cnt_arr[t] = self.bos_count
            choch_cnt_arr[t] = self.choch_count

            if self.bos_list:
                last_bos_dir_arr[t] = self.bos_list[-1].direction
            if self.choch_list:
                last_choch_dir_arr[t] = self.choch_list[-1].new_trend

            # Update confirmed swing high/low prices on historical array
            if g_phIdx != -1:
                sh_arr[t] = highs[g_phIdx]
            if g_plIdx != -1:
                sl_arr[t] = lows[g_plIdx]

        self.current_trend = int(g_trend)

        # Compute bars since
        bs_b = np.full(n_bars, -1)
        bs_c = np.full(n_bars, -1)
        last_b = -1
        last_c = -1
        for i in range(n_bars):
            if bos_arr[i] != 0:
                last_b = i
            if choch_arr[i] != 0:
                last_c = i
            if last_b != -1:
                bs_b[i] = i - last_b
            if last_c != -1:
                bs_c[i] = i - last_c

        # Vectorized bars since confirmation for swings list
        for s in self.swings:
            confirmation_idx = s.index + s.confirmation_delay
            if confirmation_idx < n_bars:
                s.bars_since_confirmation = n_bars - 1 - confirmation_idx
            else:
                s.bars_since_confirmation = -1

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

        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        return df

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
