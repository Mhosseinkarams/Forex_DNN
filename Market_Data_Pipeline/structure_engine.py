import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from Market_Data_Pipeline.structure_graph import StructureLevel, BOS, CHOCH

# Configure logger
logger = logging.getLogger('MarketStructureEngine')

class MarketStructureEngine:
    """
    Purpose:
        Extract objective Smart Money market structure from OHLC data.
        Detects Swing Highs/Lows, BOS (Break of Structure), and CHOCH (Change of Character).

    No trading decisions.
    Pure deterministic calculations.
    """
    def __init__(self, lookback: int = 3):
        self.lookback = lookback
        self.swings: List[StructureLevel] = []
        self.bos_list: List[BOS] = []
        self.choch_list: List[CHOCH] = []

        # Internal state
        self.current_trend: int = 0  # 1: Bullish, -1: Bearish, 0: Neutral
        self.last_swing_high: Optional[StructureLevel] = None
        self.last_swing_low: Optional[StructureLevel] = None
        self.protected_high: Optional[StructureLevel] = None
        self.protected_low: Optional[StructureLevel] = None

        self.bos_count: int = 0
        self.choch_count: int = 0
        self.last_bos_idx: int = -1
        self.last_choch_idx: int = -1

    def _detect_swings(self, df: pd.DataFrame):
        """
        Detect Swing Highs and Swing Lows based on configurable lookback.
        """
        highs = df['High'].values
        lows = df['Low'].values
        times = df['Datetime'].values if 'Datetime' in df.columns else [None] * len(df)

        for i in range(self.lookback, len(df) - self.lookback):
            # Swing High Detection
            is_swing_high = True
            for j in range(1, self.lookback + 1):
                if highs[i] <= highs[i-j] or highs[i] < highs[i+j]:
                    is_swing_high = False
                    break

            if is_swing_high:
                if not self.swings or not (self.swings[-1].index == i and self.swings[-1].level_type == 'SwingHigh'):
                    dt = pd.to_datetime(times[i]) if times[i] is not None else None
                    swing = StructureLevel(price=float(highs[i]), index=i, timestamp=dt, strength=1, level_type='SwingHigh')
                    self.swings.append(swing)
                    self.last_swing_high = swing

            # Swing Low Detection
            is_swing_low = True
            for j in range(1, self.lookback + 1):
                if lows[i] >= lows[i-j] or lows[i] > lows[i+j]:
                    is_swing_low = False
                    break

            if is_swing_low:
                if not self.swings or not (self.swings[-1].index == i and self.swings[-1].level_type == 'SwingLow'):
                    dt = pd.to_datetime(times[i]) if times[i] is not None else None
                    swing = StructureLevel(price=float(lows[i]), index=i, timestamp=dt, strength=1, level_type='SwingLow')
                    self.swings.append(swing)
                    self.last_swing_low = swing

    def _detect_structure_breaks(self, df: pd.DataFrame):
        """
        Detect BOS and CHOCH based on swing points and current trend.
        """
        if not self.swings:
            return

        closes = df['Close'].values
        times = df['Datetime'].values if 'Datetime' in df.columns else [None] * len(df)

        # Calculate ATR
        atr_period = 14
        prev_close = df['Close'].shift(1)
        tr = pd.concat([df['High'] - df['Low'], (df['High'] - prev_close).abs(), (df['Low'] - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean().values

        broken_swings = set()

        for i in range(1, len(df)):
            confirmed_swings = [s for s in self.swings if s.index + self.lookback <= i]
            if not confirmed_swings: continue

            latest_highs = [s for s in confirmed_swings if s.level_type == 'SwingHigh']
            latest_lows = [s for s in confirmed_swings if s.level_type == 'SwingLow']
            if not latest_highs or not latest_lows: continue

            curr_high_swing = latest_highs[-1]
            curr_low_swing = latest_lows[-1]

            dt_i = pd.to_datetime(times[i]) if times[i] is not None else None

            if closes[i] > curr_high_swing.price and curr_high_swing.index not in broken_swings:
                broken_swings.add(curr_high_swing.index)
                distance = closes[i] - curr_high_swing.price
                norm_dist = distance / atr[i] if atr[i] > 0 else 0

                if self.current_trend >= 0:
                    self.bos_list.append(BOS(index=i, direction=1, broken_level=curr_high_swing.price, timestamp=dt_i, strength=1, distance=distance, atr_normalized_distance=norm_dist))
                    self.bos_count += 1
                    self.last_bos_idx = i
                else:
                    self.choch_list.append(CHOCH(index=i, previous_trend=self.current_trend, new_trend=1, timestamp=dt_i, price=closes[i], strength=1))
                    self.choch_count += 1
                    self.last_choch_idx = i
                self.current_trend = 1
                self.protected_low = curr_low_swing

            elif closes[i] < curr_low_swing.price and curr_low_swing.index not in broken_swings:
                broken_swings.add(curr_low_swing.index)
                distance = curr_low_swing.price - closes[i]
                norm_dist = distance / atr[i] if atr[i] > 0 else 0

                if self.current_trend <= 0:
                    self.bos_list.append(BOS(index=i, direction=-1, broken_level=curr_low_swing.price, timestamp=dt_i, strength=1, distance=distance, atr_normalized_distance=norm_dist))
                    self.bos_count += 1
                    self.last_bos_idx = i
                else:
                    self.choch_list.append(CHOCH(index=i, previous_trend=self.current_trend, new_trend=-1, timestamp=dt_i, price=closes[i], strength=1))
                    self.choch_count += 1
                    self.last_choch_idx = i
                self.current_trend = -1
                self.protected_high = curr_high_swing

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
        self.swings = []; self.bos_list = []; self.choch_list = []
        self.bos_count = 0; self.choch_count = 0; self.current_trend = 0
        self.last_bos_idx = -1; self.last_choch_idx = -1
        self.last_swing_high = None; self.last_swing_low = None
        self.protected_high = None; self.protected_low = None

        self._detect_swings(df)
        self._detect_structure_breaks(df)

        # Optimized column assignment
        trend_arr = np.zeros(len(df)); bos_arr = np.zeros(len(df)); choch_arr = np.zeros(len(df))
        bos_cnt_arr = np.zeros(len(df)); choch_cnt_arr = np.zeros(len(df))
        last_bos_dir_arr = np.zeros(len(df)); last_choch_dir_arr = np.zeros(len(df))
        sh_arr = np.full(len(df), np.nan); sl_arr = np.full(len(df), np.nan)

        # Swings
        for s in self.swings:
            if s.level_type == 'SwingHigh': sh_arr[s.index + self.lookback:] = s.price
            else: sl_arr[s.index + self.lookback:] = s.price

        # BOS
        curr_bos_cnt = 0; last_bos_dir = 0
        for b in self.bos_list:
            bos_arr[b.index] = b.direction
            curr_bos_cnt += 1
            last_bos_dir = b.direction
            bos_cnt_arr[b.index:] = curr_bos_cnt
            last_bos_dir_arr[b.index:] = last_bos_dir

        # CHOCH
        curr_choch_cnt = 0; last_choch_dir = 0
        for c in self.choch_list:
            choch_arr[c.index] = c.new_trend
            curr_choch_cnt += 1
            last_choch_dir = c.new_trend
            choch_cnt_arr[c.index:] = curr_choch_cnt
            last_choch_dir_arr[c.index:] = last_choch_dir

        # Trend and Bars Since
        curr_tr = 0; last_b = -1; last_c = -1
        bs_b = np.full(len(df), -1); bs_c = np.full(len(df), -1)

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
