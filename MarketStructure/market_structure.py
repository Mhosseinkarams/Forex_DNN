import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Configure logger
logger = logging.getLogger('MarketStructureEngine')

@dataclass
class SwingPoint:
    index: int
    price: float
    type: str  # 'High' or 'Low'
    strength: int
    timestamp: Optional[datetime] = None

@dataclass
class BOS:
    index: int
    direction: int  # 1 for Bullish, -1 for Bearish
    broken_level: float
    timestamp: Optional[datetime] = None
    strength: int = 1
    distance: float = 0.0
    atr_normalized_distance: float = 0.0

@dataclass
class CHOCH:
    index: int
    previous_trend: int
    new_trend: int
    timestamp: Optional[datetime] = None
    price: float = 0.0
    strength: int = 1

class MarketStructureEngine:
    """
    SMC Market Structure Engine based on SmartMoneyConcepts.mq5.
    Extracts objective Smart Money market structure from OHLC data using pivot-to-pivot breakouts.
    """
    def __init__(self, lookback: int = 3):
        self.lookback = lookback  # Swing Length (bars)
        self.swings: List[SwingPoint] = []
        self.bos_list: List[BOS] = []
        self.choch_list: List[CHOCH] = []

        # Internal state
        self.current_trend: int = 0  # 1: Bullish, -1: Bearish, 0: Neutral
        self.last_swing_high: Optional[SwingPoint] = None
        self.last_swing_low: Optional[SwingPoint] = None

        self.bos_count: int = 0
        self.choch_count: int = 0
        self.last_bos_idx: int = -1
        self.last_choch_idx: int = -1

    def _detect_swings(self, df: pd.DataFrame):
        # Kept for backward compatibility interface, but actual work is done sequentially in process()
        pass

    def _detect_structure_breaks(self, df: pd.DataFrame):
        # Kept for backward compatibility interface, but actual work is done sequentially in process()
        pass

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
        n_bars = len(df)

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

        # Prep output arrays
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

        # Precompute pivots
        piv_highs = np.zeros(n_bars, dtype=bool)
        piv_lows = np.zeros(n_bars, dtype=bool)

        for i in range(self.lookback, n_bars - self.lookback):
            # IsPivotHigh
            p_high = highs[i]
            is_ph = True
            for k in range(1, self.lookback + 1):
                if highs[i - k] >= p_high or highs[i + k] >= p_high:
                    is_ph = False
                    break
            piv_highs[i] = is_ph

            # IsPivotLow
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

        # Calculate ATR
        atr_period = 14
        prev_close = df['Close'].shift(1)
        tr = pd.concat([df['High'] - df['Low'], (df['High'] - prev_close).abs(), (df['Low'] - prev_close).abs()], axis=1).max(axis=1)
        atr_vals = tr.rolling(window=atr_period).mean().fillna(0.0010).values

        # Sequential processing
        for t in range(n_bars):
            # Check if there is a confirmed pivot at idx = t - self.lookback
            idx = t - self.lookback
            if idx >= self.lookback:
                # Process Pivot High
                if piv_highs[idx]:
                    pivPrice = highs[idx]
                    pivTime = times[idx] if isinstance(times[idx], datetime) else (pd.to_datetime(times[idx]) if times[idx] is not None else None)

                    # Create SwingPoint
                    swing = SwingPoint(index=idx, price=float(pivPrice), type='High', strength=self.lookback, timestamp=pivTime)
                    self.swings.append(swing)
                    self.last_swing_high = swing

                    # Breakout check (BOS/CHoCH)
                    if g_phLevel > 0 and pivPrice > g_phLevel:
                        tag = "CHoCH" if g_trend == -1 else "BOS"
                        g_trend = 1

                        # Distance & ATR norm distance
                        distance = float(pivPrice - g_phLevel)
                        norm_dist = float(distance / atr_vals[t]) if atr_vals[t] > 0 else 0.0

                        if tag == "BOS":
                            bos_obj = BOS(
                                index=t,  # confirmation candle index
                                direction=1,
                                broken_level=float(g_phLevel),
                                timestamp=times[t],
                                strength=self.lookback,
                                distance=distance,
                                atr_normalized_distance=norm_dist
                            )
                            self.bos_list.append(bos_obj)
                            self.bos_count += 1
                            self.last_bos_idx = t
                        else:
                            choch_obj = CHOCH(
                                index=t,
                                previous_trend=-1,
                                new_trend=1,
                                timestamp=times[t],
                                price=float(pivPrice),
                                strength=self.lookback
                            )
                            self.choch_list.append(choch_obj)
                            self.choch_count += 1
                            self.last_choch_idx = t

                    # Update level
                    g_phLevel = pivPrice
                    g_phIdx = idx

                # Process Pivot Low
                if piv_lows[idx]:
                    pivPrice = lows[idx]
                    pivTime = times[idx] if isinstance(times[idx], datetime) else (pd.to_datetime(times[idx]) if times[idx] is not None else None)

                    # Create SwingPoint
                    swing = SwingPoint(index=idx, price=float(pivPrice), type='Low', strength=self.lookback, timestamp=pivTime)
                    self.swings.append(swing)
                    self.last_swing_low = swing

                    # Breakdown check
                    if g_plLevel > 0 and pivPrice < g_plLevel:
                        tag = "CHoCH" if g_trend == 1 else "BOS"
                        g_trend = -1

                        # Distance & ATR norm distance
                        distance = float(g_plLevel - pivPrice)
                        norm_dist = float(distance / atr_vals[t]) if atr_vals[t] > 0 else 0.0

                        if tag == "BOS":
                            bos_obj = BOS(
                                index=t,
                                direction=-1,
                                broken_level=float(g_plLevel),
                                timestamp=times[t],
                                strength=self.lookback,
                                distance=distance,
                                atr_normalized_distance=norm_dist
                            )
                            self.bos_list.append(bos_obj)
                            self.bos_count += 1
                            self.last_bos_idx = t
                        else:
                            choch_obj = CHOCH(
                                index=t,
                                previous_trend=1,
                                new_trend=-1,
                                timestamp=times[t],
                                price=float(pivPrice),
                                strength=self.lookback
                            )
                            self.choch_list.append(choch_obj)
                            self.choch_count += 1
                            self.last_choch_idx = t

                    # Update level
                    g_plLevel = pivPrice
                    g_plIdx = idx

            # Populate point-in-time state arrays
            trend_arr[t] = g_trend
            bos_cnt_arr[t] = self.bos_count
            choch_cnt_arr[t] = self.choch_count

            if self.last_bos_idx != -1:
                bos_arr[self.last_bos_idx] = self.bos_list[-1].direction if self.bos_list else 0
                last_bos_dir_arr[t] = self.bos_list[-1].direction if self.bos_list else 0
            if self.last_choch_idx != -1:
                choch_arr[self.last_choch_idx] = self.choch_list[-1].new_trend if self.choch_list else 0
                last_choch_dir_arr[t] = self.choch_list[-1].new_trend if self.choch_list else 0

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
