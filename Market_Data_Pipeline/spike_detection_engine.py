import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger("SpikeDetectionEngine")


class SpikeDetectionEngine:
    """
    Purpose:
        Analyze price candles and volumes sequentially to identify price and volume spikes.
        Helps protect strategies from entry during extreme illiquidity or flash crashes.
    """

    def __init__(self, price_spike_mult: float = 2.5, volume_spike_mult: float = 2.0, period: int = 20):
        self.price_spike_mult = price_spike_mult
        self.volume_spike_mult = volume_spike_mult
        self.period = period

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enriches DataFrame with price_spike and volume_spike boolean columns."""
        if df.empty:
            return df

        df_out = df.copy()
        n = len(df_out)

        price_spikes = [False] * n
        volume_spikes = [False] * n

        for i in range(n):
            if i < self.period:
                continue

            row = df_out.iloc[i]
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])
            open_p = float(row["Open"])
            volume = float(row.get("TickVolume", 0.0))

            candle_range = high - low
            atr = float(row.get("atr_14", 0.0001))

            # Detect price spike
            if candle_range > self.price_spike_mult * atr:
                price_spikes[i] = True

            # Detect volume spike
            slice_df = df_out.iloc[i - self.period:i]
            avg_volume = slice_df.get("TickVolume", pd.Series([0.0])).mean()
            if avg_volume > 0 and volume > self.volume_spike_mult * avg_volume:
                volume_spikes[i] = True

        df_out["price_spike"] = price_spikes
        df_out["volume_spike"] = volume_spikes
        return df_out

    def is_spike(self, df: pd.DataFrame, idx: int) -> Tuple[bool, bool]:
        """Returns (price_spike, volume_spike) at specific index."""
        if idx < 0 or idx >= len(df):
            return False, False

        row = df.iloc[idx]
        price_sp = bool(row.get("price_spike", False))
        volume_sp = bool(row.get("volume_spike", False))
        return price_sp, volume_sp
