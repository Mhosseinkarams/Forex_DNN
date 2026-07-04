import pandas as pd
import os
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("HistoricalDataFeed")

class HistoricalDataFeed:
    """
    Serves historical OHLCV data from CSV files.
    Mirroring MT5DataFeed interface where applicable.
    """
    def __init__(self, data_dir: str = "Data"):
        self.data_dir = data_dir
        self.symbol_data: Dict[str, Dict[str, pd.DataFrame]] = {} # symbol -> timeframe -> df
        self.current_indices: Dict[str, Dict[str, int]] = {} # symbol -> timeframe -> current_index

    def load_symbol_data(self, symbol: str, timeframe: str, csv_file: str):
        filepath = os.path.join(self.data_dir, csv_file)
        if not os.path.exists(filepath):
            logger.error(f"CSV file not found: {filepath}")
            return False

        try:
            df = pd.read_csv(filepath)
            # Ensure Datetime is converted
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            # Ensure common columns
            expected = ["Datetime", "Open", "High", "Low", "Close", "TickVolume", "Spread"]
            for col in expected:
                if col not in df.columns:
                    if col == "Spread":
                        df["Spread"] = 1
                    elif col == "TickVolume":
                        df["TickVolume"] = 100
                    else:
                        logger.error(f"Missing required column {col} in {csv_file}")
                        return False

            df = df[expected].sort_values("Datetime").reset_index(drop=True)

            if symbol not in self.symbol_data:
                self.symbol_data[symbol] = {}
                self.current_indices[symbol] = {}

            self.symbol_data[symbol][timeframe] = df
            self.current_indices[symbol][timeframe] = 0
            logger.info(f"Loaded {len(df)} bars for {symbol} {timeframe} from {csv_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load {csv_file}: {e}")
            return False

    def get_ohlcv(self, symbol: str, timeframe: str, count: int = 1000) -> Optional[pd.DataFrame]:
        """
        Alias for get_latest_bars to mirror MT5DataFeed wrapper.
        """
        return self.get_latest_bars(symbol, timeframe, count)

    def get_latest_bars(self, symbol: str, timeframe: str, count: int = 1000) -> Optional[pd.DataFrame]:
        """
        Returns the LATEST 'count' bars relative to the CURRENT cursor position.
        """
        if symbol not in self.symbol_data or timeframe not in self.symbol_data[symbol]:
            return None

        df = self.symbol_data[symbol][timeframe]
        idx = self.current_indices[symbol][timeframe]

        if idx < 0:
            return None

        start_idx = max(0, idx - count + 1)
        return df.iloc[start_idx : idx + 1].copy()

    def advance(self, symbol: str = None, timeframe: str = None) -> bool:
        """
        Advances the cursor for specific symbol/timeframe or ALL if None.
        """
        if symbol and timeframe:
            if symbol in self.current_indices and timeframe in self.current_indices[symbol]:
                if self.current_indices[symbol][timeframe] < len(self.symbol_data[symbol][timeframe]) - 1:
                    self.current_indices[symbol][timeframe] += 1
                    return True
                return False
            return False

        any_advanced = False
        for s in self.current_indices:
            for tf in self.current_indices[s]:
                if self.current_indices[s][tf] < len(self.symbol_data[s][tf]) - 1:
                    self.current_indices[s][tf] += 1
                    any_advanced = True
        return any_advanced

    def is_finished(self, symbol: str, timeframe: str) -> bool:
        if symbol not in self.symbol_data or timeframe not in self.symbol_data[symbol]:
            return True
        return self.current_indices[symbol][timeframe] >= len(self.symbol_data[symbol][timeframe]) - 1

    def get_current_bar(self, symbol: str, timeframe: str) -> Optional[pd.Series]:
        if symbol not in self.symbol_data or timeframe not in self.symbol_data[symbol]:
            return None
        idx = self.current_indices[symbol][timeframe]
        return self.symbol_data[symbol][timeframe].iloc[idx]

    def reset(self):
        for s in self.current_indices:
            for tf in self.current_indices[s]:
                self.current_indices[s][tf] = 0
