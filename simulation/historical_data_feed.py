import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
import os

logger = logging.getLogger("HistoricalDataFeed")

class HistoricalDataFeed:
    def __init__(self):
        self.data = {} # (symbol, timeframe) -> DataFrame
        self.current_indices = {} # (symbol, timeframe) -> int
        self.symbols = []
        self.timeframes = []
        self._is_finished = False

    def load_csv(self, symbol: str, timeframe: str, filepath: str):
        """
        Purpose:
            Loads OHLCV data from a CSV file into the simulation memory.

        Arguments:
            symbol (str): Target symbol.
            timeframe (str): Timeframe string (e.g., "M5").
            filepath (str): Path to the source CSV file.

        Returns:
            bool: True if loading was successful.
        """
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return False
        
        df = pd.read_csv(filepath)
        # Assuming CSV has 'Datetime' or 'time'
        if 'Datetime' not in df.columns and 'time' in df.columns:
            df = df.rename(columns={'time': 'Datetime'})
        
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        if df['Datetime'].dt.tz is None:
            df['Datetime'] = df['Datetime'].dt.tz_localize('UTC')
        else:
            df['Datetime'] = df['Datetime'].dt.tz_convert('UTC')
            
        df = df.sort_values('Datetime').reset_index(drop=True)
        
        self.data[(symbol, timeframe)] = df
        self.current_indices[(symbol, timeframe)] = 0
        
        if symbol not in self.symbols:
            self.symbols.append(symbol)
        if timeframe not in self.timeframes:
            self.timeframes.append(timeframe)
            
        logger.info(f"Loaded {len(df)} bars for {symbol} {timeframe} from {filepath}")
        return True

    def get_ohlcv(self, symbol: str, timeframe_str: str, count: int = 1000) -> pd.DataFrame | None:
        """
        Purpose:
            Retrieves a slice of historical data up to the current virtual time.
            Matches the interface of MT5DataFeed for strategy compatibility.

        Arguments:
            symbol (str): Target symbol.
            timeframe_str (str): Target timeframe.
            count (int): Max bars to return.

        Returns:
            pd.DataFrame | None: Data slice or None.
        """
        if (symbol, timeframe_str) not in self.data:
            return None
        
        idx = self.current_indices[(symbol, timeframe_str)]
        df = self.data[(symbol, timeframe_str)]
        
        start_idx = max(0, idx - count + 1)
        return df.iloc[start_idx : idx + 1].copy()

    def get_current_bar(self, symbol: str, timeframe_str: str) -> pd.Series | None:
        """
        Purpose:
            Retrieves the OHLCV candle at the current virtual time index.

        Arguments:
            symbol (str): Target symbol.
            timeframe_str (str): Target timeframe.

        Returns:
            pd.Series | None: The current bar's data.
        """
        if (symbol, timeframe_str) not in self.data:
            return None
        idx = self.current_indices[(symbol, timeframe_str)]
        return self.data[(symbol, timeframe_str)].iloc[idx]

    def advance(self) -> bool:
        """
        Purpose:
            Increments the current data index for all loaded symbols/timeframes.
            Moves the virtual simulation "forward" by one bar.

        Returns:
            bool: True if data is still available, False if the end of history is reached.
        """
        finished_count = 0
        for key in self.data:
            if self.current_indices[key] < len(self.data[key]) - 1:
                self.current_indices[key] += 1
            else:
                finished_count += 1
        
        if finished_count == len(self.data):
            self._is_finished = True
            return False
        return True

    def reset(self):
        for key in self.current_indices:
            self.current_indices[key] = 0
        self._is_finished = False

    def is_finished(self) -> bool:
        """
        Purpose:
            Checks if the simulation has processed all available historical data.

        Returns:
            bool: True if all data indices have reached their limit.
        """
        return self._is_finished

    def connect(self): return True
    def disconnect(self): pass
    def check_health(self, symbol): return "HEALTHY"
