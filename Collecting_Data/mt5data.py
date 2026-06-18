#!/bin/python3

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
import os

# Try to load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class MT5DataLoader:
    """
    Enhanced class to handle data acquisition directly from MetaTrader 5 terminal.
    Incorporates historical data fetching, spread analysis, and latency measurement.
    """

    def __init__(self, mt5_id=None, password=None, server=None):
        # Use provided parameters or fall back to environment variables
        self.mt5_id = int(mt5_id or os.getenv("MT5_ID", 0))
        self.password = password or os.getenv("MT5_PASSWORD", "")
        self.server = server or os.getenv("MT5_SERVER", "")

        current_dir = Path(__file__).resolve().parent
        self.project_root = current_dir.parent
        self.data_dir = self.project_root / "Data"

        # Ensure Data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Add current dir to sys.path for indicators
        if str(current_dir) not in sys.path:
            sys.path.append(str(current_dir))

        try:
            from indicators import IndicatorEngine
            self.engine = IndicatorEngine(dropna=True)
        except ImportError:
            print("Warning: IndicatorEngine not found. Indicators will not be added.")
            self.engine = None

    def initialize(self):
        """Initializes connection to MT5 terminal with provided credentials."""
        if not mt5.initialize(login=self.mt5_id, password=self.password, server=self.server):
            print("Failed to initialize MT5, error code:", mt5.last_error())
            return False
        return True

    def get_max_bars(self, symbol, timeframe):
        """Determines maximum accessible historical candles using binary search."""
        low = 0
        high = 10000000
        max_bars = 0
        last_rates = None

        while low <= high:
            mid = (low + high) // 2
            if mid <= 0:
                low = 1
                continue

            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, mid)
            if rates is None:
                high = mid - 1
            else:
                current_len = len(rates)
                if current_len == mid:
                    max_bars = mid
                    last_rates = rates
                    low = mid + 1
                else:
                    high = mid - 1
        return max_bars, last_rates

    def measure_latency(self, symbol, num_samples=10):
        """Measures average data latency and API execution time."""
        data_latencies = []
        execution_times = []

        for _ in range(num_samples):
            start_time = time.time()
            last_tick = mt5.symbol_info_tick(symbol)
            exec_time = (time.time() - start_time) * 1000  # ms

            if last_tick is not None:
                tick_time_sec = last_tick.time
                current_time = time.time()
                data_latency_ms = (current_time - tick_time_sec) * 1000

                data_latencies.append(data_latency_ms)
                execution_times.append(exec_time)

        if data_latencies:
            avg_latency = np.mean(data_latencies)
            avg_exec = np.mean(execution_times)
            print(f"Avg Data Latency: {avg_latency:.2f} ms | Avg API Exec: {avg_exec:.2f} ms")
            return avg_latency, avg_exec
        return None, None

    def get_historical_data(self, symbol="GBPUSD_i", timeframe=mt5.TIMEFRAME_M5, count=10000):
        """Fetches historical rates and current market state from MT5."""
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select symbol '{symbol}'. Error:", mt5.last_error())
            return None

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

        if rates is None:
            print("No historical data available. Error:", mt5.last_error())
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Map to project internal naming standards
        df.rename(columns={
            'time': 'Datetime',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Vol'
        }, inplace=True)

        # Standardize for IndicatorEngine
        df['TickVolume'] = df['Vol']
        df['Spread'] = 0 # Placeholder

        # Fetch the latest tick for current Ask and Bid prices
        last_tick = mt5.symbol_info_tick(symbol)
        if last_tick is not None:
            # Add Ask and Bid to the entire dataframe (representing the current market state)
            df['Ask'] = last_tick.ask
            df['Bid'] = last_tick.bid
            # Spread in points
            df['Current_Spread'] = last_tick.spread
            df['Spread'] = last_tick.spread
            print(f"Current Ask: {last_tick.ask}, Bid: {last_tick.bid}, Spread: {last_tick.spread}")

        if self.engine:
            df = self.engine.calculate(df)

        return df

if __name__ == "__main__":
    loader = MT5DataLoader()
    if loader.initialize():
        df = loader.get_historical_data()
        if df is not None:
            print("Fetched and enriched data sample:")
            print(df.head())
        mt5.shutdown()
