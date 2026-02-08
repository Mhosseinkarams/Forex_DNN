#!/bin/python3

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

class MT5DataLoader:
    """
    Enhanced class to handle data acquisition directly from MetaTrader 5 terminal.
    Incorporates historical data fetching, spread analysis, and latency measurement.
    """

    # User-provided Credentials
    DEFAULT_MT5_ID = 90874984
    DEFAULT_PASSWORD = 'Lord@7516'
    DEFAULT_SERVER = 'LiteFinance-MT5-Demo'

    def __init__(self, mt5_id=DEFAULT_MT5_ID, password=DEFAULT_PASSWORD, server=DEFAULT_SERVER):
        self.mt5_id = mt5_id
        self.password = password
        self.server = server

        current_dir = Path(__file__).resolve().parent
        self.project_root = current_dir.parent
        self.data_dir = self.project_root / "Data"

        # Ensure Data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

# Add current dir to sys.path for utils
sys.path.append(str(current_dir))
try:
    from utils import add_technical_indicators
except ImportError:
    print("Warning: utils not found. Technical indicators will not be added.")
    add_technical_indicators = lambda x: x

# MT5 Login Credentials - FILL THESE IN
MT5_ID = 12345678  # Replace with your account ID
MT5_PASSWORD = 'YourPassword'  # Replace with your password
MT5_SERVER = 'YourBrokerServer'  # Replace with your broker server

def initialize_mt5():
    """Initializes connection to MT5 terminal."""
    if not mt5.initialize():
        print("Failed to initialize MT5, error code:", mt5.last_error())
        return False

    # Optional: Login if the terminal is not already logged in
    # if not mt5.login(MT5_ID, password=MT5_PASSWORD, server=MT5_SERVER):
    #     print("Failed to login to MT5, error code:", mt5.last_error())
    #     return False

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

    # Fetch the latest tick for current Ask and Bid prices
    last_tick = mt5.symbol_info_tick(symbol)
    if last_tick is None:
        print(f"Failed to fetch latest tick for {symbol}, error code:", mt5.last_error())
    else:
        # Add Ask and Bid to the entire dataframe (representing the current market state)
        df['Ask'] = last_tick.ask
        df['Bid'] = last_tick.bid
        # Spread in points
        df['Current_Spread'] = last_tick.spread
        print(f"Current Ask: {last_tick.ask}, Bid: {last_tick.bid}, Spread: {last_tick.spread}")

        return df

while True:
    print(f"Attempting to fetch data at {time.ctime()}...")
    if initialize_mt5():
        # Configuration
        symbol = "GBPUSD"
        timeframe = mt5.TIMEFRAME_H1
        output_file = data_dir / 'GBPUSD_1h_2.csv'

        data = get_data(symbol, timeframe)

        if data is not None:
            # Add technical indicators from utils.py
            data = add_technical_indicators(data)

            # Save the DataFrame as a .csv file
            data.to_csv(output_file, index=False)
            print(f"******************* Successfully updated {output_file} *******************")

        # Shutdown connection until next iteration
        mt5.shutdown()
    else:
        print("MT5 Initialization failed. Retrying in next cycle.")

    # Wait for 15 minutes (900 seconds)
    time.sleep(900)
