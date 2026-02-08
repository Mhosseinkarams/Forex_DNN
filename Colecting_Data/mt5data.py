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

    def __init__(self, mt5_id=None, password=None, server=None):
        self.mt5_id = mt5_id
        self.password = password
        self.server = server

        current_dir = Path(__file__).resolve().parent
        self.project_root = current_dir.parent
        self.data_dir = self.project_root / "Data"

        # Ensure Data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Add current dir to sys.path for utils
        if str(current_dir) not in sys.path:
            sys.path.append(str(current_dir))

        try:
            from utils import TechnicalIndicators
            self.indicators = TechnicalIndicators()
        except ImportError:
            print("Warning: TechnicalIndicators not found. Indicators will not be added.")
            self.indicators = None

    def initialize(self):
        """Initializes connection to MT5 terminal."""
        if not mt5.initialize():
            print("Failed to initialize MT5, error code:", mt5.last_error())
            return False

        if self.mt5_id and self.password and self.server:
            if not mt5.login(self.mt5_id, password=self.password, server=self.server):
                print("Failed to login to MT5, error code:", mt5.last_error())
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
            if mid == 0:
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

    def get_historical_data(self, symbol="GBPUSD", timeframe=mt5.TIMEFRAME_H1, count=None):
        """Fetches historical rates and current market state from MT5."""
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select symbol '{symbol}'. Error:", mt5.last_error())
            return None

        if count is None:
            max_candles, rates = self.get_max_bars(symbol, timeframe)
            print(f"Maximum accessible candles for {symbol}: {max_candles}")
        else:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

        if rates is None:
            print("No historical data available. Error:", mt5.last_error())
            return None

        df = pd.DataFrame(rates)

        # Standardized columns as suggested in user sample
        expected_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        if list(df.columns) == expected_columns or len(df.columns) == len(expected_columns):
             df.columns = expected_columns

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

        # Collect Ask and Bid price for spread calculation
        last_tick = mt5.symbol_info_tick(symbol)
        if last_tick is not None:
            df['Ask'] = last_tick.ask
            df['Bid'] = last_tick.bid
            # Use MT5 tick spread (in points)
            df['Tick_Spread'] = last_tick.spread

            # Measure Latency
            current_time = time.time()
            latency = (current_time - last_tick.time) * 1000
            print(f"Tick Latency: {latency:.2f} ms | Ask: {last_tick.ask} | Bid: {last_tick.bid}")

        # Add Technical Indicators
        if self.indicators:
            df = self.indicators.add_all_indicators(df)

        return df

    def run_update_loop(self, symbol="GBPUSD", timeframe=mt5.TIMEFRAME_H1, interval=900):
        """Runs a periodic update loop to fetch and save data."""
        output_file = self.data_dir / f'{symbol}_history.csv'

        while True:
            print(f"\n--- Update Cycle Started: {time.ctime()} ---")
            if self.initialize():
                data = self.get_historical_data(symbol, timeframe)
                if data is not None:
                    data.to_csv(output_file, index=False)
                    print(f"Successfully updated {output_file}")
                mt5.shutdown()
            else:
                print("MT5 Initialization failed. Retrying in next cycle.")

            print(f"Waiting {interval} seconds...")
            time.sleep(interval)

if __name__ == "__main__":
    # Example usage based on user samples
    loader = MT5DataLoader()
    print("MT5DataLoader initialized. Ready to collect data directly from MT5.")
    # loader.run_update_loop(symbol="EURUSD_i", timeframe=mt5.TIMEFRAME_H4)
