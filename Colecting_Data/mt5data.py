#!/bin/python3

import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys
import MetaTrader5 as mt5

class MT5DataLoader:
    """
    A class to handle data acquisition from MetaTrader 5.
    """

    def __init__(self, mt5_id=None, password=None, server=None):
        self.mt5_id = mt5_id
        self.password = password
        self.server = server

        current_dir = Path(__file__).resolve().parent
        self.project_root = current_dir.parent
        self.data_dir = self.project_root / "Data"

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

    def get_historical_data(self, symbol="GBPUSD", timeframe=mt5.TIMEFRAME_H1, count=10000):
        """Fetches historical rates and current tick info from MT5."""
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select symbol {symbol}, error code:", mt5.last_error())
            return None

        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            print(f"Failed to copy rates for {symbol}, error code:", mt5.last_error())
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        df.rename(columns={
            'time': 'Datetime',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Vol'
        }, inplace=True)

        last_tick = mt5.symbol_info_tick(symbol)
        if last_tick is not None:
            df['Ask'] = last_tick.ask
            df['Bid'] = last_tick.bid
            df['Current_Spread'] = last_tick.spread

        if self.indicators:
            df = self.indicators.add_all_indicators(df)

        return df

    def run_update_loop(self, symbol="GBPUSD", timeframe=mt5.TIMEFRAME_H1, interval=900):
        """Runs a periodic update loop to fetch and save data."""
        output_file = self.data_dir / f'{symbol}_{timeframe}_live.csv'

        while True:
            print(f"Attempting to fetch data at {time.ctime()}...")
            if self.initialize():
                data = self.get_historical_data(symbol, timeframe)
                if data is not None:
                    data.to_csv(output_file, index=False)
                    print(f"******************* Successfully updated {output_file} *******************")
                mt5.shutdown()
            else:
                print("MT5 Initialization failed. Retrying in next cycle.")

            time.sleep(interval)

if __name__ == "__main__":
    loader = MT5DataLoader()
    # Note: For actual use, you'd call loader.run_update_loop()
    print("MT5DataLoader class ready.")
