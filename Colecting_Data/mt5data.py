#!/bin/python3

import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys
import MetaTrader5 as mt5

# Set up paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
data_dir = project_root / "Data"

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

def get_data(symbol="GBPUSD", timeframe=mt5.TIMEFRAME_H1, count=10000):
    """Fetches historical rates and current tick info from MT5."""
    # Ensure the symbol is visible in Market Watch
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select symbol {symbol}, error code:", mt5.last_error())
        return None

    # Fetch historical rates
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        print(f"Failed to copy rates for {symbol}, error code:", mt5.last_error())
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Rename columns to match project standards
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
