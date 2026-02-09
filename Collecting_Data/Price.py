#!/bin/python3
import numpy as np
import pandas as pd
import yfinance as yf
import time
from datetime import datetime as dt
from pathlib import Path
import os

class YFinanceDataLoader:
    """
    Class to handle data acquisition from Yahoo Finance.
    """

    def __init__(self, ticker='GBPUSD=X'):
        self.ticker = ticker
        current_dir = Path(__file__).resolve().parent
        self.project_root = current_dir.parent
        self.data_dir = self.project_root / "Data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_data(self, period='2y', interval='1h'):
        """Downloads historical data and saves to CSV."""
        print(f"Downloading data for {self.ticker}...")
        try:
            data = yf.download(self.ticker, period=period, interval=interval, auto_adjust=True)
            if data.empty:
                print("No data downloaded.")
                return None

            # Flatten MultiIndex if present (common in recent yfinance versions)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Map columns to internal standard
            data.reset_index(inplace=True)
            # yfinance returns 'Datetime' or 'Date'
            date_col = 'Datetime' if 'Datetime' in data.columns else 'Date'
            data.rename(columns={date_col: 'Datetime'}, inplace=True)

            output_file = self.data_dir / f'GBPUSD_{interval}.csv'
            data.to_csv(output_file, index=False)
            print(f"Data saved to {output_file}")
            return data
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None

    def run_update_loop(self, interval_download='1h', sleep_time=300):
        """Continuously updates data."""
        while True:
            self.download_data(interval=interval_download)
            print(f"Waiting {sleep_time} seconds before next update...")
            time.sleep(sleep_time)

if __name__ == "__main__":
    loader = YFinanceDataLoader()
    loader.download_data()
