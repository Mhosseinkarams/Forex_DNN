#!/bin/python3
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import sys
import os
from pathlib import Path

# Set up paths relative to project root
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
data_dir = project_root / "Data"

# Add the current directory to sys.path to import utils
sys.path.append(str(current_dir))
try:
    from utils import add_technical_indicators
except ImportError:
    print("Error: Colecting_Data/utils.py not found.")
    exit(1)

# Load the data from GBPUSD_1h.csv
input_file = data_dir / "GBPUSD_1h.csv"
try:
    data = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: {input_file} not found.")
    exit(1)

# Basic column selection
cols = ['Open', 'High', 'Low', 'Close', 'Volume']
data = data[cols]

# Add technical indicators
data = add_technical_indicators(data)

# Create a function to label the next candle based on price change
def label_next_candle(current_close, next_close):
    price_change =  next_close - current_close
    if price_change > 0.0005:
        return 2
    elif price_change < -0.0005:
        return 1
    else:
        return 0

# Add the classification column to the data
data['Classification'] = [label_next_candle(current_close, next_close)
                          for current_close, next_close in zip(data['Close'], data['Close'].shift(-1))]

# Drop the last row because it will have NaN in Classification due to shift(-1)
data.dropna(inplace=True)

# save the data to a csv file
output_file = data_dir / "GBPUSD_1h_preprocessed.csv"
data.to_csv(output_file, index=False)
print(f"Preprocessing complete. Saved to {output_file}")
