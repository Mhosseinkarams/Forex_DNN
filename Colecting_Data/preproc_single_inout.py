#!/bin/python3
import numpy as np
import pandas as pd
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
cols = ['Open', 'High', 'Low', 'Close', 'Vol']
# Handle cases where MT5 might output 'Volume' instead of 'Vol'
if 'Volume' in data.columns and 'Vol' not in data.columns:
    data.rename(columns={'Volume': 'Vol'}, inplace=True)
data = data[[c for c in cols if c in data.columns]]

# Add technical indicators
data = add_technical_indicators(data)

# Calculate price change
data['Price_Change'] = data['Close'].shift(-1) - data['Close']

# 1. Binary Label: Up (1) or Down (0)
data['Binary_Label'] = (data['Price_Change'] > 0).astype(int)

# 2. Multi-class Label: Direction + Size
def label_multi(change):
    if change > 0.001: return 4    # Strong Buy
    elif change > 0.0002: return 3 # Buy
    elif change < -0.001: return 0 # Strong Sell
    elif change < -0.0002: return 1 # Sell
    else: return 2                 # Neutral

data['Multi_Label'] = data['Price_Change'].apply(label_multi)

# Drop the last row because it will have NaN in Price_Change due to shift(-1)
data.dropna(inplace=True)

# save the data to a csv file
output_file = data_dir / "GBPUSD_1h_preprocessed.csv"
data.to_csv(output_file, index=False)
print(f"Preprocessing complete. Saved to {output_file}")
