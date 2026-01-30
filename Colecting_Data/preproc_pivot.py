#!/bin/python3
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Set up paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
data_dir = project_root / "Data"

# Add current dir to sys.path for utils
sys.path.append(str(current_dir))
try:
    from utils import add_technical_indicators
except ImportError:
    print("Error: Colecting_Data/utils.py not found.")
    exit(1)

def get_pivots(df, window=24):
    """
    Identifies local peaks and troughs in a given window.
    """
    close = df['Close'].values
    peaks = np.zeros(len(df))
    troughs = np.zeros(len(df))

    for i in range(window, len(df) - window):
        chunk = close[i-window : i+window+1]
        if close[i] == np.max(chunk):
            peaks[i] = 1
        if close[i] == np.min(chunk):
            troughs[i] = 1

    df['Peak'] = peaks
    df['Trough'] = troughs
    return df

def label_pivots(df, horizon=24):
    """
    Labels each step based on the next major pivot within the horizon.
    1: Next major pivot is a Peak (Trend change to Down soon)
    2: Next major pivot is a Trough (Trend change to Up soon)
    0: No major pivot in the immediate horizon
    """
    peaks = df['Peak'].values
    troughs = df['Trough'].values
    labels = np.zeros(len(df))

    for i in range(len(df) - horizon):
        next_peaks = peaks[i+1 : i+horizon+1]
        next_troughs = troughs[i+1 : i+horizon+1]

        if np.any(next_peaks == 1):
            labels[i] = 1
        elif np.any(next_troughs == 1):
            labels[i] = 2

    df['Pivot_Label'] = labels
    return df

# Load data
input_file = data_dir / "GBPUSD_1h.csv"
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    print(f"Error: {input_file} not found.")
    exit(1)

# Standardize Volume column if needed
if 'Volume' in df.columns and 'Vol' not in df.columns:
    df.rename(columns={'Volume': 'Vol'}, inplace=True)

# Add indicators
df = add_technical_indicators(df)

# Identify major pivots (24-hour window for "major")
df = get_pivots(df, window=24)

# Label data (Predicting pivot in the next 24 hours)
df = label_pivots(df, horizon=24)

# Drop rows where we can't determine labels (at the end)
df = df.iloc[:-24]

# Save
output_file = data_dir / "GBPUSD_1h_pivot.csv"
df.to_csv(output_file, index=False)
print(f"Pivot preprocessing complete. Saved to {output_file}")
