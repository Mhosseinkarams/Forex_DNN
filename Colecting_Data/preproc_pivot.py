#!/bin/python3
import pandas as pd
import numpy as np
from pathlib import Path
import sys

class PivotPreprocessor:
    """
    Preprocessor for major trend pivot detection.
    """

    def __init__(self):
        self.current_dir = Path(__file__).resolve().parent
        self.project_root = self.current_dir.parent
        self.data_dir = self.project_root / "Data"

        if str(self.current_dir) not in sys.path:
            sys.path.append(str(self.current_dir))

        try:
            from utils import TechnicalIndicators
            self.indicators = TechnicalIndicators()
        except ImportError:
            self.indicators = None

    def get_pivots(self, df, window=24):
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

    def label_pivots(self, df, horizon=24):
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

        if self.indicators:
            df = self.indicators.add_all_indicators(df)

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
