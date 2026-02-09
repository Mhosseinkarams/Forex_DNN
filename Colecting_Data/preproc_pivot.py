#!/bin/python3
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

class PivotPreprocessor:
    """
    Preprocessor for major trend pivot detection.
    """
    
    def __init__(self):
        self.current_dir = Path(__file__).resolve().parent
        self.project_root = self.current_dir.parent
        self.data_dir = self.project_root / "Data"
        
        # Ensure Colecting_Data is in path for utils import
        if str(self.current_dir) not in sys.path:
            sys.path.append(str(self.current_dir))
            
        try:
            from utils import TechnicalIndicators
            self.indicators = TechnicalIndicators()
        except ImportError:
            print("Warning: TechnicalIndicators not found in utils.py")
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

    def preprocess(self, filename="GBPUSD_1h.csv", window=24, horizon=24):
        input_file = self.data_dir / filename
        if not input_file.exists():
            print(f"Error: {input_file} not found.")
            return None

        df = pd.read_csv(input_file)
        if 'Volume' in df.columns and 'Vol' not in df.columns:
            df.rename(columns={'Volume': 'Vol'}, inplace=True)

        if self.indicators:
            df = self.indicators.add_all_indicators(df)

        df = self.get_pivots(df, window=window)
        df = self.label_pivots(df, horizon=horizon)
        df = df.iloc[:-horizon]
        
        return df

    def save(self, df, output_filename="GBPUSD_1h_pivot.csv"):
        output_path = self.data_dir / output_filename
        df.to_csv(output_path, index=False)
        print(f"Pivot preprocessing complete. Saved to {output_path}")

if __name__ == "__main__":
    preprocessor = PivotPreprocessor()
    df = preprocessor.preprocess()
    if df is not None:
        preprocessor.save(df)
