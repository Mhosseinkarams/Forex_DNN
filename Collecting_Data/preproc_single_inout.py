#!/bin/python3
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

class SingleInOutPreprocessor:
    """
    Preprocessor for single-output models (Dense networks).
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
            print("Error: Collecting_Data/utils.py not found.")
            self.indicators = None

    def preprocess(self, filename="GBPUSD_1h.csv"):
        input_file = self.data_dir / filename
        if not input_file.exists():
            print(f"Error: {input_file} not found.")
            return None

        df = pd.read_csv(input_file)

        # Standardize Vol column
        if 'Volume' in df.columns and 'Vol' not in df.columns:
            df.rename(columns={'Volume': 'Vol'}, inplace=True)

        cols = ['Open', 'High', 'Low', 'Close', 'Vol']
        df = df[[c for c in cols if c in df.columns]]

        if self.indicators:
            df = self.indicators.add_all_indicators(df)

        # Price change for labeling
        df['Price_Change'] = df['Close'].shift(-1) - df['Close']
        df['Binary_Label'] = (df['Price_Change'] > 0).astype(int)

        def label_multi(change):
            if change > 0.001: return 4
            elif change > 0.0002: return 3
            elif change < -0.001: return 0
            elif change < -0.0002: return 1
            else: return 2

        df['Multi_Label'] = df['Price_Change'].apply(label_multi)
        df.dropna(inplace=True)

        return df

    def save(self, df, output_filename="GBPUSD_1h_preprocessed.csv"):
        output_path = self.data_dir / output_filename
        df.to_csv(output_path, index=False)
        print(f"Preprocessing complete. Saved to {output_path}")

if __name__ == "__main__":
    preprocessor = SingleInOutPreprocessor()
    df = preprocessor.preprocess()
    if df is not None:
        preprocessor.save(df)
