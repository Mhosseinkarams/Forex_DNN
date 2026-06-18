#!/bin/python3

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

class MultiInOutPreprocessor:
    """
    Preprocessor for multi-input/output models (LSTM/Sequence models).
    """

    def __init__(self):
        self.current_dir = Path(__file__).resolve().parent
        self.project_root = self.current_dir.parent
        self.data_dir = self.project_root / "Data"

        if str(self.current_dir) not in sys.path:
            sys.path.append(str(self.current_dir))

        try:
            from indicators import IndicatorEngine
            self.engine = IndicatorEngine(dropna=True)
        except ImportError:
            self.engine = None

    def preprocess(self, filename="GBPUSD_1h.csv", num_input_candles=100, num_output_candles=5):
        input_file = self.data_dir / filename
        if not input_file.exists():
            print(f"Error: {input_file} not found.")
            return None, None

        df = pd.read_csv(input_file)
        
        # Standardize columns for IndicatorEngine
        if 'Volume' in df.columns and 'Vol' not in df.columns:
            df.rename(columns={'Volume': 'Vol'}, inplace=True)
        if 'TickVolume' not in df.columns and 'Vol' in df.columns:
            df['TickVolume'] = df['Vol']
        if 'Spread' not in df.columns:
            df['Spread'] = 0

        if self.engine:
            df = self.engine.calculate(df)

        input_data = []
        output_data = []

        # Use numeric columns
        cols_to_use = df.select_dtypes(include=[np.number]).columns

        for i in range(len(df) - num_input_candles - num_output_candles + 1):
            input_candles = df.iloc[i : i + num_input_candles][cols_to_use].values
            output_candles = df.iloc[i + num_input_candles : i + num_input_candles + num_output_candles][['Open', 'Close']].values

            price_diffs = output_candles[:, 1] - output_candles[:, 0]
            output = np.where(price_diffs > 0.0005, 1, np.where(price_diffs < -0.0005, -1, 0))

            input_data.append(input_candles)
            output_data.append(output)

        return np.array(input_data), np.array(output_data)

    def save_numpy(self, input_data, output_data, prefix="GBPUSD_1h_multi"):
        np.savez_compressed(self.data_dir / f"{prefix}_inout.npz", input_data=input_data)
        np.savez_compressed(self.data_dir / f"{prefix}_out.npz", output_data=output_data)
        print(f"Preprocessing complete. Saved to {self.data_dir}")

if __name__ == "__main__":
    preprocessor = MultiInOutPreprocessor()
    X, y = preprocessor.preprocess()
    if X is not None:
        preprocessor.save_numpy(X, y)
