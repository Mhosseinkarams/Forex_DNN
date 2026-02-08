import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import sys
import os

class SignalGenerator:
    """
    A class to generate signals from trained LSTM models and enrich datasets.
    """

    def __init__(self, seq_len=48):
        self.base_path = Path(__file__).resolve().parent.parent
        self.data_path = self.base_path / "Data"
        self.seq_len = seq_len
        self.scaler = StandardScaler()

    def generate_all_signals(self, input_filename="GBPUSD_1h.csv", output_filename="GBPUSD_1h_enriched.csv"):
        input_file = self.data_path / input_filename
        if not input_file.exists():
            return None

        df = pd.read_csv(input_file)

        # Add indicators
        sys.path.append(str(self.base_path / "Colecting_Data"))
        from utils import TechnicalIndicators
        df = TechnicalIndicators.add_all_indicators(df)

        exclude_cols = ['DTYYYYMMDD', '<Time>', 'Datetime', 'Classification', 'Binary_Label', 'Multi_Label', 'Pivot_Label', 'Price_Change', 'Peak', 'Trough']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        scaled_features = self.scaler.fit_transform(df[feature_cols])

        X = []
        for i in range(len(scaled_features) - self.seq_len + 1):
            X.append(scaled_features[i : i + self.seq_len])
        X = np.array(X)

        signals = pd.DataFrame(index=df.index)
        models = {
            'binary': self.base_path / 'lstm_binary_classifier.h5',
            'multi': self.base_path / 'lstm_multi_classifier.h5',
            'pivot': self.base_path / 'lstm_pivot_classifier.h5'
        }

        for name, path in models.items():
            if path.exists():
                print(f"Generating signals with {name} model...")
                model = tf.keras.models.load_model(path)
                preds = model.predict(X, batch_size=512, verbose=0)

                if name == 'binary':
                    signals.loc[self.seq_len:, 'signal_binary'] = preds.flatten()
                elif name == 'multi':
                    signals.loc[self.seq_len:, 'signal_multi'] = np.argmax(preds, axis=1)
                elif name == 'pivot':
                    signals.loc[self.seq_len:, 'signal_pivot'] = np.argmax(preds, axis=1)

        df = pd.concat([df, signals], axis=1)
        df.fillna(0, inplace=True)
        df.to_csv(self.data_path / output_filename, index=False)
        print(f"Enriched data saved to {output_filename}")
        return df

if __name__ == "__main__":
    generator = SignalGenerator()
    # generator.generate_all_signals()
    print("SignalGenerator class ready.")
