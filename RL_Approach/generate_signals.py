import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import sys
import os

# Set up paths
base_path = Path(__file__).resolve().parent.parent
data_path = base_path / "Data"
sys.path.append(str(base_path / "Colecting_Data"))
from utils import add_technical_indicators

def generate_lstm_signals():
    # Load raw data
    input_file = data_path / "GBPUSD_1h.csv"
    if not input_file.exists():
        print(f"Data not found at {input_file}")
        return

    df = pd.read_csv(input_file)
    df = add_technical_indicators(df)

    # Identify feature columns (must match what LSTM was trained on)
    # Binary/Multi/Pivot models were trained on all numeric features except labels
    exclude_cols = ['DTYYYYMMDD', '<Time>', 'Datetime', 'Classification', 'Binary_Label', 'Multi_Label', 'Pivot_Label', 'Price_Change', 'Peak', 'Trough']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])

    # Load models
    models = {
        'binary': base_path / 'lstm_binary_classifier.h5',
        'multi': base_path / 'lstm_multi_classifier.h5',
        'pivot': base_path / 'lstm_pivot_classifier.h5'
    }

    # Sequence length used during training
    seq_len = 48

    # Prepare inputs
    X = []
    for i in range(len(scaled_features) - seq_len + 1):
        X.append(scaled_features[i : i + seq_len])
    X = np.array(X)

    # Generate predictions
    signals = pd.DataFrame(index=df.index)

    for name, path in models.items():
        if path.exists():
            print(f"Generating signals with {name} model...")
            model = tf.keras.models.load_model(path)
            preds = model.predict(X, batch_size=512)

            # Align predictions with original dataframe
            # Predictions are for the step AFTER the sequence
            # So preds[0] is for index seq_len
            if name == 'binary':
                signals.loc[seq_len:, 'signal_binary'] = preds.flatten()
            elif name == 'multi':
                # Take the class with highest probability or expected movement
                # Classes: 0: Strong Sell, 1: Sell, 2: Neutral, 3: Buy, 4: Strong Buy
                signals.loc[seq_len:, 'signal_multi'] = np.argmax(preds, axis=1)
            elif name == 'pivot':
                # Classes: 0: None, 1: Peak, 2: Trough
                signals.loc[seq_len:, 'signal_pivot'] = np.argmax(preds, axis=1)
        else:
            print(f"Model {path} not found. Skipping {name} signals.")

    # Merge signals back to dataframe
    df = pd.concat([df, signals], axis=1)
    df.fillna(0, inplace=True)

    output_file = data_path / "GBPUSD_1h_enriched.csv"
    df.to_csv(output_file, index=False)
    print(f"Enriched data saved to {output_file}")

if __name__ == "__main__":
    generate_lstm_signals()
