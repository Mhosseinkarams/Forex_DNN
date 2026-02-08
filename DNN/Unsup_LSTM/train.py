import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from pathlib import Path
import sys
import os

class UnsupLSTMAutoencoder:
    """
    An unsupervised LSTM Autoencoder for anomaly detection and trend analysis.
    """

    def __init__(self, sequence_length=24, num_features=None):
        self.base_path = Path(__file__).resolve().parent.parent.parent
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        if num_features:
            self.model = self._build_model(sequence_length, num_features)

    def _build_model(self, sequence_length, num_features):
        model = Sequential([
            LSTM(units=128, activation='relu', input_shape=(sequence_length, num_features)),
            RepeatVector(sequence_length),
            LSTM(units=128, activation='relu', return_sequences=True),
            TimeDistributed(Dense(units=num_features))
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def load_and_preprocess(self, filename='GBPUSD_1h.csv'):
        data_file = self.base_path / "Data" / filename
        if not data_file.exists():
            return None

        data = pd.read_csv(data_file)

        sys.path.append(str(self.base_path / "Colecting_Data"))
        from utils import TechnicalIndicators
        data = TechnicalIndicators.add_all_indicators(data)

        features = data.select_dtypes(include=[np.number])
        self.num_features = features.shape[1]

        scaled_features = self.scaler.fit_transform(features)

        sequences = []
        for i in range(len(scaled_features) - self.sequence_length):
            sequences.append(scaled_features[i : i + self.sequence_length])

        return np.array(sequences)

    def train(self, X, epochs=50, batch_size=32):
        if self.model is None:
            self.model = self._build_model(self.sequence_length, self.num_features)

        print("Training LSTM Autoencoder...")
        history = self.model.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=1)
        return history

    def calculate_reconstruction_error(self, X):
        reconstructed = self.model.predict(X)
        mse = np.mean(np.square(X - reconstructed), axis=(1, 2))
        return mse

if __name__ == "__main__":
    autoencoder = UnsupLSTMAutoencoder()
    X = autoencoder.load_and_preprocess()
    if X is not None:
        autoencoder.train(X, epochs=10)
