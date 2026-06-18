#!/bin/python3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

class BinaryLSTMClassifier:
    """
    A class for binary sequence classification using LSTMs.
    Predicts directional movement (Up/Down).
    """

    def __init__(self, num_input_candles=48, num_features=None):
        self.base_path = Path(__file__).resolve().parent.parent.parent
        self.num_input_candles = num_input_candles
        self.num_features = num_features
        self.model = None
        self.scaler = StandardScaler()
        if num_features:
            self.model = self._build_model(num_input_candles, num_features)

    def _build_model(self, num_input_candles, num_features):
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=(num_input_candles, num_features)),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(64)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def load_and_preprocess(self, filename="GBPUSD_1h.csv"):
        data_file = self.base_path / "Data" / filename
        if not data_file.exists():
            return None, None

        data = pd.read_csv(data_file)

        # Standardize columns for IndicatorEngine
        if 'Volume' in data.columns and 'Vol' not in data.columns:
            data.rename(columns={'Volume': 'Vol'}, inplace=True)
        if 'TickVolume' not in data.columns and 'Vol' in data.columns:
            data['TickVolume'] = data['Vol']
        if 'Spread' not in data.columns:
            data['Spread'] = 0

        # Add indicators (assumes Collecting_Data/indicators.py is available)
        sys.path.append(str(self.base_path / "Collecting_Data"))
        from indicators import IndicatorEngine
        engine = IndicatorEngine(dropna=True)
        data = engine.calculate(data)

        feature_cols = [c for c in data.columns if c not in ['DTYYYYMMDD', '<Time>', 'Datetime', 'Classification', 'Binary_Label', 'Multi_Label', 'Pivot_Label', 'Price_Change', 'Peak', 'Trough']]
        self.num_features = len(feature_cols)

        data['Price_Change'] = data['Close'].shift(-1) - data['Close']
        data['Binary_Label'] = (data['Price_Change'] > 0).astype(int)
        data.dropna(inplace=True)

        # Scaling will be done in train() to avoid leakage

        X, y = [], []
        for i in range(len(data) - self.num_input_candles):
            X.append(data.iloc[i : i + self.num_input_candles][feature_cols].values)
            y.append(data.iloc[i + self.num_input_candles]['Binary_Label'])

        return np.array(X), np.array(y)

    def train(self, X, y, epochs=30, batch_size=64):
        train_split = int(len(X) * 0.8)

        # Scaling correction: Fit on train, transform both
        # X is (samples, window, features)
        X_train_raw = X[:train_split]
        X_test_raw = X[train_split:]

        # Flatten for scaling
        n_train, n_window, n_features = X_train_raw.shape
        X_train_flat = X_train_raw.reshape(-1, n_features)
        X_train_scaled = self.scaler.fit_transform(X_train_flat).reshape(n_train, n_window, n_features)

        n_test = X_test_raw.shape[0]
        X_test_flat = X_test_raw.reshape(-1, n_features)
        X_test_scaled = self.scaler.transform(X_test_flat).reshape(n_test, n_window, n_features)

        X_train, y_train = X_train_scaled, y[:train_split]
        X_test, y_test = X_test_scaled, y[train_split:]

        if self.model is None:
            self.model = self._build_model(self.num_input_candles, self.num_features)

        print("Training Binary LSTM...")
        history = self.model.fit(X_train, y_train, validation_split=0.1,
                                 epochs=epochs, batch_size=batch_size, verbose=1)

        eval_metrics = self.model.evaluate(X_test, y_test)
        return history, eval_metrics

    def save_model(self, filename='lstm_binary_classifier.h5'):
        self.model.save(self.base_path / filename)

if __name__ == "__main__":
    classifier = BinaryLSTMClassifier()
    X, y = classifier.load_and_preprocess()
    if X is not None:
        classifier.train(X, y)
        classifier.save_model()
