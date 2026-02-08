#!/bin/python3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys

class PivotLSTMClassifier:
    """
    A class for pivot detection using LSTMs.
    Predicts major trend reversal points.
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
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_and_preprocess(self, filename="GBPUSD_1h.csv"):
        data_file = self.base_path / "Data" / filename
        if not data_file.exists():
            return None, None

        data = pd.read_csv(data_file)

        sys.path.append(str(self.base_path / "Colecting_Data"))
        from utils import TechnicalIndicators
        data = TechnicalIndicators.add_all_indicators(data)

        # Define pivots (simplified inline for load_and_preprocess if preproc_pivot wasn't used)
        close = data['Close'].values
        peaks = np.zeros(len(data))
        troughs = np.zeros(len(data))
        window = 24
        for i in range(window, len(data) - window):
            chunk = close[i-window : i+window+1]
            if close[i] == np.max(chunk): peaks[i] = 1
            if close[i] == np.min(chunk): troughs[i] = 1

        data['Peak'] = peaks
        data['Trough'] = troughs

        labels = np.zeros(len(data))
        horizon = 24
        for i in range(len(data) - horizon):
            if np.any(peaks[i+1 : i+horizon+1] == 1): labels[i] = 1
            elif np.any(troughs[i+1 : i+horizon+1] == 1): labels[i] = 2
        data['Pivot_Label'] = labels
        data = data.iloc[:-horizon]

        feature_cols = [c for c in data.columns if c not in ['DTYYYYMMDD', '<Time>', 'Datetime', 'Classification', 'Binary_Label', 'Multi_Label', 'Pivot_Label', 'Price_Change', 'Peak', 'Trough']]
        self.num_features = len(feature_cols)

        data[feature_cols] = self.scaler.fit_transform(data[feature_cols])

        X, y = [], []
        for i in range(len(data) - self.num_input_candles):
            X.append(data.iloc[i : i + self.num_input_candles][feature_cols].values)
            y.append(data.iloc[i + self.num_input_candles]['Pivot_Label'])

        return np.array(X), to_categorical(np.array(y), num_classes=3)

    def train(self, X, y, epochs=30, batch_size=64):
        train_split = int(len(X) * 0.8)
        X_train, y_train = X[:train_split], y[:train_split]
        X_test, y_test = X[train_split:], y[train_split:]

        if self.model is None:
            self.model = self._build_model(self.num_input_candles, self.num_features)

        print("Training Pivot LSTM...")
        history = self.model.fit(X_train, y_train, validation_split=0.1,
                                 epochs=epochs, batch_size=batch_size, verbose=1)

        eval_metrics = self.model.evaluate(X_test, y_test)
        return history, eval_metrics

    def save_model(self, filename='lstm_pivot_classifier.h5'):
        self.model.save(self.base_path / filename)

if __name__ == "__main__":
    classifier = PivotLSTMClassifier()
    X, y = classifier.load_and_preprocess()
    if X is not None:
        classifier.train(X, y)
        classifier.save_model()
