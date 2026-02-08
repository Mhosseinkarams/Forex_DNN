#!/bin/python3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

class BinaryDNNClassifier:
    """
    A class for binary classification using a Dense Neural Network.
    Predicts directional movement (Up/Down).
    """

    def __init__(self, input_dim=None):
        self.base_path = Path(__file__).resolve().parent.parent.parent
        self.model = None
        self.scaler = StandardScaler()
        if input_dim:
            self.model = self._build_model(input_dim)

    def _build_model(self, input_dim):
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def load_data(self, filename="GBPUSD_1h_preprocessed.csv"):
        data_file = self.base_path / "Data" / filename
        if not data_file.exists():
            return None, None

        data = pd.read_csv(data_file)
        drop_cols = ['Binary_Label', 'Multi_Label', 'Price_Change']
        x = data.drop(drop_cols, axis=1)
        x = x.select_dtypes(include=[np.number])
        y = data['Binary_Label']
        return x, y

    def train(self, X, y, epochs=50, batch_size=32, val_size=0.1):
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_size, shuffle=False)

        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        if self.model is None:
            self.model = self._build_model(X_train.shape[1])

        print("Training Binary DNN Classifier...")
        history = self.model.fit(X_train, y_train, validation_split=0.1,
                                 epochs=epochs, batch_size=batch_size, verbose=1)

        eval_metrics = self.model.evaluate(X_test, y_test)
        return history, eval_metrics

    def save_model(self, filename='forex_binary_classifier.h5'):
        self.model.save(self.base_path / filename)

if __name__ == "__main__":
    classifier = BinaryDNNClassifier()
    X, y = classifier.load_data()
    if X is not None:
        classifier.train(X, y)
        classifier.save_model()
