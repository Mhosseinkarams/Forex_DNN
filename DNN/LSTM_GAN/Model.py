import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Reshape, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from pathlib import Path
import sys

class LSTM_GAN:
    """
    A Generative Adversarial Network with LSTM for Forex data synthesis and prediction.
    """

    def __init__(self, sequence_length=24, latent_dim=100, num_features=None):
        self.base_path = Path(__file__).resolve().parent.parent.parent
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.scaler = MinMaxScaler()

        self.generator = None
        self.discriminator = None
        self.gan = None
        self.lstm_model = None

        if num_features:
            self._build_all_models()

    def _build_generator(self):
        model = Sequential([
            Dense(128, input_dim=self.latent_dim),
            Dense(self.sequence_length * self.num_features, activation='tanh'),
            Reshape((self.sequence_length, self.num_features))
        ])
        return model

    def _build_discriminator(self):
        model = Sequential([
            Flatten(input_shape=(self.sequence_length, self.num_features)),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
        return model

    def _build_lstm_predictor(self):
        model = Sequential([
            LSTM(128, input_shape=(self.sequence_length, self.num_features)),
            Dense(1, activation='linear')
        ])
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def _build_all_models(self):
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.lstm_model = self._build_lstm_predictor()

        # Combine into GAN
        self.discriminator.trainable = False
        gan_input = Input(shape=(self.latent_dim,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        self.gan = Model(gan_input, gan_output)
        self.gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    def load_and_preprocess(self, filename='GBPUSD_1d_2.csv'):
        data_file = self.base_path / "Data" / filename
        if not data_file.exists():
            return None, None

        data = pd.read_csv(data_file)

        sys.path.append(str(self.base_path / "Colecting_Data"))
        from utils import TechnicalIndicators
        data = TechnicalIndicators.add_all_indicators(data)

        feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        features = data[feature_cols].values
        self.num_features = features.shape[1]

# Normalize the data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

        sequences = []
        next_closes = []
        close_idx = feature_cols.index('Close') if 'Close' in feature_cols else 3

        for i in range(len(scaled_features) - self.sequence_length - 1):
            sequences.append(scaled_features[i : i + self.sequence_length])
            next_closes.append(scaled_features[i + self.sequence_length + 1][close_idx])

        return np.array(sequences), np.array(next_closes)

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

num_features = features.shape[1]

# Build the LSTM model
lstm_model = Sequential([
    LSTM(128, input_shape=(sequence_length, num_features)),
    Dense(1, activation='linear')
])
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# Build the GAN
latent_dim = 100
generator = Sequential([
    Dense(128, input_dim=latent_dim),
    Dense(sequence_length * num_features, activation='tanh'),
    Reshape((sequence_length, num_features))
])

discriminator = Sequential([
    Flatten(input_shape=(sequence_length, num_features)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
discriminator.compile(loss='binary_crossentropy', optimizer='adam')

discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')

def train_gan(epochs=20, batch_size=64):
    for e in range(epochs):
        for _ in range(len(X_train) // batch_size):
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            generated_data = generator.predict(noise)
            real_data = X_train[np.random.randint(0, len(X_train), size=batch_size)]
            X_batch = np.concatenate([real_data, generated_data])
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X_batch, y_dis)
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y_gen)
        print(f"Epoch {e + 1}, D Loss: {d_loss}, G Loss: {g_loss}")

if __name__ == "__main__":
    gan = LSTM_GAN()
    X, y = gan.load_and_preprocess()
    if X is not None:
        gan.train_lstm(X, y, epochs=5)
        gan.train_gan(X, epochs=5)
