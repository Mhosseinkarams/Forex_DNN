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

# Set up paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
data_dir = project_root / "Data"

# Add Colecting_Data to sys.path
sys.path.append(str(project_root / "Colecting_Data"))
try:
    from utils import add_technical_indicators
except ImportError:
    print("Warning: utils not found.")
    add_technical_indicators = lambda x: x

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Load and preprocess your input data
data_file = data_dir / 'GBPUSD_1d_2.csv'
try:
    data = pd.read_csv(data_file)
except FileNotFoundError:
    print(f"Error: {data_file} not found.")
    exit(1)

# Add indicators
data = add_technical_indicators(data)
feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
features = data[feature_cols].values

# Normalize the data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

sequence_length = 24

# Create sequences
sequences = []
next_closes = []
if 'Close' in feature_cols:
    close_idx = feature_cols.index('Close')
else:
    close_idx = 3

for i in range(len(scaled_features) - sequence_length - 1):
    seq = scaled_features[i:i + sequence_length]
    target = scaled_features[i + sequence_length + 1][close_idx]
    sequences.append(seq)
    next_closes.append(target)

X = np.array(sequences)
y = np.array(next_closes)

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

train_gan()

# Prediction example (on normalized data)
test_predictions = lstm_model.predict(X_test)
print("Predictions on test data complete.")
