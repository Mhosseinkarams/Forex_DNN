import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

# Readin data from the csv file
data = pd.read_csv('Data/GBPUSD_1h.csv')

# Extract the 'close' column
prices = data['Close'].values.reshape(-1, 1)

# Normalize the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
prices_normalized = scaler.fit_transform(prices)

# Function to create sequences for training the autoencoder
def create_sequences(dataset, sequence_length):
    sequences = []
    for i in range(len(dataset) - sequence_length):
        sequence = dataset[i:i+sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

# Set the sequence length (number of time steps in each input sequence)
sequence_length = 10

# Create sequences for training
sequences = create_sequences(prices_normalized, sequence_length)

# Reshape input data to be (samples, time steps, features)
X_train = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)

# Build the LSTM autoencoder model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(sequence_length, 1)))
model.add(RepeatVector(sequence_length))
model.add(LSTM(units=50, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(units=1)))
model.compile(optimizer='adam', loss='mse')

# Train the autoencoder
model.fit(X_train, X_train, epochs=50, batch_size=32, verbose=2)

# Use the trained autoencoder to reconstruct the input sequences
reconstructed_sequences = model.predict(X_train)

# Calculate the reconstruction error (mean squared error)
mse = np.mean(np.square(X_train - reconstructed_sequences))
print(f"Mean Squared Error on Training Data: {mse}")


# Example: If reconstruction error is above a certain threshold, consider it an anomaly or trend
threshold = 0.02
anomalies = (mse > threshold).astype(int)
print("Anomalies or Trends Detected:", anomalies)

# Use the trained autoencoder to reconstruct the input sequences
reconstructed_sequences = model.predict(X_train)

# Calculate the reconstruction error (mean squared error)
mse = np.mean(np.square(X_train - reconstructed_sequences))
print(f"Mean Squared Error on Training Data: {mse}")
