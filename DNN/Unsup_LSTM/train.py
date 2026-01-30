import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
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

# Readin data from the csv file
data_file = data_dir / 'GBPUSD_1h.csv'
try:
    data = pd.read_csv(data_file)
except FileNotFoundError:
    print(f"Error: {data_file} not found.")
    exit(1)

# Add indicators
data = add_technical_indicators(data)

# Features to use (numeric only)
features = data.select_dtypes(include=[np.number]).values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
features_normalized = scaler.fit_transform(features)

# Function to create sequences
def create_sequences(dataset, sequence_length):
    sequences = []
    for i in range(len(dataset) - sequence_length):
        sequence = dataset[i:i+sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

sequence_length = 24
sequences = create_sequences(features_normalized, sequence_length)
num_features = features.shape[1]

X_train = sequences

# Build the LSTM autoencoder model
model = Sequential([
    LSTM(units=128, activation='relu', input_shape=(sequence_length, num_features)),
    RepeatVector(sequence_length),
    LSTM(units=128, activation='relu', return_sequences=True),
    TimeDistributed(Dense(units=num_features))
])
model.compile(optimizer='adam', loss='mse')

# Train the autoencoder
model.fit(X_train, X_train, epochs=30, batch_size=32, verbose=1)

# Calculate the reconstruction error
reconstructed_sequences = model.predict(X_train)
mse = np.mean(np.square(X_train - reconstructed_sequences))
print(f"Mean Squared Error on Training Data: {mse}")
