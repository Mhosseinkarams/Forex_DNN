#!/bin/python3

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Normalization, Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from pathlib import Path
import sys

# Use project root relative paths
base_path = Path(__file__).resolve().parent.parent.parent
data_path = base_path / "Data"

# Add Colecting_Data to sys.path
sys.path.append(str(base_path / "Colecting_Data"))
try:
    from utils import add_technical_indicators
except ImportError:
    # If not found, define a basic version or handle error
    print("Warning: utils not found in sys.path. Ensure Colecting_Data/utils.py exists.")
    exit(1)

#load data
try:
    data_1h_path = data_path / "GBPUSD_1d_2.csv"
    test_data_path = data_path / "GBPUSD_1h_2.csv"
    data_1h = pd.read_csv(data_1h_path)
    test_data = pd.read_csv(test_data_path)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit(1)

# Add indicators
data_1h = add_technical_indicators(data_1h)
test_data = add_technical_indicators(test_data)

# Features to use
feature_cols = [c for c in data_1h.columns if c not in ['DTYYYYMMDD', '<Time>', 'Datetime', 'Classification']]

#normalize data
scaler = StandardScaler()
data_1h[feature_cols] = scaler.fit_transform(data_1h[feature_cols])
test_data[feature_cols] = scaler.transform(test_data[feature_cols])

#Define the number of past hourly candles to consider as input
num_input_candles_1h = 24
num_output_candles = 1
num_features = len(feature_cols)

# Create the input data
input_data_1h = []
output_data = []
test_input = []
test_output_data = []

# Iterate over the data to create input and output samples
for i in range(len(data_1h) - num_input_candles_1h - num_output_candles + 1):
    input_candles = data_1h.iloc[i:i + num_input_candles_1h][feature_cols].values
    output_candles = data_1h.iloc[i + num_input_candles_1h:i + num_input_candles_1h + num_output_candles][['Open','Close']]
    price_diff = (output_candles['Close'] - output_candles['Open']).values[0]

    if price_diff > 0.01: label = 4
    elif price_diff > 0.001: label = 3
    elif price_diff < -0.01: label = 1
    elif price_diff < -0.001: label = 2
    else: label = 0

    input_data_1h.append(input_candles)
    output_data.append(label)
    
for i in range(len(test_data) - num_input_candles_1h - num_output_candles + 1):
    test_candles = test_data.iloc[i:i + num_input_candles_1h][feature_cols].values
    test_output_candles = test_data.iloc[i + num_input_candles_1h:i + num_input_candles_1h + num_output_candles][['Open','Close']]
    test_price_diff = (test_output_candles['Close'] - test_output_candles['Open']).values[0]

    if test_price_diff > 0.01: test_label = 4
    elif test_price_diff > 0.001: test_label = 3
    elif test_price_diff < -0.01: test_label = 1
    elif test_price_diff < -0.001: test_label = 2
    else: test_label = 0

    test_input.append(test_candles)
    test_output_data.append(test_label)
   
input_data_1h = np.array(input_data_1h)
output_data = to_categorical(np.array(output_data), num_classes=5)
test_input = np.array(test_input)
test_output_data = to_categorical(np.array(test_output_data), num_classes=5)

split_idx = int(len(test_input)*0.5)
h_test = test_input[:split_idx]
o_test = test_output_data[:split_idx]
h_val = test_input[split_idx:]
o_val = test_output_data[split_idx:]

#define model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(num_input_candles_1h, num_features)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(input_data_1h, output_data,
                    validation_data=(h_val, o_val),
                    epochs=50,
                    batch_size=32,
                    callbacks=[tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )]
)

loss, accuracy = model.evaluate(h_test, o_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
model.save(base_path / 'multiclassifier_2.h5')
