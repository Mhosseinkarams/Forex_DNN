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

# Use project root relative paths
base_path = Path(__file__).resolve().parent.parent.parent
data_path = base_path / "Data"

# load data
data_file = data_path / "GBPUSD_1h_pivot.csv"
try:
    data = pd.read_csv(data_file)
except FileNotFoundError:
    print(f"Error: {data_file} not found. Please run preproc_pivot.py first.")
    exit(1)

# Features to use
feature_cols = [c for c in data.columns if c not in ['DTYYYYMMDD', '<Time>', 'Datetime', 'Classification', 'Binary_Label', 'Multi_Label', 'Peak', 'Trough', 'Pivot_Label']]

# Normalize
scaler = StandardScaler()
data[feature_cols] = scaler.fit_transform(data[feature_cols])

# Parameters
num_input_candles = 48
num_features = len(feature_cols)

# Prepare sequences
X = []
y = []
for i in range(len(data) - num_input_candles):
    X.append(data.iloc[i : i + num_input_candles][feature_cols].values)
    y.append(data.iloc[i + num_input_candles]['Pivot_Label'])

X = np.array(X)
y = to_categorical(np.array(y), num_classes=3)

# Split (Temporal)
train_split = int(len(X) * 0.8)
val_split = int(len(X) * 0.9)

X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

# Build Bidirectional LSTM Model for Pivot Prediction
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(num_input_candles, num_features)),
    BatchNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(64)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax') # Classes: 0 (None), 1 (Peak), 2 (Trough)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training Pivot LSTM Classifier (Trend Change Detection)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Save
model_path = base_path / 'lstm_pivot_classifier.h5'
model.save(model_path)
print(f"Model saved to {model_path}")
