#!/bin/python3

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
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
    print("Error: utils not found.")
    exit(1)

# load data
data_file = data_path / "GBPUSD_1h.csv"
try:
    data = pd.read_csv(data_file)
except FileNotFoundError:
    print(f"Error: {data_file} not found.")
    exit(1)

# Add indicators
data = add_technical_indicators(data)

# Features to use
feature_cols = [c for c in data.columns if c not in ['DTYYYYMMDD', '<Time>', 'Datetime', 'Classification', 'Binary_Label', 'Multi_Label', 'Pivot_Label', 'Price_Change', 'Peak', 'Trough']]

# Normalize
scaler = StandardScaler()
data[feature_cols] = scaler.fit_transform(data[feature_cols])

# Parameters
num_input_candles = 48
num_features = len(feature_cols)

# Labels
data['Price_Change'] = data['Close'].shift(-1) - data['Close']
data['Binary_Label'] = (data['Price_Change'] > 0).astype(int)
data.dropna(inplace=True)

# Prepare sequences
X = []
y = []
for i in range(len(data) - num_input_candles):
    X.append(data.iloc[i : i + num_input_candles][feature_cols].values)
    y.append(data.iloc[i + num_input_candles]['Binary_Label'])

X = np.array(X)
y = np.array(y)

# Temporal Split
train_split = int(len(X) * 0.8)
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Build Model
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

print("Training Binary LSTM...")
model.fit(X_train, y_train, validation_split=0.1, epochs=30, batch_size=64,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

# Save
model.save(base_path / 'lstm_binary_classifier.h5')

# --- Risk-Aware Backtest ---
print("\n--- Risk-Aware Backtest (Test Set) ---")
preds = model.predict(X_test)
prices = data.iloc[train_split + num_input_candles:]['Close'].values

initial_balance = 1000
balance = initial_balance
spread = 0.0002
commission = 0.00005
confidence_threshold = 0.65
tp_pips = 0.0020
sl_pips = 0.0010

trades = 0
wins = 0

for i in range(len(preds)):
    prob = preds[i][0]
    current_price = prices[i]

    # Simple strategy: Enter if confidence is high
    if prob > confidence_threshold or prob < (1 - confidence_threshold):
        trades += 1
        direction = 1 if prob > confidence_threshold else -1

        # Entry cost
        balance -= initial_balance * commission
        entry_price = current_price + (spread * direction)

        # Check next few hours for SL or TP
        # (Very simplified backtest)
        result = 0
        for j in range(i + 1, min(i + 24, len(prices))):
            price_at_j = prices[j]
            change = (price_at_j - entry_price) * direction

            if change >= tp_pips:
                result = tp_pips
                wins += 1
                break
            elif change <= -sl_pips:
                result = -sl_pips
                break

        # Add result to balance (simplified sizing: 1 lot = initial_balance)
        balance += result * initial_balance

print(f"Total Trades: {trades}")
print(f"Win Rate: {wins/trades*100:.2f}%" if trades > 0 else "No trades")
print(f"Final Balance: {balance:.2f}")
print(f"Total Return: {(balance - initial_balance)/initial_balance*100:.2f}%")
