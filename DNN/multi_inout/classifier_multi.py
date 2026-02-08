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

# Features
feature_cols = [c for c in data.columns if c not in ['DTYYYYMMDD', '<Time>', 'Datetime', 'Classification', 'Binary_Label', 'Multi_Label', 'Pivot_Label', 'Price_Change', 'Peak', 'Trough']]

# Normalize
scaler = StandardScaler()
data[feature_cols] = scaler.fit_transform(data[feature_cols])

# Parameters
num_input_candles = 48
num_features = len(feature_cols)

# Labels
def label_multi(change):
    if change > 0.001: return 4    # Strong Buy
    elif change > 0.0002: return 3 # Buy
    elif change < -0.001: return 0 # Strong Sell
    elif change < -0.0002: return 1 # Sell
    else: return 2                 # Neutral

data['Price_Change'] = data['Close'].shift(-1) - data['Close']
data['Multi_Label'] = data['Price_Change'].apply(label_multi)
data.dropna(inplace=True)

# Sequences
X = []
y = []
for i in range(len(data) - num_input_candles):
    X.append(data.iloc[i : i + num_input_candles][feature_cols].values)
    y.append(data.iloc[i + num_input_candles]['Multi_Label'])

X = np.array(X)
y = to_categorical(np.array(y), num_classes=5)

# Split
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
    Dense(5, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='categorical_crossentropy', metrics=['accuracy'])

print("Training Multi-class LSTM...")
model.fit(X_train, y_train, validation_split=0.1, epochs=30, batch_size=64,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

# Save
model.save(base_path / 'lstm_multi_classifier.h5')

# --- Risk-Aware Backtest ---
print("\n--- Risk-Aware Backtest (Test Set) ---")
preds = model.predict(X_test)
prices = data.iloc[train_split + num_input_candles:]['Close'].values

initial_balance = 1000
balance = initial_balance
spread = 0.0002
commission = 0.00005

# Backtest strategy: Only trade on Strong signals (0 or 4)
trades = 0
wins = 0

for i in range(len(preds)):
    pred_class = np.argmax(preds[i])
    current_price = prices[i]

    if pred_class in [0, 4]:
        trades += 1
        direction = 1 if pred_class == 4 else -1

        # Costs
        balance -= initial_balance * commission
        entry_price = current_price + (spread * direction)

        # Fixed SL/TP based on ATR or pips
        tp_pips = 0.0030 if pred_class in [0, 4] else 0.0015
        sl_pips = 0.0015

        result = 0
        for j in range(i + 1, min(i + 48, len(prices))):
            price_at_j = prices[j]
            change = (price_at_j - entry_price) * direction

            if change >= tp_pips:
                result = tp_pips
                wins += 1
                break
            elif change <= -sl_pips:
                result = -sl_pips
                break

        balance += result * initial_balance

print(f"Total Trades: {trades}")
print(f"Win Rate: {wins/trades*100:.2f}%" if trades > 0 else "No trades")
print(f"Final Balance: {balance:.2f}")
print(f"Total Return: {(balance - initial_balance)/initial_balance*100:.2f}%")
