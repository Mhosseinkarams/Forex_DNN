#!/bin/python3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Use project root relative paths
base_path = Path(__file__).resolve().parent.parent.parent
data_file = base_path / "Data" / "GBPUSD_1h_preprocessed.csv"

# Read the data
try:
    data = pd.read_csv(data_file)
except FileNotFoundError:
    print(f"Error: Data file {data_file} not found. Please run preprocessing first.")
    exit(1)

# Features: All except labels and price change
drop_cols = ['Binary_Label', 'Multi_Label', 'Price_Change']
x = data.drop(drop_cols, axis=1)
# Ensure only numeric features
x = x.select_dtypes(include=[np.number])
y = data['Multi_Label']

# Split data (shuffle=False for time-series)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Build Multi-class Classifier Model (5 classes)
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax') # Softmax for multi-class classification
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training Multi-class Classifier (Direction + Size)...")
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=32,
                    epochs=50,
                    verbose=1)

# Evaluate
eval_metrics = model.evaluate(X_test, y_test)
print('\nTest Evaluation:', dict(zip(model.metrics_names, eval_metrics)))

# Save model
model_path = base_path / 'forex_multi_classifier.h5'
model.save(model_path)
print(f"Model saved to {model_path}")
