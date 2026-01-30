#!/bin/python3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Normalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from pathlib import Path

# Use project root relative paths
base_path = Path(__file__).resolve().parent.parent.parent
data_file = base_path / "Data" / "GBPUSD_1h_preprocessed.csv"

#Read the data
try:
    data = pd.read_csv(data_file)
except FileNotFoundError:
    print(f"Error: Data file {data_file} not found. Please run preprocessing first.")
    exit(1)

# Use all columns except Classification as features
x = data.drop(['Classification'], axis=1)
# Drop non-numeric columns if any (like Datetime)
x = x.select_dtypes(include=[np.number])
y = data['Classification']

#Split the data to train, test and validation sets (shuffle=False to prevent data leakage in time series)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

input_dim = X_train.shape[1]

#Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

# Train the model
history = model.fit(X_train,
             y_train,
             validation_data=(X_val, y_val),
             batch_size=32,
             epochs=50,
             verbose=1)

# Evaluate the model on the test set
eval_metrics = model.evaluate(X_test, y_test)

# Print the evaluation metrics
print('Evaluation metrics:', dict(zip(model.metrics_names, eval_metrics)))
model.save(base_path / 'forex_classifier_2.h5')
