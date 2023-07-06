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
import keras_tuner as kt
import tensorflow_addons as tfa

#Read the data
data = pd.read_csv("GBPUSD_1h_preprocessed.csv")
# data = data.drop(['Datetime'], axis=1)
#Define sets of 100 days data
# x = []
# y = []
# for i in range(len(data)-101):
#     if (i+101<=len(data)-101): 
#         x.append(np.array(data.iloc[i:i+100]))
#         y.append(data.iloc[i+100]['Classification'])
# # Convert the training data to numpy arrays
# x = np.array(x)
# y = np.array(y)
x = data[['Open' , 'High' , 'Low' , 'Close' , 'Volume']]
y = data['Classification']
#Split the data to train, test and validation sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05,)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5,)

#Build the 
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='tanh', input_shape=(5,)),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
model.fit(X_train, 
             y_train,
             validation_data=(X_val, y_val),
             batch_size=8, 
             epochs=500,
             workers=10,
             use_multiprocessing=True)
#Get model summary
model.summary()

# Evaluate the model on the validation set
eval_metrics = model.evaluate(X_test, y_test)

# Print the evaluation metrics
print('Evaluation metrics:', model.metrics_names)
model.save('forex_classifier_2.h5')