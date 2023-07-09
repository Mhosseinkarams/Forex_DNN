#!/bin/python3

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Normalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

#load data
data_1h = pd.read_csv("/home/mhossein/my_projects/Forex_DNN/Data/GBPUSD_1h.csv")
# data_1d = pd.read_csv("/home/mhossein/my_projects/Forex_DNN/Data/GBPUSD_1d.csv")

# Define the number of past hourly candles to consider as input
num_input_candles_1h = 3 
# num_input_candles_1d = 100

# Define the number of future hourly candles to predict
num_output_candles = 5

# Create the input data
input_data_1h = []
# input_data_1d = []
output_data = []

# Iterate over the data to create input and output samples
for i in range(len(data_1h) - num_input_candles_1h - num_output_candles + 1):
    input_candles = data_1h.iloc[i:i + num_input_candles_1h][['Open', 'High', 'Low','Close']].values
    output_candles = data_1h.iloc[i + num_input_candles_1h:i + num_input_candles_1h + num_output_candles][['Open', 'Close']].values
    
    # Calculate the price differences
    price_diffs = output_candles[:, 1] - output_candles[:, 0]
    
    # Encode the output data based on the price differences
    output = np.where(price_diffs > 0.0005, 1, np.where(price_diffs < -0.0005, -1, 0))
    
    input_data_1h.append(input_candles)
    output_data.append(output)
    
# for i in range(len(data_1d) - num_input_candles_1d - num_output_candles + 1):
#     input_candles = data_1h.iloc[i:i + num_input_candles_1d][['Open', 'High', 'Low','Close']].values
#     input_data_1d.append(input_candles)
    # print(input_data_1d)
# Convert the lists to arrays
input_data_1h = np.array(input_data_1h)
# input_data_1d = np.array(input_data_1d)
output_data = np.array(output_data)

#split data into train and test
# d_train = input_data_1d[:int(len(input_data_1d)*0.8)]
# d_test = input_data_1d[int(len(input_data_1d)*0.8):]
h_train = input_data_1h[:int(len(input_data_1h)*0.8)]
h_test = input_data_1h[int(len(input_data_1h)*0.8):]
o_train = output_data[:int(len(output_data)*0.8)]
o_test = output_data[int(len(output_data)*0.8):]

#define model
# input_1d = keras.Input(shape=(num_input_candles_1d, ))
input_1h = keras.Input(shape=(num_input_candles_1h, 4 ))

# Shared layers for both inputs
shared_layer1 = layers.Dense(64, activation='relu')
shared_layer2 = layers.Dense(32, activation='relu')

# Process daily candles
# x1 = shared_layer1(input_1d)
# x1 = shared_layer2(x1)

# Process hourly candles
x2 = shared_layer1(input_1h)
x2 = shared_layer2(x2)

# Concatenate the processed inputs
# x = layers.Concatenate()(x2)

# Additional layers for prediction
x = layers.Dense(32, activation='relu')(x2)
x = layers.Dense(16, activation='relu')(x)

# Create the output layers
outputs = []
for _ in range(num_output_candles):
    output = Dense(3, activation='softmax')(x)
    outputs.append(output)

# Create the model
model = Model(inputs=input_1h, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(h_train, [to_categorical(o_train[:, i] + 1 , num_classes=3 ) for i in range(num_output_candles)], epochs=10, batch_size=32)

# Evaluate the model on the test set
loss = model.evaluate(h_test, [to_categorical(o_test[:, i] + 1 , num_classes=3 ) for i in range(num_output_candles)])

#save model
model.save('multiclassifier_1.h5')

