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
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from pathlib import Path

# Use project root relative paths
base_path = Path(__file__).resolve().parent.parent.parent
data_path = base_path / "Data"

#load data
data_1h = pd.read_csv(data_path / "GBPUSD_1d_2.csv")
test_data = pd.read_csv(data_path / "GBPUSD_1h_2.csv")

#nomalize data
scaler = StandardScaler()
scaler.fit(data_1h[[ 'Open','High', 'Low','Close']])
data_1h[[ 'Open','High', 'Low','Close']] = scaler.transform(data_1h[[ 'Open','High', 'Low','Close']])
scaler.fit(test_data[[ 'Open','High', 'Low','Close']])
test_data[[ 'Open','High', 'Low','Close']] = scaler.transform(test_data[[ 'Open','High', 'Low','Close']])
#Define the number of past hourly candles to consider as input
num_input_candles_1h = 10
# num_input_candles_1d = 100

# Define the number of future hourly candles to predict
num_output_candles = 1

# Create the input data
input_data_1h = []
# input_data_1d = []
output_data = []
test_input = []
test_output_data = []
out = to_categorical([0,1,2,3,4],num_classes=5)

# Iterate over the data to create input and output samples
for i in range(len(data_1h) - num_input_candles_1h - num_output_candles + 1):
    input_candles = data_1h.iloc[i:i + num_input_candles_1h][[ 'High', 'Low','Close' , 'Vol']].values
    output_candles = data_1h.iloc[i + num_input_candles_1h:i + num_input_candles_1h + num_output_candles][['Open','Close']]
    # Calculate the price differences
    price_diffs = (output_candles['Close'] - output_candles['Open']).values
    # Encode the output data based on the price differences
    output = np.where(price_diffs > 0.01, out[4],
                      np.where(price_diffs > 0.001, out[3],
                                np.where(price_diffs < -0.01 , out[1],
                                    np.where(price_diffs < -0.001, out[2] ,out[0]))))
    #append data to list
    input_data_1h.append(input_candles)
    output_data.append(output)
    
for i in range(len(test_data) - num_input_candles_1h - num_output_candles + 1):
    test_candles = test_data.iloc[i:i + num_input_candles_1h][[ 'High', 'Low','Close' , 'Vol']].values
    test_output_candles = test_data.iloc[i + num_input_candles_1h:i + num_input_candles_1h + num_output_candles][['Open','Close']]
    # Calculate the price differences
    test_price_diffs = (test_output_candles['Close'] - test_output_candles['Open']).values
    # Encode the output data based on the price differences
    test_output = np.where(test_price_diffs > 0.01, out[4],
                           np.where(test_price_diffs > 0.001, out[3],
                                    np.where(test_price_diffs < -0.01 , out[1],
                                            np.where(test_price_diffs < -0.001, out[2] ,out[0]))))
    test_input.append(test_candles)
    test_output_data.append(test_output)
   
# Convert the lists to arrays
input_data_1h = np.array(input_data_1h)
output_data = np.array(output_data)
test_output_data = np.array(test_output_data)
test_input = np.array(test_input)

#spllit test and validation data
h_test = test_input[:int(len(test_input)*0.5)]
o_test = test_output_data[:int(len(test_output_data)*0.5)]
h_val = test_input[int(len(test_input)*0.5):]
o_val = test_output_data[int(len(test_output_data)*0.5):]

#define model
input_1h = keras.Input(shape=(num_input_candles_1h, 4 ))
Flat_input = keras.layers.Flatten()(input_1h)
# Shared layers for both inputs
shared_layer1 = layers.Dense(114, activation='tanh')

# Process hourly candles
x = shared_layer1(Flat_input)

# # Additional layers for prediction
x = layers.Dense(56, activation='tanh')(x)
Y = [layers.Dense(32, activation='relu')(x),
     layers.Dense(32, activation='relu')(x),
     layers.Dense(32, activation='relu')(x),
     layers.Dense(32, activation='relu')(x),
     layers.Dense(32, activation='relu')(x)]
# Create the output layers
outputs = []
for i in range(num_output_candles):
    output = Dense(5, activation='softmax')(Y[i])
    outputs.append(output)

# Create the model
model = Model(inputs=input_1h, outputs=outputs)

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Plot the model structure
plot_model(model, to_file='model.png', show_shapes=True)

# Train the model and store the history
history = model.fit(input_data_1h, output_data ,
                    validation_data=(h_val, o_val),
                    epochs=400,
                    batch_size=32,
                    callbacks=tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        min_delta=0.001,
                        patience=10,
                        verbose=0,
                        mode='auto',
                        baseline=None,
                        restore_best_weights=True,
                        start_from_epoch=10
)
)

# #Plot the loss/epoch changes
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 1, 1)
# plt.plot(history.history['loss'])
# plt.title('Loss vs. Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')

# plt.tight_layout()
# plt.show()

# Evaluate the model on the test set
loss = model.evaluate(h_test, o_test)
print('evaluate loss:', loss)
#save model
model.save(base_path / 'multiclassifier_2.h5')

print(model.predict(h_test),o_test)