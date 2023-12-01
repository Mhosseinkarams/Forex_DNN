#!/bin/python3

import numpy as np
import pandas as pd
import tensorflow as tf
import time
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler


#Open model
model = tf.keras.models.load_model('multiclassifier_2.h5')

#load data
data_1h = pd.read_csv("/home/mhossein/my_projects/Forex_DNN/Data/GBPUSD_1h_2.csv")
# data_1d = pd.read_csv("/home/mhossein/my_projects/Forex_DNN/Data/GBPUSD_1d.csv")

#nomalize data
scaler = StandardScaler()
scaler.fit(data_1h[[ 'High', 'Low','Close']])
data_1h[[ 'High', 'Low','Close']] = scaler.transform(data_1h[[ 'High', 'Low','Close']])
# Define the number of past hourly candles to consider as input
num_input_candles_1h =  10

# Create the input data
input_data_1h = []

# Iterate over the data to create input and output samples
for i in range(len(data_1h) - num_input_candles_1h + 1):
    input_candles = data_1h.iloc[i:i + num_input_candles_1h][[ 'High', 'Low','Close','Vol']].values
    input_data_1h.append(input_candles)
    
# Convert the input data to numpy arrays
input_data_1h = np.array(input_data_1h)

# Plot the model structure
plot_model(model, to_file='model.png', show_shapes=True)
# print(input_data_1h[-100],input_data_1h[-10])

#predict
# while True:
pred = model.predict(input_data_1h[-20:-3])
print(pred[-2]) 
n=0
print('*********************************************************')
for i in range(len(pred)):
    n+=1
    # print("Test #", n)
    m=0
    # for l in range(len(pred[i])):
        # print('--------------------------------------------------')
    m+=1
        # print("Prediction #", m)
        # Find the index of the highest probability
    predicted_class_index = np.argmax(pred[i])
    # print("Probability:", pred[i][l])

        # Map the index to the class label
    if predicted_class_index == 0:
        predicted_class = "Neutral"
    elif predicted_class_index == 1:
        predicted_class = "strong Down"
    elif predicted_class_index == 2:
        predicted_class = "Down"
    elif predicted_class_index == 3:
        predicted_class = "Up"
    elif predicted_class_index == 4 :
        predicted_class = "strong Up"
    else:
        predicted_class = "Unknown"
    # if predicted_class != 'Up':
    print("Predicted Class:", predicted_class)
    # time.sleep(900)