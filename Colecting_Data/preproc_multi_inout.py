#!/bin/python3

import pandas as pd
import numpy as np

#load data
data_1h = pd.read_csv("/home/mhossein/My_Projects/Forex_DNN/Data/GBPUSD_1h.csv")

# Define the number of past hourly candles to consider as input
num_input_candles_1h = 100 

# Define the number of future hourly candles to predict
num_output_candles = 5

# Create the input data
input_data_1h = []
output_data = []

# Iterate over the data to create input and output samples
for i in range(len(data_1h) - num_input_candles_1h - num_output_candles + 1):
    input_candles = data_1h.iloc[i:i + num_input_candles_1h].values
    output_candles = data_1h.iloc[i + num_input_candles_1h:i + num_input_candles_1h + num_output_candles][['Open', 'Close']].values
    
    # Calculate the price differences
    price_diffs = output_candles[:, 1] - output_candles[:, 0]
    
    # Encode the output data based on the price differences
    output = np.where(price_diffs > 0.0005, 1, np.where(price_diffs < -0.0005, -1, 0))
    
    input_data_1h.append(input_candles)
    output_data.append(output)
    


# Convert the lists to arrays
input_data_1h = np.array(input_data_1h)
output_data = np.array(output_data)
#save inputs and outputs in different files
np.savez_compressed("/home/mhossein/My_Projects/Forex_DNN/Data/GBPUSD_1h_multi_inout.npz", input_data_1h=input_data_1h)
np.savez_compressed("/home/mhossein/My_Projects/Forex_DNN/Data/GBPUSD_1h_multi_out.npz", output_data=output_data)