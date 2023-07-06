import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Load the data from XAUUSD.csv
data = pd.read_csv('GBPUSD_1h.csv')

# Normalize the price data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close']].values)

# Create a function to label the next candle based on price change
def label_next_candle(current_close, current_open):
    price_change =  current_close - current_open
    # if price_change > 0.02:
    #     return 2
    # elif price_change > 0.005:
    #     return 1
    # elif price_change < -0.02:
    #     return -2
    # elif price_change < -0.005:
    #     return -1
    # else:
    #     return 0
    if price_change > 0:
        return 1
    elif price_change < 0:
        return -1
    else:
        return 0
# Add the classification column to the data
data['Classification'] = [label_next_candle(current_close, current_open)
                          for current_close, current_open in zip(data['Close'], data['Open'])]

# save the data to a csv file
data.to_csv('GBPUSD_1h_preprocessed.csv', index=False)
