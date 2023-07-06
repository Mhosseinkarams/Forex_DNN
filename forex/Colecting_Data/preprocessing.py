import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Load the data from XAUUSD.csv
data = pd.read_csv('GBPUSD_1h.csv')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Create a function to label the next candle based on price change
def label_next_candle(current_close, next_close):
    price_change =  next_close - current_close
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
        return 2
    elif price_change < 0:
        return 1
    else:
        return 0
# Add the classification column to the data
data['Classification'] = [label_next_candle(current_close, next_close)
                          for current_close, next_close in zip(data['Close'], data['Close'].shift(-1))]
# save the data to a csv file
data.to_csv('GBPUSD_1h_preprocessed.csv', index=False)
