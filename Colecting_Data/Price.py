#!/bin/python3
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from datetime import datetime as dt
import time
import pytz

# Set the ticker as 'XAUUSD=X'
ticker = 'GBPUSD=X'

# Set the end date as the current date
tz = pytz.timezone("America/New_York")
end_date = tz.localize(dt.now())
# Function to continuously update and save data
def update_data():
    while True:
        # Create an empty DataFrame
        forex_data = pd.DataFrame()
        
        # Fetch data for each interval and append to the DataFrame
        data = yf.download(ticker, period='2y', interval='1h', auto_adjust=True)
        forex_data = pd.concat([forex_data, data])
        
        # Save the data to the file
        forex_data.to_csv('/home/mhossein/my_projects/Forex_DNN/Data/GBPUSD_1h.csv')
        
        # Wait for 5 minutes before the next update
        time.sleep(300)

# Call the function to start updating and saving data
update_data()
