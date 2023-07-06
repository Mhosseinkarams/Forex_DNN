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
print(end_date)
# Set the intervals for updates
intervals = ['1d']
start_dates = ['2003-12-01' , '2005-12-01', '2007-12-01', '2009-12-01', '2011-12-01', '2013-12-01', '2015-12-01', '2017-12-01', '2019-12-01', '2020-12-01', '2022-12-01']
end_dates = ['2005-11-28', '2007-11-29', '2009-11-29', '2011-11-29', '2013-11-29', '2015-11-29', '2017-11-29', '2019-11-29', '2020-11-29', '2022-11-29', end_date]
# Function to continuously update and save data
def update_data():
    while True:
        # Create an empty DataFrame
        forex_data = pd.DataFrame()
        
        # Fetch data for each interval and append to the DataFrame
        data = yf.download(ticker, period='2y', interval='1h', auto_adjust=True)
        forex_data = pd.concat([forex_data, data])
        
        # Save the data to the file
        forex_data.to_csv('GBPUSD_1h.csv')
        
        # Wait for 5 minutes before the next update
        time.sleep(300)

# Call the function to start updating and saving data
update_data()
