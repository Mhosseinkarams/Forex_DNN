#!/bin/python3

import pandas as pd
import time
while True:
    # Define the path to the input .prn file
    input_file = '/home/mhossein/.wine/drive_c/Program Files/RoboForex - MetaTrader 5/MQL5/Files/GBPUSD-H1.prn'

    # Define the path to the output .csv file
    output_file = '/home/mhossein/my_projects/Forex_DNN/Data/GBPUSD_1h_2.csv'

    # Open the .prn file and read the data
    with open(input_file, 'r') as f:
        lines = f.readlines()

    #drop the first row of the file
    lines = lines[1:]

    #Create an empty DataFrame to store the converted data
    data = pd.DataFrame(columns=['DTYYYYMMDD', '<Time>','Open', 'High', 'Low', 'Close', 'Vol'])

    # Iterate over the lines of the .prn file and extract the data
    for line in lines:
        # Split the line into columns based on the comma separator
        columns = line.strip().split(',')


    
        # Extract the data for each column
        dt = columns[0]
        t = columns[1]
        open_val = float(columns[2])
        high_val = float(columns[3])
        low_val = float(columns[4])
        close_val = float(columns[5])
        vol = float(columns[6])

        # Append the data as a new row to the DataFrame
        data = pd.concat([data, pd.DataFrame([[dt, t ,open_val, high_val, low_val, close_val, vol]], columns=data.columns )])

    # Save the DataFrame as a .csv file
    data.to_csv(output_file, index=False)
    print("*************************************Done*************************************************")
    time.sleep(900)