#!/bin/python3

import pandas as pd
import time
while True:
    # Define the path to the input .prn file
    input_file = '/home/mhossein/.wine/drive_c/Program Files/RoboForex - MetaTrader 5/MQL5/Files/GBPUSD-H1.prn'

    # Define the path to the output .csv file
    output_file = 'Data/GBPUSD_1h_2.csv'

    # Open the .prn file and read the data
    with open(input_file, 'r') as f:
        lines = f.readlines()

    #drop the first row of the file
    lines = lines[1:]

    # Create a list to store the converted data rows
    rows = []

    # Iterate over the lines of the .prn file and extract the data
    for line in lines:
        # Split the line into columns based on the comma separator
        columns = line.strip().split(',')

        if len(columns) < 7:
            continue

        # Extract and convert data
        rows.append({
            'DTYYYYMMDD': columns[0],
            '<Time>': columns[1],
            'Open': float(columns[2]),
            'High': float(columns[3]),
            'Low': float(columns[4]),
            'Close': float(columns[5]),
            'Vol': float(columns[6])
        })

    # Create DataFrame from all rows at once
    data = pd.DataFrame(rows)

    # Save the DataFrame as a .csv file
    data.to_csv(output_file, index=False)
    print("*************************************Done*************************************************")
    time.sleep(900)