#!/bin/python3

import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys

# Set up paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
data_dir = project_root / "Data"

# Add current dir to sys.path for utils
sys.path.append(str(current_dir))
try:
    from utils import add_technical_indicators
except ImportError:
    print("Warning: utils not found. Technical indicators will not be added.")
    add_technical_indicators = lambda x: x

while True:
    # Define the path to the input .prn file (Absolute path as requested by MT5 setup)
    input_file = Path('/home/mhossein/.wine/drive_c/Program Files/RoboForex - MetaTrader 5/MQL5/Files/GBPUSD-H1.prn')

    # Define the path to the output .csv file
    output_file = data_dir / 'GBPUSD_1h_2.csv'

    try:
        # Open the .prn file and read the data
        if not input_file.exists():
            print(f"Input file {input_file} not found. Waiting...")
            time.sleep(60)
            continue

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

        # Create DataFrame
        data = pd.DataFrame(rows)

        # Add technical indicators
        data = add_technical_indicators(data)

        # Save the DataFrame as a .csv file
        data.to_csv(output_file, index=False)
        print(f"******************* Done: {time.ctime()} *******************")
    except Exception as e:
        print(f"An error occurred: {e}")

    time.sleep(900)
