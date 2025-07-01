# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 06:45:31 2025

model output: json -> prepare for analysis; 
use glob to collect all outputs of a given dataset
"""

import pandas as pd
import os
import json
import glob

# Path to the folder with JSON files
folder_path = './'  # Change this to your folder path if needed

# Get all JSON files in the folder
json_files = glob.glob(os.path.join(folder_path, '*.json'))

# Loop through each JSON file
for json_file in json_files:
    try:
        # Read the JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Keep only 'word' and 'UPOS' columns
        df = df[['word', 'UPOS']]

        # Rename 'UPOS' to 'upos'
        df.rename(columns={'UPOS': 'upos'}, inplace=True)

        # Create Excel filename
        base_name = os.path.splitext(json_file)[0]
        excel_file = base_name + '.xlsx'

        # Save to Excel
        df.to_excel(excel_file, index=False)

        print(f"Converted: {json_file} -> {excel_file}")

    except Exception as e:
        print(f"Failed to process {json_file}: {e}")
