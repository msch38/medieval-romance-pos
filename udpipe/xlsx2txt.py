# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 21:18:51 2025

"""

import pandas as pd
import re

# Load the Excel file
file_path = 'output_with_upos.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path)

# Extract the tokens from the 'form' column
tokens = df['form'].astype(str).tolist()

# Join tokens with space, add newline after sentence-ending punctuation
text = ''
sentence_endings = {'.', '!', '?'}

for token in tokens:
    text += token + ' '
    if token[-1] in sentence_endings:
        text += '\n'

# Clean up extra spaces/newlines
text = re.sub(r'\s+\n', '\n', text).strip()

# Save to a .txt file
output_path = 'output_text.txt'  # Change this if you want a different filename
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Text saved to {output_path}")
