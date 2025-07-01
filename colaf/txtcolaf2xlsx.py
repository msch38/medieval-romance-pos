# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 17:42:37 2025

"""
import pandas as pd 

file_path = 'cat.txt'

data = []
with open(file_path, 'r') as file:
    header = file.readline().strip().split('\t')
    for line in file:
        row = line.strip().split('\t')
        data.append(row)


dfc = pd.DataFrame(data, columns=header)

dfc.to_excel("cat_colaf.xlsx",index=False)