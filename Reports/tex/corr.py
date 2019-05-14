#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun April 12 15:53:00 2019

@author: jingqian
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("../Cleaned Data/Cleaned Underlying Data.csv", index_col = 0)
newCols = ['Crude Oil WTI','Natural Gas','Gasoline RBOB','Ethanol','Coal',
        'US Corn','US Wheat', 'US Soybean Meal','US Soybean Oil','US Soybeans',
        'Lean Hogs','Live Cattle', 'Gold','Silver','Platinum','Copper']


price = data[newCols]

corr_price = price.corr()
corr_logRt = price.corr()

for col1 in newCols:
    for col2 in newCols:
        tmpdf = data[[col1,col2]].dropna()
        corr_price[col1][col2] = tmpdf[[col1]].corrwith(tmpdf[col2]).values[0]
        logdf = data[[col1,col2]].dropna()
        logdf[col1] = (np.log(logdf[col1]) - np.log(logdf[col1].shift(1))) * 100
        logdf[col2] = (np.log(logdf[col2]) - np.log(logdf[col2].shift(1))) * 100
        logdf = logdf.dropna()
        corr_logRt[col1][col2] = logdf[[col1]].corrwith(logdf[col2]).values[0]
        

plt.subplots(figsize = (12, 12))
sns_plot = sns.heatmap(corr_price, annot=True, vmin= -1, vmax= 1, square=True, cmap="RdBu_r")
fig = sns_plot.get_figure()
fig.savefig('corr_price.jpg')

plt.subplots(figsize=(12, 12))
sns_plot = sns.heatmap(corr_logRt, annot=True, vmin= -1, vmax= 1, square=True, cmap="RdBu_r")
fig = sns_plot.get_figure()
fig.savefig('corr.jpg')