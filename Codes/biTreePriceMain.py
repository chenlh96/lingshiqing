# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:46:08 2019

@author: Leheng Chen
"""

from binomialTreePricer import asianOptionBinomialTree
import pandas as pd
import numpy as np

data_path = '../Cleaned Data/ulyPrice.csv'
assets = pd.read_csv(data_path)
init_asset_price = assets.tail()
init_asset_price = init_asset_price.interpolate(method='linear',axis=0).tail(1)
test = init_asset_price.iloc[0, 1:]



treePricer = asianOptionBinomialTree(100, 0.002, 0.083, 3, 0.05)
for t in test:
    if not type(t) == str:
        price = treePricer.getOptionPrice(t, t - 1)
        print(price)
