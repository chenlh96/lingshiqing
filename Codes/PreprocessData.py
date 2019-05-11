#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:22:55 2019

@author: lueshen
"""

import pandas as pd

def process_data(filename):
    filepath = "../Underlying Data/" + filename;
    df = pd.read_csv(filepath, sep=',', header=0)
    df.index = pd.to_datetime(df['Date'])
    df = df[['Price']]
    df.columns = [filename[:-28]]
    return df

import os

path = "../Underlying Data/"

files = []

for r, d, f in os.walk(path):
    for file in f:
        if '.csv' in file:
            files.append(file)

dataList = []

for file in files:
    dataList.append(process_data(file))

data = pd.concat(dataList, sort=True, axis = 1)

data.to_csv(os.path.join(path, 'Underlying Data.csv'))

# Load underlying data from git
df_uly = pd.read_csv("../Underlying Data/Underlying Data.csv",sep=',')

# Preprocess dataframe, set up index, fill nan with latest previous values
df_uly.index = df_uly['Date']
df_uly = df_uly.drop('Date', axis = 1)
df_uly = df_uly.fillna(method='ffill')

# Load option data from git
df_opt = pd.read_csv("../Option Price Data/Option Data.csv",sep=',')
df_opt.columns = ['Start Date','Maturity Date','Strike','Put','Call','Underlying']
df_opt['Maturity Date'] = pd.to_datetime(df_opt['Maturity Date'])
df_opt['Start Date'] = pd.to_datetime('2019-3-25')
