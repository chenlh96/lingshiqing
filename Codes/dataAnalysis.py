#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun April 12 15:53:00 2019

@author: lueshen
"""

import pandas as pd
import numpy as np
import statistics
import progressbar
from datetime import datetime
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")


# =============================================================================
# Repo Rates
# =============================================================================
repoRateDict = {
                'Crude Oil WTI':    0.01,
                'Ethanol':          0.01,
                'Gold':             0.01,
                'Natural Gas':      0.01,
                'Silver':           0.01
               }

# =============================================================================
# Utility functions
# =============================================================================
commodityCurrDict = {
                        'Crude Oil WTI':    0.01,
                        'Ethanol':          0.0001,
                        'Gold':             1,
                        'Natural Gas':      0.001,
                        'Silver':           0.01
                    }

def get_currency_divisor(commodity):
    return commodityCurrDict[commodity]

# =============================================================================
# 
# =============================================================================
# Load underlying data from git
df_uly = pd.read_csv("../Underlying Data/Underlying Data.csv", sep=',')

# Preprocess dataframe, set up index, fill nan with latest previous values
df_uly.index = pd.to_datetime(df_uly['Date']).dt.date
df_uly = df_uly.drop('Date', axis = 1)

# Load option data from git
df_opt = pd.read_csv("../Option Price Data/Option Data.csv", sep=',')

# Preprocess dataframe, convert dates, calculate days to maturity
df_opt.columns = ['Start Date','Maturity Date','Strike','Put','Call','Underlying']
df_opt['Maturity Date'] = pd.to_datetime(df_opt['Maturity Date']).dt.date
df_opt['Start Date'] = '2019-3-25'
df_opt['Start Date'] = pd.to_datetime(df_opt['Start Date']).dt.date
try:
    df_opt['Exp BusDays'] = np.busday_count(df_opt['Start Date'], df_opt['Maturity Date']) + 1
except:
    # Sometimes line above may not work, use method below as an alternative
    tmp_list = []
    for i in range(len(df_opt['Start Date'])):
        tmp_list.append(np.busday_count(df_opt['Start Date'][i], df_opt['Maturity Date'][i]) + 1)
    df_opt['Exp BusDays'] = tmp_list


masterObj = {}

ulyList = np.unique(df_opt['Underlying'])
for underlying in progressbar.progressbar(ulyList):
    tmp_uly = underlying[:-8]
    TS_uly = df_uly[tmp_uly].dropna()
    TS_logRt = (np.log(TS_uly) - np.log(TS_uly.shift(1))).dropna()*100
    masterObj[tmp_uly] = {
                            'Start Price':  TS_uly[-1],
                            'Log Return':   TS_logRt,
                            'Volatility':   statistics.stdev(TS_logRt)
                         }

tmp_uly = 'Crude Oil WTI'
oilLogRt = masterObj[tmp_uly]['Log Return']

plt.figure(figsize=(20,10))
plt.title('Crude Oil WTI Daily Return')
plt.xlabel('Time')
plt.ylabel('Daily Log Return')
plt.plot(oilLogRt)
plt.grid(True)
plt.show()

oilLogRt.mean()

am = arch_model(oilLogRt,mean = 'Zero', p=1, o=1, q=1)
res = am.fit(update_freq=5, disp='off')
print(res.summary())
len()
res.num_params
res.pvalues['mu']
res.aic
a = res.forecast(horizon=1000)
std = np.sqrt(a.residual_variance.iloc[-1].values)
std
plt.figure(figsize=(20,10))
plt.title('Crude Oil WTI Volatility Prediction')
plt.xlabel('BusDays after 20190322')
plt.ylabel('Volatility (%)')
plt.plot(std)
plt.grid(True)
plt.show()


wn = res.resid/res._volatility
[lbvalue, pvalue] = acorr_ljungbox(wn, lags = 20)
print('Ljung-Box Statistics: ', lbvalue[19])
print('p-value: ', pvalue[19])

len(res._volatility)

for i in range(1):
    print(i)

import numpy as np
a = np.inf
a * 1.1
