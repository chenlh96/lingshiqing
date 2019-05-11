# -*- coding: utf-8 -*-
"""
Created on Fri May  3 19:38:56 2019

@author: User
"""

# Import necessary library
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
from arch import arch_model

# =============================================================================
# Data Preprocessing
# =============================================================================

# Load data from git
df = pd.read_csv("../Underlying Data/Underlying Data.csv",sep=',')

# Preprocess dataframe, set up index, fill nan with forward values
df.index = df['Date']
df = df.drop('Date', axis = 1)
df = df.fillna(method='ffill')

# =============================================================================
# Function to get best GARCH model
# =============================================================================
def _get_best_model(priceData, pLimit, oLimit, qLimit):
    # To do: add model checking with white noise using ljungbox test
    best_aic = np.inf
    best_order = None
    best_mdl = None

    for pValue in range(pLimit):
        for oValue in range(oLimit):
            for qValue in range(qLimit):
                try:
                    tmp_mdl = arch_model(y = priceData,
                                         p = pValue,
                                         o = oValue,
                                         q = qValue,
                                         dist = 'Normal').fit()
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = [pValue, oValue, qValue]
                        best_mdl = tmp_mdl
                except:
                    continue
    return best_aic, best_order, best_mdl


# =============================================================================
# Main part to retrieve results
# =============================================================================
pLimitInput = 5
oLimitInput = 5
qLimitInput = 5

mlRes = {}
for column in df.columns:
    TS = np.log(df[column]) - np.log(df[column].shift(1))
    TS = TS[np.isnan(TS) == False]
    mlRes[column] = _get_best_model(TS, pLimitInput, oLimitInput, qLimitInput)
    

column = df.columns[0]
TS = (np.log(df[column]) - np.log(df[column].shift(1))) * 100
TS = TS[np.isnan(TS) == False]
mlRes[column] = _get_best_model(TS, pLimitInput, oLimitInput, qLimitInput)