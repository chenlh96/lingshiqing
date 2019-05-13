#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:22:55 2019

@author: lueshen
"""

import pandas as pd
import numpy as np
import statistics
import progressbar
from datetime import datetime
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")
# =============================================================================
# Define risk free rate, reference to US treasury yield curve as of 20190322
# https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/TextView.aspx?data=yieldYear&year=2019
# 1m, 2m, 3m, 6m, 1y, 2y, 3y, 5y, 7y, 10y, 20y, 30y
# =============================================================================
# Define risk free rate according to US
yieldCurveDict = {
                    '2019-04-22': 2.49,
                    '2019-05-22': 2.48,
                    '2019-06-22': 2.46,
                    '2019-09-22': 2.48,
                    '2020-03-22': 2.45,
                    '2021-03-22': 2.31,
                    '2022-03-22': 2.24,
                    '2024-03-22': 2.24,
                    '2026-03-22': 2.34,
                    '2029-03-22': 2.44,
                    '2039-03-22': 2.69,
                    '2049-03-22': 2.88
                 }

# Derive forward rates from US treasury yield curve
curvePoints = ['2019-03-22'] + list(yieldCurveDict.keys())

forwardCurveDict = {}
fwdCurveDict = {0:0}
for i in range(len(yieldCurveDict)):
    datePoint1 = curvePoints[i]
    datePoint2 = curvePoints[i + 1]
    busDateDiff1 = np.busday_count(curvePoints[0], datePoint1)
    busDateDiff2 = np.busday_count(curvePoints[0], datePoint2)
    if (datePoint1 == curvePoints[0]):
        forwardCurveDict[datePoint2] = yieldCurveDict[datePoint2]
        fwdCurveDict[busDateDiff2] = yieldCurveDict[datePoint2]
    else:
        yieldAtDate1 = yieldCurveDict[datePoint1]
        yieldAtDate2 = yieldCurveDict[datePoint2]
        forwardCurveDict[datePoint2] = float((yieldAtDate2 * busDateDiff2 - yieldAtDate1 * busDateDiff1) / (busDateDiff2 - busDateDiff1))
        fwdCurveDict[busDateDiff2] = float((yieldAtDate2 * busDateDiff2 - yieldAtDate1 * busDateDiff1) / (busDateDiff2 - busDateDiff1))

# Function to get risk free rate given a date (datetime.date object)
def getRiskFreeRateByDate(inputDate):
    for i in range(len(forwardCurveDict)):
        datePoint1 = datetime.strptime(curvePoints[i],'%Y-%m-%d').date()
        datePoint2 = datetime.strptime(curvePoints[i + 1],'%Y-%m-%d').date()
        if (inputDate >= datePoint1 and inputDate < datePoint2):
            return forwardCurveDict[curvePoints[i + 1]]
    return 0

# Function to get risk free rate given a business date count from 20190322
def getRiskFreeRate(dayCounts):
    dayCountPoints = list(fwdCurveDict.keys())
    for i in range(len(dayCountPoints)-1):
        dayCount1 = dayCountPoints[i]
        dayCount2 = dayCountPoints[i + 1]
        if (dayCounts >= dayCount1 and dayCounts < dayCount2):
            return fwdCurveDict[dayCount2]
    return 0

# =============================================================================
# convenience yield - storage cost
# =============================================================================
commodityYieldDict = {
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

# Get best GJR GARCH model for the log return series
def get_best_model(logRtSeries, pLimit, oLimit, qLimit, predictDays):
    best_bic   = np.inf
    best_order = None
    best_mdl   = None
    best_numParams = np.inf
    isZeroMean = False

    for pValue in range(pLimit + 1):
        for oValue in range(oLimit + 1):
            for qValue in range(qLimit + 1):
                isZeroMean = False
                try:
                    tmp_mdl = arch_model(y = logRtSeries,
                                         p = pValue,
                                         o = oValue,
                                         q = qValue,
                                         dist = 'Normal')
                    tmp_res = tmp_mdl.fit(update_freq=5, disp='off')
                    
                    # Remove mean if it's not significant
                    if tmp_res.pvalues['mu'] > 0.05:
                        isZeroMean = True
                        tmp_mdl = arch_model(y = logRtSeries,
                                             mean = 'Zero',
                                             p = pValue,
                                             o = oValue,
                                             q = qValue,
                                             dist = 'Normal')
                        tmp_res = tmp_mdl.fit(update_freq=5, disp='off')
                    
                    tmp_bic = tmp_res.bic
                    tmp_numParams = tmp_res.num_params
                    tmp_wn_test = tmp_res.resid / tmp_res._volatility
                    [lbvalue, pvalue] = acorr_ljungbox(tmp_wn_test, lags = 20)
                    
                    # Make sure the model pass Ljunbox Test, and fit the time series
                    if pvalue[19] >= 0.05:
                        if best_bic / tmp_bic > 1.05:
                            best_bic = tmp_bic
                            best_order = [pValue, oValue, qValue]
                            best_mdl = tmp_res
                        # Choose simpler model
                        elif tmp_bic <= best_bic and tmp_numParams <= best_numParams:
                            best_bic = tmp_bic
                            best_order = [pValue, oValue, qValue]
                            best_mdl = tmp_res
                except:
                    continue

    # Handle situations when all models don't pass Ljunbox Test
    if (best_mdl == None):
        tmp_mdl = arch_model(y = logRtSeries,
                             p = 1,
                             o = 1,
                             q = 1,
                             dist = 'Normal')
        best_mdl = tmp_mdl.fit(update_freq=5, disp='off')
        
        # Remove mean if it's not significant
        if best_mdl.pvalues['mu'] > 0.05:
            isZeroMean = True
            tmp_mdl = arch_model(y = logRtSeries,
                                 mean = 'Zero',
                                 p = 1,
                                 o = 1,
                                 q = 1,
                                 dist = 'Normal')
            best_mdl = tmp_mdl.fit(update_freq=5, disp='off')
        
        best_bic = best_mdl.bic
        best_order = [1, 1, 1]
        
    
    # Test for first 20-lag
    wn_test = best_mdl.resid / best_mdl._volatility
    [lbvalue, pvalue] = acorr_ljungbox(wn_test, lags = 20)
    
    output = {}
    output['Zero Mean Model'] = isZeroMean
    output['Best BIC'] = best_bic
    output['Best Order'] = best_order
    output['Best Model'] = best_mdl
    volForecasts = best_mdl.forecast(horizon=predictDays)
    output['Vol Predictions'] = np.sqrt(volForecasts.residual_variance.iloc[-1].values)
    output['Ljunbox Test Statistics'] = lbvalue[19]
    output['Ljunbox Test pvalue'] = pvalue[19]

    return output

# Get affine garch model [not implemented yet]
def get_affine_garch(logRtSeries):
    tmp_mdl = arch_model(y = logRtSeries,p = 1,q = 1,dist = 'Normal')
    tmp_res = tmp_mdl.fit(update_freq=5, disp='off')
    tmp_bic = tmp_res.aic

    output = {}
    output['Best AIC'] = tmp_bic
    output['Best Model'] = tmp_res

    return output

# Use GARCH vol to price Asian option (Monte Carlo)
def garchPricer(startPrice, strikePrice, garchModel, expBusDays, numPath):
    sumCallPrice = 0
    sumPutPrice = 0
    res = garchModel['Best Model']
    mu = 0
    if garchModel['Zero Mean Model'] == False:
        mu = res.params['mu']
    vol = garchModel['Vol Predictions']

    for i in range(numPath):
        ulyPrice = startPrice
        sumUlyPrice = 0
        dt = 1 / 252
        randomGenerator = np.random.normal(0, 1, expBusDays)
        discountRate = 0

        # Simulated one path of underlying price
        for j in range(expBusDays):
            zt = randomGenerator[j]
            rt = getRiskFreeRate(j) / 100
            dLogSt = mu + vol[j] / 100 * zt
            discountRate += rt
            ulyPrice *= np.exp(dLogSt)
            sumUlyPrice += ulyPrice

        avgUlyPrice = sumUlyPrice / expBusDays

        # True for call, false for put
        sumCallPrice += max(avgUlyPrice - strikePrice, 0) * np.exp(-discountRate * dt)
        sumPutPrice  += max(strikePrice - avgUlyPrice, 0) * np.exp(-discountRate * dt)

    output = {
                'Call': sumCallPrice / numPath,
                'Put':  sumPutPrice / numPath
             }

    return output

# Pricing Asian Option with fixed vol (Monte Carlo)
def nonGarchPricer(startPrice, strikePrice, vol, costYield, expBusDays, numPath):
    sumCallPrice = 0
    sumPutPrice = 0

    for i in range(numPath):
        ulyPrice = startPrice
        sumUlyPrice = 0
        dt = 1 / 252
        randomGenerator = np.random.normal(0, np.sqrt(dt), expBusDays)
        discountRate = 0

        # Simulated one path of underlying price
        for j in range(expBusDays):
            dWt = randomGenerator[j]
            rt = getRiskFreeRate(j) / 100
            dLogSt = (rt - costYield) * dt + vol / 100 * dWt
            discountRate += rt
            ulyPrice *= np.exp(dLogSt)
            sumUlyPrice += ulyPrice

        avgUlyPrice = sumUlyPrice / expBusDays

        sumCallPrice += max(avgUlyPrice - strikePrice, 0) * np.exp(-discountRate * dt)
        sumPutPrice  += max(strikePrice - avgUlyPrice, 0) * np.exp(-discountRate * dt)

    output = {
                'Call': sumCallPrice / numPath,
                'Put':  sumPutPrice / numPath
             }

    return output

# =============================================================================
# Data Preprocessing
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

# =============================================================================
# Get best GARCH model and other info for underlyings whose options we will price later
# =============================================================================
masterObj = {}

ulyList = list(np.unique(df_opt['Underlying']))
for underlying in progressbar.progressbar(ulyList):
    tmp_uly = underlying[:-8]
    TS_uly = df_uly[tmp_uly].dropna()
    TS_logRt = (np.log(TS_uly) - np.log(TS_uly.shift(1))).dropna() * 100
    TS_logRt = TS_logRt[TS_logRt!=0]
    max_expDays = max(df_opt[df_opt['Underlying']==underlying]['Exp BusDays'])
    masterObj[tmp_uly] = {
                            'Start Price':  TS_uly[-1],
                            'Volatility':   statistics.stdev(TS_logRt),
                            'Garch Model':  get_best_model(TS_logRt, 10, 10, 10, max_expDays)
                         }

# =============================================================================
# Execute garchPricer and collect results
# =============================================================================
nonGarchFairPriceCall = []
nonGarchFairPricePut = []
garchFairPriceCall = []
garchFairPricePut = []

# Number of Monte Carlo simulated paths
numPath = 10000

# Loop through options
for row in progressbar.progressbar(df_opt.index):
    # Retrieve the name of the underlying
    tmp_uly = df_opt['Underlying'][row][:-8]
    tmp_strike = df_opt['Strike'][row] * get_currency_divisor(tmp_uly)
    tmp_maturity = df_opt['Maturity Date'][row]
    tmp_expBusDays = df_opt['Exp BusDays'][row]

    # Retrieve the underlying historical data
    tmp_s0 = masterObj[tmp_uly]['Start Price']
    tmp_vol = masterObj[tmp_uly]['Volatility']
    tmp_model = masterObj[tmp_uly]['Garch Model']

    nonGarchResults = nonGarchPricer(tmp_s0, tmp_strike, tmp_vol, commodityYieldDict[tmp_uly], tmp_expBusDays, numPath)
    nonGarchFairPriceCall.append(nonGarchResults['Call'])
    nonGarchFairPricePut.append(nonGarchResults['Put'])

    garchResults = garchPricer(tmp_s0, tmp_strike, tmp_model, tmp_expBusDays, numPath)
    garchFairPriceCall.append(garchResults['Call'])
    garchFairPricePut.append(garchResults['Put'])

df_opt['Put (MC non-GARCH)'] = nonGarchFairPricePut
df_opt['Call (MC non-GARCH)'] = nonGarchFairPriceCall
df_opt['Put (MC GARCH)'] = garchFairPricePut
df_opt['Call (MC GARCH)'] = garchFairPriceCall


# =============================================================================
# For testing only
# =============================================================================
res = masterObj[tmp_uly]['Garch Model']['Best Model']
vol = res.forecast(horizon=100)
np.sqrt(vol.residual_variance.iloc[-1].values)
res['Best Model']

df_opt.to_csv('garchPricerResult_1.0.1.csv')

TS_logRt = TS_logRt[TS_logRt!=0]

test = get_best_model(TS_logRt, 1, 1, 1, max_expDays)

testml = test['Best Model']
wn_test = testml.resid / testml._volatility
[lbvalue, pvalue] = acorr_ljungbox(wn_test, lags = 10)
pvalue[9]

plt.figure(figsize=(20,10))
plt.plot(wn_test)
plt.grid(True)
plt.show()












