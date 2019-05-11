# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:46:08 2019

@author: Leheng Chen
"""

from binomialTreePricer import asianOptionBinomialTree
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

uly_names = ['Crude Oil WTI', 'Ethanol', 'Gold', 'Silver', 'Natural Gas']
uly_init = df_uly[uly_names].tail(1)
df_opt['bdays'] = 1 + np.busday_count(df_opt['Start Date'].values.astype('datetime64[D]'), df_opt['Maturity Date'].values.astype('datetime64[D]'))

df_uly_vol = df_uly[uly_names].std(skipna=True)

oneOverRho = 3
df_vols = pd.DataFrame([[0.3, 0.01, 0.4, 0.1, 0.001]], columns = uly_names)
df_units = pd.DataFrame([[0.01, 0.0001, 1, 0.001, 0.01]], columns = uly_names)
bdays_year = 252
    
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
for i in range(len(yieldCurveDict)):
    datePoint1 = curvePoints[i]
    datePoint2 = curvePoints[i + 1]
    if (datePoint1 == curvePoints[0]):
        forwardCurveDict[datePoint2] = yieldCurveDict[datePoint2]
    else:
        yieldAtDate1 = yieldCurveDict[datePoint1]
        yieldAtDate2 = yieldCurveDict[datePoint2]
        busDateDiff1 = np.busday_count(curvePoints[0], datePoint1)
        busDateDiff2 = np.busday_count(curvePoints[0], datePoint2)
        forwardCurveDict[datePoint2] = float((yieldAtDate2 * busDateDiff2 - yieldAtDate1 * busDateDiff1) / (busDateDiff2 - busDateDiff1))

# Function to get risk free rate given a date (datetime.date object)
def getRiskFreeRate(inputDate):
    input_date = inputDate.date()
    for i in range(len(forwardCurveDict)):
        datePoint1 = datetime.strptime(curvePoints[i],'%Y-%m-%d').date()
        datePoint2 = datetime.strptime(curvePoints[i + 1],'%Y-%m-%d').date()
        if (input_date >= datePoint1 and input_date < datePoint2):
            return forwardCurveDict[curvePoints[i + 1]]
    return 0


for row in df_opt.index:
    # Retrieve the name of the underlying
    tmp_uly = df_opt['Underlying'][row][:-8]
    tmp_strike = df_opt['Strike'][row]
    tmp_maturity = df_opt['Maturity Date'][row]
    tmp_steps = df_opt['bdays'][row]
    if tmp_steps > bdays_year:
        tmp_steps = bdays_year
    tmp_init = uly_init[tmp_uly][0]
    tmp_time_period = 1 / bdays_year
    tmp_vol = df_uly_vol[tmp_uly]
    tmp_ir = get_interest_rate(tmp_steps)
    tmp_rates = [getRiskFreeRate(tmp_maturity - timedelta(d)) for d in range(tmp_steps)]
    
    tmp_call = df_opt['Call'][row]
    tmp_unit = df_units[tmp_uly][0]
    
    pricer = asianOptionBinomialTree(tmp_steps, tmp_vol, tmp_time_period, oneOverRho, tmp_rates)
    sim = pricer.getOptionPrice(tmp_init, tmp_strike * tmp_unit)
    print('undeylying: %s; bdays: %d, strile: %6.3f, init: %6.3f --> simulate: %6.3f; actual call: %6.3f' \
          % (tmp_uly, tmp_steps, tmp_strike* tmp_unit, tmp_init, sim, tmp_call))

