# lingshiqing

Description:

MAFS5130 Time series project
 

***
## Collaborators

- Lue Shen
- Jing Qian
- Chu Song
- Leheng Chen

***
## Data Resources
- Commodity prices: investing.com
- Commodity average options: cme group


***
## Work-Flow
- Download commodity price data from investing.com
- Data cleaning
- Use GARCH model to fit different categories of data
- Download commodity average price options from Bloomberg
- Comparison of theoretical options price ​​obtained by the model with actual options price
- Draw a conclusion and write report


***
## Commodity Prices

#### Energy
- Crude Oil: WTI Financial Futures
- Natural Gas: Henry Hub Natural Gas Futures
- Refined Products: RBOB Gasonline Futures
- Biofuels: Chicago Ethanol (Platts) Futures
- Coal: Coal (API2) CIF ARA (ARGUS-McCloskey) Futures
- Electricity: German Power Baseload Calendar Month Futures
- Petrochemicals: Mont Belvieu LDH Propane (OPIS) Futures


#### Agriculture
- Corns: Corns Futures
- Wheats: Chicago SRW Wheat Futures
- Soybean: Soybean Futures
- Soybean Meal: Soybean Meal Futures
- Soybean Oil: Soybean Oil Futures
- Livestock: Live Cattle Futures
- Livestock: Lean Hog Futures


#### Metals
- Gold: Gold Futures
- Silver: Silver Futures
- Platinum: Platinum Futures
- Copper: Copper Futures


## Commodity Derivatives
- [WTI Crude Oil Asian Option](https://www.cmegroup.com/trading/energy/crude-oil/west-texas-intermediate-wti-crude-oil-calendar-swap-futures_quotes_globex_options.html?exchange=CME&sector=COAL#exchange=CME&sector=COAL&optionProductId=2767&strikeRange=ATM)
- [Chicago Ethanol (Platts) Asian Option](https://www.cmegroup.com/trading/energy/ethanol/chicago-ethanol-platts-swap_quotes_globex_options.html?optionProductId=5174#optionProductId=5174&strikeRange=ATM)
- [Gold American Option](https://www.cmegroup.com/trading/metals/precious/gold_quotes_globex_options.html?optionProductId=192#optionProductId=192&strikeRange=ATM)
- [Silver American Option](https://www.cmegroup.com/trading/metals/precious/silver_quotes_globex_options.html?optionProductId=193#optionProductId=193&strikeRange=ATM)
- [Natural Gas European Option](https://www.cmegroup.com/trading/energy/natural-gas/natural-gas_quotes_globex_options.html?optionProductId=1352#optionProductId=1352&strikeRange=ATM)



***

## Model Evaluation Method

Use ARE (Average Relative Error) criterion
 
![equation](https://latex.codecogs.com/gif.latex?ARE&space;=&space;\frac&space;1N&space;\sum_{j=1}^N&space;\frac&space;{|V_j^{model}&space;-&space;V_j^{market}|}{V_j^{market}}&space;\times&space;100)
 
where N is the total number of options considered.
 
Need to find a (prevailing) market model to compare our model results.


***
 
## Work-Distributions
 
- Lue Shen: LS
- Jing Qian: JQ
- Chu Song: CS
- Leheng Chen: LC
 
[JQ ] Step 1: preprocess underlying data and option market price data into two big dataframes.
 
[CS ] Step 2: create a function, to read one historical underlying price time series, and to find out the best GARCH model for it.
 
[CS, JQ] Step 3: derive correlation infomation (say correlations of prices/percentage changes, collinearity, etc., utilizing cleaned data) between two given historical underlying price time series, analyse the results and derive some conclusions regarding differenct commodity sectors.
 
[LS ] Step 4: create a function, input with GARCH model, commence date, expiry date, strike price, output fair price for the average price options.
 
[LC ] Step 5: reproduce some prevailing pricing methods for average price options.
 
[JQ ] Step 6: use ARE criterion to compare results between prevailing methods and GARCH method.

[TBD ] Step 7: calculate greeks (based on step 4 function), implied volatility curve (reverse pricing function, and using bisection method), etc.