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
