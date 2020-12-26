import pandas as pd
import numpy as np
import os
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.config import config

# quick and dirty code
def process_stock(stock_name):
    df = pd.read_csv('data/turkish/' + stock_name)

    df = df.rename(columns={'Date': 'date', 
        'Open': 'open',
        'Close': 'close',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    })

    df['tic'] = stock_name
    df = df.drop(columns=['Currency'])

    df.sort_values(['date', 'tic'],ignore_index=True)

    df = FeatureEngineer(df.copy(),
        use_technical_indicator=True,
        tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=False,
        user_defined_feature = True).preprocess_data()

    print('completed:' + stock_name)
    return df


if __name__ == "__main__":     
    arr = os.listdir('./data/turkish')

    result = process_stock('AKBNK')
    
    for i in range(1, len(arr)):
        df = process_stock(arr[i])
        result = pd.concat([result, df])

    result.to_csv('data/preprocessed_turkish.csv')