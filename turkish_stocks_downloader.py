import pandas as pd
import numpy as np
import os
import investpy

if __name__ == "__main__":     
    BIST30 = ['AKBNK','ARCLK','ASELS','BIMAS','DOHOL',
        'EKGYO','EREGL','GARAN','GUBRF','HALKB',
        'ISCTR','KCHOL','KOZAA','KOZAL','KRDMD','MGROS',
        #'OYAKC',
        'PETKM','PGSUS','SAHOL',
        'SISE','TAVHL','TCELL','THYAO','TKFEN',
        'TSKB','TTKOM','TUPRS','VAKBN','YKBNK']

    for i in range(len(BIST30)):
        df = investpy.get_stock_historical_data(stock=BIST30[i],
            country='Turkey',
            from_date='01/01/2010',
            to_date='01/01/2020')

        df.to_csv('data/turkish/' + BIST30[i])
    