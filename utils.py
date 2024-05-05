import os
import fnmatch
import pandas as pd
import numpy as np

def checker(directory, year):
    if os.path.exists(directory):
        pattern = '*-' + str(year % 100) + '-*'
        found = False
        for filename in os.listdir(directory):
            if fnmatch.fnmatch(filename, pattern):
                found = True
                file = open(directory + '/' + filename + '/' + 'full-submission.txt')
                return file
        return None
    else:
        return None    
    
def get_competitors(sheet_name, num_neighbours, ticker):
# ticker = 'GOOGL'
# df = pd.read_csv('all_stocks.csv')
    df = pd.read_csv(sheet_name)
    x = df[(df.Symbol == ticker)]
    temp = df[df.Industry == x.Industry.iloc[0]]
    temp['Market Cap'] = temp['Market Cap'].astype(float)
    temp['cap_diff'] = np.abs(temp['Market Cap'] - float(x['Market Cap'].iloc[0]))
    temp = temp.sort_values('cap_diff').Symbol.head(num_neighbours).tolist()
    return temp