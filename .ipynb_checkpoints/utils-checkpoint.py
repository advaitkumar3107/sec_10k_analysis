import os
import fnmatch
import pandas as pd
import numpy as np

def checker(directory, year):
    """
    Inputs : 
    directory (str) : The directory to check for the file
    year (int) : The year for the SEC 10K report
    
    Outputs : The file which is present in the directory or None if no file exists
    """
    
    if os.path.exists(directory):
        pattern = '*-' + str(year % 100) + '-*'   ### Check for the pattern which is some string followed by '-last_two_digits_of_year-' followed by some string which is the format for EDGAR files
        found = False
        for filename in os.listdir(directory):
            if fnmatch.fnmatch(filename, pattern):  ## If match found, open it and return the file
                found = True
                file = open(directory + '/' + filename + '/' + 'full-submission.txt')
                return file
        return None  ## If no match found return None
    else:
        return None   ### If directory doesnt exist, return None    


def get_competitors(sheet_name, num_neighbours, ticker):
    """
    Input:
    sheet_name (str) : Name of the .csv file which has the symbol data
    num_neighbours (int) : Number of competitors that need to be loaded
    ticker (str) : The current ticker for which competitors need to be found
    
    Output:
    temp (list[str]) : List of the tickers of nearest 5 competitors (including the ticker itself)
    """
    
    df = pd.read_csv(sheet_name)
    x = df[(df.Symbol == ticker)]   ## get data for current ticker
    temp = df[df.Industry == x.Industry.iloc[0]]   ## get all the stocks of the same industry
    
    ## Get the closest (num_neighbours) number of stocks according to the absolute difference in market cap 
    temp['Market Cap'] = temp['Market Cap'].astype(float)
    temp['cap_diff'] = np.abs(temp['Market Cap'] - float(x['Market Cap'].iloc[0]))
    
    temp = temp.sort_values('cap_diff').Symbol.head(num_neighbours).tolist() ## convert to list
    return temp