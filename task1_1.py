from sec_edgar_downloader import Downloader
from utils import checker
from section_location import get_section_location
import numpy as np
import pandas as pd
from process import process_text
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from predict import prediction

def download_data(ticker):
    ## Create a dummy profile for webrequest
    dl = Downloader("MyCompanyName", "my.email@domain.com")
    
    curr_year = pd.Timestamp.now().year    ## get current year
    start_date = str(curr_year - 30) + '-01-01'   ## past 30 years' data
    end_date = str(curr_year) + '-12-31'     
    
    ### download all the data for the particular stock in the last 30 years 
    dl.get("10-K", ticker, after=start_date, before=end_date)    
    return

if __name__ == '__main__':
    user_input = input("Enter a stock ticker: ")
    download_data(user_input)
    print('Finished Downloading all the files in the local directory....')