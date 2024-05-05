from sec_edgar_downloader import Downloader
from utils import checker
from section_location import get_section_location
import numpy as np
import pandas as pd
from process import process_text
from nltk.tokenize import word_tokenize, sent_tokenize
from predict import prediction

def create_df(ticker, year, tokenizer, tokenizer_fls, model, model_fls):    
    nltk.download('punkt')
    dl = Downloader("MyCompanyName", "my.email@domain.com")
    directory = 'sec-edgar-filings/' + ticker + '/10-K/'
    file = checker(directory, year)

    if file is None:
        start_date = str(year) + '-01-01'
        end_date = str(year) + '-12-31'
        dl.get("10-K", ticker, after=start_date, before=end_date)    
        file = checker(directory, year)

    if file is None:
        print("10K report for " + ticker + " doesn't exist for " + str(year))
        return {'negatives' : np.nan, 'positives' : np.nan, 'fls' : np.nan}
    
    else:    
        text = file.read()
        text, section_df = get_section_location(text)
        
        preds_df = pd.DataFrame()
        preds_fls_df = pd.DataFrame()

        for section in ['1', '1a', '1b', '1c', '2', '3', '4', '7', '7a']:
            try:
                clean_text = process_text(text, section, section_df)
                sentences = sent_tokenize(clean_text)

                if (len(sentences) > 5):
                    preds = {}
                    preds_fls = {}
                    preds['preds'], preds['prob'] = prediction(model, tokenizer, sentences)
                    preds_fls['preds'], preds_fls['prob'] = prediction(model_fls, tokenizer_fls, sentences)

                    preds_df = pd.concat([preds_df, pd.DataFrame(preds)])
                    preds_fls_df = pd.concat([preds_fls_df, pd.DataFrame(preds_fls)])

                else:
                    print('Section ' + section + ' : has too few sentences')

            except Exception as e:
                print('Section ' + section + " doesn't exist in the report")
                print(e)
                continue

        pct_positives = len(preds_df[(preds_df.preds == 'positive') & (preds_df.prob > 0.5)])/len(preds_df[(preds_df.prob > 0.5)])
        pct_negatives = len(preds_df[(preds_df.preds == 'negative') & (preds_df.prob > 0.5)])/len(preds_df[(preds_df.prob > 0.5)])
        pct_fls = len(preds_fls_df[(preds_fls_df.preds != 'Not FLS') & (preds_fls_df.prob > 0.5)])/len(preds_fls_df[(preds_fls_df.prob > 0.5)])

        return {'negatives' : pct_negatives, 'positives' : pct_positives, 'fls' : pct_fls}
