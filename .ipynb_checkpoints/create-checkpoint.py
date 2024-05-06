from sec_edgar_downloader import Downloader
from utils import checker
from section_location import get_section_location
import numpy as np
import pandas as pd
from process import process_text
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from predict import prediction
import streamlit as st

def create_df(ticker, year, tokenizer, tokenizer_fls, model, model_fls):    
    """
    Input : 
    ticker (str) : The user defined ticker
    year (int) : The user defined year
    tokenizer (BertTokenizer) : The tokenizer to convert the sentence into corresponding tokens to feed into the model predicting sentiment
    tokenizer_fls (AutoTokenizer) : The tokenizer to convert the sentence into corresponding tokens to feed into the model predicting forward looking statements
    model (BertModelForSequenceClassification) : The NLP model to be used for predicting sentiment
    model_fls (AutoModelForSequenceClassification) : The NLP model to be used for predicting forward looking sentences

    Output : 
    dict : Contains percentage of positive sentiment sentences, negative sentiment sentences and forward looking statements (specific as well as non-specific
    """

    try:
        dl = Downloader("MyCompanyName", "my.email@domain.com")  ## create a dummy profile for web request
        directory = 'sec-edgar-filings/' + ticker + '/10-K/'    ### Directory to check if report already exists
        file = checker(directory, year)   ## Check if report exists
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
    ## If Report doesnt exist
    if file is None:
        start_date = str(year) + '-01-01'
        end_date = str(year) + '-12-31'
        dl.get("10-K", ticker, after=start_date, before=end_date)   ### Get new report for the year
        file = checker(directory, year)   ### Check again if report is finally downloaded

    ### If Report still doesnt exist
    if file is None:
        print("10K report for " + ticker + " doesn't exist for " + str(year))  ### The report wasnt available on the website
        return {'negatives' : np.nan, 'positives' : np.nan, 'fls' : np.nan}  ## Return NaNs for plotting

    #### If Report is downloaded properly
    else:    
        text = file.read()   
        try:
            text, section_df = get_section_location(text)  ## Get the df containing start and end positions of sections
        except Exception as e:
            st.error(f"An error occurred in get_section_location.py: {e}")

        preds_df = pd.DataFrame()
        preds_fls_df = pd.DataFrame()

        ### Loop over relevant sections
        for section in ['1', '1a', '1b', '1c', '2', '3', '4', '7', '7a']:
            try:
                clean_text = process_text(text, section, section_df)   ### clean and process xml text of a particular section
            except Exception as e:
                st.error(f"An error occurred in process_text.py: {e}")
                continue

            try:
                sentences = sent_tokenize(clean_text)   ### Convert the entire text into a list of sentences
            except Exception as e:
                st.error(f"An error occurred in get_section_location.py: {e}")
                continue


            ### Only if the section has enough sentences perform analysis else skip to next section
            if (len(sentences) > 5):
                preds = {}
                preds_fls = {}
                try:
                    preds['preds'], preds['prob'] = prediction(model, tokenizer, sentences)   ## Make a prediction of the sentiment
                except Exception as e:
                    st.error(f"An error occurred in finbert prediction: {e}")
                    continue

                try:
                    preds_fls['preds'], preds_fls['prob'] = prediction(model_fls, tokenizer_fls, sentences)   ## Make a prediction of forward looking nature
                except Exception as e:
                    st.error(f"An error occurred in finbert fls prediction: {e}")

                preds_df = pd.concat([preds_df, pd.DataFrame(preds)])   ## append to the existing df
                preds_fls_df = pd.concat([preds_fls_df, pd.DataFrame(preds_fls)])  ## append to existing df

            else:
                print('Section ' + section + ' : has too few sentences')

            # ### If exception raised => section wasnt present in the report
            # except:
            #     print('Section ' + section + " doesn't exist in the report")
            #     continue

        ### Calculate percentages of positive, negative and forward looking sentences (specific as well as non specific). Only count those predictions for which the confidence was above 0.5 (threshold)
        pct_positives = len(preds_df[(preds_df.preds == 'positive') & (preds_df.prob > 0.5)])/len(preds_df[(preds_df.prob > 0.5)])
        pct_negatives = len(preds_df[(preds_df.preds == 'negative') & (preds_df.prob > 0.5)])/len(preds_df[(preds_df.prob > 0.5)])
        pct_fls = len(preds_fls_df[(preds_fls_df.preds != 'Not FLS') & (preds_fls_df.prob > 0.5)])/len(preds_fls_df[(preds_fls_df.prob > 0.5)])

        return {'negatives' : pct_negatives, 'positives' : pct_positives, 'fls' : pct_fls}