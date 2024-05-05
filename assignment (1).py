#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import Counter
from sec_edgar_downloader import Downloader
import os
import fnmatch
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import yfinance as yf
import numpy as np
import torch
import scipy
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from IPython.core.debugger import set_trace
from bs4 import BeautifulSoup
nltk.download('punkt')


# In[2]:


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


# In[3]:


def get_competitors(sheet_name, num_neighbours, ticker):
# ticker = 'GOOGL'
# df = pd.read_csv('all_stocks.csv')
    df = pd.read_csv(sheet_name)
    x = df[(df.Symbol == ticker)]
    temp = df[df.Industry == x.Industry.iloc[0]]
    temp['Market Cap'] = temp['Market Cap'].astype(float).astype(int)
    temp['cap_diff'] = np.abs(temp['Market Cap'] - int(float(x['Market Cap'].iloc[0])))
    temp = temp.sort_values('cap_diff').Symbol.head(num_neighbours).tolist()
    return temp


# In[4]:


def get_section_location(text):
    regex = re.compile(r'(>Item(\s|&#160;|&nbsp;|&#xA0;)(1\.|1A|1B|1C|2|3|4|5|6|7|7A|8)\.{0,1})|(>ITEM(\s|&#160;|&nbsp;|&#xA0;)(1\.|1A|1B|1C|2|3|4|5|6|7|7A|8))')
    matches = regex.finditer(text)

    test_df = pd.DataFrame([[x.group(), x.start(), x.end()] for x in matches])
    test_df.columns = ['group', 'start_idx', 'end_idx']
    test_df.group = test_df.group.str.lower()
    
    test_df.replace('&#160;',' ',regex=True,inplace=True)
    test_df.replace('&nbsp;',' ',regex=True,inplace=True)
    test_df.replace('&#xa0;',' ',regex=True,inplace=True)

    test_df.replace(' ','',regex=True,inplace=True)
    test_df.replace('\.','',regex=True,inplace=True)
    test_df.replace('>','',regex=True,inplace=True)

    test_df = test_df.drop_duplicates('group', keep = 'last')
    test_df.index = range(len(test_df))
    test_df = test_df.set_index('group')
    
    return text, test_df


# In[5]:


def process_text(text, section, test_df):
    location = 'item' + section
    index_list = test_df.index.tolist()
    curr_index = index_list.index(location)
    item_raw = text[test_df.iloc[curr_index]['start_idx'] : test_df.iloc[curr_index+1]['start_idx']]
    soup = BeautifulSoup(item_raw, 'html.parser')
    for table in soup.find_all('table'):
        table.decompose()

    text_act = soup.get_text(separator=' ', strip=True)
    text_act = text_act.replace('\xa0', ' ')

    clean_text = ' '.join(text_act.split())
    clean_text = re.sub(r'<span style=.*$', '', clean_text)
    return clean_text


# In[6]:


def prediction(model, tokenizer, X):
    preds = []
    preds_proba = []
    tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}
    for x in X:
        with torch.no_grad():
            input_sequence = tokenizer(x, return_tensors="pt", **tokenizer_kwargs)
            logits = model(**input_sequence).logits
            scores = {
            k: v
            for k, v in zip(
                model.config.id2label.values(),
                scipy.special.softmax(logits.numpy().squeeze()),
            )
        }
        sentimentFinbert = max(scores, key=scores.get)
        probabilityFinbert = max(scores.values())
        preds.append(sentimentFinbert)
        preds_proba.append(probabilityFinbert)
    return preds, preds_proba


# In[7]:


def create_df(ticker, year, tokenizer, tokenizer_fls, model, model_fls):    
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

            except:
                print('Section ' + section + " doesn't exist in the report")
                continue

        pct_positives = len(preds_df[(preds_df.preds == 'positive') & (preds_df.prob > 0.5)])/len(preds_df[(preds_df.prob > 0.5)])
        pct_negatives = len(preds_df[(preds_df.preds == 'negative') & (preds_df.prob > 0.5)])/len(preds_df[(preds_df.prob > 0.5)])
        pct_fls = len(preds_fls_df[(preds_fls_df.preds != 'Not FLS') & (preds_fls_df.prob > 0.5)])/len(preds_fls_df[(preds_fls_df.prob > 0.5)])

        return {'negatives' : pct_negatives, 'positives' : pct_positives, 'fls' : pct_fls}


# In[25]:


def make_plots(fig, df, col, subplot_idx, title_text, legend_x, legend_y):
    bar_width = 0.2
    num_cols = len(df.columns)
    group_width = num_cols * bar_width

    for i, column in enumerate(df.columns):
        fig.add_trace(
            go.Bar(
                x=[x + i * bar_width for x in range(len(df))],
                y=df[column],
                name=column,
                text=[f"{val:.2f}" for val in df[column]],
                textposition='outside',
                hoverinfo='text',
                hovertext=[f"{val:.2f}" for val in df[column]],
                width=bar_width,
                legendgroup=f'group{subplot_idx}',  # Assign legend group
                showlegend=True  # Show legend only for the first trace of each group
            ), 
            row=1, col=subplot_idx
        )
    
    # Update subplot's x-axis
    fig.update_xaxes(
        tickvals=[x + group_width / 2 - bar_width / 2 for x in range(len(df))],
        ticktext=df.index,
        row=1, col=subplot_idx
    )

    # Customize the layout and position the legend
    fig.update_layout(
        title_text=title_text,
        barmode='group',
        showlegend=True,
        legend=dict(x=legend_x, y=legend_y, orientation="h")
    )
    return fig


# In[9]:


##### Inputs #####
ticker = 'MS'
year = 2014


# In[10]:


tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
tokenizer_fls = BertTokenizer.from_pretrained("yiyanghkust/finbert-fls")

model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model_fls = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-fls")

competitors = get_competitors('all_stocks.csv', 5, ticker)
competitors_dict = {}
historical_dict = {}

for stock in competitors:
    competitors_dict[stock] = create_df(stock, year, tokenizer, tokenizer_fls, model, model_fls)
    if (stock == ticker):
        historical_dict[year] = competitors_dict[stock].copy()

for curr_year in range(year-4,year):
    historical_dict[curr_year] = create_df(ticker, curr_year, tokenizer, tokenizer_fls, model, model_fls)


# In[11]:


historical_df = pd.DataFrame(historical_dict).transpose().sort_index()
historical_df_norm = (historical_df - historical_df.mean())/historical_df.std()
competitors_df = pd.DataFrame(competitors_dict).transpose()
competitors_df_norm = (competitors_df - competitors_df.mean())/competitors_df.std()


# In[12]:


historical_df


# In[13]:


competitors_df


# In[14]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig1 = make_subplots(rows=1, cols=2, subplot_titles=("Historical Sentiment Analysis", "Historical Sentiment Analysis Normalised"))
fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Competitor Sentiment Analysis", "Competitor Sentiment Analysis Normalised"))

# Define legend positions for each subplot
make_plots(fig1, historical_df, 1, 1, 'Historical Data Analysis', legend_x=-2, legend_y=-0.1)
make_plots(fig1, historical_df_norm, 1, 2, 'Historical Data Analysis Normalised', legend_x=0.6, legend_y=-0.1)
make_plots(fig2, competitors_df, 1, 1, 'Competitor Data Analysis', legend_x=0.1, legend_y=-0.1)
make_plots(fig2, competitors_df_norm, 1, 2, 'Competitor Data Analysis Normalised', legend_x=0.6, legend_y=-0.1)

fig1.show()
fig2.show()


# In[15]:


import yfinance as yf


# In[16]:


comps = get_competitors('all_stocks.csv', 5, ticker)

stock_prices = pd.DataFrame()
stock_rets = pd.DataFrame()

for comp in comps:    
    tickerData = yf.Ticker(comp)

    # Get the historical prices for this ticker
    tickerDf = tickerData.history(start = str(year+1) + '-01-01', end = str(year+1) + '-12-31')  # 1 month of data

    # Show only the Close prices
    close_prices = tickerDf['Close']
    close_prices_rets = close_prices/close_prices.iloc[0] - 1
    
    stock_prices = pd.concat([stock_prices, close_prices.rename(comp)], axis = 1)
    stock_rets = pd.concat([stock_rets, close_prices_rets.rename(comp)], axis = 1)


# In[17]:


import plotly.express as px


# In[18]:


px.line(stock_rets)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig1 = make_subplots(rows=1, cols=2, subplot_titles=("Historical Sentiment Analysis", "Historical Sentiment Analysis Normalised"))
fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Competitor Sentiment Analysis", "Competitor Sentiment Analysis Normalised"))

# Define legend positions for each subplot
make_plots(fig1, historical_df, 1, 1, 'Historical Data Analysis', legend_x=-2, legend_y=-0.1)
make_plots(fig1, historical_df_norm, 1, 2, 'Historical Data Analysis Normalised', legend_x=0.6, legend_y=-0.1)
make_plots(fig2, competitors_df, 1, 1, 'Competitor Data Analysis', legend_x=0.1, legend_y=-0.1)
make_plots(fig2, competitors_df_norm, 1, 2, 'Competitor Data Analysis Normalised', legend_x=0.6, legend_y=-0.1)

fig1.show()
fig2.show()


# In[ ]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import io
import base64
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from utils import get_competitors
from create import create_df
from plot import make_plots

app = Flask(__name__)

def load_tickers(sheet_name):
    df = pd.read_csv(sheet_name)
    tickers = df.Symbol.unique().tolist()
    return tickers
    
def create_figure(fig):
    # Convert Plotly figure to HTML for embedding
    return pio.to_html(fig, full_html=False)


@app.route('/', methods=['GET', 'POST'])
def index():
    year = request.form.get('year', '2020')
    ticker = request.form.get('ticker', 'AAPL')
    
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    tokenizer_fls = BertTokenizer.from_pretrained("yiyanghkust/finbert-fls")

    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model_fls = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-fls")

    competitors = get_competitors('all_stocks.csv', 5, ticker)
    competitors_dict = {}
    historical_dict = {}

    for stock in competitors:
        competitors_dict[stock] = create_df(stock, year, tokenizer, tokenizer_fls, model, model_fls)
        if (stock == ticker):
            historical_dict[year] = competitors_dict[stock].copy()

    for curr_year in range(year-4,year):
        historical_dict[curr_year] = create_df(ticker, curr_year, tokenizer, tokenizer_fls, model, model_fls)

    fig1 = make_subplots(rows=1, cols=2, subplot_titles=("Historical Sentiment Analysis", "Historical Sentiment Analysis Normalised"))
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Competitor Sentiment Analysis", "Competitor Sentiment Analysis Normalised"))

    # Define legend positions for each subplot
    make_plots(fig1, historical_df, 1, 1, 'Historical Data Analysis', legend_x=-2, legend_y=-0.1)
    make_plots(fig1, historical_df_norm, 1, 2, 'Historical Data Analysis Normalised', legend_x=0.6, legend_y=-0.1)
    make_plots(fig2, competitors_df, 1, 1, 'Competitor Data Analysis', legend_x=0.1, legend_y=-0.1)
    make_plots(fig2, competitors_df_norm, 1, 2, 'Competitor Data Analysis Normalised', legend_x=0.6, legend_y=-0.1)
    
    figure1 = create_figure(fig1)
    figure2 = create_figure(fig2)
    return render_template('index.html', figure1=figure1, figure2=figure2, selected_year=year, ticker=ticker)

tickers = load_tickers('all_stocks.csv')

if __name__ == '__main__':
    app.run(debug=True)


# In[19]:


df = pd.read_csv('all_stocks.csv')


# In[23]:


df.Symbol.unique().tolist()

