from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from utils import get_competitors
from create import create_df
from plot import make_plots
from plotly.io import show

##### Inputs #####
ticker = input("Enter the ticker that you want to analyze: ")
year = int(input("Enter the year for which you want the analysis: "))

### Load tokenizers for sentiment and FLS
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
tokenizer_fls = BertTokenizer.from_pretrained("yiyanghkust/finbert-fls")

### Load models for sentiment and FLS
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model_fls = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-fls")

### Load competitors for the ticker. all_stocks.csv contains all the Symbol data like ticker name and Market Cap
competitors = get_competitors('all_stocks.csv', 5, ticker)

### initialize empty dicts, one for competitors data another for the stocks historical analysis data
competitors_dict = {}
historical_dict = {}

## Make predictions for each competitor
for stock in competitors:
    competitors_dict[stock] = create_df(stock, year, tokenizer, tokenizer_fls, model, model_fls)
    ## If curr_stock is the user stock then populate the historical_dict as well
    if (stock == ticker):
        historical_dict[year] = competitors_dict[stock].copy()

## Make predictions for past 5 years of user input stock
for curr_year in range(year-4,year):
    historical_dict[curr_year] = create_df(ticker, curr_year, tokenizer, tokenizer_fls, model, model_fls)
    
## Convert the dicts to dfs both unnormalised and normalised
historical_df = pd.DataFrame(historical_dict).transpose().sort_index()
historical_df_norm = (historical_df - historical_df.mean())/historical_df.std()
competitors_df = pd.DataFrame(competitors_dict).transpose()
competitors_df_norm = (competitors_df - competitors_df.mean())/competitors_df.std()


#### Create 2 figures each having 2 subplots, for normalised and unnormalised versions
fig1 = make_subplots(rows=1, cols=2, subplot_titles=("Historical Sentiment Analysis", "Historical Sentiment Analysis Normalised"))
fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Competitor Sentiment Analysis", "Competitor Sentiment Analysis Normalised"))

# Define legend positions for each subplot
make_plots(fig1, historical_df, 1, 1, 'Historical Data Analysis', legend_x=-2, legend_y=-0.1)
make_plots(fig1, historical_df_norm, 1, 2, 'Historical Data Analysis Normalised', legend_x=0.6, legend_y=-0.1)
make_plots(fig2, competitors_df, 1, 1, 'Competitor Data Analysis', legend_x=0.1, legend_y=-0.1)
make_plots(fig2, competitors_df_norm, 1, 2, 'Competitor Data Analysis Normalised', legend_x=0.6, legend_y=-0.1)

show(fig1, renderer="browser")  # Adjust the renderer as needed
show(fig2, renderer="browser")  # Adjust the renderer as needed