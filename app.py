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


tickers = pd.read_csv('all_stocks.csv').Symbol.unique().tolist()
years = [x for x in range(1995,2024)]

# Title of the application
st.title('Stock Data Visualization')

# Dropdown to select the ticker
ticker = st.selectbox('Select Ticker', tickers)

# Dropdown to select the year
year = st.selectbox('Select Year', years)

if st.button('Generate Plot'):
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
        
    historical_df = pd.DataFrame(historical_dict).transpose()
    historical_df_norm = (historical_df - historical_df.mean())/historical_df.std()

    competitors_df = pd.DataFrame(competitors_dict).transpose()
    competitors_df_norm = (competitors_df - competitors_df.mean())/competitors_df.std()

    fig1 = make_subplots(rows=1, cols=2, subplot_titles=("Historical Sentiment Analysis", "Historical Sentiment Analysis Normalised"))
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Competitor Sentiment Analysis", "Competitor Sentiment Analysis Normalised"))

    # Define legend positions for each subplot
    make_plots(fig1, historical_df, 1, 1, 'Historical Data Analysis', legend_x=-2, legend_y=-0.1)
    make_plots(fig1, historical_df_norm, 1, 2, 'Historical Data Analysis Normalised', legend_x=0.6, legend_y=-0.1)
    make_plots(fig2, competitors_df, 1, 1, 'Competitor Data Analysis', legend_x=0.1, legend_y=-0.1)
    make_plots(fig2, competitors_df_norm, 1, 2, 'Competitor Data Analysis Normalised', legend_x=0.6, legend_y=-0.1)

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)