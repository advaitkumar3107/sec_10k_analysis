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