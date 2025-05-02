import yfinance as yf
import numpy as np
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import warnings
from pymongo import MongoClient
import os
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Tesla Monte Carlo Simulation", layout="wide")

with st.sidebar:
    selected = option_menu("Menu",
    ["code", "Data Analysis", "Visual representations", "MongoDB", "Team Member"],
    icons=["code-slash", "bar-chart", "image", "", "people-fill"],
    menu_icon="cast",
    default_index=0
)

@st.cache_data(ttl=86400)  # cache for 1 day
def get_tesla_data():
    tesla = yf.Ticker("TSLA")
    return tesla.history(period="max")

try:
    tesla_data = get_tesla_data()
except Exception as e:
    st.error("Error fetching Tesla data. It may be a rate-limit issue. Try again later.")
    st.stop()

tesla_data.reset_index(inplace=True)

url = "https://www.macrotrends.net/stocks/charts/TSLA/tesla/revenue"
headers = {"User-Agent": "Mozilla/5.0"}
html_data = requests.get(url, headers=headers)
soup = BeautifulSoup(html_data.content, "html.parser")
html_data = pd.read_html(html_data.text)
tesla_revenue = html_data[1]
tesla_revenue.columns = ["Date", "Revenue"]


days = 365
simulations = 1000
last_price = tesla_data['Close'].iloc[-1]
returns = tesla_data['Close'].pct_change().dropna()
mean_return = returns.mean()
std_return = returns.std()

simulated_prices = []
for _ in range(simulations):
    price_series = [last_price]
    for _ in range(days):
        daily_return = np.random.normal(mean_return, std_return)
        price_series.append(price_series[-1] * (1 + daily_return))
    simulated_prices.append(price_series)


def clean_revenue(value):
    cleaned = re.sub(r"[^\d.]", "", str(value))

    if cleaned:
        return float(cleaned)
    return float('nan')



if selected == "code":
    st.code("""
    tesla = yf.Ticker("TSLA")
    tesla_data = pd.DataFrame()
    tesla_data = tesla.history(period="max")
    tesla_data.reset_index(inplace=True)

    url = "https://www.macrotrends.net/stocks/charts/TSLA/tesla/revenue"
    headers = {"User-Agent": "Mozilla/5.0"}
    html_data = requests.get(url, headers=headers)
    soup = BeautifulSoup(html_data.content, "html.parser")
    html_data = pd.read_html(html_data.text)
    tesla_revenue = html_data[1]
    tesla_revenue.columns = ["Date","Revenue"]
    """)

    with st.echo():
        st.dataframe(tesla_revenue)

    st.code("""
    tesla_revenue["Revenue"] = tesla_revenue["Revenue"].apply(clean_revenue)
    tesla_revenue.dropna(inplace=True)
    """)

    tesla_revenue["Revenue"] = tesla_revenue["Revenue"].apply(clean_revenue)
    tesla_revenue.dropna(inplace=True)

    with st.echo():
        st.dataframe(tesla_revenue)



elif selected == "Data Analysis":

    tesla_revenue["Revenue"] = tesla_revenue["Revenue"].apply(clean_revenue)
    tesla_revenue.dropna(inplace=True)

    with st.echo():
        correlation = tesla_data['Close'].corr(tesla_revenue['Revenue'])
        st.write(f"Correlation between stock price and revenue: {correlation:0.3%}")

    with st.echo():
        volatility = tesla_data['Close'].pct_change().std()
        st.write(f"Volatility (standard deviation of daily returns): {volatility}")

    st.code("""
    days = 365  
    simulations = 1000
    last_price = tesla_data['Close'].iloc[-1]
    returns = tesla_data['Close'].pct_change().dropna()
    mean_return = returns.mean()
    std_return = returns.std()

    simulated_prices = []
    for _ in range(simulations):
        price_series = [last_price]
        for _ in range(days):
            daily_return = np.random.normal(mean_return, std_return)
            price_series.append(price_series[-1] * (1 + daily_return))
        simulated_prices.append(price_series)    
    """)


    plt.style.use('dark_background')

    st.title("🚗 Tesla Stock: Monte Carlo Simulations")

    simulations = 1000
    trading_days = 365
    last_price = tesla_data['Close'].iloc[-1]

    simulated_prices = []
    for _ in range(simulations):
        price_series = [last_price]
        for _ in range(days):
            daily_return = np.random.normal(mean_return, std_return)
            price_series.append(price_series[-1] * (1 + daily_return))
        simulated_prices.append(price_series)

    simulated_prices = np.array(simulated_prices)

    fig = plt.figure(figsize=(12, 6))
    for prices in simulated_prices:
        plt.plot(prices, alpha=0.1, color='green')

    plt.plot(np.percentile(simulated_prices, 50, axis=0), color='#FFD700', linewidth=2, label='Median Path')
    plt.plot(np.percentile(simulated_prices, 5, axis=0), color='red', linestyle=':', label='5th Percentile')
    plt.plot(np.percentile(simulated_prices, 95, axis=0), color='red', linestyle=':', label='95th Percentile')

    plt.title(f"Tesla Stock: {simulations} Monte Carlo Simulations\n(Shaded = 90% Confidence Interval)", pad=20)
    plt.xlabel("Trading Days (1 Year Horizon)")
    plt.ylabel("Price ($)")
    plt.axhline(last_price, color='magenta', linestyle='--', label=f'Last Price (${last_price:.2f})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    st.pyplot(fig)

    with st.echo():
        terminal_prices = [prices[-1] for prices in simulated_prices]
        prob_above = sum(1 for price in terminal_prices if price > last_price) / simulations
        st.write(f"Probability price increases: {prob_above:.2%}")

        # 5th and 95th percentiles (confidence intervals)
        CI = 0.9
        lower_bound = np.percentile(terminal_prices, (1 - CI) * 100 / 2)
        upper_bound = np.percentile(terminal_prices, 100 - (1 - CI) * 100 / 2)
        st.write(f"{CI:.0%} Confidence Interval: [${lower_bound:.2f}, ${upper_bound:.2f}]")

    with st.echo():
        mean_terminal = np.mean(terminal_prices)
        median_terminal = np.median(terminal_prices)
        std_terminal = np.std(terminal_prices)
        min_terminal = np.min(terminal_prices)
        max_terminal = np.max(terminal_prices)

        st.write(f"Mean terminal price: ${mean_terminal:.2f}")
        st.write(f"Median terminal price: ${median_terminal:.2f}")
        st.write(f"Std deviation: ${std_terminal:.2f}")
        st.write(f"Min terminal price: ${min_terminal:.2f}")
        st.write(f"Max terminal price: ${max_terminal:.2f}")


    fig = px.histogram(
        x=terminal_prices,
        nbins=100,
        title=f"<b>Terminal Price Distribution</b><br>Mean: ${mean_terminal:.2f} | 90% CI: [${lower_bound:.2f}, ${upper_bound:.2f}]",
        labels={'x': 'Price ($)'},
        color_discrete_sequence=['indianred']
    )

    fig.add_vline(x=last_price, line_dash="dot", line_color="purple",
                  annotation_text=f"Current Price", annotation_position="top")
    fig.add_vline(x=mean_terminal, line_dash="solid", line_color="green",
                  annotation_text=f"Mean", annotation_position="top right")

    fig.update_layout(
        hovermode="x unified",
        showlegend=False,
        xaxis_title="Terminal Price ($)",
        yaxis_title="Number of Simulations"
    )

    st.plotly_chart(fig)



elif selected == "Visual representations":

    def make_graph(stock_data, revenue_data, stock):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Historical Share Price", "Historical Revenue"), vertical_spacing=0.3)

        fig.add_trace(go.Scatter(x=pd.to_datetime(stock_data.Date),
                                 y=stock_data.Close.astype("float"),
                                 name="Share Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=pd.to_datetime(revenue_data.Date),
                                 y=revenue_data.Revenue.astype("float"),
                                 name="Revenue"), row=2, col=1)

        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($US)", row=1, col=1)
        fig.update_yaxes(title_text="Revenue ($US Millions)", row=2, col=1)
        fig.update_layout(showlegend=False, height=900, title=stock, xaxis_rangeslider_visible=True)

        st.plotly_chart(fig, use_container_width=True)

    tesla_revenue["Revenue"] = tesla_revenue["Revenue"].apply(clean_revenue)
    tesla_revenue.dropna(inplace=True)


    make_graph(tesla_data, tesla_revenue, "TSLA")

    apple = yf.Ticker("AAPL")
    apple_data = pd.DataFrame()
    apple_data = apple.history(period="max")
    apple_data.reset_index(inplace=True)

    url = "https://www.macrotrends.net/stocks/charts/AAPL/apple/revenue"
    headers = {"User-Agent": "Mozilla/5.0"}
    html_data = requests.get(url, headers=headers)
    soup = BeautifulSoup(html_data.content, "html.parser")
    html_data = pd.read_html(html_data.text)
    apple_revenue = html_data[1]
    apple_revenue.columns = ["Date", "Revenue"]

    apple_revenue["Revenue"] = apple_revenue["Revenue"].apply(clean_revenue)
    apple_revenue.dropna(inplace=True)

    make_graph(apple_data, apple_revenue, "AAPL")

    nvidia = yf.Ticker("NVDA")
    nvidia_data = pd.DataFrame()
    nvidia_data = nvidia.history(period="max")
    nvidia_data.reset_index(inplace=True)

    url = "https://www.macrotrends.net/stocks/charts/NVDA/nvidia/revenue"
    headers = {"User-Agent": "Mozilla/5.0"}
    html_data = requests.get(url, headers=headers)
    soup = BeautifulSoup(html_data.content, "html.parser")
    html_data = pd.read_html(html_data.text)
    nvidia_revenue = html_data[1]
    nvidia_revenue.columns = ["Date", "Revenue"]

    nvidia_revenue["Revenue"] = nvidia_revenue["Revenue"].apply(clean_revenue)
    nvidia_revenue.dropna(inplace=True)

    make_graph(nvidia_data, nvidia_revenue, "NVDA")

    amazon = yf.Ticker("AMZN")
    amazon_data = pd.DataFrame()
    amazon_data = amazon.history(period="max")
    amazon_data.reset_index(inplace=True)

    url = "https://www.macrotrends.net/stocks/charts/AMZN/amazon/revenue"
    headers = {"User-Agent": "Mozilla/5.0"}
    html_data = requests.get(url, headers=headers)
    soup = BeautifulSoup(html_data.content, "html.parser")
    html_data = pd.read_html(html_data.text)
    amazon_revenue = html_data[1]
    amazon_revenue.columns = ["Date", "Revenue"]

    amazon_revenue["Revenue"] = amazon_revenue["Revenue"].apply(clean_revenue)
    amazon_revenue.dropna(inplace=True)

    make_graph(amazon_data, amazon_revenue, "AMZN")


elif selected == "MongoDB":

    st.code("""
def df_to_mongodb(file, name, db_name='excel_database', connection_string='mongodb://localhost:27017/'):
    try:
        client = MongoClient(connection_string)
        db = client[db_name]

        collection_name = name
        collection = db[collection_name]

        df = file
        data = df.to_dict('records')

        collection.insert_many(data)

        print(f"Successfully uploaded {len(data)} records to collection '{collection_name}'")

    except Exception as e:
        print(f"Error uploading to MongoDB: {e}")
        return False 
    """)

    st.code("""
df_to_mongodb(tesla_data, "Tesla", db_name='manga')
df_to_mongodb(apple_data, "Apple", db_name='manga')
df_to_mongodb(nvidia_data, "Nvidia", db_name='manga')
df_to_mongodb(amazon_data, "Amazon", db_name='manga')
    """)

    st.code("""
Successfully uploaded 3730 records to collection 'Tesla'
Successfully uploaded 11183 records to collection 'Apple'
Successfully uploaded 6606 records to collection 'Nvidia'
Successfully uploaded 7031 records to collection 'Amazon'
    """)


elif selected == "Team Member":
    import pandas as pd

    Authers = {
        'Name': ["محمد مسعد نعيم محمد", "محمد عابدين محمود", "محمود رضا محمود ابوزيد", "فارس محمد حسن محمد"],
        'ID': [23011492, 23011477, 23011515, 23012193],
    }
    Authers = pd.DataFrame(Authers)
    Authers.index = Authers.index + 1


    with st.echo():
        Authers
