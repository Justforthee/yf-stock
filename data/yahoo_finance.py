import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
def get_stock_data(ticker, period='1y'):
    """
    Fetch stock data from Yahoo Finance.
    
    :param ticker: Stock ticker symbol
    :param period: Time period to fetch data for (default is '1y' for 1 year)
    :return: DataFrame with stock data
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data available for ticker {ticker}")
        
        return df
    except Exception as e:
        raise ValueError(f"Error fetching data for ticker {ticker}: {str(e)}")

def get_top_stocks(n=20):
    # List of 20 popular stock tickers
    tickers = [
        "AAPL",  # Apple Inc.
        "MSFT",  # Microsoft Corporation
        "AMZN",  # Amazon.com Inc.
        "GOOGL", # Alphabet Inc. (Google)
        "FB",    # Meta Platforms Inc. (Facebook)
        "TSLA",  # Tesla Inc.
        "NVDA",  # NVIDIA Corporation
        "JPM",   # JPMorgan Chase & Co.
        "JNJ",   # Johnson & Johnson
        "V",     # Visa Inc.
        "PG",    # Procter & Gamble Co.
        "HD",    # The Home Depot Inc.
        "MA",    # Mastercard Incorporated
        "UNH",   # UnitedHealth Group Incorporated
        "BAC",   # Bank of America Corporation
        "DIS",   # The Walt Disney Company
        "ADBE",  # Adobe Inc.
        "NFLX",  # Netflix Inc.
        "CMCSA", # Comcast Corporation
        "XOM"    # Exxon Mobil Corporation
    ]
    
    return tickers[:n]
def get_real_time_price(ticker):
    """
    Get the latest real-time price for a given stock ticker.
    
    :param ticker: Stock ticker symbol
    :return: Latest price
    """
    try:
        stock = yf.Ticker(ticker)
        return stock.info['regularMarketPrice']
    except Exception as e:
        raise ValueError(f"Error fetching real-time price for ticker {ticker}: {str(e)}")