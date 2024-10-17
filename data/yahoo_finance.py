import yfinance as yf
from datetime import datetime, timedelta

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