import pandas as pd
import numpy as np
from ta import add_all_ta_features
import warnings
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """
    Engineer features for the stock prediction model.
    
    :param df: DataFrame with stock data
    :return: DataFrame with engineered features
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Suppress specific warnings from TA library
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        # Add technical indicators
        df = add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )
    
    # Add rolling statistics
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    
    # Add price momentum
    df['Price_Momentum'] = df['Close'] / df['Close'].shift(1) - 1
    
    # Add target variable (next day's price direction)
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    
    # Fill NaN values
    df = df.ffill().bfill()
    
    # Replace infinity values with NaN and then fill them
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    if df.empty:
        raise ValueError("DataFrame is empty after feature engineering")
    
    return df