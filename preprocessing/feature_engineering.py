import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.preprocessing import StandardScaler
import warnings

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
    for window in [5, 10, 20]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'STD_{window}'] = df['Close'].rolling(window=window).std()
    
    # Add price momentum
    df['Price_Momentum'] = df['Close'] / df['Close'].shift(1) - 1
    
    # Add lag features
    for lag in [1, 5, 10]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
    
    # Add target variables
    df['Target_Short'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df['Target_Long'] = df['Close'].shift(-30) / df['Close'] - 1
    
    # Fill NaN values
    df = df.ffill().bfill()
    
    # Replace infinity values with NaN and then fill them
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    if df.empty:
        raise ValueError("DataFrame is empty after feature engineering")
    
    return df

def prepare_data(df, target_column, is_classification=True, feature_columns=None):
    """
    Prepare data for model training or prediction.
    
    :param df: DataFrame with engineered features
    :param target_column: Name of the target column
    :param is_classification: Whether it's a classification task
    :param feature_columns: List of feature column names to use (for prediction)
    :return: X (features) and y (target) as numpy arrays
    """
    if feature_columns is None:
        features = df.drop(['Target_Short', 'Target_Long', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    else:
        features = df[feature_columns]
    
    if target_column in df.columns:
        target = df[target_column]
        if is_classification:
            target = target.astype(int)
        y = target.values
    else:
        y = None
    
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    
    return X, y