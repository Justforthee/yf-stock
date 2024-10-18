import os
from data.yahoo_finance import get_stock_data
from preprocessing.feature_engineering import engineer_features, prepare_data
from models.short_term_model import ShortTermModel
from models.long_term_model import LongTermModel
from dashboard.app import app
import config
import pandas as pd
import warnings
import numpy as np
import joblib

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")  # Ignore numpy warnings


def train_models():
    print("Training models...")
    try:
        # Fetch training data
        df = get_stock_data('AAPL', period='5y')  # Using Apple stock for training
        
        if df.empty:
            raise ValueError("No data fetched for AAPL")
        
        # Engineer features
        df_features = engineer_features(df)
        
        # Prepare data for training
        X_short, y_short = prepare_data(df_features, 'Target_Short', is_classification=True)
        X_long, y_long = prepare_data(df_features, 'Target_Long', is_classification=False)
        
        # Get feature columns
        feature_columns = df_features.drop(['Target_Short', 'Target_Long', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1).columns.tolist()
        
        # Train and save short-term model
        short_term_model = ShortTermModel()
        short_term_model.train(X_short, y_short)
        if not short_term_model.is_fitted:
            raise ValueError("Short-term model failed to train properly")
        short_term_model.save('short_term_model.joblib')
        
        # Train and save long-term model
        long_term_model = LongTermModel()
        long_term_model.train(X_long, y_long)
        if not long_term_model.is_fitted:
            raise ValueError("Long-term model failed to train properly")
        long_term_model.save('long_term_model.joblib')
        
        # Save feature columns
        joblib.dump(feature_columns, 'feature_columns.joblib')
        
        print("Models and feature columns saved successfully.")
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

def models_exist():
    return os.path.exists('short_term_model.joblib') and os.path.exists('long_term_model.joblib')

if __name__ == "__main__":
    if not models_exist():
        train_models()
    app.run_server(debug=True)