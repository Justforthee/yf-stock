import os
from data.yahoo_finance import get_stock_data
from preprocessing.feature_engineering import engineer_features
from models.short_term_model import ShortTermModel
from models.long_term_model import LongTermModel
from dashboard.app import app
import config
import pandas as pd
import warnings
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.seterr(all="ignore")  # Ignore numpy warnings

def train_models():
    print("Training models...")
    try:
        # Fetch training data
        df = get_stock_data('PLTR', period='5y')  # Using PLTR stock for training
        
        if df.empty:
            raise ValueError("No data fetched for PLTR")
        
        # Engineer features
        df_features = engineer_features(df)
        
        # Prepare data for training
        X = df_features.drop(['Target'], axis=1)
        y_short = df_features['Target']
        
        # For long-term model, we'll predict the prices for the next 30 days
        y_long = pd.DataFrame({f'day_{i}': df_features['Close'].shift(-i) for i in range(1, 31)})
        
        # Train and save short-term model
        short_term_model = ShortTermModel()
        short_term_model.train(X, y_short)
        short_term_model.save('short_term_model.joblib')
        
        # Train and save long-term model
        long_term_model = LongTermModel()
        long_term_model.train(X.iloc[:-30], y_long.iloc[:-30])
        long_term_model.save('long_term_model.joblib')
        
        print("Models trained and saved successfully.")
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    # Always train new models
    train_models()
    app.run_server(debug=True)