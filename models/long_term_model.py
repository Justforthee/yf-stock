from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import joblib
import pandas as pd

class LongTermModel:
    def __init__(self):
        self.models = [GradientBoostingRegressor(n_estimators=100, random_state=42) for _ in range(30)]
        self.feature_names = None
        
    def train(self, X, y):
        """
        Train the long-term prediction models.
        
        :param X: Feature matrix
        :param y: Target DataFrame with 30 columns for 30 days of future prices
        """
        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for i in range(30):
            self.models[i].fit(X_train, y_train.iloc[:, i])
        
        # Evaluate the models
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        print(f"Long-term model MSE: {mse:.2f}")
        print(f"Long-term model MAPE: {mape:.2%}")
        
    def predict(self, X):
        """
        Make price predictions using the trained models.
        
        :param X: Feature matrix
        :return: Price predictions for 30 days
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names]  # Select only the features the model was trained on
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.shape[1] > len(self.feature_names):
                X = X[:, :len(self.feature_names)]
        
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return predictions
    
    def save(self, filename):
        """Save the models and feature names to a file."""
        joblib.dump({'models': self.models, 'feature_names': self.feature_names}, filename)
    
    def load(self, filename):
        """Load the models and feature names from a file."""
        data = joblib.load(filename)
        self.models = data['models']
        self.feature_names = data['feature_names']