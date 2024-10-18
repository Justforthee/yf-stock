from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np
import joblib

class LongTermModel:
    def __init__(self):
        self.models = [
            RandomForestRegressor(n_estimators=100, random_state=42),
            XGBRegressor(n_estimators=100, random_state=42),
            LGBMRegressor(n_estimators=100, random_state=42)
        ]
        self.is_fitted = False
        
    def train(self, X, y):
        """
        Train the long-term prediction models.
        
        :param X: Feature matrix
        :param y: Target vector
        """
        for model in self.models:
            model.fit(X, y)
        
        self.is_fitted = True
        
        # Evaluate the models using cross-validation
        cv_scores = [cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error') for model in self.models]
        avg_cv_score = np.mean([-np.mean(scores) for scores in cv_scores])
        print(f"Long-term model average CV MSE: {avg_cv_score:.2f}")
        
    def predict(self, X):
        """
        Make price predictions using the trained models.
        
        :param X: Feature matrix
        :return: Price predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'train' before using this model.")
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)
    
    def save(self, filename):
        """Save the models to a file."""
        joblib.dump(self, filename)
    
    @classmethod
    def load(cls, filename):
        """Load the models from a file."""
        return joblib.load(filename)