from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
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
        tscv = TimeSeriesSplit(n_splits=5)
        for model in self.models:
            model.fit(X, y)
        
        self.is_fitted = True

        # Evaluate using time series cross-validation
        cv_scores = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            for model in self.models:
                model.fit(X_train, y_train)
            
            pred = self.predict(X_test)
            cv_scores.append(mean_squared_error(y_test, pred, squared=False))
        
        print(f"Long-term model average CV RMSE: {np.mean(cv_scores):.4f}")
        
    def predict(self, X):
        """
        Make price predictions using the trained models.
        
        :param X: Feature matrix
        :return: Price predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'train' before using this model.")
        predictions = np.array([model.predict(X) for model in self.models])
        avg_prediction = np.mean(predictions, axis=0)

        # Constrain predictions to a maximum of 20% change per month
        constrained_prediction = np.clip(avg_prediction, -0.2, 0.2)
        return constrained_prediction
    
    def save(self, filename):
        """Save the models to a file."""
        joblib.dump(self, filename)
    
    @classmethod
    def load(cls, filename):
        """Load the models from a file."""
        return joblib.load(filename)