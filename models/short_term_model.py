from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

class ShortTermModel:
    def __init__(self):
        self.models = [
            RandomForestClassifier(n_estimators=100, random_state=42),
            XGBClassifier(n_estimators=100, random_state=42),
            LGBMClassifier(n_estimators=100, random_state=42)
        ]
        self.is_fitted = False
        
    def train(self, X, y):
        """
        Train the short-term prediction model.
        
        :param X: Feature matrix
        :param y: Target vector
        """
        for model in self.models:
            model.fit(X, y)
        
        self.is_fitted = True
        
        # Evaluate the model using cross-validation
        cv_scores = [cross_val_score(model, X, y, cv=5, scoring='accuracy') for model in self.models]
        avg_cv_score = np.mean([np.mean(scores) for scores in cv_scores])
        print(f"Short-term model average CV accuracy: {avg_cv_score:.2f}")
        
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        :param X: Feature matrix
        :return: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'train' before using this model.")
        predictions = np.array([model.predict(X) for model in self.models])
        return np.round(np.mean(predictions, axis=0)).astype(int)
    
    def save(self, filename):
        """Save the model to a file."""
        joblib.dump(self, filename)
    
    @classmethod
    def load(cls, filename):
        """Load the model from a file."""
        return joblib.load(filename)