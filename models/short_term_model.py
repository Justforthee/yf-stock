from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
import numpy as np

class ShortTermModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = None
        
    def train(self, X, y):
        """
        Train the short-term prediction model.
        
        :param X: Feature matrix
        :param y: Target vector
        """
        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Short-term model accuracy: {accuracy:.2f}")
        
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        :param X: Feature matrix
        :return: Predictions
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names]  # Select only the features the model was trained on
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.shape[1] > len(self.feature_names):
                X = X[:, :len(self.feature_names)]
        return self.model.predict(X)
    
    def save(self, filename):
        """Save the model and feature names to a file."""
        joblib.dump({'model': self.model, 'feature_names': self.feature_names}, filename)
    
    def load(self, filename):
        """Load the model and feature names from a file."""
        data = joblib.load(filename)
        self.model = data['model']
        self.feature_names = data['feature_names']