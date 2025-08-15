import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

class PropertyPricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        
    def preprocess_data(self, df):
        """Preprocess the property data for training or prediction"""
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Handle categorical variables
        categorical_cols = ['location', 'property_type', 'amenities']
        
        for col in categorical_cols:
            if col in data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
                else:
                    # Handle unseen categories during prediction
                    data[col] = data[col].astype(str)
                    unique_values = set(data[col].unique())
                    known_values = set(self.label_encoders[col].classes_)
                    
                    # Replace unknown values with most frequent known value
                    if not unique_values.issubset(known_values):
                        most_frequent = self.label_encoders[col].classes_[0]
                        data[col] = data[col].apply(lambda x: x if x in known_values else most_frequent)
                    
                    data[col] = self.label_encoders[col].transform(data[col])
        
        return data
    
    def train(self, df, target_column='price'):
        """Train the model with property data"""
        # Preprocess data
        processed_data = self.preprocess_data(df)
        
        # Separate features and target
        X = processed_data.drop(columns=[target_column])
        y = processed_data[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        predictions = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        self.is_trained = True
        
        return {
            'mae': mae,
            'r2_score': r2,
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
    
    def predict(self, property_data):
        """Predict property price"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess input data
        processed_data = self.preprocess_data(property_data)
        
        # Scale features
        scaled_data = self.scaler.transform(processed_data)
        
        # Make prediction
        prediction = self.model.predict(scaled_data)
        
        return prediction
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.is_trained = model_data['is_trained']
            return True
        return False
