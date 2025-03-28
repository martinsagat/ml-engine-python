import numpy as np
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional

class CarPredictionInterface:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url

    def create_sample_car_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Create sample car data with realistic features."""
        np.random.seed(42)
        
        # Generate realistic car data
        data = {
            'mileage': np.random.normal(50000, 20000, n_samples).clip(0),  # km
            'age': np.random.randint(0, 20, n_samples),  # years
            'engine_size': np.random.choice([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0], n_samples),
            'horsepower': np.random.normal(120, 40, n_samples).clip(60, 300),
            'fuel_efficiency': np.random.normal(7.5, 1.5, n_samples).clip(4, 12),  # L/100km
            'price': np.zeros(n_samples)  # Will be calculated based on features
        }
        
        # Calculate price based on features (with some randomness)
        base_price = 30000
        price = (
            base_price
            - data['mileage'] * 0.1  # Price decreases with mileage
            - data['age'] * 1000     # Price decreases with age
            + data['engine_size'] * 2000  # Price increases with engine size
            + data['horsepower'] * 50     # Price increases with horsepower
            - data['fuel_efficiency'] * 500  # Price decreases with fuel consumption
            + np.random.normal(0, 2000, n_samples)  # Add some randomness
        ).clip(5000, 100000)  # Ensure prices are realistic
        
        data['price'] = price
        return pd.DataFrame(data)

    def train_model(self, n_samples: int = 1000) -> Dict:
        """Train a model to predict car prices."""
        print("Generating sample car data...")
        df = self.create_sample_car_data(n_samples)
        
        # Split features and target
        X = df[['mileage', 'age', 'engine_size', 'horsepower', 'fuel_efficiency']].values
        y = df['price'].values
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("\nTraining car price prediction model...")
        # Prepare data for API
        data = {
            "features": X_train.tolist(),
            "targets": y_train.tolist(),
            "task_type": "regression"
        }
        
        # Train model
        response = requests.post(f"{self.api_url}/train", json=data)
        training_response = response.json()
        print("Training response:", training_response)
        
        # Make predictions on test set
        print("\nMaking predictions on test set...")
        pred_data = {"features": X_test.tolist()}
        response = requests.post(f"{self.api_url}/predict", json=pred_data)
        predictions = response.json()["predictions"]
        
        # Calculate metrics
        mse = np.mean((np.array(predictions) - y_test) ** 2)
        rmse = np.sqrt(mse)
        print(f"\nTest Set RMSE: ${rmse:.2f}")
        
        # Save model
        print("\nSaving model...")
        response = requests.post(f"{self.api_url}/model/save", json={"name": "car_price_model"})
        save_response = response.json()
        print("Save response:", save_response)
        
        return {
            "training_response": training_response,
            "save_response": save_response,
            "rmse": rmse,
            "predictions": predictions,
            "actual_values": y_test.tolist()
        }

    def predict_price(self, car_features: List[float]) -> float:
        """Predict the price for a single car based on its features."""
        pred_data = {"features": [car_features]}
        response = requests.post(f"{self.api_url}/predict", json=pred_data)
        return response.json()["predictions"][0]

    def get_example_predictions(self, n_examples: int = 5) -> List[Dict]:
        """Get example predictions for random cars."""
        df = self.create_sample_car_data(n_examples)
        X = df[['mileage', 'age', 'engine_size', 'horsepower', 'fuel_efficiency']].values
        y = df['price'].values
        
        predictions = []
        for i in range(n_examples):
            pred = self.predict_price(X[i].tolist())
            predictions.append({
                "actual_price": float(y[i]),
                "predicted_price": float(pred),
                "features": {
                    "mileage": float(X[i][0]),
                    "age": float(X[i][1]),
                    "engine_size": float(X[i][2]),
                    "horsepower": float(X[i][3]),
                    "fuel_efficiency": float(X[i][4])
                }
            })
        return predictions 