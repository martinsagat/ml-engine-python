import numpy as np
import requests
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# API endpoint
API_URL = "http://localhost:8000"

def train_classification_example():
    """Example of training a classification model."""
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5)
    
    # Prepare data for API
    data = {
        "features": X.tolist(),
        "targets": y.tolist(),
        "task_type": "classification"
    }
    
    # Train model
    response = requests.post(f"{API_URL}/train", json=data)
    print("Training response:", response.json())
    
    # Make predictions
    X_test = X[:5]  # Use first 5 samples for testing
    pred_data = {"features": X_test.tolist()}
    response = requests.post(f"{API_URL}/predict", json=pred_data)
    print("Predictions:", response.json())
    
    # Save model
    response = requests.post(f"{API_URL}/model/save", json={"name": "classification_model"})
    print("Save response:", response.json())

def train_regression_example():
    """Example of training a regression model."""
    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
    
    # Prepare data for API
    data = {
        "features": X.tolist(),
        "targets": y.tolist(),
        "task_type": "regression"
    }
    
    # Train model
    response = requests.post(f"{API_URL}/train", json=data)
    print("Training response:", response.json())
    
    # Make predictions
    X_test = X[:5]  # Use first 5 samples for testing
    pred_data = {"features": X_test.tolist()}
    response = requests.post(f"{API_URL}/predict", json=pred_data)
    print("Predictions:", response.json())
    
    # Save model
    response = requests.post(f"{API_URL}/model/save", json={"name": "regression_model"})
    print("Save response:", response.json())

if __name__ == "__main__":
    print("Starting ML Engine examples...")
    
    print("\nTraining classification model...")
    train_classification_example()
    
    print("\nTraining regression model...")
    train_regression_example()
    
    print("\nExamples completed!") 