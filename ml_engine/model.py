import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Union, Optional, Dict, Any
import os

class AutonomousMLEngine:
    def __init__(self, task_type: str = "classification"):
        """
        Initialize the autonomous ML engine.
        
        Args:
            task_type (str): Type of ML task - "classification" or "regression"
        """
        self.task_type = task_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = "models"
        os.makedirs(self.model_path, exist_ok=True)
        
    def _initialize_model(self, n_features: int) -> None:
        """Initialize the appropriate model based on task type."""
        if self.task_type == "classification":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the model with new data.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Target values
            
        Returns:
            Dict[str, Any]: Training metrics
        """
        if not self.is_trained:
            self._initialize_model(X.shape[1])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X) if not self.is_trained else self.scaler.transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training metrics
        train_score = self.model.score(X_scaled, y)
        
        return {
            "status": "success",
            "train_score": float(train_score),
            "n_samples": X.shape[0],
            "n_features": X.shape[1]
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Features to predict on
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, model_name: str) -> Dict[str, Any]:
        """
        Save the current model state.
        
        Args:
            model_name (str): Name to save the model as
            
        Returns:
            Dict[str, Any]: Save status
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_file = os.path.join(self.model_path, f"{model_name}.joblib")
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "task_type": self.task_type,
            "is_trained": self.is_trained
        }, model_file)
        
        return {
            "status": "success",
            "message": f"Model saved as {model_name}",
            "path": model_file
        }
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """
        Load a saved model.
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            Dict[str, Any]: Load status
        """
        model_file = os.path.join(self.model_path, f"{model_name}.joblib")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model {model_name} not found")
        
        saved_data = joblib.load(model_file)
        self.model = saved_data["model"]
        self.scaler = saved_data["scaler"]
        self.task_type = saved_data["task_type"]
        self.is_trained = saved_data["is_trained"]
        
        return {
            "status": "success",
            "message": f"Model {model_name} loaded successfully",
            "task_type": self.task_type,
            "is_trained": self.is_trained
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current model status.
        
        Returns:
            Dict[str, Any]: Model status
        """
        return {
            "is_trained": self.is_trained,
            "task_type": self.task_type,
            "model_type": type(self.model).__name__ if self.model else None
        } 