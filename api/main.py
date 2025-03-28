from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from ml_engine.model import AutonomousMLEngine

app = FastAPI(
    title="Autonomous ML Engine API",
    description="A REST API for the autonomous machine learning engine",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML engine
ml_engine = AutonomousMLEngine()

class TrainingData(BaseModel):
    features: List[List[float]]
    targets: List[float]
    task_type: str = "classification"

class PredictionData(BaseModel):
    features: List[List[float]]

class ModelName(BaseModel):
    name: str

@app.post("/train")
async def train_model(data: TrainingData):
    """Train the model with provided data."""
    try:
        X = np.array(data.features)
        y = np.array(data.targets)
        
        # Update task type if provided
        if data.task_type != ml_engine.task_type:
            ml_engine.task_type = data.task_type
        
        result = ml_engine.train(X, y)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
async def predict(data: PredictionData):
    """Make predictions using the trained model."""
    try:
        X = np.array(data.features)
        predictions = ml_engine.predict(X)
        return {
            "predictions": predictions.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/status")
async def get_model_status():
    """Get the current model status."""
    return ml_engine.get_status()

@app.post("/model/save")
async def save_model(model_name: ModelName):
    """Save the current model state."""
    try:
        return ml_engine.save_model(model_name.name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/model/load")
async def load_model(model_name: ModelName):
    """Load a saved model."""
    try:
        return ml_engine.load_model(model_name.name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 