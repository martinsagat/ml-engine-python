from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
from ml_engine.model import AutonomousMLEngine
import uuid

app = FastAPI(
    title="Autonomous ML Engine API",
    description="A REST API for the autonomous machine learning engine",
    version="1.0.0",
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
ml_engine = AutonomousMLEngine(task_type="conversation")

# Store conversations
conversations = {}

class TrainingData(BaseModel):
    features: List[List[float]]
    targets: List[float]
    task_type: str = "classification"

class PredictionData(BaseModel):
    features: List[List[float]]

class ModelName(BaseModel):
    name: str

class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class TextAnalysis(BaseModel):
    text: str
    max_length: Optional[int] = None

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
        return {"predictions": predictions.tolist()}
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

@app.post("/chat")
async def chat(data: ChatMessage):
    """Start or continue a conversation."""
    try:
        if not data.conversation_id:
            # Start new conversation
            conversation_id = str(uuid.uuid4())
            conversations[conversation_id] = []
            response = "Hello! I'm your AI assistant. I can help you with:\n" + \
                      "1. General conversation and questions\n" + \
                      "2. Sentiment analysis of text\n" + \
                      "3. Entity extraction from text\n" + \
                      "4. Text summarization\n\n" + \
                      "What would you like to do?"
        else:
            conversation_id = data.conversation_id
            if conversation_id not in conversations:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            # Get conversation history
            history = conversations[conversation_id]
            
            # Generate response using the ML engine
            response = ml_engine.generate_response(data.message, history)
        
        # Add message to conversation history
        conversations[conversation_id].append({
            "role": "user",
            "content": data.message
        })
        
        # Add response to conversation history
        conversations[conversation_id].append({
            "role": "assistant",
            "content": response
        })
        
        return {
            "conversation_id": conversation_id,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/chat/history/{conversation_id}")
async def get_chat_history(conversation_id: str):
    """Get the history of a conversation."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversations[conversation_id]

@app.delete("/chat/{conversation_id}")
async def end_chat(conversation_id: str):
    """End a conversation and clean up resources."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    del conversations[conversation_id]
    return {"status": "success"}

@app.post("/analyze/sentiment")
async def analyze_sentiment(data: TextAnalysis):
    """Analyze the sentiment of text."""
    try:
        # Placeholder - replace with actual sentiment analysis
        return {"sentiment": "positive", "score": 0.8}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/entities")
async def extract_entities(data: TextAnalysis):
    """Extract named entities from text."""
    try:
        # Placeholder - replace with actual entity extraction
        return {"entities": ["example", "test"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/summarize")
async def summarize_text(data: TextAnalysis):
    """Generate a summary of text."""
    try:
        # Placeholder - replace with actual text summarization
        return {"summary": data.text[:100] + "..."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model/learning/status")
async def get_learning_status():
    """Get the current learning status and statistics."""
    return ml_engine.get_learning_status()

@app.post("/model/learning/toggle")
async def toggle_learning(enable: Optional[bool] = None):
    """Toggle the automatic learning mode."""
    return ml_engine.toggle_learning(enable)
