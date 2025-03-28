import requests
from typing import Dict, List, Optional

class ConversationInterface:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url

    def start_conversation(self, initial_message: str) -> Dict:
        """Start a new conversation with the model."""
        data = {
            "message": initial_message,
            "conversation_id": None  # New conversation
        }
        response = requests.post(f"{self.api_url}/chat", json=data)
        return response.json()

    def continue_conversation(self, message: str, conversation_id: str) -> Dict:
        """Continue an existing conversation."""
        data = {
            "message": message,
            "conversation_id": conversation_id
        }
        response = requests.post(f"{self.api_url}/chat", json=data)
        return response.json()

    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """Get the history of a conversation."""
        response = requests.get(f"{self.api_url}/chat/history/{conversation_id}")
        return response.json()

    def end_conversation(self, conversation_id: str) -> Dict:
        """End a conversation and clean up resources."""
        response = requests.delete(f"{self.api_url}/chat/{conversation_id}")
        return response.json()

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze the sentiment of a piece of text."""
        data = {"text": text}
        response = requests.post(f"{self.api_url}/analyze/sentiment", json=data)
        return response.json()

    def extract_entities(self, text: str) -> Dict:
        """Extract named entities from text."""
        data = {"text": text}
        response = requests.post(f"{self.api_url}/analyze/entities", json=data)
        return response.json()

    def summarize_text(self, text: str, max_length: Optional[int] = None) -> Dict:
        """Generate a summary of the text."""
        data = {
            "text": text,
            "max_length": max_length
        }
        response = requests.post(f"{self.api_url}/analyze/summarize", json=data)
        return response.json() 