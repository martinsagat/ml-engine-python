import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import joblib

class ChatModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.conversation_pairs = []
        self.tfidf_matrix = None
        self.is_trained = False
        self.model_path = "models"
        os.makedirs(self.model_path, exist_ok=True)

    def train(self, json_file_path):
        """Train the chat model with conversation data."""
        try:
            # Load conversation data
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            # Extract conversations
            inputs = []
            self.conversation_pairs = []
            
            for conv in data['conversations']:
                inputs.append(conv['input'])
                self.conversation_pairs.append({
                    'input': conv['input'],
                    'response': conv['response']
                })
            
            # Create TF-IDF matrix
            self.tfidf_matrix = self.vectorizer.fit_transform(inputs)
            self.is_trained = True
            
            return {
                'status': 'success',
                'message': f'Trained on {len(inputs)} conversations',
                'n_features': self.tfidf_matrix.shape[1]
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def get_response(self, user_input):
        """Get a response for the user input."""
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        # Transform user input
        user_vector = self.vectorizer.transform([user_input])
        
        # Calculate similarities
        similarities = cosine_similarity(user_vector, self.tfidf_matrix)[0]
        
        # Find most similar conversation
        most_similar_idx = np.argmax(similarities)
        similarity_score = similarities[most_similar_idx]
        
        # Get response
        response = self.conversation_pairs[most_similar_idx]['response']
        
        return {
            'response': response,
            'confidence': float(similarity_score)
        }

    def save_model(self, model_name):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_file = os.path.join(self.model_path, f"{model_name}.joblib")
        model_data = {
            'vectorizer': self.vectorizer,
            'conversation_pairs': self.conversation_pairs,
            'tfidf_matrix': self.tfidf_matrix,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, model_file)
        
        return {
            'status': 'success',
            'message': f'Model saved as {model_name}',
            'path': model_file
        }

    def load_model(self, model_name):
        """Load a saved model."""
        model_file = os.path.join(self.model_path, f"{model_name}.joblib")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model {model_name} not found")
        
        model_data = joblib.load(model_file)
        
        self.vectorizer = model_data['vectorizer']
        self.conversation_pairs = model_data['conversation_pairs']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.is_trained = model_data['is_trained']
        
        return {
            'status': 'success',
            'message': f'Model {model_name} loaded successfully'
        } 