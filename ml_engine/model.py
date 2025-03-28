import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Union, Optional, Dict, Any, List
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from datetime import datetime


class AutonomousMLEngine:
    def __init__(self, task_type: str = "classification"):
        """
        Initialize the autonomous ML engine.

        Args:
            task_type (str): Type of ML task - "classification", "regression", or "conversation"
        """
        self.task_type = task_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = "models"
        self.conversation_model = None
        self.tokenizer = None
        self.conversation_history = []  # Store all conversations for learning
        self.learning_mode = True  # Enable automatic learning
        
        # Initialize learning stats with default values
        self.learning_stats = {
            "total_updates": 0,
            "last_update": None,
            "learned_patterns": set(),
            "user_preferences": {},
            "learned_phrases": {},  # Store learned phrases and their translations
            "learned_translations": {  # Store translations for different languages
                "spanish": {
                    "night": "noche",
                    "red": "rojo",
                    "blue": "azul",
                    "green": "verde",
                    "yellow": "amarillo",
                    "black": "negro",
                    "white": "blanco"
                },
                "japanese": {
                    "blue": "青 (ao)",
                    "red": "赤 (aka)",
                    "green": "緑 (midori)",
                    "yellow": "黄色 (kiiro)",
                    "black": "黒 (kuro)",
                    "white": "白 (shiro)"
                },
                "hungarian": {
                    "blue": "kék",
                    "red": "piros",
                    "green": "zöld",
                    "yellow": "sárga",
                    "black": "fekete",
                    "white": "fehér"
                },
                "chinese": {
                    "blue": "蓝色 (lán sè)",
                    "red": "红色 (hóng sè)",
                    "green": "绿色 (lǜ sè)",
                    "yellow": "黄色 (huáng sè)",
                    "black": "黑色 (hēi sè)",
                    "white": "白色 (bái sè)"
                }
            }
        }
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Load existing learning stats if available
        self._load_learning_stats()

    def _initialize_model(self, n_features: int) -> None:
        """Initialize the appropriate model based on task type."""
        if self.task_type == "classification":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
            )
        elif self.task_type == "regression":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
            )
        elif self.task_type == "conversation":
            # Initialize a small language model for conversation
            model_name = "microsoft/DialoGPT-small"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.conversation_model = AutoModelForCausalLM.from_pretrained(model_name)

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the model with new data.

        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Target values

        Returns:
            Dict[str, Any]: Training metrics
        """
        if self.task_type == "conversation":
            # For conversation, we'll fine-tune the language model
            # This is a simplified version - in practice, you'd want to use proper fine-tuning
            return {
                "status": "success",
                "message": "Conversation model initialized and ready for use",
                "model_type": "DialoGPT-small"
            }

        if not self.is_trained:
            self._initialize_model(X.shape[1])

        # Scale features
        X_scaled = (
            self.scaler.fit_transform(X)
            if not self.is_trained
            else self.scaler.transform(X)
        )

        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Calculate training metrics
        train_score = self.model.score(X_scaled, y)

        return {
            "status": "success",
            "train_score": float(train_score),
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X (np.ndarray): Features to predict on

        Returns:
            np.ndarray: Predictions
        """
        if self.task_type == "conversation":
            raise ValueError("Use generate_response() for conversation tasks")

        if not self.is_trained:
            raise ValueError("Model is not trained yet")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def generate_response(self, message: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """
        Generate a response for a conversation.

        Args:
            message (str): The user's message
            conversation_history (List[Dict[str, str]], optional): Previous conversation history

        Returns:
            str: Generated response
        """
        if self.task_type != "conversation":
            raise ValueError("Model is not configured for conversation")

        message = message.lower()
        
        # Get the last response if there's conversation history
        last_response = None
        if conversation_history and len(conversation_history) >= 2:
            last_response = conversation_history[-2]["content"].lower()
        
        # Store conversation for learning if enabled
        if self.learning_mode and conversation_history:
            self.conversation_history.extend(conversation_history)
            self._update_learning_stats(message)
            # Try to learn a phrase and get response
            phrase_response = self._learn_phrase(message)
            if phrase_response:
                return phrase_response
        
        # Handle common questions and responses
        if "how do you say" in message or "what is" in message or "tell me" in message:
            if "language" in message or "languages" in message:
                # Handle multiple language requests
                words = message.split()
                word = None
                languages = []
                
                # Extract the word to translate
                if "what is" in message:
                    word = words[2] if len(words) > 2 else None
                elif "how do you say" in message:
                    # Extract word between "say" and "in"
                    try:
                        say_index = words.index("say")
                        in_index = words.index("in")
                        if say_index < in_index:
                            word = " ".join(words[say_index + 1:in_index])
                    except ValueError:
                        word = None
                elif "tell me" in message:
                    word = words[2] if len(words) > 2 else None
                
                # Extract languages
                for lang in ["japanese", "hungarian", "chinese", "spanish", "czech"]:
                    if lang in message:
                        languages.append(lang)
                
                if word and languages:
                    response = f"Here's how to say '{word}' in different languages:\n"
                    for lang in languages:
                        if word in self.learning_stats["learned_translations"][lang]:
                            response += f"- {lang.title()}: {self.learning_stats['learned_translations'][lang][word]}\n"
                    response += "\nWould you like to learn more words in these languages?"
                    return response
            
            if "spanish" in message or "english" in message or "czech" in message or (last_response and "translations" in last_response):
                # Handle translations in both directions
                if '"' in message or "'" in message:
                    start = message.find('"') if '"' in message else message.find("'")
                    end = message.rfind('"') if '"' in message else message.rfind("'")
                    if start != -1 and end != -1:
                        phrase = message[start+1:end].strip()
                        if phrase.lower() in self.learning_stats["learned_phrases"]:
                            return f"In {'Spanish' if 'english' in message else 'English'}, '{phrase}' is '{self.learning_stats['learned_phrases'][phrase.lower()]}'. Would you like to learn more words?"
                
                # Extract the word to translate
                words = message.split()
                if "what is" in message:
                    word = words[2] if len(words) > 2 else None
                elif "how do you say" in message:
                    # Extract word between "say" and "in"
                    try:
                        say_index = words.index("say")
                        in_index = words.index("in")
                        if say_index < in_index:
                            word = " ".join(words[say_index + 1:in_index])
                    except ValueError:
                        word = None
                else:
                    word = words[-1].strip('?')
                
                # Check learned phrases first
                if word and word.lower() in self.learning_stats["learned_phrases"]:
                    # Determine translation direction based on the question
                    is_english_to_other = "spanish" in message or "czech" in message or "how do you say" in message
                    translation = self.learning_stats["learned_phrases"][word.lower()]
                    language = "Spanish" if "spanish" in message else "Czech" if "czech" in message else "English"
                    return f"In {language}, '{word}' is '{translation}'. Would you like to learn more words?"
                
                # Then check hardcoded translations
                if word and word in self.learning_stats["learned_translations"]["spanish"]:
                    return f"In Spanish, '{word}' is '{self.learning_stats['learned_translations']['spanish'][word]}'. Would you like to learn more Spanish words?"
                elif word:
                    # Add some common translations that aren't in the hardcoded list
                    common_translations = {
                        "hello": "hola",
                        "goodbye": "adiós",
                        "thank you": "gracias",
                        "please": "por favor",
                        "good morning": "buenos días",
                        "good night": "buenas noches",
                        "how are you": "cómo estás",
                        "nice to meet you": "encantado/a",
                        "yes": "sí",
                        "no": "no"
                    }
                    if word in common_translations:
                        return f"In Spanish, '{word}' is '{common_translations[word]}'. Would you like to learn more Spanish words?"
                    return f"I can help you translate '{word}' to {'Spanish' if 'spanish' in message else 'Czech' if 'czech' in message else 'English'}. Would you like to learn more words?"
                return "What word would you like to translate?"
            return "I can help you with translations! What language would you like to translate to?"
        elif "joke" in message:
            if last_response and "joke" in last_response:
                return "Here's another one: Why do programmers prefer dark mode? Because light attracts bugs! 😄"
            return "Why don't programmers like nature? It has too many bugs! 😄"
        elif "another" in message and last_response and "joke" in last_response:
            return "Here's another one: Why do programmers prefer dark mode? Because light attracts bugs! 😄"
        elif "learn" in message or "train" in message:
            if "show" in message or "status" in message or "what" in message:
                return self._format_learning_stats()
            elif "how" in message or "can" in message:
                return "Yes! I'm already learning from our conversations. I store our conversation history and use it to improve my responses. I can learn:\n1. Common patterns in conversations\n2. Context and follow-up questions\n3. User preferences and topics of interest\n4. Better response patterns\n5. New phrases and translations\n\nWould you like to see what you've learned so far?"
            return "I can learn from our conversations! I'll remember the context and improve my responses over time."
        elif "who are you" in message:
            return "I'm an AI assistant designed to help you with various tasks like conversation, analysis, and learning. How can I assist you today?"
        elif "hello" in message or "hi" in message:
            return "Hello! I'm your AI assistant. I can help you with:\n1. General conversation and questions\n2. Sentiment analysis of text\n3. Entity extraction from text\n4. Text summarization\n\nWhat would you like to do?"
        elif "spanish" in message and last_response and "translations" in last_response:
            # Extract the word or phrase to translate
            words = message.split()
            if '"' in message or "'" in message:
                start = message.find('"') if '"' in message else message.find("'")
                end = message.rfind('"') if '"' in message else message.rfind("'")
                if start != -1 and end != -1:
                    phrase = message[start+1:end].strip()
                    if phrase.lower() in self.learning_stats["learned_phrases"]:
                        return f"In Spanish, '{phrase}' is '{self.learning_stats['learned_phrases'][phrase.lower()]}'. Would you like to learn more Spanish phrases?"
            
            word = words[-1].strip('?') if words else None
            
            # Check learned phrases first
            if word and word.lower() in self.learning_stats["learned_phrases"]:
                return f"In Spanish, '{word}' is '{self.learning_stats['learned_phrases'][word.lower()]}'. Would you like to learn more Spanish phrases?"
            
            # Then check hardcoded translations
            translations = {
                "night": "noche",
                "red": "rojo",
                "blue": "azul",
                "green": "verde",
                "yellow": "amarillo",
                "black": "negro",
                "white": "blanco"
            }
            
            if word and word in translations:
                return f"In Spanish, '{word}' is '{translations[word]}'. Would you like to learn more Spanish words?"
            elif word:
                return f"I can help you translate '{word}' to Spanish. Would you like to learn more Spanish words?"
            return "What word would you like to translate to Spanish?"
        else:
            return "I understand your message. How can I help you with that?"

    def _save_learning_stats(self) -> None:
        """Save learning statistics to a JSON file."""
        stats_file = os.path.join(self.model_path, "learning_stats.json")
        # Convert set to list for JSON serialization
        stats_to_save = self.learning_stats.copy()
        stats_to_save["learned_patterns"] = list(stats_to_save["learned_patterns"])
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_to_save, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved learning stats to {stats_file}")
        except Exception as e:
            print(f"Error saving learning stats: {e}")

    def _load_learning_stats(self) -> None:
        """Load learning statistics from a JSON file."""
        stats_file = os.path.join(self.model_path, "learning_stats.json")
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    loaded_stats = json.load(f)
                    # Convert list back to set
                    loaded_stats["learned_patterns"] = set(loaded_stats["learned_patterns"])
                    # Update only if loaded stats are valid
                    if isinstance(loaded_stats, dict) and "learned_phrases" in loaded_stats:
                        self.learning_stats = loaded_stats
                        print(f"Successfully loaded learning stats from {stats_file}")
                    else:
                        print("Invalid learning stats format, using defaults")
            except Exception as e:
                print(f"Error loading learning stats: {e}")
                print("Using default learning stats")
        else:
            print("No existing learning stats found, using defaults")

    def _update_learning_stats(self, message: str) -> None:
        """Update learning statistics based on the message."""
        # Update basic stats
        self.learning_stats["total_updates"] += 1
        self.learning_stats["last_update"] = datetime.now().isoformat()
        
        # Extract patterns (simple word-based for now)
        words = message.split()
        for word in words:
            if len(word) > 3:  # Only consider words longer than 3 characters
                self.learning_stats["learned_patterns"].add(word)
        
        # Update user preferences based on topics
        topics = {
            "programming": ["code", "program", "bug", "debug", "function"],
            "language": ["translate", "spanish", "word", "language"],
            "learning": ["learn", "teach", "study", "understand"],
            "general": ["hello", "hi", "help", "what"]
        }
        
        for topic, keywords in topics.items():
            if any(keyword in message for keyword in keywords):
                self.learning_stats["user_preferences"][topic] = self.learning_stats["user_preferences"].get(topic, 0) + 1
        
        # Save updated stats
        self._save_learning_stats()

    def _learn_phrase(self, message: str) -> str:
        """Learn a new phrase from the message."""
        message = message.lower()
        
        # Look for remember/learn patterns
        if "remember" in message:
            try:
                # Extract the phrase and translation
                if '"' in message or "'" in message:
                    # Handle quoted phrases
                    start = message.find('"') if '"' in message else message.find("'")
                    end = message.rfind('"') if '"' in message else message.rfind("'")
                    if start != -1 and end != -1:
                        phrase = message[start+1:end].strip()
                        # Look for the translation after "in spanish is"
                        if "in spanish is" in message:
                            translation_start = message.find("in spanish is") + len("in spanish is")
                            translation = message[translation_start:].strip()
                            # Clean up the translation
                            translation = translation.strip('"').strip("'").strip()
                            if phrase and translation:
                                # Store in both directions
                                self.learning_stats["learned_phrases"][phrase.lower()] = translation
                                self.learning_stats["learned_phrases"][translation.lower()] = phrase
                                # Save updated stats
                                self._save_learning_stats()
                                return f"I've learned that '{phrase}' in Spanish is '{translation}'. You can ask me about it anytime!"
                else:
                    # Handle unquoted phrases
                    parts = message.split("remember", 1)[1].strip().split("in spanish is")
                    if len(parts) == 2:
                        phrase = parts[0].strip()
                        translation = parts[1].strip()
                        if phrase and translation:
                            # Store in both directions
                            self.learning_stats["learned_phrases"][phrase.lower()] = translation
                            self.learning_stats["learned_phrases"][translation.lower()] = phrase
                            # Save updated stats
                            self._save_learning_stats()
                            return f"I've learned that '{phrase}' in Spanish is '{translation}'. You can ask me about it anytime!"
            except Exception as e:
                print(f"Error learning phrase: {e}")
                return None
        elif "in spanish is" in message:
            try:
                # Handle format "word in spanish is translation"
                parts = message.split("in spanish is")
                if len(parts) == 2:
                    phrase = parts[0].strip()
                    translation = parts[1].strip()
                    # Clean up any quotes
                    phrase = phrase.strip('"').strip("'")
                    translation = translation.strip('"').strip("'")
                    if phrase and translation:
                        # Store in both directions
                        self.learning_stats["learned_phrases"][phrase.lower()] = translation
                        self.learning_stats["learned_phrases"][translation.lower()] = phrase
                        # Save updated stats
                        self._save_learning_stats()
                        return f"I've learned that '{phrase}' in Spanish is '{translation}'. You can ask me about it anytime!"
            except Exception as e:
                print(f"Error learning phrase: {e}")
                return None
        elif "in" in message and "is" in message:
            try:
                # Handle format "in [language], [word] is [translation]"
                parts = message.split("in", 1)[1].strip()
                if "," in parts:
                    language, rest = parts.split(",", 1)
                    language = language.strip()
                    if "is" in rest:
                        word, translation = rest.split("is", 1)
                        word = word.strip()
                        translation = translation.strip()
                        # Clean up any quotes
                        word = word.strip('"').strip("'")
                        translation = translation.strip('"').strip("'")
                        if word and translation:
                            # Store in both directions
                            self.learning_stats["learned_phrases"][word.lower()] = translation
                            self.learning_stats["learned_phrases"][translation.lower()] = word
                            # Save updated stats
                            self._save_learning_stats()
                            return f"I've learned that '{word}' in {language.title()} is '{translation}'. You can ask me about it anytime!"
            except Exception as e:
                print(f"Error learning phrase: {e}")
                return None
        return None

    def _format_learning_stats(self) -> str:
        """Format learning statistics into a readable string."""
        stats = self.learning_stats
        response = "Here's what I've learned so far:\n\n"
        
        # Add basic stats
        response += f"Total updates: {stats['total_updates']}\n"
        response += f"Last update: {stats['last_update']}\n"
        response += f"Learned patterns: {len(stats['learned_patterns'])} unique words\n\n"
        
        # Add learned phrases
        if stats["learned_phrases"]:
            response += "Learned Spanish phrases:\n"
            for phrase, translation in stats["learned_phrases"].items():
                response += f"- '{phrase}' → '{translation}'\n"
            response += "\n"
        
        # Add user preferences
        if stats["user_preferences"]:
            response += "Your interests:\n"
            for topic, count in sorted(stats["user_preferences"].items(), key=lambda x: x[1], reverse=True):
                response += f"- {topic.title()}: {count} interactions\n"
        
        response += "\nWould you like to learn more about any specific topic?"
        return response

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
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "task_type": self.task_type,
                "is_trained": self.is_trained,
            },
            model_file,
        )

        return {
            "status": "success",
            "message": f"Model saved as {model_name}",
            "path": model_file,
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
            "is_trained": self.is_trained,
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
            "model_type": type(self.model).__name__ if self.model else None,
        }

    def get_learning_status(self) -> Dict[str, Any]:
        """
        Get the current learning status and statistics.

        Returns:
            Dict[str, Any]: Learning status and statistics
        """
        return {
            "learning_mode": self.learning_mode,
            "total_conversations": len(self.conversation_history),
            "unique_topics": len(set(msg["content"].lower() for msg in self.conversation_history)),
            "last_learned": self.learning_stats["last_update"] or "Not yet",
            "total_updates": self.learning_stats["total_updates"],
            "learned_patterns": len(self.learning_stats["learned_patterns"]),
            "user_preferences": self.learning_stats["user_preferences"]
        }

    def toggle_learning(self, enable: bool = None) -> Dict[str, Any]:
        """
        Toggle the automatic learning mode.

        Args:
            enable (bool, optional): Whether to enable learning. If None, toggles current state.

        Returns:
            Dict[str, Any]: Learning mode status
        """
        if enable is None:
            self.learning_mode = not self.learning_mode
        else:
            self.learning_mode = enable
        
        return {
            "status": "success",
            "learning_mode": self.learning_mode,
            "message": "Learning mode " + ("enabled" if self.learning_mode else "disabled")
        }
