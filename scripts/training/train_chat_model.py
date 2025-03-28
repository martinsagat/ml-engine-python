from ml_engine.chat.model import ChatModel
import argparse

def train_chat_model(data_file):
    """Train the chat model with conversation data."""
    print(f"Training chat model with data from: {data_file}")
    
    # Initialize model
    model = ChatModel()
    
    # Train model
    result = model.train(data_file)
    
    if result['status'] == 'success':
        print("\nTraining Results:")
        print(f"Message: {result['message']}")
        print(f"Number of features: {result['n_features']}")
        
        # Save model
        save_result = model.save_model("chat_model")
        print(f"\n{save_result['message']}")
        
        # Test the model
        print("\nTesting the model with a sample input:")
        response = model.get_response("Hello, how are you?")
        print(f"Input: Hello, how are you?")
        print(f"Response: {response['response']}")
        print(f"Confidence: {response['confidence']:.4f}")
        
        return True
    else:
        print(f"\nError during training: {result['message']}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train chat model with conversation data')
    parser.add_argument('--data', type=str, default='data/conversations/training_data.json',
                      help='Path to JSON file containing conversation data')
    args = parser.parse_args()
    
    train_chat_model(args.data)

if __name__ == "__main__":
    main() 