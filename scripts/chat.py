from ml_engine.chat.model import ChatModel
import sys

def chat():
    """Start an interactive chat session."""
    print("Initializing chat model...")
    model = ChatModel()
    
    try:
        # Load the trained model
        model.load_model("chat_model")
        print("\nChat model loaded successfully!")
    except FileNotFoundError:
        print("\nNo trained model found. Please train the model first using:")
        print("python scripts/training/train_chat_model.py")
        sys.exit(1)
    
    print("\nChat started! Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for quit command
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nGoodbye! Have a great day!")
            break
        
        # Get model response
        try:
            result = model.get_response(user_input)
            response = result['response']
            confidence = result['confidence']
            
            print(f"\nBot: {response}")
            print(f"(confidence: {confidence:.4f})")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            break

if __name__ == "__main__":
    chat() 