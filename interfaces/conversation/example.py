from interface import ConversationInterface

def main():
    # Initialize the interface
    conv_interface = ConversationInterface()
    
    # Start a new conversation
    print("Starting a new conversation...")
    response = conv_interface.start_conversation("Hello! I'd like to learn more about machine learning.")
    conversation_id = response["conversation_id"]
    print(f"Model: {response['response']}")
    
    # Continue the conversation
    print("\nUser: What are the main types of machine learning?")
    response = conv_interface.continue_conversation(
        "What are the main types of machine learning?",
        conversation_id
    )
    print(f"Model: {response['response']}")
    
    # Analyze sentiment of the conversation
    print("\nAnalyzing sentiment of the conversation...")
    sentiment = conv_interface.analyze_sentiment(response['response'])
    print(f"Sentiment analysis: {sentiment}")
    
    # Extract entities
    print("\nExtracting entities from the conversation...")
    entities = conv_interface.extract_entities(response['response'])
    print(f"Extracted entities: {entities}")
    
    # Get conversation history
    print("\nRetrieving conversation history...")
    history = conv_interface.get_conversation_history(conversation_id)
    print("Conversation history:")
    for message in history:
        print(f"{message['role']}: {message['content']}")
    
    # End the conversation
    print("\nEnding conversation...")
    conv_interface.end_conversation(conversation_id)
    print("Conversation ended.")

if __name__ == "__main__":
    main() 