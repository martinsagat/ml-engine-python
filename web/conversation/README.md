# AI Conversation Web Interface

A modern web interface for interacting with the AI conversation model. This interface provides a chat-like experience with additional features for text analysis.

## Features

- Real-time chat interface
- Sentiment analysis
- Entity extraction
- Text summarization
- Responsive design
- Modern UI with Tailwind CSS

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the Flask application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Start a conversation by typing a message and pressing Enter or clicking the send button
2. Continue the conversation by sending more messages
3. Use the analysis tools below the chat:
   - Analyze Sentiment: Get the emotional tone of the text
   - Extract Entities: Identify named entities in the text
   - Summarize: Generate a summary of the text

## Security Note

Make sure to change the `secret_key` in `app.py` to a secure value before deploying to production.

## Development

The web interface is built with:
- Flask for the backend
- Tailwind CSS for styling
- Font Awesome for icons
- Vanilla JavaScript for frontend functionality

## API Endpoints

- `/start_conversation`: Start a new conversation
- `/continue_conversation`: Continue an existing conversation
- `/end_conversation`: End the current conversation
- `/analyze_sentiment`: Analyze text sentiment
- `/extract_entities`: Extract named entities
- `/summarize`: Generate text summary 