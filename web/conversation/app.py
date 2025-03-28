import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from interfaces.conversation import ConversationInterface
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Initialize the conversation interface
conv_interface = ConversationInterface()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    message = request.json.get('message', '')
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    response = conv_interface.start_conversation(message)
    session['conversation_id'] = response['conversation_id']
    return jsonify(response)

@app.route('/continue_conversation', methods=['POST'])
def continue_conversation():
    message = request.json.get('message', '')
    conversation_id = session.get('conversation_id')
    
    if not message or not conversation_id:
        return jsonify({'error': 'Message and conversation ID are required'}), 400
    
    response = conv_interface.continue_conversation(message, conversation_id)
    return jsonify(response)

@app.route('/end_conversation', methods=['POST'])
def end_conversation():
    conversation_id = session.get('conversation_id')
    if not conversation_id:
        return jsonify({'error': 'No active conversation'}), 400
    
    response = conv_interface.end_conversation(conversation_id)
    session.pop('conversation_id', None)
    return jsonify(response)

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    text = request.json.get('text', '')
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    response = conv_interface.analyze_sentiment(text)
    return jsonify(response)

@app.route('/extract_entities', methods=['POST'])
def extract_entities():
    text = request.json.get('text', '')
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    response = conv_interface.extract_entities(text)
    return jsonify(response)

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.json.get('text', '')
    max_length = request.json.get('max_length')
    
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    response = conv_interface.summarize_text(text, max_length)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 