import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from flask import Flask, render_template, request, jsonify
from interfaces.car_prediction import CarPredictionInterface

app = Flask(__name__)

# Initialize the car prediction interface
car_interface = CarPredictionInterface()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = [
            float(data['mileage']),
            float(data['age']),
            float(data['engine_size']),
            float(data['horsepower']),
            float(data['fuel_efficiency'])
        ]
        
        predicted_price = car_interface.predict_price(features)
        return jsonify({
            'success': True,
            'predicted_price': predicted_price
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/train', methods=['POST'])
def train_model():
    try:
        results = car_interface.train_model()
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/examples', methods=['GET'])
def get_examples():
    try:
        examples = car_interface.get_example_predictions(5)
        return jsonify({
            'success': True,
            'examples': examples
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001) 