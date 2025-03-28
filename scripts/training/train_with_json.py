import json
import requests
import numpy as np
import os
import argparse

# API endpoint
API_URL = "http://localhost:8000"

def train_model_with_json(json_file_path):
    """
    Train the model using data from a JSON file.
    
    Expected JSON format:
    {
        "cars": [
            {
                "mileage": 50000,
                "age": 3,
                "engine_size": 2.0,
                "horsepower": 150,
                "fuel_efficiency": 7.0,
                "price": 25000
            },
            ...
        ]
    }
    """
    try:
        # Read JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        # Extract features and targets
        features = []
        targets = []
        
        for car in data['cars']:
            features.append([
                car['mileage'],
                car['age'],
                car['engine_size'],
                car['horsepower'],
                car['fuel_efficiency']
            ])
            targets.append(car['price'])
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(targets)
        
        # Prepare data for API
        training_data = {
            "features": X.tolist(),
            "targets": y.tolist(),
            "task_type": "regression"
        }
        
        # Train model
        print(f"Training model with {len(features)} cars...")
        response = requests.post(f"{API_URL}/train", json=training_data)
        
        if response.status_code == 200:
            result = response.json()
            print("\nTraining Results:")
            print(f"Training Score: {result['train_score']:.4f}")
            print(f"Number of samples: {result['n_samples']}")
            print(f"Number of features: {result['n_features']}")
            
            # Save the model
            save_response = requests.post(
                f"{API_URL}/model/save",
                json={"name": "car_price_model"}
            )
            print("\nModel saved successfully!")
            
            return True
        else:
            print(f"Error training model: {response.text}")
            return False
            
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
        return False
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return False
    except KeyError:
        print("Error: JSON data must contain a 'cars' array with car objects")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train car price prediction model with JSON data')
    parser.add_argument('--data', type=str, default='data/cars.json',
                      help='Path to JSON file containing car data')
    args = parser.parse_args()
    
    # Train the model with the JSON data
    train_model_with_json(args.data)

if __name__ == "__main__":
    main() 