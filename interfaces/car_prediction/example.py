from interface import CarPredictionInterface

def main():
    # Initialize the interface
    car_interface = CarPredictionInterface()
    
    # Train the model
    print("Starting Car Price Prediction Model Training...")
    results = car_interface.train_model()
    print("\nTraining completed!")
    
    # Get some example predictions
    print("\nExample predictions:")
    examples = car_interface.get_example_predictions(5)
    for example in examples:
        print(f"\nCar Features:")
        for feature, value in example["features"].items():
            print(f"  {feature}: {value}")
        print(f"Actual price: ${example['actual_price']:.2f}")
        print(f"Predicted price: ${example['predicted_price']:.2f}")
    
    # Example of predicting a specific car
    print("\nPredicting price for a specific car:")
    specific_car = [50000, 5, 2.0, 200, 7.5]  # Example features
    predicted_price = car_interface.predict_price(specific_car)
    print(f"Predicted price: ${predicted_price:.2f}")

if __name__ == "__main__":
    main() 