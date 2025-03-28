import requests
import numpy as np

# API endpoint
API_URL = "http://localhost:8000"

def predict_car_price(mileage, age, engine_size, horsepower, fuel_efficiency):
    """
    Predict the price of a car based on its features.
    
    Args:
        mileage (float): Car's mileage in kilometers
        age (int): Car's age in years
        engine_size (float): Engine size in liters
        horsepower (float): Engine horsepower
        fuel_efficiency (float): Fuel consumption in L/100km
    """
    # Prepare the features
    features = [[mileage, age, engine_size, horsepower, fuel_efficiency]]
    
    # Make prediction
    response = requests.post(f"{API_URL}/predict", json={"features": features})
    prediction = response.json()["predictions"][0]
    
    return prediction

def main():
    # Example 1: A relatively new, efficient car
    print("\nExample 1: New Efficient Car")
    print("-" * 50)
    print("Features:")
    print("Mileage: 20,000 km")
    print("Age: 2 years")
    print("Engine size: 1.6L")
    print("Horsepower: 120")
    print("Fuel efficiency: 6.5 L/100km")
    
    price = predict_car_price(20000, 2, 1.6, 120, 6.5)
    print(f"\nPredicted price: ${price:,.2f}")
    
    # Example 2: An older, powerful car
    print("\nExample 2: Older Powerful Car")
    print("-" * 50)
    print("Features:")
    print("Mileage: 150,000 km")
    print("Age: 8 years")
    print("Engine size: 3.0L")
    print("Horsepower: 250")
    print("Fuel efficiency: 9.5 L/100km")
    
    price = predict_car_price(150000, 8, 3.0, 250, 9.5)
    print(f"\nPredicted price: ${price:,.2f}")
    
    # Example 3: A mid-range car
    print("\nExample 3: Mid-Range Car")
    print("-" * 50)
    print("Features:")
    print("Mileage: 75,000 km")
    print("Age: 5 years")
    print("Engine size: 2.0L")
    print("Horsepower: 180")
    print("Fuel efficiency: 7.5 L/100km")
    
    price = predict_car_price(75000, 5, 2.0, 180, 7.5)
    print(f"\nPredicted price: ${price:,.2f}")

if __name__ == "__main__":
    print("Car Price Prediction Examples")
    print("=" * 50)
    main() 