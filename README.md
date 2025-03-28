# Car Price Prediction System

A machine learning system for predicting car prices based on various features. The system includes a web interface for easy interaction and scripts for training and prediction.

## Project Structure

```
.
├── api/                    # API layer
│   ├── main.py            # FastAPI application
│   └── routes.py          # API routes
├── ml_engine/             # Core ML engine
│   ├── __init__.py
│   ├── model.py          # ML model implementation
│   └── trainer.py        # Training logic
├── scripts/               # Utility scripts
│   ├── training/         # Training-related scripts
│   │   └── train_with_json.py
│   └── prediction/       # Prediction-related scripts
│       └── predict_car_price.py
├── web/                  # Web interface
│   ├── static/          # Static files (CSS, JS)
│   │   ├── style.css
│   │   └── script.js
│   ├── templates/       # HTML templates
│   │   └── index.html
│   └── server.py        # Web server
├── data/                # Data directory
│   └── sample_cars.json # Sample training data
├── models/              # Saved models
├── requirements.txt     # Project dependencies
└── Makefile            # Build and development commands
```

## Setup

1. Create and activate virtual environment:
```bash
make install
```

2. Start the ML API server:
```bash
make run
```

3. Start the web interface (in a new terminal):
```bash
cd web
source ../venv/bin/activate
uvicorn server:app --reload --port 8080
```

## Usage

### Web Interface
Access the web interface at `http://localhost:8080` to make predictions through a user-friendly interface.

### Training with JSON Data
To train the model using JSON data:
```bash
python3 scripts/training/train_with_json.py
```

### Making Predictions
To make predictions using the command line:
```bash
python3 scripts/prediction/predict_car_price.py
```

## Data Format

### Training Data (JSON)
```json
{
    "cars": [
        {
            "mileage": 50000,
            "age": 3,
            "engine_size": 2.0,
            "horsepower": 150,
            "fuel_efficiency": 7.0,
            "price": 25000
        }
    ]
}
```

## Development

- `make install` - Create virtual environment and install dependencies
- `make run` - Start the API server
- `make test` - Run example script
- `make clean` - Remove virtual environment and cached files
- `make lint` - Run linter (flake8)
- `make format` - Format code using black
- `make dev-setup` - Complete development setup
- `make check` - Run all checks (linting and tests)
- `make dev` - Start development server with hot reload 