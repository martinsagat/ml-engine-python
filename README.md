# Autonomous Machine Learning Engine

A powerful, self-learning machine learning engine that provides a REST API interface for training, prediction, and model management. This engine can be easily integrated into other applications and supports both classification and regression tasks.

## Features

- **Autonomous Learning**: Automatically selects and configures appropriate models based on the task type
- **Multiple Task Types**: Supports both classification and regression problems
- **REST API Interface**: Easy integration with any application
- **Model Management**: Save and load trained models
- **Feature Scaling**: Automatic feature preprocessing
- **Real-time Training**: Train models with new data at any time
- **Comprehensive Metrics**: Detailed training and model performance information

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- make (for using Makefile commands)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-engine
```

2. Set up the development environment:
```bash
make dev-setup
```
This command will:
- Create a virtual environment
- Install all required dependencies
- Format the code
- Run the linter

## Usage

### Starting the Server

Start the API server:
```bash
make run
```

The server will start at `http://localhost:8000`

### API Documentation

Once the server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Available Endpoints

1. **Train Model**
```http
POST /train
Content-Type: application/json

{
    "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    "targets": [0, 1],
    "task_type": "classification"
}
```

2. **Make Predictions**
```http
POST /predict
Content-Type: application/json

{
    "features": [[1.0, 2.0, 3.0]]
}
```

3. **Get Model Status**
```http
GET /model/status
```

4. **Save Model**
```http
POST /model/save
Content-Type: application/json

{
    "name": "my_model"
}
```

5. **Load Model**
```http
POST /model/load
Content-Type: application/json

{
    "name": "my_model"
}
```

### Example Usage

Run the example script to see the ML engine in action:
```bash
make test
```

This will:
1. Train a classification model with sample data
2. Make predictions
3. Train a regression model with sample data
4. Make predictions
5. Save both models

## Development

### Available Make Commands

- `make install` - Create virtual environment and install dependencies
- `make run` - Start the API server
- `make test` - Run example script
- `make clean` - Remove virtual environment and cached files
- `make lint` - Run linter (flake8)
- `make format` - Format code using black
- `make dev-setup` - Complete development setup
- `make check` - Run all checks (linting and tests)
- `make dev` - Start development server with hot reload

### Code Quality

The project uses:
- `black` for code formatting
- `flake8` for linting

Run code quality checks:
```bash
make check
```

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
├── utils/                 # Utility functions
│   └── __init__.py
├── example.py            # Example usage script
├── requirements.txt      # Project dependencies
├── Makefile             # Build and development commands
└── README.md            # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 