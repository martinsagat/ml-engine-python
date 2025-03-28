.PHONY: install run test clean lint format help

# Python interpreter to use
PYTHON = python3

# Virtual environment directory
VENV = venv

# API server settings
HOST = 0.0.0.0
PORT = 8000

help:
	@echo "Available commands:"
	@echo "  make install    - Create virtual environment and install dependencies"
	@echo "  make run       - Start the API server"
	@echo "  make test      - Run example script to test the ML engine"
	@echo "  make clean     - Remove virtual environment and cached files"
	@echo "  make lint      - Run linter (flake8) on the code"
	@echo "  make format    - Format code using black"
	@echo "  make help      - Show this help message"

install:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Installing dependencies..."
	. $(VENV)/bin/activate && pip install -r requirements.txt
	@echo "Installation complete!"

run:
	@echo "Starting API server..."
	. $(VENV)/bin/activate && uvicorn api.main:app --host $(HOST) --port $(PORT) --reload

test:
	@echo "Running example script..."
	. $(VENV)/bin/activate && $(PYTHON) example.py

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	@echo "Cleanup complete!"

lint:
	@echo "Running linter..."
	. $(VENV)/bin/activate && flake8 api/ ml_engine/ example.py

format:
	@echo "Formatting code..."
	. $(VENV)/bin/activate && black api/ ml_engine/ example.py

# Development setup
dev-setup: install format lint

# Run all checks
check: lint test

# Start development server with hot reload
dev: run 