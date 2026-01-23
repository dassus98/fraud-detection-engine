# Variables
PYTHON = python3
PIP = pip
VENV = venv

# Setup: Create env and install dependencies
setup:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && $(PIP) install -r requirements.txt

# Training: Run the full training pipeline
train:
	. $(VENV)/bin/activate && $(PYTHON) -m src.pipeline.train_pipeline

# API: Run the FastAPI server locally (Dev mode)
serve:
	. $(VENV)/bin/activate && uvicorn src.api.main:app --reload

# Docker: Build the container
docker-build:
	docker build -t fraud-engine .

# Docker: Run the full stack (API + Redis)
up:
	docker-compose up --build

# Testing: Run unit tests
test:
	. $(VENV)/bin/activate && pytest tests/

# Clean: Remove cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +