.DEFAULT_GOAL := help
.PHONY: help install format lint typecheck test test-fast test-integration \
        test-lineage data-download train serve docker-up docker-down clean

# Load .env so API_HOST / API_PORT flow into `make serve`.
# Uses `-include` so the target doesn't break before `.env` is created.
-include .env
export

help:  ## Show this help message.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make <target>\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install:  ## Install dependencies via uv and register pre-commit hooks.
	uv sync --all-extras
	uv run pre-commit install

format:  ## Format code with ruff.
	uv run ruff format src tests scripts

lint:  ## Lint with ruff.
	uv run ruff check src tests scripts

typecheck:  ## Type-check src/ with mypy strict mode.
	uv run mypy src

test:  ## Run the full test suite with coverage.
	uv run pytest

test-fast:  ## Run unit tests only, no coverage, quiet.
	uv run pytest tests/unit -q --no-cov

test-integration:  ## Run integration tests (requires Redis, Postgres).
	uv run pytest tests/integration

test-lineage:  ## Run schema-lineage tests.
	uv run pytest tests/lineage

data-download:  ## Download raw datasets. Implemented in Sprint 1.
	@echo "data-download: implemented in Sprint 1"; exit 1

train:  ## Train models. Implemented in Sprint 3.
	@echo "train: implemented in Sprint 3"; exit 1

# `serve` points at the real uvicorn entrypoint so Sprint 5 flips it live
# with zero Makefile edits. Until then, it fails loudly at import time.
serve:  ## Start the FastAPI server (requires Sprint 5 api module).
	uv run uvicorn fraud_engine.api.main:app --host $(API_HOST) --port $(API_PORT)

docker-up:  ## Start the docker-compose stack. Implemented in Sprint 5.
	@echo "docker-up: implemented in Sprint 5"; exit 1

docker-down:  ## Stop the docker-compose stack. Implemented in Sprint 5.
	@echo "docker-down: implemented in Sprint 5"; exit 1

clean:  ## Remove test / type-check / build caches.
	rm -rf .pytest_cache .ruff_cache .mypy_cache .pyright htmlcov .coverage coverage.xml build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
