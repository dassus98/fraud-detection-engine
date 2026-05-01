.DEFAULT_GOAL := help
.PHONY: help install format lint typecheck test test-fast test-integration \
        test-lineage nb-test notebooks data-download data-profile sprint1-baseline \
        train serve docker-up docker-down docker-ps clean

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

test:  ## Run the full test suite with coverage (includes notebook smoke).
	uv run python -m pytest
	$(MAKE) nb-test

test-fast:  ## Run unit tests only, no coverage, quiet.
	uv run python -m pytest tests/unit -q --no-cov

test-integration:  ## Run integration tests (requires Redis, Postgres).
	uv run python -m pytest tests/integration

test-lineage:  ## Run schema-lineage tests.
	uv run python -m pytest tests/lineage

nb-test:  ## Execute notebooks end-to-end via nbmake (catches util-rename drift).
	uv run python -m pytest --no-cov --nbmake notebooks

notebooks:  ## Rebuild + execute every committable notebook in place (commit-ready).
	uv run python scripts/_build_eda_notebook.py
	uv run python scripts/_build_graph_analysis_notebook.py
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/00_observability_demo.ipynb

data-download:  ## Fetch IEEE-CIS from Kaggle into data/raw/ and write the manifest.
	uv run python scripts/download_data.py

data-profile:  ## Render reports/raw_profile.{html,json} from the merged raw frame.
	uv run python scripts/profile_raw.py

sprint1-baseline:  ## Fit the Sprint 1 LightGBM baseline (random + temporal) on the full dataset.
	uv run python scripts/run_sprint1_baseline.py

train:  ## Train models. Implemented in Sprint 3.
	@echo "train: implemented in Sprint 3"; exit 1

# `serve` points at the real uvicorn entrypoint so Sprint 5 flips it live
# with zero Makefile edits. Until then, it fails loudly at import time.
serve:  ## Start the FastAPI server (requires Sprint 5 api module).
	uv run uvicorn fraud_engine.api.main:app --host $(API_HOST) --port $(API_PORT)

docker-up:  ## Start the dev compose stack (Postgres, Redis, MLflow, Prometheus, Grafana).
	docker compose -f docker-compose.dev.yml up -d

docker-down:  ## Stop the dev compose stack.
	docker compose -f docker-compose.dev.yml down

docker-ps:  ## Show the dev stack's service status (including healthchecks).
	docker compose -f docker-compose.dev.yml ps

clean:  ## Remove test / type-check / build caches.
	rm -rf .pytest_cache .ruff_cache .mypy_cache .pyright htmlcov .coverage coverage.xml build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
