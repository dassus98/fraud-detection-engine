# Fraud Detection Engine

> Real-time fraud detection system built on the IEEE-CIS dataset,
> architected as a production engagement for a Canadian fintech.

This project scores card-not-present transactions end-to-end:
LightGBM-based classification with SHAP explainability, economic
threshold optimisation, a FastAPI serving layer backed by Redis and
Postgres, and Prometheus + Grafana monitoring with PSI drift
detection. The build is organised as seven sprints — this repository
is the reference implementation, not a Kaggle submission.

**Standards source of truth:** [docs/CONVENTIONS.md](docs/CONVENTIONS.md)
(reader-facing) mirrors the agent-facing [CLAUDE.md](CLAUDE.md).
Architecture decisions are recorded under
[docs/ADR/](docs/ADR/).

---

## Quick Start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies from the locked manifest
uv sync --all-extras

# 2. Seed local config (defaults are safe for local dev)
cp .env.example .env

# 3. Install pre-commit hooks
make install

# 4. Green the Sprint 0 acceptance gate
uv run python scripts/verify_bootstrap.py
```

Day-to-day commands:

```bash
make lint              # ruff check
make format            # ruff format
make typecheck         # mypy --strict on src/
make test-fast         # pytest tests/unit (no coverage)
make test              # full suite with coverage
make test-lineage      # schema + temporal integrity tests
```

CI runs the same gate on every push and pull request; see
[.github/workflows/ci.yml](.github/workflows/ci.yml).

---

## Architecture Overview

```
Raw IEEE-CIS data
    ↓ [data loader + schema validation]
Cleaned interim data (Parquet)
    ↓ [feature pipeline: T1 basic → T2 aggregations → T3 behavioral
        → T4 exponential decay → T5 graph]
Feature-engineered data (Parquet)
    ↓ [LightGBM training + Optuna tuning + calibration]
Production model (joblib)
    ↓ [economic threshold optimization with cost function]
Calibrated, threshold-tuned model
    ↓ [FastAPI + Redis (real-time features) + Postgres (batch features)]
Production API (Docker)
    ↓ [SHAP explainability + shadow mode + prediction logging]
    ↓ [Prometheus + Grafana monitoring + PSI drift detection]
```

- **Production model:** LightGBM (Model A). Chosen for inference
  latency (< 15 ms) and SHAP interpretability.
- **Diversity models:** FraudNet entity-embedding NN (Model B,
  shadow-deployable) and FraudGNN PyTorch-Geometric graph model
  (Model C, batch-only — its outputs feed Model A as features).
- **Decision threshold:** Driven by the economic cost function in
  Sprint 4, not by 0.5 or F1.
- **Latency budget:** < 100 ms P95 end-to-end, including Redis
  lookup, inference, SHAP, and logging.
- **Target metrics:** AUC ≥ 0.93 on temporal split; > 90% fraud
  capture at < 2% false-positive rate.

The full architecture rationale is in
[docs/ADR/0001-tech-stack.md](docs/ADR/0001-tech-stack.md).

---

## Repository Layout

```
fraud-detection-engine/
├── .github/workflows/         # CI configuration
├── configs/                   # YAML configs (schemas, hyperparameters, costs)
├── data/                      # gitignored; raw, interim, processed
├── docs/                      # ADRs, conventions, architecture, model card
├── logs/                      # gitignored; structured JSON + per-run dirs
├── notebooks/                 # numbered, exploratory + demo
├── scripts/                   # CLI entry points (download, train, evaluate)
├── src/fraud_engine/          # library source (api, config, data, features,
│                              # models, evaluation, monitoring, schemas, utils)
├── sprints/                   # per-prompt completion reports
└── tests/                     # unit, integration, lineage
```

---

## Sprint Status

| Sprint | Scope                                              | Status      |
| ------ | -------------------------------------------------- | ----------- |
| 0      | Foundation & environment (config, logging, CI)     | in progress |
| 1      | Data profiling, EDA, temporal split, baseline      | pending     |
| 2      | Feature engineering tiers 1–3                      | pending     |
| 3      | Advanced features & models (tuned LGBM, NN, GNN)   | pending     |
| 4      | Economic evaluation & threshold optimisation       | pending     |
| 5      | Production API & shadow mode                       | pending     |
| 6      | Monitoring & documentation                         | pending     |

Per-prompt completion reports live under
[sprints/sprint_X/prompt_Y_report.md](sprints/).

> This README is refreshed incrementally each sprint. Sprint 6 adds
> the full operations, onboarding, and model-card sections.

---

## Previous Iteration

The original Fraud Detection Engine lives on the
[`archive/v1-original`](../../tree/archive/v1-original) branch and is
preserved as portfolio history. That branch is not referenced or
copied from during the current rebuild — the new build is
independent and held to stricter standards (see
[CLAUDE.md §9](CLAUDE.md) anti-pattern 11).
