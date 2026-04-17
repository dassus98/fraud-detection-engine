# fraud-engine

> Production-grade real-time fraud detection engine.

This repository is being built sprint-by-sprint. Sprint 0 lays the
foundation (config, logging, seeding, test harness, CI gate). Later
sprints add data, features, models, evaluation, serving, and
hardening. See the [Sprint Status](#sprint-status) table below.

**The source of truth for code conventions is
[docs/CONVENTIONS.md](docs/CONVENTIONS.md). Read it before contributing.**

Architectural decisions live in [docs/ADR/](docs/ADR/).

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
cp .env.example .env        # fill in any secrets you have; defaults work for local
make install                # installs pre-commit hooks
```

## Quickstart

```bash
make lint                         # ruff check
make typecheck                    # mypy --strict
make test-fast                    # pytest tests/unit
uv run python scripts/verify_bootstrap.py   # green-row acceptance gate
```

## Sprint Status

| Sprint | Scope                               | Status       |
| ------ | ----------------------------------- | ------------ |
| 0      | Bootstrap: config, logging, tests   | in progress  |
| 1      | Data ingestion and schema contracts | pending      |
| 2      | Feature engineering                 | pending      |
| 3      | Models and hyperparameter tuning    | pending      |
| 4      | Evaluation and cost thresholding    | pending      |
| 5      | Serving and monitoring              | pending      |
| 6      | Hardening and documentation         | pending      |

> This README is a Sprint 0 placeholder. Sprint 6 rewrites it with full
> architecture, operations, and onboarding content.

## Layout

```
src/fraud_engine/   # package source (api, config, data, features,
                    # models, evaluation, monitoring, schemas, utils)
configs/            # YAML configs (schemas, logging)
tests/              # unit, integration, lineage
scripts/            # ops scripts (verify_bootstrap, data-download, ...)
docs/               # ADRs and conventions
sprints/            # per-sprint reports
```
