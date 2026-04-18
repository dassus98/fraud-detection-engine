# Sprint 0 — Bootstrap Report

**Branch:** `sprint-0/bootstrap`
**Tag (on completion):** `sprint-0-complete`
**Date:** 2026-04-17 → 2026-04-18

## Summary

Sprint 0 lays the foundation for every later sprint: a reproducible
Python 3.11 / uv toolchain, a pinned dependency set covering
tabular ML (LightGBM, SHAP, cleanlab, Optuna), graph ML (torch,
torch-geometric), serving (FastAPI, uvicorn, Redis), and
observability (structlog, MLflow, Prometheus); an empty but
import-clean `fraud_engine` package organised for Sprints 1–5; a
Pydantic `Settings` class that centralises every tunable value
including explicit fraud/FP/TP cost weights for Sprint 4 cost-curve
optimisation; deterministic JSON-to-stdout + text-to-file logging via
structlog; a seeding helper that stabilises `random`, `numpy`, and
`torch`; pytest fixtures for tmp-isolated settings and deterministic
synthetic DataFrames; a parametrised smoke test that imports every
submodule; a `scripts/verify_bootstrap.py` acceptance gate that runs
ruff/mypy/pytest/settings and prints a green/red table; a GitHub
Actions workflow that runs the same gate on every push; and an ADR
plus CONVENTIONS doc codifying the tech choices and engineering
rules. The acceptance gate is green on Windows/WSL.

## What was built

Each item below maps to a single commit on `sprint-0/bootstrap`.

| # | Artefact | Commit |
|---|----------|--------|
| 1 | Directory tree + top-level `.gitignore` | [9f88036](#) |
| 2 | Pinned dependencies in `pyproject.toml` (incl. two drift bumps, see Deviations) | [2c160a5](#) |
| 3 | `.env.example` + POSIX `Makefile` with 14 targets | [ca9389c](#) |
| 4 | `ruff.toml`, `mypy.ini`, `.pre-commit-config.yaml`, `.gitattributes` | [1c94733](#) |
| 5 | Placeholder `configs/schemas.yaml` + `configs/logging.yaml` | [21d2a28](#) |
| 6 | `src/fraud_engine/__init__.py` + subpackage inits (api, config, data, features, models, evaluation, monitoring, schemas, utils) | [c81dad1](#) |
| 7 | `.gitignore` anchor fix + added the two subpackage inits silently dropped by the over-broad rule | [c4dcefa](#) |
| 8 | `src/fraud_engine/config/settings.py` — Pydantic `Settings` with paths, seeds, economic costs, LightGBM defaults, API, DB URLs, credentials, logging, decision threshold | [95fd500](#) |
| 9 | `src/fraud_engine/utils/logging.py` — `configure_logging`, `get_logger`, `new_run_id`, JSON stdout + text file mirror | [0553939](#) |
| 10 | `src/fraud_engine/utils/seeding.py` — `set_all_seeds` covering `random` / `numpy` / `PYTHONHASHSEED` / torch CPU+CUDA / cuDNN determinism | [c4c38b7](#) |
| 11 | `docs/ADR/0001-tech-stack.md` — LightGBM / Redis / FastAPI / pandera / structlog rationale | [88f4113](#) |
| 12 | `docs/CONVENTIONS.md` — Universal Standards (Code Quality / Logging / Testing / Git & CI / Data Contracts / Reproducibility) | [cf2f476](#) |
| 13 | `README.md` placeholder with setup, quickstart, sprint-status table | [8f76c3c](#) |
| 14 | `tests/conftest.py` — `tmp_data_dir`, `mock_settings`, `small_transactions_df`, `small_identity_df` | [f580454](#) |
| 15 | `tests/unit/test_smoke.py` — version, parametrised submodule imports, settings, `ensure_directories`, logger, seed reproducibility | [a7a864e](#) |
| 16 | `scripts/verify_bootstrap.py` — Click-based acceptance gate printing the 4-row status table | [7252727](#) |
| 17 | `.github/workflows/ci.yml` — lint → format-check → type-check → unit-test → verify_bootstrap on Ubuntu + uv | [cb52e47](#) |
| 18 | `fix: green up verification gate on Windows/WSL` — `python -m pytest` shim, UP017 UTC alias, ruff format, asyncio_default_fixture_loop_scope | [9f83423](#) |

## What was tested

### `uv run ruff check src tests scripts`

```
All checks passed!
```

### `uv run ruff format --check src tests scripts`

```
15 files already formatted
```

### `uv run mypy src`

```
Success: no issues found in 13 source files
```

### `uv run python -m pytest tests/unit --no-cov -q`

```
..................                                                       [100%]
18 passed in 49.16s
```

(18 tests: 1 version check, 12 parametrised submodule imports,
2 settings checks, 1 logger-emits check, 2 seeding checks.)

### `uv run python scripts/verify_bootstrap.py`

```
[ OK ] ruff       ( 1.80s)
[ OK ] mypy       (47.27s)
[ OK ] pytest     (43.12s)
[ OK ] settings   ( 2.85s)

Bootstrap: GREEN
```

Timings measured on the WSL-share filesystem, which is the primary
driver of the mypy/pytest runtimes — the same gate on the CI runner
(native Linux) runs in a fraction of the time.

## Deviations from prompt

1. **cleanlab pin bumped 2.7.0 → 2.9.0.** The 2.7.0 series requires
   `numpy < 2.dev0`, which conflicts with the `numpy==2.2.0` pin.
   2.9.0 is the first cleanlab release to support numpy 2.x.
2. **mlflow pin bumped 2.19.0 → 3.11.1.** 2.19.0 requires
   `pyarrow < 19`, which conflicts with the `pyarrow==19.0.0` pin.
   3.11.1 relaxes the ceiling to `< 24`.
3. **Added `.gitattributes`.** Not in the prompt's file list but
   required on Windows to keep line endings LF end-to-end and avoid
   `ruff format --check` failures after a fresh clone.
4. **Added `.secrets.baseline`.** The `detect-secrets` pre-commit
   hook refuses to run without it. Empty baseline committed so
   future secrets are caught by diff.
5. **Added `.github/workflows/ci.yml`.** The prompt shipped an empty
   `.github/workflows/` directory but the Universal Standards
   mandate CI. Implemented the gate in Sprint 0 rather than waiting
   for Sprint 5.
6. **`serve` Makefile target is a live uvicorn invocation, not a
   stub.** Fails loudly until Sprint 5 adds `fraud_engine.api.main`,
   but needs zero Makefile edits when the module lands.
7. **Worked on branch `sprint-0/bootstrap`, not `main`.** Per the
   Universal Standards rule "Merge to main only after verification
   passes" and "branch per prompt". Merge to main is a user-gated
   action.

One additional runtime-only fix, recorded in commit 18
(`9f83423`), made the gate pass on the developer's WSL-share
workstation:

- Invoke pytest via `python -m pytest` rather than the `pytest`
  shim. On the WSL share the shim's `sys.executable` resolves to a
  `\\?\UNC\…` extended path that trips numpy 2.2's source-tree guard
  inside `conftest.py`. `python -m` avoids the shim entirely.
  Applied in `Makefile`, `.github/workflows/ci.yml`, and
  `scripts/verify_bootstrap.py`.

## Known gaps / handoffs to Sprint 1

- **`configs/schemas.yaml` is a header comment only.** Sprint 1
  populates it with pandera schema definitions for the transaction,
  identity, and feature tables.
- **`make data-download` is a `exit 1` stub.** Sprint 1 implements
  the Kaggle IEEE-CIS fetch + checksum verification.
- **`src/fraud_engine/data/`, `features/`, `models/`, `evaluation/`,
  `monitoring/`, `schemas/`, `api/` are empty package stubs** with
  only `__init__.py` containing `from __future__ import annotations`
  and a one-line docstring. Each sprint's first commit lands real
  code into its subpackage.
- **`make serve` will fail until Sprint 5.** Documented in the
  target's docstring.
- **Docker build + API smoke-test are not in CI.** The workflow
  comment cites this deferral; Sprint 5 adds both stages.
- **`asyncio_default_fixture_loop_scope = "function"`** was chosen
  to silence the pytest-asyncio deprecation. If Sprint 5 adds
  async integration tests that share a loop across tests, revisit.

## Acceptance criteria checklist

- [x] **`make install` succeeds from a clean clone.**
  Evidence: `uv sync --all-extras --frozen` installs the full
  dependency set (~230 wheels, ~2 GB on disk) on Python 3.11.
  `pre-commit install` succeeds after install.
- [x] **`make lint`, `make typecheck`, `make test-fast` are green.**
  Evidence: outputs above — `All checks passed!`, `Success: no
  issues found in 13 source files`, `18 passed in 49.16s`.
- [x] **`python scripts/verify_bootstrap.py` prints four green rows
  and exits 0.** Evidence: `[ OK ]` rows for ruff, mypy, pytest,
  settings → `Bootstrap: GREEN`.
- [x] **`git log --oneline` shows conventional-commit history.**
  Evidence: 18 commits from `9f88036` (`chore: scaffold directory
  tree`) through `9f83423` (`fix: green up verification gate`), all
  following the `type(scope): subject` format.
