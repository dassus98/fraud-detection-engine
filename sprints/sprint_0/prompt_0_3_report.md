# Sprint 0 — Prompt 3 Report: Observability Layer

**Branch (target):** `sprint-0/observability`
**Date:** 2026-04-18
**Status:** ready for John to commit — **all implementable verification green; Docker stack deferred for manual run (daemon not installed in this environment)**

## Summary

Prompt 3 puts the shared observability spine in place before any Sprint 1+
work can accumulate retrofit debt. It extends `utils/logging.py` with a
`request_id` contextvar (for API-era correlation) and a `log_dataframe`
stage-boundary emitter; adds a new `utils/tracing.py` with a `Run`
dataclass, a `run_context` context manager that persists
`logs/runs/{run_id}/{run.json,stdout.log,stderr.log,artifacts/}`, and an
`attach_artifact` dispatch covering Path / DataFrame / Figure / dict /
list / generic joblib objects; a new `utils/mlflow_setup.py` that
centralises tracking-URI wiring, experiment bootstrap, and two logging
helpers (`log_dataframe_stats`, `log_economic_metrics`); and a new
`utils/metrics.py` exposing `economic_cost`, `precision_recall_at_k`,
`recall_at_fpr`, `compute_psi` — the four numbers Sprint 4's threshold
optimiser and Sprint 6's drift alarm will both call. A
`docker-compose.dev.yml` brings up Postgres / Redis / MLflow (SQLite on a
named volume) / Prometheus / Grafana with healthchecks and
`127.0.0.1`-only bindings; Prometheus is pre-wired to scrape
`host.docker.internal:8000/metrics` (Sprint 5 will expose the endpoint),
and Grafana gets its Prometheus datasource auto-provisioned. `nbmake` is
added to the dev extras; `notebooks/00_observability_demo.ipynb` runs
end-to-end under `--nbmake` and exercises every new surface. All
implementable verification gates pass (ruff + ruff format + mypy strict
+ 75 unit tests + 8 lineage tests + nbmake + `verify_bootstrap.py`);
the five-service Docker stack must be brought up manually by John because
Docker Desktop isn't installed in this environment.

## What was built

Each row is one logical change; the final git grouping is John's call.

| # | Artefact | Purpose |
|---|----------|---------|
| 1 | `src/fraud_engine/utils/logging.py` (extended) | `_REQUEST_ID` ContextVar + `bind_request_id` / `get_request_id` / `reset_request_id`; `log_dataframe()` snapshot emitter that fingerprints the first row via SHA-256 and never logs values |
| 2 | `src/fraud_engine/utils/tracing.py` (new) | Frozen `Run` dataclass; `run_context(pipeline, *, metadata, capture_streams)` creates `logs/runs/{run_id}/` tree, captures stdout/stderr via `_TeeStream`, writes `run.json` with status/duration/traceback; `attach_artifact(run, obj, *, name)` dispatches on `Path` / DataFrame / matplotlib Figure / dict\|list / joblib fallback |
| 3 | `src/fraud_engine/utils/mlflow_setup.py` (new) | `configure_mlflow()` sets tracking URI; `setup_experiment(name=None)` resolves or creates, returns experiment_id; `log_dataframe_stats(df, prefix)` logs shape/memory/dtypes as params and NaN/duplicate counts as metrics; `log_economic_metrics(fn_rate, fp_rate, total_cost_usd)` raises if called outside an active run |
| 4 | `src/fraud_engine/utils/metrics.py` (new) | `economic_cost` (default costs from Settings; per-call overrides); `precision_recall_at_k`; `recall_at_fpr` (sklearn roc_curve + degenerate-fallback); `compute_psi` (equal-frequency quantile bins + `1e-4` epsilon floor) |
| 5 | `src/fraud_engine/utils/__init__.py` (replaced) | Re-exports: 20 symbols — logging (7) + tracing (3) + mlflow_setup (4) + metrics (4) + seeding/new_run_id (2) |
| 6 | `src/fraud_engine/config/settings.py` (extended) | `mlflow_experiment_name`, `mlflow_port`, `prometheus_port`, `grafana_port`, `grafana_admin_user`, `grafana_admin_password` (SecretStr) |
| 7 | `.env.example` (extended) | Mirrors every new Settings field with inline business-meaning comments |
| 8 | `docker-compose.dev.yml` (new) | Five services (postgres 16.4, redis 7.4, mlflow v3.11.1, prometheus v3.1.0, grafana 11.4.0); healthchecks per service; named volumes; `127.0.0.1`-only ports; `extra_hosts: host.docker.internal:host-gateway` on Prometheus for WSL2 host scrape |
| 9 | `docker/prometheus/prometheus.yml` (new) | 15 s scrape interval; two jobs — `prometheus` self-scrape and `fraud-api` pre-wired at `host.docker.internal:${API_PORT}` (intentionally DOWN until Sprint 5) |
| 10 | `docker/grafana/provisioning/datasources/prometheus.yml` (new) | Prometheus datasource auto-provisioned at `http://prometheus:9090`, marked default + editable |
| 11 | `pyproject.toml` (extended) | `nbmake==1.5.5` in `[project.optional-dependencies.dev]` |
| 12 | `uv.lock` (regenerated) | `Resolved 241 packages` — added `nbmake` and its deps |
| 13 | `Makefile` (extended) | `docker-up` / `docker-down` / `docker-ps` targets wired to the new compose file; `nb-test` target runs `pytest --nbmake notebooks`; `test` chains to `nb-test` (fast suite unchanged) |
| 14 | `mypy.ini` (extended) | `sklearn.*` and `joblib.*` added to `ignore_missing_imports` — both ship without py.typed stubs |
| 15 | `tests/conftest.py` (extended) | `mock_settings` fixture now monkeypatches `MODELS_DIR` and `LOGS_DIR` alongside `DATA_DIR` so code paths calling `get_settings()` directly (like `run_context`) land on tmp disk |
| 16 | `tests/unit/test_logging.py` (renamed from `test_log_call.py`, extended) | 17 tests: preserves `TestDescribe` (6) + `TestLogCall` (4); adds `TestRequestId` (5 — bind/reset/async-isolation/structlog-render) + `TestLogDataframe` (2 — event shape + secret-never-leaked assertion) |
| 17 | `tests/unit/test_tracing.py` (new) | 9 tests: `TestRunContext` (4 — directory tree, success status, failure + reraise, metadata round-trip) + `TestAttachArtifact` (5 — Path/DataFrame/Figure/dict/joblib branches) |
| 18 | `tests/unit/test_metrics.py` (new) | 15 tests: `TestEconomicCost` (4) + `TestPrecisionRecallAtK` (3) + `TestRecallAtFPR` (3) + `TestComputePSI` (4) + `test_metrics_import_smoke` |
| 19 | `tests/unit/test_mlflow_setup.py` (new) | 6 tests: `TestConfigureMLflow` (1) + `TestSetupExperiment` (2) + `TestLogDataframeStats` (1) + `TestLogEconomicMetrics` (2 — raises-outside-run + records-three-metrics) |
| 20 | `notebooks/00_observability_demo.ipynb` (new) | Seven-cell walkthrough (logging → `log_dataframe` → `run_context` + `attach_artifact` → MLflow helpers → the four metrics), all synthetic data so nbmake runs without raw CSVs |
| 21 | `docs/OBSERVABILITY.md` (new) | Six-section operator guide: instrumenting new code, reading the logs (`jq` recipes + directory layout), tracing a lineage issue end-to-end, log-level table for this repo, local dev stack URLs, related references |

## What was tested

Verbatim output, in the order the plan specifies.

### 1. `uv run ruff check src tests scripts`

```
All checks passed!
```

### 2. `uv run ruff format --check src tests scripts`

```
27 files already formatted
```

### 3. `uv run mypy src`

```
Success: no issues found in 18 source files
```

### 4. `uv run python -m pytest tests/unit --no-cov -q`

```
75 passed, 4 warnings in 103.36s (0:01:43)
```

(Warnings are MLflow's "FileStore deprecated in 2026" informational notice from `tests/unit/test_mlflow_setup.py` — benign; we track it under Known gaps.)

### 5. `uv run python -m pytest tests/lineage --no-cov -q`

```
8 passed in 198.99s (0:03:18)
```

All Sprint 0.2 lineage contracts still hold after the settings touch.

### 6. `uv run python -m pytest --no-cov --nbmake notebooks`

```
notebooks\00_observability_demo.ipynb .                                  [100%]

============================= 1 passed in 57.61s ==============================
```

### 7. `uv run python scripts/verify_bootstrap.py`

```
[ OK ] ruff       ( 1.98s)
[ OK ] mypy       (99.72s)
[ OK ] pytest     (107.47s)
[ OK ] settings   ( 3.18s)

Bootstrap: GREEN
```

### 8. Docker dev stack — **deferred**

`docker compose -f docker-compose.dev.yml up -d` could not execute: Docker
Desktop is not installed in this environment (`docker: command not found`
in the WSL shell and no `Docker/` under `/c/Program Files/`). The compose
file was instead validated structurally via `yaml.safe_load`, which
confirms the five expected services (`postgres`, `redis`, `mlflow`,
`prometheus`, `grafana`), five named volumes, and parseable port /
healthcheck / provisioning shapes. `docker/prometheus/prometheus.yml`
and `docker/grafana/provisioning/datasources/prometheus.yml` parse
cleanly and expose the expected scrape targets (`prometheus` self +
`fraud-api` at `host.docker.internal:8000`) and datasource (`Prometheus`
→ `http://prometheus:9090`).

John should run the stack manually once Docker Desktop is running:

```bash
make docker-up
docker compose -f docker-compose.dev.yml ps          # healthchecks green
curl -f http://127.0.0.1:5000/                       # MLflow UI
curl -f http://127.0.0.1:9090/-/ready                # Prometheus ready
curl -f http://127.0.0.1:3000/api/health             # Grafana health
redis-cli -h 127.0.0.1 -p 6379 ping                  # PONG
psql "postgresql://fraud:fraud@127.0.0.1:5432/fraud" -c 'SELECT 1'
make docker-down
```

## Deviations from prompt

1. **`tests/unit/test_log_call.py` renamed to `tests/unit/test_logging.py`** —
   the prompt specifies the new name; renaming (rather than creating a
   parallel stub) keeps the file count tidy and the new request_id /
   log_dataframe classes live alongside the preserved `TestDescribe` /
   `TestLogCall` suites.
2. **`test_logger_includes_request_id` uses `structlog.wrap_logger` with
   a manual pipeline instead of `structlog.testing.capture_logs`.**
   `capture_logs()` in structlog 25.1 strips the entire processor chain
   — including `merge_contextvars` — so the bound `request_id` would not
   appear on the captured record. The test builds a minimal ad-hoc
   pipeline (`merge_contextvars` → capture → `DropEvent`) to exercise
   exactly the production rendering path.
3. **`test_independent_contexts` compares against a list, not a tuple** —
   `asyncio.gather()` returns `list[...]`. The assertion uses
   `== ["req-A", "req-B"]`.
4. **`attach_artifact`'s Figure check lifted to module scope via
   `_FIGURE_TYPES` tuple** — the original in-function `try/except
   ImportError` triggered ruff N806 (local variable should be lowercase)
   on the class binding; resolving at import time keeps ruff happy and
   doubles as a micro-perf win (no per-call re-import).
5. **`mypy.ini` gained `sklearn.*` and `joblib.*` entries** — neither
   ships `py.typed`, so mypy strict flags their imports as
   `import-untyped`. Mirrors the existing pattern for `mlflow.*`,
   `pandera.*`, etc.
6. **`conftest.mock_settings` now monkeypatches `MODELS_DIR` and
   `LOGS_DIR`** — tracing code paths call `get_settings()` directly;
   without the env monkeypatches, `run.run_dir` would have resolved to
   the real `logs/` directory and polluted the repo during test runs.
7. **Docker stack validated structurally only** — see §8 above. All
   YAML parses; manual bring-up is John's action.

## Known gaps / handoffs

- **Grafana dashboards are deferred to Sprint 6.** The Prometheus
  datasource is provisioned, but no dashboards are pre-loaded — the
  metrics they would render don't exist yet.
- **Prometheus `fraud-api` target is DOWN.** Intentional: the target is
  pre-wired so the scrape config exists and is visible on
  `/targets`, but `host.docker.internal:8000/metrics` doesn't respond
  until Sprint 5 stands up the FastAPI `/metrics` endpoint. That's a
  diagnostic signal, not a defect.
- **`@lineage_step` decorator (CLAUDE.md §7.2) is deferred to Sprint 1.**
  `src/fraud_engine/data/lineage.py` doesn't exist yet and was explicitly
  out of scope for this prompt.
- **PyTorch artifact support in `attach_artifact` is deferred to
  Sprint 3.** The joblib fallback covers sklearn / LightGBM / generic
  pickleables today; when Sprint 3 introduces `FraudNet` / `FraudGNN`,
  the dispatch grows a `torch.Tensor` / `nn.Module` branch.
- **MLflow FileStore deprecation warning.** MLflow 3.11.1 warns that
  the file store is deprecated as of 2026-02 in favour of SQLite. The
  docker-compose service already uses SQLite (`--backend-store-uri
  sqlite:////mlflow/mlflow.db`); the unit tests still use the file
  store because each test gets a fresh tmp dir and SQLite would add
  fixture complexity without changing what's being tested.

## Acceptance checklist

- [x] `utils/logging.py` exposes `request_id` contextvar (`bind_request_id`
      / `get_request_id` / `reset_request_id`) — [src/fraud_engine/utils/logging.py:74](src/fraud_engine/utils/logging.py:74)
- [x] `utils/logging.py` exposes `log_dataframe` with SHA-256 fingerprint
      and no value leakage — [src/fraud_engine/utils/logging.py:295](src/fraud_engine/utils/logging.py:295)
- [x] `utils/tracing.py` exposes `Run`, `run_context`, `attach_artifact` —
      [src/fraud_engine/utils/tracing.py](src/fraud_engine/utils/tracing.py)
- [x] `utils/mlflow_setup.py` exposes `configure_mlflow`,
      `setup_experiment`, `log_dataframe_stats`, `log_economic_metrics` —
      [src/fraud_engine/utils/mlflow_setup.py](src/fraud_engine/utils/mlflow_setup.py)
- [x] `utils/metrics.py` exposes `economic_cost`, `precision_recall_at_k`,
      `recall_at_fpr`, `compute_psi` — [src/fraud_engine/utils/metrics.py](src/fraud_engine/utils/metrics.py)
- [x] `docker-compose.dev.yml` with Postgres / Redis / MLflow /
      Prometheus / Grafana, healthchecks, named volumes — [docker-compose.dev.yml](docker-compose.dev.yml)
- [x] Prometheus scrape config + Grafana datasource provisioning —
      [docker/prometheus/prometheus.yml](docker/prometheus/prometheus.yml) / [docker/grafana/provisioning/datasources/prometheus.yml](docker/grafana/provisioning/datasources/prometheus.yml)
- [x] `notebooks/00_observability_demo.ipynb` runs end-to-end under
      `nbmake` — `1 passed in 57.61s`
- [x] `docs/OBSERVABILITY.md` covers instrumenting code, reading logs,
      tracing lineage, log levels — [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md)
- [x] Settings gains `mlflow_experiment_name` + service ports + Grafana
      creds — [src/fraud_engine/config/settings.py:157](src/fraud_engine/config/settings.py:157)
- [x] `.env.example` mirrors all new Settings fields — [.env.example](.env.example)
- [x] ruff check + ruff format --check + mypy strict — all green
- [x] 75 unit tests + 8 lineage tests — all pass
- [x] `verify_bootstrap.py` — GREEN
- [ ] Docker stack end-to-end `up` / healthcheck / service curl —
      **deferred to John's manual run** (Docker Desktop not available
      in this environment; YAML validated structurally in lieu)

Ready for John to commit the work, run the Docker stack once locally to
confirm §8 of the verification plan, and then tag
`sprint-0-complete`. I will not run any git commands.
