# Observability Runbook

How to instrument new code, read the logs, trace a lineage issue, and pick the right log level. Matches the surface delivered in Sprint 0.3 (`src/fraud_engine/utils/logging.py`, `tracing.py`, `mlflow_setup.py`, `metrics.py`).

For a hands-on walk-through, run [`notebooks/00_observability_demo.ipynb`](../notebooks/00_observability_demo.ipynb).

---

## 1. Instrumenting new code

### 1.1 Entry point: `configure_logging`

Call this **once** at the top of a script / CLI entry point. It binds `run_id` + `pipeline` to structlog's contextvars and wires both stdout (JSON) and a text file at `logs/{pipeline}/{run_id}.log`.

```python
from fraud_engine.utils.logging import configure_logging, get_logger

run_id = configure_logging("feature-pipeline")
logger = get_logger(__name__)
logger.info("pipeline.start", input_rows=590_540)
```

Library code should **not** call `configure_logging`. It calls `get_logger(__name__)`; if `configure_logging` never ran, a JSON-to-stderr fallback kicks in so nothing crashes.

### 1.2 Function instrumentation: `@log_call`

Wraps a function so every call logs entry, exit (with `duration_ms`), and failure:

```python
from fraud_engine.utils.logging import log_call

@log_call
def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    ...
```

Emits `{qualname}.start`, `{qualname}.done`, `{qualname}.failed`. Shapes only — never full values.

### 1.3 DataFrame snapshots: `log_dataframe`

At pipeline-stage boundaries, emit a richer DataFrame fingerprint:

```python
from fraud_engine.utils.logging import log_dataframe

log_dataframe(features, name="post_merge")
```

Records rows, cols, memory_mb, dtype histogram, NaN count, SHA-256 of the first row. Never emits the actual values.

### 1.4 Per-request correlation: `bind_request_id`

Sprint 5 middleware calls these on every inbound request:

```python
from fraud_engine.utils.logging import bind_request_id, reset_request_id

request_id = bind_request_id()            # generates UUID4 if none given
try:
    ...
finally:
    reset_request_id()
```

`request_id` is a `ContextVar`, so parallel requests under asyncio / thread pools each see their own ID.

### 1.5 Per-run lineage: `run_context`

For non-model scripts (feature builds, one-off downloads, EDA notebooks):

```python
from fraud_engine.utils.tracing import attach_artifact, run_context

with run_context("feature-build", metadata={"source": "Sprint 2"}) as run:
    features = build_features(raw)
    attach_artifact(run, features.head(1000), name="feature_head")
    attach_artifact(run, eda_figure, name="amount_distribution")
```

On entry, creates `logs/runs/{run_id}/` with `run.json` (`status="running"`), `stdout.log`, `stderr.log`, `artifacts/`. On success, rewrites `run.json` with `status="success"` + `duration_ms`. On exception, writes `status="failed"` + `exception_type` / `exception_message` / `traceback` and re-raises. Streams are teed, not redirected — console output still visible in dev.

`attach_artifact` dispatches on type: `Path` → copy, `DataFrame` → `.parquet`, `matplotlib.Figure` → `.png`, `dict|list` → `.json`, anything else → `joblib.dump`.

### 1.6 MLflow for model work: `mlflow_setup`

Every Sprint 3 / 4 script uses the same three calls so runs land in the same experiment tree:

```python
from fraud_engine.utils.mlflow_setup import (
    configure_mlflow, setup_experiment,
    log_dataframe_stats, log_economic_metrics,
)
import mlflow

configure_mlflow()
exp_id = setup_experiment()                 # defaults to settings.mlflow_experiment_name
with mlflow.start_run(experiment_id=exp_id):
    log_dataframe_stats(train_df, prefix="train")
    log_dataframe_stats(val_df, prefix="val")
    # ... train model ...
    log_economic_metrics(fn_rate, fp_rate, total_cost_usd)
```

Cross-reference `run_context` and MLflow via a shared `run_id`:

```python
with run_context("train") as run, mlflow.start_run(run_name=run.run_id):
    ...
```

---

## 2. Reading the logs

### 2.1 Directory layout

```
logs/
├── {pipeline}/{run_id}.log       # human-readable text mirror (local tail -f)
└── runs/{run_id}/
    ├── run.json                  # Run summary: status, metadata, duration
    ├── stdout.log                # tee'd stdout (when capture_streams=True)
    ├── stderr.log                # tee'd stderr
    └── artifacts/
        ├── feature_head.parquet
        ├── amount_distribution.png
        └── summary.json
```

stdout always emits the same JSON line that the text file captures — Sprint 5's log aggregator (ELK / Loki) consumes the JSON stream.

### 2.2 `jq` recipes

```bash
# Every event from one run
jq 'select(.run_id == "abc123...")' logs/feature-pipeline/*.log

# Every slow function (>500ms)
jq 'select(.duration_ms != null and .duration_ms > 500)' logs/**/*.log

# All DataFrame snapshots
jq 'select(.event == "dataframe.snapshot")' logs/**/*.log

# Errors only
jq 'select(.level == "error")' logs/**/*.log

# Request-scoped trail (Sprint 5)
jq 'select(.request_id == "req-xyz")' logs/**/*.log
```

Pipe to `| head -n 5` for readability; drop `--compact-output` to pretty-print.

### 2.3 Run summary

`run.json` is the smallest thing to read to understand what happened:

```bash
cat logs/runs/abc123.../run.json | jq
```

Fields: `run_id`, `pipeline`, `start_time`, `end_time`, `duration_ms`, `status` (`running` / `success` / `failed`), `metadata`, and on failure `exception_type` / `exception_message` / `traceback`.

---

## 3. Tracing a lineage issue end-to-end

A production flow goes **API request → feature lookup → model inference → SHAP → logging**. Reversing that trail after the fact:

1. **Start from the prediction log.** The API (Sprint 5) records every prediction with `request_id`, `decision`, `probability`, and the `run_id` of the model that produced it.

   ```bash
   jq 'select(.event == "prediction.scored" and .request_id == "req-xyz")' \
       logs/api/*.log
   ```

2. **Pivot on `request_id`** to see the full per-request trail: feature-store lookup, inference, SHAP computation, response emission.

   ```bash
   jq 'select(.request_id == "req-xyz")' logs/api/*.log
   ```

3. **Pivot on `run_id`** (from the prediction record) to see how that model was trained:

   ```bash
   cat logs/runs/{run_id}/run.json | jq .metadata
   ```

   The `metadata` block names the training data fingerprint and the MLflow run.

4. **Pivot on the training data hash** to find the raw CSV that fed it. Sprint 0.2 wrote [`data/raw/MANIFEST.json`](../data/raw/MANIFEST.json); match the `first_row_sha256` from the `dataframe.snapshot` event at ingest time against the SHA-256 column there.

That chain — prediction → request → model run → training data — is what this observability layer exists to preserve.

---

## 4. Log levels for this repo

Follow CLAUDE.md §5.5. Concrete fraud-pipeline examples:

| Level | When | Example |
|---|---|---|
| `DEBUG` | Per-row detail, feature values, intermediate computations. Off in prod because feature values may be PII-adjacent. | `logger.debug("feature.computed", user_id=..., velocity_ewm=0.42)` |
| `INFO` | Pipeline stage entry/exit, row counts, durations, run boundaries. | `logger.info("pipeline.start", rows=590540)` / `log_dataframe(df, name="post_merge")` |
| `WARNING` | Degraded but recoverable: schema drift, missing-data fraction above threshold, fallback path taken, PSI > 0.1. | `logger.warning("psi.moderate", feature="amount", psi=0.18)` |
| `ERROR` | Recoverable failure — retry succeeded, fallback model served. | `logger.error("feature_lookup.redis_miss", user_id=..., fell_back_to="postgres")` |
| `CRITICAL` | Unrecoverable, process should terminate. | `logger.critical("model.load_failed", path=..., exc_info=True)` |

Common pitfalls:

- **Don't log at `INFO` inside a hot loop.** Move it to `DEBUG` and aggregate at the loop boundary. A 590,540-row INFO log at one-per-row is ~590k records per run.
- **Don't log feature values at `INFO` even in dev.** DEBUG-only. Production scrubbing relies on this invariant.
- **Don't use `print()`.** CLAUDE.md §5.5. Every `print` hides from the aggregator.

---

## 5. Local dev stack

`make docker-up` runs [`docker-compose.dev.yml`](../docker-compose.dev.yml): Postgres, Redis, MLflow (SQLite backend), Prometheus, Grafana. All ports bind to `127.0.0.1` so nothing leaks to the LAN.

| Service | URL | Purpose |
|---|---|---|
| MLflow | http://localhost:5000 | Experiment tracking UI |
| Prometheus | http://localhost:9090 | Metric scrape + ad-hoc queries |
| Grafana | http://localhost:3000 | Dashboards (Sprint 6 populates) |
| Postgres | 127.0.0.1:5432 | Offline feature store (Sprint 2+) |
| Redis | 127.0.0.1:6379 | Online feature store (Sprint 5+) |

The Prometheus `fraud-api` target will show **DOWN** on `/targets` until Sprint 5 stands up the API's `/metrics` endpoint. That's expected — it's a diagnostic signal that the scrape is wired correctly.

`make docker-down` tears the stack down; volumes persist across restarts.

---

## 6. Related references

- [CLAUDE.md](../CLAUDE.md) §5.5 — authoritative logging rules.
- [`src/fraud_engine/utils/logging.py`](../src/fraud_engine/utils/logging.py) — implementation.
- [`src/fraud_engine/utils/tracing.py`](../src/fraud_engine/utils/tracing.py) — `run_context` + `attach_artifact`.
- [`src/fraud_engine/utils/mlflow_setup.py`](../src/fraud_engine/utils/mlflow_setup.py) — MLflow helpers.
- [`src/fraud_engine/utils/metrics.py`](../src/fraud_engine/utils/metrics.py) — cost & calibration metrics.
- [`notebooks/00_observability_demo.ipynb`](../notebooks/00_observability_demo.ipynb) — runnable demo.
