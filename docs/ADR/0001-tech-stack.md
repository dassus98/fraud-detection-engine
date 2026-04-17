# ADR 0001: Core Tech Stack

- **Status:** Accepted
- **Date:** 2026-04-17
- **Sprint:** 0 (Bootstrap)

## Context

We are building a real-time fraud detection engine that must score
card-not-present transactions with a p99 latency budget of a few
milliseconds, produce explanations that analysts can action, and
survive a production rollout with audit-grade traceability. Every
component choice below is in service of one or more of those
constraints: **latency, interpretability, observability, or
reproducibility**.

This ADR freezes the Sprint 0 stack. Swapping a component later is
fine but requires a new ADR justifying the move.

## Decision

| Layer             | Choice                 |
| ----------------- | ---------------------- |
| Language          | Python 3.11            |
| Dep mgmt          | uv (locked uv.lock)    |
| Primary model     | LightGBM               |
| Explainability    | SHAP (TreeExplainer)   |
| Graph / identity  | torch-geometric + networkx |
| Noisy-label audit | cleanlab               |
| HPO               | Optuna                 |
| Online store      | Redis                  |
| Serving framework | FastAPI + uvicorn      |
| Validation        | Pydantic + pandera     |
| Logging           | structlog              |
| Tracking          | MLflow                 |
| Metrics           | prometheus-client      |

### Why LightGBM

- **Latency.** A compiled C++ inference path with pre-built histograms
  hits < 2 ms p99 on tabular inputs with hundreds of features, well
  inside our serving budget.
- **Interpretability.** Gradient-boosted trees surface feature-level
  importance natively and combine cleanly with SHAP for per-transaction
  explanations — a hard requirement for analyst review in Sprint 5.
- **Tabular dominance.** On tabular fraud benchmarks (IEEE-CIS,
  Sparkov, Vesta), GBMs consistently beat deep learning by 1–3 AUC
  points with 10× less training time and 100× less inference compute.
- **Missing-value handling.** Native `NaN` support means less
  pre-processing and fewer schema constraints.

**Trade-off:** LightGBM cannot learn non-tabular structure (graphs,
sequences). We compensate in Sprint 2 with torch-geometric features
that flatten graph signal into node embeddings, then feed those into
LightGBM. Deep end-to-end models were considered and rejected: the
latency and interpretability gap is not worth the small headline AUC
gain.

### Why Redis

- **Sub-millisecond lookups.** Feature serving reads 100+ cached
  aggregates per scoring call; Redis's in-memory key-value model
  supports that at O(1) with predictable tail latency.
- **Shared across replicas.** A horizontally-scaled FastAPI fleet
  needs one consistent view of aggregate state — in-process caches
  can't provide that.
- **Durability via AOF.** The append-only log gives us point-in-time
  replay for debugging without the overhead of a full SQL journal.

**Trade-off:** Redis is an extra operational surface (memory sizing,
eviction policy, failover). We accept the cost in exchange for
latency headroom. For offline feature store and audit log we use
Postgres, not Redis.

### Why FastAPI + uvicorn

- **Async I/O.** Serving path is I/O-bound on Redis + Postgres lookups;
  async lets a single worker handle multiple inflight requests.
- **Pydantic-native.** Request / response schemas are the same
  Pydantic models we use for Settings validation, so one mental model
  for contracts.
- **Auto OpenAPI.** Free spec for downstream integration teams.

**Trade-off:** Starlette's middleware ecosystem is thinner than
Flask's. Acceptable — we need very few middlewares (auth, request ID,
metrics).

### Why pandera

- **Runtime enforcement.** Pandera validates DataFrames at each
  pipeline stage boundary, catching schema drift before it corrupts a
  training run or inference request. Pure type hints can't catch a
  renamed column in a parquet file.
- **Declarative.** Schemas live in `src/fraud_engine/schemas/` and in
  `configs/schemas.yaml`, reviewed like any other contract.

**Trade-off:** A few ms of validation overhead per batch. Negligible
at our volume; invaluable for "why did this pipeline silently produce
garbage" debugging.

### Why structlog

- **Machine-readable.** JSON output drops directly into Loki / ELK
  without a parser. Every record is already key-value structured.
- **Context propagation.** `contextvars` attach `run_id` and
  `pipeline` at render time, so logs from child tasks correlate.
- **Bridges stdlib.** Third-party libraries that use Python's stdlib
  `logging` inherit our formatting via `ProcessorFormatter`, so we
  don't get double-emitted records or split log formats.

**Trade-off:** More ceremony than stdlib logging (processor chain,
configuration). The payoff — traceable pipelines — is worth it.

## Consequences

- **Positive.** Shared tooling between dev and prod (JSON logs, MLflow
  runs). Fast inference path from day one. Interpretability is
  built-in, not bolted on. Schema violations fail loud at ingest.
- **Negative.** uv is younger than Poetry / pip-tools; if the tool
  becomes abandoned we need a migration plan (escape hatch:
  `uv export -o requirements.txt`). torch + torch-geometric inflate
  install size by ~2 GB — we mitigate with a future CPU-only extra.
- **Revisit triggers.** A GBM cap at ~95 % AUC that DL could close, a
  Redis write-amplification issue under load, or an mlflow-3.x API
  regression — any of these warrants a new ADR.

## References

- IEEE-CIS Fraud dataset: https://www.kaggle.com/c/ieee-fraud-detection
- LightGBM paper: Ke et al., 2017
- SHAP paper: Lundberg & Lee, 2017
- structlog docs: https://www.structlog.org
