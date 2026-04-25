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

| Layer              | Choice                      |
| ------------------ | --------------------------- |
| Language           | Python 3.11                 |
| Dep mgmt           | uv (locked uv.lock)         |
| Primary model      | LightGBM                    |
| Experimental ML    | PyTorch                     |
| Explainability     | SHAP (TreeExplainer)        |
| Graph / identity   | torch-geometric + networkx  |
| Noisy-label audit  | cleanlab                    |
| HPO                | Optuna                      |
| Online store       | Redis                       |
| Batch store        | PostgreSQL                  |
| Serving framework  | FastAPI + uvicorn           |
| Validation         | Pydantic + pandera          |
| Logging            | structlog                   |
| Tracking           | MLflow                      |
| Metrics            | prometheus-client           |
| Dashboards         | Grafana                     |

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

### Why PyTorch (experimental track only)

- **Diversity model lineage.** Model B (FraudNet entity-embedding NN)
  and Model C (FraudGNN via torch-geometric) both need PyTorch as
  their execution engine. Keeping one deep-learning framework means
  one set of CUDA/CPU wheels, one autograd model, one
  serialisation format.
- **torch-geometric requires it.** PyG is not portable to TF or JAX;
  adopting PyG (see next row) forces the PyTorch decision anyway.
- **Scoped to experimental.** PyTorch is not on the production
  serving path — Model A (LightGBM) is. Model B is shadow-deployable
  and Model C is batch-only. This keeps the production image slim
  and the inference path free of torch runtime overhead.

**Trade-off:** ~1.5 GB install footprint even on CPU, and a second
model-format surface (`.pt` vs `.joblib`) to document in the model
card. Accepted for the interpretability-vs-recall diversity story.

### Why Redis + PostgreSQL (two-tier feature store)

- **Sub-millisecond lookups (Redis).** Feature serving reads 100+
  cached aggregates per scoring call; Redis's in-memory key-value
  model supports that at O(1) with predictable tail latency.
- **Shared across replicas (Redis).** A horizontally-scaled FastAPI
  fleet needs one consistent view of aggregate state — in-process
  caches can't provide that.
- **Durability via AOF (Redis).** The append-only log gives us
  point-in-time replay for debugging without the overhead of a full
  SQL journal.
- **Batch store (Postgres).** Nightly-refreshed behavioural features
  (Tier 3) and the prediction audit log live in Postgres. SQL gives
  us ad-hoc joins for analyst investigations; Redis TTLs would
  erase the audit trail. Postgres also hosts the offline training
  feature snapshots so batch and online reads can be reconciled.

**Trade-off:** Two stateful services to operate, not one. We accept
the split because real-time and batch have genuinely different
access patterns (hot-key vs analytical), and collapsing them into
one store compromises one of them.

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

### Why Prometheus + Grafana

- **Pull-based scraping (Prometheus).** `prometheus-client` exposes
  counters / histograms on a `/metrics` endpoint; Prometheus pulls
  them on a fixed cadence. No agent sidecar, no push gateway to
  operate, and the scrape loop is the same contract at dev and
  prod.
- **Fraud-native dashboards (Grafana).** Sprint 6 ships pre-built
  panels: P95 latency heatmap, fraud-capture rate by day, PSI
  drift per feature group, Redis hit ratio, threshold-decision
  breakdown by cost bucket. These are the signals oncall watches;
  wiring them into Grafana means the alerting surface is defined
  with the rest of the stack, not bolted on later.
- **Open-source and local-friendly.** Both run from the project's
  `docker-compose.dev.yml` on a laptop; no vendor account needed
  to reproduce the production observability posture.

**Trade-off:** Grafana dashboards are JSON blobs that drift from
the codebase they describe. We mitigate by version-controlling the
dashboard JSON under `configs/grafana/` and treating dashboard
changes as code review items.

## Consequences

- **Positive.** Shared tooling between dev and prod (JSON logs, MLflow
  runs, Grafana dashboards). Fast inference path from day one.
  Interpretability is built-in, not bolted on. Schema violations fail
  loud at ingest. Two-tier feature store matches real production
  patterns at Canadian fintech peers.
- **Negative.** uv is younger than Poetry / pip-tools; if the tool
  becomes abandoned we need a migration plan (escape hatch:
  `uv export -o requirements.txt`). torch + torch-geometric inflate
  install size by ~2 GB — we mitigate with a future CPU-only extra.
  Redis + Postgres + Prometheus + Grafana is four stateful services
  to stand up for a full local reproduction; `docker-compose.dev.yml`
  is the mitigation.
- **Neutral.** Stack skews heavily toward the Python / PyData
  ecosystem. A future Go- or Rust-based serving rewrite for the
  absolute latency floor is possible but not on the current roadmap;
  that would be a follow-up ADR, not a blocker here. MLflow locks us
  to its tracking-server contract; if we move to Weights & Biases or
  a bespoke registry, the experiment history is exportable but
  migration is not free.
- **Revisit triggers.** A GBM cap at ~95 % AUC that DL could close, a
  Redis write-amplification issue under load, an mlflow-3.x API
  regression, or a Grafana / Prometheus replacement driven by a
  managed-observability mandate — any of these warrants a new ADR.

## References

- IEEE-CIS Fraud dataset: https://www.kaggle.com/c/ieee-fraud-detection
- LightGBM paper: Ke et al., 2017
- SHAP paper: Lundberg & Lee, 2017
- structlog docs: https://www.structlog.org
