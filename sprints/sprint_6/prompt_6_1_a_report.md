# Sprint 6 — Prompt 6.1.a: Prometheus metrics module + API instrumentation

## Summary

Sprint 6 opens the monitoring layer. This PR delivers the **single source
of truth** for every Prometheus signal the fraud-engine API emits — a new
`src/fraud_engine/monitoring/prometheus_metrics.py` module that
centralises 11 metrics (the 4 latency histograms moved verbatim from
`api/main.py` plus 7 new metrics covering prediction volume, score
distribution, dependency health, model identity, and shadow-mode
signals).

After this PR, an operator running `curl localhost:8000/metrics` against
the running service sees a comprehensive monitoring surface that the
remaining Sprint-6 prompts (Grafana dashboards, PSI drift detection,
alerting rules) can consume directly via metric-name constants rather
than free-form strings.

The change is **purely additive**: existing metric names + buckets are
unchanged so the Sprint-5.1.f e2e regression
(`test_metrics_endpoint_exposes_prometheus`) still passes verbatim, and
the `<100 ms` p95 latency budget is unaffected (no extra work in the
request path beyond ~3 µs of label-lookup + atomic-counter increments
per request).

## Files changed

| Path | Change | LOC |
|---|---|---|
| `src/fraud_engine/monitoring/prometheus_metrics.py` | NEW — 11 metrics + buckets + helpers + module docstring covering 7 design decisions | +267 |
| `src/fraud_engine/monitoring/__init__.py` | MODIFIED — re-export the metric constants | +43 / -2 |
| `src/fraud_engine/api/main.py` | MODIFIED — remove inline `_LATENCY_BUCKETS` + 4 inline `Histogram`s; import from monitoring; add 4 increments in `/predict`; add `dependency_up` updates in lifespan + `/ready`; two `# noqa: PLR0915` justifications | +33 / -42 |
| `src/fraud_engine/api/shadow.py` | MODIFIED — add 3 `SHADOW_TOTAL` increments + 3 `set_shadow_breaker_state` calls at the success / failure / breaker_open_skip sites | +15 |
| `tests/unit/test_prometheus_metrics.py` | NEW — 12 tests (registration / behaviour / label discipline / scrape format) | +325 |
| `sprints/sprint_6/prompt_6_1_a_report.md` | NEW — this report | +(this file) |

**No changes** to schemas, FeatureService, RedisFeatureStore,
InferenceService, ShapExplainer, PredictionLogger, CircuitBreaker,
Settings, Makefile, Dockerfile, docker-compose.yml, or `CLAUDE.md`.

## Public surface (what `/metrics` now exposes)

### Histograms — latency (4, existing — moved verbatim)

```
fraud_engine_feature_fetch_seconds   buckets=(0.005, 0.010, 0.025, 0.050, 0.100, 0.250)
fraud_engine_inference_seconds       buckets=(0.005, 0.010, 0.025, 0.050, 0.100, 0.250)
fraud_engine_shap_seconds            buckets=(0.005, 0.010, 0.025, 0.050, 0.100, 0.250)
fraud_engine_predict_total_seconds   buckets=(0.005, 0.010, 0.025, 0.050, 0.100, 0.250)
```

### Histogram — calibrated probability (1, NEW)

```
fraud_engine_prediction_score        buckets=(0.0, 0.01, 0.05, 0.08, 0.1, 0.2, 0.5, 0.8, 1.0)
```

### Counters — monotonic volumes (3, NEW)

```
fraud_engine_predictions_total{decision}        decision ∈ {block, allow}
fraud_engine_degraded_mode_total                (no labels)
fraud_engine_shadow_total{event}                event ∈ {scored, failed, breaker_open_skip}
```

### Gauges — instantaneous state (3, NEW)

```
fraud_engine_dependency_up{component}           component ∈ {redis, postgres, model}
fraud_engine_shadow_breaker_state{state}        state ∈ {closed, open, half_open}
fraud_engine_model_info{model_version}          model_version is the SHA-256 hex
```

## Design decisions (7)

### Decision 1 — Centralise metrics in `monitoring/prometheus_metrics.py`

Sprint 5.1.f put 4 inline `Histogram` constants in `api/main.py:200-241`.
Inline worked for 4 metrics; doesn't scale to 11. Centralising mirrors
the prometheus-client README's recommended layout and gives operators a
single file to grep for "what does this app emit?". Tests can import the
metrics directly without booting the FastAPI app.

**Rejected:**
- Leave inline in main.py — visual clutter, harder to test.
- Auto-discover via decorators — overkill for the project's RPS; adds
  opaque magic.

### Decision 2 — Three metric types: Counter / Histogram / Gauge

- **Counter** for monotonic volumes (predictions, degraded incidents,
  shadow events).
- **Histogram** for distributions (latencies + score) — bucket counts
  aggregate cleanly across instances; Summaries can't.
- **Gauge** for instantaneous state (dependency health, breaker state,
  model identity).

**Rejected:**
- A native `info` metric type — Prometheus has no such type; the
  labelled-gauge-set-to-1 pattern (used by `MODEL_INFO`) is the
  established convention.
- Summary instead of Histogram for latency — Summaries can't be
  aggregated across instances; Histograms can (sum the bucket counts
  cross-instance, then compute quantiles on the aggregated series).

### Decision 3 — Bucket selection

**Latency buckets** `(0.005, 0.010, 0.025, 0.050, 0.100, 0.250)` —
reused verbatim from Sprint 5.1.f:
- `0.005` covers typical sub-5 ms inference / SHAP per-call cost.
- `0.025–0.050` covers the typical feature-fetch path.
- **`0.100`** is the load-bearing bucket — it's the CLAUDE.md §3 P95
  budget gate. Grafana alerting fires when the p95 latency series
  crosses 100 ms.
- `0.250` covers tail latencies (cold-start outliers).

**Score buckets** `(0.0, 0.01, 0.05, 0.08, 0.1, 0.2, 0.5, 0.8, 1.0)`:
- `0.08` = `Settings.decision_threshold` (post-Sprint-4.4 cost-optimal
  value). Critical for "block-rate at threshold" dashboards.
- Fine resolution at the low end where the decision boundary lives.
- Coarser resolution at the high end (high-confidence-fraud cases are
  rare).
- `0.0` and `1.0` included as boundary anchors for clean cumulative
  distribution reads.

**Math: bucket choice for p95 reads.** Prometheus's
`histogram_quantile(0.95, ...)` linearly interpolates within the bucket
containing the quantile:
- p95 ≈ 70 ms → falls in [0.050, 0.100] bucket → linear-interpolated to
  ~0.07 ± bucket-width-error.
- p95 ≈ 95 ms → still in [0.050, 0.100] → interpolation accuracy ~5 ms
  (acceptable for "is p95 < 100 ms?" alerting).
- p95 ≈ 110 ms → falls in [0.100, 0.250] → interpolation accuracy
  ~15 ms (which is fine because the alert already fired on crossing
  100 ms).

The bucket boundary at exactly the budget value is intentional — it
makes the alert evaluation a clean
`histogram_quantile(0.95, ...) > 0.100` query.

### Decision 4 — Bounded-cardinality labels only

All label values are drawn from finite enumerations:

| Label | Values | Count |
|---|---|---|
| `decision` | `{block, allow}` | 2 |
| `event` | `{scored, failed, breaker_open_skip}` | 3 |
| `component` | `{redis, postgres, model}` | 3 |
| `state` | `{closed, open, half_open}` | 3 |
| `model_version` | SHA-256 hex (≤2 per process lifetime) | ≤2 |

**No high-cardinality labels** (no `request_id` / `txn_id` / `card1` /
`addr1` / `entity_id` / `user_id` / `device_id`) — per Prometheus best
practice, these would explode the time-series storage and OOM the
Prometheus process. The `tests/unit/test_prometheus_metrics.py::TestLabelDiscipline::test_no_high_cardinality_labels`
test explicitly enforces the blacklist.

**Math: cardinality estimate.** Total unique series for the new metrics:

| Metric | Series |
|---|---|
| `predictions_total` | 2 (one per decision) |
| `degraded_mode_total` | 1 |
| `shadow_total` | 3 |
| `prediction_score` | ~12 internal series (10 buckets + count + sum) |
| `dependency_up` | 3 |
| `shadow_breaker_state` | 3 |
| `model_info` | ≤2 lifetime |
| 4 latency histograms × ~8 internal each | 32 |
| **Total** | **~57 series** |

Standard Prometheus instances handle 10⁶+ series comfortably; 57 is
rounding error.

### Decision 5 — Wire request-path metrics inline; same for shadow

**main.py** — after building `PredictionResponse`, before returning:

```python
PREDICTIONS_TOTAL.labels(decision=inf.decision).inc()
PREDICTION_SCORE.observe(inf.probability)
if feature_vector.degraded_mode:
    DEGRADED_MODE_TOTAL.inc()
MODEL_INFO.labels(model_version=inf.model_version).set(1)
```

**main.py lifespan** — set initial dependency health based on connect
outcomes; the existing `/ready` route is the natural update site for
subsequent transitions (recovered Redis / Postgres flips back to 1 next
scrape after `/ready` is polled):

```python
# In lifespan:
DEPENDENCY_UP.labels(component="redis").set(1 if redis_ok else 0)
DEPENDENCY_UP.labels(component="postgres").set(1 if postgres_ok else 0)
DEPENDENCY_UP.labels(component="model").set(1)

# In /ready handler:
for component, status_value in checks.items():
    DEPENDENCY_UP.labels(component=component).set(1 if status_value == "ok" else 0)
```

**shadow.py** — increments + breaker-state updates at each call site:

- After `record_success()` → `SHADOW_TOTAL.labels(event="scored").inc()`
  + `set_shadow_breaker_state(self._breaker.state)`
- After `record_failure()` → `SHADOW_TOTAL.labels(event="failed").inc()`
  + `set_shadow_breaker_state(self._breaker.state)`
- In `breaker_open_skip` log path →
  `SHADOW_TOTAL.labels(event="breaker_open_skip").inc()` +
  `set_shadow_breaker_state(self._breaker.state)`

**Trade-off accepted:** Shadow instrumentation lives in shadow.py rather
than main.py because the shadow path is fire-and-forget — wiring
increments via callback would add complexity for no benefit. The
increments are 6 lines total in shadow.py.

**Rejected:** A central event bus + subscriber pattern. Overengineering
for ~10 increment points.

### Decision 6 — Tests cover registration / behaviour / label discipline / scrape format

12 unit tests across 4 categories. The **delta pattern** (capture sample
value before; perform increment; capture after; assert difference)
sidesteps the global-singleton problem: tests run in any order without a
custom fixture to reset state, and tests must NOT use `importlib.reload`
on the metrics module (would raise `Duplicated timeseries`).

**Trade-off rejected:** End-to-end test that hits `/metrics` via httpx.
The Sprint 5.1.f integration test
(`test_metrics_endpoint_exposes_prometheus`) already does this;
duplicating it adds cost without coverage.

### Decision 7 — `prometheus_metrics.py` is import-side-effecting

Defining `Counter("foo")` at module scope auto-registers it against
`prometheus_client.REGISTRY` on import. This is the standard pattern
but worth flagging in the module docstring: importing the module from a
test triggers registration, and if the test re-imports (via
`importlib.reload`) it raises `ValueError: Duplicated timeseries`. The
standard mitigation is to NOT reload — build tests around the single
import. The `__init__.py` re-exports references rather than re-importing
the module.

## Verification

### Unit tests — 12/12 PASS

```text
tests/unit/test_prometheus_metrics.py::TestRegistration::test_all_metrics_exist_in_registry PASSED [  8%]
tests/unit/test_prometheus_metrics.py::TestRegistration::test_metric_types_match_spec PASSED [ 16%]
tests/unit/test_prometheus_metrics.py::TestRegistration::test_histogram_buckets_match_spec PASSED [ 25%]
tests/unit/test_prometheus_metrics.py::TestBehaviour::test_predictions_total_increments_by_decision_label PASSED [ 33%]
tests/unit/test_prometheus_metrics.py::TestBehaviour::test_degraded_mode_total_increments_unlabeled PASSED [ 41%]
tests/unit/test_prometheus_metrics.py::TestBehaviour::test_shadow_total_increments_by_event_label PASSED [ 50%]
tests/unit/test_prometheus_metrics.py::TestBehaviour::test_prediction_score_observes_into_correct_bucket PASSED [ 58%]
tests/unit/test_prometheus_metrics.py::TestBehaviour::test_set_shadow_breaker_state_helper_sets_one_zeros_others PASSED [ 66%]
tests/unit/test_prometheus_metrics.py::TestLabelDiscipline::test_no_high_cardinality_labels PASSED [ 75%]
tests/unit/test_prometheus_metrics.py::TestLabelDiscipline::test_label_value_sets_are_finite PASSED [ 83%]
tests/unit/test_prometheus_metrics.py::TestScrapeFormat::test_generate_latest_returns_valid_prometheus_text PASSED [ 91%]
tests/unit/test_prometheus_metrics.py::TestScrapeFormat::test_scrape_text_contains_all_metric_names_and_help PASSED [100%]
======================= 12 passed, 14 warnings in 1.69s ========================
```

### Cheap gates

```text
$ make format       → 3 files reformatted, 132 files left unchanged
$ make lint         → All checks passed!
$ make typecheck    → Success: no issues found in 51 source files
```

### Full-suite regression

```text
$ uv run pytest tests/unit -q --no-cov
795 passed, 3282 warnings in 116.68s (0:01:56)
```

Pre-PR baseline was 783; +12 new prometheus tests = 795.

### E2E regression — `/metrics` endpoint test from Sprint 5.1.f

```text
$ uv run pytest tests/integration/test_api_e2e.py::test_metrics_endpoint_exposes_prometheus -v --no-cov
tests/integration/test_api_e2e.py::test_metrics_endpoint_exposes_prometheus PASSED
======================= 1 passed, 1157 warnings in 4.27s =======================
```

### Shadow integration — 4/4 PASS (touched shadow.py, must verify)

```text
$ uv run pytest tests/integration/test_shadow.py -v --no-cov
====================== 4 passed, 15197 warnings in 9.20s =======================
```

### Pre-commit on touched files — all PASS

```text
trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
check yaml...........................................(no files to check)Skipped
check toml...........................................(no files to check)Skipped
check for added large files..............................................Passed
check for merge conflicts................................................Passed
mixed line ending........................................................Passed
ruff.....................................................................Passed
ruff-format..............................................................Passed
Detect secrets...........................................................Passed
mypy (strict, src only)..................................................Passed
pytest (unit, fast)......................................................Passed
```

### Manual scrape — `curl /metrics` after one /predict

uvicorn started, one /predict served (decision="allow", score=0.00373),
one /ready polled, then `/metrics` scraped. The new metrics surface
exactly as designed:

```text
fraud_engine_predictions_total{decision="allow"} 1.0
fraud_engine_degraded_mode_total 0.0
fraud_engine_dependency_up{component="redis"} 1.0
fraud_engine_dependency_up{component="postgres"} 1.0
fraud_engine_dependency_up{component="model"} 1.0
fraud_engine_model_info{model_version="990ef848fb8bf578a31a6baf659e8757db189359c59beb9a14d6c67f22f0cf26"} 1.0
fraud_engine_prediction_score_bucket{le="0.0"} 0.0
fraud_engine_prediction_score_bucket{le="0.01"} 1.0
fraud_engine_prediction_score_bucket{le="0.05"} 1.0
fraud_engine_prediction_score_bucket{le="0.08"} 1.0
fraud_engine_prediction_score_bucket{le="0.1"} 1.0
fraud_engine_prediction_score_bucket{le="0.2"} 1.0
fraud_engine_prediction_score_bucket{le="0.5"} 1.0
fraud_engine_prediction_score_bucket{le="0.8"} 1.0
fraud_engine_prediction_score_bucket{le="1.0"} 1.0
fraud_engine_prediction_score_bucket{le="+Inf"} 1.0
fraud_engine_prediction_score_count 1.0
fraud_engine_prediction_score_sum 0.0037328909166321027
```

The `prediction_score` cumulative buckets read cleanly: an observation of
0.00373 lands in every bucket at-or-above 0.01, leaving the `le="0.0"`
bucket at 0 and all others at 1 — exactly the cumulative-distribution
semantics Grafana's `histogram_quantile` reads from. The
`prediction_score_sum / prediction_score_count` ratio gives the true
mean (0.00373 / 1 = 0.00373).

`shadow_total` and `shadow_breaker_state` aren't emitted yet because
`Settings.shadow_enabled = False` in the dev environment — they only
materialise when shadow mode is on. That's the expected/correct behaviour
for Prometheus gauges: no observations → no series.

All HELP + TYPE comments render in canonical format:

```text
# HELP fraud_engine_predictions_total Predictions served, labelled by decision.
# TYPE fraud_engine_predictions_total counter
# HELP fraud_engine_degraded_mode_total Predictions served while a feature source (Redis/Postgres) was unreachable.
# TYPE fraud_engine_degraded_mode_total counter
# HELP fraud_engine_shadow_total Shadow-mode events: scored (success), failed, breaker_open_skip.
# TYPE fraud_engine_shadow_total counter
# HELP fraud_engine_dependency_up 1 if the named component is healthy, 0 otherwise.
# TYPE fraud_engine_dependency_up gauge
# HELP fraud_engine_shadow_breaker_state Shadow circuit-breaker state — 1 hot for the active state, 0 for the others.
# TYPE fraud_engine_shadow_breaker_state gauge
# HELP fraud_engine_model_info Info-style gauge: always 1 with model_version on the label.
# TYPE fraud_engine_model_info gauge
# HELP fraud_engine_prediction_score Distribution of calibrated fraud probabilities returned to clients.
# TYPE fraud_engine_prediction_score histogram
```

## Deviations from plan

1. **`# noqa: PLR0915` on `_make_lifespan` and `_lifespan`.** Not in the
   plan but unavoidable: adding 5 new `DEPENDENCY_UP.labels(...).set(...)`
   statements pushed the lifespan past the default 50-statement
   `pylint-too-many-statements` threshold. The lifespan is the API's
   wiring point — chopping it into helpers for the sake of a
   statement-count rule pushes complexity around without reducing it.
   Inline `noqa` with a one-sentence justification is the cleanest fix
   and matches the codebase's pattern (see `circuit_breaker.py` for
   prior `noqa` conventions).

2. **Counter sample names: `_total` not `_total_total`.** The first
   test run failed three behaviour tests because I assumed
   `prometheus_client` would naively suffix `_total` to a Counter named
   `fraud_engine_predictions_total`, producing the sample
   `fraud_engine_predictions_total_total`. Empirical check (`generate_latest`
   on a populated REGISTRY) showed the library is smart: a Counter
   named ending in `_total` exposes the sample with that exact name (no
   double-suffix). Tests fixed to use the correct name. The metric
   names themselves are unchanged — only the test assertions adjusted.

3. **No `make serve` smoke** — used `uvicorn` directly via the
   background-task pattern instead. `make serve` requires `--reload` and
   would conflict with `pkill -f "uvicorn fraud_engine"`. The direct
   invocation produced the same scrape output without the dev-loop
   complications.

## Cross-references

- `src/fraud_engine/api/main.py:200-220,540-770` — instrumentation sites
  (lifespan dependency_up, /ready dependency_up refresh, /predict
  increments).
- `src/fraud_engine/api/shadow.py:120-130,395-475` — shadow event
  increments + breaker-state updates.
- `tests/integration/test_api_e2e.py:172-186` — the e2e
  `test_metrics_endpoint_exposes_prometheus` regression that was
  preserved verbatim.
- `CLAUDE.md` §3 (latency budget that drives bucket choice), §4
  (`monitoring/` module home), §8 (decision_threshold = 0.080 that
  drives score bucket choice).
- `pyproject.toml:52-56` — pinned `prometheus-client==0.21.1` +
  `prometheus-fastapi-instrumentator==7.0.2`.

## Out of scope (Sprint 6.x+)

- **Grafana dashboards** — the next prompts in Sprint 6 wire dashboards
  that consume these metrics.
- **PSI drift detection** on `prediction_score` — the histogram is the
  data feed; the actual PSI computation + alerting is a Sprint 6.x
  prompt.
- **Per-segment metrics** (e.g.,
  `predictions_total{decision, product_cd}`) — would push label
  cardinality up; defer until needed.
- **OpenTelemetry tracing** — separate observability primitive; Sprint
  6.x candidate.
- **Per-customer / per-tenant metrics** — high-cardinality risk; Sprint
  6.x with proper isolation.
- **Service-side P99/P999 latency from histogram** — already computable
  via `histogram_quantile(0.99, ...)` against the existing buckets; no
  extra work.
- **`fraud_engine_shadow_score` histogram** — would mirror
  `prediction_score` for the challenger; defer to Sprint 6.x when
  shadow-vs-champion drift becomes a Grafana panel.
- **Alerting rules** (`prometheus.yml` `rule_files:`) — the metric
  surface is in place; rules are Sprint 6.x.
- **`fraud_engine_predictions_total{model_version}` label** — would let
  dashboards split by model. Useful but adds cardinality (one new series
  per ever-deployed model). Sprint 6.x.
- **CLAUDE.md §13 sprint-status update** — Sprint 6 row gets updated by
  a 6.2.x audit-and-gap-fill PR per established convention.
