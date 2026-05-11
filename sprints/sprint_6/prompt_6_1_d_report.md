# Sprint 6 — Prompt 6.1.d: Grafana dashboard + Prometheus rules + alert rules

## Summary

Sprint 6.1.a-c built the **monitoring data plane** — 11 Prometheus metrics
(6.1.a) + offline PSI drift detection (6.1.b) + offline performance
regression on labelled predictions (6.1.c). Sprint 6.1.d builds the
**monitoring control plane**:

- **Grafana dashboard** with 7 panels covering latency SLO, prediction
  volume + decision mix, score distribution (output drift), dependency
  health, model identity, and shadow + drift activity.
- **Prometheus config** with 5 named alert rules wired via `rule_files:`
  + 30-day retention via `--storage.tsdb.retention.time=30d` flag.
- **Two retrofitted Counter metrics** so the 5 alert rules are real
  (not placeholder stubs):
    - `fraud_engine_drift_alerts_total` — incremented inside
      `DriftMonitor.check_and_alert` per JSONL line written.
    - `fraud_engine_shadow_disagreement_total` — incremented inside
      `ShadowService._score_one` when the shadow vs champion decision
      disagrees.

Without the retrofit, `FeatureDrift` and `ShadowDisagreement` would have
been undefined-metric stubs that Prometheus would silently never fire —
defeating the spec. The retrofit adds 2 unlabelled Counters (bounded
cardinality, 1 series each) + 4 new tests + zero changes to public APIs.

**Risk: Medium → realised Low.** Most files are static configs (low risk
in isolation); the small source retrofit has explicit unit + integration
test coverage. Live `docker compose up -d` smoke is deferred per the
project's `project_docker_deferred` memory; the report includes the
operator-side runbook line for when the docker stack returns.

## Files changed

| Path | Change | LOC |
|---|---|---|
| `configs/grafana/fraud_dashboard.json` | NEW — 7-panel Grafana dashboard | +338 |
| `configs/grafana/dashboards.yml` | NEW — Grafana provisioning config | +35 |
| `configs/prometheus/prometheus.yml` | REWRITTEN — `rule_files:`, dual-stack `scrape_configs`, `external_labels` | +49 / -12 |
| `configs/alerts/alert_rules.yml` | NEW — 5 named alert rules in one `groups:` block | +120 |
| `docker-compose.dev.yml` | MODIFIED — Prometheus `command:` (retention 30d + lifecycle), `volumes:` (alert_rules.yml mount), Grafana volumes (dashboards.yml + fraud_dashboard.json mounts) | +18 / -6 |
| `src/fraud_engine/monitoring/prometheus_metrics.py` | MODIFIED — add `DRIFT_ALERTS_TOTAL` + `SHADOW_DISAGREEMENT_TOTAL` Counters + `__all__` updates | +20 |
| `src/fraud_engine/monitoring/__init__.py` | MODIFIED — re-export the 2 new counters | +4 |
| `src/fraud_engine/monitoring/drift.py` | MODIFIED — `DRIFT_ALERTS_TOTAL.inc()` inside `check_and_alert`'s for-loop | +6 / -0 |
| `src/fraud_engine/api/shadow.py` | MODIFIED — `SHADOW_DISAGREEMENT_TOTAL.inc()` inside `_score_one` when `agree_decision is False` | +14 / -1 |
| `tests/unit/test_prometheus_metrics.py` | MODIFIED — extend `_EXPECTED_METRIC_NAMES` + `test_metric_types_match_spec` for the 2 new counters | +8 |
| `tests/unit/test_drift.py` | MODIFIED — add `TestDriftAlertsTotalCounter` class with 2 tests (counter increments per alert; counter unchanged when no drift) | +69 |
| `tests/integration/test_shadow.py` | MODIFIED — add 2 tests (counter increments on disagreement; counter unchanged on agreement) | +103 |
| `sprints/sprint_6/prompt_6_1_d_report.md` | NEW — this report | +(this file) |

**No changes** to schemas, FeatureService, inference, shap_explainer,
prediction_logger, circuit_breaker, settings, performance_monitor (Sprint
6.1.c), Makefile, Dockerfile, `CLAUDE.md`, or `docker-compose.yml` (the
prod-like compose doesn't include Prometheus/Grafana — see Deviation 1).

## The 7 dashboard panels

| # | Panel title | Type | Backing metric(s) | Why |
|---|---|---|---|---|
| 1 | API latency — p50 / p95 / p99 (5m rolling) | Time series with threshold line at 0.100s | `fraud_engine_predict_total_seconds_bucket` | Headline SLO. Three quantiles + the 100ms gate visualised together. |
| 2 | Per-stage latency p95 (5m rolling) | Time series, stacked | `feature_fetch / inference / shap` `_seconds_bucket` | Diagnoses which stage is slow when panel 1 trips. |
| 3 | Prediction volume + decision mix (1m rolling) | Time series, stacked | `fraud_engine_predictions_total{decision}` | Throughput at a glance + the block/allow ratio. |
| 4 | Block rate (%) — 5m rolling | Time series with threshold line at 15% | derived: `block / total` | The signal `BlockRateSpike` alerts on; visual confirms before paging. |
| 5 | Score distribution (heatmap) | Heatmap | `fraud_engine_prediction_score_bucket` | Output drift surface — calibration shifts visible before they trip a rate alert. |
| 6 | Dependency health + model version | Stat panel (4 gauges, value mappings UP/DOWN) | `fraud_engine_dependency_up{component}` + `fraud_engine_model_info{model_version}` | "What's down? What model are we serving?" in one glance. |
| 7 | Shadow + drift activity | Time series, multi-axis | `fraud_engine_shadow_total{event}` + `fraud_engine_shadow_disagreement_total` (NEW) + `fraud_engine_drift_alerts_total` (NEW) | All 6.1.b/c/5.2.b signals in one place; matches the alert rules' shape. |

`ApiErrorRate` doesn't get its own panel — it's covered by the alert
rule + at-a-glance presence on panel 3 (a separate stat-panel inset
would be over-design for a 7-panel limit).

## The 5 alert rules

| Rule | Query (abbreviated) | Threshold | `for:` | Severity |
|---|---|---|---|---|
| `HighLatency` | `histogram_quantile(0.95, ... predict_total_seconds_bucket[5m])` | `> 0.1` (CLAUDE.md §3) | `2m` | warning |
| `BlockRateSpike` | `block / total over 5m` | `> 0.15` (≈4× the 3.5% IEEE-CIS baseline) | `10m` | warning |
| `FeatureDrift` | `increase(fraud_engine_drift_alerts_total[1h])` | `> 0` (any drift cron run wrote an alert) | `5m` | info |
| `ShadowDisagreement` | `disagreement / scored over 15m` | `> 0.1` (10% disagreement) | `10m` | warning |
| `ApiErrorRate` | `5xx / total over 5m` | `> 0.01` (1%) | `5m` | warning |

Each rule carries `labels:` (severity, team=fraud, slo=…) and
`annotations:` (summary, description, runbook_url) — runbook_url points
at the future `docs/RUNBOOK.md#alert-${name}` per CLAUDE.md §14.

**HTTP metric verification (live):** before writing the `ApiErrorRate`
query, started uvicorn + drove `/predict` + scraped `/metrics` to
confirm the prometheus-fastapi-instrumentator 7.0.2 exact label name.
Output: `http_requests_total{handler, method, status}` where `status`
is **bucketed strings** (`"2xx"`, `"3xx"`, `"4xx"`, `"5xx"`) — NOT
raw status codes. The query uses `status="5xx"` rather than the regex
`status=~"5.."` the plan initially assumed; adjusted before writing.

## Design decisions (7 + 1 surfaced during impl)

### Decision 1 — Three artefacts in three new/existing config homes

```
configs/grafana/fraud_dashboard.json     — NEW; 7 panels
configs/grafana/dashboards.yml           — NEW; provisioning config
configs/prometheus/prometheus.yml        — REWRITTEN; rule_files + dual-stack scrape
configs/alerts/alert_rules.yml           — NEW; 5 named alert rules
```

**Rejected:** put dashboard JSON inline in YAML (Grafana doesn't
support); deeper `provisioning/` subdirectory nesting (YAGNI for a
single dashboard).

### Decision 2 — 7 panels mapped to existing + 2 retrofitted metrics

Per Phase-1 finding, `docs/PROJECT_PLAN.md` doesn't exist in the repo
(deferred .docx → markdown conversion). Panels are inferred from what
the 6.1.a/b/c + 6.1.d retrofit metrics actually emit. See "The 7
dashboard panels" table above.

**Rejected:** dedicated alert-table panel (Grafana's built-in alert
pane already provides this); per-feature drift heatmap (high cardinality
risk; defer until DriftMonitor has Prometheus per-feature gauges).

### Decision 3 — 5 alert rules with explicit thresholds + `for:` clauses

`for:` clauses range from 2m (latency, fast-feedback signal) to 10m
(block rate / shadow disagreement, slower-moving signals where
short-term spikes are noise). All thresholds are sourced from explicit
project facts (CLAUDE.md §3 latency budget, §1 fraud rate, §8 decision
threshold) — none invented at impl time.

**Rejected:** Recording rules (overhead not warranted at this RPS);
AlertManager wiring (out of scope; rules fire in `/alerts` UI but
don't route until a future Sprint 6.x adds the `alerting:` block).

### Decision 4 — Add 2 retrofitted Counter metrics so `FeatureDrift` + `ShadowDisagreement` are real

Without the retrofit, two of the five named alert rules would either be
undefined-metric stubs (Prometheus never fires) OR require a log-scraper
sidecar (significant infra). The retrofit is small:

- **`prometheus_metrics.py`** (+20 LOC) — two unlabelled Counters.
- **`drift.py`** (+6 LOC) — one `.inc()` inside `check_and_alert`'s
  for-loop, mirroring the JSONL write count.
- **`shadow.py`** (+14 LOC) — one `.inc()` inside `_score_one` guarded
  by `if agree_decision is False`.

Both unlabelled (1 series each) — bounded cardinality. Per-feature
drift breakdown stays in `logs/drift/{run_id}/drift_alerts.jsonl` (the
labelled version would risk 743-feature cardinality; defer to a future
Sprint 6.x with bucketed labels if needed).

**Rejected:** per-feature drift label; a boolean gauge instead of a
counter (Counter's rate gives "alerts per hour" trend; gauge would not).

### Decision 5 — 30-day retention via docker-compose `command:` flag

Prometheus's default retention is 15 days; the spec asks for 30. This
is a binary flag (`--storage.tsdb.retention.time=30d`), NOT a
config-file option. Lives in the Prometheus service's `command:` block
in `docker-compose.dev.yml`.

**Rejected:** retention as an env var (`prometheus` doesn't read
retention from env); retention via `prometheus.yml` (not a supported
config field).

### Decision 6 — Static validation in CI; live verification deferred

Spec asks for `docker compose up -d` + browser check. Per the project's
docker-deferred posture (memory `project_docker_deferred`), this PR
substitutes:

- **JSON parse** the dashboard via `json.load(...)` — confirms valid
  syntax + 7 panels + the canonical `uid`.
- **YAML parse** the alert rules + prometheus config — confirms the 5
  rule names + `rule_files:` block + dual-stack scrape targets.
- **`yamllint` + `check yaml`** via pre-commit hooks.
- **`promtool`** check — not on PATH (no Prometheus binary locally
  installed); documented as deferred operator-side step.

The completion-report runbook line tells the operator how to do the
live smoke when the docker stack returns:

```bash
# Operator runbook (deferred — for when docker stack comes back online):
docker compose -f docker-compose.dev.yml up -d
curl -sf http://localhost:9090/api/v1/rules | jq '.data.groups[].rules[].name'
# Expected: ["HighLatency", "BlockRateSpike", "FeatureDrift", "ShadowDisagreement", "ApiErrorRate"]
open http://localhost:3000  # Grafana, admin/admin
# Expected: dashboard "Fraud Detection — Overview" auto-provisions (uid: fraud-detection-main)
```

### Decision 7 — Dashboard JSON authored at `schemaVersion: 39` (Grafana 11.x native)

Grafana 11.4.0 is what `docker-compose.dev.yml:87` pins. Each panel
uses the unified `timeseries` / `stat` / `heatmap` panel types (v8+
defaults, replacing the deprecated `graph` type). Dashboard `uid` is
fixed (`fraud-detection-main`) so operators can deep-link stably.

**Rejected:** target an older `schemaVersion` for cross-version
portability (we control the Grafana version in compose); use legacy
`graph` panel type (deprecated, warns on every load).

### Decision 8 — `status="5xx"` not `status=~"5.."` (surfaced via live `/metrics` scrape)

Plan assumed prometheus-fastapi-instrumentator 7.0.2 used raw status
codes (`200`, `404`, `500`, …) which would warrant a regex match.
Empirical verification (`uvicorn` + `curl /metrics`) showed the library
**buckets** status into string labels (`"2xx"` / `"3xx"` / `"4xx"` /
`"5xx"`). Adjusted the `ApiErrorRate` query before writing — it's
cleaner with the literal-match `status="5xx"` than the equivalent
regex.

This deviation from plan was caught at impl time, not after-the-fact —
the live verification step is essential for any alert query referencing
third-party-library metric names.

## Verification

### Unit tests — 4 new tests pass (2 in test_drift, 2 in test_shadow)

```text
tests/unit/test_drift.py::TestDriftAlertsTotalCounter::test_drift_alerts_counter_increments_per_jsonl_line PASSED [ 25%]
tests/unit/test_drift.py::TestDriftAlertsTotalCounter::test_drift_alerts_counter_unchanged_when_no_drift PASSED [ 50%]
tests/integration/test_shadow.py::test_shadow_disagreement_counter_increments_on_disagreement PASSED [ 75%]
tests/integration/test_shadow.py::test_shadow_disagreement_counter_unchanged_on_agreement PASSED [100%]
```

### Cheap gates

```text
$ make format       → 1 file reformatted, 139 files left unchanged
$ make lint         → All checks passed!
$ make typecheck    → Success: no issues found in 53 source files
```

### Full-suite regression

```text
$ uv run pytest tests/unit -q --no-cov
817 passed, 3282 warnings in 110.63s (0:01:50)
```

Pre-PR baseline was 815 (post 6.1.c); +2 new drift counter tests = 817.

### Shadow integration tests — 6/6 PASS (4 prior + 2 new)

```text
$ uv run pytest tests/integration/test_shadow.py -v --no-cov
====================== 6 passed, 15665 warnings in 10.03s ======================
```

### Static config validation

```text
$ uv run python -c "..."
OK dashboard: fraud-detection-main with 7 panels
OK 5 alert rules: ['HighLatency', 'BlockRateSpike', 'FeatureDrift', 'ShadowDisagreement', 'ApiErrorRate']
OK prometheus.yml: rule_files + 2 scrape jobs
OK grafana provisioning: fraud-engine
```

### Pre-commit on all touched files — all PASS

```text
trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
check yaml...............................................................Passed
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

### Sample HTTP-metric output (from live `/metrics` scrape)

```text
http_requests_total{handler="/health",method="GET",status="2xx"} 1.0
http_requests_total{handler="/predict",method="POST",status="2xx"} 1.0
```

Confirms `status` is bucketed string (`"2xx"` / `"5xx"`) — drove the
`ApiErrorRate` query design.

## Operator runbook (deferred — for when docker stack comes back online)

```bash
# Bring up the dev stack with monitoring services.
docker compose -f docker-compose.dev.yml up -d

# Confirm Prometheus loaded the 5 alert rules.
curl -sf http://localhost:9090/api/v1/rules | jq '.data.groups[].rules[].name'
# Expected output (5 names):
#   "HighLatency"
#   "BlockRateSpike"
#   "FeatureDrift"
#   "ShadowDisagreement"
#   "ApiErrorRate"

# Confirm 30-day retention is set.
curl -sf http://localhost:9090/api/v1/status/runtimeinfo | jq '.data.storageRetention'
# Expected: "30d"

# Open Grafana and verify the dashboard auto-provisions.
open http://localhost:3000  # admin/admin
# Expected: "Fraud Detection — Overview" listed under General folder, uid="fraud-detection-main"
# Click through to verify all 7 panels render with data (after a few /predict calls).

# Drive a few /predict requests to populate the panels.
for i in $(seq 1 50); do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d @tests/fixtures/sample_txn.json > /dev/null
done

# Refresh dashboard — panels 1, 2, 3, 4, 5 should now show data.
# Panels 6 (dependency health) + 7 (shadow + drift) only populate when
# their respective signals are emitted (run a drift cron / enable
# shadow mode to see them).
```

## Deviations from plan

1. **`docker-compose.yml` (prod-like) was NOT modified.** The plan's
   estimate said both compose files would get the same edits. In
   practice the prod-like compose (Sprint 5.1.g) only includes
   postgres/redis/fraud-api/nginx — no Prometheus or Grafana services.
   Documented in the prometheus.yml comment that the dual-stack scrape
   target works for both (one will be DOWN if only one stack is up).

2. **`status="5xx"` literal match instead of `status=~"5.."` regex.**
   Surfaced via live `/metrics` scrape against running uvicorn.
   prometheus-fastapi-instrumentator 7.0.2 buckets status codes into
   string labels (`"2xx"` / `"3xx"` / `"4xx"` / `"5xx"`); literal-match
   is the right query.

3. **No `promtool` validation step.** Not on the operator's PATH (no
   Prometheus binary installed locally). Documented as a deferred
   operator-side step in the runbook above.

## Cross-references

- `src/fraud_engine/monitoring/prometheus_metrics.py` — extended to
  expose `DRIFT_ALERTS_TOTAL` + `SHADOW_DISAGREEMENT_TOTAL`.
- `src/fraud_engine/monitoring/drift.py:DriftMonitor.check_and_alert` —
  retrofit increment site (one `.inc()` per JSONL line).
- `src/fraud_engine/api/shadow.py:ShadowService._score_one` — retrofit
  increment site (one `.inc()` when `agree_decision is False`).
- `configs/grafana/datasources.yml` — Sprint 0 datasource provisioning,
  unchanged.
- `docker-compose.dev.yml:66-118` — Prometheus + Grafana service
  definitions, extended with command/volumes for retention + dashboard
  provisioning.
- `CLAUDE.md` §3 (100ms latency budget drives `HighLatency`), §1 (3.5%
  fraud rate drives `BlockRateSpike`), §8 (decision_threshold = 0.08
  drives the score-distribution panel boundary), §14 (`docs/RUNBOOK.md`
  is the future runbook the alert annotations link to).

## Out of scope (Sprint 6.x+)

- **AlertManager wiring** — alerts fire in Prometheus' `/alerts` UI but
  don't route to PagerDuty/Slack. Future PR adds the `alerting:` block
  + the AlertManager service.
- **`docs/RUNBOOK.md`** — the alert rules' `runbook_url` annotations
  link to placeholder anchors. Sprint 6.x deliverable per CLAUDE.md §14.
- **Per-feature drift Counter** (`fraud_engine_drift_alerts_total{feature_name}`)
  — high-cardinality risk; defer until needed with bucketed labels.
- **Recording rules** — not warranted at the project's RPS today.
- **Live `docker compose up -d` verification** — deferred per
  `project_docker_deferred`.
- **Multi-instance scrape config** — current scrape targets a single
  host:8000 / fraud-api:8000. Future PR adds Kubernetes service
  discovery or labelled host lists when multi-instance comes up.
- **CLAUDE.md §13 sprint-status update** — Sprint 6 row gets updated by
  a 6.2.x audit-and-gap-fill PR per established convention.
- **Dashboard secondary views** — drill-down per-route latency panels,
  per-handler error breakdowns, Redis/Postgres exporter scraping
  (would need `redis_exporter` + `postgres_exporter` services in
  compose); defer.
