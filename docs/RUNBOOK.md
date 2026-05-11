# Operations Runbook — Fraud Detection Engine

> Companion to [`docs/OBSERVABILITY.md`](OBSERVABILITY.md).
> When PagerDuty pages you about an alert from `configs/alerts/alert_rules.yml`,
> open this file, jump to the alert section, and follow the ordered steps.

This runbook covers:

- **The 5 named alert rules** from Sprint 6.1.d (`HighLatency`, `BlockRateSpike`, `FeatureDrift`, `ShadowDisagreement`, `ApiErrorRate`) — what they mean, how to investigate, how to mitigate.
- **Two cross-cutting procedures** — how to safely roll back a model, and how to trigger a retraining run.
- **Three appendices** — log file locations, useful `jq` recipes for the JSONL streams, and useful PromQL queries beyond the alert rules.

The 5 alert rules' `runbook_url` annotations all point at `docs/RUNBOOK.md#alert-${name}` — the anchors below match those URLs verbatim.

## Monitoring at a glance

The monitoring stack lives across three layers (Sprint 6.1.a → 6.1.d):

| Layer | Sprint | What it gives you |
|---|---|---|
| Live request-path metrics | [6.1.a](../sprints/sprint_6/prompt_6_1_a_report.md) | 11 Prometheus metrics surfaced on `/metrics`: 4 latency histograms, score distribution, per-decision counter, dependency gauges, model_info. |
| Offline input-feature drift | [6.1.b](../sprints/sprint_6/prompt_6_1_b_report.md) | `DriftMonitor` computes PSI on 743 features against a frozen training baseline; alerts written to `logs/drift/{run_id}/drift_alerts.jsonl`. |
| Offline labelled-prediction performance | [6.1.c](../sprints/sprint_6/prompt_6_1_c_report.md) | `PerformanceMonitor` tracks rolling AUC / AUC-PR / cost vs training baseline; alerts written to `logs/performance/{run_id}/performance_alerts.jsonl`. |
| Dashboard + alert rules + Prometheus config | [6.1.d](../sprints/sprint_6/prompt_6_1_d_report.md) | 7-panel Grafana dashboard at `localhost:3000` (uid `fraud-detection-main`); 5 named alert rules with 30-day Prometheus retention. |
| End-to-end integration test | [6.1.e](../sprints/sprint_6/prompt_6_1_e_report.md) | `tests/integration/test_monitoring_e2e.py` exercises the full chain: live `/predict` → `/metrics` scrape → counter assertions; synthetic drift → counter delta → alert-threshold static eval. |

The data plane (6.1.a + retrofits in 6.1.d) feeds the control plane (6.1.d's dashboard + alert rules). This runbook is the operator-side closure.

## How to use this runbook

- **Alert anchor convention.** Each alert section's heading carries an explicit anchor (e.g. `{#alert-highlatency}`) that matches the `runbook_url` annotation in `configs/alerts/alert_rules.yml`. The `tests/integration/test_monitoring_e2e.py::test_dashboard_and_alerts_reference_emitted_metrics` test enforces alert ↔ emitted-metric agreement; a future Sprint 6.x can add a similar test for runbook-anchor ↔ alert-rule agreement.
- **Each alert section has the same shape:** *what it means → common causes → investigation (numbered, copy-pasteable commands) → mitigation (least- to most-disruptive).*
- **Docker stack assumed up.** Many investigation commands assume `docker compose -f docker-compose.dev.yml up -d` is running. Where that assumption matters, the command block is prefixed with `# Once docker stack is up:` per the `project_docker_deferred` posture.
- **Severity guides response speed.** `warning` severity → 1-hour SLO to acknowledge + 4-hour SLO to remediate. `info` severity → next-business-day acceptable.

---

## Alert: HighLatency {#alert-highlatency}

**Rule:** `histogram_quantile(0.95, sum by (le) (rate(fraud_engine_predict_total_seconds_bucket[5m]))) > 0.1`
**Threshold:** p95 above 100 ms (CLAUDE.md §3 SLO budget)
**Trigger window:** sustained for 2 m
**Severity:** warning

### What it means

The 95th-percentile end-to-end `/predict` latency over the last 5 minutes exceeds the project's 100 ms SLO budget. The budget covers feature fetch + LightGBM inference + SHAP top-k + response serialization, and excludes network round-trip (the SLO is measured at the application boundary, not the wire).

A single slow request doesn't trip this — `histogram_quantile` over a 5 m window smooths transient blips. A sustained breach indicates a systemic slowdown: something in the request path got slower and stayed that way.

### Common causes

- Redis lookup slowness (network blip, replica failover, large entity history).
- Postgres pool saturation (other workloads competing for connections).
- LightGBM model artefact unusually slow to serialise/deserialise (rare; only on cold start).
- Python GIL contention from a sibling process on the same host.
- A recent deploy regressed a per-stage cost (look at panel 2 of the dashboard).

### Investigation

1. **Open dashboard panel 1** (API latency p50/p95/p99) at `http://localhost:3000/d/fraud-detection-main`. Confirm the alert is real (the panel shows the same time series the rule evaluates).

2. **Open dashboard panel 2** (per-stage latency p95) to see which stage is slow:
   - **feature_fetch slow** → suspect Redis or Postgres.
   - **inference slow** → suspect LightGBM artefact corruption or GIL contention.
   - **shap slow** → suspect SHAP top-k cardinality (rare).

3. **Check dependency health (panel 6).** If Redis or Postgres shows DOWN, the `/predict` path is in degraded mode (Tier-1 features only); cost is high because the FeatureService falls through to population defaults. The `/ready` probe also returns 503 — confirm with:
   ```bash
   # Once docker stack is up:
   curl -s http://localhost:8000/ready | jq .
   ```

4. **If feature_fetch is slow but Redis + Postgres look healthy**, check Redis latency directly:
   ```bash
   docker compose -f docker-compose.dev.yml exec redis redis-cli --latency
   ```

5. **If inference is slow**, verify the loaded model version matches expectation:
   ```bash
   curl -s -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d @tests/fixtures/sample_txn.json | jq .model_version
   ```
   Compare to `models/sprint3/lightgbm_model_manifest.json:content_hash`. If they differ, the API is serving an older artefact than what's on disk — see [How to safely roll back a model](#how-to-safely-roll-back-a-model) for the swap procedure (in reverse).

6. **Inspect uvicorn logs** for any `WARNING` or `ERROR` level entries in the request path:
   ```bash
   docker compose -f docker-compose.dev.yml logs fraud-api --tail 200 | jq -c 'select(.level == "warning" or .level == "error")'
   ```

### Mitigation

1. **If the cause is a transient Redis/Postgres blip**, no action needed — the alert auto-clears once latency drops back below 100 ms for 2 m.

2. **If the cause is a recent deploy regressing per-stage latency**, roll back the model OR the API container — see [How to safely roll back a model](#how-to-safely-roll-back-a-model).

3. **If the cause is a sustained Redis slowness** (e.g., entity history grew very large), increase the per-feature TTL in `configs/redis_feature_store.yaml` to reduce churn, OR scale Redis vertically.

4. **If the cause is GIL contention from a sibling process**, isolate the FastAPI worker on its own host or set CPU affinity.

5. **If none of the above apply**, escalate to the on-call ML platform engineer with a link to this alert + the dashboard time range.

---

## Alert: BlockRateSpike {#alert-blockratespike}

**Rule:** `(sum(rate(fraud_engine_predictions_total{decision="block"}[5m])) / sum(rate(fraud_engine_predictions_total[5m]))) > 0.15`
**Threshold:** block rate above 15% (≈4× the IEEE-CIS baseline fraud rate of 3.5%)
**Trigger window:** sustained for 10 m
**Severity:** warning

### What it means

The fraction of `/predict` requests that returned `decision="block"` over the last 5 minutes is above 15%. Expected baseline is around 3.5% (the IEEE-CIS fraud rate from CLAUDE.md §1) at the post-Sprint-4.4 cost-optimal threshold (`Settings.decision_threshold = 0.08`). A spike to 15%+ suggests one of: a real attack wave, a feature-pipeline regression, a marketing-driven traffic-mix shift, or a model calibration drift.

The 10 m sustained-window guards against bursty traffic samples — a single 30-second spike from a known-batch caller doesn't trip it.

### Common causes

- An actual fraud attack (the hardest case to rule in or out — needs business context).
- Feature pipeline regression: a Tier-2/3 aggregation broken, so the model sees null-imputed features and over-blocks.
- Marketing-driven traffic mix shift (new geo, new device, new product type) where the model is mis-calibrated.
- Score-distribution drift moving mass across the 0.08 decision boundary.
- Decision threshold accidentally lowered (`Settings.decision_threshold` overridden in `.env`).

### Investigation

1. **Open dashboard panel 4** (block rate %) — confirm the alert by visual.

2. **Open dashboard panel 5** (score distribution heatmap) — look for mass shifting toward the high end. If the bulk of probability mass moved across the 0.08 boundary, the model's calibration has drifted (see [FeatureDrift](#alert-featuredrift) and [ShadowDisagreement](#alert-shadowdisagreement)).

3. **Check the predictions audit log** for the top-k blocked features:
   ```bash
   # Once docker stack is up:
   docker compose -f docker-compose.dev.yml exec postgres psql -U fraud -d fraud -c \
     "SELECT decision, count(*), avg(score), avg(latency_ms) FROM predictions
      WHERE created_at > NOW() - INTERVAL '15 minutes'
      GROUP BY decision;"
   ```

4. **Inspect the top-reasons of recent blocks** to see if a single feature is dominating:
   ```bash
   docker compose -f docker-compose.dev.yml exec postgres psql -U fraud -d fraud -c \
     "SELECT top_reasons FROM predictions
      WHERE decision = 'block' AND created_at > NOW() - INTERVAL '15 minutes'
      ORDER BY created_at DESC LIMIT 50;" | head -100
   ```
   If one `feature_name` appears in ≥80% of blocks, that's likely the regression.

5. **Confirm the decision threshold** matches the post-Sprint-4.4 value:
   ```bash
   curl -s http://localhost:8000/health
   # decision_threshold isn't surfaced on /health by default; check .env
   grep DECISION_THRESHOLD .env
   ```

6. **Compare against the most-recent shadow comparison report** (Sprint 5.2.c) to see if the challenger model also blocks at the elevated rate. If yes → likely a real attack. If no → likely a champion-model regression.
   ```bash
   ls -la reports/shadow_compare_*.md | tail -3
   cat reports/shadow_compare_$(date +%Y-%m-%d).md  # if today's report exists
   ```

### Mitigation

1. **If the cause is a confirmed real attack**, no action needed — the model is doing its job. Notify the fraud-ops team so they can monitor for downstream patterns (chargebacks, refund requests).

2. **If the cause is a feature-pipeline regression**, fix the upstream pipeline stage and redeploy. Running a quick test:
   ```bash
   uv run pytest tests/integration/test_api_e2e.py::test_predict_valid_payload_returns_response -v
   ```

3. **If the cause is calibration drift**, trigger a retraining run — see [How to trigger retraining](#how-to-trigger-retraining).

4. **If the cause is a marketing-driven traffic mix shift**, raise the `decision_threshold` in `.env` (e.g., `DECISION_THRESHOLD=0.12`) to reduce sensitivity until the next retraining run incorporates the new traffic. Restart the API to pick up the new value.

5. **As a last resort**, [roll back to the previous model](#how-to-safely-roll-back-a-model) if recent deploy correlates with the spike start.

---

## Alert: FeatureDrift {#alert-featuredrift}

**Rule:** `increase(fraud_engine_drift_alerts_total[1h]) > 0`
**Threshold:** any drift alert written in the last hour
**Trigger window:** sustained for 5 m
**Severity:** info

### What it means

`DriftMonitor.check_and_alert` (Sprint 6.1.b, run periodically as an offline batch) wrote at least one `drift_alerts.jsonl` line to `logs/drift/{run_id}/` in the last hour. Each line corresponds to one feature whose PSI exceeds `Settings.psi_alert_threshold` (default 0.20, industry-conservative).

PSI bands (industry convention):
- `< 0.10` — no significant population shift.
- `0.10–0.25` — moderate shift; investigate.
- `> 0.25` — significant shift; model re-fit likely needed.

This is `info` severity (not `warning`) because feature drift is often benign — a marketing campaign, seasonality, an upstream data-pipeline change. The alert tells you to look; it doesn't tell you to act immediately.

### Common causes

- Seasonal traffic shift (holiday week, end-of-month payday, etc.).
- Marketing campaign brings new geo / device / product mix.
- Upstream data-pipeline change (a new column added, a missing-value imputation rule changed).
- Real population shift requiring a model retrain.
- The drift cron's `recent_window` was unusually small (e.g., a low-traffic hour) — high PSI from sampling noise rather than real drift.

### Investigation

1. **Find the most-recent `drift_alerts.jsonl` file**:
   ```bash
   ls -la logs/drift/*/drift_alerts.jsonl | tail -5
   ```

2. **List the drifted features sorted by PSI**:
   ```bash
   cat logs/drift/*/drift_alerts.jsonl | jq -s 'sort_by(-.psi) | .[:20] | .[]'
   ```

3. **For the top-3 features**, check whether the drift is plausible (look at the feature name semantically):
   - Velocity / EWM features → expected to shift with traffic patterns.
   - Identity / device features → real drift if the source IP geo distribution changed.
   - Tier-3 target-encoded features → suspicious if drifted; suggests training distribution mismatch.

4. **Check sample sizes** (small recent windows give noisy PSI):
   ```bash
   cat logs/drift/*/drift_alerts.jsonl | jq -s '[.[] | .n_recent] | min, max, length'
   ```
   If `n_recent` is below 100 for many alerts, the drift detection is statistically unreliable — adjust the cron to use larger windows.

5. **Cross-reference with [BlockRateSpike](#alert-blockratespike)**. If both alerts fire together, the drift is moving model decisions; if FeatureDrift fires alone, the drift is in upstream features but the model is still making the same decisions (less urgent).

### Mitigation

1. **If the drift is benign** (seasonal / marketing-driven), document in the next sprint's audit report and move on.

2. **If the drift is real and persistent** (>1 day, multiple features), trigger a retraining run — see [How to trigger retraining](#how-to-trigger-retraining).

3. **If the drift is small-sample noise**, raise the `psi_alert_threshold` in `.env` (e.g., `PSI_ALERT_THRESHOLD=0.25`) and increase the `recent_window` size in the cron.

4. **If the drift correlates with an upstream data-pipeline change**, fix the upstream pipeline first and re-baseline before retraining the model.

---

## Alert: ShadowDisagreement {#alert-shadowdisagreement}

**Rule:** `(sum(rate(fraud_engine_shadow_disagreement_total[15m])) / sum(rate(fraud_engine_shadow_total{event="scored"}[15m]))) > 0.1`
**Threshold:** disagreement rate above 10% over a 15-minute rolling window
**Trigger window:** sustained for 10 m
**Severity:** warning

### What it means

When `Settings.shadow_enabled = True`, every `/predict` fires a fire-and-forget shadow scoring against the FraudNet challenger (Sprint 5.2.b). The `agree_decision` field on each `shadow.scored` log line tracks whether champion + challenger agreed at the decision boundary. This alert fires when sustained disagreement exceeds 10% — meaning the two models are diverging in production.

### Common causes

- Genuine model divergence (champion and challenger optimised different objectives / on different data).
- Champion model degraded due to drift; challenger still well-calibrated.
- Challenger model degraded; champion fine.
- `decision_threshold` mismatch — shadow uses the same threshold as champion (`Settings.decision_threshold`); if the threshold isn't right for the challenger's score distribution, disagreement is artefactual.
- Recent shadow model swap brought in a different-calibration challenger.

### Investigation

1. **Open dashboard panel 7** (shadow + drift activity) — confirm the disagreement-rate time series.

2. **Pull recent disagreement events from the structlog stream**:
   ```bash
   # Once docker stack is up:
   docker compose -f docker-compose.dev.yml logs fraud-api --tail 5000 | \
     jq -c 'select(.event == "shadow.scored" and .agree_decision == false) |
       {request_id, champion_score, shadow_score, champion_decision, shadow_decision}' | head -50
   ```

3. **Look for a pattern in the disagreement direction**:
   - Champion blocks, shadow allows → champion is more aggressive; check for champion overfit.
   - Champion allows, shadow blocks → shadow is more aggressive; check for shadow staleness or calibration mismatch.

4. **Compare model versions**:
   ```bash
   # Champion model_version surfaces in /predict response:
   curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
     -d @tests/fixtures/sample_txn.json | jq .model_version
   # Shadow model_version is in shadow.scored events:
   docker compose -f docker-compose.dev.yml logs fraud-api --tail 100 | \
     jq -c 'select(.event == "shadow.scored") | .shadow_model_version' | head -1
   ```

5. **Run the offline shadow comparison report** (Sprint 5.2.c) to compute champion-vs-challenger AUC + economic cost on labelled data:
   ```bash
   uv run python scripts/shadow_compare_report.py --sample
   ls -la reports/shadow_compare_*.md | tail -1
   ```

### Mitigation

1. **If the comparison report shows the challenger has materially better economic cost**, plan a champion-promotion exercise (out of scope for this runbook — see Sprint 5.2.c report for the promotion criteria).

2. **If the comparison shows the champion is fine and the challenger is degraded**, swap the shadow artefact for a known-good challenger. Procedure mirrors [How to safely roll back a model](#how-to-safely-roll-back-a-model) but for `models/sprint3/fraudnet/` instead of the LightGBM artefacts.

3. **If both models are fine but disagreement is high**, the threshold may be miscalibrated for the challenger. Raise the issue in the next model-review meeting; meanwhile, set `Settings.shadow_enabled = False` to disable shadow if the disagreement-driven log volume is a problem.

4. **If the challenger is failing more than scoring** (look at `fraud_engine_shadow_total{event="failed"}` rate vs `event="scored"` rate), the breaker may have tripped — check `fraud_engine_shadow_breaker_state{state="open"}`. Fix the underlying shadow failure (logs from `shadow.failed` events).

---

## Alert: ApiErrorRate {#alert-apierrorrate}

**Rule:** `(sum(rate(http_requests_total{status="5xx"}[5m])) / sum(rate(http_requests_total[5m]))) > 0.01`
**Threshold:** 5xx error rate above 1% over a 5-minute window
**Trigger window:** sustained for 5 m
**Severity:** warning

### What it means

More than 1% of HTTP responses over the last 5 minutes returned a 5xx status (server error). The `prometheus-fastapi-instrumentator` library buckets status codes into string labels (`"2xx"` / `"3xx"` / `"4xx"` / `"5xx"`), so this query uses a literal `status="5xx"` match rather than a regex.

5xx errors are uniformly bad — they indicate either an unhandled exception inside the request path OR an upstream dependency returning an error the API doesn't gracefully degrade from.

### Common causes

- Unhandled exception in a recent deploy (a code path that wasn't covered by the integration tests).
- Postgres pool exhausted under unusual concurrency.
- Redis connection refused (the FeatureService catches and degrades; if the catch is incomplete, 5xx surfaces).
- Pydantic validation error on a request the schema should reject as 4xx but the route raises a 500 instead.
- Asyncpg connection-pool deadlock.

### Investigation

1. **Look at uvicorn error logs** for the recent traceback:
   ```bash
   # Once docker stack is up:
   docker compose -f docker-compose.dev.yml logs fraud-api --tail 500 | \
     jq -c 'select(.level == "error" or .level == "critical")' | head -10
   ```

2. **Check the per-handler 5xx breakdown** (which route is failing):
   ```bash
   curl -s http://localhost:8000/metrics | grep '^http_requests_total{.*status="5xx"'
   ```

3. **Verify dependency health**:
   ```bash
   curl -s http://localhost:8000/ready | jq .
   ```
   If `/ready` returns 503 for a specific component, fix that first.

4. **Drive a single `/predict` to isolate the failure**:
   ```bash
   curl -v -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d @tests/fixtures/sample_txn.json
   ```
   The verbose response carries the full error.

### Mitigation

1. **If the cause is a recent deploy**, [roll back to the previous model](#how-to-safely-roll-back-a-model). For code-only deploys, redeploy the prior container image.

2. **If the cause is Postgres pool exhaustion**, increase pool size in `.env` (`PG_POOL_SIZE`) and restart the API.

3. **If the cause is an unhandled exception in a specific code path**, file an incident ticket with the traceback, then deploy a hotfix.

4. **If the cause is a validation-error-as-500 bug**, the schemas in `src/fraud_engine/api/schemas.py` should be tightened; that's a code fix, not a runtime mitigation.

---

## How to safely roll back a model

The `InferenceService` reads its artefacts at lifespan-startup (`src/fraud_engine/api/main.py` line 313). There's currently **no hot-reload admin endpoint** — rollback is file-swap + container restart. (A future Sprint 6.x can add `POST /admin/reload-model` that calls the existing programmatic `InferenceService.reload()` method.)

**Three artefacts must be swapped together** (any one alone leaves the model in an inconsistent state):

| Artefact | Path | Why it matters |
|---|---|---|
| Model joblib | `models/sprint3/lightgbm_model.joblib` | The LightGBM Booster + isotonic calibrator pickled together. |
| Calibrator joblib | `models/sprint3/calibrator.joblib` | Standalone calibrator; loaded separately by `InferenceService`. |
| Manifest | `models/sprint3/lightgbm_model_manifest.json` | Carries the `content_hash` (returned in `PredictionResponse.model_version`), `feature_names` (the 743-element list FeatureService aligns to), `best_score`, `best_iteration`. |

### Procedure

1. **Identify the rollback target.** Each prior model's full bundle should live in your artefact registry (or `models/sprint3/archive/{content_hash}/`). Verify it has all three files.

2. **Stop accepting new traffic** (load-balancer drain or compose stop):
   ```bash
   # Once docker stack is up:
   docker compose -f docker-compose.dev.yml stop fraud-api
   ```

3. **Swap the three artefacts atomically** — copy the rollback bundle into place:
   ```bash
   cp models/sprint3/archive/{rollback_content_hash}/lightgbm_model.joblib models/sprint3/
   cp models/sprint3/archive/{rollback_content_hash}/calibrator.joblib models/sprint3/
   cp models/sprint3/archive/{rollback_content_hash}/lightgbm_model_manifest.json models/sprint3/
   ```

4. **Restart the API** so the lifespan re-loads from disk:
   ```bash
   docker compose -f docker-compose.dev.yml start fraud-api
   ```

5. **Verify the served model_version matches the rollback target**:
   ```bash
   curl -s -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d @tests/fixtures/sample_txn.json | jq .model_version
   # Expected: the rollback content_hash from the manifest
   ```

6. **Verify Prometheus picks up the version change**:
   ```bash
   curl -s http://localhost:8000/metrics | grep '^fraud_engine_model_info'
   # The label `model_version="<sha256>"` should match the rolled-back manifest's content_hash
   ```

7. **Document the rollback** in a brief incident note: which alert prompted it, which model_version went out, which model_version came back in, expected duration of the rollback.

### Anti-patterns

- **Don't swap just the joblib without the manifest** — `model_version` would lie about which artefact is loaded.
- **Don't restart without the file swap** — the lifespan re-loads the SAME current artefact.
- **Don't `kill -9` the API** — the lifespan's shutdown drains pending shadow tasks (Sprint 5.2.b); `kill -9` skips that, which can corrupt the in-flight shadow scoring queue. Use `docker compose stop` (which sends SIGTERM with a 10 s grace period) or `pkill -TERM`.

---

## How to trigger retraining

**Note:** `make train` is a no-op stub that prints "implemented in Sprint 3" and exits 1 (`Makefile:62-63`). The retraining flow is invoked via the explicit `uv run python scripts/...` form below.

The end-to-end pipeline is **data refresh → feature pipeline → train → evaluate → deploy**. Each step writes its outputs to known paths the next step reads from.

### Step 1 — Refresh raw data

If new IEEE-CIS-style data has landed (or a Kaggle re-release):
```bash
uv run python scripts/download_data.py
# Writes data/raw/{train,test}_*.csv per the Sprint 1 loader contract.
```

### Step 2 — Run the feature pipeline through Tier-5

```bash
uv run python scripts/build_features_all_tiers.py
# Writes data/processed/tier{1..5}_{train,val,test}.parquet
# Wall-clock: ~15-25 minutes on a laptop for the full IEEE-CIS dataset.
```

### Step 3 — Retrain the LightGBM champion (Sprint 3.3.b)

```bash
# Full retrain with 100-trial Optuna sweep (wall-clock: ~2-4 hours):
uv run python scripts/train_lightgbm.py

# Faster reuse of prior tuning (wall-clock: ~5-10 minutes):
uv run python scripts/train_lightgbm.py --skip-tuning

# Quick smoke (wall-clock: <1 minute):
uv run python scripts/train_lightgbm.py --quick
```

This writes:
- `models/sprint3/lightgbm_model.joblib`
- `models/sprint3/calibrator.joblib`
- `models/sprint3/lightgbm_model_manifest.json` (with new `content_hash`)
- `reports/training_report_lightgbm.md` (Optuna summary + validation metrics)

### Step 4 — (Optional) Retrain the diversity models

```bash
# FraudNet (Sprint 3.2.a) — feeds the shadow channel:
uv run python scripts/train_neural.py

# FraudGNN (Sprint 3.3.d) — batch-only; outputs feed Tier-5:
uv run python scripts/train_gnn.py
```

### Step 5 — Re-run economic evaluation (Sprint 4.4)

```bash
uv run python scripts/run_economic_evaluation.py
# Writes reports/economic_evaluation.md with the new optimal threshold.
```

If the new optimal threshold differs materially from the current `Settings.decision_threshold`, update `.env`:
```bash
sed -i 's/^DECISION_THRESHOLD=.*/DECISION_THRESHOLD=0.085/' .env
```

### Step 6 — Update the drift baseline + performance baselines

```bash
# Drift baseline (Sprint 6.1.b) — re-run after every model retrain:
uv run python scripts/build_drift_baseline.py
# Writes data/baselines/distributions.parquet

# Performance baselines (Sprint 6.1.c) — operator-curated; update Settings:
# Read the new validation AUC from the manifest:
jq .best_score models/sprint3/lightgbm_model_manifest.json
# Then update .env:
sed -i 's/^PERFORMANCE_TRAINING_AUC=.*/PERFORMANCE_TRAINING_AUC=0.835/' .env
# (AUC-PR + cost baselines are placeholder until a future Sprint 6.x retrofit
# adds them to the manifest — see Sprint 6.1.c report.)
```

### Step 7 — Deploy the new model

Two paths:

- **Blue/green:** spin up a second API container with the new artefacts on a different port, drive synthetic traffic, compare metrics, then cut over via the load balancer.
- **In-place:** stop the API, files are already in place from step 3, restart the API. (Same procedure as the rollback above, just without the `archive/` source.)

### Step 8 — Validate

```bash
# Drive the full integration test suite:
uv run pytest tests/integration -v

# Confirm the new model_version is being served:
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d @tests/fixtures/sample_txn.json | jq .model_version
```

Compare to `models/sprint3/lightgbm_model_manifest.json:content_hash`.

---

## Appendix A — log file locations

| Source | Path | Format | Sprint |
|---|---|---|---|
| DriftMonitor alerts | `logs/drift/{run_id}/drift_alerts.jsonl` | JSONL | 6.1.b |
| PerformanceMonitor alerts | `logs/performance/{run_id}/performance_alerts.jsonl` | JSONL | 6.1.c |
| Pipeline lineage | `logs/lineage/{run_id}/lineage.jsonl` | JSONL | 1 |
| Per-pipeline text logs | `logs/{pipeline_name}/{run_id}.log` | text | 0+ |
| PredictionLogger | Postgres `predictions` table | SQL | 5.2.a |
| Structlog (live API) | uvicorn stdout (JSON per line) | JSON | 5.1.f |

In a docker-compose stack, the structlog stream is captured by Docker:
```bash
# Once docker stack is up:
docker compose -f docker-compose.dev.yml logs fraud-api --tail 200 | jq -c .
```

---

## Appendix B — `jq` recipes

### List recent prediction-logger writes from the structlog stream
```bash
docker compose -f docker-compose.dev.yml logs fraud-api --tail 5000 | \
  jq -c 'select(.event == "prediction.logged")'
```

### Top-10 drifted features across all recent runs
```bash
cat logs/drift/*/drift_alerts.jsonl | jq -s '
  group_by(.feature_name) |
  map({feature: .[0].feature_name, max_psi: (map(.psi) | max)}) |
  sort_by(-.max_psi) | .[:10]'
```

### Find shadow disagreements with high score divergence
```bash
docker compose -f docker-compose.dev.yml logs fraud-api --tail 10000 | \
  jq -c 'select(.event == "shadow.scored" and .agree_decision == false) |
    select((.champion_score - .shadow_score) | fabs > 0.3)'
```

### Performance-monitor alerts grouped by metric
```bash
cat logs/performance/*/performance_alerts.jsonl | jq -s '
  group_by(.metric) | map({metric: .[0].metric, n_alerts: length, max_degradation: (map(.degradation) | max)})'
```

### Predictions audit log — block-rate over the last hour by hour
```bash
docker compose -f docker-compose.dev.yml exec postgres psql -U fraud -d fraud -c "
  SELECT date_trunc('minute', created_at) AS minute,
         decision,
         count(*) as n
  FROM predictions
  WHERE created_at > NOW() - INTERVAL '1 hour'
  GROUP BY minute, decision
  ORDER BY minute DESC;"
```

---

## Appendix C — useful PromQL queries

Beyond the 5 named alert rules, these queries answer common operator questions.

### What's the median feature_fetch latency right now?
```promql
histogram_quantile(0.5, sum by (le) (rate(fraud_engine_feature_fetch_seconds_bucket[5m])))
```

### What's the per-handler request rate?
```promql
sum by (handler) (rate(http_requests_total[5m]))
```

### Has the model been swapped recently?
```promql
count by (model_version) (fraud_engine_model_info)
# Returns one series per model_version that's been seen since process start.
```

### What fraction of the last hour was in degraded mode?
```promql
sum(increase(fraud_engine_degraded_mode_total[1h])) /
sum(increase(fraud_engine_predictions_total[1h]))
```

### Is the shadow circuit breaker open?
```promql
fraud_engine_shadow_breaker_state{state="open"} == 1
```

### Drift alerts per feature (if a future sprint retrofits the per-feature label)
```promql
# Currently the counter is unlabelled; a future label would enable:
sum by (feature_name) (increase(fraud_engine_drift_alerts_total[1d]))
```
