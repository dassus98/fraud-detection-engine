# Sprint 6 — Prompt 6.1.e: Monitoring integration test + operations runbook

## Summary

Sprint 6.1.a-d built the **monitoring stack** (data plane: 13 Prometheus metrics + 2 JSONL alert streams + DriftMonitor + PerformanceMonitor; control plane: 7-panel Grafana dashboard + 5 named alert rules + Prometheus scrape config + 30-day retention). Sprint 6.1.e closes the loop with two operationalising artefacts:

1. **`tests/integration/test_monitoring_e2e.py`** — 5 end-to-end integration tests: 2 prove that `/predict` traffic is reflected in the live `/metrics` scrape's counter / histogram values (parsed via `prometheus_client.parser.text_string_to_metric_families`); 2 prove that synthetic drift triggered via `DriftMonitor.check_and_alert` increments `fraud_engine_drift_alerts_total` and that the `FeatureDrift` alert rule's threshold WOULD fire on the post-drift counter state (static evaluation since no live Prometheus is available); 1 enforces alert/dashboard ↔ emitted-metric agreement to catch dangling-reference regressions in future PRs.

2. **`docs/RUNBOOK.md`** — operations runbook with ordered remediation steps for each of the 5 alert rules, plus two cross-cutting procedures: "How to safely roll back a model" and "How to trigger retraining". Sprint 6.1.d's alert rules pre-baked `runbook_url` annotations pointing at this file (`docs/RUNBOOK.md#alert-${name}`) — this PR makes those anchors real.

**Risk: Medium → realised Low.** All 5 tests pass cleanly; no source code changes; the runbook anchor-validation script confirms all 5 `runbook_url` annotations resolve to real headings.

## Files changed

| Path | Change | LOC |
|---|---|---|
| `tests/integration/test_monitoring_e2e.py` | NEW — 5 end-to-end integration tests | +443 |
| `docs/RUNBOOK.md` | NEW — operations runbook | +536 |
| `sprints/sprint_6/prompt_6_1_e_report.md` | NEW — this report | +(this file) |

**No changes** to source code, configs, schemas, settings, compose files, Makefile, Dockerfile, `CLAUDE.md`, or any prior monitoring module / report.

## The 5 integration tests

| # | Category | Test | What it asserts |
|---|---|---|---|
| 1 | Predict-volume scrape | `test_predict_volume_reflected_in_metrics_counters` | Drive 5 `/predict`; assert `fraud_engine_predictions_total{decision}` delta sums to 5 across labels. |
| 2 | Predict-volume scrape | `test_predict_latency_histograms_observe_each_request` | Drive 5 `/predict`; assert `fraud_engine_predict_total_seconds_count` delta == 5; +Inf bucket delta ≥ 5. |
| 3 | Synthetic-drift scrape | `test_synthetic_drift_increments_drift_alerts_counter` | Build synthetic baseline + +1.5σ-shifted recent window; run `DriftMonitor.check_and_alert`; assert `fraud_engine_drift_alerts_total` delta ≥ 1. |
| 4 | Synthetic-drift scrape | `test_drift_alert_rule_threshold_would_fire_after_synthetic_drift` | Same drift trigger; load `alert_rules.yml`'s `FeatureDrift` rule; regex out `> 0` threshold; assert post-drift counter delta exceeds it. |
| 5 | Config-correctness | `test_dashboard_and_alerts_reference_emitted_metrics` | Walk dashboard JSON `panels[*].targets[*].expr` + alert rule YAML `expr`; regex out `fraud_engine_*` names; assert each appears in `/metrics` registered families. |

All 5 use the existing `deps_reachable` + `client` + `sample_request_payload` fixtures from `test_api_e2e.py`'s pattern (mirrored inline since `tests/integration/conftest.py` doesn't exist). Each test scrapes `/metrics` over HTTP — exercising the actual `prometheus-fastapi-instrumentator` wiring, not just the in-process REGISTRY.

## The runbook structure

```
# Operations Runbook — Fraud Detection Engine
## Monitoring at a glance                       [links to 6.1.a/b/c/d/e]
## How to use this runbook                      [anchor convention; severity SLOs]

## Alert: HighLatency             {#alert-highlatency}
## Alert: BlockRateSpike          {#alert-blockratespike}
## Alert: FeatureDrift            {#alert-featuredrift}
## Alert: ShadowDisagreement      {#alert-shadowdisagreement}
## Alert: ApiErrorRate            {#alert-apierrorrate}

## How to safely roll back a model
## How to trigger retraining

## Appendix A — log file locations
## Appendix B — `jq` recipes
## Appendix C — useful PromQL queries
```

Each per-alert section follows the same shape: **What it means → Common causes → Investigation (numbered with copy-pasteable commands) → Mitigation (least- to most-disruptive).** Investigation steps reference the dashboard panels by number, the `psql` queries, the `jq` recipes, and the `docker compose logs` commands the operator needs.

The model-rollback procedure documents the file-swap + container-restart approach (no `POST /admin/reload-model` admin endpoint exists yet — flagged as Sprint 6.x). All three artefacts (`lightgbm_model.joblib`, `calibrator.joblib`, `lightgbm_model_manifest.json`) must be swapped together; verification via `curl /predict | jq .model_version` against the rolled-back manifest's `content_hash`.

The retraining procedure is the explicit `uv run python scripts/...` form (`make train` is currently a no-op stub per `Makefile:62-63`). Eight-step end-to-end flow: data refresh → feature pipeline → train LightGBM → train diversity models → re-run economic evaluation → update drift baseline + performance baselines → deploy → validate.

## Design decisions (7)

### Decision 1 — 5 tests across 3 categories

Two on predict-volume scrape, two on synthetic-drift scrape, one on config-correctness. The config-correctness test (test 5) is the highest-leverage addition — it catches the dashboard-references-deleted-metric regression class that no other test covers. **Rejected:** a 6th test that spins up a local `prometheus` binary to evaluate alert rules literally (requires `promtool` on PATH; not available; static threshold extraction in test 4 covers the same intent for the FeatureDrift rule).

### Decision 2 — Static evaluation of alert thresholds (test 4)

Test 4 extracts the literal threshold from the alert rule's `expr` via a narrow regex (`r">\s*([\d.]+)\s*$"`). For the `FeatureDrift` rule (`increase(...[1h]) > 0`), this gives `threshold = 0.0`; the test asserts the post-drift counter delta exceeds it. Couples the assertion to `alert_rules.yml` so a future threshold tweak automatically updates the test. **Rejected:** a full PromQL parser/evaluator (massive dependency); hardcoding the `> 0` literal (couples test brittleness to the rule file).

### Decision 3 — Test 5 walks dashboard panels recursively + suffix-tolerant matching

Grafana JSON nests `targets[*].expr` under `panels[*]`, with row-grouped layouts adding a second nesting level. Test 5 walks both. For metric-name matching: builds the registered family-root set from `text_string_to_metric_families(...)`'s `family.name` attribute (the root, not sample names) — this is the right source-of-truth because Counters with no observed labels (e.g., `SHADOW_TOTAL` when shadow is disabled by default) emit a HELP/TYPE block but no samples; only `family.name` exposes them.

### Decision 4 — DriftMonitor isolated to `tmp_path`

Tests 3 + 4 invoke `DriftMonitor.check_and_alert` which writes JSONL to `logs/drift/{run_id}/...`. We pass `alert_log_dir=tmp_path/"drift"` so writes never touch the real `logs/` directory.

### Decision 5 — Mirror `test_api_e2e.py`'s `deps_reachable` + `client` fixtures inline

`tests/integration/conftest.py` doesn't exist; integration tests own their fixtures inline. Mirror the existing pattern verbatim — same Redis + Postgres probe, same per-test-scoped lifespan-managed client. **Rejected:** mocking Redis + Postgres at the FeatureService boundary (requires a separate code path; the live integration is the value-add).

### Decision 6 — Counter delta pattern (mirrors 6.1.a-d test convention)

Each test uses the **delta pattern** (capture pre-value, do work, capture post-value, assert difference). Order-independent against the global Prometheus REGISTRY singleton — same convention as `test_prometheus_metrics.py`, `test_drift.py::TestDriftAlertsTotalCounter`, `test_shadow.py::test_shadow_disagreement_counter_*`.

### Decision 7 — Runbook adopts `OBSERVABILITY.md` style

`docs/OBSERVABILITY.md` is the operator-focused exemplar in the codebase: numbered sections, `jq` recipes, code blocks for every command, real file paths, structured-logging emphasis. RUNBOOK.md mirrors that. Each per-alert section's heading carries an explicit `{#alert-name}` anchor matching the `runbook_url` annotation in `alert_rules.yml`. **Rejected:** ADR-style format; pure prose; auto-generated runbook from the alert YAML's annotations (annotations are 1-line summaries; the runbook needs paragraphs).

## Verification

### New integration tests — 5/5 PASS

```text
tests/integration/test_monitoring_e2e.py::test_predict_volume_reflected_in_metrics_counters PASSED [ 20%]
tests/integration/test_monitoring_e2e.py::test_predict_latency_histograms_observe_each_request PASSED [ 40%]
tests/integration/test_monitoring_e2e.py::test_synthetic_drift_increments_drift_alerts_counter PASSED [ 60%]
tests/integration/test_monitoring_e2e.py::test_drift_alert_rule_threshold_would_fire_after_synthetic_drift PASSED [ 80%]
tests/integration/test_monitoring_e2e.py::test_dashboard_and_alerts_reference_emitted_metrics PASSED [100%]
```

### Cheap gates — all PASS

```text
$ make format       → 1 file reformatted, 140 files left unchanged
$ make lint         → All checks passed!
$ make typecheck    → Success: no issues found in 53 source files
```

### Integration regression — 16/16 PASS (test_api_e2e + test_shadow)

```text
$ uv run pytest tests/integration/test_api_e2e.py tests/integration/test_shadow.py --no-cov
===================== 16 passed, 39767 warnings in 18.08s ======================
```

### Anchor verification — 5/5 runbook URLs resolve

```text
Referenced anchors (alert → expected runbook anchor):
  HighLatency               -> #alert-highlatency              [OK]
  BlockRateSpike            -> #alert-blockratespike           [OK]
  FeatureDrift              -> #alert-featuredrift             [OK]
  ShadowDisagreement        -> #alert-shadowdisagreement       [OK]
  ApiErrorRate              -> #alert-apierrorrate             [OK]

OK — all 5 runbook_url anchors resolve to real RUNBOOK.md headings.
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

## Sample alert section (excerpt from RUNBOOK.md)

````markdown
## Alert: HighLatency {#alert-highlatency}

**Rule:** `histogram_quantile(0.95, sum by (le) (rate(fraud_engine_predict_total_seconds_bucket[5m]))) > 0.1`
**Threshold:** p95 above 100 ms (CLAUDE.md §3 SLO budget)
**Trigger window:** sustained for 2 m
**Severity:** warning

### What it means
[paragraph about the SLO + smoothing window]

### Common causes
[5 bulleted causes]

### Investigation
1. Open dashboard panel 1 (API latency p50/p95/p99)...
2. Open dashboard panel 2 (per-stage latency p95)...
3. Check dependency health (panel 6)...
4. Check Redis latency directly: `docker compose ... redis-cli --latency`
5. Verify loaded model version: `curl ... | jq .model_version`
6. Inspect uvicorn logs: `docker compose ... logs fraud-api --tail 200 | jq -c ...`

### Mitigation
1. If transient blip → no action; auto-clears.
2. If recent deploy regression → roll back model.
3. ...
````

## Deviations from plan

1. **Test 5 originally used sample-name → root suffix-stripping; switched to using `family.name` directly** because Counters with no observed labels emit no samples (so SHADOW_TOTAL when shadow is disabled left no `fraud_engine_shadow*` samples in the parsed scrape). Caught on the first test run; the fix uses `text_string_to_metric_families(...).name` which IS the family root regardless of observation state. No plan-level decision changed.

2. **No `make train` target wrapping was added.** The plan noted the existing stub (`Makefile:62-63`) and documented the explicit `uv run python scripts/...` form in the runbook. A future Sprint 6.x can wrap the explicit invocations.

3. **No automatic anchor-validation pre-commit hook was added.** The manual one-shot script in the verification section confirms all 5 anchors resolve. A future Sprint 6.x can promote it to a hook once the alert + runbook surface stabilises.

## Cross-references

- `tests/integration/test_api_e2e.py:49,89-141,172-186` — pattern this PR mirrors (`LifespanManager` + `ASGITransport` + `deps_reachable` + the existing `/metrics` substring-grep test).
- `tests/integration/test_shadow.py` — sibling pattern.
- `src/fraud_engine/monitoring/drift.py:DriftBaselineBuilder + DriftMonitor` — the synthetic-drift trigger path.
- `src/fraud_engine/monitoring/prometheus_metrics.py` — 13 metrics + 2 retrofit Counters.
- `configs/alerts/alert_rules.yml` — 5 rules whose `runbook_url` annotations this PR makes real.
- `configs/grafana/fraud_dashboard.json` — 7 panels test 5 walks.
- `docs/OBSERVABILITY.md` — runbook style template.
- `models/sprint3/lightgbm_model_manifest.json:content_hash` — the rollback verification target.
- `scripts/train_lightgbm.py + train_neural.py + train_gnn.py` — retraining script names.
- `CLAUDE.md` §3 (latency budget), §14 (RUNBOOK.md was named here as a Sprint 6 deliverable).

## Out of scope (Sprint 6.x+)

- **`POST /admin/reload-model` admin endpoint** — would let the runbook's rollback procedure skip the container restart.
- **`make train` Makefile target** — currently a no-op stub. Wrapping the explicit script invocations.
- **AlertManager integration** — the runbook documents the alerts; they don't route anywhere yet.
- **Automatic anchor-validation pre-commit hook** — would catch `runbook_url` ↔ heading drift at commit time (rather than the integration-test runtime).
- **Markdown linting** — project's pre-commit doesn't include `markdownlint-cli`; manually proof-read for now.
- **Live-Prometheus integration test** — would require `promtool` on PATH or a Prometheus binary. The static threshold-evaluation in test 4 covers the same intent for the FeatureDrift rule.
- **Per-stratum runbook sections** (e.g., per-`ProductCD`) — once the stratified dashboard panels exist.
- **CLAUDE.md §13 sprint-status update** — Sprint 6 row gets updated by a 6.2.x audit-and-gap-fill PR per established convention.
