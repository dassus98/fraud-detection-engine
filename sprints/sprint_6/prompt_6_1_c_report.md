# Sprint 6 — Prompt 6.1.c: PerformanceMonitor (rolling AUC / AUC-PR / cost vs training baseline)

## Summary

Sprint 6's monitoring tripod is now complete:

- **6.1.a** — *output drift* via the `fraud_engine_prediction_score` Prometheus
  histogram (live request-path signal).
- **6.1.b** — *input drift* via PSI on 743 features against a frozen training
  baseline (offline batch).
- **6.1.c** — *labelled-prediction performance regression* via rolling AUC,
  AUC-PR, and economic-cost vs the training-time baseline (offline batch).

Both 6.1.a and 6.1.b detect that *something has shifted* but neither answers
"is the model performing worse on the labels we care about?" PSI on a
feature distribution can shift for benign reasons (a marketing campaign
brings new geo mix); only labelled-prediction comparison confirms model
degradation. This PR closes that loop.

In production, the labelled-prediction stream is fed by chargeback feeds
with a 30–90 day lag. For this project's offline-portfolio context the
labels come from the IEEE-CIS test set we already have on disk; the
`PerformanceMonitor` itself is data-source-agnostic — caller hands it a
DataFrame, mirroring `DriftMonitor`'s contract from 6.1.b.

**Risk: Low–Medium.** Reuses the battle-tested `economic_cost` primitive
from `utils/metrics.py:68-161` (Sprint 4) for the cost math and
`sklearn.metrics.roc_auc_score / average_precision_score` for the
discriminative metrics (already imported in `evaluation/stratified.py:95`
— no new dependency). Net-new surface is the baseline-comparison logic,
the alert-file convention (`logs/performance/{run_id}/...`), and the
labelled-window contract. No request-path changes, no schema changes,
no API changes.

## Files changed

| Path | Change | LOC |
|---|---|---|
| `src/fraud_engine/monitoring/performance_monitor.py` | NEW — `PerformanceMonitor` + `_Degradation` frozen dataclass + module docstring covering 8 design decisions | +401 |
| `src/fraud_engine/monitoring/__init__.py` | MODIFIED — re-export `PerformanceMonitor` | +3 |
| `src/fraud_engine/config/settings.py` | MODIFIED — add 6 new monitoring fields (`performance_window_size`, `performance_degradation_threshold`, `performance_alert_log_dir`, `performance_training_auc`, `performance_training_auc_pr`, `performance_training_cost`); extend `_resolve_relative_to_project_root` field-validator with `performance_alert_log_dir` | +69 |
| `tests/unit/test_performance_monitor.py` | NEW — 9 tests across baseline-comparison / cost-direction / contract / edge-cases / alerting | +377 |
| `sprints/sprint_6/prompt_6_1_c_report.md` | NEW — this report | +(this file) |

**No changes** to schemas, FeatureService, inference, shap_explainer,
shadow, prediction_logger, circuit_breaker, prometheus_metrics (Sprint
6.1.a), drift (Sprint 6.1.b), API routes, Makefile, Dockerfile,
docker-compose.yml, `CLAUDE.md`, evaluation/, or utils/.

## Public surface

```python
class PerformanceMonitor:
    def __init__(self, settings: Settings | None = None) -> None: ...

    @log_call
    def compute_rolling_metrics(self, recent_window: pd.DataFrame) -> dict[str, float]:
        """Return {'auc', 'auc_pr', 'cost', 'n_recent'} on the given window.
        AUC + AUC-PR are NaN for single-class windows (no false alerts)."""

    @log_call
    def compare_to_baseline(self, metrics: dict[str, float]) -> list[_Degradation]:
        """One _Degradation per metric whose drop exceeds the threshold."""

    @log_call
    def check_and_alert(
        self,
        recent_window: pd.DataFrame,
        *,
        run_id: str | None = None,
        alert_log_dir: Path | None = None,
    ) -> int:
        """Compute → compare → JSONL per degraded metric. Returns count."""
```

### Recent-window column contract

```
Required: score (float ∈ [0, 1]), label (int ∈ {0, 1})
Optional: decision (int ∈ {0, 1})
    - if present: used directly for cost computation (mirrors the
      Sprint 5.2.a `predictions` table's persisted decision column)
    - if absent:  scores thresholded internally via Settings.decision_threshold
```

### Alert record schema

```json
{
  "timestamp": "2026-05-10T22:14:30.123456+00:00",
  "run_id": "f8c4...e3a1",
  "metric": "auc",
  "baseline": 0.85,
  "current": 0.78,
  "degradation": 0.0824,
  "threshold": 0.05,
  "n_recent": 1000
}
```

Joinable by `run_id` against `logs/lineage/{run_id}/lineage.jsonl`,
`logs/drift/{run_id}/drift_alerts.jsonl` (Sprint 6.1.b), and the Sprint
5.2.a predictions audit log.

## Design decisions (7 + 1 surfaced during impl)

### Decision 1 — Stateless analyzer (mirrors DriftMonitor)

`PerformanceMonitor` carries no per-prediction history. Caller slices
the most-recent N labelled predictions and hands in a DataFrame. "Rolling"
is satisfied by the caller-side window choice (`Settings.performance_window_size`
hints at the expected size, but `PerformanceMonitor` doesn't enforce it).

**Why:** consistency with 6.1.b; avoids a stateful global; simpler tests;
matches the offline-cron expected workflow. A future Sprint 6.x can add
a `StreamingPerformanceMonitor` once a real chargeback feed exists.

### Decision 2 — Recent-window DataFrame contract

Required `score` + `label`; optional `decision`. AUC + AUC-PR always come
from `score`; cost uses `decision` if present, else thresholds `score`
internally via `Settings.decision_threshold`. This handles both the
production case (caller forwards the persisted decision column) and
the offline-replay case (caller has only scores).

### Decision 3 — Three metrics: AUC, AUC-PR, economic cost

- **AUC (ROC):** discrimination quality; stable under class-balance shifts.
- **AUC-PR (`average_precision_score`):** the right metric for the 3.5%
  positive-rate fraud setting per CLAUDE.md §1; catches degradations
  that ROC AUC misses on imbalanced classes.
- **Economic cost (USD):** the business-meaning metric; AUC may stay
  flat while cost rises when score-distribution bunching changes near
  the decision threshold.

Three metrics, three independent alerts per `check_and_alert` call.

### Decision 4 — Fractional degradation `(baseline - current) / baseline > 0.05`

Portable across metrics with different scales. Cost flips the sign
internally (higher cost = worse) — `_compute_degradation(metric, baseline,
current)` handles the sign per metric name.

Worked examples:

| Metric | Baseline | Current | Degradation | Alerts (>5%)? |
|---|---|---|---|---|
| AUC | 0.85 | 0.78 | (0.85 - 0.78) / 0.85 = 8.24% | ✓ |
| AUC | 0.85 | 0.82 | (0.85 - 0.82) / 0.85 = 3.53% | ✗ |
| Cost | $42K | $50K | (50 - 42) / 42 = 19.05% | ✓ |
| Cost | $42K | $43K | (43 - 42) / 42 = 2.38% | ✗ |

### Decision 5 — Training baselines as Settings fields, operator-curated

Three new Settings fields seed the comparison:

- `performance_training_auc = 0.8281` — sourced from
  `models/sprint3/lightgbm_model_manifest.json:best_score` (the only
  baseline currently persisted machine-readably).
- `performance_training_auc_pr = 0.50` — placeholder; no AUC-PR
  artefact exists yet.
- `performance_training_cost = 0.0` — placeholder; a 0.0 cost baseline
  effectively disables cost alerts (the fractional-degradation
  comparison requires a positive baseline; `compare_to_baseline` skips
  metrics whose baseline is ≤ 0).

Operator updates these on each model retrain. A Sprint 6.x housekeeping
PR can extend the model manifest with AUC-PR + cost so the operator step
becomes "edit the manifest after `train.py` writes it".

**Rejected:** auto-load from manifest (manifest doesn't have AUC-PR or
cost yet — Sprint 3 retrofit is out of scope here); separate
`configs/training_baselines.yaml` (extra moving part; Settings is the
project's prescribed config home).

### Decision 6 — Append-only JSONL alerts at `logs/performance/{run_id}/...`

Mirrors 6.1.b's `logs/drift/{run_id}/drift_alerts.jsonl` exactly. Each
record is grep-able + joinable by `run_id`. If no metric degrades, the
file is NOT created (operators grep for the file's existence as a clean
alerted/not-alerted signal).

**Rejected:** Postgres `performance_alerts` table — slow-moving log
stream; JSONL ships with no schema migration.

### Decision 7 — Three-method API: compute → compare → alert

```python
metrics = monitor.compute_rolling_metrics(recent_window)   # raw numbers
degradations = monitor.compare_to_baseline(metrics)        # filtered list
n = monitor.check_and_alert(recent_window, ...)            # writes + counts
```

Each method does one thing; tests can exercise each layer independently.
`check_and_alert` is the ergonomic top-level entry point that does all
three under the hood.

### Decision 8 — single-class detection upfront (surfaced during testing)

`sklearn.metrics.roc_auc_score` raises `ValueError` on a single-class
window, but `sklearn.metrics.average_precision_score` silently returns
`0.0` (with only a UserWarning). Naively wrapping just the AUC call in
try/except let AUC-PR slip through as `0.0` — which would then trigger a
false "100% degradation" alert.

Solution: detect single-class windows up-front (`np.unique(y_true)`) and
return NaN for both metrics together. Cost is still computable (just no
true positives). The `compare_to_baseline` skip-NaN logic stays clean.

This deviation from the original plan added ~12 LOC and one constant
(`_MIN_CLASSES_FOR_AUC = 2`); the 7-decision count in the plan held.

## Verification

### Unit tests — 9/9 PASS

```text
tests/unit/test_performance_monitor.py::TestBaselineComparison::test_alert_when_auc_degrades_above_threshold PASSED [ 11%]
tests/unit/test_performance_monitor.py::TestBaselineComparison::test_no_alert_when_auc_degrades_below_threshold PASSED [ 22%]
tests/unit/test_performance_monitor.py::TestBaselineComparison::test_each_metric_alerts_independently PASSED [ 33%]
tests/unit/test_performance_monitor.py::TestCostDirection::test_cost_degradation_uses_higher_is_worse_sign PASSED [ 44%]
tests/unit/test_performance_monitor.py::TestRecentWindowContract::test_compute_rolling_metrics_uses_decision_column_when_present PASSED [ 55%]
tests/unit/test_performance_monitor.py::TestRecentWindowContract::test_compute_rolling_metrics_thresholds_scores_when_decision_absent PASSED [ 66%]
tests/unit/test_performance_monitor.py::TestEdgeCases::test_returns_nan_metrics_when_window_is_single_class PASSED [ 77%]
tests/unit/test_performance_monitor.py::TestAlerting::test_check_and_alert_writes_jsonl_per_degraded_metric PASSED [ 88%]
tests/unit/test_performance_monitor.py::TestAlerting::test_check_and_alert_writes_nothing_when_no_degradation PASSED [100%]
======================== 9 passed, 14 warnings in 1.66s ========================
```

### Cheap gates

```text
$ make format       → 2 files reformatted, 138 files left unchanged
$ make lint         → All checks passed!
$ make typecheck    → Success: no issues found in 53 source files
```

### Full-suite regression

```text
$ uv run pytest tests/unit -q --no-cov
815 passed, 3282 warnings in 82.13s (0:01:22)
```

Pre-PR baseline was 805 (post 6.1.b); +9 new performance-monitor tests
= 814 expected. Observed 815 — the +1 is the `test_settings.py::TestDefaults::test_defaults_load_without_error`
flake (documented in Sprint 5.1.f) recovering after this PR adds new
Settings defaults that change the baseline initialisation path.

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

### Sample alert payload (synthetic-degradation test output)

```json
{
  "timestamp": "2026-05-10T23:08:14.555123+00:00",
  "run_id": "test-run-001",
  "metric": "auc_pr",
  "baseline": 0.31,
  "current": 0.155,
  "degradation": 0.50,
  "threshold": 0.05,
  "n_recent": 1000
}
```

## Deviations from plan

1. **Single-class detection moved to a guard upfront** (Decision 8 above).
   Plan assumed catching `roc_auc_score`'s `ValueError` was sufficient;
   `average_precision_score` doesn't raise — it returns 0.0 silently.
   Caught during the first test run; fix added 12 LOC + 1 constant.

2. **5 of 9 tests had to be retuned for the synthetic-AUC calibration.**
   First-pass test design used `score_separation=1.2-2.0` which produces
   AUC ≈ 0.95-0.996 — too clean to test below-baseline degradation, and
   the `* 1.5` baseline multipliers exceeded Pydantic's `le=1.0` cap.
   Recalibrated to `sep=0.5` (AUC ≈ 0.81) which leaves headroom for
   realistic baseline values. Updated test docstrings to document the
   actual numerical bands.

3. **Two `# noqa: PLR0913` justifications** on test helpers `_make_settings`
   and `_synth_window` — both have one knob per logical dimension and
   collapsing them into a dict would obscure test-side intent at every
   call site. Matches the project's prior `noqa` pattern (e.g.,
   `utils/metrics.py:68` `economic_cost`).

## Cross-references

- `src/fraud_engine/utils/metrics.py:68-161` — `economic_cost(y_true,
  y_pred, ...)` returns a dict; we extract `total_cost` only and discard
  the rest.
- `src/fraud_engine/evaluation/stratified.py:95,783-784` — sklearn
  `roc_auc_score` / `average_precision_score` import precedent.
- `src/fraud_engine/monitoring/drift.py` (Sprint 6.1.b) — sibling module
  whose patterns this one mirrors (stateless, caller-passes-DataFrame,
  JSONL alerts, frozen `_Internal` dataclass).
- `src/fraud_engine/monitoring/prometheus_metrics.py` (Sprint 6.1.a) —
  the live-metrics surface; operators see all three monitoring layers in
  concert.
- `models/sprint3/lightgbm_model_manifest.json:best_score` — AUC = 0.8281
  baseline source.
- `CLAUDE.md` §3 (PSI drift detection as a Sprint-6 endpoint;
  performance monitoring is the implicit third leg), §4 (`monitoring/`
  module home: "Drift, performance, Prometheus metrics"), §5.4 (no
  hardcoded thresholds), §5.5 (logging discipline).

## Out of scope (Sprint 6.x+)

- **Stateful `StreamingPerformanceMonitor`** that maintains an
  N-prediction sliding window across calls — useful when a real
  chargeback feed lands events incrementally; defer until that infra
  exists.
- **Modelling chargeback lag (30–90 day delay)** — current simulation
  assumes instant labels. A Sprint 6.x can add a `simulate_lag(days)`
  parameter for replay scenarios.
- **Auto-load training baselines from manifest** — would need
  `models/sprint3/lightgbm_model_manifest.json` retrofit to carry
  AUC-PR + economic cost. Sprint 6.x housekeeping.
- **Per-stratum performance** (per `ProductCD`, per `card4` issuer,
  etc.) — Sprint 4's stratified evaluator already has the primitive;
  a future prompt can wire it in for slice-aware alerting.
- **Prometheus Counter for performance alerts**
  (`fraud_engine_performance_alerts_total{metric}`) — useful for Grafana
  panels; bounded labels (3 metrics) so cardinality is fine. Defer to
  Sprint 6.x.
- **Calibration drift** (Brier score / ECE) — discrimination (AUC) and
  business value (cost) are the load-bearing two; calibration is a
  useful third dimension for Sprint 6.x.
- **CLAUDE.md §13 sprint-status update** — Sprint 6 row gets updated by
  a 6.2.x audit-and-gap-fill PR per established convention.
- **Wiring `PerformanceMonitor` into a scheduled cron / Airflow DAG** —
  operator concern outside the codebase's deployment scope.
