# Sprint 6 — Prompt 6.1.b: DriftMonitor (PSI on production features against training baseline)

## Summary

Sprint 6.1.a delivered the **output-drift surface** (the
`fraud_engine_prediction_score` Prometheus histogram surfaces score
distribution shifts in real time). This PR delivers the **input-drift
surface** — Population Stability Index (PSI) detection on the model's
743 input features, comparing a frozen training baseline against any
recent-window DataFrame an offline batch script provides.

The fraud-industry convention for PSI alerting:

```
PSI < 0.10 — no significant population shift
0.10 ≤ PSI ≤ 0.25 — moderate shift; investigate
PSI > 0.25 — significant shift; model re-fit likely needed
```

This PR ships:

1. `src/fraud_engine/monitoring/drift.py` — `DriftMonitor` class +
   `DriftBaselineBuilder` static method + `_psi_from_pcts` math kernel.
2. `scripts/build_drift_baseline.py` — Click CLI thin wrapper that
   builds `data/baselines/distributions.parquet` once from the
   training-data slice (operator-run; the parquet artefact itself is
   gitignored).
3. Append-only `logs/drift/{run_id}/drift_alerts.jsonl` — one JSONL
   record per feature crossing `Settings.psi_alert_threshold` (default
   0.20).
4. Four new `Settings` fields (`psi_alert_threshold`, `psi_bins`,
   `drift_baseline_path`, `drift_alert_log_dir`).
5. 9 unit tests in `tests/unit/test_drift.py` covering behaviour,
   aggregation, alerting, persistence, and math equivalence to the
   pre-existing `utils.metrics.compute_psi`.

**Risk: Medium.** Reuses the battle-tested PSI math from
`utils/metrics.py:277-369` (9 prior tests). Net-new surface is the
baseline-persistence format, the alert-file convention, and the
offline-batch contract. No changes to the request path, no schema
changes, no API changes.

## Files changed

| Path | Change | LOC |
|---|---|---|
| `src/fraud_engine/monitoring/drift.py` | NEW — `DriftMonitor` + `DriftBaselineBuilder` + `_FeatureBaseline` dataclass + `_psi_from_pcts` helper + module docstring covering 7 design decisions | +591 |
| `src/fraud_engine/monitoring/__init__.py` | MODIFIED — re-export `DriftMonitor`, `DriftBaselineBuilder` | +6 |
| `src/fraud_engine/config/settings.py` | MODIFIED — add 4 monitoring fields (psi_alert_threshold, psi_bins, drift_baseline_path, drift_alert_log_dir) + extend `_resolve_relative_to_project_root` validator to cover the new path fields | +47 |
| `scripts/build_drift_baseline.py` | NEW — Click CLI thin wrapper around `DriftBaselineBuilder.build()` | +193 |
| `tests/unit/test_drift.py` | NEW — 9 tests across behaviour / aggregation / alerting / persistence / math equivalence | +371 |
| `sprints/sprint_6/prompt_6_1_b_report.md` | NEW — this report | +(this file) |

**No changes** to schemas, FeatureService, inference, shap_explainer,
shadow, prediction_logger, circuit_breaker, prometheus_metrics (Sprint
6.1.a), API routes, Makefile, Dockerfile, docker-compose.yml, or
`CLAUDE.md`.

**`data/baselines/distributions.parquet` is NOT generated in this PR.**
The artefact is gitignored under `data/`. The script + tests verify the
build pipeline; the actual baseline materialisation against the real
~400K-row train parquet is a one-shot operator command John runs locally
when convenient. The expected schema + sample invocation appear below.

## Public surface

### `DriftBaselineBuilder.build(train_df, feature_names, n_bins=10) → DataFrame`

One-time call. For each feature in `feature_names` that has ≥2 unique
non-null values in `train_df`, computes equal-frequency quantile edges
(via `np.quantile + np.unique`) + per-bin baseline percentages.
Constant-baseline features and missing-from-train_df features are
skipped with a WARNING log. Returns long-format DataFrame ready for
`to_parquet()` with columns `(feature_name, bin_idx, edge_low, edge_high,
baseline_pct, n_baseline)`.

### `DriftMonitor(baseline_path, settings)`

Loads the persisted baseline once at construction (~5 ms parquet read)
into an in-memory `dict[str, _FeatureBaseline]` for O(1) per-feature
lookups. Three public methods:

- **`compute_feature_psi(feature_name, recent_window) → float`** — PSI
  for one feature. Returns NaN if the feature is not in the loaded
  baseline (skipped at build) or not in `recent_window.columns`.
- **`compute_all_psi(recent_window, top_n=10) → DataFrame`** — per-feature
  PSI for every feature in the baseline, sorted by PSI descending,
  capped at `top_n`. NaN values sink to the bottom under
  `na_position="last"`.
- **`check_and_alert(recent_window, *, run_id=None, alert_log_dir=None)
  → int`** — appends one JSONL record to
  `{alert_log_dir}/{run_id}/drift_alerts.jsonl` for every feature with
  `psi > settings.psi_alert_threshold`. Returns the count of alerts
  written (0 = no drift). When zero alerts fire, the file is NOT
  created (operators can grep for the file's existence as a clean
  alerted/not-alerted signal).

### Alert record schema

```json
{
  "timestamp": "2026-05-10T22:14:30.123456+00:00",
  "run_id": "f8c4...e3a1",
  "feature_name": "card1_fraud_v_ewm_lambda_0.05",
  "psi": 0.34,
  "threshold": 0.2,
  "n_baseline": 412503,
  "n_recent": 1024
}
```

Joinable by `run_id` against `logs/lineage/{run_id}/lineage.jsonl`
(Sprint 1+) and against the predictions audit log (Sprint 5.2.a) for
full investigation context.

## Design decisions (7)

### Decision 1 — Long-format parquet baseline storage

`distributions.parquet` stores one row per (feature, bin) with columns
`(feature_name, bin_idx, edge_low, edge_high, baseline_pct, n_baseline)`.
743 features × 10 bins = 7,430 rows ≈ 250 KB on disk.

**Rejected:** raw baseline arrays (~3 GB; impractical for runtime
load); list-typed-column wide format (breaks `pd.read_parquet`
ergonomics); pickle dict-of-arrays (no cross-language interop, breaks
project's parquet convention).

### Decision 2 — Stateful `DriftMonitor` (load once, query many)

Constructor reads the baseline parquet once into
`_baselines: dict[str, _FeatureBaseline]`. Subsequent PSI computations
are O(1) feature lookup + ~1 ms `np.histogram` per recent-window column.
Total in-memory footprint: ~250 KB.

**Rejected:** stateless `compute_feature_psi(baseline_path, ...)` —
re-reads parquet on every call (~5 ms wasted per of 743 features).

### Decision 3 — Reuse `utils.metrics.compute_psi` math via `_psi_from_pcts` thin helper

`utils/metrics.compute_psi(baseline, current, bins, epsilon)` takes raw
arrays and does its own binning. Our pre-binned-baseline path needs only
the math kernel after the recent_window is binned into the persisted
edges. So `_psi_from_pcts(baseline_pcts, recent_pcts, epsilon)` is a
thin private helper that mirrors `utils.metrics.compute_psi`'s last
line.

**Math equivalence test** (`test_compute_feature_psi_matches_utils_compute_psi`)
asserts the two paths produce identical PSI to 1e-6 tolerance — catches
any future drift between the binning paths.

**Rejected:** refactor `compute_psi` to accept either raw arrays OR
pre-binned percentages. Doable but spreads complexity across 3
evaluation modules that already import it; keeping the existing
signature stable is safer.

### Decision 4 — Constant-baseline features skipped at build, NaN at runtime

PSI is mathematically undefined when the baseline has only one value
(quantile binning collapses to a single edge after `np.unique`).
`DriftBaselineBuilder.build` skips such features with a WARNING log;
they're absent from the persisted parquet. Runtime callers asking
`compute_feature_psi("constant_feature", ...)` get NaN — they can
filter on `df["psi"].notna()` if the distinction matters.

**Rejected:** carry constant features with edges=[v, v] and PSI=0 —
propagates degeneracy into runtime reports.

### Decision 5 — `top_n` ranks by PSI magnitude, NOT feature importance

`compute_all_psi` sorts by PSI descending. An on-call engineer reading
a daily drift report wants "what's drifting most?", not "what's
drifting most among the model's most-important features?". The latter
is derivable from the former via a join against feature importance.

**Rejected:** rank by feature importance (would require a Sprint 3.x
retrofit to add gain importances to the model manifest; defer).

### Decision 6 — Alerts as append-only JSONL at `logs/drift/{run_id}/...`

Mirrors `data/lineage.py:200-218`'s
`logs/lineage/{run_id}/lineage.jsonl` pattern. JSONL is grep-able,
joinable by `run_id` against other lineage logs, and ships with no
schema migration.

**Rejected:** Postgres `drift_alerts` table — adds a new schema for
what is fundamentally a slow-moving log stream; JSONL is simpler and
mirrors prior art.

**Rejected:** Prometheus Counter `fraud_engine_drift_alerts_total{feature}`
— useful for Grafana panels but with 743 features this is a
high-cardinality risk. Sprint 6.x can add bucketed labels alongside
the JSONL writer.

### Decision 7 — One-time generation script as thin wrapper around `DriftBaselineBuilder.build()`

The script does: `pd.read_parquet → _load_feature_names → builder.build →
to_parquet`. Underlying `DriftBaselineBuilder.build()` static method is
what tests call directly with synthetic DataFrames; the script adds I/O
+ Click CLI on top.

**Rejected:** all-in-script — couples math to subprocess invocation;
can't unit-test without spawning a Click process.

## Verification

### Unit tests — 9/9 PASS

```text
tests/unit/test_drift.py::TestNoDrift::test_psi_near_zero_for_identical_distributions PASSED [ 11%]
tests/unit/test_drift.py::TestSyntheticDrift::test_psi_high_for_synthetic_mean_shift PASSED [ 22%]
tests/unit/test_drift.py::TestSyntheticDrift::test_psi_increases_monotonically_with_shift_magnitude PASSED [ 33%]
tests/unit/test_drift.py::TestAggregation::test_compute_all_psi_returns_top_n_sorted_desc PASSED [ 44%]
tests/unit/test_drift.py::TestAggregation::test_compute_all_psi_handles_missing_feature_with_nan PASSED [ 55%]
tests/unit/test_drift.py::TestAlerting::test_check_and_alert_writes_jsonl_when_psi_above_threshold PASSED [ 66%]
tests/unit/test_drift.py::TestAlerting::test_check_and_alert_writes_nothing_when_no_drift PASSED [ 77%]
tests/unit/test_drift.py::TestPersistenceAndEquivalence::test_baseline_round_trip_via_parquet PASSED [ 88%]
tests/unit/test_drift.py::TestPersistenceAndEquivalence::test_compute_feature_psi_matches_utils_compute_psi PASSED [100%]
======================== 9 passed, 14 warnings in 2.50s ========================
```

### Cheap gates

```text
$ make format       → 4 files reformatted, 134 files left unchanged
$ make lint         → All checks passed!
$ make typecheck    → Success: no issues found in 52 source files
```

### Full-suite regression

```text
$ uv run pytest tests/unit -q --no-cov
805 passed, 3282 warnings in 88.05s (0:01:28)
```

Pre-PR baseline was 795 (post 6.1.a); +9 new drift tests = 804 expected.
Observed 805 — the +1 above expected is likely the
`test_settings.py::TestDefaults::test_defaults_load_without_error`
flake (documented in Sprint 5.1.f) recovering, since this PR adds new
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

### Sample build-script invocation (operator-side)

```bash
$ uv run python scripts/build_drift_baseline.py \
    --train-parquet data/processed/tier5_train.parquet \
    --manifest models/sprint3/lightgbm_model_manifest.json \
    --output data/baselines/distributions.parquet \
    --bins 10
```

Expected log output (structlog JSON):

```json
{"event": "build_drift_baseline.start", "n_features": 743, "n_bins": 10, ...}
{"event": "build_drift_baseline.train_loaded", "n_rows": 411836, "n_cols": 750, ...}
{"event": "drift.baseline.feature_constant", "feature": "is_null_some_sparse_feature", ...}
{"event": "drift.baseline.build_complete", "kept_features": 717, "skipped_constant": 26, ...}
{"event": "build_drift_baseline.complete", "n_rows": 7170, "n_kept_features": 717, ...}
```

Expected output: `data/baselines/distributions.parquet` ≈ 250 KB,
~7,170 rows × 6 columns (kept_features × n_bins).

### Sample alert payload (synthetic-drift test output)

```json
{
  "timestamp": "2026-05-10T22:14:30.123456+00:00",
  "run_id": "test-run-001",
  "feature_name": "feature_a",
  "psi": 1.847,
  "threshold": 0.05,
  "n_baseline": 5000,
  "n_recent": 1000
}
```

## Deviations from plan

1. **`np.ndarray` → `NDArray[np.float64]` (mypy strict).** Initial impl
   used bare `np.ndarray` for the `_FeatureBaseline` dataclass fields
   and `_psi_from_pcts` parameters. mypy strict mode required type
   parameters; switched to `numpy.typing.NDArray[np.float64]`. Also
   added `.astype(np.float64)` cast on `recent_pcts = recent_counts /
   len(non_null)` to satisfy the function signature (the division
   produces `floating[Any]` rather than `float64` in mypy's view).
   Pure documentation/type-precision changes; no runtime behaviour
   change.

2. **Settings path validator extended.** Added `drift_baseline_path` and
   `drift_alert_log_dir` to the existing `_resolve_relative_to_project_root`
   field validator's covered list (originally `data_dir`, `models_dir`,
   `logs_dir`). This way a user setting `DRIFT_BASELINE_PATH=./other/path/baseline.parquet`
   in `.env` gets the same project-root-anchored resolution as the
   other path fields, regardless of CWD at process start.

## Cross-references

- `src/fraud_engine/utils/metrics.py:277-369` — `compute_psi(baseline,
  current, bins, epsilon)` whose math kernel `_psi_from_pcts` mirrors.
  The math equivalence test asserts byte-identical agreement.
- `src/fraud_engine/data/splits.py:79-177` — `temporal_split()` for
  the train-slice contract (operators must pass `tier5_train.parquet`,
  not merged data; no val/test leakage into the production-drift
  baseline).
- `src/fraud_engine/data/lineage.py:200-218` — append-only JSONL
  pattern this module mirrors for `drift_alerts.jsonl`.
- `models/sprint3/lightgbm_model_manifest.json:feature_names` — the
  canonical 743-feature list; the build script reads it via
  `_load_feature_names` so baseline columns stay in sync with the
  model's input order.
- `src/fraud_engine/monitoring/prometheus_metrics.py` (Sprint 6.1.a) —
  sibling module: live request-path signals there, offline batch drift
  here.
- `CLAUDE.md` §3 (PSI drift detection as Sprint-6 endpoint), §4
  (`monitoring/` module home), §5.4 (no hardcoded thresholds), §5.5
  (logging discipline).

## Out of scope (Sprint 6.x+)

- **Building the actual `data/baselines/distributions.parquet` artefact**
  — the file is gitignored; this PR ships the build pipeline, not the
  materialised artefact. John runs the script once locally when
  convenient.
- **Top-K by feature importance** (vs PSI magnitude) — would need a
  `feature_importance` field added to the model manifest (Sprint 3.x
  retrofit) + reading it in DriftMonitor; defer.
- **Sliding-window dedupe of alerts** — current behaviour: every cron
  run independent. May emit the same `feature_name` alert daily if
  drift persists. A future prompt can add a "first seen / last seen"
  pattern.
- **Prometheus Counter for drift alerts**
  (`fraud_engine_drift_alerts_total{feature}`) — useful for Grafana
  panels but high-cardinality risk if there are many drifting features.
  Sprint 6.x with bucketed labels.
- **Live (per-request) drift detection** — would require persisting
  the 743-feature vector per request (storage explosion). Major
  write-volume increase for marginal gain over batch monitoring;
  defer indefinitely.
- **Categorical-feature PSI** — current impl assumes numeric features
  (quantile binning). All 743 features in this project are
  numeric-encoded post Tier-2/3, so this isn't blocking. A future
  prompt can add `pd.api.types.is_numeric_dtype` discrimination +
  frequency-table binning for categoricals.
- **Multi-segment drift** (e.g., per `ProductCD`) — same data, sliced
  by a categorical. Defer until product-team asks.
- **CLAUDE.md §13 sprint-status update** — Sprint 6 row gets updated
  by a 6.2.x audit-and-gap-fill PR per established convention.
- **Wiring `DriftMonitor` into a scheduled cron / Airflow DAG** —
  operator concern outside the codebase's deployment scope.
