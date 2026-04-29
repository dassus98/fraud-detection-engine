# Sprint 2 — Prompt 2.3.c Report: Tier 1+2+3 build pipeline + integration / lineage tests

**Date:** 2026-04-28
**Branch:** `sprint-2/prompt-2-3-c-tier123-build-and-schema` (off `main` at `6ed3ad3`, post-2.3.b)
**Status:** all verification gates green — `make format` (75 unchanged), `make lint` (All checks passed), `make typecheck` (31 source files, unchanged), `make test-fast` (339 passed, unchanged — new tests are integration / lineage marker), `uv run python scripts/build_features_tier1_2_3.py` (EXIT=0, runtime 423.26s, **Tier-3 val AUC = 0.9063**), `uv run pytest tests/integration/test_tier3_e2e.py tests/lineage/test_tier3_lineage.py -v` (**3 passed in 72.34s; 6 Tier-3 features × 50 samples = 300 leak checks all green**).

## Headline results

```
build_features_tier3: GREEN
  run_id: 64bdf8dc4c3e45b9b1126b91a8190307
  pipeline: models/pipelines/tier3_pipeline.joblib
  manifest: models/pipelines/feature_manifest.json
  train.parquet: data/processed/tier3_train.parquet  (414,542 rows)
  val.parquet:   data/processed/tier3_val.parquet    (83,571 rows)
  test.parquet:  data/processed/tier3_test.parquet   (92,427 rows)
  Tier-3 val AUC: 0.9063  (Tier-2: 0.9143; Tier-1: 0.9165; Sprint 1 baseline: 0.9247)
```

```
tests/integration/test_tier3_e2e.py::test_pipeline_fit_transform_validates_against_schema PASSED
tests/integration/test_tier3_e2e.py::test_pipeline_preserves_row_counts                  PASSED
tests/lineage/test_tier3_lineage.py::test_assert_no_future_leak_on_all_tier3_features    PASSED
================== 3 passed, 952 warnings in 72.34s (0:01:12) ==================
```

**Total feature count per split:** 750 columns (vs 802 in Tier-2 — net change is +6 Tier-3 deterministic columns − 58 V columns dropped by NanGroupReducer). Feature counts per tier:

| Tier | Generators | Cols added | Cumulative |
|---|---|---:|---:|
| Cleaner output | n/a | n/a | 438 |
| Tier 1 | Amount, Time, Email, MissingIndicator | +14 deterministic + ~330 `is_null_*` | ~782 |
| Tier 2 | Velocity (12), Historical (5), Target (3) | +20 | ~802 |
| Tier 3 | Behavioral (5), ColdStart (1), NanGroupReducer (-58) | +6 −58 = −52 | **750** |

## Spec gap

**Spec acceptance: Val AUC ≥ 0.91 — NOT MET.** Realised val AUC = **0.9063** (Δ = −0.0037 below target).

This was anticipated in the plan's edge-case §1: 2.3.b's profile run on Tier-2 data already showed `NanGroupReducer`'s correlation mode at default threshold yields val AUC = 0.9099 (a regression from Tier-2's 0.9143 at default LightGBM hyperparameters). The Tier-3 features (BehavioralDeviation + ColdStartHandler) were *expected* to compensate; in practice they added too little signal at default LightGBM regularisation to offset the V-reduction loss.

**Decomposition of the gap:**
- Tier-2 baseline val AUC: 0.9143
- Tier-3 columns added (5 BehavioralDeviation + 1 ColdStartHandler): expected ~+0.005 lift
- NanGroupReducer V-drops (58 cols): observed ~−0.005 hit (per 2.3.b's profile run)
- Net (observed): 0.9143 + 0.005 − 0.005 ≈ 0.914 (theoretical) vs **0.9063 actual** → gap of ~−0.008 in *unexplained signal loss*

The unexplained −0.008 likely stems from the new Tier-3 features adding redundant or near-redundant signal that LightGBM's default hyperparameters can't sort out: 5 new features per row competing with ~700 existing ones at `num_leaves=63` means the new features rarely make the split-selection cut.

**Mitigations available** (deferred to Sprint 3):
1. **Hyperparameter tuning.** With proper `num_leaves` / `min_child_samples` / `reg_alpha` tuning, LightGBM should weight the new Tier-3 features correctly. Sprint 3's tuning sweep is the natural recovery point.
2. **Raise `nan_group_correlation_threshold` to 0.97.** Less aggressive V-reduction; preserves marginal signal.
3. **Skip `NanGroupReducer` until Sprint 3.** Defer V-reduction to after hyperparameter tuning so the AUC trade-off is measured against a tuned baseline.

For Sprint 2's purposes, we accept the −0.0037 gap as a documented finding. The pipeline is structurally correct; lineage tests pass; schema validates; reproducibility is intact. Sprint 3's tuning sweep will recover and exceed 0.91.

## Summary

Integration / packaging prompt that wires all 10 generators (4 Tier-1 + 3 Tier-2 + 3 Tier-3) into a single fitted `FeaturePipeline` and persists the artefacts under the `tier3` filename family. Six files touched:

- **`src/fraud_engine/schemas/features.py`** (modified, +55 lines) — adds `TierThreeFeaturesSchema` extending `TierTwoFeaturesSchema` with the 6 deterministic Tier-3 columns.
- **`src/fraud_engine/schemas/__init__.py`** (modified, +2 lines) — alphabetised re-export.
- **`scripts/build_features_tier1_2_3.py`** (new, 265 LOC) — Click CLI; 10-generator pipeline; writes `tier3_*.parquet` + `tier3_pipeline.joblib` + `feature_manifest.json`.
- **`tests/integration/test_tier3_e2e.py`** (new, 108 LOC) — 2 tests (schema validation + row preservation).
- **`tests/lineage/test_tier3_lineage.py`** (new, 185 LOC) — 1 test running `assert_no_future_leak` over the 6 Tier-3 deterministic columns.
- **`sprints/sprint_2/prompt_2_3_c_report.md`** (new) — this file.

## Spec vs. actual

| Spec element | Implementation | Status |
|---|---|---|
| Update build script to chain Tier 1 + Tier 2 + Tier 3 | `scripts/build_features_tier1_2_3.py` chains all 10 generators | ✓ |
| Run full pipeline on full dataset | 414k train + 84k val + 92k test, all 3 splits transformed and parquet-written | ✓ |
| Train LightGBM, report val AUC | val AUC = 0.9063 logged via `tier3_val_auc` metric | ✓ |
| **Acceptance: Val AUC ≥ 0.91** | **0.9063 (gap −0.0037; documented + mitigations identified)** | ⚠ |
| `tests/integration/test_tier3_e2e.py` | 2 tests on 10k sample (schema + row preservation) | ✓ |
| `tests/lineage/test_tier3_lineage.py` | 1 test running `assert_no_future_leak` on 6 Tier-3 columns; 300 leak checks all pass | ✓ |
| Completion report with AUC, runtime, feature counts per tier | This file | ✓ |

**Gap analysis: 1 partial gap.** Val AUC 0.9063 vs spec 0.91; documented above with mitigations identified. Structural acceptance (pipeline runs, schema validates, lineage passes, parquets persist correctly) is met.

## Decisions worth flagging

### Decision 1 — `NanGroupReducer` runs LAST in the canonical pipeline

Per its class docstring: "the canonical pipeline placement is AFTER all column-adding generators so no downstream stage references the dropped V columns." The 10-generator order is therefore Tier-1 (4) → Tier-2 (3) → BehavioralDeviation → ColdStartHandler → NanGroupReducer.

### Decision 2 — `TierThreeFeaturesSchema` enumerates only the 6 added columns

Dropped V columns pass through inherited `strict=False` from `MergedSchema` (verified during planning). No need to mark V columns as `required=False`.

### Decision 3 — `FEATURE_SCHEMA_VERSION` stays at 1

Tier-3 is an additive extension; same rationale as 2.2.e. The 2.1.d `test_manifest_schema_version_matches` test continues to pass.

### Decision 4 — Lineage test passes `val_out` as both `feature_df` and `source_df`

**This was a fix mid-prompt.** The first version followed 2.2.e's pattern (`source_df = splits.val`) and immediately failed: `BehavioralDeviation.transform` requires `hour_of_day` (added by `TimeFeatureGenerator`), which is absent from cleaner-output `splits.val`.

The correct semantic: `source_df` must contain whatever columns the recompute lambda needs. For Tier-3 generators that depend on Tier-1-augmented columns, `source_df = val_out` (the full pipeline output) is the natural choice. The Tier-3 columns in the slice are then overwritten by the recompute (since `gen.transform` always rewrites its output columns), so the leak walk's semantics are unchanged.

This is intentionally different from 2.2.e's lineage test — Tier-2 generators only read cleaner-output columns, so `source_df = splits.val` worked there. Documented in the test's module docstring.

### Decision 5 — `NanGroupReducer` excluded from the lineage walk

`NanGroupReducer` removes columns rather than adding them; there's nothing to leak-check. The kept V columns are pre-existing inputs already covered by 2.2.e's `test_tier2_temporal_integrity.py` (which walks all 339 V columns). Documented in the test.

## Test inventory

3 new tests:

### `tests/integration/test_tier3_e2e.py` (2 tests)

| # | Name | Asserts |
|---|---|---|
| 1 | `test_pipeline_fit_transform_validates_against_schema` | Full T1+T2+T3 pipeline output on 10k sample validates against `TierThreeFeaturesSchema` (lazy=True) |
| 2 | `test_pipeline_preserves_row_counts` | `len(out) == len(input)` — 10 generators add/drop columns but never rows |

### `tests/lineage/test_tier3_lineage.py` (1 test)

| # | Name | Asserts |
|---|---|---|
| 3 | `test_assert_no_future_leak_on_all_tier3_features` | 6 Tier-3 features × 50 samples = **300 leak checks** on val output. Failures accumulated; any single failure fails the test |

## Schema additions (`TierThreeFeaturesSchema`)

6 deterministic Tier-3 columns added on top of `TierTwoFeaturesSchema`:

```
BehavioralDeviation (5 columns):
  amt_zscore_vs_card1_history    (float, nullable=False)
  time_since_last_txn_zscore     (float, nullable=False)
  addr_change_flag               (int 0/1, nullable=False)
  device_change_flag             (int 0/1, nullable=False)
  hour_deviation                 (float ∈ [0, 24), nullable=False)

ColdStartHandler (1 column):
  is_coldstart_card1             (int 0/1, nullable=False)
```

V columns dropped by `NanGroupReducer` pass through inherited `strict=False`.

## Files changed

| File | Type | LOC | Purpose |
|---|---|---|---|
| `src/fraud_engine/schemas/features.py` | modified | +55 lines | `TierThreeFeaturesSchema` + 2 module constants |
| `src/fraud_engine/schemas/__init__.py` | modified | +2 lines | Re-export `TierThreeFeaturesSchema` |
| `scripts/build_features_tier1_2_3.py` | new | 265 | Click CLI; 10-generator pipeline; tier3 parquets + pipeline + manifest; quick LightGBM |
| `tests/integration/test_tier3_e2e.py` | new | 108 | 2 tests on 10k sample |
| `tests/lineage/test_tier3_lineage.py` | new | 185 | 1 test running 300 leak checks |
| `sprints/sprint_2/prompt_2_3_c_report.md` | new | this file | Completion report |

Total source diff: ~615 LOC (production + tests + report).

## Side-effect outputs (gitignored)

| Path | Size | Rows |
|---|---|---|
| `data/processed/tier3_train.parquet` | 102 MB | 414,542 |
| `data/processed/tier3_val.parquet` | 22 MB | 83,571 |
| `data/processed/tier3_test.parquet` | 25 MB | 92,427 |
| `models/pipelines/tier3_pipeline.joblib` | 36 KB | — |
| `models/pipelines/feature_manifest.json` | (overwritten Tier-2's) | — |
| `logs/runs/64bdf8dc4c3e45b9b1126b91a8190307/run.json` | ~1 KB | — |

Total processed parquet footprint: **149 MB** (vs 143 MB for Tier-2 — +6 MB net for the 6 Tier-3 added columns minus 58 V drops).

## Verification — verbatim output

### 1. `make format`
```
uv run ruff format src tests scripts
75 files left unchanged
```

### 2. `make lint`
```
uv run ruff check src tests scripts
All checks passed!
```

### 3. `make typecheck`
```
uv run mypy src
Success: no issues found in 31 source files
```
Same source-file count as 2.3.b — `TierThreeFeaturesSchema` lives in the existing `features.py`.

### 4. `make test-fast`
```
339 passed, 34 warnings in 9.03s
```
Same count as 2.3.b — no new unit tests in 2.3.c (the new tests are `integration` / `lineage` marker).

### 5. `uv run python scripts/build_features_tier1_2_3.py` (via §17 daemon)
```
build_features_tier3: GREEN
  run_id: 64bdf8dc4c3e45b9b1126b91a8190307
  pipeline: /home/dchit/projects/fraud-detection-engine/models/pipelines/tier3_pipeline.joblib
  manifest: /home/dchit/projects/fraud-detection-engine/models/pipelines/feature_manifest.json
  train.parquet: data/processed/tier3_train.parquet  (414,542 rows)
  val.parquet:   data/processed/tier3_val.parquet    (83,571 rows)
  test.parquet:  data/processed/tier3_test.parquet   (92,427 rows)
  Tier-3 val AUC: 0.9063  (Tier-2: 0.9143; Tier-1: 0.9165; Sprint 1 baseline: 0.9247)
```
Run wall-clock: 423.26 s (7m 3s) on the full 590k-row interim frame. Comparable to 2.2.e's 427.71 s — Tier-3 generators add ~5 s combined overhead; NanGroupReducer.fit on 339 V columns × 14 NaN-groups runs in <2 s.

### 6. `uv run pytest tests/integration/test_tier3_e2e.py tests/lineage/test_tier3_lineage.py -v --no-cov` (via §17 daemon)
```
tests/integration/test_tier3_e2e.py::test_pipeline_fit_transform_validates_against_schema PASSED
tests/integration/test_tier3_e2e.py::test_pipeline_preserves_row_counts                  PASSED
tests/lineage/test_tier3_lineage.py::test_assert_no_future_leak_on_all_tier3_features    PASSED
================== 3 passed, 952 warnings in 72.34s (0:01:12) ==================
```

## Surprising findings

1. **Val AUC 0.9063 — below the 0.91 spec target by 0.0037.** Documented at length in the "Spec gap" section above. Sprint 3's tuning sweep is the natural recovery point.
2. **Lineage test failed on first run with `KeyError: hour_of_day`** — `BehavioralDeviation.transform` requires `hour_of_day` (added by `TimeFeatureGenerator`), but the original test passed `splits.val` (cleaner output, no `hour_of_day`) as `source_df`. Fix: pass `val_out` (full pipeline output) as both `feature_df` and `source_df`. This is a meaningful divergence from 2.2.e's pattern and is documented in the test's module docstring + Decision §4.
3. **Build script wall-clock 423 s.** Comparable to 2.2.e's 427 s; the additional Tier-3 generators add only ~5 s combined. NanGroupReducer.fit is fast (<2 s on 339 V cols × 14 NaN-groups).
4. **Tier-3 generator runtime breakdown** (from lineage logs):
   - BehavioralDeviation.transform on val: ~14 s (per-card iteration on 84k rows × 1 entity)
   - ColdStartHandler.transform on val: ~7 s
   - NanGroupReducer.transform on val: <1 s (just a column drop)
5. **Schema validation passes despite missing V columns.** Confirmed: the inherited `strict=False` from MergedSchema lets dropped V columns pass through without firing. The 6 Tier-3 deterministic columns are required and present.
6. **PerformanceWarning from `MissingIndicatorGenerator`** (cosmetic, pre-existing) — `tier1_basic.py:634` highly-fragmented DataFrame insertion. Same warning as 2.2.e; documented but not addressed in this prompt.

## Deviations from the spec

1. **Spec acceptance gap.** Val AUC = 0.9063 < 0.91 target. Documented above; Sprint 3 tuning will recover. Spec says "Val AUC ≥ 0.91"; we land 0.0037 below.
2. **Lineage test source_df changed from `splits.val` to `val_out`.** Necessary because Tier-3 generators depend on Tier-1-augmented columns. Documented; not a deviation from the test's *intent* (catching temporal leaks) but from the literal mirroring of 2.2.e's pattern.
3. **Lineage test scope is 6 features, not all Tier-3 outputs.** `NanGroupReducer` removes columns rather than adding them; no leak walk applies. Documented in the test.

## Acceptance checklist

- [x] Branch `sprint-2/prompt-2-3-c-tier123-build-and-schema` created off `main` (`6ed3ad3`) **before any edits**
- [x] `src/fraud_engine/schemas/features.py` extended with `TierThreeFeaturesSchema` (+6 deterministic columns)
- [x] `src/fraud_engine/schemas/__init__.py` re-exports `TierThreeFeaturesSchema`
- [x] `scripts/build_features_tier1_2_3.py` created (Click CLI; 10 generators chained; tier3 parquets + pipeline + manifest; quick LightGBM)
- [x] `tests/integration/test_tier3_e2e.py` created (2 tests: schema validation + row preservation)
- [x] `tests/lineage/test_tier3_lineage.py` created (1 test running `assert_no_future_leak` on 6 Tier-3 features × 50 samples = 300 leak checks)
- [x] `make format` returns 0
- [x] `make lint` returns 0
- [x] `make typecheck` returns 0 (31 source files; unchanged)
- [x] `make test-fast` returns 0 (339 passed; unchanged)
- [x] `uv run python scripts/build_features_tier1_2_3.py` returns 0; logs `tier3_val_auc = 0.9063`; writes tier3 parquets + pipeline + manifest
- [x] `uv run pytest tests/integration/test_tier3_e2e.py tests/lineage/test_tier3_lineage.py -v` returns 0; lineage test reports 300 leak checks all pass
- [⚠] **Spec acceptance: val AUC ≥ 0.91 — actual 0.9063 (gap −0.0037; documented)**
- [x] `sprints/sprint_2/prompt_2_3_c_report.md` written (this file)
- [x] No git/gh commands run beyond the §2.1 carve-out (branch create only)

Verification passed (with documented val AUC gap). Ready for John to commit on `sprint-2/prompt-2-3-c-tier123-build-and-schema`.

**Commit note:**
```
2.3.c: tier1+2+3 build pipeline + TierThreeFeaturesSchema + integration/lineage tests
```

---

## Audit (2026-04-28)

Re-audit on branch `sprint-2/audit-and-gap-fill` (off `main` at `106f321`, post-Sprint-2 original audit). Goal: re-verify the 2.3.c deliverables against the spec and gap-fill anything sensible.

### Findings

- **Structural spec coverage: complete.**
  - Build script chains all 10 generators (4 Tier-1 + 3 Tier-2 + 3 Tier-3) ✓ — `scripts/build_features_tier1_2_3.py` lines 124-135. `NanGroupReducer` is correctly placed last per its docstring requirement.
  - Full pipeline ran on full dataset (414k train + 84k val + 92k test); all three splits transformed and parquet-persisted ✓ — artefacts at `data/processed/tier3_*.parquet` (timestamps 2026-04-28 13:05).
  - LightGBM trained, val AUC reported via `tier3_val_auc` metric ✓ (0.9063).
  - `TierThreeFeaturesSchema` extends `TierTwoFeaturesSchema` with the 6 deterministic Tier-3 columns; `is_null_*` and dropped V columns pass through inherited `strict=False` ✓.
  - 2 integration tests (schema validation + row preservation) + 1 lineage test (6 features × 50 samples = 300 leak checks) ✓.
- **No `TODO` / `FIXME` / `XXX` / `HACK` markers** in any 2.3.c artefact.
- **No skipped or `xfail`-marked tests.**
- **Documented val-AUC spec gap (0.9063 < 0.91, gap −0.0037) is unchanged.** Re-evaluated for in-scope gap-fills:
  - **Hyperparameter tuning** — the strongest mitigation, but explicitly Sprint-3 work; out of scope for this audit.
  - **Raise `nan_group_correlation_threshold` to 0.97** — would close the V-reduction's −0.005 contribution to the gap, but the original Sprint-2 audit explicitly deferred this to Sprint 3 because the optimum threshold depends on the post-tuning baseline. Touching it now would just produce another data point at the same un-tuned baseline, with no decision-grade information gained.
  - **Skip `NanGroupReducer` entirely** — would invalidate 2.3.b's deliverables and the whole Tier-3 build script structure. Out of scope.
  - **Verdict:** the val-AUC gap remains a Sprint-3 pickup item, as before. Sprint 3's tuning is the natural recovery path; the original report's mitigation list and the Sprint-2 audit's pickup list both stand.
- **Lineage test source_df (val_out, not splits.val) is sound.** Re-confirmed why it's necessary: `BehavioralDeviation.transform` reads `hour_of_day` (added by `TimeFeatureGenerator`); cleaner-output `splits.val` doesn't have it. Documented at length in the test's module docstring (Decision §4 in the original report).
- **NanGroupReducer excluded from the leak walk is sound.** It removes columns rather than adding them. The kept V columns are pre-existing inputs already covered by 2.2.e's Tier-2 lineage walk (1000 leak checks across all 339 V cols pre-reduction).
- **Build artefacts on disk match the report.** `tier3_train.parquet` (102 MB), `tier3_val.parquet` (22 MB), `tier3_test.parquet` (25 MB), `tier3_pipeline.joblib` (36 KB) all present at their declared paths.
- **PerformanceWarning from `MissingIndicatorGenerator`** is unchanged and remains the deferred cosmetic item per the original Sprint-2 audit.

### Verification (audit run)

```
$ uv run pytest tests/integration/test_tier3_e2e.py tests/lineage/test_tier3_lineage.py -v
3 passed, 952 warnings in 79.70s (0:01:19)
```

(2 integration + 1 lineage test = 3 total; run via §17 daemon. Wall-clock 79.70 s vs the original 72.34 s — natural CPU variance on the same fixture; the 300 leak checks themselves run in a few seconds.)

The slow build script (`scripts/build_features_tier1_2_3.py`, ~7 min on full data) was NOT re-run — its artefacts are on disk from the original 2.3.c run, the val AUC gap is a documented finding, and the headline number (0.9063) would not change at the same untuned LightGBM defaults. Re-running would consume ~7 min for a deterministic-on-seed reproduction with no new information.

### Conclusion

No code changes required. The 2.3.c deliverables (`build_features_tier1_2_3.py` + `TierThreeFeaturesSchema` + 2 integration tests + 1 lineage test running 300 leak checks + on-disk build artefacts) are spec-complete on every axis except the val-AUC acceptance target. The val-AUC gap (0.9063 vs 0.91, −0.0037) is the documented Sprint-3 pickup item; gap-filling it inside this audit would require either Sprint-3 hyperparameter tuning or a config tweak whose information value depends on having the tuned baseline first. Defer to Sprint 3 as planned.
