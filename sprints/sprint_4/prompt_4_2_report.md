# Sprint 4 — Prompt 4.2: `StratifiedEvaluator`

**Date:** 2026-05-09
**Branch:** `sprint-4/prompt-4-2-stratified-evaluator` (off `main` @ `d8180c7` — post 4.1 merge)
**Status:** Verification passed; all spec gates met.

## Headline

| Acceptance gate | Realised | Status |
|---|---|---|
| Synthetic imbalanced segments produce expected differential metrics | 4 sub-gates + headline test all pass on `_small_fixture` | PASS |
| Five stratification axes (amount bucket, ProductCD, device type, identity coverage, month) | All five computed via private per-axis helpers; orchestrated by single `evaluate()` | PASS |
| Output: DataFrame + heatmap plot | Long-format `pd.DataFrame` (9 columns) + `plot_heatmap` returning `Axes` with z-score-normalised cells, sample-size annotations, and direction-aware (cost-flipped) colormap | PASS |
| Skip-with-warning on missing column / `month=None` | `axis_skipped` warning logged; that axis simply absent from the output DataFrame | PASS |

4 of 4 spec gates met. Plus: `make format`, `ruff check`, `mypy --strict src` all green; **38 tests in `test_stratified.py` pass in 2.63 s**; full unit-test regression at **522 passed** (483 baseline + 38 new + 1 misc).

## Summary

- **`src/fraud_engine/evaluation/stratified.py`** (NEW, ~830 LOC including comprehensive docstring) ships `StratifiedEvaluator` — a stateless class with one orchestrator (`evaluate`) that fans out to five private per-axis helpers (`_evaluate_by_amount_bucket`, `_evaluate_by_product_cd`, `_evaluate_by_device_type`, `_evaluate_by_identity_coverage`, `_evaluate_by_month`), one workhorse `_stratum_metrics` for per-stratum AUC / PR-AUC / cost computation, and one `plot_heatmap` method for the visualisation. Cost computation is delegated to an injected `EconomicCostModel` (DI in the constructor; default = `EconomicCostModel()`).
- **`src/fraud_engine/evaluation/__init__.py`** updated to re-export `StratifiedEvaluator` (alphabetised in `__all__`) and refresh the package docstring with the 4.2 paragraph.
- **`tests/unit/test_stratified.py`** (NEW, ~570 LOC) ships 38 tests across the five spec-mandated contract surfaces (TestInit, TestEvaluate, TestPerAxisLogic, TestPlotHeatmap, TestErrorHandling) plus two synthetic-data generators (`_small_fixture`, `_amount_only_fixture`) following `test_economic.py`'s pattern.
- **No changes** to `economic.py`, `calibration.py`, `utils/metrics.py`, `config/settings.py`, or any model files.

## Spec vs. actual

| Spec line | Actual |
|---|---|
| `StratifiedEvaluator` class | `class StratifiedEvaluator` with `__init__(cost_model=None, threshold=None, min_stratum_size=50)` and three public methods + read-only properties (`threshold`, `min_stratum_size`, `cost_model`). |
| Stratify by amount bucket: <$50, $50-200, $200-500, $500-1K, >$1K | `_DEFAULT_AMOUNT_BUCKETS` constant with half-open `[low, high)` intervals. Edge case ($50.0 → `$50-200`, not `<$50`) pinned in test. |
| Stratify by ProductCD | `_evaluate_by_product_cd` emits one row per unique value (sorted) plus a `(missing)` bucket for NaN rows. |
| Stratify by device type (mobile / desktop / null) | `_evaluate_by_device_type` emits one row per unique `DeviceType` value plus an explicit `(null)` bucket for NaN rows. |
| Stratify by identity coverage (has / no identity) | `_evaluate_by_identity_coverage` produces exactly two rows: `has_identity` (probe `id_01.notna()`) and `no_identity`. Probe column choice documented per CLAUDE.md §1. |
| Stratify by temporal (month 5 vs month 6) | `_evaluate_by_month` emits one row per unique value of the caller-supplied `month: pd.Series`. Tier-5 parquet drops `timestamp`, so the caller passes `month` explicitly; pin "5 vs 6" not hardcoded — strata are whatever values appear in the series. |
| Output: DataFrame + heatmap plot | `evaluate(...)` returns a long-format DataFrame with columns `stratum_axis, stratum_value, n_rows, n_pos, fraud_rate, auc, pr_auc, total_cost, cost_per_txn`. `plot_heatmap(eval_df, ...)` returns a matplotlib `Axes` with one heatmap (rows = stratum, cols = chosen metrics, colour = z-score with cost columns sign-flipped, cell annotations include raw value + sample size). |
| Test: synthetic imbalanced segments produce expected differential metrics | `TestPerAxisLogic` — four axis-specific gates plus the consolidated headline test, all pass on `_small_fixture` (4000 rows, seed=42). |
| `uv run pytest tests/unit/test_stratified.py -v` | **38 passed in 2.63 s** |

## Test inventory

38 tests across 5 contract surfaces:

| Class | Count | Coverage |
|---|---|---|
| `TestInit` | 9 | Default cost-model from Settings; default threshold from Settings; explicit cost-model / threshold / min_stratum_size overrides; threshold > 1 raises; threshold < 0 raises; min_stratum_size = 0 raises; min_stratum_size negative raises |
| `TestEvaluate` | 8 | Long-format DataFrame columns; `stratum_axis` ⊆ canonical axes; `n_rows` per axis sums to `len(y_true)`; `fraud_rate = n_pos/n_rows` identity; all 5 amount buckets emitted on full-range synthetic; missing column → warning + axis absent; `month=None` → axis absent; canonical axis ordering |
| `TestPerAxisLogic` | 8 | **Spec gates:** (a) low-amount fraud_rate > high-amount fraud_rate by > 0.20; (b) ProductCD W (separable) AUC > C (overlapping) AUC by > 0.10; (c) `has_identity` AUC > `no_identity` AUC; (d) `month=5` (noisy) cost_per_txn > `month=6` (clean) cost_per_txn; **headline consolidated test**; amount-bucket boundary semantics (50.0 → `$50-200`); device-type `(null)` bucket; identity-coverage exactly two groups |
| `TestPlotHeatmap` | 7 | Returns `Axes`; respects `ax` kwarg; cell annotations include `n=`; cost cells include `$` prefix; `fig.savefig` smoke; unknown metric raises; empty `eval_df` → placeholder Axes |
| `TestErrorHandling` | 6 | `y_scores ∉ [0, 1]` raises (positive overflow); negative scores raise; `y_true` / `y_scores` shape mismatch raises; frame length mismatch raises; month length mismatch raises; single-class stratum → NaN AUC/PR-AUC + warning logged + cost still computed |

## Files-changed table

| File | Change | LOC |
|---|---|---|
| `src/fraud_engine/evaluation/stratified.py` | new (`StratifiedEvaluator` class + 5 axis helpers + `_stratum_metrics` workhorse + `_z_score_for_heatmap` + 2 formatters + module docstring with 10 trade-offs) | ~830 |
| `src/fraud_engine/evaluation/__init__.py` | add `StratifiedEvaluator` import + `__all__` entry; refresh docstring | +9 |
| `tests/unit/test_stratified.py` | new (38 tests across 5 classes + 2 synthetic generators) | ~570 |
| `sprints/sprint_4/prompt_4_2_report.md` | this file | (this file) |

**No changes** to `economic.py`, `calibration.py`, `utils/metrics.py`, `config/settings.py`, or any model files.

## Decisions worth flagging

1. **Stateless class wrapping a stateful primitive.** Mirrors `EconomicCostModel`'s shape — config (cost-model + threshold + min-stratum floor) in the constructor; no learned state. The closest pattern analog is `select_calibration_method` (sweep + pick), not `PlattScaler` / `IsotonicCalibrator`. No `is_fitted_` flag, no pre-fit guard.

2. **Single `evaluate()` orchestrator + 5 private helpers**, NOT public per-axis methods. A reviewer wants one call returning one frame; per-axis methods would balloon the public surface with no flexibility win — the long-format frame is already filterable via `groupby('stratum_axis')`. Validated once at the top of `evaluate`; helpers receive clean arrays (no re-validation per call).

3. **`min_stratum_size: int = 50` floor.** Strata with fewer rows return NaN AUC/PR-AUC (cost still computed; warning logged). AUC = 1.0 on n=3 rows is meaningless noise that would discolour the heatmap. 50 is the smallest size where a 10 % fraud rate gives expected n_pos = 5 — barely enough for AUC to be non-trivial.

4. **`id_01.notna()` as identity-coverage probe.** Highest-non-null `id_*` column per CLAUDE.md §1 ("only 24 % of transactions have device/identity data"). Single-column probe is reliable and avoids conflating identity-coverage with the device-type axis (which `DeviceType.notna()` would do, smearing two signals across the same mask).

5. **Skip-with-warning on missing column** rather than raise. A tier-1-only experimental frame missing `DeviceType` shouldn't crash the evaluator — log `WARNING(stratified.axis_skipped, axis=..., reason=...)` and omit that axis from the output. Reviewer-friendly on partial frames; the alternative (raise) would force every caller to pre-strip axes.

6. **Include single-class strata with NaN metrics + warning**, not drop them. A reviewer wants to see "this stratum had only positives" — dropping hides the skew. Cost is still computed (well-defined on a single class); only AUC / PR-AUC are NaN.

7. **Heatmap z-scores per metric column with diverging colormap (`RdYlBu_r`)**, with axis-aware sign flip for cost columns so red always means "this stratum is worse" regardless of metric direction. Cells annotated with raw value + sample size (`"AUC=0.920\n(n=15.2K)"`) so a reader can judge cell trustworthiness at a glance. NaN cells render light grey via `cmap.set_bad("lightgray")`.

8. **Pandas DataFrame return for `evaluate`, matplotlib `Axes` return for `plot_heatmap`** — consistent with `economic.py` (DataFrame for cost curves) and `calibration.py:reliability_diagram` (Axes for plots). Caller saves figures via `ax.figure.savefig`.

9. **Month axis is a keyword arg on `evaluate`, not derived from `frame['timestamp']`.** Tier-5 parquet drops `timestamp` (`build_features_all_tiers.py:110-111`). Making the caller pass `month` explicitly is least-surprising; pass `month=None` to skip the axis cleanly. Strata are whatever unique values appear in the series — "month 5 vs month 6" is not pinned in code.

10. **MLflow logging deferred to Sprint 4.x+** (mirrors 4.1's deferral). Inline TODO comment marks the integration point in `plot_heatmap`. The `evaluate()` DataFrame and the returned `Axes` are clean handoff points for a future MLflow-aware reporter.

## Surprising findings

1. **NaN handling in the heatmap is simpler than expected.** Both `np.ma.array(...)` and `np.ma.masked_invalid(...)` are flagged by mypy as untyped (numpy stubs gap). Switching to `imshow(z_data, ...)` with `cmap.set_bad("lightgray")` works just as well — matplotlib honours `set_bad` for plain-ndarray NaN cells without needing a masked array wrapper. One less type-ignore.

2. **The synthetic-fixture composition needs care for spec gates to pass independently.** With `_small_fixture` layering four axis-specific biases simultaneously (amount-driven y-rate, ProductCD-driven score quality, identity-driven score blur, month-driven additive noise), each gate's effect could conflict with another. Empirically the four gates pass cleanly with the chosen Beta-distribution parameters and bias magnitudes, but the test suite has both the consolidated headline test AND four axis-specific tests with isolated fixtures (`_amount_only_fixture` for gate (a)) so a flake in one doesn't mask the others.

3. **Heatmap row count is variable** (depends on which axes are present in the frame and how many unique values each axis has). A frame with ProductCD={W,C,R,H,S} + DeviceType={mobile,desktop,(null)} + identity={has,no} + month={5,6} produces 5+5+3+2+2 = 17 rows; dropping ProductCD drops 5 rows. The heatmap auto-sizes via `figsize=(8, 0.4*n_rows + 1)` — empirically readable from 2 to ~25 strata.

## Verbatim verification output

### Cheap gates

```
$ uv run ruff format src/fraud_engine/evaluation/stratified.py \
                     src/fraud_engine/evaluation/__init__.py \
                     tests/unit/test_stratified.py
3 files already formatted

$ uv run ruff check src/fraud_engine/evaluation/stratified.py \
                    src/fraud_engine/evaluation/__init__.py \
                    tests/unit/test_stratified.py
All checks passed!

$ uv run mypy src
Success: no issues found in 40 source files
```

### Spec verification

```
$ uv run pytest tests/unit/test_stratified.py -v --no-cov
======================= 38 passed, 14 warnings in 2.63s ========================
```

### Unit-test regression

```
$ uv run pytest tests/unit -q --no-cov
522 passed, 34 warnings in 106.85s (0:01:46)
```

(Up from 483 post-4.1 baseline by +39: 38 new in `test_stratified.py` + 1 misc baseline shift in another module. No regressions.)

## Out of scope (Sprint 4.x+)

- **MLflow logging of `evaluate()` DataFrame and heatmap PNG** — deferred per plan; inline TODO at the integration point in `plot_heatmap`.
- **Cross-axis interactions** (e.g., amount × device cell-level AUC heatmap). Spec is per-axis only; cross-axis is a Sprint 4.x experiment.
- **Persistence of stratum-specific thresholds** (Sprint 5 territory; per-segment thresholds would consume `EconomicCostModel.optimize_threshold` in a loop over strata).
- **Updating CLAUDE.md §13 sprint status table** (per `docs/CONTRIBUTING.md` §4: handled in the next sprint's first PR, not as its own commit).

## Acceptance checklist

- [x] Branch `sprint-4/prompt-4-2-stratified-evaluator` off `main` (`d8180c7`, post 4.1 merge)
- [x] `src/fraud_engine/evaluation/stratified.py` created (~830 LOC; `StratifiedEvaluator` class + 5 axis helpers + workhorse + heatmap + module docstring with 10 trade-offs + cross-references)
- [x] `src/fraud_engine/evaluation/__init__.py` re-exports `StratifiedEvaluator` (alphabetised in `__all__`)
- [x] `tests/unit/test_stratified.py` created (38 tests across 5 classes + 2 synthetic generators)
- [x] Spec gate (a): low-amount fraud_rate > high-amount fraud_rate by > 0.20 — PASS
- [x] Spec gate (b): separable ProductCD AUC > overlapping ProductCD AUC by > 0.10 — PASS
- [x] Spec gate (c): `has_identity` AUC > `no_identity` AUC — PASS
- [x] Spec gate (d): noisy-month cost_per_txn > clean-month cost_per_txn — PASS
- [x] Headline consolidated test: synthetic imbalanced segments produce expected differential metrics — PASS
- [x] DataFrame output: long-format with 9 columns
- [x] Heatmap output: matplotlib `Axes` with z-score normalised colour, cost-flipped direction, sample-size annotations
- [x] `make format && make lint && make typecheck` all return 0
- [x] `uv run pytest tests/unit/test_stratified.py -v` returns 0 (38 passed in 2.63 s)
- [x] `uv run pytest tests/unit -q` returns 0 (522 passed; no regressions vs 483 baseline)
- [x] `sprints/sprint_4/prompt_4_2_report.md` written
- [x] No git/gh commands run beyond §2.1 carve-out (branch create only; commit + push + merge happen after John's sign-off)

Verification passed. Ready for John to commit on `sprint-4/prompt-4-2-stratified-evaluator`.

**Commit note:**
```
4.2: StratifiedEvaluator (per-axis AUC / PR-AUC / cost + heatmap)
```
