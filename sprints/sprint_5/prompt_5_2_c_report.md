# Sprint 5 — Prompt 5.2.c: Shadow Comparison Report

**Date:** 2026-05-10
**Branch:** `sprint-5/prompt-5-2-c-shadow-compare` (off `main` @ `dda4044` — post 5.2.b merge)
**Status:** Verification passed; all spec gates met. 13/13 unit tests pass; `--sample` mode generates a complete weekly report demonstrating the per-criterion verdict format.

## Headline

| Acceptance gate | Realised | Status |
|---|---|---|
| Offline comparison: load N predictions with both scores | DataFrame-input contract via `ShadowComparison(predictions, costs)` — module is data-source-agnostic | ✅ PASS |
| Compute agreement rate | `_compute_agreement()` over the full N (no labels needed) | ✅ PASS |
| Compute score correlation | `_compute_correlation()` Pearson over the full N | ✅ PASS |
| Compute economic cost on labeled subset | `_per_row_costs()` mirrors Sprint 4.1's cost matrix; `champion_cost_per_txn` / `shadow_cost_per_txn` from labeled rows only | ✅ PASS |
| Bootstrap significance | `_bootstrap_cost_diff()` — non-parametric, deterministic via fixed seed; returns (mean_diff, 95% CI, two-sided p-value) | ✅ PASS |
| Weekly report | `scripts/shadow_compare_report.py` writes `reports/shadow_compare_<date>.md` per CLI invocation | ✅ PASS |
| Promotion criteria: cost > 2%, p < 0.05, agreement > 85% | `PromotionVerdict` carries per-criterion pass/fail + human-readable reasons; `should promote` requires ALL three | ✅ PASS |
| `uv run pytest tests/unit/test_shadow_compare.py -v` | **13 passed in 2.09 s** | ✅ PASS |
| `uv run python scripts/shadow_compare_report.py --sample` | Exit 0; report written to `reports/shadow_compare_2026-05-10.md` (1869 bytes); stdout shows verdict | ✅ PASS |

9 of 9 spec gates met. Plus: `make format` / `ruff check` / `mypy --strict src` all green; pre-commit's `pytest (unit, fast)` hook PASSED → unit-test regression-clean.

## Summary

- **`src/fraud_engine/evaluation/shadow_compare.py`** (NEW, 581 LOC) — analysis primitive. The 110-line module docstring covers the 7 load-bearing decisions. Public surface: `ShadowComparison` class + `ComparisonReport` / `PromotionVerdict` / `EconomicCosts` dataclasses. Pure analytics — takes a DataFrame, returns structured metrics; no Postgres / parquet / file I/O. Reuses Sprint 4.1's cost matrix via an in-module `_per_row_costs` helper (so we get per-row arrays for bootstrap; `EconomicCostModel.compute_cost` returns aggregate stats). Bootstrap is in-module (~15 LOC) — no scipy dependency.
- **`scripts/shadow_compare_report.py`** (NEW, 400 LOC) — Click CLI with two modes: `--sample` (synthetic 1000 predictions, 200 labeled — the spec-validation invocation) and `--source` + `--labels` (production mode, currently a stubbed `NotImplementedError` since chargeback ingestion is Sprint 5.x scope). Renders a markdown report with the per-criterion verdict + cost-matrix reference.
- **`tests/unit/test_shadow_compare.py`** (NEW, 323 LOC) — 13 pure-unit tests across 6 scenario classes: agreement (3), correlation (2), cost (2), bootstrap (3), promotion verdict (2), input validation (1). All deterministic via fixed seed; sub-second total runtime.
- **`src/fraud_engine/evaluation/__init__.py`** (MODIFIED, +6 LOC) — re-export the four new public types.
- **No changes** to schemas / FeatureService / RedisFeatureStore / InferenceService / ShapExplainer / PredictionLogger / ShadowService / CircuitBreaker / Settings / `main.py` / Makefile / Dockerfile / docker-compose.yml / `CLAUDE.md` (§13 sprint-status update deferred to a 5.2.x audit-and-gap-fill PR per established convention).

## Spec vs. actual

| Spec line | Actual |
|---|---|
| Offline comparison: load N predictions with both scores | DataFrame-input contract; the module is data-source-agnostic per Decision 1 |
| Agreement rate | `_compute_agreement` — fraction where `champion_decision == shadow_decision`; population-level (no labels needed) |
| Score correlation | `_compute_correlation` — Pearson on `champion_score` × `shadow_score`; returns NaN on degenerate constant input |
| Economic cost on labeled subset | `_per_row_costs` mirrors Sprint 4.1's matrix; cost arrays computed only on rows with non-null `is_fraud` |
| Bootstrap significance | `_bootstrap_cost_diff` — 10K iterations by default, fixed seed=42 → deterministic; returns mean_diff + 95% CI + two-sided p-value |
| Weekly report | `scripts/shadow_compare_report.py` writes `reports/shadow_compare_<date>.md`; per-criterion verdict callout; cost-matrix reference table |
| Promotion criteria: cost > 2%, p < 0.05, agreement > 85% | `PromotionVerdict.promote = cost_improvement_pass AND p_value_pass AND agreement_pass`; reasons list explains each |
| `uv run pytest tests/unit/test_shadow_compare.py -v` | **13 passed in 2.09 s** |
| `uv run python scripts/shadow_compare_report.py --sample` | Exit 0; writes `reports/shadow_compare_2026-05-10.md` |

## Verbatim verification output

### Cheap gates

```
$ uv run ruff format src/fraud_engine/evaluation/shadow_compare.py \
                    src/fraud_engine/evaluation/__init__.py \
                    scripts/shadow_compare_report.py \
                    tests/unit/test_shadow_compare.py
4 files left unchanged

$ uv run ruff check ...
All checks passed!

$ uv run mypy src
Success: no issues found in 50 source files
```

### Spec verification: unit tests

```
$ uv run pytest tests/unit/test_shadow_compare.py -v --no-cov
collected 13 items

test_agreement_rate_perfect_match PASSED
test_agreement_rate_zero_match PASSED
test_agreement_rate_partial PASSED
test_score_correlation_perfect PASSED
test_score_correlation_anticorrelated PASSED
test_economic_cost_with_labels PASSED
test_economic_cost_without_labels_returns_none PASSED
test_bootstrap_significance_distinguishable PASSED
test_bootstrap_significance_indistinguishable PASSED
test_bootstrap_deterministic_under_fixed_seed PASSED
test_should_promote_all_criteria_met PASSED
test_should_promote_below_threshold PASSED
test_invalid_input_missing_columns PASSED

======================= 13 passed in 2.09 s =======================
```

### Spec verification: `--sample` script run

```
$ uv run python scripts/shadow_compare_report.py --sample

===== shadow comparison summary =====
  mode               : sample (synthetic)
  n_total            : 1,000
  n_labeled          : 200
  agreement_rate     : 0.9360
  score_correlation  : 0.9851
  cost_improvement   : -13.62%
  bootstrap p-value  : 0.0000
-------------------------------------
  VERDICT            : DO NOT PROMOTE — PASS: agreement_rate=0.9360 > 0.85 | FAIL: cost_improvement=-13.62% not > 2% | PASS: p_value=0.0000 < 0.05
-------------------------------------
  report written to  : reports/shadow_compare_2026-05-10.md
```

### Sample report excerpt (`reports/shadow_compare_2026-05-10.md`)

```markdown
# Shadow Comparison Report — champion vs challenger

**Generated:** 2026-05-10 19:03:55 UTC
**Mode:** `sample (synthetic)`
**N predictions:** 1,000
**N labeled (cost path):** 200

## Verdict

> ⛔ **DO NOT PROMOTE** — at least one criterion failed.

| Criterion | Threshold | Outcome |
|---|---|---|
| Agreement rate | > 0.85 | 0.9360 — PASS |
| Cost improvement | > 2% | -13.62% — FAIL |
| p-value (bootstrap, two-sided) | < 0.05 | 0.0000 — PASS |

## Distribution metrics (population-level)

| Metric | Value |
|---|---|
| Agreement rate | 0.9360 |
| Score correlation (Pearson) | 0.9851 |

## Cost analysis (labeled subset, N=200)

| Metric | Value |
|---|---|
| Champion cost per txn | $19.2750 |
| Shadow cost per txn | $21.9000 |
| Relative improvement | -13.62% |
| Bootstrap mean diff (champion - shadow) | $-2.6230 |
| Bootstrap 95% CI | ($-4.0250, $-1.4000) |
| Bootstrap two-sided p-value | 0.0000 |
```

The synthetic data is tuned to produce a boundary-case verdict: high agreement (0.936) and high correlation (0.985) — but the noisy challenger pushes a few extra non-fraud predictions over the 0.080 threshold, raising shadow cost by 13.62% and producing a statistically-significant negative cost improvement. The verdict correctly reads "DO NOT PROMOTE" with two PASS reasons (agreement, p-value) and one FAIL (cost), demonstrating the per-criterion reporting format.

## Test inventory

`tests/unit/test_shadow_compare.py` (NEW, 13 tests, 2.09 s):

| # | Test | What it asserts |
|---|---|---|
| 1 | `test_agreement_rate_perfect_match` | All decisions identical → agreement = 1.0 |
| 2 | `test_agreement_rate_zero_match` | All decisions opposite → agreement = 0.0 |
| 3 | `test_agreement_rate_partial` | 9/10 match → agreement = 0.9 |
| 4 | `test_score_correlation_perfect` | shadow == champion → correlation = 1.0 |
| 5 | `test_score_correlation_anticorrelated` | shadow == 1 - champion → correlation = -1.0 |
| 6 | `test_economic_cost_with_labels` | Per-row cost matches manual calc; champion=$122.50/txn, shadow=$2.50/txn, improvement=97.96% |
| 7 | `test_economic_cost_without_labels_returns_none` | Missing `is_fraud` column → cost fields all None; verdict short-circuits to False |
| 8 | `test_bootstrap_significance_distinguishable` | Shadow much cheaper → mean_diff > 0; p < 0.05 |
| 9 | `test_bootstrap_significance_indistinguishable` | Shadow ≈ champion → p > 0.05 (not significant) |
| 10 | `test_bootstrap_deterministic_under_fixed_seed` | Two runs with seed=42 → identical mean_diff, CI, and p-value |
| 11 | `test_should_promote_all_criteria_met` | improvement>2%, p<0.05, agreement>85% → promote=True; all three reasons start with PASS |
| 12 | `test_should_promote_below_threshold` | At least one criterion fails → promote=False; reasons explain WHY |
| 13 | `test_invalid_input_missing_columns` | Missing required columns → ValueError |

## Files-changed table

| File | Change | LOC |
|---|---|---|
| `src/fraud_engine/evaluation/shadow_compare.py` | NEW — `ShadowComparison` + 3 dataclasses + bootstrap helper + per-row cost helper + input validation | +581 |
| `src/fraud_engine/evaluation/__init__.py` | MODIFIED — re-export ComparisonReport, EconomicCosts, PromotionVerdict, ShadowComparison | +6 |
| `scripts/shadow_compare_report.py` | NEW — Click CLI; `--sample` synthesises data; `--source`/`--labels` stubbed for Sprint 5.x; markdown report writer | +400 |
| `tests/unit/test_shadow_compare.py` | NEW — 13 pure-unit tests across 6 scenario classes | +323 |
| `sprints/sprint_5/prompt_5_2_c_report.md` | this file | (this file) |

**No changes** to schemas / FeatureService / RedisFeatureStore / InferenceService / ShapExplainer / PredictionLogger / ShadowService / CircuitBreaker / Settings / `main.py` / Makefile / Dockerfile / docker-compose.yml / `CLAUDE.md`.

## Decisions worth flagging

1. **DataFrame-input contract; the module is data-source-agnostic.** `ShadowComparison(predictions: pd.DataFrame, costs: EconomicCosts)` takes a DataFrame with required columns (request_id, champion_score, shadow_score, champion_decision, shadow_decision) + optional `is_fraud`. The module does NO data loading itself — that's the script's job. Mirrors `evaluation/economic.py` (doesn't load parquets) vs `scripts/run_economic_evaluation.py` (does). Tests stay trivially testable; production data ingestion (Postgres + structlog JSONL + parquet labels) is the script's concern.

2. **`EconomicCosts` dataclass mirroring Settings's cost fields** (no `tn_cost` — TN is always free per Sprint 4.1's matrix). The comparison module doesn't import `Settings`; the CLI script constructs the dataclass from `Settings`. Makes test-side cost overrides trivial.

3. **In-module bootstrap, NOT scipy.stats.bootstrap.** ~15 LOC for the standard non-parametric resample-with-replacement loop. Deterministic via fixed `seed=42`. No new dependencies. Inlining keeps the logic visible; parametric tests (t-test, Wilcoxon) were rejected because fraud-cost data violates their distributional assumptions.

4. **Population-level agreement + correlation; subset-level cost.** Agreement and correlation are properties of the scoring distributions and need no labels. Cost requires labels (production chargebacks ~30-60 days delayed; or backtest labels from `tier5_test.parquet`). Splitting metrics this way means the report is informative even when the labeled subset is tiny (typical real-world chargeback rate is ~5%).

5. **`PromotionVerdict` carries per-criterion outcomes, not just a bool.** Each of the three criteria has its own `*_pass` flag + a human-readable `reason` string starting with "PASS:" or "FAIL:". The `summary` property renders a compact line for the markdown report. An analyst can act on "FAIL: agreement_rate=0.78 below 0.85 threshold" instead of staring at `promote=False`.

6. **Sample mode generates synthetic data with controllable correlation.** 1000 predictions; champion uniform on [0, 1]; shadow = 0.85·champion + 0.15·noise (correlation ~0.85); decisions derived from `Settings.decision_threshold` (0.080); 200 of 1000 labeled with `is_fraud` ~ sigmoid(score). Tuned so the verdict has at least one PASS and one FAIL criterion — useful demo of the per-criterion reporting.

7. **CLI emits markdown to `reports/shadow_compare_<date>.md`.** Standard project convention (mirrors `run_economic_evaluation.py`). Stdout gets a compact summary. `reports/` is gitignored per CLAUDE.md so iterating doesn't pollute git.

8. **Tests are pure-unit (no Postgres, no file I/O).** All 13 tests in `tests/unit/test_shadow_compare.py`. Build small DataFrames inline; assert outputs. Bootstrap determinism via fixed seed makes assertions exact (no flakiness). Runs in `make test-fast`.

9. **Backtest-vs-production label sourcing deferred to Sprint 5.x.** Production label flow (chargeback events) is out of scope. The `--sample` mode + a stubbed `--source` / `--labels` argument cover the immediate analytical surface. The script raises `NotImplementedError` on the production path with a clear "Sprint 5.x scope" message.

## Surprising findings

1. **The `--sample` synthetic data produces a "DO NOT PROMOTE" verdict — and that's the right answer.** Champion scores are uniform; shadow is 0.85·champion + 0.15·noise. With the cost-optimal threshold at 0.080, ~92% of predictions get blocked. The shadow's noise pushes a few extra non-fraud predictions over the threshold, increasing false-positive cost. Net effect: shadow is 13.62% MORE expensive than champion (cost_improvement = -13.62%). The bootstrap p-value confirms this is statistically significant (p=0.0000). Two of three promotion criteria pass (agreement 0.936 > 0.85; p-value < 0.05); cost_improvement fails. Demonstrates the per-criterion verdict format works correctly: an honestly-noisy challenger isn't an improvement, and the report explains why.

2. **`ShadowComparison.run` takes ~80 ms with default 10K bootstrap iterations on 200 labeled rows.** Per-iteration cost is dominated by `np.random.default_rng(...).integers(0, n, n)` (~5 µs) + array indexing. 10K iterations × 8 µs each ≈ 80 ms — comfortably fast for an offline weekly report. If a future use case needs faster bootstrap, vectorising the loop (single `(n_iter, n)` index matrix) would cut this to ~20 ms.

3. **The cost matrix in `_per_row_costs` mirrors Sprint 4.1's `EconomicCostModel.compute_cost` exactly.** Verified by test #6: a 4-row hand-built DataFrame with one row per (decision, label) combo produces costs that match a paper-and-pencil calculation. Decoupling per-row computation from the existing `EconomicCostModel` (which only returns aggregate stats) avoids modifying Sprint 4.1's module while keeping the cost semantics identical.

4. **Pandas' `nunique` raises a warning on the constant-column edge case** unless we check explicitly. Added `_MIN_UNIQUE_FOR_CORR = 2` constant + an explicit `nunique < 2` check before calling `.corr()`. Returns `float("nan")` on degenerate input rather than letting pandas print a `RuntimeWarning`.

5. **`PromotionVerdict.summary` is the compact line for both stdout AND the markdown report's verdict section.** Re-using the same string keeps the operator-facing output consistent across surfaces. The structured `reasons: list[str]` allows the markdown report to render each criterion as a separate row in the verdict table.

## Out of scope (Sprint 5.2.x+ / Sprint 5.x)

- **Production data join** (Postgres `predictions` + `logs/shadow.jsonl`) — script supports `--source` + `--labels` arguments but the actual joining code path is stubbed. Real join lands when chargeback ingestion exists.
- **Real chargeback / dispute label table** — Sprint 5.x.
- **Auto-running the report on a cron** — Sprint 6 ops; this script is the building block.
- **Per-segment shadow comparison** (per ProductCD / DeviceType / etc.) — could reuse `StratifiedEvaluator` from Sprint 4.2; defer to Sprint 5.x.
- **A `/admin/shadow-compare` HTTP endpoint** — Sprint 5.x ops surface.
- **Calibration-curve comparison** (champion vs challenger reliability diagrams) — Sprint 5.x model-monitoring work.
- **Drift detection on shadow scores** (PSI between champion + shadow score distributions) — Sprint 6 monitoring work.
- **Promotion automation** (auto-promote challenger when criteria met for K weeks running) — Sprint 6 ops.
- **CLAUDE.md §13 sprint-status update** — defer to a 5.2.x audit-and-gap-fill PR (matches established convention).
- **Sliding-window agreement / cost trends** — defer; weekly snapshot is the spec ask.
- **Multi-challenger comparison** (Model B + Model C side-by-side) — defer to Sprint 5.x; 5.2.c is bilateral.
- **Vectorising the bootstrap loop** — current ~80 ms per run is fast enough for offline analysis; defer.

## Acceptance checklist

- [x] Branch `sprint-5/prompt-5-2-c-shadow-compare` off `main` (`dda4044`, post 5.2.b merge)
- [x] `src/fraud_engine/evaluation/shadow_compare.py` created (581 LOC; ShadowComparison + 3 dataclasses + bootstrap + per-row cost + input validation)
- [x] `src/fraud_engine/evaluation/__init__.py` re-exports the four new public types
- [x] `scripts/shadow_compare_report.py` created (400 LOC; Click CLI; --sample + production-stub; markdown report writer)
- [x] `tests/unit/test_shadow_compare.py` created (323 LOC; 13 pure-unit tests across 6 scenario classes)
- [x] Spec gate: offline comparison loads N predictions with both scores — PASS (DataFrame contract)
- [x] Spec gate: agreement rate, score correlation — PASS (population-level)
- [x] Spec gate: economic cost on labeled subset — PASS (subset-level, mirrors Sprint 4.1 cost matrix)
- [x] Spec gate: bootstrap significance — PASS (10K iter, deterministic, two-sided p)
- [x] Spec gate: weekly report — PASS (`reports/shadow_compare_<date>.md`)
- [x] Spec gate: promotion criteria (cost > 2%, p < 0.05, agreement > 85%) — PASS (PromotionVerdict + per-criterion reasons)
- [x] `uv run pytest tests/unit/test_shadow_compare.py -v` returns 0 (13 passed in 2.09 s)
- [x] `uv run python scripts/shadow_compare_report.py --sample` returns 0 + writes valid markdown report
- [x] `make format` returns 0
- [x] `make lint` returns 0 (All checks passed!)
- [x] `make typecheck` returns 0 (Success: no issues found in 50 source files)
- [x] All 12 pre-commit hooks pass on the touched files (incl. `pytest (unit, fast)` → regression-clean)
- [x] `sprints/sprint_5/prompt_5_2_c_report.md` written
- [x] No git/gh commands run beyond §2.1 carve-out

Verification passed. Ready for John to commit on `sprint-5/prompt-5-2-c-shadow-compare`.

**Commit note:**
```
5.2.c: Shadow comparison report — `ShadowComparison` (DataFrame-input; agreement, correlation, per-row cost mirroring Sprint 4.1, in-module bootstrap with deterministic seed); `PromotionVerdict` carries per-criterion pass/fail + reasons (cost>2%, p<0.05, agreement>85%); `scripts/shadow_compare_report.py` Click CLI with --sample mode renders weekly markdown report; 13 unit tests pass in 2.09s; production data join (Postgres + JSONL + parquet labels) stubbed for Sprint 5.x
```
