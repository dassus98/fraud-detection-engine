# Sprint 4 — Prompt 4.4: full economic evaluation on test set

**Date:** 2026-05-09
**Branch:** `sprint-4/prompt-4-4-run-economic-evaluation` (off `main` @ `9ebcf6c` — post 4.3 merge)
**Status:** Verification passed; **2 of 3 spec gates met**, optimum-band gate honestly documented as a finding (not a bug).

## Headline

| Acceptance gate | Realised | Status |
|---|---|---|
| Optimal τ in [0.3, 0.5] given default costs | **0.0800** | ❌ GAP — but matches the analytical Bayes limit τ* ≈ 0.073 (see "Surprising findings") |
| Annual savings ≥ $500K on 1M/month portfolio | **$28,957,772.08** | ✅ PASS — 58× the floor |
| Sensitivity stable within ±20% cost variation (spread < 0.20) | **0.0600** | ✅ PASS — well under the ceiling |

3 deliverables produced + 1 gap-fix to a 4.2 unit test that was implicitly coupled to `Settings.decision_threshold`. Cheap gates green; integration test 27/27 pass; full unit-test regression at **522 passed**.

## Summary

- **`scripts/run_economic_evaluation.py`** (NEW, ~880 LOC) — Click CLI orchestrator that loads Model A + the calibrator (`models/sprint3/`), scores the held-out test parquet (`tier5_test.parquet`, 92,427 rows), runs `EconomicCostModel.optimize_threshold` + `sensitivity_analysis` (Sprint 4.1 surface), runs `StratifiedEvaluator.evaluate` + `plot_heatmap` (Sprint 4.2 surface), computes annual savings via linear extrapolation (per-txn delta × monthly volume × 12), writes the markdown report, saves the cost-curve PNG and stratified-heatmap PNG, and updates `.env`'s `DECISION_THRESHOLD` in place with a `.env.bak` backup. Mirrors `train_lightgbm.py`'s structure (Click CLI with `--quick`, `Final[type]` constants block, `_render_*_report` builder). `--dry-run` and `--no-update-env` are explicit escape hatches for the `.env` mutation.
- **`tests/integration/test_run_economic_evaluation.py`** (NEW, ~480 LOC) ships 27 tests across 6 contract surfaces (TestRunSmoke, TestOutputFiles, TestEnvUpdate, TestUpdateEnvThresholdHelper, TestAcceptanceGates, TestErrorHandling). Module-scoped fixture (lessons from the 3.3.d / 3.4.a fixture-scope audit) so the smoke pipeline runs once.
- **`reports/economic_evaluation.md`** (NEW, auto-generated) — 11-section report carrying the optimal threshold, baseline comparison, annual savings table, sensitivity top/bottom-5, full stratified table, both figure embeds, `.env` update log, the spec-mandated **Caveats** section, artefacts paths, and references.
- **`reports/figures/economic_cost_curve.png` and `economic_stratified_heatmap.png`** (NEW, auto-generated). Cost curve: cost-per-txn vs threshold sweep with optimal τ (red dashed) and baseline τ=0.5 (green dotted) marked. Heatmap: Sprint 4.2's z-score-normalised per-segment view.
- **`.env` updated**: `DECISION_THRESHOLD=0.5` → `DECISION_THRESHOLD=0.080000`. `.env.bak` carries the prior placeholder for rollback.
- **One real gap-fix to `tests/unit/test_stratified.py::test_month_with_drift_has_higher_cost_per_txn`** — see "Gap-fix" section below.

## Spec vs. actual

| Spec line | Actual |
|---|---|
| `scripts/run_economic_evaluation.py` produces optimal threshold, sensitivity table, stratified performance table, annual savings estimate, caveats | All five present in `reports/economic_evaluation.md`. Optimal τ = 0.0800; sensitivity table renders top-5 / bottom-5 of the 125-cell grid; stratified table is the long-format DataFrame from `StratifiedEvaluator.evaluate`; annual savings = $28,957,772 on 1M/month default; Caveats section is the 7-bullet block enumerating cost-value uncertainty, snapshot-test, portfolio default, calibration dependency, per-segment opportunity, month-axis decision, and savings-vs-model-value framing. |
| Update `.env` with `DECISION_THRESHOLD=<optimal>` | `.env`'s `DECISION_THRESHOLD=0.5` → `DECISION_THRESHOLD=0.080000`. `.env.bak` backup written first. Logged in the report's `.env update` section with rollback command (`mv .env.bak .env`). |
| Document `.env` update in the report | Section 8 of `reports/economic_evaluation.md` carries the explicit log: "`DECISION_THRESHOLD` updated `0.50` → `0.080000` in `/home/.../`.env` on 2026-05-09. Backup: `/home/.../.env.bak`. To roll back: `mv .env.bak .env`." |
| Acceptance: Optimal threshold in [0.3, 0.5] given default costs | **GAP — realised 0.0800.** Analytical Bayes limit at default costs is τ* ≈ 35/(35 + 450 − 5) ≈ 0.073; the empirical 0.08 matches it almost exactly (one threshold-grid step away). The spec's [0.3, 0.5] expectation was based on a different intuition. ADR 0003's "Why minimum-expected-cost τ" subsection already documents this gap: "the empirical optimum lands ABOVE the analytical limit because the swept grid operates on finite-sample empirical FN/FP rates rather than the theoretical decision boundary" — but on calibrated test data the gap is small. |
| Acceptance: Annual savings ≥ $500K/year on 1M/month portfolio | **PASS — $28.96M / year.** ~58× the floor. Per-txn delta is $2.41 (cost @ τ=0.5 = $11.87, cost @ τ=0.08 = $9.46). |
| Acceptance: Sensitivity shows robustness (τ* stable within ±20% cost variation) | **PASS — spread = 0.0600** (max τ* − min τ* across the 125-cell grid). Well under the 0.20 ceiling. |
| Verification: `uv run python scripts/run_economic_evaluation.py` returns 0 | Script returned 0; full-test-set scoring took ~5 s; report + figures + `.env` mutation all completed. |
| Verification: `cat reports/economic_evaluation.md` shows the report | Report renders with all 11 sections + tables + image embeds. |
| Verification: `grep DECISION_THRESHOLD .env` shows the new value | `DECISION_THRESHOLD=0.080000` |

## Test inventory

### Integration: `tests/integration/test_run_economic_evaluation.py` (NEW, 27 tests in 10.01 s)

| Class | Count | Coverage |
|---|---|---|
| `TestRunSmoke` | 6 | `EvaluationResult` populated; optimal τ ∈ (0, 1); baseline = 0.5; annual savings positive; n_test_rows ≤ smoke cap; portfolio_monthly recorded. |
| `TestOutputFiles` | 5 | Markdown report + cost-curve PNG + heatmap PNG all written + non-empty; report carries the 9 required sections; report includes the realised optimal τ value. |
| `TestEnvUpdate` | 5 | `update_env=True` mutates fake `.env` + writes `.bak`; `.bak` carries prior 0.5; new value matches optimum within 1e-5; `dry_run=True` leaves `.env` untouched + no `.bak`; `update_env=False` (CLI's `--no-update-env`) same. |
| `TestUpdateEnvThresholdHelper` | 4 | Replaces existing `DECISION_THRESHOLD=` line; preserves surrounding lines; appends with provenance comment when key missing; missing `.env` raises `FileNotFoundError`; trailing newline preserved. |
| `TestAcceptanceGates` | 6 | Sensitivity DataFrame shape (125, 6) + columns; cost curve shape (99, 7); stratified emits at least amount_bucket axis; gate-pass fields are bool not numpy-bool; annual_cost = cost_per_txn × annual_volume arithmetic identity. |
| `TestErrorHandling` | 1 | Pointing at non-existent `.env` raises with helpful message. |

### Unit-test regression: 522 passed (matches post-4.3 baseline)

No new unit tests in 4.4. The gap-fix to `test_stratified.py::test_month_with_drift_has_higher_cost_per_txn` is described in the "Gap-fix" section below — pinning `threshold=0.5` explicitly so the test is environment-independent.

## Files-changed table

| File | Change | LOC |
|---|---|---|
| `scripts/run_economic_evaluation.py` | new (Click CLI; `EvaluationResult` dataclass; `run_economic_evaluation` orchestrator; `_render_economic_evaluation_report` builder; `_save_cost_curve_figure` helper; `_update_env_threshold` helper) | ~880 |
| `tests/integration/test_run_economic_evaluation.py` | new (27 tests across 6 classes; module-scoped fixture; fake-`.env` smoke pattern) | ~480 |
| `tests/unit/test_stratified.py` | **modify** — pin `threshold=0.5` in `test_month_with_drift_has_higher_cost_per_txn`; add docstring explaining the env-independence rationale | +12 / -1 |
| `reports/economic_evaluation.md` | new (auto-generated) | ~120 |
| `reports/figures/economic_cost_curve.png` | new (auto-generated; 70 KB) | (binary) |
| `reports/figures/economic_stratified_heatmap.png` | new (auto-generated; 210 KB) | (binary) |
| `.env` | one-line edit (`DECISION_THRESHOLD=0.5` → `0.080000`) | (gitignored) |
| `.env.bak` | new backup | (gitignored) |
| `sprints/sprint_4/prompt_4_4_report.md` | this file | ~280 |

**No changes** to `economic.py`, `stratified.py`, `calibration.py`, `lightgbm_model.py`, `Settings`, or any other src module.

## Decisions worth flagging

1. **Script loads existing artefacts; does NOT retrain.** The 4.4 spec is "evaluate on held-out test", not "re-train". `LightGBMFraudModel.load(models_dir)` is the contract; if the model joblib is stale, that's a `train_lightgbm.py` re-run concern, not 4.4's. The report's "Artefacts" section cites the producer script for both model and calibrator so a reviewer can trace lineage.

2. **Calibrated test scores via `calibrator.transform(...)`** — the production-realistic flow. Uncalibrated probabilities would give a wrong cost surface (ADR 0003: calibration is the load-bearing dependency for the Bayes-decision argument). The script logs the score-range bounds (`proba_min=0.0, proba_max=1.0`) at INFO so a reviewer can verify calibration produced valid probabilities.

3. **Annual-savings baseline = τ = 0.5** (the placeholder this script replaces). Alternative baselines (no-model = 100% FN cost; F1-optimal τ; "block everything") answer different questions ("model value" vs "threshold-optimisation value"). The Caveats section frames this explicitly so a reviewer doesn't mis-read the savings number as the model's total contribution.

4. **`.env` mutated in place with `.env.bak` backup.** Settings reads `.env`; the chosen τ has to land there or Sprint 5 won't see it. The backup is the rollback path. `--dry-run` and `--no-update-env` are escape hatches. The script logs both paths so a reviewer can `mv .env.bak .env` if needed.

5. **Month axis skipped on test (`month=None`).** Tier-5 parquet drops `timestamp` (`build_features_all_tiers.py:110-111`); the test set is approximately one calendar month per the temporal split. Within-test month stratification would be degenerate; cross-month drift is Sprint 6 monitoring territory. Documented in the report's Caveats.

6. **Cost-curve PNG and heatmap PNG both committed** to `reports/figures/` (not gitignored). The PNGs are the durable artefact a reviewer reads when the markdown image links go stale. ~280 KB combined; trivial.

7. **Annual savings is a linear extrapolation** (per-txn delta × monthly volume × 12). The Caveats section warns: "Scale linearly for a different portfolio volume." Premature precision via Monte Carlo or per-segment volumes would be over-engineering for a cost-curve evaluation.

8. **Three acceptance gates reported, NOT enforced** at the script level. Like `train_lightgbm.py`, the script runs to completion regardless of which gates pass. The integration test enforces only the catastrophic floor (annual_savings > 0). Spec gates are not asserted on the 5K smoke (too noisy to pin); they're inspection artefacts in the report.

9. **PR's `.env` change is gitignored** so the PR diff doesn't carry it, but the report's "`.env` update" section is the durable audit trail. A reviewer reading the PR sees the `.env` mutation explicitly through the report, not through a missing diff.

10. **Gap-fix to `test_stratified.py`** is a real find: 4.2's `test_month_with_drift_has_higher_cost_per_txn` was implicitly coupled to `Settings.decision_threshold = 0.5` via `StratifiedEvaluator()` no-args default. After 4.4 mutates `.env`, the default became 0.08, which collapsed the noise-vs-clean cost discrimination the test asserted. Pinning `threshold=0.5` explicitly in that one test makes it environment-independent. Other `StratifiedEvaluator()` no-args tests in `test_stratified.py` aren't threshold-sensitive (they assert AUC, fraud_rate, or shape — all threshold-free), so the fix is targeted to the one affected test.

## Surprising findings

1. **Optimal τ = 0.08 matches the analytical Bayes-decision limit almost exactly.** ADR 0003's "Why minimum-expected-cost τ" subsection derives the analytical limit:

   ```
   τ* = fp_cost / (fp_cost + fraud_cost − tp_cost)
      = 35 / (35 + 450 − 5) ≈ 0.0729
   ```

   The empirical optimum on the 92K test set is 0.0800 — one grid step (0.01) above the theoretical 0.0729, well within the swept grid's resolution. **This is calibration + Bayes-theory working as designed.** The spec's [0.3, 0.5] expectation reflects an intuition (probably about uncalibrated probabilities or different cost ratios) that doesn't match the realised math. Not a bug — a sign the calibration is solid and the cost ratio (13×) is doing what the theory says it should.

   This finding doesn't fail the prompt; it shows the prompt is doing what it should. The completion report flags the gate-band miss honestly so a reviewer doesn't ignore the discrepancy, but the deeper conclusion is that the analytical and empirical optima agree, which is the strongest possible validation that the cost-based optimisation is correctly implemented.

2. **Annual savings of $28.96M is ~58× the spec's $500K floor.** Per-txn savings are $2.41 ($11.87 baseline → $9.46 optimal). At 1M txns/month × 12 months = 12M annual transactions, that's $28.96M. Even at a more modest 100K-txn/month deployment (`--quick` portfolio default), annual savings would be ~$2.9M. **The threshold optimisation is materially valuable**, not a marginal improvement. A senior reviewer should anchor on this: the "right τ" question matters in dollars, not basis points.

3. **Sensitivity spread = 0.06** — well under the 0.20 ceiling. Across the 125-cell ±20% grid, τ* ranges from 0.04 to 0.10. The cost-input uncertainty band (±20% on each axis) doesn't shift τ* meaningfully; the optimum is genuinely robust. CLAUDE.md §8's stability rule holds on real Model A test data.

4. **`Settings.decision_threshold`-coupled test in 4.2.** Sprint 4.2's `test_month_with_drift_has_higher_cost_per_txn` was using `StratifiedEvaluator()` with no explicit threshold, which makes the test depend on whatever value `Settings.decision_threshold` happens to hold. Before 4.4 that was 0.5 (the placeholder); after 4.4 mutates `.env` it's 0.08. The test was tuned for 0.5 and broke at 0.08. The gap-fix (pin `threshold=0.5` explicitly in that one test) is a small change; the deeper insight is that **any test reading Settings defaults is implicitly coupled to the env state**. The other `StratifiedEvaluator()` no-args tests in `test_stratified.py` happen to assert threshold-free metrics (AUC, PR-AUC, fraud_rate, shape), so they're robust to the env change. Sprint 5+ should consider a project-wide pattern of pinning thresholds explicitly in tests.

## Gap-fix (discovered during 4.4 verification)

**Issue:** `tests/unit/test_stratified.py::TestPerAxisLogic::test_month_with_drift_has_higher_cost_per_txn` failed after `.env`'s `DECISION_THRESHOLD` was mutated from 0.5 → 0.08 by `run_economic_evaluation.py`.

**Root cause:** The test instantiated `StratifiedEvaluator()` with no explicit `threshold`, which means the evaluator picked up `Settings.decision_threshold` at construction time — i.e. whatever value `.env` happened to hold. The test's assertion (`month=5` cost > `month=6` cost) was tuned for τ=0.5; at τ=0.08 most predictions on both noisy and clean strata become positive, which collapses the cost discrimination.

**Fix:** Pin `threshold=0.5` explicitly in the failing test:

```python
# Before:
out = StratifiedEvaluator().evaluate(y, s, frame, month=month)

# After:
out = StratifiedEvaluator(threshold=0.5).evaluate(y, s, frame, month=month)
```

Plus a docstring paragraph explaining the env-independence rationale and the Sprint 4.4 link.

**Why not fix all `StratifiedEvaluator()` no-args calls in test_stratified.py?** Audited each: only this one asserts on `cost_per_txn` (threshold-sensitive). All others assert on AUC, PR-AUC, fraud_rate, n_rows, or shape — threshold-free. Targeted fix; the wider env-independence pattern is a Sprint 5+ test-discipline question.

**Verified:** Unit-test regression passes (522 passed; matches post-4.3 baseline).

## Verbatim verification output

### Cheap gates

```
$ uv run ruff format scripts/run_economic_evaluation.py tests/integration/test_run_economic_evaluation.py
2 files reformatted

$ uv run ruff check scripts/run_economic_evaluation.py tests/integration/test_run_economic_evaluation.py
All checks passed!

$ uv run mypy src
Success: no issues found in 40 source files

$ uv run mypy scripts/run_economic_evaluation.py
Success: no issues found in 1 source file
```

### Integration test

```
$ uv run pytest tests/integration/test_run_economic_evaluation.py -v --no-cov
======================= 27 passed, 14 warnings in 10.01s =======================
```

### Unit-test regression

```
$ uv run pytest tests/unit -q --no-cov
522 passed, 34 warnings in 69.63s (0:01:09)
```

(Matches post-4.3 baseline; 1 pre-existing test fixed via gap-fix above.)

### Spec verification — script + report + .env

```
$ uv run python scripts/run_economic_evaluation.py
[~5 s wall-time on full 92K test set]

run_economic_evaluation: COMPLETE
  optimal τ          = 0.0800  (GAP vs [0.3, 0.5])
  baseline τ         = 0.50
  cost / txn         = $9.46 (baseline $11.87)
  annual savings     = $28,957,772.08  (PASS vs $500,000)
  sensitivity spread = 0.0600  (PASS vs <0.2)
  n_test_rows        = 92,427
  portfolio monthly  = 1,000,000
  report             = /home/dchit/projects/fraud-detection-engine/reports/economic_evaluation.md
  cost-curve figure  = /home/dchit/projects/fraud-detection-engine/reports/figures/economic_cost_curve.png
  heatmap figure     = /home/dchit/projects/fraud-detection-engine/reports/figures/economic_stratified_heatmap.png
  .env updated       = /home/dchit/projects/fraud-detection-engine/.env
  .env backup        = /home/dchit/projects/fraud-detection-engine/.env.bak

$ grep DECISION_THRESHOLD .env
DECISION_THRESHOLD=0.080000

$ grep DECISION_THRESHOLD .env.bak
DECISION_THRESHOLD=0.5
```

## Out of scope (Sprint 4.x+ / 5+)

- **Per-segment thresholds.** The stratified table likely shows segments where the global τ underperforms; per-segment τ is Sprint 5 territory. The heatmap PNG is the visualisation a Sprint 5 prompt will read from.
- **MLflow logging of optimal τ + cost curve** as model-run metadata. Sprint 4.x+ / Sprint 5.
- **Drift detection on the chosen τ.** Sprint 6 monitoring stack — re-run economic evaluation on production data, compare to this report's optimum, alert if spread exceeds the sensitivity-grid bound.
- **Re-training Model A on full data with cost-aware loss.** This script worked with a fixed Model A; cost-aware retraining is a future experiment.
- **Project-wide audit of `Settings.*` no-default tests.** The 4.2 test fixed here is one instance; other tests may be coupled to env state in non-obvious ways. Sprint 5+ test-discipline pass.
- **Updating CLAUDE.md §13 sprint status table** (per CONTRIBUTING.md §4: handled in the next sprint's first PR).
- **Extending `make typecheck` to cover `scripts/`** (fourth-time-cited Sprint 6 follow-on; the Sprint 3 audit + 4.4 both confirm the gap).

## Acceptance checklist

- [x] Branch `sprint-4/prompt-4-4-run-economic-evaluation` off `main` (`9ebcf6c`, post 4.3 merge)
- [x] `scripts/run_economic_evaluation.py` created (~880 LOC; Click CLI; `EvaluationResult` dataclass; `run_economic_evaluation` orchestrator; `.env` mutation helper; cost-curve plot; report builder)
- [x] `tests/integration/test_run_economic_evaluation.py` created (27 tests across 6 classes; module-scoped fixture; fake-`.env` pattern)
- [x] `reports/economic_evaluation.md` generated by the script (11 sections per spec)
- [x] `reports/figures/economic_cost_curve.png` + `economic_stratified_heatmap.png` generated
- [x] `.env`'s `DECISION_THRESHOLD` updated `0.5` → `0.080000`; `.env.bak` carries prior value
- [x] Spec gate: optimal τ in [0.3, 0.5] — **GAP (realised 0.0800; matches Bayes limit 0.073)** — honestly documented
- [x] Spec gate: annual savings ≥ $500K — **PASS ($28.96M; 58× the floor)**
- [x] Spec gate: sensitivity stable < 0.20 — **PASS (spread = 0.06)**
- [x] Gap-fix: `test_stratified.py::test_month_with_drift_has_higher_cost_per_txn` pinned to `threshold=0.5` (env-independent)
- [x] `make format && make lint && make typecheck` all return 0 (src + new script)
- [x] `uv run pytest tests/integration/test_run_economic_evaluation.py -v` returns 0 (27 passed in 10 s)
- [x] `uv run pytest tests/unit -q` returns 0 (522 passed; matches post-4.3 baseline; gap-fix applied)
- [x] `uv run python scripts/run_economic_evaluation.py` returns 0 (full 92K test set, ~5 s wall)
- [x] `cat reports/economic_evaluation.md` shows the rendered report
- [x] `grep DECISION_THRESHOLD .env` shows `0.080000`
- [x] `sprints/sprint_4/prompt_4_4_report.md` written
- [x] No git/gh commands run beyond §2.1 carve-out (branch create only; commit + push + merge happen after John's sign-off)

Verification passed. Ready for John to commit on `sprint-4/prompt-4-4-run-economic-evaluation`.

**Commit note:**
```
4.4: run_economic_evaluation script + test-set economic eval report + .env DECISION_THRESHOLD update
```
