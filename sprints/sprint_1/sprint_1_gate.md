# Sprint 1 Gate Report

**Date:** 2026-04-27
**Branch:** `sprint-1/sprint-gate` (off `main` at `9e29fa5`, post-1.3)
**Status:** **GREEN — all gates passed.** One finding worth flagging: the full-dataset temporal-split baseline AUC (0.9247) is **above** the spec's stated 0.88–0.91 expected range by ~0.015. This is "better than expected", not a regression — the LightGBM-on-raw baseline performs more strongly than the prompt's planning estimate predicted. The runner is the canonical place this is observed; integration-test bounds and Sprint 4's economic-cost work are unaffected.

## Sprint 1 prompt history

| Prompt | Title | PR | Squash commit |
|---|---|---|---|
| 1.1.a–c | EDA scaffolding + missingness + temporal/cleanlab summaries | #1, #2, #3 | (multiple) |
| 1.2.a | Audit-and-affirm temporal split surface | #4 | `443df26` |
| 1.2.b | TransactionCleaner + InterimTransactionSchema | #5 | `aa440f8` |
| 1.2.c | Lineage tracking primitive | #6 | `9bc35d5` |
| 1.2.d | Interim build pipeline + lineage verifier | #7 | `eeba7f4` |
| 1.3 | Baseline metric surface (AUC-PR + log loss, train + val) + reload test | #8 | `9e29fa5` |

`main` advances strictly through squash-merges of feature branches. No direct commits.

## Verification results

### 1. `make lint`
```
uv run ruff check src tests scripts
All checks passed!
```
Status: **GREEN**.

### 2. `make typecheck`
```
uv run mypy src
Success: no issues found in 23 source files
```
Status: **GREEN**. Source-file count has been stable since 1.2.c (lineage module added 1 file; 1.2.d/1.3 added scripts and modified models, both already counted).

### 3. `make test` (full suite via §17 detached-daemon)
```
================= 233 passed, 40 warnings in 253.74s (0:04:13) =================
========================= 2 passed in 72.56s (0:01:12) =========================
```
Status: **GREEN — 235 tests passed total.** First line is the pytest invocation (unit + integration + lineage); second line is the `make nb-test` step that follows (`nbmake` smoke tests on the two committable notebooks `notebooks/01_eda.ipynb` and `notebooks/00_observability_demo.ipynb`).

### 4. `uv run python scripts/verify_lineage.py`
```
{"run_id": "4e75a21749ed4d93ac5926f50d29e326", "event": "verify.start", ...}
Lineage verification: GREEN
  run_id: 4e75a21749ed4d93ac5926f50d29e326
  steps:  5 (load_merged, interim_clean, split_train, split_val, split_test)
{"run_id": "4e75a21749ed4d93ac5926f50d29e326", "n_steps": 5, "event": "verify.passed", ...}
```
Status: **GREEN**. The verifier checks the most-recent `lineage.jsonl` (run from 1.2.d's `build_interim`) against six contracts: every expected step present, `load.output_rows == clean.input_rows`, cleaner never invents rows, `sum(splits.output_rows) == clean.output_rows`, each parquet's row count matches its split-step record, manifest `schema_version` matches the pinned literal. All six pass.

### 5. `uv run jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output /tmp/01_done.ipynb`
```
[NbConvertApp] Writing 117055 bytes to /tmp/01_done.ipynb
```
Status: **GREEN**. Notebook executes end-to-end against the real IEEE-CIS data and renders to a 117 KB output notebook. No execution errors. Stage took ~3 minutes (load + 5 EDA sections).

### 6. `uv run python scripts/run_sprint1_baseline.py` (full-dataset baseline run)
```
============================================================
Sprint 1 baseline — AUC summary
============================================================
  random    AUC=0.9615  AUC-PR=0.8072  LogLoss=0.0552  model=baseline_random_c4dc58d6150d.joblib
  temporal  AUC=0.9247  AUC-PR=0.6009  LogLoss=0.0826  model=baseline_temporal_fbd8f9501675.joblib
============================================================
```
Run wall-clock: 111.37 seconds. Two MLflow runs registered under experiment `fraud-detection`:

| variant | mlflow_run_id | val AUC | val AUC-PR | val log loss | train AUC | n_train | n_val |
|---|---|---:|---:|---:|---:|---:|---:|
| random | `aba2734bed5647e48d9d2b2671bfb93f` | 0.9615 | 0.8072 | 0.0552 | 0.9874 | 472,432 | 118,108 |
| temporal | `254b175352e44a74a6212df91ad195ff` | 0.9247 | 0.6009 | 0.0826 | 0.9907 | 414,542 | 83,571 |

**Reproducibility check:** the AUC values match exactly to the run done during 1.3 development (`d8cb259dba65454a9d21ca17f7584d9c` random / `fef1facf5bf14734bb6f9013f56e3010` temporal — same content_hashes `c4dc58d6150d…` / `fbd8f9501675…`, identical model files on disk). The training pipeline is fully deterministic at the `seed=42` setting.

Status: **GREEN** — but see "Findings" below for the AUC-vs-spec-range question.

## Findings

### Finding 1 — Temporal AUC exceeds the spec's expected upper bound

The prompt states the temporal-split baseline AUC should fall in **0.88–0.91**. Actual: **0.9247** — above the upper bound by 0.0147.

This is **not a regression**. Three observations:

1. **The model is genuinely stronger than expected.** LightGBM with default hyperparameters (`num_leaves=63`, `learning_rate=0.05`, `n_estimators=500`) and native categorical handling produces a meaningfully higher AUC than the planning estimate suggested. The 1.3 report already noted this ("at the upper edge of the prompt's quoted 0.88–0.91 range") — this gate corrects that to "above the upper edge by 0.015".

2. **Random > Temporal still holds.** Random AUC=0.9615; Temporal AUC=0.9247; gap=0.037. The leakage signal the spec design relies on is intact and visible.

3. **No downstream sprint depends on the AUC being inside 0.88–0.91 specifically.** Sprint 2's feature engineering tracks the *gap* (random − temporal), not the absolute level. Sprint 4's economic-cost evaluation tunes a threshold against the temporal model regardless of its AUC. Sprint 3's Optuna sweep starts from this number and is expected to lift it further. So the spec's range was a planning estimate, not a contract.

**Recommendation:** treat 0.9247 as the new Sprint 1 baseline-temporal anchor. Sprint 2's report should compare engineered-feature performance against 0.9247 as the starting line, not 0.88–0.91. The integration test floor (`_AUC_FLOOR = 0.75` on the 10k sample) and the unit-test contract are unchanged.

### Finding 2 — Lineage trail is clean and complete

The most-recent lineage run (`4e75a21749ed4d93ac5926f50d29e326`) carries all 5 expected steps in chronological order. Row counts chain cleanly: load 0→590,540 → clean 590,540→590,540 (0 dropped) → splits sum to 590,540. Schema fingerprints differ across the load/clean boundary (`44136fa3 → eb35259c → 0c241302`) and stay constant across the three split slices (correct: `temporal_split` is a pure row filter). Manifest `schema_version=1` matches the pinned literal in `verify_lineage.py`.

### Finding 3 — Notebook commit-policy compliance

`notebooks/01_eda.ipynb` executes end-to-end from a fresh kernel via `jupyter nbconvert --execute` and renders to 117 KB of executed-output `.ipynb`. The committed copy under `notebooks/01_eda.ipynb` carries those rendered outputs (committed in earlier 1.1.x prompts). `make nb-test` continues to pass alongside `make test`. CLAUDE.md §16 ("notebooks ship with rendered outputs") is satisfied.

## Sprint 1 acceptance checklist

- [x] EDA notebook executes end-to-end without errors
- [x] Temporal-split baseline runs on the full 590k-row dataset
- [x] Random + temporal variants produce distinct AUCs (random > temporal by 0.037)
- [x] Lineage trail is complete and consistent (5 steps, row-count chain, schema fingerprints)
- [x] All unit + integration + lineage tests pass (235 total: 233 pytest + 2 nbmake)
- [x] `make lint` / `make typecheck` / `make test` / `make notebooks` / `make nb-test` all return 0
- [x] Sprint history is squash-merged on `main` (5 PRs from 1.2.a through 1.3, plus the earlier 1.1.x trio)
- [x] Per the user directive: this audit produces no source-code changes; the gate report is the only artefact and lives on `sprint-1/sprint-gate` for John to commit.

## Conclusion

**Sprint 1 is gated GREEN.** All verification commands return 0; the lineage-data layer is clean and reproducible; the temporal baseline beats expectations. The single deviation (temporal AUC 0.9247 vs. the spec's 0.88–0.91 estimate) is a positive surprise, not a regression — flagged here so Sprint 2's planning uses the actual anchor.

Ready for John to commit on `sprint-1/sprint-gate`.
