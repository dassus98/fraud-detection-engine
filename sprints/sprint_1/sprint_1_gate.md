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

---

## Post-audit re-verification (2026-04-27)

After the comprehensive audit pass on `sprint-1/audit-and-gap-fill` — which touched 7 source / test / notebook / config files plus added the rolling `sprints/sprint_1/audit_findings.md` — the full Sprint 1 verification protocol was re-run. **Every gate is still green.**

### Verification results

| Gate | Pre-audit (PR #9) | Post-audit |
|---|---|---|
| `make lint` | All checks passed | All checks passed |
| `make typecheck` | 23 source files | 23 source files |
| `make test` (pytest unit + integration + lineage) | 233 passed in 253.74s | **235 passed** in 213.12s |
| `make nb-test` (nbmake harness) | 2 passed in 72.56s | 2 passed in 68.55s |
| `verify_lineage.py` | GREEN, 5 steps | GREEN, 5 steps |
| `nbconvert 01_eda.ipynb` | 117,055 bytes | **124,237 bytes** (+7 KB) |
| `run_sprint1_baseline.py` | 110.37s wall, 2 MLflow runs | 105.40s wall, 2 MLflow runs |

### Test count delta: +2

The two new tests are both from the 1.2.a audit, both in `tests/unit/test_splits.py::TestValidateNoOverlap`:

- `test_rejects_val_test_temporal_overlap` — covers `splits.py:243` (the val↔test contiguity branch the original suite missed; existing test only covered train↔val)
- `test_rejects_split_size_mismatch` — covers `splits.py:226` (manifest-sum vs ID-set-sum branch — defends against a caller bypassing `temporal_split` and constructing a malformed `SplitFrames` directly)

`splits.py` coverage: **95% → 100%** (70/70 stmts, 18/18 branches). The 1.3 audit also expanded `tests/unit/test_baseline.py::TestMLflowLogging` (renamed `test_logs_auc_metric` → `test_logs_train_and_val_metrics`, asserting all 6 MLflow metrics + `BaselineResult` field cross-checks) — but that's a test rename, not a count change.

### Notebook size delta: +7 KB

`notebooks/01_eda.ipynb` grew from 117,055 → 124,237 bytes (+7,182 bytes). Source: 19 new markdown cells added in the 1.1.a audit — 5 in Section A (4 intro mds + 1 per-plot interp) + 14 in Section B (7 intro mds + 7 per-plot interps). The cells satisfy the 1.1.a spec wording *"every code cell has a markdown cell above it explaining what we're looking at and why"* and *"every plot has … a 1-sentence interpretation below it"* — both were structurally missing in the original PR #1 delivery despite the substantive content being correct.

### AUC numbers: bit-identical

Random AUC = 0.9615, temporal AUC = 0.9247. Match the pre-audit numbers exactly down to the floating-point digit, which confirms **no model behaviour changed** during the audit — every code change was either (a) docstring / comment, (b) test addition or rename, (c) classmethod conversion (functionally equivalent), (d) markdown cells in the notebook, (e) gitignore exception. The model's training loop, hyperparameters, and feature-column selection are all untouched.

### What the audit branch contributes vs `main`

| File | Change | Audit prompt |
|---|---|---|
| `.gitignore` | Surgical `!/reports/sprint1_eda_summary.md` exception so the executive-summary file is now tracked (was gitignored despite being a 1.1.c "Produces" deliverable) | 1.1.c |
| `notebooks/01_eda.ipynb` | Regenerated; +7 KB from 19 new markdown cells in Sections A and B | 1.1.a |
| `notebooks/00_observability_demo.ipynb` | Re-executed in place by `make notebooks` (output cells refreshed) | 1.1.a side-effect |
| `scripts/_build_eda_notebook.py` | +19 markdown cells in Sections A + B (intros above each code cell + interps below each plot) | 1.1.a |
| `scripts/build_interim.py` | Removed unused public `EXPECTED_STEPS` constant (CLAUDE.md §5.7 dead code rule) | 1.2.d |
| `scripts/verify_lineage.py` | Collapsed `LineageLog(...).read()` 2-line pattern to single classmethod call; updated stale comment that referenced the now-removed `EXPECTED_STEPS` | 1.2.c, 1.2.d |
| `src/fraud_engine/data/cleaner.py` | Docstring clarification on `clean()` warning future callers about the `optimize=False` requirement and the exact `SchemaErrors` failure mode | 1.2.b |
| `src/fraud_engine/data/lineage.py` | `LineageLog.read` converted from instance method to `@classmethod` per the original spec wording (divergence not flagged in PR #6's report) | 1.2.c |
| `tests/unit/test_baseline.py` | Renamed `test_logs_auc_metric` → `test_logs_train_and_val_metrics`; expanded body from 1 metric assertion to 6 + dataclass-field cross-check + sanity bounds | 1.3 |
| `tests/unit/test_lineage.py` | 5 call sites switched to classmethod; one obsolete `log = LineageLog(...)` line removed | 1.2.c |
| `tests/unit/test_splits.py` | +2 tests in `TestValidateNoOverlap` (val↔test temporal overlap + manifest size mismatch) closing the 95% → 100% coverage gap | 1.2.a |
| `tests/lineage/test_interim_lineage.py` | 3 call sites switched to classmethod; docstring reference updated | 1.2.c |
| `reports/sprint1_eda_summary.md` | Two stale captions fixed (lines 23 + 235); both still described the *daily / 7-day rolling* plot from 1.1.a after Section E switched to the *weekly + Wilson-CI* plot in 1.1.c. **Now tracked** under the new `.gitignore` exception | 1.1.c |
| `sprints/sprint_1/audit_findings.md` | New rolling audit log (one section per audited prompt: 1.1.a, 1.1.b, 1.1.c, 1.2.a, 1.2.b, 1.2.c, 1.2.d, 1.3) | this audit |
| `sprints/sprint_1/sprint_1_gate.md` | Appended this **Post-audit re-verification** section | this gate |

### Conclusion (post-audit)

**Sprint 1 final gate: GREEN.** Comprehensive audit closed real gaps in 6 of the 8 audited prompts (1.1.a, 1.1.c, 1.2.a, 1.2.b, 1.2.c, 1.2.d, 1.3 — only 1.1.b was already to spec) without changing model behaviour. Test count up 2; coverage on `splits.py` up to 100%; the executive-summary deliverable is now actually committed (was previously local-only); and a real classmethod-vs-instance-method API divergence in the lineage primitive was corrected to match the spec. Ready for Sprint 2.
