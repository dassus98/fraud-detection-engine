# Sprint 1 — Prompt 1.1.a Report: EDA Notebook Sections A + B

**Branch:** `sprint-1/prompt-1-1-a-eda-overview-target`
**Date:** 2026-04-25
**Status:** ready for John to commit — **all six verification gates green** (ruff lint, ruff format, mypy strict, 183 unit tests, 13 lineage tests, nbmake harness on both notebooks)

## Summary

Prompt 1.1.a is the first of Sprint 1's eight fine-grained prompts. It
audits and gap-fills **Section A (Data Overview)** and **Section B
(Target Analysis)** of `notebooks/01_eda.ipynb` against the new spec.
The prior monolithic Prompt 1 had stood up A–G as scaffolding; this
prompt brings A and B to the spec while leaving C–G untouched (later
prompts own those). The stale roll-up scaffold report
`sprints/sprint_1/prompt_1_1_report.md` is renamed to
`sprints/sprint_1/prompt_1_1_scaffold_report.md` so the
`prompt_1_<x>_<y>_report.md` namespace is clean for the new
fine-grained reports.

The notebook is built programmatically by
`scripts/_build_eda_notebook.py`. The `.ipynb` is regenerated from
that builder; **never hand-edit the JSON**. Section A picks up
calendar derivation from `Settings.transaction_dt_anchor_iso` (kept as
a standalone `event_dt` Series so Section F's cleanlab classifier
keeps its feature-column selection intact), a daily volume plot, and
identity-coverage + has-id-vs-no-id fraud-rate analysis with binomial
CIs. Section B picks up Wilson CIs on every group-rate, fraud rate by
`card4` / `card6` (filtered to `n ≥ 100`), a day-of-week × hour-of-day
heatmap, top-20 `P_emaildomain` (filtered to `n ≥ 500`), and a
log-scale `TransactionAmt` overlay fraud-vs-non-fraud. The Wilson CI
helper (`wilson_ci`) is defined inline in Section A.5 (the first cell
that uses it) and re-used through B; vectorised on numpy arrays for
clean composition with `groupby().agg(...)`.

End-to-end execution under `nbconvert --execute` succeeds; under the
`nbmake` harness both `00_observability_demo.ipynb` and the regenerated
`01_eda.ipynb` pass in 57.36 s.

## What was built

| # | Artefact | Purpose |
|---|----------|---------|
| 1 | `scripts/_build_eda_notebook.py` (Sections A + B reshaped) | Section A: 1 markdown header + 5 code cells (overview, calendar derivation, daily volume, identity coverage, has-id-vs-no-id fraud rate with `wilson_ci` helper). Section B: 1 markdown header + 8 code cells (overall + CI, amount bucket + CIs, ProductCD + CIs, hour-of-day + CI band, card4/card6 + CIs, dow×hour heatmap, top-20 P_emaildomain + CIs, log-scale TransactionAmt overlay). Sections C–G left intact for later prompts. |
| 2 | `notebooks/01_eda.ipynb` (regenerated artefact) | Regenerated from the updated builder. 25 cells (was 16). |
| 3 | `sprints/sprint_1/prompt_1_1_report.md` → `sprints/sprint_1/prompt_1_1_scaffold_report.md` | Rename. Preserves the prior roll-up record as portfolio history and frees the namespace for fine-grained `prompt_1_<x>_<y>_report.md` files. |
| 4 | `tests/integration/test_sprint1_baseline.py` (1 line) | Updated module docstring link to point at the renamed scaffold report. |
| 5 | `sprints/sprint_1/prompt_1_1_a_report.md` (this file) | Completion report for prompt 1.1.a. |

The Sprint 0 report `sprints/sprint_0/prompt_0_1_a_report.md` retains
its references to the old `prompt_1_1_report.md` filename — it is
historical portfolio content and is not rewritten.

## Reusable utilities leveraged

| Utility | Path | Use |
|---|---|---|
| `RawDataLoader.load_merged` | `src/fraud_engine/data/loader.py` | Already wired in Setup; no change. |
| `Settings.transaction_dt_anchor_iso` | `src/fraud_engine/config/settings.py` | Anchor for `event_dt` derivation in Section A. |
| `attach_artifact(run, obj, *, name)` | `src/fraud_engine/utils/tracing.py` | Persists `Figure`, `dict`, `list` objects via isinstance dispatch. Used in every code cell across A and B. |
| `configure_logging`, `get_logger` | `src/fraud_engine/utils/logging.py` | Already wired in Setup; no change. |
| `run_context` | `src/fraud_engine/utils/tracing.py` | Already wired in Setup; no change. |

## Inline implementations (no project helper exists)

- **`wilson_ci(k, n, *, alpha=0.05)`** — vectorised 95% Wilson
  binomial CI via `scipy.stats.norm.ppf`. Defined inline in Section
  A.5 (first usage) and reused throughout Section B. Returns
  `(low, high)` numpy arrays; `n=0` entries return `NaN`. No
  project-wide helper introduced — single-file scope, will move to
  `src/fraud_engine/utils/` only if a second module needs it.
- **Calendar derivation** — `pd.Timestamp(SETTINGS.transaction_dt_anchor_iso)`
  + `pd.to_timedelta(merged["TransactionDT"], unit="s")`. Inline,
  one cell.

## What was tested

### 1. `make lint`

```
uv run ruff check src tests scripts
All checks passed!
```

### 2. `uv run ruff format --check scripts/_build_eda_notebook.py`

```
1 file already formatted
```

### 3. `make typecheck`

```
uv run mypy src
Success: no issues found in 20 source files
```

(mypy gate is `src/` only per the Makefile; `scripts/` is not in
scope for type-checking.)

### 4. `make test-fast`

```
183 passed, 34 warnings in 7.81s
```

### 5. `make test-lineage`

```
13 passed, 14 warnings in 191.43s (0:03:11)
```

### 6. `uv run jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output /tmp/01_executed.ipynb`

```
[NbConvertApp] Converting notebook notebooks/01_eda.ipynb to notebook
[NbConvertApp] Writing 60158 bytes to /tmp/01_executed.ipynb
```

### 7. `make nb-test`

```
notebooks/00_observability_demo.ipynb::00_observability_demo.ipynb PASSED [ 50%]
notebooks/01_eda.ipynb::01_eda.ipynb PASSED                              [100%]
============================== 2 passed in 57.36s ==============================
```

A first invocation of `make nb-test` (immediately after `make
test-lineage`) reported `01_eda.ipynb` failing on the
`MANIFEST.json` existence check at ~9.98 s with no other
diagnostics; re-running cleanly passed. The `data/raw/MANIFEST.json`
file's mtime did not change between the two runs, and 01_eda.ipynb
in isolation (`pytest --nbmake notebooks/01_eda.ipynb`) consistently
passes. Treated as a transient race / resource-contention artefact
of running heavy lineage tests immediately before nbmake; not a
correctness defect in the notebook.

## Deviations from prompt

- **Wilson CI helper kept inline**, not promoted to
  `src/fraud_engine/utils/`. The spec is silent on this; one-file
  scope is correct per CLAUDE.md §9 #9 ("don't introduce
  abstractions beyond what the task requires"). If a second module
  later needs binomial CIs, lift it then.
- **`event_dt` is a standalone `pd.Series`**, not a column on
  `merged`. The plan called this out: adding it to `merged` would
  cause Section F's `_select_feature_columns` to pick `event_dt` as
  a model feature (it's not in `_NON_FEATURE_COLUMNS`), and a
  datetime feature would either fail to fit or leak. The standalone
  Series form is the same data, decoupled from the model-input
  contract.
- **`hour_of_day` derivation switched** from
  `(TransactionDT // 3600) % 24` to `event_dt.dt.hour` to match the
  spec ("derive hour from `event_dt`, not from raw `TransactionDT`").
  Numerically equivalent for IEEE-CIS (the anchor falls on midnight
  UTC), but the spec form is the one that survives if the anchor
  ever changes.
- **`P_emaildomain` chosen over `R_emaildomain`** for the top-20
  email-domain chart. The spec said "P_emaildomain (and/or
  R_emaildomain)"; `P_emaildomain` (purchaser) has higher coverage
  in IEEE-CIS train, so the top-20 is more informative there. A
  later prompt (1.1.b or similar) can add the symmetric
  `R_emaildomain` chart if needed.
- **`card4` / `card6` filter at `n ≥ 100`**, `P_emaildomain` filter
  at `n ≥ 500`. Both numbers are documented inline as the
  noise-floor where a single fraudster's 5 transactions stops
  dominating the chart; they are not in `Settings` because they are
  visualisation policy, not pipeline configuration.
- **No new figure removed; a stale figure remains.** Prior Section B
  produced `reports/figures/target_analysis.png` (a 1×3 panel chart).
  The new Section B replaces that with eight separate, more focused
  figures. The prior PNG is still on disk from a stale run; no
  cleanup is performed because the figure file is gitignored and the
  next `nbconvert --execute` writes the new figures alongside.

## Known gaps / handoffs

- **Sections C–G are untouched.** Their refinement against the new
  spec belongs to later prompts in Sprint 1. The Section G findings
  list still reflects the prior monolithic Prompt 1 numbers; a later
  prompt will refresh those after all section-specific work is done.
- **Section G's link** to the (renamed) scaffold report has been
  updated. No other notebook references break under the rename.
- **Wilson CI helper is not in `utils/`.** If Sprint 2's feature
  pipeline or Sprint 4's evaluator needs binomial CIs, lift the
  helper to `src/fraud_engine/utils/metrics.py` at that point.

## Acceptance checklist

- [x] **Section A covers the spec.** Memory + dtype histogram,
  calendar derivation, daily volume, identity coverage, has-id-vs-no-id
  fraud rate with CIs.
- [x] **Section B covers the spec.** Overall fraud rate + CI; fraud
  rate by amount bucket / ProductCD / hour-of-day / card4 / card6 /
  dow×hour heatmap / top-20 P_emaildomain — all with 95% Wilson CIs;
  log-scale TransactionAmt fraud-vs-non-fraud overlay.
- [x] **Notebook is regenerated from the builder.** No hand-editing
  of `01_eda.ipynb`.
- [x] **Sections C–G unchanged.** Diff shows only the link in
  Section G's markdown was updated.
- [x] **Stale scaffold report renamed.** `prompt_1_1_report.md` →
  `prompt_1_1_scaffold_report.md`; non-historical references
  updated; Sprint 0 report left intact.
- [x] **All six verification gates green** (lint, format, mypy,
  test-fast, test-lineage, nbmake).
- [x] **No git commits or pushes by the agent.** Changes are in the
  working tree on `sprint-1/prompt-1-1-a-eda-overview-target`,
  awaiting John's commit.

Ready for John to commit on `sprint-1/prompt-1-1-a-eda-overview-target`.
