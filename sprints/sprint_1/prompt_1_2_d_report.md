# Sprint 1 — Prompt 1.2.d Report: Interim Build Pipeline + Lineage Verifier

**Date:** 2026-04-27
**Status:** all verification gates green — `make lint`, `make typecheck` (23 source files, unchanged), `make test-fast` (211 passed, unchanged), `make test-lineage` (16 passed in 170.69s via §17 detached-daemon), `uv run python scripts/build_interim.py` (44.5s, 5 lineage records, 3 parquets + manifest written), `uv run python scripts/verify_lineage.py` (GREEN, 5 expected steps, every cross-check passes).

## Summary

Prompt 1.2.d delivers the **canonical interim-build pipeline** that materialises the layer Sprint 2+ consumes, plus the **lineage-verification gate** that closes the loop on CLAUDE.md §7.2. Two new scripts:

- **`scripts/build_interim.py`** — Click CLI; opens a `Run("build_interim")`; lineage-logs five named steps (`load_merged`, `interim_clean`, `split_train`, `split_val`, `split_test`); writes `data/interim/{train,val,test}.parquet` and `data/interim/splits_manifest.json`; attaches the manifest to the run's `artifacts/`; emits 6 run metrics (per-split row counts + per-split fraud rates) and 4 run params (output dir, train/val cut points, anchor).
- **`scripts/verify_lineage.py`** — Click CLI; finds the most recent `lineage.jsonl` by mtime (or accepts `--run-id`); runs six independent contracts (every expected step present; `load.output_rows == clean.input_rows`; cleaner never invents rows; `sum(splits.output_rows) == clean.output_rows`; each parquet's row count matches its split-step record; manifest `schema_version` matches the pinned literal). Collects every failure rather than failing fast, exits 0 on green and 1 on red, with a structured `verify.passed` / `verify.failed` event in the structlog trail.

The lineage decorator (`@lineage_step`) was authored in 1.2.c against `DataFrame → DataFrame` callables. 1.2.d is the first integration that exercises it on the **real merged 590k-row IEEE-CIS train frame** end-to-end and proves the JSONL artefact is queryable, the row-count chain is consistent, and the split partitioning is loss-less.

## Spec vs. actual

| Spec element | Implementation | Status |
|---|---|---|
| Click CLI that opens a `Run("build_interim")` | `Run("build_interim", settings=settings, metadata={"output_dir": ...})` from `fraud_engine.utils.tracing` | ✓ |
| Loads raw train data via `RawDataLoader` | `RawDataLoader(settings=settings).load_merged(optimize=False)` — see Surprising Finding §1 for the `optimize=False` decision | ✓ |
| Applies `TransactionCleaner` | `lineage_step("interim_clean")(cleaner.clean)(merged)` | ✓ |
| Applies `temporal_split` with boundaries from EDA | `temporal_split(interim, settings=settings)` — boundaries default to `Settings.train_end_dt=10,454,400` (Day 121) and `val_end_dt=13,046,400` (Day 151), pinned in 1.2.b's report | ✓ |
| Writes `data/interim/{train,val,test}.parquet` | `splits.{train,val,test}.to_parquet(target_dir / f"{name}.parquet")` | ✓ |
| Writes `data/interim/splits_manifest.json` | `write_split_manifest(splits, target_dir / "splits_manifest.json")` | ✓ |
| Every step is lineage-logged | 5 records: `load_merged` (decorator on void→data wrapper), `interim_clean` (decorator on bound method), `split_{train,val,test}` (decorator on captured-slice passthroughs) | ✓ |
| `verify_lineage.py` validates parquet schema, lineage completeness, row-count consistency, no schema-version regressions | 6 independent checks (4 lineage + 2 parquet/manifest); see `_run_checks` | ✓ |

## Files changed

| File | Type | LOC | Purpose |
|---|---|---|---|
| `scripts/build_interim.py` | new | 308 | End-to-end interim-build pipeline (Run + lineage + parquet) |
| `scripts/verify_lineage.py` | new | 247 | Lineage + parquet invariant checker |
| `sprints/sprint_1/prompt_1_2_d_report.md` | new | this file | Completion report |

No `src/` changes; no test changes. The decision to keep this prompt's scope to `scripts/` only (no `src/` module additions) was deliberate — `lineage_step` and `Run` already exist; this prompt is the integration that wires them together.

## Side-effect outputs (gitignored)

| Path | Size | Rows |
|---|---|---|
| `data/interim/train.parquet` | 58.9 MB | 414,542 |
| `data/interim/val.parquet` | 12.3 MB | 83,571 |
| `data/interim/test.parquet` | 14.7 MB | 92,427 |
| `data/interim/splits_manifest.json` | 472 B | — |
| `logs/runs/4e75a21749ed4d93ac5926f50d29e326/run.json` | ~700 B | — |
| `logs/lineage/4e75a21749ed4d93ac5926f50d29e326/lineage.jsonl` | ~1 KB | 5 records |

Total interim parquet footprint: **~86 MB** for the full train slice (590,540 rows).

## Lineage trail

Five records emitted on run `4e75a21749ed4d93ac5926f50d29e326`:

| Step | input_rows | output_rows | input_schema_hash | output_schema_hash | duration_ms |
|---|---:|---:|---|---|---:|
| `load_merged` | 0 | 590,540 | `44136fa3…` (empty) | `eb35259c…` | 32,078 |
| `interim_clean` | 590,540 | 590,540 | `eb35259c…` | `0c241302…` | 5,115 |
| `split_train` | 590,540 | 414,542 | `0c241302…` | `0c241302…` | 0.001 |
| `split_val` | 590,540 | 83,571 | `0c241302…` | `0c241302…` | 0.001 |
| `split_test` | 590,540 | 92,427 | `0c241302…` | `0c241302…` | 0.000 |

Row-count chain: `0 → 590,540 → 590,540 → (414,542 + 83,571 + 92,427) = 590,540` ✓
Split fingerprints all equal `0c24130248754b96` (correct: `temporal_split` is a pure row-filter — no dtype or column changes).

## Splits manifest (verbatim)

```json
{
  "fraud_rate_overall": 0.03499000914417313,
  "fraud_rate_test": 0.03476256937907754,
  "fraud_rate_train": 0.035219591742211884,
  "fraud_rate_val": 0.03410273898840507,
  "max_transaction_dt": 15811131,
  "min_transaction_dt": 86400,
  "n_original": 590540,
  "n_test": 92427,
  "n_train": 414542,
  "n_val": 83571,
  "schema_version": 1,
  "seed": 42,
  "train_end_dt": 10454400,
  "transaction_dt_anchor_iso": "2017-12-01T00:00:00+00:00",
  "val_end_dt": 13046400
}
```

Fraud rates per split (train 3.52% / val 3.41% / test 3.48%) match the overall rate (3.50%) within ±0.1pp — the calendar boundaries do not introduce class-imbalance drift across windows.

## Verification — verbatim output

### `make lint`
```
uv run ruff check src tests scripts
All checks passed!
```

### `make typecheck`
```
uv run mypy src
Success: no issues found in 23 source files
```
(Unchanged from 1.2.c — both new files are under `scripts/`, which is not in the canonical `make typecheck` scope. They were independently mypy-verified: `uv run mypy scripts/build_interim.py scripts/verify_lineage.py` → `Success: no issues found in 2 source files`.)

### `make test-fast`
```
211 passed, 34 warnings in 7.88s
```
Unchanged from 1.2.c. No test files added in this prompt.

### `make test-lineage` (via §17 detached-daemon)
```
16 passed, 14 warnings in 170.69s (0:02:50)
```
Unchanged from 1.2.c. No test files added; existing suite still green.

### `uv run python scripts/build_interim.py`
```
build_interim: GREEN
  run_id: 4e75a21749ed4d93ac5926f50d29e326
  output: data/interim
  rows:   train=414,542 val=83,571 test=92,427
```
Total wall-clock: 44.5 seconds (load 32s + clean 5.1s + split 0.8s + parquet write 5.4s + bookkeeping).

### `uv run python scripts/verify_lineage.py`
```
Lineage verification: GREEN
  run_id: 4e75a21749ed4d93ac5926f50d29e326
  steps:  5 (load_merged, interim_clean, split_train, split_val, split_test)
```
All six contracts (steps-present, load↔clean chain, drop invariant, splits sum, parquet rows match lineage, manifest schema_version) green.

## Surprising findings

1. **`load_merged(optimize=True)` is incompatible with `cleaner.clean()` on the real data.** The loader's `_optimize` downcasts dtypes (int64→int32, float64→float32, object→category) AFTER its internal `MergedSchema` validation. The cleaner then re-validates the optimised frame against `InterimTransactionSchema`, which inherits `MergedSchema`'s strict dtype constraints — and rejects every downcast. No existing pipeline hits this combination: `run_sprint1_baseline.py` does `optimize=True + temporal_split` (no schema re-validation), and `tests/lineage/test_interim_lineage.py` uses a synthetic fixture with default int64/float64/object dtypes. **Resolution:** the build script passes `optimize=False`. Memory cost: the merged frame is ~2.6 GB during cleaning instead of ~924 MB. Disk impact on the parquets is none — pyarrow's column compression at write time is independent of the in-memory dtypes. **Filed for follow-up:** a future Sprint should either relax `MergedSchema` to accept the optimised dtype family or post-optimise the cleaned interim frame before parquet write. Both are scope-creep for 1.2.d.

2. **The cleaner dropped 0 rows on the real data.** The defensive `TransactionAmt > 0` filter found no offenders — every IEEE-CIS train transaction satisfies the constraint at ingest. The drop logic remains in place per CLAUDE.md §7.3 ("loud signal" defence-in-depth) but is dormant on this dataset. Recorded by both the cleaner's `cleaner.report` event (`rows_in=590540, rows_out=590540, rows_dropped=0`) and the lineage step (`input_rows=output_rows=590540`).

3. **The three split records share an output fingerprint but differ in row counts.** Each passthrough's `output_schema_hash == 0c24130248754b96`, identical to the cleaner's output hash. This is correct: `temporal_split` is a pure row-filter — no column additions, no dtype changes — so every slice's schema is identical to the parent's. The three records are still individually meaningful because their `output_rows` differ (414,542 / 83,571 / 92,427), and the verifier asserts the partition-sum invariant.

4. **`Run` was already authored in `fraud_engine.utils.tracing`.** The spec says "Opens a `Run("build_interim")`"; `Run` is a class, not a free helper. It's the right primitive — `__enter__` calls `configure_logging`, creates `logs/runs/{run_id}/`, and `__exit__` writes the success/failure record into `run.json`. The script's `with Run("build_interim") as run:` block is exactly the spec-aligned shape; `run.log_param`/`run.log_metric`/`run.attach_artifact` populated the run directory automatically.

5. **`pyarrow.parquet` ships without type stubs.** `verify_lineage.py` uses `pq.read_metadata(path).num_rows` for an O(1) row-count check, but mypy complains about the missing `py.typed` marker. Resolved with a single `# type: ignore[import-untyped]` on the import. The alternative — materialising the parquet via `pd.read_parquet` for `len()` — would have read 86 MB just to count rows.

## Deviations from the prompt

1. **The script does not delete pre-existing parquets before writing.** Pandas' `to_parquet` overwrites in place. A future iteration that keeps run-stamped copies under `data/interim/runs/{run_id}/` would be additive to this contract; the current behaviour matches `run_sprint1_baseline.py`'s manifest-write semantics (overwrite by path).

2. **The split records use a closure-captured passthrough rather than wrapping `temporal_split` itself.** `temporal_split` returns `SplitFrames`, not a `pd.DataFrame`, so `@lineage_step` (which requires `DataFrame → DataFrame`) cannot decorate it directly. The passthrough trick (each `_make_split_step(name, slice).{decorated}(interim)` call records one record with input=interim, output=slice) was the cleanest path to one record per partition without exposing private lineage helpers. Documented in `_make_split_step`'s docstring.

3. **`scripts/verify_lineage.py` pins `schema_version=1` as a literal**, not via `from fraud_engine.data.splits import _MANIFEST_SCHEMA_VERSION`. The script's whole purpose is to detect regressions; importing the live constant would mask a silent bump that this gate is supposed to catch. Documented in the script's module-level "Trade-offs considered" section.

## Acceptance checklist

- [x] `scripts/build_interim.py` created (Click CLI, opens `Run`, lineage-logs every step, writes 3 parquets + manifest)
- [x] `scripts/verify_lineage.py` created (lineage + parquet invariant checker)
- [x] `data/interim/train.parquet` (414,542 rows, 58.9 MB)
- [x] `data/interim/val.parquet` (83,571 rows, 12.3 MB)
- [x] `data/interim/test.parquet` (92,427 rows, 14.7 MB)
- [x] `data/interim/splits_manifest.json` (schema_version=1, fraud rates within ±0.1pp of overall)
- [x] `make lint` returns 0
- [x] `make typecheck` returns 0 (23 src files unchanged; both new scripts independently mypy-verified)
- [x] `make test-fast` returns 0 (211 passed, unchanged from 1.2.c)
- [x] `make test-lineage` returns 0 (16 passed, unchanged from 1.2.c)
- [x] `uv run python scripts/build_interim.py` returns 0 (44.5s, 5 lineage records emitted)
- [x] `uv run python scripts/verify_lineage.py` returns 0 (GREEN, all 6 contracts pass)
- [x] `sprints/sprint_1/prompt_1_2_d_report.md` written (this file)
- [x] Per the user directive: no git operations performed by the agent. PR/commit are John's.

Verification passed. End of prompt.
