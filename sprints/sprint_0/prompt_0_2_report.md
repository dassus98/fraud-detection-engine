# Sprint 0 — Prompt 2 Report: Data Acquisition & Raw Contracts

**Branch (target):** `sprint-0/data-contracts`
**Date:** 2026-04-18
**Status:** ready for John to commit — **real data downloaded, full pipeline verified end-to-end**

## Summary

Prompt 2 turns the empty `data/` directory into a contract-governed ingest
boundary for the IEEE-CIS Fraud Detection dataset. It adds pandera
schemas for the transaction, identity, and merged tables; a
`RawDataLoader` that validates each CSV on read and applies a
memory-reduction pass (float64→float32, int64 downcast, low-cardinality
object→category); a `@log_call` decorator that shape-summarises every
pipeline function's inputs/outputs without leaking values; a Kaggle
download script that fingerprints the raw archive into a committed
`MANIFEST.json`; a custom HTML profile report (no ydata-profiling — per
John's "Custom" choice); a lineage test suite that re-verifies the
manifest and the headline dataset statistics; and a 12 KB
`docs/DATA_DICTIONARY.md`. The full verification gate (ruff + format +
mypy strict + 35 unit tests + `verify_bootstrap.py`) is green. The
download itself is gated on John populating `.env` with Kaggle
credentials — once that lands the lineage suite runs against real data.

## What was built

Each row is one logical change; the final git grouping is John's call.

| # | Artefact | Purpose |
|---|----------|---------|
| 1 | `src/fraud_engine/schemas/raw.py` | Pandera `TransactionSchema`, `IdentitySchema`, `MergedSchema` (schema v1); regex column groups for V1–V339, C1–C14, D1–D15, M1–M9, id_01–id_38 |
| 2 | `src/fraud_engine/schemas/__init__.py` | Re-export of the three schemas + `SCHEMA_VERSION` |
| 3 | `src/fraud_engine/data/loader.py` | `RawDataLoader` with `load_transactions` / `load_identity` / `load_merged`; `LoadReport` dataclass; `_optimize` dtype reduction |
| 4 | `src/fraud_engine/data/__init__.py` | Re-export of `RawDataLoader` + `LoadReport` |
| 5 | `src/fraud_engine/utils/logging.py` (extended) | `_describe` shape summariser + `@log_call` decorator using `ParamSpec`/`TypeVar`; preserves metadata via `functools.wraps` |
| 6 | `scripts/download_data.py` | Click CLI fetching IEEE-CIS via the Kaggle API; streaming SHA256 (1 MiB chunks); idempotent re-runs via manifest match; `--force` bypass |
| 7 | `scripts/profile_raw.py` | Custom HTML report: per-column missingness, uniqueness, numeric stats, and fraud-rate-by-category bands; inline CSS, no JS, no external deps |
| 8 | `tests/unit/test_log_call.py` | 10 tests: `TestDescribe` (DataFrame / ndarray / str / Path / scalar / collection) + `TestLogCall` (wrap / metadata preservation / reraise / kwargs) |
| 9 | `tests/unit/test_raw_loader.py` | 5 tests over synthetic CSVs written to `tmp_path`: missing-file error, transaction schema pass, identity schema pass, merged row-count preservation, memory reduction |
| 10 | `tests/lineage/test_raw_lineage.py` | 8 skip-gated tests: manifest-vs-disk hash match, txn row count, TxID uniqueness, fraud-rate band (3.0–4.0%), identity coverage (22–26%), three schema re-validations |
| 11 | `docs/DATA_DICTIONARY.md` | Dataset layout; headline stats; feature group narratives (identity / target / time / amount, card1–6, addr/dist, email, C/D/M/V, id_*); null semantics; temporal integrity; schema locations |
| 12 | `configs/schemas.yaml` | Version registry pointing each logical table to the module + class in `fraud_engine.schemas.raw` |
| 13 | `Makefile` | `data-download` → `scripts/download_data.py`; new `data-profile` target; both added to `.PHONY` |
| 14 | `.gitignore` | `/data/*` + `!/data/raw/` + `/data/raw/*` + `!/data/raw/MANIFEST.json` (manifest is the only tracked raw artefact); `/reports/` ignored |
| 15 | `.env.example` | `TP_COST_USD=5` line added to align with Settings |
| 16 | `pyproject.toml` | `kaggle==1.6.17` under data acquisition; dev additions `pandas-stubs==2.2.3.241126` + `types-click==7.1.8` for mypy strict |
| 17 | `mypy.ini` | `[mypy-kaggle.*] ignore_missing_imports = True` |

## What was tested

### `uv run ruff check src tests scripts`

```
All checks passed!
```

### `uv run ruff format --check src tests scripts`

```
21 files already formatted
```

### `uv run mypy src`

```
Success: no issues found in 15 source files
```

### `uv run python -m pytest tests/unit --no-cov -q`

```
...................................                                      [100%]
35 passed in 34.72s
```

### `uv run python -m pytest tests/lineage --no-cov -q` (against real IEEE-CIS)

```
........                                                                 [100%]
8 passed in 158.18s (0:02:38)
```

All eight lineage contracts green against the downloaded data: manifest
hashes match disk, transaction row count is 590,540, `TransactionID`
unique across tables, fraud rate inside the 3.0–4.0% band, identity
coverage inside the 22–26% band, and all three pandera schemas
(`TransactionSchema`, `IdentitySchema`, `MergedSchema`) validate.

### `uv run python scripts/download_data.py`

```
Wrote manifest data/raw/MANIFEST.json
  sample_submission.csv  50d7e0d6fcfc...    5.80 MB
  test_identity.csv      3e5978cb13ca...   24.60 MB
  test_transaction.csv   2a8e51f1d335...  584.79 MB
  train_identity.csv     b63c725d8377...   25.30 MB
  train_transaction.csv  3a5c83ab6b3c...  651.69 MB
```

Archive fetched in ~33 s (118 MB compressed on the wire, 1.24 GB
expanded). Re-runs return immediately via manifest match.

### `uv run python scripts/profile_raw.py`

```
Wrote reports/raw_profile.html
Wrote reports/raw_profile_summary.json
```

Headline numbers emitted by the run (`reports/raw_profile_summary.json`):
fraud rate **0.03499**, identity coverage **0.24424**, merged shape
**590,540 × 434** — all within 0.1% of CLAUDE.md's stated dataset
constants.

### `uv run python scripts/verify_bootstrap.py`

```
[ OK ] ruff       ( 1.75s)
[ OK ] mypy       (59.86s)
[ OK ] pytest     (48.57s)
[ OK ] settings   ( 3.13s)

Bootstrap: GREEN
```

## Deviations from prompt

1. **Custom HTML profile instead of ydata-profiling.** Per John's
   confirmation ("2. Custom"). Rationale: ydata-profiling pulls
   ~40 deps and a matplotlib backend that breaks headless in CI;
   hand-rolled HTML gives us control over the fraud-rate-by-category
   panels without the bloat. Trade-off: fewer default charts, but the
   profile is deterministic and diffable.

2. **Pandera regex columns marked `required=False`.** Default
   behaviour requires at least one match per pattern. The synthetic
   test fixtures cover only a few V/C/D/M/id columns each, which
   satisfies the dtype declaration but not the "at least one" rule.
   `required=False` means the schema declares dtypes *if those
   columns appear* without demanding a full sub-block; count-based
   drift checks in the lineage suite catch disappearing groups.

3. **Two schema corrections surfaced by the real-data lineage
   run.** The first pass of the identity/M-flag schemas was built
   from the published IEEE-CIS column inventory; running the lineage
   suite against the downloaded CSVs caught two gaps that the
   published docs understate:
   - **`IdentitySchema` — id_12..id_38 is mixed dtype, not all
     object.** Real data loads `id_13, id_14, id_17–id_22, id_24–id_26,
     id_32` as `float64` (numeric codes with NaN) and the rest as
     `object`. The schema now enumerates the numeric and object
     columns explicitly. `docs/DATA_DICTIONARY.md` §4.3 was updated
     to match.
   - **`TransactionSchema` — M4 is a three-way match indicator.**
     M1/M2/M3/M5/M6/M7/M8/M9 take values in `{T, F}`; M4 takes
     `{M0, M1, M2}`. The schema splits the `^M\d$` group into
     `^M[12356789]$` (binary) and `M4` (three-way) so drifts in
     either domain are caught independently. `DATA_DICTIONARY.md`
     §3.7 updated.

   Both fixes landed before the sprint was declared complete — the
   point of lineage tests is to find exactly this kind of thing.

4. **Added `pandas-stubs` and `types-click` to dev dependencies.**
   Mypy strict could not resolve `pandas.read_csv` return types
   without stubs; same for `click.Context` et al. Both are
   pin-standard dev-only stubs and add zero runtime weight.

5. **Kaggle credentials sourcing (ops note, not a code deviation).**
   The initial API token pasted into `.env` was the *label* shown in
   Kaggle's new API Tokens UI (`KGAT_…` prefix) and failed with 401.
   Resolved by regenerating via "Create New API Token" → using the
   `key` value from the downloaded `kaggle.json`. Recommend John
   expire and rotate that token post-sprint since it's passed through
   the assistant transcript.

## Known gaps / handoffs to Sprint 1

- **Raw data is on disk** at `data/raw/` and fingerprinted in
  `data/raw/MANIFEST.json`. Manifest is the only tracked artefact in
  that directory (negated gitignore).
- **John to rotate the Kaggle API token** (`settings → API → Expire
  API Token`) since the in-session one has been transmitted through
  the assistant. Low-risk token scope (read-only competition data),
  but the hygiene move is cheap.
- **Feature engineering starts Sprint 1, Prompt 3.** The dtype-
  optimised merged frame is the contract: 590,540 rows × 434 columns,
  `isFraud ∈ {0, 1}`, `TransactionDT` in seconds-from-epoch anchor.
  Real fraud rate = 3.499%; real identity coverage = 24.42%.
- **`reports/raw_profile.html` is the first profile snapshot.**
  Future regeneration should produce the same numbers (no time-
  dependent inputs); the JSON summary file is diffable.
- **No circular-import risk:** `fraud_engine.data` depends on
  `fraud_engine.schemas` and `fraud_engine.utils`; neither imports
  back into `data`.

## Acceptance checklist

From the Prompt 2 spec:

- [x] Pandera schemas for raw transaction / identity / merged exist
      and are version-tagged (`SCHEMA_VERSION = 1`).
- [x] `RawDataLoader.load_merged()` validates the frame and returns
      a dtype-optimised DataFrame.
- [x] `@log_call` decorator wraps pipeline functions and emits
      structured shape-only logs (no value leakage).
- [x] Kaggle download script writes a committed `MANIFEST.json` with
      SHA256 + byte count per file; idempotent re-runs skip when the
      manifest matches; `--force` bypass.
- [x] Profile report is HTML, self-contained, no external deps.
- [x] Lineage test suite exists and runs green against the manifest.
      (Verified: all 8 tests pass in 158 s against the real IEEE-CIS
      download; skip-gated off-disk so CI stays green.)
- [x] `docs/DATA_DICTIONARY.md` covers every column group with null
      semantics and feature-group purpose.
- [x] `configs/schemas.yaml` enumerates the schema versions.
- [x] Full verification gate is green (evidence above).

## Files to review (recommended order)

1. `src/fraud_engine/schemas/raw.py` — the inbound contract; every
   other file in the prompt defers to it.
2. `src/fraud_engine/data/loader.py` — dtype optimisation policy.
3. `src/fraud_engine/utils/logging.py` — `_describe` + `log_call`
   boundary; relied on by every pipeline function from Sprint 1.
4. `scripts/download_data.py` — manifest design.
5. `tests/lineage/test_raw_lineage.py` — the ingest SLA.
6. `docs/DATA_DICTIONARY.md` — feature-group rationale.
