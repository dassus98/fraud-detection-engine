# Sprint 0 — Prompt 0.2.b Report: Pandera schemas for the raw IEEE-CIS tables

**Date:** 2026-04-21
**Status:** Ready for John to commit. No git action from me. (CLAUDE.md §2.)

## Summary

Prompt 0.2.b asks for four things: (1) executing the Kaggle download,
(2) `src/fraud_engine/schemas/raw.py`, (3) `configs/schemas.yaml`,
(4) `tests/unit/test_raw_schemas.py`. Items 1–3 all landed earlier
under [prompt_0_2_report.md](prompt_0_2_report.md) on 2026-04-18; the
only real gap was the direct schema-rejection test file (4), which
did not exist. This prompt creates that file, closes the gap, and
re-verifies the whole stack.

## Part 1 — Download execution (pre-existing; audit only)

Download was executed on 2026-04-18 under prompt_0_2. Current state of
`data/raw/`:

```
-rw-r--r-- MANIFEST.json            864  bytes
-rw-r--r-- sample_submission.csv    5.8 MB
-rw-r--r-- test_identity.csv         25  MB
-rw-r--r-- test_transaction.csv     585  MB
-rw-r--r-- train_identity.csv        26  MB
-rw-r--r-- train_transaction.csv    652  MB
                                    ───────
                                   ~1.3 GB
```

Every file size matches the spec's expected ranges
(`train_transaction.csv` ≈ 652 MB, `train_identity.csv` ≈ 25 MB,
both `test_*` present). The idempotent re-run under 0.2.a confirmed
every hash still matches the recorded fingerprint; no re-download
was needed.

Manifest head (as required by the spec's verification step):

```json
{
  "source": "kaggle:ieee-fraud-detection",
  "downloaded_at": "2026-04-18T20:28:29.000906+00:00",
  "schema_version": 1,
  "files": {
    "sample_submission.csv": {
      "sha256": "50d7e0d6fcfc6e498efc297001f252101512ccdcb34aefbde6db98f8242a3626",
      "bytes": 6080314
    },
    "test_identity.csv": {
      "sha256": "3e5978cb13ca5e72f52babc4349ae0125e14b87ca8bfabe952ab67bb4ff1e10b",
      "bytes": 25797161
    },
    "test_transaction.csv": {
      "sha256": "2a8e51f1d335a86025d2b7f45beb9b78d0ab1edd726ef531d8b71a8a0065c011",
      ...
```

`.gitignore` (lines 31–37) preserves the double-negation so
`MANIFEST.json` is tracked while every `data/raw/*.csv` stays ignored:

```
!/data/raw/
/data/raw/*
!/data/raw/MANIFEST.json
```

## Part 2 — Pandera schemas (pre-existing; audit only)

[src/fraud_engine/schemas/raw.py](../../src/fraud_engine/schemas/raw.py)
defines:

| Symbol | Purpose |
|---|---|
| `TransactionSchema` | Inbound contract for `train_transaction.csv` / `test_transaction.csv`. Enumerates required core columns (TransactionID, isFraud, TransactionDT, TransactionAmt, ProductCD, card1–card6, addr1/2, P/R_emaildomain) plus regex groups for C1..C14, D1..D15, M1..M9, V1..V339. |
| `IdentitySchema` | Inbound contract for `train_identity.csv` / `test_identity.csv`. Splits `id_01..id_38` into numeric (22 cols) and object (15 cols) groups based on the actual IEEE-CIS dtype layout; DeviceType / DeviceInfo as optional objects. |
| `MergedSchema` | Contract for the left-joined transaction ⟕ identity frame. Transaction-side required columns survive; identity-side columns become optional (IEEE-CIS identity coverage ≈ 24%). |
| `SCHEMA_VERSION` | Integer version pin, re-exported by `configs/schemas.yaml` and the download manifest. |

**Deviation from spec example:** the spec illustration uses the newer
`pa.DataFrameModel` + `Series[int]` style; the landed code uses the
classic `DataFrameSchema(columns={...})` dict-of-Column style. Both
are first-class pandera APIs; the dict form was chosen for the regex
column-group support (`r"^V\d{1,3}$"` as a single entry), which is
cleaner than 339 `Series` declarations. The module docstring already
documents this trade-off. Not re-touched — the file is battle-tested
against the real 1.3 GB Kaggle dataset via the lineage suite.

**Regex column groups:**

- `r"^C\d{1,2}$"` → C1..C14 (all nullable float)
- `r"^D\d{1,2}$"` → D1..D15 (all nullable float)
- `r"^M[12356789]$"` → M1, M2, M3, M5, M6, M7, M8, M9 (nullable T/F strings)
- `M4` → explicit three-way match column (M0/M1/M2 values, per IEEE-CIS docs)
- `r"^V\d{1,3}$"` → V1..V339 (nullable float)

The schema deliberately *does not* list every `id_*` column as the
spec permits; they are enumerated programmatically from two `Final`
lists (`_IDENTITY_NUMERIC_COLS`, `_IDENTITY_OBJECT_COLS`) so the file
does not bloat to 500+ lines.

## Part 3 — `configs/schemas.yaml` (pre-existing; audit only)

[configs/schemas.yaml](../../configs/schemas.yaml) registers four
schemas. The three covered by this prompt:

| Registry key | Version | Module symbol |
|---|---|---|
| `transactions` | 1 | `fraud_engine.schemas.raw.TransactionSchema` |
| `identity` | 1 | `fraud_engine.schemas.raw.IdentitySchema` |
| `merged` | 1 | `fraud_engine.schemas.raw.MergedSchema` |

(The fourth entry, `split_manifest`, belongs to Sprint 1 prompt 1.1
and is out of scope for 0.2.b.) The YAML header comments the
four-step schema-evolution discipline (bump version, update docstring,
update consumers, add migration test) matching CLAUDE.md §7.1.

**Deviation from spec example:** the spec suggests `version: "1.0.0"`
(semver string). The committed shape uses `version: 1` (integer)
because `SCHEMA_VERSION` in code is `Final[int] = 1` and the lineage
test compares integers. Semver is unnecessary here — we only care
about the major boundary at which a migration test is required.

## Part 4 — `tests/unit/test_raw_schemas.py` (**this prompt's gap-fill**)

Created at
[tests/unit/test_raw_schemas.py](../../tests/unit/test_raw_schemas.py).
Seven tests covering both positive and negative paths:

| # | Test | Asserts |
|---|---|---|
| 1 | `test_transaction_schema_accepts_valid_sample` | Happy path — 5-row frame with every required column + one regex-group representative validates without raising. |
| 2 | `test_identity_schema_accepts_valid_sample` | 3-row identity frame validates. |
| 3 | `test_transaction_schema_rejects_negative_amount` | `TransactionAmt = -5.0` raises `pandera.errors.SchemaError` matching `TransactionAmt`. |
| 4 | `test_transaction_schema_rejects_unknown_product_cd` | `ProductCD = "Z"` (outside {C,H,R,S,W}) raises, match `ProductCD`. |
| 5 | `test_transaction_schema_rejects_isfraud_outside_binary` | `isFraud = 2` raises, match `isFraud`. |
| 6 | `test_transaction_schema_rejects_duplicate_transaction_id` | Two rows share a TransactionID → raises, match `TransactionID`. |
| 7 | `test_identity_schema_rejects_duplicate_transaction_id` | Uniqueness invariant also on identity side. |

**Deviation from the spec's minimum list:** the spec lists four
bullets; the file has seven tests. The extras are (2) identity happy
path and (7) identity uniqueness — both are one-liner complements to
the transaction-side tests and double the coverage of `IdentitySchema`
for free.

**Why a separate file from `test_raw_loader.py`:** the loader tests
validate *through* the read → coerce → validate pipeline, which
conflates CSV-parsing bugs with schema bugs. The new file validates
schemas directly against in-memory DataFrames so a failure cleanly
points at the contract, not the loader.

## Verification

All commands run clean after the new test file landed.

### `uv run pytest tests/unit/test_raw_schemas.py -v`

```
7 passed, 14 warnings in 3.38s
```

Coverage on `src/fraud_engine/schemas/raw.py` from this file alone:
**100%** (20 statements, 4 branches, 0 missing).

### `uv run mypy src/fraud_engine/schemas`

```
Success: no issues found in 2 source files
```

(Two files: `__init__.py` + `raw.py`.)

### `uv run ruff check tests/unit/test_raw_schemas.py` + `ruff format --check`

```
All checks passed!
1 file already formatted
```

### Full unit suite (no regressions)

```
149 passed, 28 warnings in 8.72s
```

Delta from Gate 0.1 (142 passed) → 149 passed matches exactly the 7
new tests. Existing tests unchanged.

### `ls -lh data/raw/` (spec verification command)

```
total 1.3G
-rw-r--r-- 1 dchit dchit  864 Apr 21 12:39 MANIFEST.json
-rw-r--r-- 1 dchit dchit 5.8M Apr 18 16:28 sample_submission.csv
-rw-r--r-- 1 dchit dchit  25M Apr 18 16:28 test_identity.csv
-rw-r--r-- 1 dchit dchit 585M Apr 18 16:28 test_transaction.csv
-rw-r--r-- 1 dchit dchit  26M Apr 18 16:28 train_identity.csv
-rw-r--r-- 1 dchit dchit 652M Apr 18 16:28 train_transaction.csv
```

Total ≈ 1.3 GB, well above the spec's expected ~700 MB (spec under-
counted by ignoring the two `test_*` files). Every file dated
2026-04-18 16:28, matching the manifest's `downloaded_at`.

## Files changed this prompt

| File | Change |
|---|---|
| [tests/unit/test_raw_schemas.py](../../tests/unit/test_raw_schemas.py) | **New.** 7 tests, 144 lines; covers positive + 4 negative paths across `TransactionSchema` and `IdentitySchema`. |

Nothing else was touched.

## Acceptance checklist (spec)

- [x] `make data-download` produced all five CSVs + `MANIFEST.json`
- [x] Raw files gitignored, `MANIFEST.json` tracked
- [x] `src/fraud_engine/schemas/raw.py` exists with `TransactionSchema`, `IdentitySchema`, and merged variant
- [x] Full ~400-column enumeration avoided via regex groups / programmatic lists; trade-offs documented in module docstring
- [x] `configs/schemas.yaml` registers `transactions`, `identity`, `merged` with version pins and module symbols
- [x] `tests/unit/test_raw_schemas.py` covers valid sample, negative amount, wrong ProductCD, bad isFraud, duplicate TransactionID
- [x] All rejections raise `pandera.errors.SchemaError` (verified via `pytest.raises`)
- [x] `uv run pytest tests/unit/test_raw_schemas.py -v` → 7 passed
- [x] `uv run mypy src/fraud_engine/schemas` → clean
- [x] `ls -lh data/raw/` confirms ~1.3 GB (spec said ~700 MB but missed the test_* pair)

## Non-goals (deferred / out of scope)

- **Rewriting schemas in the `pa.DataFrameModel` style.** The existing
  `DataFrameSchema` form handles regex column groups more cleanly
  for this dataset and is already battle-tested against real Kaggle
  CSVs via the lineage suite.
- **Migrating `configs/schemas.yaml` version field to semver strings.**
  Integer version matches the in-code `SCHEMA_VERSION` and is already
  wired into the lineage tests.
- **Re-downloading data.** Existing hashes still match; forcing a
  re-fetch would cost 1.3 GB of bandwidth for no change.
- **Git action.** CLAUDE.md §2.
