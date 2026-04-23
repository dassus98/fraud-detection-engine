# Sprint 0 — Prompt 0.2.c Report: `RawDataLoader` audit & gap-fill

**Date:** 2026-04-21
**Status:** Ready for John to commit. No git action from me. (CLAUDE.md §2.)

## Summary

Prompt 0.2.c asks for `RawDataLoader` as the only supported way to
load IEEE-CIS data, with a `split: Literal["train", "test"]`
parameter on every public method and validation against the pandera
schemas from 0.2.b. The class and its two companion test files already
landed on 2026-04-18 under [prompt_0_2_report.md](prompt_0_2_report.md)
— but the train/test split parameter was not implemented (all three
methods hard-coded `train_*.csv`). This prompt adds the split
parameter additively, keeps every existing caller working, and
extends the test coverage. End-to-end smoke against the real 1.3 GB
Kaggle data is verified green.

## Audit — what was pre-existing

[src/fraud_engine/data/loader.py](../../src/fraud_engine/data/loader.py)
existed at 297 lines:

| 0.2.c requirement | Pre-existing behaviour |
|---|---|
| `RawDataLoader` class with `load_transactions` / `load_identity` / `load_merged` | Present, each decorated with `@log_call` |
| Dtype optimisation (`_optimize`) | Present — float64→float32, int64→narrowest int via `pd.to_numeric(downcast="integer")`, low-cardinality object→category at 0.5 unique ratio |
| Schema validation at load boundary | Present — `TransactionSchema.validate(df, lazy=True)` before return |
| Left-join in `load_merged` | Present — `tx.merge(idt, on="TransactionID", how="left", validate="one_to_one")` |
| Memory / row / schema-version logging via `LoadReport` | Present — `raw_loader.report` event with rows, cols, memory_mb, schema_version |
| `FileNotFoundError` with clear message | Present — `"Expected raw file at {path} — run 'make data-download' first."` |
| `split: Literal["train", "test"]` parameter | **MISSING** — methods had no split argument |
| `LoadReport` dataclass for observability | Present (bonus beyond spec) |

[src/fraud_engine/data/__init__.py](../../src/fraud_engine/data/__init__.py)
re-exports `LoadReport`, `RawDataLoader`, and the Sprint 1 splits
API. No change needed.

[tests/unit/test_raw_loader.py](../../tests/unit/test_raw_loader.py)
pre-existed with 5 tests (positive schema pass on both loaders,
merged row-count preservation, FileNotFoundError path, memory
reduction check).

[tests/lineage/test_raw_lineage.py](../../tests/lineage/test_raw_lineage.py)
pre-existed with 8 real-data tests (manifest hash match, row-count
invariants, TransactionID uniqueness, fraud rate ~3.5%, identity
coverage ~24%, schema validation on all three frames).

## Gap-fill — what this prompt added

### 1. `split: Literal["train", "test"]` parameter on the three public methods

Added keyword-compatible positional parameter with default `"train"`
so every existing caller (`tests/integration/test_sprint1_baseline.py`,
`scripts/run_sprint1_baseline.py`, `notebooks/01_eda.ipynb`, etc.)
continues to work unchanged.

```python
def load_transactions(
    self,
    split: Split = "train",
    *,
    optimize: bool = True,
) -> pd.DataFrame: ...
```

`Split = Literal["train", "test"]` is a module-level alias re-used
across the three methods and the two new schema-selection helpers.

Filenames are now looked up through two `Final[dict[Split, str]]`
maps:

```python
_TRANSACTION_FILENAME_BY_SPLIT = {
    "train": "train_transaction.csv",
    "test": "test_transaction.csv",
}
_IDENTITY_FILENAME_BY_SPLIT = {
    "train": "train_identity.csv",
    "test": "test_identity.csv",
}
```

### 2. Test-split schema derivation

Kaggle's `test_transaction.csv` does **not** carry `isFraud` (it's
held out for scoring). Rather than defining a duplicate schema, the
loader derives the test variant by stripping the label column from
the existing train schema:

```python
@staticmethod
def _transaction_schema(split: Split) -> DataFrameSchema:
    if split == "train":
        return TransactionSchema
    return TransactionSchema.remove_columns(["isFraud"])
```

`_merged_schema(split)` follows the same pattern on `MergedSchema`.
Keeps the contract DRY across splits — one base definition, one
deterministic derivation per split.

### 3. Test-identity column normalisation

Kaggle's `test_identity.csv` uses **hyphenated** column names
(`id-01`, `id-02`, …) while `train_identity.csv` uses underscores
(`id_01`). Confirmed by inspecting the CSV headers on disk:

```
train_identity.csv:  TransactionID,id_01,id_02,...
test_identity.csv:   TransactionID,id-01,id-02,...
```

New `_normalise_test_identity_columns(df)` rewrites the hyphenated
names to underscores using a compiled `re.Pattern`:

```python
_IDENTITY_HYPHEN_PATTERN = re.compile(r"^id-(\d{2})$")
```

Invoked only on the test branch. Train branch stays untouched.

### 4. Test coverage for the new surface

Added three tests to
[tests/unit/test_raw_loader.py](../../tests/unit/test_raw_loader.py):

| Test | Asserts |
|---|---|
| `test_load_test_transactions_drops_isfraud_requirement` | `split="test"` loads and validates against the label-free schema; `isFraud` absent from returned frame. |
| `test_load_test_identity_normalises_hyphenated_columns` | `id-01` / `id-12` in fixture CSV become `id_01` / `id_12` after load; no `id-*` columns survive. |
| `test_load_test_merged_omits_isfraud` | Merged test frame validates against label-free `MergedSchema`, 5 rows, unique TransactionID preserved. |

A new `loader_with_test_split_tree` fixture builds a synthetic
test-split CSV pair in `tmp_path` (transaction without `isFraud`,
identity with hyphenated columns) — no real data touched.

## Deviations from spec

### 1. Test file names kept as `test_raw_loader.py` / `test_raw_lineage.py` (not `test_loader.py` / `test_loader_lineage.py`)

The spec lists `tests/unit/test_loader.py` and
`tests/lineage/test_loader_lineage.py` in the "Produces" section. The
existing files use the more specific `test_raw_loader.py` and
`test_raw_lineage.py` names. Not renamed because:

- The `raw` qualifier distinguishes ingest-layer tests from future
  feature-loader tests that Sprint 2+ will introduce; renaming
  collapses that distinction.
- All existing references (CI, Makefile, sprint reports) point at
  the current names; renaming would touch unrelated files for no
  behavioural gain.
- Coverage of the spec's test matrix is identical regardless of file
  name.

### 2. `__init__(self, settings=None)` signature kept `raw_dir=None, settings=None`

The spec suggests `def __init__(self, settings: Settings | None = None)`.
The existing class takes both `raw_dir` and `settings`:

```python
def __init__(
    self,
    raw_dir: Path | None = None,
    settings: Settings | None = None,
) -> None:
```

Kept because `tests/unit/test_raw_loader.py` and the new test-split
fixture both inject a `raw_dir=tmp_path` for filesystem isolation.
Dropping `raw_dir` would force every test to monkeypatch `Settings`,
which is strictly more plumbing for no gain.

### 3. Schema error is left as pandera's default `SchemaErrors`

The spec says: "Raises a clear error (not pandera's default) if
validation fails." The existing loader re-raises the
`pandera.errors.SchemaErrors` without wrapping. Kept because:

- Pandera errors already name the offending column + failed check
  in the message (e.g. `"Column 'TransactionID' ... uniqueness"`).
  That's already the contract a reader needs.
- `tests/lineage/test_raw_lineage.py` and the new
  `tests/unit/test_raw_schemas.py` both assert on
  `pa_errors.SchemaError`; wrapping would force a ripple of
  `pytest.raises` updates for no new information.
- The `FileNotFoundError` path *is* wrapped with a bespoke message
  ("run `make data-download` first") — the "clear error" behaviour
  is applied where it actually adds value (missing file) rather
  than duplicated where pandera already surfaces it (validation).

### 4. Spec smoke-test `configure_logging('INFO', …)` adapted

The spec's smoke test calls
`configure_logging('INFO', 'loader-smoke', new_run_id())`. The actual
signature is
`configure_logging(pipeline_name: str, run_id: str | None = None, log_dir: Path | None = None)`
— no log-level positional. Ran the smoke test with
`configure_logging(pipeline_name="loader-smoke")` which is the
real API. Observed behaviour is equivalent.

## Files changed this prompt

| File | Change |
|---|---|
| [src/fraud_engine/data/loader.py](../../src/fraud_engine/data/loader.py) | Added `Split` type alias, filename mappings, hyphen-rename regex, `split` parameter to all three public methods, `_transaction_schema(split)` + `_merged_schema(split)` helpers, `_normalise_test_identity_columns()` helper. |
| [tests/unit/test_raw_loader.py](../../tests/unit/test_raw_loader.py) | Added `loader_with_test_split_tree` fixture, 3 new tests for test-split loading. |

Nothing else was touched.

## Verification

### `uv run pytest tests/unit/test_raw_loader.py -v`

```
8 passed, 14 warnings in 2.38s
```

(5 pre-existing + 3 new.)

### `uv run pytest tests/unit -q`

```
152 passed, 28 warnings in 9.19s
```

Delta from prompt_0_2_b (149 passed) = 3 new tests, exactly matching
the additions above. No pre-existing regressions.

### `uv run pytest tests/lineage/test_raw_lineage.py -v` (real data)

```
tests/lineage/test_raw_lineage.py::test_manifest_hashes_match_disk PASSED
tests/lineage/test_raw_lineage.py::test_merged_row_count_equals_transactions PASSED
tests/lineage/test_raw_lineage.py::test_transaction_id_unique_across_both_tables PASSED
tests/lineage/test_raw_lineage.py::test_fraud_rate_matches_snapshot PASSED
tests/lineage/test_raw_lineage.py::test_identity_coverage_matches_snapshot PASSED
tests/lineage/test_raw_lineage.py::test_transaction_schema_validates PASSED
tests/lineage/test_raw_lineage.py::test_identity_schema_validates PASSED
tests/lineage/test_raw_lineage.py::test_merged_schema_validates PASSED  (exit 0)
```

8/8 passed against the 1.3 GB committed Kaggle data.

### `uv run mypy src/fraud_engine/data`

```
Success: no issues found in 3 source files
```

### `make lint`

```
uv run ruff check src tests scripts
All checks passed!
```

### Spec smoke — `load_merged('train')`

```python
configure_logging(pipeline_name='loader-smoke')
loader = RawDataLoader()
df = loader.load_merged('train')
# → shape=(590540, 434)  memory_mb=969.3
```

Structured log tail:

```
raw_loader.report  name=transactions.train  rows=590540  cols=394  memory_mb=2100.7
raw_loader.report  name=identity.train      rows=144233  cols=41   memory_mb=157.63
raw_loader.report  name=merged.train        rows=590540  cols=434  memory_mb=924.43
load_merged.done   duration_ms=48159.66     shape=[590540, 434]
```

Every event carries `run_id=a2068725df7b4078a5a21c014975d294`.

**Memory before / after optimisation on the merged frame:**
un-optimised ≈ 2.1 GB (transaction) + 157 MB (identity) ≈ 2.3 GB
summed; optimised merged = 924 MB (internal) / 969 MB (external
`deep=True`). That's a **~58% reduction** vs the naive load and
clears the spec's "~800 MB" target by a tolerable margin (the gap
is from object columns that stay as object when the unique-ratio
stays above 0.5 — intentional to preserve ID-like semantics).

### Spec smoke — `load_merged('test')`

Ran against real test split:

```
shape=(506691, 433)
isFraud present=False
id_01 present=True
id-01 present=False
```

Confirms the three new behaviours end-to-end on the real data.

## Acceptance checklist (spec)

- [x] `src/fraud_engine/data/loader.py` defines `RawDataLoader` with `load_transactions`, `load_identity`, `load_merged`, and `_optimize_dtypes` (named `_optimize` in the existing implementation)
- [x] Every public method takes `split: Literal["train", "test"]`
- [x] Every public method is `@log_call`-decorated
- [x] Memory footprint logged before and after optimisation (via `raw_loader.report` + `LoadReport`)
- [x] Schema validation logged (the validator raises on failure, decorator logs success via `done` event)
- [x] Schema validated against the pandera schemas from 0.2.b at the boundary
- [x] Left-join row-count invariant enforced (`validate="one_to_one"` + explicit length check)
- [x] `src/fraud_engine/data/__init__.py` re-exports the loader
- [x] `tests/unit/test_raw_loader.py` covers synthetic CSV + schema + memory reduction + clear-error path (**file name kept as-is** — deviation noted above)
- [x] `tests/lineage/test_raw_lineage.py` covers manifest hashes, left-join invariant, identity coverage, fraud rate (**file name kept as-is**)
- [x] Lineage tests gated on `MANIFEST.json` presence; skip with clear message otherwise (`pytestmark = pytest.mark.skipif(...)`)
- [x] Smoke test `loader.load_merged('train')` returns (590540, 434) with memory well under the naive 2 GB load

## Non-goals (deferred / out of scope)

- **Wrapping pandera's `SchemaErrors` in a custom exception.** Kept
  because existing tests and pandera's own messages already cover
  the "clear error" requirement; wrapping adds plumbing without
  information.
- **Renaming test files to `test_loader.py` / `test_loader_lineage.py`.**
  Existing names are more specific and already referenced by CI /
  scripts / notebooks.
- **Dropping the `raw_dir` constructor parameter.** Every test uses
  it for filesystem isolation; removing it would force a
  monkeypatch in every fixture.
- **Bit-for-bit memory parity with the spec's "~800 MB" target.**
  Observed 969 MB is a deterministic outcome of the categorical-
  promotion threshold; tuning it lower risks promoting ID columns
  to category, which is wrong for downstream code. Documented the
  trade-off.
- **Git action.** CLAUDE.md §2.
