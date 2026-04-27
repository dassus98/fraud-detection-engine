# Sprint 2 — Prompt 2.1.c Report: Tier-1 Email + Missing-Indicator generators

**Date:** 2026-04-27
**Branch:** `sprint-2/prompt-2-1-c-tier1-email-and-missing` (off `main` at `3c86664`, post-2.1.b)
**Status:** all verification gates green — `make format` (53 files unchanged after one auto-format pass), `make lint` (All checks passed), `make typecheck` (26 source files unchanged — no new module, only additions to `tier1_basic.py`), `uv run pytest tests/unit/test_tier1_email_missing.py -v` (18 passed in 1.68 s), `make test-fast` (259 passed, was 241 — +18 net).

## Summary

Third concrete tier-1 feature module. Two more generators land in `src/fraud_engine/features/tier1_basic.py` (additions only — `AmountTransformer` + `TimeFeatureGenerator` from 2.1.b are untouched):

- **`EmailDomainExtractor`** — splits `P_emaildomain` and `R_emaildomain` into `{provider, tld}`; flags `is_free` and `is_disposable` against curated lists from a new YAML config. Produces 4 columns per email column = **8 features total**. Stateless `fit` (provider lists are loaded at `__init__`); `transform` does null-safe `rsplit(".", 1)` + `pd.Int8Dtype` (nullable Int8) flags so null source rows propagate as `<NA>`.
- **`MissingIndicatorGenerator`** — learns at fit time which columns exceed a missingness threshold (default 5%, configurable via the same YAML); at transform time emits `is_null_{col}` indicator columns for that learned set, even if the column is fully present in the transform frame. Schema-drift guard: column absent at transform → all-1s indicator (the column is "missing for every row").

Plus a new config file `configs/email_providers.yaml` (~25 free providers + ~15 disposable + the missing threshold). Two new dependencies added via `uv add`: `pyyaml` (runtime) + `types-pyyaml` (dev).

## Spec vs. actual

| Spec element | Implementation | Status |
|---|---|---|
| `EmailDomainExtractor` splits into `{provider, tld}` | `_split_domain` does `as_str.str.rsplit(".", 1, expand=True)`; null-safe via `pd.NA` mask | ✓ |
| `is_disposable_email`, `is_free_provider` flags | `{col}_is_disposable` + `{col}_is_free` per email column (4 columns × 2 columns = 8) | ✓ |
| Curated list in `configs/email_providers.yaml` | Created (~25 free + ~15 disposable) | ✓ |
| Nulls passed through as-is | All 4 derived fields are `<NA>` for null source rows; `pd.Int8Dtype` for flag columns preserves `<NA>` (not `0`) | ✓ |
| `MissingIndicatorGenerator` >5% threshold | Default loaded from YAML (`missing_indicator_threshold: 0.05`); explicit constructor override supported | ✓ |
| Add `is_null_{col}` per column | One indicator per learned column; `target_columns` sorted alphabetically for deterministic manifest ordering | ✓ |
| Learns columns at fit; transforms same set for val/test | `fit` populates `target_columns`; `transform` emits the same set even when val has zero nulls in those columns | ✓ |
| Test: known inputs produce known outputs | `TestEmailDomainExtractorSpec` has 7 known-input tests covering free / disposable / unknown / null / explicit-override / both-columns / default-config-load | ✓ |
| Test: config-driven threshold changes behavior | `test_config_driven_threshold_changes_behavior` (threshold=0.01 vs 0.05 on the same frame → different `target_columns`) | ✓ |

**Gap analysis: zero substantive gaps.** Two intentional design choices documented in class docstrings:

1. **Sharing `email_providers.yaml` for the missing-indicator threshold** is the spec literal. Mildly weird semantically (threshold isn't an email concern) but follows the wording. A future cleanup could split into `email_providers.yaml` + `tier1_features.yaml`; out of scope.
2. **Schema-drift handling in `MissingIndicatorGenerator.transform`** — column absent at transform → emit `is_null_{col} = 1` (all-1s indicator) rather than raise. Stricter alternative was rejected because Sprint 5 serving may legitimately ingest frames whose shape varies; the lenient form preserves the feature surface.

## Decisions worth flagging

### Decision 1 — `pd.Int8Dtype` for flag columns (nullable Int8)

The flag columns `{col}_is_free` and `{col}_is_disposable` are `pd.Int8Dtype` (nullable Int8) so a null source row produces `<NA>`, not `0`. Tree models (LightGBM, XGBoost) handle nullable ints; defaulting to `0` would silently overwrite the "unknown" semantics. Without this distinction, a downstream model can't separate "we don't know if this email is free" from "this email is definitely not free."

### Decision 2 — `_split_domain` differentiates null source from no-dot source

A naive `tld.fillna("")` clobbers null-source-row TLDs from `<NA>` to `""`, losing the null-passes-through contract. The fix uses a two-stage mask:

```python
null_source = as_str.isna()
no_dot_with_value = ~null_source & tld.isna()
tld = tld.mask(no_dot_with_value, "")
```

Result:
- `"gmail.com"` (with dot, non-null) → `provider="gmail"`, `tld="com"`
- `"weirdvalue"` (no dot, non-null) → `provider="weirdvalue"`, `tld=""`
- `<NA>` (null source) → `provider=<NA>`, `tld=<NA>`

This was caught by the `test_null_passes_through_as_na` test on the first run; the initial impl had used `fillna("")` indiscriminately and was corrected before commit.

### Decision 3 — Config loader is module-level helper, not class method

`_resolve_default_config_path()` and `_load_yaml(path)` are module-level free functions. Both generators call them. Keeps the configuration surface in one place; subclasses or alternative configs can re-use without coupling to a specific class.

## Test inventory

18 tests total (15 in 4 classes per the plan + 3 added during implementation for explicit coverage):

### `TestEmailDomainExtractorSpec` (7 tests)

| # | Name | Asserts |
|---|---|---|
| 1 | `test_known_free_provider` | `gmail.com` → provider=`gmail`, tld=`com`, is_free=`1`, is_disposable=`0` |
| 2 | `test_known_disposable_provider` | `guerrillamail.com` → is_disposable=`1`, is_free=`0` |
| 3 | `test_unknown_provider` | `weirdvalue.io` → provider=`weirdvalue`, tld=`io`, both flags `0` |
| 4 | `test_null_passes_through_as_na` | `pd.NA` source → all 4 derived fields `<NA>` |
| 5 | `test_explicit_lists_override_yaml` | Custom `free_providers={"unusual-free.com"}` flags it as free |
| 6 | `test_handles_both_email_columns` | Both `P_emaildomain` and `R_emaildomain` produce 4 derived columns each |
| 7 | `test_email_loads_default_config` | **Only** test that exercises the on-disk YAML: `EmailDomainExtractor()` with no args → `gmail.com` is_free=`1` |

### `TestMissingIndicatorGeneratorSpec` (6 tests)

| # | Name | Asserts |
|---|---|---|
| 8 | `test_learns_columns_above_threshold` | threshold=`0.05`, col_a=`10%`, col_b=`4%`, col_c=`0%` → `target_columns == ["col_a"]` |
| 9 | `test_transforms_same_set_for_val` | Fit on train (col_a 10% missing); val with 0% missing → `is_null_col_a` still emitted (all zeros) |
| 10 | `test_config_driven_threshold_changes_behavior` | threshold=`0.01` → `{col_a, col_b}`; threshold=`0.05` → `{col_a}` |
| 11 | `test_transform_before_fit_raises` | `AttributeError` matching `/must be fit/` |
| 12 | `test_target_columns_sorted_alphabetically` | `target_columns == sorted(target_columns)` |
| 13 | `test_schema_drift_emits_all_ones` | Drop col_a from val; `is_null_col_a` is all-1s indicator |

### `TestContractCompliance` (4 tests)

| # | Name | Asserts |
|---|---|---|
| 14 | `test_email_feature_names` | `len(get_feature_names()) == 8` for both email columns; all 8 expected names present |
| 15 | `test_email_rationale_non_empty` | rationale length > 50 chars |
| 16 | `test_missing_feature_names_pre_and_post_fit` | Pre-fit: `[]`; post-fit: `["is_null_col_a"]` |
| 17 | `test_missing_rationale_non_empty` | rationale length > 50 chars |

### `TestPipelineIntegration` (1 test)

| # | Name | Asserts |
|---|---|---|
| 18 | `test_email_and_missing_in_pipeline` | `FeaturePipeline([EmailDomainExtractor(), MissingIndicatorGenerator(threshold=0.05)]).fit_transform(df)` produces all expected feature columns |

## Files changed

| File | Type | LOC | Purpose |
|---|---|---|---|
| `configs/email_providers.yaml` | new | 60 | ~25 free + ~15 disposable + `missing_indicator_threshold: 0.05` |
| `src/fraud_engine/features/tier1_basic.py` | modified | +280 | `EmailDomainExtractor` + `MissingIndicatorGenerator` + 2 module helpers + 3 new constants; existing `AmountTransformer` + `TimeFeatureGenerator` untouched |
| `tests/unit/test_tier1_email_missing.py` | new | 230 | 18 tests across 4 classes + 2 inline helpers |
| `pyproject.toml` | modified | +2 lines | `uv add pyyaml` (runtime) + `uv add --dev types-pyyaml` (dev) |
| `uv.lock` | modified | (auto) | `uv add` regeneration |
| `sprints/sprint_2/prompt_2_1_c_report.md` | new | this file | Completion report |

Total source diff: ~570 LOC (production + tests + config + report). Production additions are 280 LOC; test additions are 230 LOC.

## Verification — verbatim output

### 1. `make format`
```
uv run ruff format src tests scripts
53 files left unchanged
```
(After 1 auto-format pass on the test file's hypothesis-style imports.)

### 2. `make lint`
```
uv run ruff check src tests scripts
All checks passed!
```

### 3. `make typecheck`
```
uv run mypy src
Success: no issues found in 26 source files
```
(Unchanged from 2.1.b — no new modules, only additions to `tier1_basic.py`.)

### 4. `uv run pytest tests/unit/test_tier1_email_missing.py -v`
```
18 passed, 14 warnings in 1.68s
```

### 5. `make test-fast`
```
259 passed, 34 warnings in 6.55s
```
(Was 241 passed pre-2.1.c; +18 net for the new tests.)

## Surprising findings

1. **The `tld.fillna("")` bug was caught on the first test run** — `test_null_passes_through_as_na` failed with `assert pd.isna(out["P_emaildomain_tld"].iloc[0])` because `fillna("")` clobbered null-source TLDs to `""`. Fix: differentiate null source from no-dot source via `as_str.isna()` mask. Now all 18 tests pass and the null-passes-through contract holds.
2. **`pd.Int8Dtype` for flag columns** allows `<NA>` to propagate from null source rows, but requires `pd.NA` (not `np.nan`) when explicitly setting null values. The interaction with `astype("Int8").mask(null_mask, pd.NA)` works cleanly; the alternative (using `np.nan`) would silently coerce the column to `Float64`.
3. **`yaml.safe_load` requires PyYAML to be a runtime dep**, not just dev. Both feature generators read the YAML at production runtime (Sprint 5 serving), so `pyyaml` goes in `[project] dependencies`, not `[project.optional-dependencies.dev]`. `types-pyyaml` is dev-only (mypy stubs).
4. **`ruff format` reformatted exactly 1 file** on the first run — the test file's import block (combined `from hypothesis import` lines was the same auto-fix as 2.1.b). No other files needed reformatting; the new generators in `tier1_basic.py` were format-clean on first pass.

## Deviations from the spec

1. **`{col}_is_free` and `{col}_is_disposable`** instead of bare `is_free_provider` / `is_disposable_email`. The spec used the bare names but didn't specify multi-column handling; the column-prefix form is necessary because both `P_emaildomain` and `R_emaildomain` get flags, and the prefix disambiguates them in the manifest.
2. **`pd.Int8Dtype` (nullable Int8) for flags** rather than plain `int` (which can't represent `<NA>`). Required to satisfy the spec's "Nulls passed through as-is" semantics for the flag columns.
3. **Schema-drift handling emits all-1s** rather than raising. Documented in class docstring; the lenient form supports Sprint 5 serving where input shape may vary.

## Acceptance checklist

- [x] Branch `sprint-2/prompt-2-1-c-tier1-email-and-missing` created off `main` (`3c86664`) **before any edits**
- [x] `uv add pyyaml` + `uv add --dev types-pyyaml` (both reflected in `pyproject.toml` and `uv.lock`)
- [x] `configs/email_providers.yaml` created (~25 free + ~15 disposable + threshold)
- [x] `src/fraud_engine/features/tier1_basic.py` extended with `EmailDomainExtractor` + `MissingIndicatorGenerator`
- [x] `tests/unit/test_tier1_email_missing.py` created (18 tests across 4 classes)
- [x] `make format` returns 0
- [x] `make lint` returns 0
- [x] `make typecheck` returns 0 (26 src files, unchanged)
- [x] `make test-fast` returns 0 (259 passed, was 241 — +18)
- [x] `uv run pytest tests/unit/test_tier1_email_missing.py -v` returns 0 (18 passed)
- [x] `sprints/sprint_2/prompt_2_1_c_report.md` written (this file)
- [x] No source files outside the listed set are modified

Verification passed. Ready for John to commit on `sprint-2/prompt-2-1-c-tier1-email-and-missing`.
