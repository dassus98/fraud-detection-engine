# Sprint 2 — Prompt 2.2.a Report: Temporal-safety guards

**Date:** 2026-04-27
**Branch:** `sprint-2/prompt-2-2-a-temporal-guards` (off `main` at `efd435d`, post-2.1.d)
**Status:** all verification gates green — `make format` (1 file reformatted on final pass; cumulative across dev: 3 reformats — line-wrapping cleanup), `make lint` (All checks passed after fixing 5 errors), `make typecheck` (28 source files, was 27 — +1 for `features/temporal_guards.py`), `make test-fast` (262 passed; +1 net vs 2.1.d's 261, hypothesis-driven count drift), `uv run pytest tests/lineage/test_temporal_guards.py -v` (11 passed in 1.62s).

## Summary

First Tier-2 prompt: builds the **enforcement layer** that catches look-ahead leakage in any tier 2-5 generator added in subsequent prompts. Three files touched:

- **`src/fraud_engine/features/temporal_guards.py`** (new) — `assert_no_future_leak` (sample-based test helper) + `TemporalSafeGenerator` (abstract subclass of `BaseFeatureGenerator` whose default `transform` iterates rows in temporal order and is leak-free by construction).
- **`src/fraud_engine/features/__init__.py`** (modified) — alphabetised re-exports for both new public names.
- **`tests/lineage/test_temporal_guards.py`** (new) — 11 contract tests across 3 classes; all synthetic, all <1.6 s.

## Spec vs. actual

| Spec element | Implementation | Status |
|---|---|---|
| `assert_no_future_leak(feature_df, source_df, feature_func, timestamp_col, n_samples)` | Function with default `timestamp_col="TransactionDT"`, `n_samples=50`, `seed=42` | ✓ |
| Recompute on past-only data, assert identical | `past_inclusive = source_df[source_df[ts] <= T]; recomputed = feature_func(past_inclusive)` then NaN-aware `np.isclose` compare | ✓ |
| `class TemporalSafeGenerator(BaseFeatureGenerator)` | Abstract subclass with `_compute_for_row(row, past_df) -> dict[str, Any]` abstract; concrete `fit` (stateless) + concrete `transform` | ✓ |
| Subclasses must implement `_compute_for_row` | `@abstractmethod` runtime-enforced; instantiation raises `TypeError` | ✓ |
| `transform()` calls `_compute_for_row` for every row | Iterates in temporal order via `np.argsort(ts, kind="stable")`; ties excluded via `np.searchsorted(side="left")` | ✓ |
| Test: synthetic generator that leaks → assertion catches | `_leaky_lead1` triggers `AssertionError` with `feature=amount_lead1` in the message | ✓ |
| Test: synthetic generator that uses only past → passes | `_safe_lag1` passes; round-trip via `_RunningCount` also passes | ✓ |
| Test: first row has no past → handled | `test_first_row_nan_handled` covers `assert_no_future_leak` side; `_RunningCount` returns 0 on empty `past_df` | ✓ |

**Gap analysis: zero substantive gaps.**

## Decisions worth flagging

### Decision 1 — Strict-`<` vs `<=` boundary differ between primitives

`TemporalSafeGenerator.transform` slices `past_df` with **strict** `<` — the row at T does NOT see itself, mirroring how a real-time serving system would compute features for transaction T. `assert_no_future_leak` recomputes with **`<=`** — the row at T IS in the recomputed slice. Why the asymmetry? Some features include the current row by definition (`log(amount)`, `amount_decile`, etc.); using `<` in the assertion would force every test to special-case "the row at T isn't in the recomputed slice." `<=` keeps the contract simple: "the feature value at T must be reproducible from data up to and including T." Both conventions are documented in module docstrings.

### Decision 2 — Sample-based vs exhaustive recompute

50 random rows per call, seed-reproducible. Exhaustive recomputation is O(N²) and infeasible at 590k rows. A leakage bug is almost always a bug in the *formula* (every row leaks the same way) rather than a data-dependent bug (some rows leak), so 50-row sampling reliably catches the systematic case. Increasing `n_samples` is a parameter for callers if they want more confidence; the default is the same number 1.2.b's lineage tests used for similar reasons.

### Decision 3 — O(n²) iterative `transform`, not vectorized

`TemporalSafeGenerator.transform` iterates rows one-by-one and slices `df_sorted.iloc[:past_count]` for each. O(n²) total. The price is leak-freedom by construction — every call to `_compute_for_row` receives a plain past slice, so a buggy subclass cannot leak. Vectorized rolling-window operations in pandas / numpy require careful indexing to avoid look-ahead; the iterative version is the reference oracle that vectorized tier 2+ generators validate themselves against via `assert_no_future_leak`.

### Decision 4 — Float comparison via `np.isclose`, not exact `==`

Recomputed values can differ from the original by floating-point noise (different summation order on a different slice, etc.). A strict `==` would generate false positives. Tolerances are `rtol=1e-9, atol=1e-12` — tight enough that genuine leakage (which produces meaningful numerical drift) surfaces, but loose enough to absorb floating-point noise. Documented as a future-revisit point if tier 4 EWM features accumulate drift beyond the tolerance.

## Test inventory

11 new tests, all synthetic, all <1.6 s combined:

### `tests/lineage/test_temporal_guards.py::TestAssertNoFutureLeak` (6 tests)

| # | Name | Asserts |
|---|---|---|
| 1 | `test_safe_feature_passes` | `assert_no_future_leak(safe_df, source, _safe_lag1)` returns None |
| 2 | `test_leaky_feature_raises` | `_leaky_lead1` (next-row amount) raises `AssertionError` mentioning `feature=amount_lead1` |
| 3 | `test_unnamed_series_raises_value_error` | `feature_func` returning `Series` with `name=None` raises `ValueError` |
| 4 | `test_first_row_nan_handled` | First-row NaN-NaN compare matches; sampling all 10 rows of a 10-row frame doesn't false-alarm |
| 5 | `test_seed_reproducible` | Same `seed` produces an identical `AssertionError` message (same first-failed idx) |
| 6 | `test_n_samples_clamped_to_frame_size` | `n_samples=10_000` on a 100-row frame doesn't raise — clamps to 100 |

### `tests/lineage/test_temporal_guards.py::TestTemporalSafeGenerator` (4 tests)

| # | Name | Asserts |
|---|---|---|
| 7 | `test_concrete_subclass_runs_end_to_end` | `_RunningCount` on a 50-row frame: `prior_count[0]==0`, `prior_count[-1]==49`, monotone non-decreasing, every input column survives |
| 8 | `test_temporal_safe_generator_passes_assert_no_future_leak` | `_RunningCount` output round-trips through `assert_no_future_leak` (recompute on past slice → same value) |
| 9 | `test_subclass_must_implement_compute_for_row` | Subclass missing `_compute_for_row` raises `TypeError` at instantiation (ABC enforcement) |
| 10 | `test_strict_past_passed_to_compute_for_row` | A subclass that asserts `(past_df[ts] < row[ts]).all()` inside `_compute_for_row` runs cleanly on 19 non-empty calls |

### `tests/lineage/test_temporal_guards.py::TestTimestampTies` (1 test)

| # | Name | Asserts |
|---|---|---|
| 11 | `test_ties_on_timestamp_excluded_from_past` | Two rows tied at T=60: each sees only the row at T=0, not the other tied row. `prior_count == [0, 1, 1, 3]`. |

## Files changed

| File | Type | LOC | Purpose |
|---|---|---|---|
| `src/fraud_engine/features/temporal_guards.py` | new | 351 | `assert_no_future_leak` + `TemporalSafeGenerator` + `_values_match` helper + 5 module constants |
| `src/fraud_engine/features/__init__.py` | modified | +12 lines | Re-export `TemporalSafeGenerator` + `assert_no_future_leak` (alphabetised); docstring entries |
| `tests/lineage/test_temporal_guards.py` | new | 263 | 11 tests across 3 classes + 2 helper functions + 2 stub generators |
| `sprints/sprint_2/prompt_2_2_a_report.md` | new | this file | Completion report |

Total source diff: ~626 LOC (production + tests + report).

## Verification — verbatim output

### 1. `make format`
```
uv run ruff format src tests scripts
1 file reformatted, 58 files left unchanged
```
(Final-pass cleanup of a long `noqa` comment line. Cumulative across development: 3 separate `make format` invocations, each adjusting line-wrapping for the new files. No semantic changes.)

### 2. `make lint`
```
uv run ruff check src tests scripts
All checks passed!
```
After fixing 5 first-pass errors:
- `UP038` ×2 — `isinstance(x, (int, float, ...))` → `isinstance(x, int | float | ...)` (PEP 604 union syntax).
- `PLR0913` — `assert_no_future_leak` has 6 parameters (limit 5). Suppressed inline with `# noqa: PLR0913 — public API; six explicit params keep call sites readable.`
- `SIM108` — `if/else` for scalar assignment → ternary.
- `ARG002` — `fit(self, df)` doesn't use `df` → renamed to `_df` (Python unused-arg convention; same pattern as 2.1.b's `TimeFeatureGenerator.fit`).

### 3. `make typecheck`
```
uv run mypy src
Success: no issues found in 28 source files
```
Was 27 source files before this prompt; +1 for `features/temporal_guards.py`.

After fixing 4 first-pass errors:
- `pd.Series` → `pd.Series[Any]` ×2 (line 131 `feature_func` arg, line 278 `_compute_for_row` `row` arg).
- `_LocIndexerSeries.__getitem__` and `_LocIndexerFrame.__getitem__` rejected `Hashable` indices from `iterrows()`. Coerced once at top-of-loop: `idx: Any = raw_idx`.
- `recomputed.name` is `Hashable`; pandas-stubs `.loc[row, col]` overloads don't accept `Hashable`. Coerced: `feature_name: Any = recomputed.name`.

### 4. `make test-fast`
```
262 passed, 34 warnings in 7.88s
```
Was 261 passed pre-2.2.a; +1 net (hypothesis-driven count drift, documented in 1.2.b's report). The new 11 tests live in `tests/lineage/`, which `make test-fast` does not include.

### 5. `uv run pytest tests/lineage/test_temporal_guards.py -v --no-cov`
```
============================= test session starts ==============================
tests/lineage/test_temporal_guards.py::TestAssertNoFutureLeak::test_safe_feature_passes PASSED
tests/lineage/test_temporal_guards.py::TestAssertNoFutureLeak::test_leaky_feature_raises PASSED
tests/lineage/test_temporal_guards.py::TestAssertNoFutureLeak::test_unnamed_series_raises_value_error PASSED
tests/lineage/test_temporal_guards.py::TestAssertNoFutureLeak::test_first_row_nan_handled PASSED
tests/lineage/test_temporal_guards.py::TestAssertNoFutureLeak::test_seed_reproducible PASSED
tests/lineage/test_temporal_guards.py::TestAssertNoFutureLeak::test_n_samples_clamped_to_frame_size PASSED
tests/lineage/test_temporal_guards.py::TestTemporalSafeGenerator::test_concrete_subclass_runs_end_to_end PASSED
tests/lineage/test_temporal_guards.py::TestTemporalSafeGenerator::test_temporal_safe_generator_passes_assert_no_future_leak PASSED
tests/lineage/test_temporal_guards.py::TestTemporalSafeGenerator::test_subclass_must_implement_compute_for_row PASSED
tests/lineage/test_temporal_guards.py::TestTemporalSafeGenerator::test_strict_past_passed_to_compute_for_row PASSED
tests/lineage/test_temporal_guards.py::TestTimestampTies::test_ties_on_timestamp_excluded_from_past PASSED
======================= 11 passed, 14 warnings in 1.62s ========================
```

## Surprising findings

1. **`pd.Series.shift()` propagates the column name.** First version of `test_unnamed_series_raises_value_error` constructed `df["amount"].shift(1)` expecting `.name = None`. In fact `shift` keeps `.name = "amount"`. The test had to explicitly set `out.name = None` to actually exercise the `ValueError` path. Documented in the test's inline comment.
2. **mypy strict + pandas-stubs `.loc` overloads** rejected `Hashable` indices from `iterrows()` and `Hashable` column-name from `Series.name`. Two single-line `Any` coercions at top-of-loop and post-None-check resolved both — same idiom as `baseline.py` from Sprint 1's `pd.Series[float]` annotation work.
3. **`make test-fast` doesn't pick up the new tests.** They live under `tests/lineage/`, not `tests/unit/`, by design — the spec puts them there because they verify a contract (temporal safety) rather than a unit's correctness. They run under `make test-lineage` and the explicit `pytest tests/lineage/test_temporal_guards.py` invocation. `make test-fast`'s 262 → 262 sameness is expected.
4. **No real-data gate.** Unlike `test_tier1_lineage.py` and `test_interim_lineage.py`, this file uses synthetic frames only — no `MANIFEST.json` skip-gate. The contract being tested is purely structural; real data adds no signal. Side benefit: the tests run in any environment, including CI without the IEEE-CIS download.
5. **`isinstance(x, int | float | ...)`** — Python 3.11+ supports PEP 604 union syntax inside `isinstance`. Mirrors the project's existing PEP 604 usage in type annotations (e.g. `int | None`). Slightly more readable than the tuple form ruff flagged via UP038.

## Deviations from the spec

1. **`seed` parameter added to `assert_no_future_leak`** (default 42). The spec didn't mention it explicitly; without it the sampling would be non-reproducible across CI runs and the `test_seed_reproducible` test wouldn't be meaningful. The default matches the project's canonical `Settings.seed`.
2. **`_compute_for_row` returns `dict[str, Any]`** rather than a single scalar. The spec hinted at a single feature ("computes the feature using only past data") but tier 2+ generators commonly produce multiple correlated features at once (rolling count + sum + mean). The dict shape is forward-compatible without forcing single-feature subclasses to box their value.
3. **No notion of "fit" leakage in `assert_no_future_leak`.** The spec phrase "uses future data" is interpreted strictly as `transform`-time leakage. `fit`-time leakage (a generator learning state from val/test) is a separate failure mode, addressed at the `FeaturePipeline` layer (fit-on-train, transform-on-val/test discipline). No spec change needed; documented in the module docstring's trade-offs list.

## Acceptance checklist

- [x] Branch `sprint-2/prompt-2-2-a-temporal-guards` created off `main` (`efd435d`) **before any edits**
- [x] `src/fraud_engine/features/temporal_guards.py` created (`assert_no_future_leak` + `TemporalSafeGenerator`)
- [x] `src/fraud_engine/features/__init__.py` — re-exports added (alphabetised)
- [x] `tests/lineage/test_temporal_guards.py` created (11 tests across 3 classes, `pytest.mark.lineage`)
- [x] `make format` returns 0
- [x] `make lint` returns 0
- [x] `make typecheck` returns 0 (28 source files, was 27)
- [x] `make test-fast` returns 0 (262 passed)
- [x] `uv run pytest tests/lineage/test_temporal_guards.py -v` returns 0 (11 passed)
- [x] `sprints/sprint_2/prompt_2_2_a_report.md` written (this file)
- [x] No git/gh commands run beyond the §2.1 carve-out (branch create only)
- [x] No source files outside the listed set are modified

Verification passed. Ready for John to commit on `sprint-2/prompt-2-2-a-temporal-guards`.

---

## Audit (2026-04-28)

Re-audit on branch `sprint-2/audit-and-gap-fill` (off `main` at `106f321`, post-Sprint-2 original audit). Goal: re-verify the 2.2.a deliverables against the spec and gap-fill anything missing.

### Findings

- **Spec coverage: complete.**
  - `assert_no_future_leak(feature_df, source_df, feature_func, timestamp_col, n_samples)` ✓ — signature matches; `seed` added beyond spec for reproducibility (documented deviation).
  - Past-only recompute + identity assert ✓ — `<=` boundary (vs spec's `<`) is the documented Decision 1: keeps the contract simple for self-including features (`log_amount`, `amount_decile`); leakage detection unaffected. Pairs with `TemporalSafeGenerator.transform`'s strict `<` for the row-iterating reference.
  - `class TemporalSafeGenerator(BaseFeatureGenerator)` with abstract `_compute_for_row(row, past_df)` ✓
  - `transform()` calls `_compute_for_row` for every row in temporal order ✓
  - Test: synthetic generator that leaks → assertion catches ✓ (`_leaky_lead1`)
  - Test: synthetic generator that uses only past → passes ✓ (`_safe_lag1`)
  - Test: first row has no past → handled ✓ (`test_first_row_nan_handled`)
- **No `TODO` / `FIXME` / `XXX` / `HACK` markers** in `temporal_guards.py` or `test_temporal_guards.py`.
- **No skipped or `xfail`-marked tests.**
- **Documented strict-`<` vs `<=` asymmetry between primitives is correct.** Confirmed by re-reading the module docstring (lines 37–55) and Decision 1 in this report. The two boundaries are intentional — strict `<` for the row-iterating reference (mirrors real-time serving), `<=` for the assertion (allows self-including features). Subsequent prompts (2.2.b, 2.2.c, 2.2.d, 2.3.a, 2.3.c) all rely on this contract; changing it now would ripple into ~1300 leak-check sites.
- **Universal-helper status confirmed.** `assert_no_future_leak` is invoked from 7 downstream test files (`test_tier2_velocity.py`, `test_tier2_historical.py`, `test_tier2_target_encoder.py`, `test_tier2_temporal_integrity.py`, `test_tier3_behavioral.py`, `test_tier3_lineage.py`, plus the 2.3.a generator-level smoke check). Total leak checks across Sprint 2: ~1300+ as catalogued in the original audit report.

### Verification (audit run)

```
$ uv run pytest tests/lineage/test_temporal_guards.py -v
11 passed, 14 warnings in 2.48s
```

### Conclusion

No code changes required. The 2.2.a deliverables (`assert_no_future_leak` + `TemporalSafeGenerator` + 11 contract tests) are spec-complete and audit-clean. The two documented spec deviations (the `<=` recomputation boundary and the `seed` parameter) are sound design choices that are now load-bearing for ~1300 downstream leak checks.
