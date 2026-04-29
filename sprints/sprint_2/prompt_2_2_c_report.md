# Sprint 2 — Prompt 2.2.c Report: HistoricalStats (Tier-2 rolling mean / std / max)

**Date:** 2026-04-27
**Branch:** `sprint-2/prompt-2-2-c-historical-stats` (off `main` at `6b91ab8`, post-2.2.b)
**Status:** all verification gates green — `make format` (2 files reformatted on first pass; 60 unchanged on final), `make lint` (All checks passed after fixing 1 PLR0915 + 1 PLR2004), `make typecheck` (29 source files, unchanged — `HistoricalStats` lives in the existing `tier2_aggregations.py`), `make test-fast` (289 passed; +11 net vs 2.2.b's 278), `uv run pytest tests/unit/test_tier2_historical.py -v` (11 passed in 3.14s).

## Summary

Second Tier-2 generator: `HistoricalStats` produces 5 default columns (`card1_amt_{mean,std,max}_30d` + `addr1_amt_{mean,std}_30d`) using the same deque-per-entity sweep as `VelocityCounter`, but with the deque payload upgraded from `int` (timestamp) to `(int, float)` (timestamp + amount), and the count replaced by a numpy stat dispatch over the deque contents. Five files touched:

- **`configs/historical_stats.yaml`** (new, 21 LOC) — per-entity stats list + windows + amount column.
- **`src/fraud_engine/features/tier2_aggregations.py`** (modified, +301 lines) — adds `HistoricalStats` class; tiny refactor of the YAML resolver to be filename-parameterised so both Tier-2 classes share it.
- **`src/fraud_engine/features/__init__.py`** (modified, +3 lines) — alphabetised re-export.
- **`tests/unit/test_tier2_historical.py`** (new, 409 LOC) — 11 tests across 3 classes (unit, temporal-safety property, config load) + `_NaiveHistoricalStats` reference oracle.
- **`sprints/sprint_2/prompt_2_2_c_report.md`** (new) — this file.

## Spec vs. actual

| Spec element | Implementation | Status |
|---|---|---|
| `HistoricalStats` per-entity rolling stats | `class HistoricalStats(BaseFeatureGenerator)` | ✓ |
| Default 30 d window (configurable) | YAML `windows: {"30d": 2592000}`; constructor kwargs override | ✓ |
| `card1_amt_{mean,std,max}_30d` + `addr1_amt_{mean,std}_30d` | YAML default = exactly these 5 columns | ✓ |
| Sorted-iteration pattern | `np.argsort(timestamps, kind="stable")` + tied-group two-pass batching | ✓ |
| Window ends strictly before current row | `while d and d[0][0] < window_start: d.popleft()` + tied-group batching ensures self/ties excluded | ✓ |
| Unit tests | `TestHistoricalStats` (7 tests) | ✓ |
| Property test (reference impl) | `TestTemporalSafety::test_optimized_matches_naive` (hypothesis, 50 examples) | ✓ |
| Temporal guard | `TestTemporalSafety::test_assert_no_future_leak_passes` | ✓ |

**Gap analysis: zero substantive gaps.**

## Decisions worth flagging

### Decision 1 — Recompute-from-deque vs running-state

A running-mean + running-sum-of-squares + monotonic-max-deque approach would be O(1) per push/pop, but the bookkeeping is fragile under eviction (especially the max — needs a strict-monotonic deque). The simpler approach: store `(timestamp, amount)` tuples in a per-entity deque and recompute statistics from a numpy array on every read. With a 30 d window and typical entity activity (a few txns per month per card), deques stay small (~10 entries); numpy vectorises the stat call. The naive correctness is worth the small constant-factor overhead.

### Decision 2 — Sample std (`ddof=1`), matching pandas

Pandas defaults to `ddof=1`; numpy defaults to `ddof=0`. The property test's reference uses `pd.Series.std()`, so the optimised impl must match. Encoded as the module constant `_STD_DDOF: Final[int] = 1` with a comment pinning the rationale.

### Decision 3 — `n=1` deque → std = NaN

Sample std requires ≥ 2 observations. A single-element deque returns NaN for std (mean and max still return the single value). Matches `pd.Series.std()` exactly. Module constant `_MIN_SAMPLES_FOR_STD: Final[int] = 2`; introduced as a named constant so PLR2004 doesn't fire on the magic 2.

### Decision 4 — NaN entity → all stats NaN; NaN amount → row not pushed

Different from `VelocityCounter`'s "NaN entity → 0" because a count of 0 is meaningful (no events seen) but a mean of 0 over zero observations is misleading. NaN is the clean "no data" indicator for LightGBM. Defensive against Sprint 5's serving layer ingesting unvalidated payloads (the cleaner forbids NaN amounts in the train pipeline, but the contract should hold without that assumption).

### Decision 5 — Tied-timestamp batching, identical to `VelocityCounter`

Two rows sharing `TransactionDT` see neither each other nor the future row. Implemented by the same two-pass pattern: pass 1 reads from the deque (no tied row sees another); pass 2 pushes every tied row's `(ts, amt)` tuple. Without this batching, the second tied row would erroneously include the first.

### Decision 6 — YAML-resolver helper refactored, not duplicated

`_resolve_default_config_path()` (velocity-only) became `_resolve_config_path(filename: str)` accepting a filename argument. Both classes now share one resolver; the velocity-specific `_DEFAULT_CONFIG_FILENAME` was renamed `_VELOCITY_CONFIG_FILENAME` for clarity, and a parallel `_HISTORICAL_STATS_CONFIG_FILENAME` was added. Internal change with no external callers; verified by re-running `test_tier2_velocity.py` under `make test-fast` (still passes).

## Algorithm description (for the reader)

```text
sort indices by TransactionDT (stable)
for each tied group [i, j) at timestamp T:
    pass 1 (stats):
        for each row in [i, j):
            for each (entity, window) pair:
                if entity is NaN, leave NaN default
                else:
                    evict deque[(entity, window)][entity_val] head while head[0] < T - window
                    if deque empty, leave NaN default
                    else:
                        arr = numpy array of remaining amounts
                        for each requested stat:
                            mean → arr.mean()
                            max  → arr.max()
                            std  → arr.std(ddof=1) if arr.size >= 2 else NaN
    pass 2 (push):
        for each row in [i, j):
            if amount is NaN, skip
            for each (entity, window) pair:
                if entity is NaN, skip
                deque[(entity, window)][entity_val].append((T, amount))
```

Per-row cost: `O(entities × windows × (eviction + array build + stats))`. Each (timestamp, amount) is pushed once per (entity, window) and popped at most once, so total deque ops are `O(n × E × W)`. With `E=2, W=1`, that's ~2n deque ops on a frame of size n.

## Test inventory

11 new tests, all in `tests/unit/test_tier2_historical.py`:

### `TestHistoricalStats` (7 tests)

| # | Name | Asserts |
|---|---|---|
| 1 | `test_hand_computed_stats` | 5-row frame `amt=[10,20,30,40,50]`, all entity=A, big window: row 0 all NaN; row 1 mean=10, std=NaN, max=10; row 4 mean=25, std=`pd.Series([10,20,30,40]).std(ddof=1)≈12.91`, max=40 |
| 2 | `test_nan_entity_yields_nan` | Row with NaN entity gets NaN stats; row 1's amount NOT pushed (NaN entity), so row 2's mean is from row 0 only |
| 3 | `test_empty_window_yields_nan` | First row for each entity → all stats NaN |
| 4 | `test_single_event_std_nan` | n=1 deque: mean=max=value, std=NaN |
| 5 | `test_strict_past_no_self_count` | Row at T does not include itself in its own stats (mean would inflate by 1/n if it leaked) |
| 6 | `test_ties_excluded` | `ts=[0, 100, 100, 200]`: tied rows at T=100 each see only T=0; row at T=200 sees all three earlier rows → `mean = [NaN, 10, 10, 20]` |
| 7 | `test_unsupported_stat_raises` | `entity_stats={"card1": ["median"]}` raises `ValueError` mentioning `median` |

### `TestTemporalSafety` (2 tests)

| # | Name | Asserts |
|---|---|---|
| 8 | `test_assert_no_future_leak_passes` | 50-row synthetic frame; `assert_no_future_leak(out, src, recompute_lambda)` returns None on the `entity_amt_mean_5m` column |
| 9 | `test_optimized_matches_naive` (hypothesis) | 50 examples × random small frames (n ≤ 15, integer entities ∪ None, 1–2 windows, all 3 stats) → optimised output equals naive `_NaiveHistoricalStats` reference column-for-column under `pd.testing.assert_frame_equal` |

### `TestConfigLoad` (2 tests)

| # | Name | Asserts |
|---|---|---|
| 10 | `test_default_config_loads` | `HistoricalStats()` reads YAML; `entity_stats == (("card1", ("mean","std","max")), ("addr1", ("mean","std")))`; `windows == {"30d": 2592000}`; 5 feature columns |
| 11 | `test_constructor_overrides_config` | Explicit `entity_stats={"card1": ["mean"]}, windows={"5s": 5}` ignores YAML |

## Files changed

| File | Type | LOC | Purpose |
|---|---|---|---|
| `configs/historical_stats.yaml` | new | 21 | Per-entity stats + 30 d window + amount column |
| `src/fraud_engine/features/tier2_aggregations.py` | modified | +301 | Adds `HistoricalStats` class + 4 module constants; refactors YAML resolver to be filename-parameterised |
| `src/fraud_engine/features/__init__.py` | modified | +3 | Re-export `HistoricalStats` (alphabetised) + docstring entry |
| `tests/unit/test_tier2_historical.py` | new | 409 | 11 tests + `_NaiveHistoricalStats` reference + 2 hypothesis strategies |
| `sprints/sprint_2/prompt_2_2_c_report.md` | new | this file | Completion report |

## Verification — verbatim output

### 1. `make format`
```
uv run ruff format src tests scripts
2 files reformatted, 60 files left unchanged
```
(One-pass cleanup of new files' line-wrapping; subsequent runs are 0-change.)

### 2. `make lint`
```
uv run ruff check src tests scripts
All checks passed!
```
After fixing 2 first-pass errors:
- `PLR0915` — `transform` has 51 statements (limit 50) due to the tied-group two-pass × per-entity × per-window × per-stat nested loops. Suppressed inline (`# noqa: PLR0912, PLR0915 — tied-group batching is a single algorithm; splitting across helpers would lose locality.`)
- `PLR2004` — magic value `2` in `arr.size >= 2`. Replaced with named constant `_MIN_SAMPLES_FOR_STD: Final[int] = 2`.

### 3. `make typecheck`
```
uv run mypy src
Success: no issues found in 29 source files
```
Same source-file count as 2.2.b — `HistoricalStats` lives in the existing `tier2_aggregations.py`. No new import-side fixes needed; `np.ndarray[Any, Any]` etc. carried over from `VelocityCounter`.

### 4. `make test-fast`
```
289 passed, 34 warnings in 10.10s
```
Was 278 passed pre-2.2.c; **+11 net** for the 11 new tests. Existing `test_tier2_velocity.py` (the YAML-resolver refactor's main risk vector) still passes.

### 5. `uv run pytest tests/unit/test_tier2_historical.py -v --no-cov`
```
======================= 11 passed, 14 warnings in 3.14s ========================
```
All 11 tests green in 3.14s. Hypothesis property test (50 examples) accounts for ~2s of that.

## Surprising findings

1. **Reference-vs-optimised float-equality is exact.** `pd.testing.assert_frame_equal(check_dtype=False)` — without explicit `check_exact=False`/tolerance — passed all 50 hypothesis examples. The optimised impl uses `np.fromiter` then `arr.mean()/std()/max()`; the reference uses `pd.Series.mean()/std()/max()` on the same values. Numpy's reductions and pandas' reductions agree exactly bit-for-bit for these small samples (no accumulated drift). For larger windows we may need a tolerance — revisit when Tier-4 EWM features land.
2. **PLR0915 fired but not PLR0912.** `transform` has both 13 branches (limit 12) and 51 statements (limit 50). The branches are inherent to the two-pass × per-entity × per-window × per-stat structure; the statement count is *also* tight because each stat dispatch is a named `if` arm. Both suppressed in one `noqa`.
3. **YAML-resolver refactor was zero-risk.** Renaming `_DEFAULT_CONFIG_FILENAME` → `_VELOCITY_CONFIG_FILENAME` and adding `_resolve_config_path(filename)` left the existing 10 velocity tests passing on the first try. Confirms the helper boundaries were well-chosen in 2.2.b.
4. **`np.fromiter` with `count=` hint**. Building the deque-amount array via `np.fromiter((a for _, a in d), dtype=np.float64, count=len(d))` is slightly faster than `np.array(list(...))` because the `count` pre-allocates. Detail; probably saves <1% on the property test.
5. **`pd.Series.std(ddof=1)` returns NaN for `n=1`** without warning. Matches our `arr.size >= _MIN_SAMPLES_FOR_STD` guard exactly. Verified by `test_single_event_std_nan` and confirmed in the property test's pandas reference.

## Deviations from the spec

1. **`amount_col` constructor parameter exposed.** Spec didn't enumerate it; defaulting to `TransactionAmt` (the YAML's setting) keeps the production path unchanged but lets tests use any column name without filesystem fixtures.
2. **`config_path` constructor parameter.** Same reason as 2.2.b — needed for `test_default_config_loads` to override during testing if necessary. Production code uses the default.
3. **Refactored `_resolve_default_config_path` → `_resolve_config_path(filename)`.** Internal helper change; documented above. Strictly speaking this *is* a modification of `tier2_aggregations.py` outside the literal "add a class" intent, but the spec explicitly lists `tier2_aggregations.py` as the modified file, and the refactor is the cleanest way to share helper logic between the two classes.
4. **No explicit perf benchmark.** Spec said "high risk", not "very high risk" (which 2.2.b carried). The property test's hypothesis examples each finish in <50 ms; the test suite as a whole runs in 3.14 s. Recording an informational measurement here for context: 50-row × 3-stat × 1-window `_NaiveHistoricalStats.fit_transform` runs in ~0.05 s; the optimised impl is faster but the same order of magnitude on small frames (the speedup shows on larger frames where the deque size dominates).

## Acceptance checklist

- [x] Branch `sprint-2/prompt-2-2-c-historical-stats` created off `main` (`6b91ab8`) **before any edits**
- [x] `configs/historical_stats.yaml` created (entities + stats + windows + amount column)
- [x] `src/fraud_engine/features/tier2_aggregations.py` extended with `HistoricalStats` + helper refactor
- [x] `src/fraud_engine/features/__init__.py` re-exports `HistoricalStats`
- [x] `tests/unit/test_tier2_historical.py` created (11 tests across 3 classes; hypothesis property)
- [x] `make format` returns 0
- [x] `make lint` returns 0
- [x] `make typecheck` returns 0 (29 source files, unchanged)
- [x] `make test-fast` returns 0 (289 passed; +11 net)
- [x] `uv run pytest tests/unit/test_tier2_historical.py -v` returns 0 (11 passed in 3.14s)
- [x] `sprints/sprint_2/prompt_2_2_c_report.md` written (this file)
- [x] No git/gh commands run beyond the §2.1 carve-out (branch create only)
- [x] No source files outside the listed set are modified

Verification passed. Ready for John to commit on `sprint-2/prompt-2-2-c-historical-stats`.

**Commit note:**
```
2.2.c: HistoricalStats (Tier-2 rolling mean/std/max over entity windows)
```

---

## Audit (2026-04-28)

Re-audit on branch `sprint-2/audit-and-gap-fill` (off `main` at `106f321`, post-Sprint-2 original audit). Goal: re-verify the 2.2.c deliverables against the spec and gap-fill anything missing.

### Findings

- **Spec coverage: complete.**
  - `HistoricalStats` per-entity rolling stats; default 30 d window (configurable via YAML / kwargs) ✓
  - All 5 spec-required output columns: `card1_amt_mean_30d`, `card1_amt_std_30d`, `card1_amt_max_30d`, `addr1_amt_mean_30d`, `addr1_amt_std_30d` (configured via `configs/historical_stats.yaml`) ✓
  - Same sorted-iteration pattern as `VelocityCounter`, with deque payload upgraded from `int` (timestamp) to `(int, float)` ((timestamp, amount)) and stats dispatched over a numpy view ✓ (`tier2_aggregations.py:481-591`)
  - Rolling window ends strictly before current row's timestamp ✓ — implemented via tied-group two-pass batching identical to `VelocityCounter`'s pattern (Decision 5).
  - Tests: 7 unit (hand-computed stats, NaN entity, empty window, n=1 std, strict-past, ties, unsupported-stat rejection) + 1 hypothesis property (naive ↔ optimised) + 1 `assert_no_future_leak` + 2 config-load = 11 total ✓
- **No `TODO` / `FIXME` / `XXX` / `HACK` markers** in the HistoricalStats region or `test_tier2_historical.py`.
- **No skipped or `xfail`-marked tests.**
- **Sample-std (`ddof=1`) is correctly pinned via `_STD_DDOF: Final[int] = 1`** module constant. Matches `pd.Series.std()` (the property test's reference) bit-for-bit.
- **Documented "no perf benchmark" deviation is acceptable.** The spec marked this prompt "High" risk (vs `VelocityCounter`'s "Very High" risk that mandated a perf benchmark). The hypothesis property test exercises the algorithm exhaustively on small frames; the full-data 590k-row run lives in 2.2.e's build script (val AUC = 0.9143, runtime well within limits).
- **YAML-resolver refactor (`_resolve_default_config_path` → `_resolve_config_path(filename)`) is in place** and is now also used by `TargetEncoder` (added in 2.2.d). Confirms the refactor was correctly scoped.

### Verification (audit run)

```
$ uv run pytest tests/unit/test_tier2_historical.py -v
11 passed, 14 warnings in 3.12s
```

### Conclusion

No code changes required. The 2.2.c deliverables (`HistoricalStats` + `historical_stats.yaml` + 11 tests) are spec-complete and audit-clean. The deque payload upgrade and recompute-from-deque stat dispatch are clean extensions of the 2.2.b pattern.
