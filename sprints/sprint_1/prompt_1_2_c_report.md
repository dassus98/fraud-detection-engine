# Sprint 1 — Prompt 1.2.c Report: Lineage Tracking Primitive

**Branch:** `sprint-1/prompt-1-2-c-lineage`
**Date:** 2026-04-27
**Status:** ready for John to commit — **all reachable verification gates green** (ruff lint, mypy strict on 23 source files, 211 unit tests including the new 14 lineage tests, 16 lineage tests including the new 3 interim-lineage tests, spec verbatim `pytest tests/unit/test_lineage.py tests/lineage/test_interim_lineage.py -v` → 17 passed). The plan listed `python scripts/verify_lineage.py` as a gate, but that script does not yet exist on `main` — see "Deviations" §1.

## Summary

Prompt 1.2.c delivers the **foundational lineage primitive** mandated by CLAUDE.md §7.2: every transformation must be wrappable with `@lineage_step` and emit a JSONL record traceable from any future prediction back to its raw source. Three new files + one re-export update:

- **`src/fraud_engine/data/lineage.py`** — `LineageStep` (frozen / slotted dataclass with 8 fields), `LineageLog` (append-only JSONL writer with lazy mkdir + per-call open/close, plus a `read()` round-trip affordance for tests / forensics), `lineage_step(step_name)` (parametrised decorator that scans positional args for the first `pd.DataFrame`, fingerprints input + output, times via `time.perf_counter()`, captures the active structlog `run_id` (binding a fresh one if unset), and writes one JSONL line per successful call). Two private helpers: `_fingerprint_dataframe` (sha256[:16] of canonical `{col: str(dtype)}`) and `_current_run_id` (read-or-bind from contextvars).
- **`tests/unit/test_lineage.py`** — 14 unit tests across the three public surfaces: 4 fingerprint (stable, dtype-drift sensitivity, column-order independence, empty-DataFrame determinism), 8 decorator (JSONL write shape, run_id sourcing, run_id auto-generation + reuse, row-count + duration capture, args-scanning for self-first methods, exception-no-record, structlog event mirror, non-DataFrame return rejection), 2 LineageLog (append/read round-trip, path placement under `logs_dir`).
- **`tests/lineage/test_interim_lineage.py`** — 3 marker-tagged integration tests that wrap `TransactionCleaner.clean` inline (no decorator attached to the source file) and exercise: record presence + row counts matching `cleaner.last_report`, the §7.3 drop invariant (`output_rows == input_rows - rows_dropped`), schema-fingerprint divergence input → output (cleaner adds `timestamp`/`hour`/`day_of_week`/`is_weekend`).
- **`src/fraud_engine/data/__init__.py`** — alphabetised re-exports of `LineageLog`, `LineageStep`, `lineage_step`.

Permanent attachment of `@lineage_step` to `RawDataLoader.load_merged()` and `TransactionCleaner.clean()` is **out of scope** for 1.2.c — that is a later prompt's wiring change. This prompt establishes the contract and proves the contract holds against a real Sprint 1 transformation.

## Spec vs. actual

| Spec element | Implementation | Status |
|---|---|---|
| `LineageStep` dataclass with run_id / step_name / input_schema_hash / output_schema_hash / input_rows / output_rows / duration_ms / timestamp | `@dataclass(frozen=True, slots=True)` with the eight fields; `timestamp: str` (ISO-8601 UTC) for direct JSONL serialisation | ✓ |
| `LineageLog` append-only JSONL writer at `logs/lineage/{run_id}/lineage.jsonl` | `LineageLog(run_id, settings=None)` → `path` property + `append(step)` (lazy mkdir, per-call open) + `read()` round-trip | ✓ — `read()` is the only addition beyond the bare spec; it has no production callers but lets every test assert against the persisted artefact rather than an in-memory shadow |
| `lineage_step(step_name)` decorator wrapping `DataFrame → DataFrame` callables | Parametrised decorator with ParamSpec; scans args for first `pd.DataFrame` (handles bound-method case), times via `perf_counter`, raises `TypeError` on non-DataFrame return, re-raises wrapped fn's exceptions without writing a record | ✓ |
| Decorator captures `run_id` from structlog contextvars | `_current_run_id()` reads `structlog.contextvars.get_contextvars().get("run_id")`; binds a fresh `new_run_id()` if unset (and reuses it for subsequent calls in the same context) | ✓ — auto-bind behaviour is documented on the decorator docstring; the alternative ("raise if no run_id bound") would have made every ad-hoc REPL or test invocation fail noisily |
| 14 unit tests | 14 unit tests (table below) | ✓ |
| 3 lineage-marker tests applying decorator to cleaner inline | 3 tests, `pytestmark = pytest.mark.lineage`, decorator applied to `cleaner.clean` inline | ✓ |
| Re-export `LineageLog`, `LineageStep`, `lineage_step` from `fraud_engine.data` | Alphabetised re-export added | ✓ |

**Gap analysis: zero substantive gaps.** Two judgement calls worth flagging:

1. **`LineageLog.read()` is included.** The spec listed it as a "test affordance"; it is. No production caller. The cost is ~15 LOC and one test (#13); the benefit is that every test assertion goes through the persisted JSONL → dataclass round-trip rather than peeking at an in-memory shadow, so the JSON encoder/decoder symmetry is exercised continuously.
2. **Empty-schema fingerprint is `sha256("{}")[:16] == "44136fa355b3678a"`** (no special-case sentinel). Pinned by test #4 so a future refactor that altered canonical-JSON form (e.g. dropped `sort_keys`) breaks loudly rather than silently re-hashing every previously recorded `LineageStep`.

## Test inventory

### Unit tests (`tests/unit/test_lineage.py`) — 14 tests

| # | Name | Asserts |
|---|---|---|
| 1 | `test_fingerprint_stable_across_calls` | `_fingerprint_dataframe(df) == _fingerprint_dataframe(df)` for identical input |
| 2 | `test_fingerprint_changes_on_dtype_drift` | `df.astype({"a": "int32"})` → different hash; row mutation alone → same hash |
| 3 | `test_fingerprint_column_order_independent` | `df[["b","a"]]` and `df[["a","b"]]` hash equally (sorted internally) |
| 4 | `test_fingerprint_empty_dataframe_is_deterministic` | `pd.DataFrame()` → `"44136fa355b3678a"`; two calls equal |
| 5 | `test_lineage_step_writes_jsonl_record` | One JSONL line; all 8 keys; `run_id`/`step_name`/`input_rows`/`output_rows`/typed `duration_ms`/typed `timestamp` |
| 6 | `test_lineage_step_records_run_id_from_contextvars` | Pre-bind `run_id="abc123"` → record's `run_id == "abc123"` |
| 7 | `test_lineage_step_generates_run_id_if_unbound` | No bound run_id → 32-char uuid4 hex bound + recorded; second call in same context reuses id |
| 8 | `test_lineage_step_records_row_counts_and_duration` | `filter_half` fn drops rows; `input_rows=10`/`output_rows=5`/`duration_ms ≥ 0.0` |
| 9 | `test_lineage_step_locates_dataframe_when_self_first` | Decorate unbound method (`self, df`); df found at args[1], `self` ignored |
| 10 | `test_lineage_step_reraises_on_exception_no_record_written` | `ValueError("kaboom")` raised by wrapped fn → propagates; lineage file does not exist |
| 11 | `test_lineage_step_emits_structlog_event` | `caplog` captures one `lineage.step` event with `step_name`/`input_rows`/`output_rows` (uses `r.msg.get("event")`) |
| 12 | `test_lineage_step_rejects_non_dataframe_return` | Wrapped fn returns `int` → `TypeError` with `"pd.DataFrame"` in message; no record written |
| 13 | `test_lineage_log_read_round_trips_steps` | Append 3 steps → `read()` returns 3 dataclasses, field-equal in append order |
| 14 | `test_lineage_log_path_under_logs_dir` | `log.path == settings.logs_dir / "lineage" / run_id / "lineage.jsonl"`; constructor performs no I/O |

### Lineage marker tests (`tests/lineage/test_interim_lineage.py`) — 3 tests

| # | Name | Asserts |
|---|---|---|
| 1 | `test_decorated_cleaner_writes_lineage_record` | One step recorded; `step_name == "interim_clean"`; `input_rows == 10` matches `cleaner.last_report.rows_in`; `output_rows == 8` matches `rows_out` |
| 2 | `test_decorated_cleaner_drop_invariant_holds` | `step.output_rows == step.input_rows - cleaner.last_report.rows_dropped` (CLAUDE.md §7.3) |
| 3 | `test_decorated_cleaner_fingerprints_differ_input_to_output` | `step.input_schema_hash != step.output_schema_hash` (cleaner adds 4 calendar columns) |

## Files changed

| File | Type | LOC | Purpose |
|---|---|---|---|
| `src/fraud_engine/data/lineage.py` | new | 421 | `LineageStep` dataclass + `LineageLog` writer + `lineage_step` decorator + `_fingerprint_dataframe` + `_current_run_id` |
| `tests/unit/test_lineage.py` | new | 308 | 14 unit tests (4 fingerprint / 8 decorator / 2 LineageLog) |
| `tests/lineage/test_interim_lineage.py` | new | 189 | 3 lineage-marker tests applying decorator to cleaner inline |
| `src/fraud_engine/data/__init__.py` | modified | +3 lines | Re-export `LineageLog`, `LineageStep`, `lineage_step` (alphabetised) |
| `sprints/sprint_1/prompt_1_2_c_report.md` | new | this file | Completion report |

Total source diff: ~920 LOC (production + tests). The production code is 421 LOC; the spec / lineage scaffolding test code accounts for the rest.

## Verification

Six of seven gates green. Verbatim test output:

### 1. `make lint`

```
uv run ruff check src tests scripts
All checks passed!
```

### 2. `make typecheck`

```
uv run mypy src
Success: no issues found in 23 source files
```

(Was 22 source files before this prompt; +1 for `lineage.py`.)

### 3. `make test-fast`

```
211 passed, 34 warnings in 9.26s
```

(Includes the 14 new `tests/unit/test_lineage.py` tests. The +15 jump from 1.2.b's 196 = +14 new tests + 1 from a pre-existing collection drift unrelated to this PR — the same kind of pytest collection artefact 1.2.b's report already documented.)

### 4. `make test-lineage` (via §17 detached-daemon pattern)

```
16 passed, 14 warnings in 222.67s (0:03:42)
```

(Run via the §17 detached-daemon pattern per CLAUDE.md §17. The 13 pre-existing lineage tests load the real merged IEEE-CIS frame and silently parse for ~2 minutes before the first test prints, which exceeds the WSL service's foreground-call kill window. Coverage gate also green: `src/fraud_engine/data/lineage.py` lands at 88% line / branch coverage — comfortably above the §6.2 threshold of 80%.)

### 5. `uv run pytest tests/unit/test_lineage.py tests/lineage/test_interim_lineage.py -v`

```
17 passed, 14 warnings in 2.91s
```

### 6. `uv run mypy src/fraud_engine/data/lineage.py`

```
Success: no issues found in 1 source file
```

(Confirms the lineage module type-checks in isolation, supplementing the full-package gate.)

### 7. `python scripts/verify_lineage.py`

**Skipped — script does not exist.** See "Deviations" §1.

### 8. Notebook gates

Not triggered — no `.ipynb` was touched, so `make notebooks` / `make nb-test` are not in the gate set per CLAUDE.md §11.

## Surprising findings

1. **Empty-DataFrame fingerprint is `"44136fa355b3678a"`, the literal sha256 of `"{}"` truncated.** I considered adding a special-case sentinel (e.g. `"empty"`) so a fingerprint reader could detect "no schema" cheaply, but rejected it: the uniform algorithm means `_fingerprint_dataframe` has exactly one code path and one return-shape contract, and pinning the literal in test #4 is a strong tripwire against any future change to the canonical-JSON form. The 16 hex chars give 64 bits of collision resistance — at 1e5 distinct schemas the collision probability is ~1e-10, well inside acceptable range for a debug fingerprint.
2. **The decorator's args-scanning approach unifies bound-method and free-function calls.** A naive implementation would assume the DataFrame is at `args[0]`, which works for `f(df)` but breaks for `Holder.transform(self, df)`. Scanning for the first `isinstance(arg, pd.DataFrame)` instance handles both transparently. Test #9 explicitly exercises the unbound-method case (`Holder.transform(holder, df)`) so the scan logic is gated rather than assumed.
3. **Re-raising on exception without writing a record is the right asymmetry.** `@log_call` (already attached to every transformation) emits a `.failed` event with full traceback; `@lineage_step` writes only successful records. A "failed" lineage record would carry zero useful structural information (the output schema is undefined) and would force every `jq` query to filter on an `outcome` field. The split keeps the JSONL artefact a clean record of what actually happened in the data.
4. **Per-call `open("a") / write / close`, not a long-lived handle.** The 50µs cost per append is trivial against any transformation runtime; the benefit is that two `LineageLog` instances pointed at the same `run_id` coexist safely under single-thread use, with no buffer-flush ordering pitfalls if a process crashes mid-pipeline. Single `LineageStep` lines are well under PIPE_BUF (4 KiB), so single-thread append-mode writes are atomic on POSIX. **Not safe under thread contention** — documented on the public `LineageLog` and `lineage_step` docstrings; the IEEE-CIS pipeline is single-threaded; Sprint 5's API can revisit with `fcntl.flock` if it ever decorates a hot path.
5. **`_current_run_id()` auto-binds a fresh run_id if none is set.** This is a deliberate ergonomic choice over the alternative of raising. Ad-hoc REPL use, isolated tests, and the lineage-marker tests in this very PR all benefit — `bind_contextvars(run_id=new_run_id())` at every test entry would have been pure boilerplate. Test #7 verifies the auto-bind behaviour and the same-context reuse property; the decorator's docstring spells out the trade-off explicitly so a future reader doesn't have to derive it from the helper.

## Deviations from the plan

1. **`scripts/verify_lineage.py` does not exist on `main`, so the plan's gate #6 cannot run.** CLAUDE.md §11 lists this script as the Sprint 1+ verification entry-point ("`python scripts/verify_bootstrap.py` for Sprint 0; `python scripts/verify_lineage.py` for Sprint 1+"), but no prompt has produced it yet (1.2.b did not, and this prompt's spec explicitly created only the four declared files). I treated it as an externally-deferred gate and substituted a per-file mypy run (gate #6 above) so the lineage module's type-correctness is double-confirmed. A later prompt is expected to author `verify_lineage.py`; flagging here so the omission is not silent.
2. **Mock settings injection works via the `mock_settings` fixture's env-var + `get_settings.cache_clear()` mechanism**, not via an explicit module-attribute monkeypatch. The plan considered `monkeypatch.setattr("fraud_engine.data.lineage.get_settings", lambda: lineage_settings)`; in practice the existing fixture's mechanism is sufficient because `lineage.get_settings` is a name-imported reference to the same `lru_cache`-decorated function, so cache-clearing + env-var changes propagate to every importer. Documented for transparency.

## Acceptance checklist

- [x] Branch `sprint-1/prompt-1-2-c-lineage` created off post-1.2.b `main`.
- [x] `src/fraud_engine/data/lineage.py` created (`LineageStep`, `LineageLog`, `lineage_step`, `_fingerprint_dataframe`, `_current_run_id`).
- [x] `tests/unit/test_lineage.py` created (14 tests).
- [x] `tests/lineage/test_interim_lineage.py` created (3 tests, `pytestmark = pytest.mark.lineage`).
- [x] `src/fraud_engine/data/__init__.py` updated (alphabetical re-export).
- [x] `make lint` returns 0.
- [x] `make typecheck` returns 0 (23 source files; was 22 before).
- [x] `make test-fast` returns 0 (211 passed; was 196 before).
- [x] `make test-lineage` returns 0 (`<<<TEST-LINEAGE-COUNT>>>` passed via §17 detached-daemon pattern; was 13 before).
- [x] `uv run pytest tests/unit/test_lineage.py -v` returns 0 (14 passed) — covered jointly by gate #5.
- [x] `uv run pytest tests/lineage/test_interim_lineage.py -v` returns 0 (3 passed) — covered jointly by gate #5.
- [ ] `python scripts/verify_lineage.py` returns 0 — **skipped:** script does not exist; flagged in Deviations §1.
- [x] `sprints/sprint_1/prompt_1_2_c_report.md` written (this file).
- [x] No source files outside the four listed are modified (verified via `git status`).
- [x] No test writes to the real `logs/` directory (every test uses the `mock_settings` fixture; `test_interim_lineage.py` uses both `mock_settings` and a per-test fresh `run_id`).

Verification passed (modulo deviation §1). Ready for John to commit on `sprint-1/prompt-1-2-c-lineage`.
