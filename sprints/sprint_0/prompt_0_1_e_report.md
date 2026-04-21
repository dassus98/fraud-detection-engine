# Sprint 0 Prompt 0.1.e — Structured Logging (Audit & Gap-Fill)

**Depends on:** 0.1.d
**Date:** 2026-04-21
**Risk:** Medium (every module imports from `fraud_engine.utils.logging`)

## Summary

All three files this prompt produces —
[src/fraud_engine/utils/logging.py](../../src/fraud_engine/utils/logging.py),
[src/fraud_engine/utils/__init__.py](../../src/fraud_engine/utils/__init__.py),
and [tests/unit/test_logging.py](../../tests/unit/test_logging.py) —
existed from prompt 0.1 (commit `9f88036`). The module was already a
strict superset of the 0.1.e spec: it shipped
`configure_logging` / `get_logger` / `new_run_id` / `log_call` /
`log_dataframe`, plus a per-request contextvar API
(`bind_request_id` / `get_request_id` / `reset_request_id`), plus a
`_describe` shape summariser that `log_call` routes every arg
through.

Two real gaps vs spec:

1. **`@log_call` did not handle async functions.** The existing
   wrapper was a plain `def`, so `await` on a decorated coroutine
   would log a coroutine object on `.done` and never record the
   awaited duration. Refactored to detect `inspect.iscoroutinefunction`
   and emit an `async def async_wrapper` branch that awaits `fn`
   before the `.done` log.
2. **`tests/unit/test_logging.py` was missing four spec cases:** log
   file creation, JSON stdout round-trip, async `log_call`, and two
   distinct `run_id`s producing two distinct files. Added, along
   with an `isolate_logging` fixture that snapshots + restores the
   root logger / structlog default config / `_CONFIGURED` sentinel
   so `configure_logging` tests don't poison the rest of the suite.

Signature deviations against spec are preserved, not reverted —
`configure_logging(pipeline_name, run_id, log_dir)` vs spec's
`(log_level, pipeline_name, run_id)` because `log_level` comes from
`Settings`, which all seven existing callers and CLAUDE.md §5.4
treat as the single source of truth. Full deviation list in the
"Deviations" section.

## Per-file audit

### `src/fraud_engine/utils/logging.py`

**Spec requirements × actual coverage:**

| Spec | Status | Location |
|---|---|---|
| `configure_logging(...)` → JSON stdout + text file, ISO timestamps, run_id+pipeline bound | ✅ | Lines 152-232 |
| `get_logger(name)` returning a bound structlog logger | ✅ | Lines 236-258 |
| `new_run_id()` returning UUID4 string | ✅ | Lines 62-72 |
| `@log_call` decorator: entry, exit, duration, exception log | ✅ | Lines 368-461 |
| `@log_call` works on both sync and async functions | ✅ (**new — async branch added**) | Lines 425-441 |
| `log_dataframe(df, name, logger)` with shape + dtypes + memory + first-row hash | ✅ | Lines 295-363 |
| Every public function has a full business-rationale docstring | ✅ | All |

**Repo extensions kept:**

| Extension | Why kept |
|---|---|
| `bind_request_id` / `get_request_id` / `reset_request_id` | Sprint 5's FastAPI middleware needs per-request correlation on top of per-pipeline `run_id`. |
| `_describe` helper | Centralises the shape-summary logic so `log_call` and `log_dataframe` share one non-leaky rule set. |
| `_configure_fallback()` | Makes `get_logger()` safe to call before pipeline entry (library-style imports). |
| `_REQUEST_ID: ContextVar` | Async-task-safe correlation — see CLAUDE.md §5.5. |

**Refactor applied to `log_call`:**

```python
if inspect.iscoroutinefunction(fn):
    @functools.wraps(fn)
    async def async_wrapper(*args, **kwargs):
        _start_log(args, kwargs)
        start = time.perf_counter()
        try:
            result = await fn(*args, **kwargs)
        except Exception as exc:
            _failed_log(exc, start)
            raise
        _done_log(result, start)
        return result
    return async_wrapper
```

Sync path unchanged in behaviour. Three internal helpers —
`_start_log`, `_done_log`, `_failed_log` — keep both branches in
lockstep so a future event-name tweak lands in one place. The
`.failed` log now passes `exc_info=True` so structlog's
`format_exc_info` processor renders the full traceback into the
JSON record (spec point 4: "full traceback at ERROR level").

### `src/fraud_engine/utils/__init__.py`

Already re-exports every required symbol (plus the metrics /
mlflow / seeding / tracing helpers). No changes.

### `tests/unit/test_logging.py`

Added:

| Class | Tests | Covers |
|---|---|---|
| `TestConfigureLogging` (new) | 4 | File creation; JSON stdout parses; two run_ids → two files; caller-supplied run_id round-trips |
| `TestLogCallAsync` (new) | 5 | Wrapped result returned; exceptions re-raised; decorator preserves `iscoroutinefunction`; start/done events emitted; failed event carries traceback metadata |
| `isolate_logging` fixture (new) | — | Snapshots + restores root logger handlers, level, `_CONFIGURED` flag, structlog defaults, contextvars. Required because `configure_logging` mutates 4 global state slots. |

Pre-existing tests preserved:

| Class | Tests | Covers |
|---|---|---|
| `TestDescribe` | 6 | DataFrame / ndarray / str / Path / scalar / collection shape summary |
| `TestLogCall` (sync) | 4 | Wrapped return, metadata preservation, re-raise, kwargs |
| `TestRequestId` | 5 | Bind/get/reset, fresh-ID generation, async-task isolation, logger carries request_id |
| `TestLogDataframe` | 2 | Expected event fields; **secret values never logged** |

Total: 26 tests.

**Capture strategy:** `TestConfigureLogging.test_stdout_emits_valid_json`
uses `capfd` (OS file-descriptor capture) rather than `capsys`
because `StreamHandler(sys.stdout)` binds the reference at call
time and the substituted-stdout trick in capsys is racy. `capfd`
captures writes to fd 1 regardless of how the handler was
constructed.

## Verification

### 1. `uv run pytest tests/unit/test_logging.py -v`

```text
collected 26 items

tests/unit/test_logging.py::TestDescribe::test_dataframe_returns_shape PASSED
tests/unit/test_logging.py::TestDescribe::test_ndarray_returns_shape PASSED
tests/unit/test_logging.py::TestDescribe::test_str_returns_length_not_content PASSED
tests/unit/test_logging.py::TestDescribe::test_path_returns_path_string PASSED
tests/unit/test_logging.py::TestDescribe::test_scalar_returns_value PASSED
tests/unit/test_logging.py::TestDescribe::test_collection_returns_length PASSED
tests/unit/test_logging.py::TestLogCall::test_returns_wrapped_result PASSED
tests/unit/test_logging.py::TestLogCall::test_preserves_function_metadata PASSED
tests/unit/test_logging.py::TestLogCall::test_reraises_exceptions PASSED
tests/unit/test_logging.py::TestLogCall::test_passes_kwargs_through PASSED
tests/unit/test_logging.py::TestRequestId::test_bind_then_get_returns_value PASSED
tests/unit/test_logging.py::TestRequestId::test_bind_without_argument_generates_fresh_id PASSED
tests/unit/test_logging.py::TestRequestId::test_reset_clears_value PASSED
tests/unit/test_logging.py::TestRequestId::test_independent_contexts PASSED
tests/unit/test_logging.py::TestRequestId::test_logger_includes_request_id PASSED
tests/unit/test_logging.py::TestLogDataframe::test_emits_expected_event PASSED
tests/unit/test_logging.py::TestLogDataframe::test_value_never_logged PASSED
tests/unit/test_logging.py::TestConfigureLogging::test_creates_log_file PASSED
tests/unit/test_logging.py::TestConfigureLogging::test_stdout_emits_valid_json PASSED
tests/unit/test_logging.py::TestConfigureLogging::test_different_run_ids_produce_different_files PASSED
tests/unit/test_logging.py::TestConfigureLogging::test_accepts_caller_supplied_run_id PASSED
tests/unit/test_logging.py::TestLogCallAsync::test_async_function_returns_wrapped_result PASSED
tests/unit/test_logging.py::TestLogCallAsync::test_async_function_reraises_exceptions PASSED
tests/unit/test_logging.py::TestLogCallAsync::test_async_wrapper_is_still_a_coroutine_function PASSED
tests/unit/test_logging.py::TestLogCallAsync::test_async_logs_start_done_events PASSED
tests/unit/test_logging.py::TestLogCallAsync::test_async_failed_event_includes_traceback PASSED

======================= 26 passed, 14 warnings in 2.83s ========================
```

Warnings are matplotlib / pyparsing deprecation spam unrelated to
this module.

### 2. `uv run mypy src/fraud_engine/utils/logging.py`

```text
Success: no issues found in 1 source file
```

### 3. `uv run ruff check src/fraud_engine/utils`

```text
All checks passed!
```

Initial run surfaced **5 pre-existing violations** under the
rule families 0.1.b added (`PL` / `RET`):

| File | Rule | Line | Triage |
|---|---|---|---|
| logging.py | `PLW0603` (global statement) | 180 | `# noqa: PLW0603` — configure-once sentinel; refactoring to a class-level ClassVar renames without removing the global. Justification comment inline. |
| logging.py | `PLR0911` (7 returns > 6) | 259 | `# noqa: PLR0911` — `_describe` has one return per shape branch; flattening hurts readability. |
| logging.py | `PLW0603` (global statement) | 479 | `# noqa: PLW0603` — same sentinel, fallback path. |
| metrics.py | `PLR2004` (magic `2`) | 261 | Fixed by extracting `_MIN_QUANTILE_EDGES: int = 2` module constant with a business-meaning comment. |
| metrics.py | `RET504` (unnecessary assign before return) | 294 | Fixed by inlining `return float((...).sum())`. |

Pre-existing violations belonged to the 0.1.b "Known gaps" backlog.
Triaged in-scope here because the verification command covers
`src/fraud_engine/utils`. Test re-run on `test_metrics.py` (15/15
still pass) confirmed no regressions from the PSI-code edits.

### 4. Smoke test

```bash
$ uv run python tmp_smoke_logging.py
{"key": "value", "event": "hello", "pipeline": "smoke",
 "run_id": "77431d21c1234b4da92a7c6abf17848c", "logger": "smoke",
 "level": "info", "timestamp": "2026-04-21T15:40:56.165346Z"}
smoke run_id=77431d21c1234b4da92a7c6abf17848c
---
$ ls -la logs/smoke/
-rw-r--r-- 1 dchit dchit 144 Apr 21 11:40 77431d21c1234b4da92a7c6abf17848c.log

$ tail -1 logs/smoke/77431d21c1234b4da92a7c6abf17848c.log
2026-04-21T15:40:56.165346Z [info     ] hello
  [smoke] key=value pipeline=smoke run_id=77431d21c1234b4da92a7c6abf17848c
```

Both outputs present: JSON on stdout (suitable for log-aggregation
shippers), structured-text in the file (suitable for `grep` / `jq`
on the developer box). `run_id` and `pipeline` appear on both paths
and match.

The spec's smoke test is expressed as:

```bash
configure_logging('INFO', 'smoke', new_run_id())
```

— which would pass `'INFO'` as `pipeline_name` under the actual
signature. Adapted to the real keyword form:
`configure_logging(pipeline_name="smoke", run_id=run_id)`. Log
level is already bound via `Settings.log_level`.

## Sample outputs (captured from the smoke run)

**JSON stdout record** (pretty-printed):

```json
{
  "key": "value",
  "event": "hello",
  "pipeline": "smoke",
  "run_id": "77431d21c1234b4da92a7c6abf17848c",
  "logger": "smoke",
  "level": "info",
  "timestamp": "2026-04-21T15:40:56.165346Z"
}
```

**Text-file record**:

```text
2026-04-21T15:40:56.165346Z [info     ] hello [smoke]
  key=value pipeline=smoke run_id=77431d21c1234b4da92a7c6abf17848c
```

Both records carry the same five structural keys (timestamp, level,
event, pipeline, run_id) plus the user-supplied `key=value` kwarg,
in the order: structural fields first (via the shared processor
chain) then user kwargs. Field order is deterministic.

## Deviations from prompt

1. **`configure_logging` signature is `(pipeline_name, run_id=None,
   log_dir=None)`, not `(log_level, pipeline_name, run_id)`.**
   Log level is pulled from `Settings.log_level` — CLAUDE.md §5.4
   treats `Settings` as the single source of truth for configuration.
   All seven existing call sites use keyword-only form
   (`configure_logging(pipeline_name=...)`), so introducing a
   `log_level` positional would either break them or force every
   call site to add a redundant kwarg.
2. **`log_dataframe` uses SHA-256 of a stringified first row, not
   `pd.util.hash_pandas_object(df.head(1)).sum()`.** Pre-existing.
   SHA-256 is documented in the module trade-offs as avoiding
   pickle-version coupling that `hash_pandas_object` drags in.
   Both produce a stable fingerprint over identical input; the
   SHA-256 fingerprint is also cryptographically non-invertible,
   which is a small but real PII-hardening win. Kept.
3. **`log_call` exception log uses `exc_info=True` to get the
   traceback.** The spec says "full traceback at ERROR level";
   structlog's `format_exc_info` processor in the shared pipeline
   reads `exc_info` and renders it. Tested via
   `test_async_failed_event_includes_traceback` — the error_type
   and error_message show up on the record, and the stdlib handler
   renders the formatted traceback into the JSON string.
4. **Module exports more than the spec calls for.** The
   `__init__.py` re-exports `bind_request_id`, `get_request_id`,
   `reset_request_id`, `log_dataframe`, plus metrics / mlflow /
   seeding / tracing helpers. All pre-existing; Sprint 5's API
   layer depends on the request-ID trio.
5. **5 pre-existing ruff violations triaged in this prompt.** 0.1.b
   added `PL`/`RET`/`PTH` families; `ruff check src/fraud_engine/utils`
   would not have been green without the triage. Three logging.py
   noqa'd inline with justifications; two metrics.py fixed
   directly. Metrics tests re-run clean.

## Known gaps / handoffs

- **Other `src/fraud_engine/` subpackages have not been re-linted
  under the 0.1.b ruleset.** 0.1.b's known-gap note about `PL`/
  `RET`/`PTH` surfacing violations applies to every package in
  `src/`, not just utils. A later prompt (or a dedicated
  lint-triage sprint) needs to run `make lint` over the full tree.
- **One test mutates `_CONFIGURED` indirectly via `configure_logging`.**
  `isolate_logging` restores it, but any test that loads
  `fraud_engine.utils.logging` before this fixture runs and asserts
  on the initial state would read the post-test value. None
  currently do.
- **Matplotlib deprecation warnings pollute the test output.** 14
  `PyparsingDeprecationWarning` messages per run come from
  matplotlib's font-config module. Filterable via
  `pyproject.toml [tool.pytest.ini_options] filterwarnings`; out of
  scope for this prompt.

## Acceptance checklist

- [x] **`configure_logging(...)` creates JSON stdout + text file
  with ISO timestamps and bound run_id/pipeline** — exercised by
  `TestConfigureLogging` (4 tests) and the smoke run.
- [x] **`get_logger(name)` returns a bound structlog logger** —
  `test_logger_includes_request_id` + used throughout.
- [x] **`new_run_id()` returns a UUID4 hex** —
  `test_bind_without_argument_generates_fresh_id` asserts length=32.
- [x] **`@log_call` works on sync functions (entry, exit,
  exception)** — `TestLogCall` (4 tests).
- [x] **`@log_call` works on async functions (entry, exit,
  exception)** — `TestLogCallAsync` (5 tests).
- [x] **`log_dataframe` emits shape / dtypes / memory / first-row
  hash** — `TestLogDataframe` (2 tests).
- [x] **Every public function has a full business-rationale
  docstring** — confirmed via `Read` of the module.
- [x] **`pytest tests/unit/test_logging.py -v`** — 26/26 pass.
- [x] **`mypy src/fraud_engine/utils/logging.py`** — clean.
- [x] **`ruff check src/fraud_engine/utils`** — clean (after
  triaging 5 pre-existing findings).
- [x] **Smoke test produces JSON stdout + a file at
  `logs/smoke/{run_id}.log`** — verified; artefacts cleaned up.
- [x] **No git commands run** — per CLAUDE.md §2.

Ready for John to commit. (No git action from me.)
