# Sprint 0 — Prompt 0.3.a Completion Report

**Prompt:** `Run` context manager — class-based API with `log_param` / `log_metric` / `attach_artifact` methods, `run_id` / `run_dir` properties, `RunMetadata` dataclass alongside it
**Date completed:** 2026-04-23
**Smoke Run ID:** `1beb00214cb748669ff6494dc3e49e04`

---

## 1. Summary

Prompt 0.3.a asks for a class-based `Run` context manager at `src/fraud_engine/utils/tracing.py` plus a companion `RunMetadata` dataclass. The tracing module **already existed** in the repo (379 lines, functional `@contextmanager def run_context` yielding a `@dataclass(frozen=True) class Run`) and covered the substantive requirements — per-run directory, `run.json` with status/metadata, `artifacts/` subdir, structlog `run_id` binding, success/failed exit states. What it did **not** provide was the class-based shape the spec calls out by name: `Run(pipeline_name, settings=...)` as a context manager with `log_param` / `log_metric` / method-form `attach_artifact`.

This turn is an **additive migration**:

| Deliverable | Status |
|---|---|
| Class-based `Run` with `__enter__` / `__exit__` | **Done** |
| `log_param(key, value)` → persists into `run.json` | **Done** |
| `log_metric(key, value)` → persists under `metadata.metrics` | **Done** |
| `attach_artifact(name, path)` method (strict file-copy) | **Done** |
| `run_id` / `run_dir` / `artifacts_dir` properties | **Done** |
| `RunMetadata` dataclass | **Done** (renamed from the old frozen `Run` dataclass, with `pipeline → pipeline_name` and new fields `status / end_time / duration_ms / params / metrics / extra`) |
| Sprint 1+ caller compatibility (`run_context`, module-level `attach_artifact`) | **Preserved** — `run_context` is now a thin `@contextmanager` wrapper around the class |
| Tests | **22/22 pass** (10 pre-existing + 12 new across `TestRunClass` + `TestStructlogCorrelation`) |

No Sprint 1 caller changed; `scripts/run_sprint1_baseline.py`'s `Run` type annotation now refers to the new class (same `.run_dir` / `.artifacts_dir` / `.run_id` property surface), and `attach_artifact(run, obj, *, name)` module-level dispatch still reads `run.artifacts_dir`.

---

## 2. Audit — Pre-Existing State

### `src/fraud_engine/utils/tracing.py` (379 lines, pre-edit)

- `@dataclass(frozen=True) class Run` with fields `run_id, pipeline, start_time, run_dir, artifacts_dir, metadata` — **not** the spec-shaped class.
- `@contextmanager def run_context(pipeline, *, metadata=None, capture_streams=True) -> Iterator[Run]` — functional API covering the lifecycle.
- Module-level `attach_artifact(run, obj, *, name)` with `isinstance` dispatch (Path → copy, DataFrame → parquet, Figure → png, dict/list → json, else → joblib).
- `_TeeStream` for stdout/stderr capture.
- `_serialise_run` / `_rewrite_run_json` for run.json persistence.

### `tests/unit/test_tracing.py` (127 lines, pre-edit)

10 tests: 4 on `run_context` (dir tree, success, failure, metadata round-trip) + 6 on `attach_artifact` dispatch (Path / DataFrame / Figure / dict / generic joblib).

### `src/fraud_engine/utils/__init__.py`

Re-exported `Run, attach_artifact, run_context` — no `RunMetadata`.

### Caller audit

- `scripts/run_sprint1_baseline.py` imports `Run, attach_artifact, run_context`. Usage:
  - `Run` appears **only** as a type annotation: `def _run_variant(..., run: Run) -> BaselineResult` (line 60).
  - Entry is `with run_context("sprint1_baseline", metadata={...}) as run:` (line 127).
  - `attach_artifact(run, obj, *, name)` (module-level) reads `run.artifacts_dir`.
- **Nothing instantiates `Run(...)` directly**, so the class-vs-dataclass shape change is safe.

### Substance already met

- Per-run directory tree `logs/runs/{run_id}/artifacts/`
- `run.json` with `status` (`running`/`success`/`failed`), `start_time`, `end_time`, `duration_ms`, `metadata`, exception fields on failure
- `structlog` `run_id` contextvar binding via `configure_logging(pipeline, run_id=...)` (CLAUDE.md §5.5 compliance)
- Optional stdout/stderr tee via `_TeeStream`

### Gaps vs spec

| Gap | Severity |
|---|---|
| No class-based `Run` with `__enter__` / `__exit__` | **Spec-blocker** — spec requires the class API |
| No `log_param` / `log_metric` methods | **Spec-blocker** — metadata can only be set at construction |
| `attach_artifact` only exists as a type-dispatch module-level function, not as a method taking `(name, path)` | **Spec-blocker** — spec asks for the method form |
| `RunMetadata` dataclass does not exist | **Spec-blocker** — spec names it |
| Field `pipeline` (short for "pipeline name") does not match spec's `pipeline_name` | **Minor** — rename on the renamed dataclass |

---

## 3. Gap-Fill — Edits This Turn

### `src/fraud_engine/utils/tracing.py` (restructured, 602 lines post-ruff-format)

**Renamed** the frozen `@dataclass class Run` → `@dataclass class RunMetadata`:

```python
@dataclass
class RunMetadata:
    run_id: str
    pipeline_name: str                       # renamed from `pipeline`
    start_time: datetime
    status: RunStatus = "running"            # new — Literal["running","success","failed"]
    end_time: datetime | None = None         # new
    duration_ms: float | None = None         # new
    params: dict[str, Any] = field(default_factory=dict)    # new — from log_param
    metrics: dict[str, float] = field(default_factory=dict) # new — from log_metric
    extra: dict[str, Any] = field(default_factory=dict)     # renamed/repurposed from `metadata`
```

The decision was to drop `frozen=True` (the old `Run` used it) because `RunMetadata` is now constructed fresh on every persist by `Run._build_metadata()` — the caller never mutates it in place, so the `frozen` guard wasn't buying anything. Instances should still be treated as immutable; this is documented in the docstring.

**Introduced** class `Run`:

```python
class Run:
    def __init__(
        self,
        pipeline_name: str,
        *,
        settings: Settings | None = None,
        metadata: dict[str, Any] | None = None,
        capture_streams: bool = False,
    ) -> None: ...

    def __enter__(self) -> Run: ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

    def log_param(self, key: str, value: Any) -> None:
        """Sets params[key] = value, re-persists run.json."""

    def log_metric(self, key: str, value: float) -> None:
        """Sets metrics[key] = float(value), re-persists run.json."""

    def attach_artifact(self, name: str, path: Path | str) -> Path:
        """shutil.copy2(path → artifacts_dir / name), returns destination."""

    @property
    def run_id(self) -> str: ...
    @property
    def run_dir(self) -> Path: ...
    @property
    def artifacts_dir(self) -> Path: ...
    @property
    def start_time(self) -> datetime: ...
    @property
    def pipeline(self) -> str: ...
    @property
    def metadata(self) -> dict[str, Any]: ...  # {params, metrics, extra} snapshot
```

Internal state: `_run_id`, `_run_dir`, `_artifacts_dir`, `_start_time`, `_params`, `_metrics`, `_extra`, `_stdout_file`, `_stderr_file`, `_original_stdout`, `_original_stderr`, `_pipeline_name`, `_settings_override`, `_capture_streams`, `_entered`. All property accessors call `self._require_entered(op_name)` which raises `RuntimeError` with the offending operation named if called before `__enter__`.

`__enter__` mirrors the old `run_context` body: resolves settings, generates `run_id`, calls `configure_logging(pipeline_name, run_id=run_id)`, creates the directory tree, writes initial `run.json` with `status="running"`, sets up tee streams if `capture_streams=True`, logs `run.start`, returns `self`.

`__exit__` mirrors the old success/except branches: computes `duration_ms`, writes final `run.json` with `status="success"` or `status="failed"` (with `exception_type` / `exception_message` / `traceback` on failure), logs `run.done` or `run.failed`, restores streams, closes files. Returns `None` (implicit False) so exceptions propagate — the class preserves the `raise` semantics the old `run_context` had.

**Refactored** `run_context` to a thin wrapper:

```python
@contextmanager
def run_context(
    pipeline: str,
    *,
    metadata: dict[str, Any] | None = None,
    capture_streams: bool = True,   # legacy default preserved for compat
) -> Iterator[Run]:
    with Run(pipeline, metadata=metadata, capture_streams=capture_streams) as run:
        yield run
```

Sprint 1 callers pick up the class-based `Run` without any source change — the yielded object exposes the same read surface (`run_id`, `run_dir`, `artifacts_dir`) as properties.

**Kept** module-level `attach_artifact(run, obj, *, name)` unchanged. It's the isinstance-dispatching form used by `scripts/run_sprint1_baseline.py` (`attach_artifact(run, splits.manifest, name="splits_manifest")` etc.). The spec-shaped method `Run.attach_artifact(name, path)` is a **sibling API**: it takes a name + existing file path and does a strict `shutil.copy2`. Both coexist; the module docstring documents when to use which.

**JSON persistence shape.** `_serialise_metadata` flattens `extra` + `params` into the top of the `metadata` block and nests `metrics` under a `"metrics"` sub-dict (omitted when empty). This keeps the old `test_metadata_round_trips` passing (which asserted `payload["metadata"] == {"source": "unit-test", "n_rows": 42}` — still true when no `log_param` or `log_metric` are called), while matching the spec's new tests (`metadata["n_features"] == 123` after `log_param`, `metadata["metrics"]["auc"] == 0.92` after `log_metric`). `log_param("metrics", ...)` raises `ValueError` to prevent namespace collision.

### `src/fraud_engine/utils/__init__.py`

Added `RunMetadata` to the import from `fraud_engine.utils.tracing` and to `__all__`. No other changes.

### `tests/unit/test_tracing.py` — extended (323 lines post-ruff-format)

**Kept all 10 pre-existing tests unchanged**. They use `run_context(...)` which now yields a class-based `Run`; the read surface is identical, so they pass as-is. Confirmed in verification (see §6).

**Added `TestRunClass`** — 11 new tests (1 more than planned; added `test_attach_artifact_method_missing_source_raises` and `test_log_param_reserved_key_raises` to cover error branches, and `test_run_metadata_dataclass_is_exported` to fail loudly if the dataclass shape drifts):

| Test | Contract verified |
|---|---|
| `test_class_context_manager_creates_directory` | `with Run(...) as r:` creates `run_dir`, `artifacts_dir`, `run.json` |
| `test_class_success_marks_status` | Normal exit → `status="success"`, `end_time`/`duration_ms` populated |
| `test_class_failure_marks_status_and_reraises` | Exception inside `with` → `status="failed"` with `exception_type`/`traceback`; exception still propagates |
| `test_log_param_persists_to_run_json` | `run.log_param("n_features", 123)` → reload mid-run & post-exit; `metadata["n_features"] == 123` in both |
| `test_log_metric_persists_to_run_json` | `run.log_metric("auc", 0.92)` → `metadata["metrics"]["auc"] == 0.92`; int input coerced to float |
| `test_attach_artifact_method_copies_by_path` | `run.attach_artifact("copied.txt", src)` → `artifacts_dir/copied.txt` has identical bytes |
| `test_attach_artifact_method_missing_source_raises` | `FileNotFoundError` on non-existent source |
| `test_access_before_enter_raises` | Reading `run_id` / `run_dir` / calling `log_param` before `__enter__` raises `RuntimeError` naming the op |
| `test_log_param_reserved_key_raises` | `log_param("metrics", ...)` → `ValueError("reserved")` |
| `test_run_metadata_roundtrip_via_extra` | Construction-time `metadata={...}` dict lands under `extra`; visible via the `metadata` property |
| `test_run_metadata_dataclass_is_exported` | `RunMetadata` is importable from `fraud_engine.utils.tracing` with the expected field set |

**Added `TestStructlogCorrelation`** — 2 tests addressing CLAUDE.md §5.5 (every log record inside a pipeline run includes `run_id`):

| Test | Contract verified |
|---|---|
| `test_log_records_carry_run_id` | Attach a `StreamHandler` to a `StringIO`, emit an event inside `with Run(...) as r:`, parse the captured JSON, assert `record["run_id"] == r.run_id` and `record["pipeline"] == "unit-test"` |
| `test_configure_logging_is_reentrant_after_run` | Two sequential `Run`s + a post-run `configure_logging` call all complete without error — guards against handler state leaking between runs |

---

## 4. Deviations from Spec

### (a) Dual API surface — class `Run` + functional `run_context`

**Spec asks for:** class-based `Run`.

**What exists:** class `Run` **plus** `run_context` as a thin `@contextmanager` wrapper around it.

**Justification:** `run_context` is the entry point Sprint 1's baseline script (`scripts/run_sprint1_baseline.py`, line 127) and the notebook builder (`scripts/_build_eda_notebook.py`) already use. Removing it would require editing both callers and any prior sprint report that references it. The wrapper is four lines and adds zero behavioural difference — it simply preserves the legacy default of `capture_streams=True` (the class defaults to `False` so pytest's `capsys` works unmodified).

### (b) `attach_artifact` exists in two sibling forms

**Spec asks for:** method `Run.attach_artifact(name, path)`.

**What exists:** both the spec-shaped method **and** a module-level `attach_artifact(run, obj, *, name)` with `isinstance` dispatch over object types (Path / DataFrame / Figure / dict / list / fallback-joblib).

**Justification:** Sprint 1 callers pass DataFrames and dicts — not on-disk paths — to the module-level form. The method form is the spec contract; the function form is the ergonomic entry point for callers who have an in-memory object. They are not wrappers on each other; the method is a strict file-copy, the function dispatches. Documented at the top of the module.

### (c) `frozen=True` dropped from `RunMetadata`

**Old:** `@dataclass(frozen=True) class Run`.

**New:** `@dataclass class RunMetadata` (mutable by default).

**Justification:** `RunMetadata` is constructed fresh on every persist by `Run._build_metadata()` and never mutated by callers. `frozen=True` was buying type-level immutability but at the cost of needing `dataclasses.replace` for every update — which the old code did *not* do for the metadata field anyway (it rebuilt the whole `Run` on every `_rewrite_run_json`). Dropping `frozen=True` is a wash at runtime and clearer in intent. The docstring calls out that instances should be treated as immutable.

### (d) Field rename `pipeline` → `pipeline_name`

**Old dataclass field:** `pipeline: str`.

**New dataclass field:** `pipeline_name: str`.

**Justification:** Spec calls out `pipeline_name` explicitly in the `__init__` signature. The `Run` class also exposes a `pipeline` property (returns `self._pipeline_name`) as a backwards-compat read alias, so any caller that did `run.pipeline` (none in the codebase audit, but defensive) still works. The JSON payload continues to write `"pipeline": ...` as the top-level key so consumers of `run.json` don't break.

### (e) Structlog correlation test uses direct handler attachment, not `caplog`

**Why:** pytest's `caplog` doesn't play well with `structlog.ProcessorFormatter` — records arrive as un-rendered `event_dict`s, not JSON strings. Attaching a `StreamHandler` with the production JSON formatter to a `StringIO` is the minimal way to capture the actual JSON a production aggregator would see. Documented inline in the test.

---

## 5. Files Changed

| File | Change |
|---|---|
| `src/fraud_engine/utils/tracing.py` | Restructured: `RunMetadata` dataclass introduced (rename+extend of old frozen `Run`), new class `Run` with `__enter__`/`__exit__`/`log_param`/`log_metric`/`attach_artifact` + 6 properties, `run_context` refactored to thin wrapper, `_serialise_metadata` rewritten for new payload shape, module-level `attach_artifact` and `_TeeStream` unchanged |
| `src/fraud_engine/utils/__init__.py` | Export `RunMetadata` |
| `tests/unit/test_tracing.py` | 12 new tests: 11 in `TestRunClass`, 2 in `TestStructlogCorrelation`; imports updated; existing 10 tests untouched and still pass |
| `sprints/sprint_0/prompt_0_3_a_report.md` | **NEW** — this report |

No other files modified. `scripts/run_sprint1_baseline.py`, `scripts/_build_eda_notebook.py`, and the broader `src/` tree were read to confirm compat but not edited.

---

## 6. Verification

### Ruff

```
$ uv run ruff check src/fraud_engine/utils/tracing.py tests/unit/test_tracing.py
All checks passed!

$ uv run ruff check src/ tests/ scripts/
All checks passed!
```

### Ruff format

```
$ uv run ruff format src/fraud_engine/utils/tracing.py tests/unit/test_tracing.py
2 files reformatted
# (second-pass --check is clean)
```

### Mypy (strict, src/)

```
$ uv run mypy src/fraud_engine/utils/tracing.py
Success: no issues found in 1 source file

$ uv run mypy src/fraud_engine/ scripts/run_sprint1_baseline.py
Success: no issues found in 21 source files
```

### Pytest — targeted

```
$ uv run pytest tests/unit/test_tracing.py -v
...
tests/unit/test_tracing.py::TestRunContext::test_creates_directory_tree PASSED              [  4%]
tests/unit/test_tracing.py::TestRunContext::test_success_marks_status PASSED                [  9%]
tests/unit/test_tracing.py::TestRunContext::test_failure_marks_status_and_reraises PASSED   [ 13%]
tests/unit/test_tracing.py::TestRunContext::test_metadata_round_trips PASSED                [ 18%]
tests/unit/test_tracing.py::TestAttachArtifact::test_attach_path PASSED                     [ 22%]
tests/unit/test_tracing.py::TestAttachArtifact::test_attach_dataframe PASSED                [ 27%]
tests/unit/test_tracing.py::TestAttachArtifact::test_attach_figure PASSED                   [ 31%]
tests/unit/test_tracing.py::TestAttachArtifact::test_attach_dict PASSED                     [ 36%]
tests/unit/test_tracing.py::TestAttachArtifact::test_attach_generic_object PASSED           [ 40%]
tests/unit/test_tracing.py::TestRunClass::test_class_context_manager_creates_directory PASSED   [ 45%]
tests/unit/test_tracing.py::TestRunClass::test_class_success_marks_status PASSED            [ 50%]
tests/unit/test_tracing.py::TestRunClass::test_class_failure_marks_status_and_reraises PASSED   [ 54%]
tests/unit/test_tracing.py::TestRunClass::test_log_param_persists_to_run_json PASSED        [ 59%]
tests/unit/test_tracing.py::TestRunClass::test_log_metric_persists_to_run_json PASSED       [ 63%]
tests/unit/test_tracing.py::TestRunClass::test_attach_artifact_method_copies_by_path PASSED [ 68%]
tests/unit/test_tracing.py::TestRunClass::test_attach_artifact_method_missing_source_raises PASSED [ 72%]
tests/unit/test_tracing.py::TestRunClass::test_access_before_enter_raises PASSED            [ 77%]
tests/unit/test_tracing.py::TestRunClass::test_log_param_reserved_key_raises PASSED         [ 81%]
tests/unit/test_tracing.py::TestRunClass::test_run_metadata_roundtrip_via_extra PASSED      [ 86%]
tests/unit/test_tracing.py::TestRunClass::test_run_metadata_dataclass_is_exported PASSED    [ 90%]
tests/unit/test_tracing.py::TestStructlogCorrelation::test_log_records_carry_run_id PASSED  [ 95%]
tests/unit/test_tracing.py::TestStructlogCorrelation::test_configure_logging_is_reentrant_after_run PASSED [100%]

======================= 22 passed, 14 warnings in 3.82s ========================
```

Line coverage on `src/fraud_engine/utils/tracing.py` reported at **83 %** (195 stmts, 27 missing — the missing lines are the matplotlib-only dispatch branch, the `_TeeStream.isatty` delegation, and defensive `assert` guards). Pre-existing coverage posture; unchanged.

### Pytest — full unit suite

```
$ uv run pytest tests/unit -q --no-cov
...
165 passed, 28 warnings in 6.80s
```

All 165 pre-existing unit tests still pass — no regression in neighbouring test files (loader / splits / baseline / metrics / mlflow_setup / logging / seeding).

### Spec smoke

```
$ uv run python -c '
from fraud_engine.utils.tracing import Run
with Run("smoke", capture_streams=False) as r:
    r.log_param("k", "v")
    r.log_metric("auc", 0.931)
    print(r.run_dir)
'
{"run_id": "1beb00214cb748669ff6494dc3e49e04", "pipeline": "smoke",
 "run_dir": "/home/dchit/projects/fraud-detection-engine/logs/runs/1beb00214cb748669ff6494dc3e49e04",
 "event": "run.start", ...}
/home/dchit/projects/fraud-detection-engine/logs/runs/1beb00214cb748669ff6494dc3e49e04
{"run_id": "1beb00214cb748669ff6494dc3e49e04", "duration_ms": 2.16,
 "event": "run.done", "pipeline": "smoke", ...}

$ ls logs/runs/
12c2527c219b49cf80e036aa421c27af
1beb00214cb748669ff6494dc3e49e04       # ← this turn's smoke
62fe056ee0db492b88dd3621a62d17be
8b34d64b3e2c4eb4844009821ba1a240
b53124a7293f4d37a147f45ab69183c1
bf46f04edf9847a38af6f89538fcaba3

$ cat logs/runs/1beb00214cb748669ff6494dc3e49e04/run.json
{
  "run_id": "1beb00214cb748669ff6494dc3e49e04",
  "pipeline": "smoke",
  "start_time": "2026-04-23T19:07:13.952564+00:00",
  "status": "success",
  "metadata": {
    "k": "v",
    "metrics": {
      "auc": 0.931
    }
  },
  "end_time": "2026-04-23 19:07:13.954724+00:00",
  "duration_ms": 2.16
}
```

`log_param("k", "v")` landed as `metadata["k"] == "v"` (flat in metadata per spec). `log_metric("auc", 0.931)` landed as `metadata["metrics"]["auc"] == 0.931` (nested per spec). `status: "success"`; `run.start` and `run.done` structlog events both emitted with `run_id` attached. Wall time for the smoke run: **~2 ms** (0.931 seconds-of-auc, not seconds-of-runtime).

---

## 7. Acceptance Checklist

From the 0.3.a spec:

- [x] `src/fraud_engine/utils/tracing.py` exports class `Run` with `__init__(pipeline_name, *, settings=None)`, `__enter__`/`__exit__`, `log_param`, `log_metric`, `attach_artifact` (method form), and `run_id` / `run_dir` / `artifacts_dir` properties
- [x] `src/fraud_engine/utils/tracing.py` exports `RunMetadata` dataclass
- [x] Entering `Run` creates `logs/runs/{run_id}/` with `run.json` + `artifacts/`
- [x] `run.json` carries `status` (`running`/`success`/`failed`), `start_time`, `end_time`, `duration_ms`, `metadata`, and exception fields on failure
- [x] Every log line inside the `with` block carries `run_id` (verified via `TestStructlogCorrelation::test_log_records_carry_run_id`)
- [x] `log_param(key, value)` persists into `run.json` before the run exits
- [x] `log_metric(key, value)` persists under `metadata.metrics`
- [x] `attach_artifact(name, path)` copies the file into `artifacts_dir` and returns the destination Path
- [x] `tests/unit/test_tracing.py` passes (22/22)
- [x] Ruff clean on modified files and whole tree
- [x] Mypy clean on `src/` + Sprint 1 script
- [x] Spec smoke command runs, prints `run_dir`, `ls logs/runs/` shows the UUID
- [x] No git commands executed (CLAUDE.md §2)

---

## 8. Non-Goals

- **Removing `run_context`:** Deferred indefinitely. It is a 4-line wrapper that preserves Sprint 1 caller compat; removing it would force edits to `scripts/run_sprint1_baseline.py` and `scripts/_build_eda_notebook.py` for zero behavioural gain.
- **Removing module-level `attach_artifact`:** Deferred indefinitely. It's the type-dispatching entry point Sprint 1 callers use for DataFrames / Figures / dicts. The spec-shaped `Run.attach_artifact(name, path)` method covers the file-copy case; both coexist.
- **Porting `archive/v1-original`'s mlflow-integration hooks:** Out of scope per CLAUDE.md §9 rule 11. Sprint 3 will revisit if MLflow integration needs to thread a `Run` through.
- **`Run.log_artifact_from_object(obj)` convenience (dispatching via the method form):** Not asked for. If a future caller wants one-method ergonomics, can add in a later sprint.
- **Git action:** CLAUDE.md §2 — no stage, commit, or push from Claude Code.

---

Verification passed. Ready for John to commit. No git action from me.
