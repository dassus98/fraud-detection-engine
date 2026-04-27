"""Lineage tracking primitive for DataFrame transformations.

Every transformation in the pipeline emits one `LineageStep` per call:
step name, input/output schema fingerprints, row counts, wall-clock
duration, and the active `run_id`. Steps are appended (one per JSONL
line) to ``{settings.logs_dir}/lineage/{run_id}/lineage.jsonl`` so any
prediction made by the trained model can be traced back through every
transformation that produced its training data.

Business rationale:
    CLAUDE.md §7.2 mandates that every prediction be traceable to its
    raw source via grep-able lineage artefacts. JSONL is `jq`-queryable
    without a database, append-only writes are crash-safe, and a
    16-character schema fingerprint catches dtype drift cheaply (the
    silent ``object → category`` / ``int64 → Int64`` promotions that
    routinely break downstream models). The decorator layers cleanly
    on top of `@log_call` (which already emits start/done/failed
    events for every transformation): `@lineage_step` writes the
    durable JSONL artefact while `@log_call` carries the
    human-readable text-log trail.

Trade-offs considered:
    - `LineageStep.timestamp` is an ISO-8601 UTC string rather than a
      `datetime`. JSONL needs string serialisation; `datetime` would
      force every reader to call `datetime.fromisoformat` before
      filtering. ISO-8601 strings sort lexically, which is how `jq`
      filters time ranges (``select(.timestamp >= "2026-...")``).
    - The decorator fingerprints the *runtime DataFrame's* columns
      and dtypes, not the declared pandera schema. Two reasons: (1)
      pandera 0.22.1 has no native schema-fingerprint method, and (2)
      the runtime shape is what actually shipped — if upstream code
      accidentally promoted a dtype, the fingerprint reflects that
      drift loudly.
    - On exception, the decorator re-raises **without writing a
      `LineageStep`**. Lineage records only successful transformations;
      `@log_call`'s `.failed` event already carries the failure trail
      with full traceback.
    - `LineageLog.append` opens / writes / closes per call. Multiple
      `LineageLog` instances against the same `run_id` therefore
      coexist safely under single-thread use (no shared handle, no
      buffer reordering on crash). Single `LineageStep` lines are
      well under PIPE_BUF (4 KiB) so single-thread append is atomic
      on POSIX. **Not** safe under thread contention — the IEEE-CIS
      pipeline is single-threaded; if Sprint 5's API ever decorates
      a hot path, revisit with `fcntl.flock`.
    - Schema fingerprint is sha256[:16] = 64 bits → collision
      probability ~1e-10 across 1e5 schemas; sufficient for a debug
      fingerprint, not a security primitive.
    - `step_name` is **not** enforced unique within a run. A pipeline
      that loops over chunks may legitimately emit one step per
      chunk; `jq` queries should group on ``(step_name, timestamp)``.

Version history:
    1 — initial. `LineageStep` schema = 8 fields; JSONL one record
        per line; subdir layout ``logs/lineage/{run_id}/lineage.jsonl``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, ParamSpec

import pandas as pd
from structlog.contextvars import bind_contextvars, get_contextvars

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.utils.logging import get_logger, new_run_id

# Subdirectory under `settings.logs_dir` that holds per-run lineage
# trees. Mirrors the existing ``logs/{pipeline_name}/{run_id}.log``
# layout but with one extra namespace so lineage artefacts never
# collide with pipeline text logs.
_LINEAGE_DIR_NAME: Final[str] = "lineage"

# One JSONL file per run_id holds every successful transformation's
# step record. Append-only; one line == one `LineageStep`.
_LINEAGE_FILENAME: Final[str] = "lineage.jsonl"

# Truncate the sha256 schema fingerprint to 16 hex chars (64 bits).
# Collision probability on 1e5 schemas is ~1e-10 — sufficient for a
# debug fingerprint, well short of cryptographic guarantees.
_FINGERPRINT_HEX_CHARS: Final[int] = 16

# Encoding used both for the JSONL file write and for the bytes fed
# to sha256. Keeping them in sync via one constant means a future
# encoding change (e.g. utf-8-sig) cannot drift between the writer
# and the hasher.
_FINGERPRINT_ENCODING: Final[str] = "utf-8"

# structlog event name for the mirrored step record. Carries every
# `LineageStep` field plus the contextvar-bound `run_id`/`pipeline`
# (added automatically by `merge_contextvars` in the project's
# logging configuration) so the regular text log is queryable in
# parallel with the JSONL artefact.
_LINEAGE_EVENT_NAME: Final[str] = "lineage.step"


@dataclass(frozen=True, slots=True)
class LineageStep:
    """One immutable record describing a single transformation call.

    Attributes:
        run_id: The pipeline-run UUID4 hex from structlog contextvars.
            Threads every step in one run together.
        step_name: Caller-supplied label, e.g. ``"interim_clean"``.
            Not unique within a run — chunked pipelines emit one step
            per chunk.
        input_schema_hash: 16-character truncated sha256 of
            ``{column: str(dtype)}`` for the input DataFrame, with
            keys alphabetised. Catches column renames and dtype
            drift; ignores row content.
        output_schema_hash: Same fingerprint computed on the
            transformation's return value. Equality with
            `input_schema_hash` indicates a row-only transformation
            (filter, sort); inequality indicates a structural change.
        input_rows: ``len(input_df)`` at decorator entry.
        output_rows: ``len(output_df)`` at decorator exit. Combined
            with `input_rows` gives a row-delta that lineage tests
            cross-check against `CleanReport.rows_dropped` etc.
        duration_ms: Wall-clock duration in milliseconds, measured
            via `time.perf_counter()`. Matches `@log_call`'s metric.
        timestamp: ISO-8601 UTC string captured at decorator exit.
            String, not `datetime`, so the dataclass round-trips
            through ``json.dumps(asdict(...))`` without a custom
            encoder; lexical comparison matches chronological order
            for `jq` filters.
    """

    run_id: str
    step_name: str
    input_schema_hash: str
    output_schema_hash: str
    input_rows: int
    output_rows: int
    duration_ms: float
    timestamp: str


class LineageLog:
    """Append-only JSONL writer for `LineageStep` records.

    Business rationale:
        Lineage queries (`jq` filters in CI debugging, ad-hoc data
        forensics) are line-oriented: one step per line lets a
        ``cat lineage.jsonl | jq 'select(.step_name=="interim_clean")'``
        pipeline do the work without a database.

    Trade-offs considered:
        - Per-call open / write / close rather than a long-lived
          file handle. The cost (~50µs per append) is trivial against
          a transformation runtime; the benefit is that two
          `LineageLog` instances pointed at the same `run_id`
          coexist safely under single-thread use, with no
          buffer-flush ordering pitfalls if a process crashes
          mid-pipeline.
        - Lazy directory creation: `__init__` does no I/O so
          constructing a `LineageLog` is free. The first `append`
          call mkdirs the parent. Tests that build a log and never
          write therefore leave no artefacts on disk.
        - Single `LineageStep` lines are ~250 bytes, well under
          PIPE_BUF (4096 bytes), so append-mode writes are atomic
          on POSIX for a single thread / process. **Not** safe under
          thread contention; document the boundary on the public
          decorator.
    """

    def __init__(self, run_id: str, settings: Settings | None = None) -> None:
        """Construct a log handle. Performs no I/O.

        Args:
            run_id: The pipeline-run UUID4 hex. The constructor does
                NOT validate the id; callers should pass a value
                obtained from `new_run_id()` or
                `structlog.contextvars.get_contextvars()`.
            settings: Override for the `Settings` singleton. Tests
                pass a `tmp_path`-rooted instance so writes never
                touch the real `logs/` directory.
        """
        self._run_id: str = run_id
        self._settings: Settings = settings or get_settings()

    @property
    def path(self) -> Path:
        """Return the JSONL file path for this run.

        Returns:
            ``{settings.logs_dir}/lineage/{run_id}/lineage.jsonl``.
            Path is computed on every access; the file may or may
            not exist yet.
        """
        return self._settings.logs_dir / _LINEAGE_DIR_NAME / self._run_id / _LINEAGE_FILENAME

    def append(self, step: LineageStep) -> None:
        """Append one `LineageStep` as a single JSONL line.

        Lazy-creates the parent directory. The write goes through a
        per-call ``open(...) / write / close`` cycle so multiple
        `LineageLog` instances against the same `run_id` interleave
        safely under single-thread use.

        Args:
            step: The step to persist.
        """
        target = self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        # `sort_keys=True` and `separators=(",", ":")` give a stable,
        # whitespace-free JSONL line — predictable byte counts in the
        # log directory and stable diffs in tests.
        line = json.dumps(asdict(step), sort_keys=True, separators=(",", ":"))
        with target.open("a", encoding=_FINGERPRINT_ENCODING) as f:
            f.write(line + "\n")

    @classmethod
    def read(cls, run_id: str, settings: Settings | None = None) -> list[LineageStep]:
        """Round-trip the JSONL file for `run_id` into a list of dataclasses.

        Stateless reader — does not require an existing `LineageLog`
        instance. Constructs one internally to derive the path; the
        constructor is I/O-free, so this is cheap. Returns an empty
        list if the backing file is absent or empty.

        Useful for tests, ad-hoc forensics (`scripts/verify_lineage.py`),
        and any future callsite that wants to read without committing to
        a write loop.

        Args:
            run_id: Pipeline-run UUID4 hex whose lineage trail should
                be loaded.
            settings: Override for the `Settings` singleton. Tests
                pass a `tmp_path`-rooted instance.

        Returns:
            Step records in append order. An empty list if the
            backing file is absent or empty.
        """
        instance = cls(run_id, settings)
        if not instance.path.exists():
            return []
        steps: list[LineageStep] = []
        for line in instance.path.read_text(encoding=_FINGERPRINT_ENCODING).splitlines():
            if not line.strip():
                continue
            payload: dict[str, Any] = json.loads(line)
            steps.append(LineageStep(**payload))
        return steps


_P = ParamSpec("_P")


def lineage_step(
    step_name: str,
) -> Callable[[Callable[_P, pd.DataFrame]], Callable[_P, pd.DataFrame]]:
    """Decorate a `DataFrame → DataFrame` callable to emit lineage records.

    Business rationale:
        CLAUDE.md §7.2 mandates one step record per transformation.
        Writing the boilerplate by hand on every loader / cleaner /
        feature module would drift; a parametrised decorator keeps
        the contract mechanical. `@lineage_step` layers on top of
        `@log_call`: `@log_call` handles the entry/exit/failure
        trail in the human-readable log file; `@lineage_step` writes
        the durable JSONL artefact for forensic queries.

    Trade-offs considered:
        - Locates the input DataFrame by **scanning args** for the
          first `pd.DataFrame` instance, not assuming positional 0.
          Handles bound methods (``cleaner.clean(self, df)``) and
          free functions (``f(df)``) without separate decorators.
        - On exception inside the wrapped fn, re-raises **without
          writing a `LineageStep`**. The lineage file therefore
          holds only successful transformations; `@log_call`'s
          `.failed` event covers the failure side.
        - On non-DataFrame return, raises `TypeError`. Catches
          generator-function misuse and the `temporal_split`-style
          ``DataFrame → SplitFrames`` mismatch loudly.
        - If no `run_id` is bound in structlog contextvars, the
          decorator generates a fresh one via `new_run_id()` AND
          binds it. Subsequent calls in the same context reuse it.
          A later `configure_logging("foo")` call without an
          explicit `run_id=` will reuse the bound id; this is
          acceptable (single run id per process) but documented.

    Args:
        step_name: A short, stable identifier for the transformation,
            e.g. ``"interim_clean"`` or ``"feature_t1"``. Used as
            the `step_name` field on every emitted `LineageStep`.

    Returns:
        A decorator that wraps a callable taking a `pd.DataFrame` as
        one of its positional args and returning a `pd.DataFrame`.
    """

    def decorator(
        fn: Callable[_P, pd.DataFrame],
    ) -> Callable[_P, pd.DataFrame]:
        @functools.wraps(fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> pd.DataFrame:
            input_df = _find_dataframe(args)
            input_rows = int(len(input_df))
            input_hash = _fingerprint_dataframe(input_df)

            start = time.perf_counter()
            result = fn(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000.0

            if not isinstance(result, pd.DataFrame):
                raise TypeError(
                    f"lineage_step({step_name!r}) expected a pd.DataFrame "
                    f"return value, got {type(result).__name__}"
                )

            output_rows = int(len(result))
            output_hash = _fingerprint_dataframe(result)

            run_id = _current_run_id()
            step = LineageStep(
                run_id=run_id,
                step_name=step_name,
                input_schema_hash=input_hash,
                output_schema_hash=output_hash,
                input_rows=input_rows,
                output_rows=output_rows,
                duration_ms=round(duration_ms, 3),
                timestamp=datetime.now(UTC).isoformat(),
            )
            LineageLog(run_id).append(step)
            # Mirror to the structlog text log so `tail -f` watchers
            # observe the same shape as the JSONL artefact. `run_id`
            # and `pipeline` come in via `merge_contextvars` in the
            # project's logging configuration.
            get_logger(fn.__module__).info(
                _LINEAGE_EVENT_NAME,
                step_name=step.step_name,
                input_schema_hash=step.input_schema_hash,
                output_schema_hash=step.output_schema_hash,
                input_rows=step.input_rows,
                output_rows=step.output_rows,
                duration_ms=step.duration_ms,
            )
            return result

        return wrapper

    return decorator


def _find_dataframe(args: tuple[Any, ...]) -> pd.DataFrame:
    """Return the first `pd.DataFrame` instance in `args`.

    Scans positionals so the decorator handles both free functions
    (``f(df)``) and bound methods (``cleaner.clean(self, df)``) in
    one wrapper.

    Args:
        args: Positional arguments captured by the wrapper.

    Returns:
        The first `pd.DataFrame` instance found.

    Raises:
        TypeError: If no positional arg is a `pd.DataFrame`.
    """
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            return arg
    raise TypeError(
        "lineage_step expected at least one pd.DataFrame in positional " "args; none was found"
    )


def _fingerprint_dataframe(df: pd.DataFrame) -> str:
    """Compute a deterministic 16-char schema fingerprint.

    Hashes ``{column: str(dtype)}`` for every column in `df`, sorted
    by column name. Catches column renames (different keys), dtype
    drift (different values), and column additions / removals
    (different set of keys). Ignores row count, row content, index,
    and pandera Check constraints.

    An empty-columns DataFrame hashes the empty dict
    (``sha256("{}").hexdigest()[:16]``) — no special-case
    sentinel; uniform algorithm. The exact value is pinned by
    ``test_fingerprint_empty_dataframe_is_deterministic``.

    Args:
        df: The DataFrame to fingerprint. May be empty
            (``len(df) == 0``); the schema is still well-defined.

    Returns:
        First 16 hex chars of the sha256 of the canonical-JSON
        ``{column: dtype}`` map. Two DataFrames with identical
        column-set and dtypes always produce the same hash; any
        structural difference produces a different hash with
        overwhelming probability.
    """
    canonical: dict[str, str] = {col: str(df[col].dtype) for col in sorted(df.columns)}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode(_FINGERPRINT_ENCODING)).hexdigest()
    return digest[:_FINGERPRINT_HEX_CHARS]


def _current_run_id() -> str:
    """Return the active `run_id`; bind a fresh one if none is set.

    Reads from `structlog.contextvars`. If no `run_id` is bound (or
    the bound value is not a string), generates a UUID4 hex via
    `new_run_id()` AND binds it so subsequent calls in the same
    context observe the same id.

    Returns:
        A 32-character UUID4 hex run_id.
    """
    rid = get_contextvars().get("run_id")
    if isinstance(rid, str):
        return rid
    new_id = new_run_id()
    bind_contextvars(run_id=new_id)
    return new_id


__all__ = ["LineageLog", "LineageStep", "lineage_step"]
