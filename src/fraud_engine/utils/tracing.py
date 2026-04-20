"""Per-run directory tracing for non-model pipeline work.

`run_context` is the sibling of MLflow: MLflow tracks model-training
runs (params, metrics, artifacts, lineage into the model registry);
`run_context` tracks *everything else* — data downloads, feature
builds, profiling jobs, one-off scripts — that benefits from a
persistent on-disk record of what happened.

Business rationale:
    Sprint 1's EDA, Sprint 2's feature-pipeline regens, and Sprint 6's
    drift-monitor runs all need a "what did I just do, and can I
    inspect the artefacts after the fact" trail. MLflow is the wrong
    tool for those (they have no model to register). Without
    `run_context`, these runs leave nothing behind but stdout and
    whatever files happened to be written to the cwd. `run_context`
    imposes a fixed layout — `logs/runs/{run_id}/{run.json,stdout.log,
    stderr.log,artifacts/}` — so after-the-fact inspection is
    mechanical.

Trade-offs considered:
    - `@dataclass(frozen=True)` on `Run` makes the handle safe to pass
      between threads / async tasks. Metadata updates go through
      `_rewrite_run_json` which regenerates the JSON from the current
      object — callers that want mutable metadata call
      `dataclasses.replace` and then `_rewrite_run_json`.
    - Stream capture uses a `TeeStream` wrapper so stdout/stderr still
      reach the terminal. A full `redirect_stdout` would make loud
      scripts appear silent. The tee doubles I/O cost but that's
      acceptable for script-scale work.
    - `attach_artifact` dispatches by `isinstance` rather than a
      protocol because we want a known, narrow set of handled types;
      a plugin protocol would encourage silent extensions. The
      `joblib` fallback covers sklearn / LightGBM / generic
      pickleables. PyTorch support lands alongside Sprint 3.
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback as tb_mod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import joblib

# Resolve `matplotlib.figure.Figure` once so `attach_artifact` can dispatch
# on it without re-importing on every call. An empty tuple keeps the
# fallback branch (joblib) reachable when matplotlib is absent.
try:
    from matplotlib.figure import Figure as _MplFigure

    _FIGURE_TYPES: tuple[type, ...] = (_MplFigure,)
except ImportError:  # pragma: no cover - matplotlib is a declared dep
    _FIGURE_TYPES = ()

from fraud_engine.config.settings import get_settings
from fraud_engine.utils.logging import configure_logging, get_logger, new_run_id


@dataclass(frozen=True)
class Run:
    """Immutable handle to an active (or completed) run directory.

    Attributes:
        run_id: 32-char hex UUID; shared with the structlog contextvar
            and any MLflow run opened under the same pipeline.
        pipeline: Short label for the work being done, e.g.
            "feature-build", "profile-raw".
        start_time: UTC entry timestamp.
        run_dir: `{settings.logs_dir}/runs/{run_id}/`.
        artifacts_dir: `{run_dir}/artifacts/`.
        metadata: Caller-supplied payload written into `run.json`.
            Immutable via `frozen=True`; rewrite via
            `dataclasses.replace` + `_rewrite_run_json`.
    """

    run_id: str
    pipeline: str
    start_time: datetime
    run_dir: Path
    artifacts_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)


class _TeeStream:
    """File-like object that mirrors writes to two underlying streams.

    Used by `run_context` to split stdout/stderr between the console
    (so users still see live output) and a per-run log file (so the
    output is archival).
    """

    def __init__(self, primary: TextIO, mirror: TextIO) -> None:
        """Store the two streams without taking ownership of either.

        Args:
            primary: The original stream (usually sys.stdout).
            mirror: The duplicate target (usually the run log file).
        """
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        """Write to both streams, returning the primary's byte count."""
        written = self._primary.write(data)
        self._mirror.write(data)
        return written

    def flush(self) -> None:
        """Flush both streams."""
        self._primary.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        """Delegate TTY-ness to the primary stream.

        Libraries that branch on isatty (tqdm, colourised output) will
        see the original terminal capabilities rather than the log
        file's lack thereof.
        """
        return self._primary.isatty()


def _serialise_run(run: Run, status: str, **extras: Any) -> dict[str, Any]:
    """Render a Run + status into the JSON payload written to disk.

    Private helper; `_rewrite_run_json` is the public write path.

    Args:
        run: The Run handle.
        status: One of "running", "success", "failed".
        **extras: Additional top-level keys (`end_time`, `duration_ms`,
            `exception_type`, `exception_message`, `traceback`).

    Returns:
        A JSON-safe dict.
    """
    payload: dict[str, Any] = {
        "run_id": run.run_id,
        "pipeline": run.pipeline,
        "start_time": run.start_time.isoformat(),
        "status": status,
        "metadata": run.metadata,
    }
    payload.update(extras)
    return payload


def _rewrite_run_json(run: Run, status: str, **extras: Any) -> None:
    """Write (or overwrite) `{run_dir}/run.json` with the current state.

    Business rationale:
        Callers need an authoritative JSON file they can parse from
        outside the Python process — a Grafana dashboard or a
        `jq`-based audit script. Rewriting the file atomically on
        entry/exit is simpler than a streaming JSON-lines format and
        the whole file is < 1 KB.

    Args:
        run: The Run whose directory the file lives in.
        status: "running" / "success" / "failed".
        **extras: Extra top-level fields merged into the payload.
    """
    payload = _serialise_run(run, status, **extras)
    target = run.run_dir / "run.json"
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


@contextmanager
def run_context(
    pipeline: str,
    *,
    metadata: dict[str, Any] | None = None,
    capture_streams: bool = True,
) -> Iterator[Run]:
    """Enter a per-run directory with managed logging and stream capture.

    Creates `{settings.logs_dir}/runs/{run_id}/` and its `artifacts/`
    subdirectory, calls `configure_logging(pipeline, run_id=run_id)`,
    optionally tees stdout/stderr into per-run log files, writes an
    initial `run.json` with `status="running"`, yields the `Run`
    handle, and on exit rewrites `run.json` with a success or failure
    status. Exceptions propagate after the failure record is written.

    Business rationale:
        Every non-model pipeline run (data download, feature build,
        profiling) wants the same after-the-fact trail: a directory,
        captured logs, attached artefacts, a machine-readable summary.
        Packaging that behind a context manager means callers get it
        with one line of overhead.

    Trade-offs considered:
        - The context manager re-enters `configure_logging` even if it
          was already called. `configure_logging` is idempotent (it
          replaces root handlers), so the worst case is a one-time
          handler churn.
        - `capture_streams=True` breaks pytest's stdout/stderr capture.
          Tests must pass `capture_streams=False`; the notebook leaves
          it True to mirror production.

    Args:
        pipeline: Short identifier used for both the structlog
            contextvar and the log subdirectory.
        metadata: Arbitrary JSON-safe payload captured in `run.json`.
            Copied on entry; the dict is not held by reference.
        capture_streams: If True, tee stdout/stderr into
            `stdout.log` / `stderr.log` under the run dir.

    Yields:
        A `Run` handle the caller uses with `attach_artifact`.
    """
    settings = get_settings()
    run_id = new_run_id()
    configure_logging(pipeline, run_id=run_id)
    logger = get_logger(__name__)

    run_dir = settings.logs_dir / "runs" / run_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run = Run(
        run_id=run_id,
        pipeline=pipeline,
        start_time=datetime.now(UTC),
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        metadata=dict(metadata or {}),
    )
    _rewrite_run_json(run, status="running")

    logger.info(
        "run.start",
        run_id=run_id,
        pipeline=pipeline,
        run_dir=str(run_dir),
    )

    stdout_file: TextIO | None = None
    stderr_file: TextIO | None = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if capture_streams:
        stdout_file = (run_dir / "stdout.log").open("w", encoding="utf-8")
        stderr_file = (run_dir / "stderr.log").open("w", encoding="utf-8")
        sys.stdout = _TeeStream(original_stdout, stdout_file)
        sys.stderr = _TeeStream(original_stderr, stderr_file)

    try:
        yield run
    except BaseException as exc:
        duration_ms = (datetime.now(UTC) - run.start_time).total_seconds() * 1000.0
        _rewrite_run_json(
            run,
            status="failed",
            end_time=datetime.now(UTC).isoformat(),
            duration_ms=round(duration_ms, 3),
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback=tb_mod.format_exc(),
        )
        logger.error(
            "run.failed",
            run_id=run_id,
            duration_ms=round(duration_ms, 3),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise
    else:
        duration_ms = (datetime.now(UTC) - run.start_time).total_seconds() * 1000.0
        _rewrite_run_json(
            run,
            status="success",
            end_time=datetime.now(UTC).isoformat(),
            duration_ms=round(duration_ms, 3),
        )
        logger.info(
            "run.done",
            run_id=run_id,
            duration_ms=round(duration_ms, 3),
        )
    finally:
        if capture_streams:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            if stdout_file is not None:
                stdout_file.close()
            if stderr_file is not None:
                stderr_file.close()


def attach_artifact(run: Run, obj: Any, *, name: str) -> Path:
    """Persist `obj` into the run's artifacts directory.

    Dispatches by `isinstance` check:
        - `Path` (existing file): `shutil.copy2` preserving extension.
        - `pandas.DataFrame`: `to_parquet` → `{name}.parquet`.
        - `matplotlib.figure.Figure`: `savefig` → `{name}.png`.
        - `dict` or `list`: JSON-serialised → `{name}.json`.
        - Anything else: `joblib.dump` → `{name}.joblib`.

    Business rationale:
        A consistent artefact layout means Sprint 6's audit scripts
        (and John reading the run after the fact) know exactly what
        extension to expect for each artefact type. A bespoke
        per-caller save path would rot immediately.

    Trade-offs considered:
        - We import matplotlib lazily inside the isinstance check so
          headless environments that don't install matplotlib still
          import this module successfully.
        - `default=str` on the JSON dump covers non-serialisable
          leaves (e.g. numpy scalars, Paths) rather than failing
          loudly. If you want strict serialisation, pre-convert before
          calling.
        - `dpi=150, bbox_inches="tight"` are the matplotlib defaults
          we've standardised on for reports; change them in one place
          if needed rather than in every caller.

    Args:
        run: The `Run` whose `artifacts_dir` receives the file.
        obj: The object to persist.
        name: The file stem (extension is chosen by the dispatch).

    Returns:
        The on-disk Path that was written.

    Raises:
        FileNotFoundError: If `obj` is a Path but doesn't exist.
    """
    artifacts = run.artifacts_dir

    if isinstance(obj, Path):
        if not obj.exists():
            raise FileNotFoundError(f"attach_artifact: source Path {obj} does not exist")
        target = artifacts / (name + obj.suffix)
        shutil.copy2(obj, target)
        return target

    # pandas is a core runtime dependency; import locally to avoid a
    # cycle with any downstream module that imports tracing at
    # package-init time.
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        target = artifacts / f"{name}.parquet"
        obj.to_parquet(target)
        return target

    # matplotlib is a declared dependency but we only reach the import
    # when the caller passes a Figure — keeps headless imports cheap on
    # non-plotting paths.
    if _FIGURE_TYPES and isinstance(obj, _FIGURE_TYPES):
        target = artifacts / f"{name}.png"
        obj.savefig(target, dpi=150, bbox_inches="tight")  # type: ignore[attr-defined]
        return target

    if isinstance(obj, dict | list):
        target = artifacts / f"{name}.json"
        target.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
        return target

    target = artifacts / f"{name}.joblib"
    joblib.dump(obj, target)
    return target


__all__ = [
    "Run",
    "attach_artifact",
    "run_context",
]
