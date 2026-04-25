"""Per-run directory tracing for non-model pipeline work.

The class `Run` (and its thin functional sibling `run_context`) is the
counterpart to MLflow: MLflow tracks model-training runs (params,
metrics, artifacts, lineage into the model registry); `Run` tracks
*everything else* — data downloads, feature builds, profiling jobs,
one-off scripts — that benefits from a persistent on-disk record of
what happened.

Business rationale:
    Sprint 1's EDA, Sprint 2's feature-pipeline regens, and Sprint 6's
    drift-monitor runs all need a "what did I just do, and can I
    inspect the artefacts after the fact" trail. MLflow is the wrong
    tool for those (they have no model to register). Without a `Run`,
    these jobs leave nothing behind but stdout and whatever files
    happened to be written to the cwd. `Run` imposes a fixed layout —
    `logs/runs/{run_id}/{run.json,stdout.log,stderr.log,artifacts/}` —
    so after-the-fact inspection is mechanical.

Trade-offs considered:
    - Class-based `Run` exposes `log_param` / `log_metric` /
      `attach_artifact` methods that mutate live state and re-persist
      `run.json`. A purely functional API (the earlier
      `run_context`-only shape) required callers to thread the
      `metadata` dict through at construction time, which broke down
      as soon as a pipeline wanted to log a metric discovered mid-run.
      The mutable class is the natural fit; tests assert the JSON
      persists after each mutation.
    - `RunMetadata` is a read-only snapshot dataclass emitted into
      `run.json`; it exists so external consumers (Grafana panels,
      `jq`-based audits) have a documented schema for the file rather
      than relying on `dict[str, Any]`.
    - Stream capture uses a `TeeStream` wrapper so stdout/stderr still
      reach the terminal. A full `redirect_stdout` would make loud
      scripts appear silent. The tee doubles I/O cost but that's
      acceptable for script-scale work. Tests must pass
      `capture_streams=False` so pytest's capsys sees the output.
    - Module-level `attach_artifact(run, obj, *, name)` dispatches by
      `isinstance` on the *object* type — kept as a separate entry
      point for Sprint 1+ callers that pass DataFrames / Figures /
      dicts. The spec-shaped method `Run.attach_artifact(name, path)`
      is a strict file-copy: simpler contract, no type dispatch.
      Neither form is a wrapper on the other; they're sibling APIs.
    - `run_context` is retained as a thin `@contextmanager` wrapper so
      existing callers (Sprint 1 baseline script, notebook builders)
      keep working without migration. It now yields the class-based
      `Run`; the read surface (`run_id`, `run_dir`, `artifacts_dir`,
      `metadata`) is identical via properties.
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
from types import TracebackType
from typing import Any, Literal, TextIO

import joblib

# Resolve `matplotlib.figure.Figure` once so `attach_artifact` can dispatch
# on it without re-importing on every call. An empty tuple keeps the
# fallback branch (joblib) reachable when matplotlib is absent.
try:
    from matplotlib.figure import Figure as _MplFigure

    _FIGURE_TYPES: tuple[type, ...] = (_MplFigure,)
except ImportError:  # pragma: no cover - matplotlib is a declared dep
    _FIGURE_TYPES = ()

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.utils.logging import configure_logging, get_logger, new_run_id

# Literal type for the three terminal states we write into run.json.
# Centralised so tests, consumers, and the dataclass all agree.
RunStatus = Literal["running", "success", "failed"]


@dataclass
class RunMetadata:
    """Read-only snapshot of a `Run`'s metadata written into `run.json`.

    This is the documented schema external consumers (Grafana panels,
    `jq`-based audit scripts, the Sprint 6 drift dashboard) key off.
    The dataclass is *not* frozen because `Run._build_metadata()`
    constructs it fresh on every persist; callers should treat
    instances as immutable.

    Business rationale:
        Without a typed schema for `run.json`, every consumer would
        duplicate the same string keys and drift when fields are
        added. `RunMetadata` makes the schema discoverable and
        type-checkable from the Python side; the JSON on disk still
        looks the same.

    Attributes:
        run_id: 32-char hex UUID shared with the structlog contextvar
            and any MLflow run opened under the same pipeline.
        pipeline_name: Short label for the work being done, e.g.
            "feature-build", "profile-raw".
        start_time: UTC entry timestamp.
        status: One of "running", "success", "failed" — the terminal
            state (or the in-flight state for running jobs).
        end_time: UTC exit timestamp (None while status == "running").
        duration_ms: Wall-clock duration at exit (None while running).
        params: Caller-supplied parameters, set via
            `Run.log_param(key, value)`. JSON-safe.
        metrics: Caller-supplied metrics, set via
            `Run.log_metric(key, value)`. Values are floats.
        extra: Free-form metadata passed at construction
            (e.g. the Sprint 1 baseline's `{"variants_requested": ...}`).
    """

    run_id: str
    pipeline_name: str
    start_time: datetime
    status: RunStatus = "running"
    end_time: datetime | None = None
    duration_ms: float | None = None
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class _TeeStream:
    """File-like object that mirrors writes to two underlying streams.

    Used by `Run` to split stdout/stderr between the console (so users
    still see live output) and a per-run log file (so the output is
    archival).
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


class Run:
    """Context-manager handle to an active (or completed) run directory.

    `Run` owns the per-run directory tree, structlog configuration,
    optional stdout/stderr tee, and the `run.json` file. Callers enter
    it as a context manager:

        with Run("feature-build") as run:
            run.log_param("variant", "temporal")
            run.log_metric("auc", 0.93)
            run.attach_artifact("splits.parquet", path_to_file)

    Business rationale:
        A per-run directory with captured logs, attached artefacts,
        and a machine-readable summary is the cheapest observability
        that still lets us reconstruct what a pipeline did after the
        fact. The mutable `log_param` / `log_metric` methods are
        necessary because pipelines discover parameters and metrics
        mid-run (e.g. after a model fits and reports its validation
        AUC), not all at construction time.

    Trade-offs considered:
        - Each `log_param` / `log_metric` call rewrites `run.json` so
          an external observer sees the latest state. The cost is
          trivial (< 1KB JSON write) and the benefit is live
          visibility without a streaming JSON-lines format.
        - Accessors (`run_id`, `run_dir`, etc.) raise `RuntimeError`
          if read before `__enter__` rather than returning partial
          state — a half-configured `Run` shouldn't be passed around.
        - `capture_streams` defaults to False. Production scripts opt
          in via `run_context(...)` (which keeps the legacy default of
          True for backwards compat); tests and direct library use
          default to False so pytest's capsys works unmodified.
        - `__exit__` returns None (implicit False) so exceptions
          propagate after the failure record is written. Swallowing
          would mask bugs; the existing `run_context` did this and the
          class preserves the contract.
    """

    def __init__(
        self,
        pipeline_name: str,
        *,
        settings: Settings | None = None,
        metadata: dict[str, Any] | None = None,
        capture_streams: bool = False,
    ) -> None:
        """Construct a Run. The directory is not created until `__enter__`.

        Args:
            pipeline_name: Short identifier used for both the structlog
                contextvar and the log subdirectory.
            settings: Optional Settings override. Defaults to
                `get_settings()` on `__enter__` so tests that
                monkeypatch env vars pick up fresh values without
                needing to construct a Settings eagerly.
            metadata: Free-form payload captured in `run.json` under
                the `extra` key. Copied on entry; the dict is not held
                by reference.
            capture_streams: If True, tee stdout/stderr into
                `stdout.log` / `stderr.log` under the run dir.
        """
        self._pipeline_name = pipeline_name
        self._settings_override = settings
        self._extra: dict[str, Any] = dict(metadata or {})
        self._capture_streams = capture_streams

        # Lazily-initialised state. Populated in __enter__; accessors
        # raise RuntimeError if read beforehand.
        self._entered: bool = False
        self._run_id: str | None = None
        self._run_dir: Path | None = None
        self._artifacts_dir: Path | None = None
        self._start_time: datetime | None = None
        self._params: dict[str, Any] = {}
        self._metrics: dict[str, float] = {}
        self._stdout_file: TextIO | None = None
        self._stderr_file: TextIO | None = None
        self._original_stdout: TextIO | None = None
        self._original_stderr: TextIO | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> Run:
        """Create the directory tree, configure logging, persist run.json.

        Returns:
            This instance, so `with Run(...) as r` binds `r` to it.
        """
        settings = self._settings_override or get_settings()
        run_id = new_run_id()
        configure_logging(self._pipeline_name, run_id=run_id)

        run_dir = settings.logs_dir / "runs" / run_id
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._run_id = run_id
        self._run_dir = run_dir
        self._artifacts_dir = artifacts_dir
        self._start_time = datetime.now(UTC)
        self._entered = True

        self._write_run_json(status="running")

        logger = get_logger(__name__)
        logger.info(
            "run.start",
            run_id=run_id,
            pipeline=self._pipeline_name,
            run_dir=str(run_dir),
        )

        if self._capture_streams:
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            self._stdout_file = (run_dir / "stdout.log").open("w", encoding="utf-8")
            self._stderr_file = (run_dir / "stderr.log").open("w", encoding="utf-8")
            sys.stdout = _TeeStream(self._original_stdout, self._stdout_file)
            sys.stderr = _TeeStream(self._original_stderr, self._stderr_file)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Finalise run.json, restore streams, let exceptions propagate.

        Args:
            exc_type: Exception class if the with-block raised.
            exc_val: Exception instance if the with-block raised.
            exc_tb: Traceback object if the with-block raised.
        """
        # _start_time is set in __enter__; the None guard keeps mypy
        # happy and is defensive against misuse.
        start = self._start_time or datetime.now(UTC)
        end_time = datetime.now(UTC)
        duration_ms = (end_time - start).total_seconds() * 1000.0
        logger = get_logger(__name__)

        if exc_val is not None:
            self._write_run_json(
                status="failed",
                end_time=end_time,
                duration_ms=round(duration_ms, 3),
                exception_type=type(exc_val).__name__,
                exception_message=str(exc_val),
                traceback=tb_mod.format_exc(),
            )
            logger.error(
                "run.failed",
                run_id=self._run_id,
                duration_ms=round(duration_ms, 3),
                error_type=type(exc_val).__name__,
                error_message=str(exc_val),
            )
        else:
            self._write_run_json(
                status="success",
                end_time=end_time,
                duration_ms=round(duration_ms, 3),
            )
            logger.info(
                "run.done",
                run_id=self._run_id,
                duration_ms=round(duration_ms, 3),
            )

        if self._capture_streams:
            if self._original_stdout is not None:
                sys.stdout = self._original_stdout
            if self._original_stderr is not None:
                sys.stderr = self._original_stderr
            if self._stdout_file is not None:
                self._stdout_file.close()
            if self._stderr_file is not None:
                self._stderr_file.close()

        # Implicit None return → exceptions propagate. Spelled out for
        # reviewers: we do NOT swallow.

    # ------------------------------------------------------------------
    # public API — mutate + persist
    # ------------------------------------------------------------------

    def log_param(self, key: str, value: Any) -> None:
        """Record a parameter and persist to `run.json`.

        Business rationale:
            Parameters are the *inputs* to a run (hyperparameters,
            dataset versions, variant flags). Separating them from
            metrics — the *outputs* — mirrors MLflow's convention so
            a reviewer scanning `run.json` sees two clearly distinct
            sections. Persisting on every call means the JSON is a
            live record even if the run later crashes.

        Args:
            key: Parameter name. `"metrics"` is reserved (used for
                the nested log_metric sub-dict in run.json); passing
                it raises ValueError.
            value: JSON-safe value. Stringified by the JSON encoder if
                not natively serialisable (via `default=str`).

        Raises:
            RuntimeError: If called before `__enter__`.
            ValueError: If `key == "metrics"`.
        """
        self._require_entered("log_param")
        if key == "metrics":
            raise ValueError(
                "'metrics' is reserved for the log_metric sub-dict in run.json; "
                "use a different parameter name."
            )
        self._params[key] = value
        self._write_run_json(status="running")

    def log_metric(self, key: str, value: float) -> None:
        """Record a numeric metric and persist to `run.json`.

        Business rationale:
            Metrics carry the run's outcomes — validation AUC, fraud
            capture rate, expected-cost-per-decision. Nesting them
            under a dedicated `metrics` key means Sprint 4's
            economic evaluator and Sprint 6's drift monitor can emit
            the same shape without clashing with `params`.

        Args:
            key: Metric name.
            value: Numeric value. Coerced to float — passing a numpy
                scalar or an int works, passing a non-numeric value
                raises TypeError at the coercion step.
        """
        self._require_entered("log_metric")
        self._metrics[key] = float(value)
        self._write_run_json(status="running")

    def attach_artifact(self, name: str, path: Path | str) -> Path:
        """Copy an existing file into `artifacts/` under the given name.

        This is the spec-shaped method — it takes an existing file on
        disk and copies it verbatim. For type-dispatching (DataFrame
        → parquet, Figure → png, dict → json, anything → joblib),
        use the module-level `attach_artifact(run, obj, *, name)`
        function instead.

        Business rationale:
            Some artefacts are already on disk by the time the caller
            reaches this point (e.g. a Parquet split manifest, a
            downloaded Kaggle zip). Forcing a round-trip through
            pandas or joblib would be wasteful and lossy; a direct
            copy preserves the file bit-for-bit.

        Args:
            name: Destination filename inside `artifacts/`. The caller
                is responsible for including the correct extension.
            path: Source path on disk. `str` is accepted for
                ergonomic reasons; it's promoted to Path immediately.

        Returns:
            The destination Path (`artifacts_dir / name`).

        Raises:
            RuntimeError: If called before `__enter__`.
            FileNotFoundError: If `path` does not exist.
        """
        self._require_entered("attach_artifact")
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Run.attach_artifact: source path {source} does not exist")
        # Cast for mypy: _artifacts_dir is set by _require_entered's
        # precondition, but the attribute type is `Path | None`.
        assert self._artifacts_dir is not None  # noqa: S101
        destination = self._artifacts_dir / name
        shutil.copy2(source, destination)
        return destination

    # ------------------------------------------------------------------
    # public API — read surface (properties)
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        """Return the run's UUID4 hex string.

        Raises:
            RuntimeError: If accessed before `__enter__`.
        """
        self._require_entered("run_id")
        assert self._run_id is not None  # noqa: S101 — _require_entered guarantees
        return self._run_id

    @property
    def run_dir(self) -> Path:
        """Return the absolute path of the run directory.

        Raises:
            RuntimeError: If accessed before `__enter__`.
        """
        self._require_entered("run_dir")
        assert self._run_dir is not None  # noqa: S101
        return self._run_dir

    @property
    def artifacts_dir(self) -> Path:
        """Return the absolute path of the `artifacts/` subdirectory.

        Raises:
            RuntimeError: If accessed before `__enter__`.
        """
        self._require_entered("artifacts_dir")
        assert self._artifacts_dir is not None  # noqa: S101
        return self._artifacts_dir

    @property
    def start_time(self) -> datetime:
        """Return the UTC entry timestamp.

        Raises:
            RuntimeError: If accessed before `__enter__`.
        """
        self._require_entered("start_time")
        assert self._start_time is not None  # noqa: S101
        return self._start_time

    @property
    def pipeline(self) -> str:
        """Return the pipeline name passed at construction.

        This is readable *before* `__enter__` because it's part of the
        construction contract, not runtime state.
        """
        return self._pipeline_name

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a read-only view of the metadata currently on the run.

        The dict is a fresh shallow copy on every call so callers
        can't mutate private state by reference. For writes, use
        `log_param` / `log_metric`.

        Returns:
            A dict with keys `params`, `metrics`, `extra` mirroring
            the sections of `run.json`.
        """
        return {
            "params": dict(self._params),
            "metrics": dict(self._metrics),
            "extra": dict(self._extra),
        }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _require_entered(self, operation: str) -> None:
        """Raise if `__enter__` hasn't populated runtime state yet.

        Args:
            operation: Name of the operation being attempted, used in
                the error message so stack traces point at the right
                API call.

        Raises:
            RuntimeError: If the Run hasn't been entered.
        """
        if not self._entered:
            raise RuntimeError(
                f"Run.{operation} called before entering the context manager; "
                "use `with Run(...) as run:` first."
            )

    def _build_metadata(self, status: RunStatus) -> RunMetadata:
        """Assemble a `RunMetadata` snapshot for the current state.

        Args:
            status: The status field to stamp on the snapshot.

        Returns:
            A fresh `RunMetadata`.
        """
        assert self._run_id is not None  # noqa: S101
        assert self._start_time is not None  # noqa: S101
        return RunMetadata(
            run_id=self._run_id,
            pipeline_name=self._pipeline_name,
            start_time=self._start_time,
            status=status,
            params=dict(self._params),
            metrics=dict(self._metrics),
            extra=dict(self._extra),
        )

    def _write_run_json(self, *, status: RunStatus, **extras: Any) -> None:
        """Atomically rewrite `{run_dir}/run.json` with current state.

        Business rationale:
            External tooling (Grafana, `jq`, the Sprint 6 audit
            scripts) wants an authoritative JSON file they can parse
            outside the Python process. Rewriting on every mutation is
            simpler than a streaming JSON-lines format and the whole
            file is < 1KB.

        Args:
            status: Current lifecycle state.
            **extras: Additional top-level fields merged into the
                payload (e.g. `end_time`, `duration_ms`,
                `exception_type`, `traceback` at exit).
        """
        assert self._run_dir is not None  # noqa: S101 — only called after __enter__
        payload = _serialise_metadata(self._build_metadata(status), **extras)
        (self._run_dir / "run.json").write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )


def _serialise_metadata(meta: RunMetadata, **extras: Any) -> dict[str, Any]:
    """Render a `RunMetadata` into the JSON payload written to disk.

    Private helper. Kept separate from `Run._write_run_json` so the
    payload shape is testable in isolation if we ever need it.

    Shape of the `metadata` block:
        - `extra` keys are merged flat at the top.
        - `params` keys (from `log_param`) are merged flat at the top
          as well — the spec calls for `metadata["n_features"] == 123`
          after `run.log_param("n_features", 123)`.
        - `metrics` keys (from `log_metric`) are nested under a
          dedicated `metrics` sub-dict so output metrics live apart
          from inputs. The sub-dict is omitted when empty so runs that
          never call `log_metric` produce the minimal payload.

    Args:
        meta: The snapshot to serialise.
        **extras: Additional top-level keys (`end_time`,
            `duration_ms`, `exception_type`, `exception_message`,
            `traceback`).

    Returns:
        A JSON-safe dict.
    """
    metadata_block: dict[str, Any] = dict(meta.extra)
    # log_param keys flatten into metadata so a reviewer scanning
    # run.json sees them inline next to the caller-supplied extras.
    metadata_block.update(meta.params)
    if meta.metrics:
        # Keep metrics nested under a reserved key so they can't
        # collide with params/extras on casual scans.
        metadata_block["metrics"] = dict(meta.metrics)

    payload: dict[str, Any] = {
        "run_id": meta.run_id,
        "pipeline": meta.pipeline_name,
        "start_time": meta.start_time.isoformat(),
        "status": meta.status,
        "metadata": metadata_block,
    }
    payload.update(extras)
    return payload


@contextmanager
def run_context(
    pipeline: str,
    *,
    metadata: dict[str, Any] | None = None,
    capture_streams: bool = True,
) -> Iterator[Run]:
    """Thin functional wrapper around `Run` for backwards compatibility.

    Sprint 1's baseline script and the EDA notebook builder use this
    form; migrating them is unnecessary since a `Run` yielded here has
    the same read surface (`run_id`, `run_dir`, `artifacts_dir`) and
    accepts the module-level `attach_artifact(run, obj, *, name)`
    dispatcher.

    Business rationale:
        The class-based `Run` is the richer API (live `log_param` /
        `log_metric`), but not every caller needs it. Preserving
        `run_context` as a one-liner entry point keeps short scripts
        ergonomic while routing through the same underlying
        implementation.

    Args:
        pipeline: Short identifier used for both the structlog
            contextvar and the log subdirectory.
        metadata: Arbitrary JSON-safe payload captured under the
            `extra` section of the Run. Copied on entry; the dict is
            not held by reference.
        capture_streams: If True (the default, matching the legacy
            behaviour), tee stdout/stderr into `stdout.log` /
            `stderr.log` under the run dir.

    Yields:
        A `Run` handle the caller uses with `attach_artifact`,
        `log_param`, `log_metric`.
    """
    with Run(
        pipeline,
        metadata=metadata,
        capture_streams=capture_streams,
    ) as run:
        yield run


def attach_artifact(run: Run, obj: Any, *, name: str) -> Path:
    """Persist `obj` into the run's artifacts directory.

    Module-level entry point that dispatches by `isinstance`:
        - `Path` (existing file): `shutil.copy2` preserving extension.
        - `pandas.DataFrame`: `to_parquet` → `{name}.parquet`.
        - `matplotlib.figure.Figure`: `savefig` → `{name}.png`.
        - `dict` or `list`: JSON-serialised → `{name}.json`.
        - Anything else: `joblib.dump` → `{name}.joblib`.

    Use this when you're handing the function *a Python object* that
    could be any of several types. Use `Run.attach_artifact(name,
    path)` when you already have a file on disk and want a strict
    copy.

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
    "RunMetadata",
    "RunStatus",
    "attach_artifact",
    "run_context",
]
