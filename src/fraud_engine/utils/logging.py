"""Structured logging setup.

Configures structlog to emit JSON to stdout and human-readable text to
`logs/{pipeline_name}/{run_id}.log`. Every log record carries a
`run_id` contextvar so logs from a single pipeline execution can be
stitched together across subprocesses and services.

Business rationale:
    JSON stdout feeds the same log aggregation pipeline (ELK / Loki)
    that production services use, so dev and prod observability share
    tooling. The text-file mirror means engineers can `tail -f` a log
    without standing up that infrastructure locally.

Trade-offs considered:
    - `structlog.contextvars.merge_contextvars` attaches `run_id` at
      render time; this costs one dict merge per record but guarantees
      correlation even when logs are emitted from pools / async tasks.
    - `ProcessorFormatter` bridges structlog to stdlib handlers so
      third-party libraries that use stdlib logging also inherit our
      JSON/text formatting. The alternative (pure structlog output)
      would double-emit records whenever a library logs directly.
"""

from __future__ import annotations

import functools
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar
from uuid import uuid4

import structlog
from structlog.contextvars import bind_contextvars
from structlog.stdlib import BoundLogger, LoggerFactory, ProcessorFormatter
from structlog.typing import Processor

from fraud_engine.config.settings import get_settings

_P = ParamSpec("_P")
_R = TypeVar("_R")

# Module-level sentinel — configure_logging() is idempotent via this flag.
_CONFIGURED: bool = False


def new_run_id() -> str:
    """Return a fresh UUID4 hex string for a pipeline run.

    The run_id is generated once at pipeline entry and propagated via
    structlog contextvars. Downstream logs, metric tags, and MLflow
    run names all key off this value.

    Returns:
        A 32-character hex string.
    """
    return uuid4().hex


def _shared_processors() -> list[Processor]:
    """Return the processor chain applied to every record.

    Shared between the structlog-native path and the stdlib-bridged
    path so both emit identically-shaped records.

    Returns:
        A list of structlog processors, ordered.
    """
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]


def configure_logging(
    pipeline_name: str,
    run_id: str | None = None,
    log_dir: Path | None = None,
) -> str:
    """Wire structlog + stdlib logging and return the active run_id.

    Sets up two handlers:
        1. stdout → JSON (for aggregation pipelines)
        2. {log_dir}/{pipeline_name}/{run_id}.log → human-readable text

    Binds `run_id` and `pipeline` to the structlog contextvars so every
    record in this process includes them without per-call plumbing.

    Args:
        pipeline_name: Short identifier for the pipeline stage, e.g.
            "ingest", "train", "serve". Determines the log subdir.
        run_id: Existing run_id to reuse (e.g. from an upstream stage).
            If None, a fresh UUID4 hex is generated.
        log_dir: Override for the base log directory. Defaults to
            `settings.logs_dir`.

    Returns:
        The run_id that was bound. Callers should thread this through
        to child processes / subprocess invocations.

    Raises:
        OSError: If `log_dir/pipeline_name` can't be created.
    """
    global _CONFIGURED

    settings = get_settings()
    effective_run_id = run_id or new_run_id()
    effective_log_dir = log_dir or settings.logs_dir
    pipeline_log_dir = effective_log_dir / pipeline_name
    pipeline_log_dir.mkdir(parents=True, exist_ok=True)

    log_file = pipeline_log_dir / f"{effective_run_id}.log"

    shared = _shared_processors()

    # Structlog → stdlib bridge. ProcessorFormatter.wrap_for_formatter
    # transfers the event_dict to the stdlib handler, which then applies
    # the configured ProcessorFormatter to actually render.
    structlog.configure(
        processors=[
            *shared,
            ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=LoggerFactory(),
        wrapper_class=BoundLogger,
        cache_logger_on_first_use=True,
    )

    json_formatter = ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared,
    )
    text_formatter = ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=False),
        foreign_pre_chain=shared,
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(json_formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(text_formatter)

    root_logger = logging.getLogger()
    # Replace any prior handlers so idempotent re-entry during tests
    # doesn't stack duplicate handlers (and duplicate records).
    for existing in list(root_logger.handlers):
        root_logger.removeHandler(existing)
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(settings.log_level)

    bind_contextvars(run_id=effective_run_id, pipeline=pipeline_name)
    _CONFIGURED = True
    return effective_run_id


def get_logger(name: str, **initial_values: Any) -> BoundLogger:
    """Return a structlog logger bound to the given name.

    If `configure_logging()` has not been called, stands up a minimal
    JSON-to-stderr configuration so library code imported in isolation
    still logs safely. Tests that assert on log output should call
    `configure_logging()` explicitly.

    Args:
        name: Logger name; conventionally `__name__`.
        **initial_values: Key-value pairs bound to every record emitted
            from the returned logger.

    Returns:
        A BoundLogger ready for `.info()`, `.warning()`, etc.
    """
    if not _CONFIGURED:
        _configure_fallback()
    logger = structlog.get_logger(name)
    if initial_values:
        logger = logger.bind(**initial_values)
    return logger  # type: ignore[no-any-return]


def _describe(value: Any) -> dict[str, Any]:
    """Return a small, JSON-safe shape summary of `value`.

    Used by `log_call` so logs carry "what went in and what came out"
    without dumping full DataFrame contents (PII risk, log bloat).

    Rules:
        - DataFrames / ndarrays / anything with `.shape`: emit the shape.
        - Strings: emit length only (never the content).
        - Path objects: emit the path string — assumed non-sensitive.
        - Collections (list/dict/tuple/set): emit the length.
        - Scalars (int/float/bool/None): emit the value.
        - Anything else: emit only the type name.

    Args:
        value: The object to summarise.

    Returns:
        A flat dict with a `type` key and one of
        `shape`/`length`/`value`/`path` depending on the branch taken.
    """
    type_name = type(value).__name__
    if hasattr(value, "shape"):
        return {"type": type_name, "shape": list(value.shape)}
    if isinstance(value, str):
        return {"type": "str", "length": len(value)}
    if isinstance(value, Path):
        return {"type": "Path", "path": str(value)}
    if isinstance(value, bool | int | float) or value is None:
        return {"type": type_name, "value": value}
    if hasattr(value, "__len__"):
        try:
            return {"type": type_name, "length": len(value)}
        except TypeError:
            return {"type": type_name}
    return {"type": type_name}


def log_call(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """Decorate a function so every call logs entry, exit, and duration.

    Emits three event names keyed on `fn.__qualname__`:
        - `{qualname}.start` — arg shapes (positional + keyword)
        - `{qualname}.done` — result shape + `duration_ms`
        - `{qualname}.failed` — raised exception type / message +
          `duration_ms`, then the exception is re-raised.

    Business rationale:
        CLAUDE.md §5.5 mandates that every function touching data logs
        input shape, output shape, and duration. Writing that by hand
        on every function drifts; a decorator keeps the discipline
        mechanical and consistent.

    Trade-offs considered:
        - The decorator only peeks at `.shape` / `__len__` / `Path`,
          never the underlying data. The alternative (full `repr`)
          would risk spilling PII into logs.
        - `time.perf_counter` is used for duration. It has ms-scale
          precision on Windows/WSL which is plenty for pipeline timing.
        - `functools.wraps` + `ParamSpec` preserves the wrapped
          signature for mypy and IDE introspection.
        - Exceptions are logged then re-raised. Swallowing them would
          mask failures; eating and returning `None` would be worse
          still.

    Args:
        fn: The callable to wrap. Methods work too — `self` shows up
            in the shape log as an opaque object.

    Returns:
        A wrapped callable with the same signature as `fn`.
    """

    @functools.wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        logger = get_logger(fn.__module__)
        event = fn.__qualname__
        shapes: dict[str, Any] = {f"arg_{i}": _describe(a) for i, a in enumerate(args)}
        shapes.update({f"kw_{k}": _describe(v) for k, v in kwargs.items()})
        logger.info(f"{event}.start", **shapes)

        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.error(
                f"{event}.failed",
                duration_ms=round(duration_ms, 3),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            f"{event}.done",
            duration_ms=round(duration_ms, 3),
            result=_describe(result),
        )
        return result

    return wrapper


def _configure_fallback() -> None:
    """Minimal config so `get_logger()` works before pipeline entry.

    Emits JSON to stderr. No file handler (no `run_id` yet). Called
    once, on first `get_logger()` if `configure_logging()` wasn't run.
    """
    global _CONFIGURED

    shared = _shared_processors()
    structlog.configure(
        processors=[
            *shared,
            ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=LoggerFactory(),
        wrapper_class=BoundLogger,
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=shared,
        )
    )
    root_logger = logging.getLogger()
    for existing in list(root_logger.handlers):
        root_logger.removeHandler(existing)
    root_logger.addHandler(handler)
    root_logger.setLevel(get_settings().log_level)
    _CONFIGURED = True
