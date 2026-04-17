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

import logging
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog
from structlog.contextvars import bind_contextvars
from structlog.stdlib import BoundLogger, LoggerFactory, ProcessorFormatter
from structlog.typing import Processor

from fraud_engine.config.settings import get_settings

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
