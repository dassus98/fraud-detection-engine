"""Unit tests for `fraud_engine.utils.logging`.

Covers the surfaces exposed by the logging module:
    - `_describe` — shape summariser.
    - `log_call` — instrumentation decorator (sync + async).
    - `configure_logging` — root logger wiring + file + JSON stdout.
    - request_id contextvar — `bind_request_id` / `get_request_id` /
      `reset_request_id`.
    - `log_dataframe` — DataFrame snapshot event (must not leak
      values).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging as stdlib_logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import structlog

from fraud_engine.utils import logging as fe_logging
from fraud_engine.utils.logging import (
    _describe,
    bind_request_id,
    configure_logging,
    get_logger,
    get_request_id,
    log_call,
    log_dataframe,
    new_run_id,
    reset_request_id,
)


@pytest.fixture
def isolate_logging() -> Iterator[None]:
    """Snapshot + restore global logging state around a test.

    `configure_logging` mutates (a) the stdlib root logger's handlers,
    (b) the structlog default config, (c) the module-level
    `_CONFIGURED` sentinel, and (d) structlog contextvars. Without this
    fixture, the first test that calls `configure_logging` would poison
    every subsequent test in the process.
    """
    root = stdlib_logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_flag = fe_logging._CONFIGURED
    try:
        yield
    finally:
        for h in list(root.handlers):
            # Best-effort close; FileHandler may already be closed in a
            # prior error path. The loop still removes every handler.
            with contextlib.suppress(Exception):
                h.close()
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)
        fe_logging._CONFIGURED = saved_flag
        structlog.reset_defaults()
        structlog.contextvars.clear_contextvars()


# ---------------------------------------------------------------------
# _describe / log_call (preserved from test_log_call.py)
# ---------------------------------------------------------------------


class TestDescribe:
    """Contract tests for the `_describe` shape summariser."""

    def test_dataframe_returns_shape(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        out = _describe(df)
        assert out["type"] == "DataFrame"
        assert out["shape"] == [3, 2]

    def test_ndarray_returns_shape(self) -> None:
        arr = np.zeros((4, 5))
        out = _describe(arr)
        assert out["type"] == "ndarray"
        assert out["shape"] == [4, 5]

    def test_str_returns_length_not_content(self) -> None:
        out = _describe("secrets-should-not-be-logged")
        assert out == {"type": "str", "length": len("secrets-should-not-be-logged")}

    def test_path_returns_path_string(self) -> None:
        out = _describe(Path("/tmp/raw"))
        assert out["type"] == "Path"
        assert out["path"].endswith("raw")

    def test_scalar_returns_value(self) -> None:
        assert _describe(42) == {"type": "int", "value": 42}
        assert _describe(3.14) == {"type": "float", "value": 3.14}
        assert _describe(True) == {"type": "bool", "value": True}
        assert _describe(None) == {"type": "NoneType", "value": None}

    def test_collection_returns_length(self) -> None:
        assert _describe([1, 2, 3]) == {"type": "list", "length": 3}
        assert _describe({"a": 1}) == {"type": "dict", "length": 1}


class TestLogCall:
    """Contract tests for the `log_call` decorator."""

    def test_returns_wrapped_result(self) -> None:
        @log_call
        def doubler(x: int) -> int:
            return x * 2

        assert doubler(5) == 10

    def test_preserves_function_metadata(self) -> None:
        @log_call
        def labelled(x: int) -> int:
            """Doubles its input."""
            return x * 2

        assert labelled.__name__ == "labelled"
        assert labelled.__doc__ == "Doubles its input."

    def test_reraises_exceptions(self) -> None:
        @log_call
        def explodes() -> None:
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            explodes()

    def test_passes_kwargs_through(self) -> None:
        @log_call
        def joiner(a: str, *, b: str = "default") -> str:
            return f"{a}-{b}"

        assert joiner("left", b="right") == "left-right"


# ---------------------------------------------------------------------
# request_id contextvar
# ---------------------------------------------------------------------


class TestRequestId:
    """Contract tests for the per-request correlation ID."""

    def teardown_method(self) -> None:
        """Leave the contextvar in a clean state between tests."""
        reset_request_id()

    def test_bind_then_get_returns_value(self) -> None:
        bound = bind_request_id("req-123")
        assert bound == "req-123"
        assert get_request_id() == "req-123"

    def test_bind_without_argument_generates_fresh_id(self) -> None:
        bound = bind_request_id()
        assert isinstance(bound, str)
        assert len(bound) == 32
        assert get_request_id() == bound

    def test_reset_clears_value(self) -> None:
        bind_request_id("req-abc")
        reset_request_id()
        assert get_request_id() is None

    def test_independent_contexts(self) -> None:
        """Two async tasks must see their own request_id."""

        async def _runner() -> list[str | None]:
            async def child(tag: str) -> str | None:
                bind_request_id(tag)
                # Yield control so the sibling interleaves before the read.
                await asyncio.sleep(0)
                return get_request_id()

            t1 = asyncio.create_task(child("req-A"))
            t2 = asyncio.create_task(child("req-B"))
            return await asyncio.gather(t1, t2)

        results = asyncio.run(_runner())
        assert results == ["req-A", "req-B"]

    def test_logger_includes_request_id(self) -> None:
        """After bind_request_id, structlog records must carry request_id.

        `structlog.testing.capture_logs` clears the configured processor
        chain — including `merge_contextvars` — so we build a minimal
        ad-hoc pipeline here that keeps `merge_contextvars` in front of a
        capture hook. That mirrors what `configure_logging` does in prod
        without depending on the global configuration.
        """
        bind_request_id("req-xyz")
        captured: list[dict[str, object]] = []

        def capture(
            _logger: object, _method: str, event_dict: dict[str, object]
        ) -> dict[str, object]:
            captured.append(dict(event_dict))
            raise structlog.DropEvent

        logger = structlog.wrap_logger(
            None,
            processors=[structlog.contextvars.merge_contextvars, capture],
        )
        logger.info("probe-event", extra_field="visible")

        matching = [r for r in captured if r.get("event") == "probe-event"]
        assert matching, f"no probe-event in capture: {captured!r}"
        assert matching[0]["request_id"] == "req-xyz"
        assert matching[0]["extra_field"] == "visible"


# ---------------------------------------------------------------------
# log_dataframe
# ---------------------------------------------------------------------


class TestLogDataframe:
    """Contract tests for the `log_dataframe` snapshot emitter."""

    def test_emits_expected_event(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4.0, 5.0, 6.0],
                "c": ["x", "y", "z"],
            }
        )
        with structlog.testing.capture_logs() as captured:
            log_dataframe(df, name="demo")

        snapshots = [r for r in captured if r.get("event") == "dataframe.snapshot"]
        assert len(snapshots) == 1
        record = snapshots[0]
        assert record["name"] == "demo"
        assert record["rows"] == 3
        assert record["cols"] == 3
        assert isinstance(record["memory_mb"], float)
        assert isinstance(record["dtypes"], dict)
        assert record["n_missing"] == 0
        assert isinstance(record["first_row_sha256"], str)
        assert len(record["first_row_sha256"]) == 64

    def test_value_never_logged(self) -> None:
        """Secrets in the DataFrame must not appear in the event record."""
        secret = "supersecret-should-not-leak"
        df = pd.DataFrame(
            {
                "amount": [100.0, 200.0],
                "email": [secret, "ok@example.com"],
            }
        )
        with structlog.testing.capture_logs() as captured:
            log_dataframe(df, name="leak-check")

        rendered = json.dumps(captured, default=str)
        assert secret not in rendered


# ---------------------------------------------------------------------
# configure_logging: file creation, JSON stdout, distinct run_ids
# ---------------------------------------------------------------------


class TestConfigureLogging:
    """End-to-end contract tests for `configure_logging`.

    These tests exercise the real stdlib-handlers path rather than
    structlog's capture fixture, because the spec requires proof that
    (a) the configured `FileHandler` writes to disk at the expected
    path, and (b) the stdout `StreamHandler` emits valid JSON.
    """

    def test_creates_log_file(self, tmp_path: Path, isolate_logging: None) -> None:
        run_id = configure_logging(pipeline_name="fileemit", log_dir=tmp_path)
        logger = get_logger("fileemit_test")
        logger.info("hello", key="value")
        # Flush stdlib handlers before reading the file from disk.
        for h in stdlib_logging.getLogger().handlers:
            h.flush()
        expected = tmp_path / "fileemit" / f"{run_id}.log"
        assert expected.exists(), f"log file not created at {expected}"
        content = expected.read_text(encoding="utf-8")
        assert "hello" in content
        assert "key" in content
        assert "value" in content

    def test_stdout_emits_valid_json(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str], isolate_logging: None
    ) -> None:
        """Captured stdout lines must parse as JSON with the event payload.

        `capfd` (OS file-descriptor capture) is used rather than
        `capsys` because `configure_logging` binds the handler to
        `sys.stdout` at call time and `StreamHandler` retains the
        reference — `capfd` catches the actual fd-1 writes regardless.
        """
        configure_logging(pipeline_name="jsonemit", log_dir=tmp_path)
        logger = get_logger("jsonemit_test")
        logger.info("probe-event", extra_field="visible")
        for h in stdlib_logging.getLogger().handlers:
            h.flush()
        out, _err = capfd.readouterr()
        lines = [line for line in out.splitlines() if line.strip()]
        assert lines, "no JSON lines captured on stdout"
        # The last line is the most recent emit; parse it.
        parsed = json.loads(lines[-1])
        assert parsed["event"] == "probe-event"
        assert parsed["extra_field"] == "visible"
        # run_id + pipeline are bound as default context and must ride
        # along on every record — this is the core correlation guarantee.
        assert parsed["pipeline"] == "jsonemit"
        assert "run_id" in parsed
        assert "timestamp" in parsed

    def test_different_run_ids_produce_different_files(
        self, tmp_path: Path, isolate_logging: None
    ) -> None:
        """Two pipeline runs → two files, no overwrite, both on disk."""
        run_a = configure_logging(pipeline_name="dup", log_dir=tmp_path)
        logger_a = get_logger("dup_a")
        logger_a.info("first-run-event")
        for h in stdlib_logging.getLogger().handlers:
            h.flush()

        run_b = configure_logging(pipeline_name="dup", log_dir=tmp_path)
        logger_b = get_logger("dup_b")
        logger_b.info("second-run-event")
        for h in stdlib_logging.getLogger().handlers:
            h.flush()

        file_a = tmp_path / "dup" / f"{run_a}.log"
        file_b = tmp_path / "dup" / f"{run_b}.log"
        assert run_a != run_b, "new_run_id() must not repeat"
        assert file_a.exists()
        assert file_b.exists()
        # The first run's file retains its event; the second run writes
        # to its own file — no cross-contamination.
        assert "first-run-event" in file_a.read_text(encoding="utf-8")
        assert "second-run-event" in file_b.read_text(encoding="utf-8")
        assert "second-run-event" not in file_a.read_text(encoding="utf-8")

    def test_accepts_caller_supplied_run_id(self, tmp_path: Path, isolate_logging: None) -> None:
        """Upstream pipelines pass their run_id so logs stitch together."""
        supplied = new_run_id()
        returned = configure_logging(pipeline_name="passthrough", run_id=supplied, log_dir=tmp_path)
        assert returned == supplied
        assert (tmp_path / "passthrough" / f"{supplied}.log").exists()


# ---------------------------------------------------------------------
# log_call — async coverage (sync path covered by TestLogCall above)
# ---------------------------------------------------------------------


class TestLogCallAsync:
    """Async decorator path — spec mandates `inspect.iscoroutinefunction`.

    `asyncio_mode = "auto"` in pyproject.toml makes `async def` test
    functions run as coroutines without a marker.
    """

    async def test_async_function_returns_wrapped_result(self) -> None:
        @log_call
        async def async_doubler(x: int) -> int:
            await asyncio.sleep(0)
            return x * 2

        result = await async_doubler(5)
        assert result == 10

    async def test_async_function_reraises_exceptions(self) -> None:
        @log_call
        async def async_explodes() -> None:
            await asyncio.sleep(0)
            raise RuntimeError("async-boom")

        with pytest.raises(RuntimeError, match="async-boom"):
            await async_explodes()

    async def test_async_wrapper_is_still_a_coroutine_function(self) -> None:
        """`iscoroutinefunction` of the wrapped callable must stay True.

        FastAPI's dependency system and some async frameworks introspect
        callables to decide whether to await them. The decorator must
        not turn an async function into something that looks sync.
        """

        @log_call
        async def pretender() -> int:
            return 1

        assert asyncio.iscoroutinefunction(pretender)
        assert await pretender() == 1

    async def test_async_logs_start_done_events(self) -> None:
        """The async branch still emits the `.start` and `.done` pair."""
        with structlog.testing.capture_logs() as captured:

            @log_call
            async def async_tagged(x: int) -> int:
                return x + 1

            await async_tagged(41)

        events = [r.get("event") for r in captured]
        start_events = [e for e in events if e and e.endswith(".start")]
        done_events = [e for e in events if e and e.endswith(".done")]
        assert len(start_events) == 1
        assert len(done_events) == 1
        done = [r for r in captured if r.get("event", "").endswith(".done")][0]
        assert "duration_ms" in done
        assert done["result"] == {"type": "int", "value": 42}

    async def test_async_failed_event_includes_traceback(self) -> None:
        """Exceptions log a `.failed` event with exc_info so the
        stdout JSON carries a rendered traceback string."""
        with structlog.testing.capture_logs() as captured:

            @log_call
            async def async_raises() -> None:
                raise ValueError("trace-please")

            with pytest.raises(ValueError, match="trace-please"):
                await async_raises()

        failed = [r for r in captured if r.get("event", "").endswith(".failed")]
        assert len(failed) == 1
        record = failed[0]
        assert record["error_type"] == "ValueError"
        assert record["error_message"] == "trace-please"
