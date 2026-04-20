"""Unit tests for `fraud_engine.utils.logging`.

Covers the four surfaces exposed by the logging module:
    - `_describe` — shape summariser.
    - `log_call` — instrumentation decorator.
    - request_id contextvar — `bind_request_id` / `get_request_id` /
      `reset_request_id`.
    - `log_dataframe` — DataFrame snapshot event (must not leak
      values).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import structlog

from fraud_engine.utils.logging import (
    _describe,
    bind_request_id,
    get_request_id,
    log_call,
    log_dataframe,
    reset_request_id,
)

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
