"""Unit tests for `fraud_engine.data.lineage`.

Three public surfaces are exercised:

- `_fingerprint_dataframe`: deterministic 16-character schema hash.
- `LineageLog`: append-only JSONL writer + round-trip reader.
- `lineage_step`: `DataFrame → DataFrame` decorator that emits one
  step record per call.

All tests run against a `tmp_path`-rooted `Settings` (via the shared
`mock_settings` fixture) so writes never touch the real `logs/`
directory. An autouse fixture clears the structlog `run_id`
contextvar before and after each test so a binding in one test never
leaks into the next.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pandas as pd
import pytest
from structlog.contextvars import bind_contextvars, clear_contextvars, get_contextvars

from fraud_engine.config.settings import Settings
from fraud_engine.data.lineage import (
    LineageLog,
    LineageStep,
    _fingerprint_dataframe,
    lineage_step,
)

# sha256("{}").hexdigest()[:16]. Locked in by the empty-schema test
# below; a future refactor that altered the canonical-JSON form
# (e.g. dropped `sort_keys` or changed separators) would break the
# reproducibility of every previously recorded `LineageStep`. Pinning
# the literal here makes that breakage loud.
_EMPTY_SCHEMA_HASH = "44136fa355b3678a"  # pragma: allowlist secret

# UUID4 hex is exactly 32 lowercase hex characters. Pinned so the
# unbound-run_id test does not silently pass against e.g. a UUID5
# replacement that differs in length.
_UUID4_HEX_LEN = 32

# Every `LineageStep` JSONL record carries exactly these keys. Pinned
# here so a renamed / dropped / added field surfaces as a single
# focused failure rather than a death-of-a-thousand-cuts spread
# across every other decorator test.
_STEP_KEYS = frozenset(
    {
        "run_id",
        "step_name",
        "input_schema_hash",
        "output_schema_hash",
        "input_rows",
        "output_rows",
        "duration_ms",
        "timestamp",
    }
)


@pytest.fixture(autouse=True)
def _clear_run_id_contextvar() -> Iterator[None]:
    """Ensure no bound `run_id` leaks between tests.

    structlog contextvars are process-wide, so a test that binds
    `run_id="abc"` and forgets to unbind would pollute its neighbours.
    Clearing both before and after is belt-and-braces against tests
    that raise mid-binding.
    """
    clear_contextvars()
    yield
    clear_contextvars()


def _identity(df: pd.DataFrame) -> pd.DataFrame:
    """Trivial DataFrame → DataFrame fn shared by decorator tests."""
    return df


# ---------------- Fingerprint helper ---------------- #


def test_fingerprint_stable_across_calls() -> None:
    """Identical input → identical fingerprint, repeatable."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert _fingerprint_dataframe(df) == _fingerprint_dataframe(df)


def test_fingerprint_changes_on_dtype_drift() -> None:
    """Dtype change → different hash; row mutation alone → same hash."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    original = _fingerprint_dataframe(df)
    # Dtype drift: int64 → int32 must shift the hash.
    drifted = _fingerprint_dataframe(df.astype({"a": "int32"}))
    assert drifted != original
    # Row-only mutation must NOT shift the hash.
    df_mutated = df.copy()
    df_mutated.loc[0, "a"] = 99
    assert _fingerprint_dataframe(df_mutated) == original


def test_fingerprint_column_order_independent() -> None:
    """Column reorder does not affect the hash (sorted internally)."""
    df_ab = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df_ba = df_ab[["b", "a"]]
    assert _fingerprint_dataframe(df_ab) == _fingerprint_dataframe(df_ba)


def test_fingerprint_empty_dataframe_is_deterministic() -> None:
    """Empty (no columns) DataFrame hashes to sha256({})[:16]."""
    empty = pd.DataFrame()
    assert _fingerprint_dataframe(empty) == _EMPTY_SCHEMA_HASH
    assert _fingerprint_dataframe(empty) == _fingerprint_dataframe(empty)


# ---------------- Decorator ---------------- #


def test_lineage_step_writes_jsonl_record(mock_settings: Settings) -> None:
    """One call → one JSONL line with all 8 fields correctly typed."""
    bind_contextvars(run_id="testrun5")
    df = pd.DataFrame({"a": [1, 2, 3]})
    decorated = lineage_step("foo")(_identity)
    decorated(df)

    log_path = mock_settings.logs_dir / "lineage" / "testrun5" / "lineage.jsonl"
    assert log_path.is_file()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert frozenset(payload.keys()) == _STEP_KEYS
    assert payload["run_id"] == "testrun5"
    assert payload["step_name"] == "foo"
    assert payload["input_rows"] == 3
    assert payload["output_rows"] == 3
    assert isinstance(payload["duration_ms"], float)
    assert isinstance(payload["timestamp"], str)


def test_lineage_step_records_run_id_from_contextvars(
    mock_settings: Settings,
) -> None:
    """Pre-bound run_id appears verbatim on the recorded step."""
    bind_contextvars(run_id="abc123")
    decorated = lineage_step("bar")(_identity)
    decorated(pd.DataFrame({"a": [1]}))
    steps = LineageLog("abc123", settings=mock_settings).read()
    assert len(steps) == 1
    assert steps[0].run_id == "abc123"


def test_lineage_step_generates_run_id_if_unbound(
    mock_settings: Settings,
) -> None:
    """No bound run_id → decorator generates and binds one; reuse on retry."""
    df = pd.DataFrame({"a": [1]})
    decorated = lineage_step("bar")(_identity)
    decorated(df)
    bound = get_contextvars().get("run_id")
    assert isinstance(bound, str)
    assert len(bound) == _UUID4_HEX_LEN
    log = LineageLog(bound, settings=mock_settings)
    assert len(log.read()) == 1
    # A second call in the same context reuses the same id and
    # appends a second record.
    decorated(df)
    assert get_contextvars().get("run_id") == bound
    assert len(log.read()) == 2


def test_lineage_step_records_row_counts_and_duration(
    mock_settings: Settings,
) -> None:
    """input/output row counts reflect the wrapped fn; duration is non-negative."""
    bind_contextvars(run_id="dur1")
    df = pd.DataFrame({"a": list(range(10))})

    def filter_half(d: pd.DataFrame) -> pd.DataFrame:
        return d[d["a"] >= 5].copy()

    decorated = lineage_step("filter_half")(filter_half)
    decorated(df)
    steps = LineageLog("dur1", settings=mock_settings).read()
    assert len(steps) == 1
    assert steps[0].input_rows == 10
    assert steps[0].output_rows == 5
    assert isinstance(steps[0].duration_ms, float)
    # `time.perf_counter()` may round to 0 on extremely fast platforms;
    # non-negativity is the meaningful invariant.
    assert steps[0].duration_ms >= 0.0


def test_lineage_step_locates_dataframe_when_self_first(
    mock_settings: Settings,
) -> None:
    """A non-DataFrame leading positional (e.g. `self`) is skipped."""
    bind_contextvars(run_id="bound1")

    class Holder:
        """Stand-in for an instance-method receiver."""

        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

    holder = Holder()
    # Decorate the unbound function so the wrapper sees args=(self, df).
    decorated = lineage_step("bound_method")(Holder.transform)
    decorated(holder, pd.DataFrame({"a": [1, 2]}))
    steps = LineageLog("bound1", settings=mock_settings).read()
    assert len(steps) == 1
    assert steps[0].input_rows == 2


def test_lineage_step_reraises_on_exception_no_record_written(
    mock_settings: Settings,
) -> None:
    """Wrapped fn raises → exception propagates; lineage file is absent."""
    bind_contextvars(run_id="boom")

    def bad(df: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("kaboom")

    decorated = lineage_step("bad")(bad)
    with pytest.raises(ValueError, match="kaboom"):
        decorated(pd.DataFrame({"a": [1]}))
    log_path = mock_settings.logs_dir / "lineage" / "boom" / "lineage.jsonl"
    assert not log_path.exists()


def test_lineage_step_emits_structlog_event(
    mock_settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The decorator emits one `lineage.step` event with key fields.

    Mirrors the cleaner test's pattern: structured events arrive on
    `record.msg` as a dict via the project's structlog→stdlib bridge.
    """
    bind_contextvars(run_id="ev1")
    decorated = lineage_step("emit")(_identity)
    df = pd.DataFrame({"a": [1, 2, 3]})
    with caplog.at_level(logging.INFO):
        decorated(df)
    events = [
        r.msg
        for r in caplog.records
        if isinstance(r.msg, dict) and r.msg.get("event") == "lineage.step"
    ]
    assert len(events) == 1
    event = events[0]
    assert event["step_name"] == "emit"
    assert event["input_rows"] == 3
    assert event["output_rows"] == 3


def test_lineage_step_rejects_non_dataframe_return(
    mock_settings: Settings,
) -> None:
    """Wrapped fn returning a non-DataFrame raises TypeError; no record written."""
    bind_contextvars(run_id="bad-ret")

    def returns_int(df: pd.DataFrame) -> int:
        return 7

    # Intentional mistype: lineage_step's signature requires a
    # DataFrame return, and we are exercising the runtime guard.
    decorated = lineage_step("bad")(returns_int)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="pd.DataFrame"):
        decorated(pd.DataFrame({"a": [1]}))
    log_path = mock_settings.logs_dir / "lineage" / "bad-ret" / "lineage.jsonl"
    assert not log_path.exists()


# ---------------- LineageLog ---------------- #


def test_lineage_log_read_round_trips_steps(mock_settings: Settings) -> None:
    """Append three records → `read()` returns equal dataclasses in order."""
    log = LineageLog("rt", settings=mock_settings)
    steps = [
        LineageStep(
            run_id="rt",
            step_name=f"step_{i}",
            input_schema_hash="aaaaaaaaaaaaaaaa",
            output_schema_hash="bbbbbbbbbbbbbbbb",
            input_rows=i * 10,
            output_rows=i * 10 - 1,
            duration_ms=float(i) * 1.5,
            timestamp=f"2026-01-0{i + 1}T00:00:00+00:00",
        )
        for i in range(3)
    ]
    for s in steps:
        log.append(s)
    assert log.read() == steps


def test_lineage_log_path_under_logs_dir(mock_settings: Settings) -> None:
    """Path is `logs_dir / lineage / run_id / lineage.jsonl`; constructor is I/O-free."""
    log = LineageLog("xyz", settings=mock_settings)
    expected = mock_settings.logs_dir / "lineage" / "xyz" / "lineage.jsonl"
    assert log.path == expected
    # No I/O at construction — the parent directory does not exist yet.
    assert not log.path.parent.exists()
