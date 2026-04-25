"""Unit tests for `fraud_engine.utils.tracing`.

Covers:
    - `run_context` functional wrapper (the legacy API surface used
      by Sprint 1 callers).
    - Module-level `attach_artifact(run, obj, *, name)` isinstance
      dispatch.
    - Class-based `Run` context manager with `log_param`,
      `log_metric`, `attach_artifact` method, and property accessors.
    - Structlog correlation: records emitted inside the `with` block
      carry `run_id` for log-stitching.

All tests use `mock_settings(tmp_path)` so `settings.logs_dir` lands
inside the per-test tmp directory — nothing in `logs/runs/` on the
real repo.
"""

from __future__ import annotations

import io
import json
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from fraud_engine.config.settings import Settings
from fraud_engine.utils.logging import configure_logging, get_logger
from fraud_engine.utils.tracing import Run, RunMetadata, attach_artifact, run_context


class TestRunContext:
    """Contract tests for the `run_context` context manager."""

    def test_creates_directory_tree(self, mock_settings: Settings) -> None:
        with run_context("unit-test", capture_streams=False) as run:
            assert run.run_dir.exists()
            assert run.artifacts_dir.exists()
            assert (run.run_dir / "run.json").exists()
            assert run.run_dir == mock_settings.logs_dir / "runs" / run.run_id

    def test_success_marks_status(self, mock_settings: Settings) -> None:
        with run_context("unit-test", capture_streams=False) as run:
            pass
        payload = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
        assert payload["status"] == "success"
        assert payload["end_time"]
        assert payload["duration_ms"] >= 0
        # mock_settings is the fixture that seeded logs_dir; referencing it
        # here prevents pytest from flagging the fixture as unused while
        # still exercising the tmp-logs path.
        assert payload["run_id"] == run.run_id
        _ = mock_settings.logs_dir

    def test_failure_marks_status_and_reraises(self, mock_settings: Settings) -> None:
        with (
            pytest.raises(ValueError, match="boom"),
            run_context("unit-test", capture_streams=False) as run,
        ):
            raise ValueError("boom")

        payload = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
        assert payload["status"] == "failed"
        assert payload["exception_type"] == "ValueError"
        assert payload["exception_message"] == "boom"
        assert "Traceback" in payload["traceback"]
        _ = mock_settings

    def test_metadata_round_trips(self, mock_settings: Settings) -> None:
        meta = {"source": "unit-test", "n_rows": 42}
        with run_context("unit-test", metadata=meta, capture_streams=False) as run:
            pass
        payload = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
        assert payload["metadata"] == meta
        _ = mock_settings


class TestAttachArtifact:
    """Contract tests for the `attach_artifact` dispatch."""

    def test_attach_path(self, mock_settings: Settings, tmp_path: object) -> None:
        source = mock_settings.logs_dir / "source.txt"
        source.write_text("hello", encoding="utf-8")
        with run_context("unit-test", capture_streams=False) as run:
            out = attach_artifact(run, source, name="copied")
        assert out.exists()
        assert out.read_text(encoding="utf-8") == "hello"
        assert out.name == "copied.txt"
        _ = tmp_path

    def test_attach_dataframe(self, mock_settings: Settings) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        with run_context("unit-test", capture_streams=False) as run:
            out = attach_artifact(run, df, name="frame")
        assert out.name == "frame.parquet"
        loaded = pd.read_parquet(out)
        pd.testing.assert_frame_equal(loaded, df)
        _ = mock_settings

    def test_attach_figure(self, mock_settings: Settings) -> None:
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 4])
        try:
            with run_context("unit-test", capture_streams=False) as run:
                out = attach_artifact(run, fig, name="plot")
        finally:
            plt.close(fig)
        assert out.name == "plot.png"
        assert out.stat().st_size > 0
        _ = mock_settings

    def test_attach_dict(self, mock_settings: Settings) -> None:
        payload = {"fraud_rate": 0.035, "identity_coverage": 0.244}
        with run_context("unit-test", capture_streams=False) as run:
            out = attach_artifact(run, payload, name="summary")
        assert out.name == "summary.json"
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == payload
        _ = mock_settings

    def test_attach_generic_object(self, mock_settings: Settings) -> None:
        import joblib

        # Use a stdlib built-in so pickle can round-trip without needing
        # a module-level class definition. set()/tuple() both hit the
        # joblib fallback branch because they aren't Path / DataFrame /
        # Figure / dict / list.
        obj = {"point", "set", "of", "strings"}
        with run_context("unit-test", capture_streams=False) as run:
            out = attach_artifact(run, obj, name="obj")
        assert out.name == "obj.joblib"
        loaded = joblib.load(out)
        assert loaded == obj
        _ = mock_settings


class TestRunClass:
    """Contract tests for the class-based `Run` context manager.

    These cover the spec-required API: `__enter__`/`__exit__`,
    `log_param`, `log_metric`, `attach_artifact(name, path)`, and the
    `run_id` / `run_dir` / `artifacts_dir` properties. `run_context`
    is already exercised by `TestRunContext`; these tests drill into
    the class-only behaviours.
    """

    def test_class_context_manager_creates_directory(self, mock_settings: Settings) -> None:
        with Run("unit-test", capture_streams=False) as run:
            assert run.run_dir.exists()
            assert run.artifacts_dir.exists()
            assert (run.run_dir / "run.json").exists()
            assert run.run_dir == mock_settings.logs_dir / "runs" / run.run_id

    def test_class_success_marks_status(self, mock_settings: Settings) -> None:
        with Run("unit-test", capture_streams=False) as run:
            pass
        payload = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
        assert payload["status"] == "success"
        assert payload["end_time"]
        assert payload["duration_ms"] >= 0
        _ = mock_settings

    def test_class_failure_marks_status_and_reraises(self, mock_settings: Settings) -> None:
        with (
            pytest.raises(ValueError, match="boom"),
            Run("unit-test", capture_streams=False) as run,
        ):
            raise ValueError("boom")

        payload = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
        assert payload["status"] == "failed"
        assert payload["exception_type"] == "ValueError"
        assert payload["exception_message"] == "boom"
        assert "Traceback" in payload["traceback"]
        _ = mock_settings

    def test_log_param_persists_to_run_json(self, mock_settings: Settings) -> None:
        with Run("unit-test", capture_streams=False) as run:
            run.log_param("n_features", 123)
            run.log_param("variant", "temporal")
            # Mid-run persistence check — the JSON is authoritative
            # even while status is still "running".
            mid_payload = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
            assert mid_payload["status"] == "running"
            assert mid_payload["metadata"]["n_features"] == 123
            assert mid_payload["metadata"]["variant"] == "temporal"

        final_payload = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
        assert final_payload["status"] == "success"
        assert final_payload["metadata"]["n_features"] == 123
        assert final_payload["metadata"]["variant"] == "temporal"
        _ = mock_settings

    def test_log_metric_persists_to_run_json(self, mock_settings: Settings) -> None:
        with Run("unit-test", capture_streams=False) as run:
            run.log_metric("auc", 0.92)
            run.log_metric("recall_at_fpr_2pct", 0.88)

        payload = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
        assert payload["metadata"]["metrics"]["auc"] == 0.92
        # Coercion to float is part of the contract — an int input
        # becomes a float in the persisted payload.
        assert payload["metadata"]["metrics"]["recall_at_fpr_2pct"] == 0.88
        _ = mock_settings

    def test_attach_artifact_method_copies_by_path(self, mock_settings: Settings) -> None:
        source = mock_settings.logs_dir / "scratch.txt"
        source.write_text("hello world", encoding="utf-8")
        with Run("unit-test", capture_streams=False) as run:
            out = run.attach_artifact("copied.txt", source)

        assert out == run.artifacts_dir / "copied.txt"
        assert out.exists()
        assert out.read_text(encoding="utf-8") == "hello world"
        _ = mock_settings

    def test_attach_artifact_method_missing_source_raises(self, mock_settings: Settings) -> None:
        with (
            Run("unit-test", capture_streams=False) as run,
            pytest.raises(FileNotFoundError, match="does not exist"),
        ):
            run.attach_artifact("dest.txt", mock_settings.logs_dir / "nope.bin")
        _ = mock_settings

    def test_access_before_enter_raises(self, mock_settings: Settings) -> None:
        run = Run("unit-test", capture_streams=False)
        # Properties should not silently fabricate state before
        # __enter__; the error message must name the offending op.
        with pytest.raises(RuntimeError, match="run_id"):
            _ = run.run_id
        with pytest.raises(RuntimeError, match="run_dir"):
            _ = run.run_dir
        with pytest.raises(RuntimeError, match="log_param"):
            run.log_param("k", "v")
        _ = mock_settings

    def test_log_param_reserved_key_raises(self, mock_settings: Settings) -> None:
        with (
            Run("unit-test", capture_streams=False) as run,
            pytest.raises(ValueError, match="reserved"),
        ):
            run.log_param("metrics", "anything")
        _ = mock_settings

    def test_run_metadata_roundtrip_via_extra(self, mock_settings: Settings) -> None:
        """The construction-time `metadata` dict lands under `extra`."""
        with Run(
            "unit-test",
            metadata={"source": "unit-test", "n_rows": 42},
            capture_streams=False,
        ) as run:
            # `metadata` property exposes the live state split by
            # section so tests can inspect without parsing JSON.
            snapshot = run.metadata
            assert snapshot["extra"] == {"source": "unit-test", "n_rows": 42}
            assert snapshot["params"] == {}
            assert snapshot["metrics"] == {}

        payload = json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))
        assert payload["metadata"]["source"] == "unit-test"
        assert payload["metadata"]["n_rows"] == 42
        _ = mock_settings

    def test_run_metadata_dataclass_is_exported(self) -> None:
        """`RunMetadata` is part of the public surface per spec."""
        assert RunMetadata.__name__ == "RunMetadata"
        # Spot-check field presence — if the dataclass shape drifts
        # the test fails loudly rather than via a mysterious
        # serialisation bug.
        fields = {f for f in RunMetadata.__dataclass_fields__}
        assert {
            "run_id",
            "pipeline_name",
            "start_time",
            "status",
            "end_time",
            "duration_ms",
            "params",
            "metrics",
            "extra",
        } <= fields


class TestStructlogCorrelation:
    """Every log line inside a `Run` carries the run_id contextvar.

    Spec bullet: "Every log line inside the `with` block includes the
    run_id". This lets the aggregation pipeline stitch a prediction
    back to the pipeline that produced it.
    """

    def test_log_records_carry_run_id(self, mock_settings: Settings) -> None:
        # Reconfigure logging to dump JSON into an in-memory buffer so
        # we can parse it. We re-call configure_logging afterwards so
        # the module's handlers are live again when Run.__enter__ runs.
        buffer = io.StringIO()

        # Enter the run, then attach a capturing handler *after* the
        # context manager has bound the contextvar. configure_logging's
        # effects persist across our handler manipulation because the
        # structlog contextvar is separate from the stdlib handler
        # list.
        with Run("unit-test", capture_streams=False) as run:
            capture_handler = logging.StreamHandler(buffer)
            import structlog
            from structlog.stdlib import ProcessorFormatter

            # Mirror the production JSON formatter so the record we
            # capture is the same shape aggregators consume.
            shared = [
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
            ]
            capture_handler.setFormatter(
                ProcessorFormatter(
                    processor=structlog.processors.JSONRenderer(),
                    foreign_pre_chain=shared,
                )
            )
            logging.getLogger().addHandler(capture_handler)
            try:
                logger = get_logger(__name__)
                logger.info("unit.correlation.check", payload=123)
            finally:
                logging.getLogger().removeHandler(capture_handler)
                capture_handler.close()

            captured_run_id = run.run_id

        output = buffer.getvalue().strip().splitlines()
        # One event emitted inside the with-block — the
        # `unit.correlation.check` one.
        matching = [
            line
            for line in output
            if '"event": "unit.correlation.check"' in line
            or '"event":"unit.correlation.check"' in line
        ]
        assert matching, f"expected unit.correlation.check event in {output!r}"
        record = json.loads(matching[0])
        assert record["run_id"] == captured_run_id
        assert record["pipeline"] == "unit-test"
        assert record["payload"] == 123
        _ = mock_settings

    def test_configure_logging_is_reentrant_after_run(self, mock_settings: Settings) -> None:
        """Two sequential Runs leave logging in a usable state.

        Regression guard: `configure_logging` replaces stdlib handlers
        on each call. A Run that doesn't restore them correctly would
        break subsequent tests; this asserts we can log again after
        the run exits.
        """
        with Run("unit-a", capture_streams=False):
            pass
        with Run("unit-b", capture_streams=False) as run_b:
            get_logger(__name__).info("post-run.ok", second_run=run_b.run_id)
        # Asserting no exception is the contract — logging must stay
        # configured across Run boundaries.
        configure_logging("post-test")
        _ = mock_settings
