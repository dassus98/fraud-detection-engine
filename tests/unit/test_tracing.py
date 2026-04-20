"""Unit tests for `fraud_engine.utils.tracing.run_context` / `attach_artifact`.

All tests use `mock_settings(tmp_path)` so `settings.logs_dir` lands
inside the per-test tmp directory — nothing in `logs/runs/` on the
real repo.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from fraud_engine.config.settings import Settings
from fraud_engine.utils.tracing import attach_artifact, run_context


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
