"""Unit tests for `fraud_engine.data.splits`.

Each test builds a small synthetic frame with `TransactionID`,
`TransactionDT`, and `isFraud` columns so assertions are independent
of real data. Use `mock_settings` for boundary overrides via
Settings.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_engine.config.settings import Settings
from fraud_engine.data.splits import (
    SplitFrames,
    temporal_split,
    validate_no_overlap,
    write_split_manifest,
)


def _make_frame(
    *,
    n_rows: int = 100,
    dt_start: int = 0,
    dt_end: int = 1_000_000,
    fraud_rate: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic merged frame with the three required columns."""
    rng = np.random.default_rng(seed)
    dts = np.linspace(dt_start, dt_end, n_rows, dtype=np.int64)
    labels = (rng.uniform(size=n_rows) < fraud_rate).astype(np.int64)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n_rows, dtype=np.int64),
            "TransactionDT": dts,
            "isFraud": labels,
        }
    )


class TestTemporalSplit:
    """Contract tests for `temporal_split`."""

    def test_split_partitions_cleanly(self, mock_settings: Settings) -> None:
        df = _make_frame(n_rows=100, dt_start=0, dt_end=90)
        splits = temporal_split(
            df,
            train_end_dt=30,
            val_end_dt=60,
            settings=mock_settings,
        )
        assert len(splits.train) + len(splits.val) + len(splits.test) == 100
        assert splits.train["TransactionDT"].max() < 30
        assert splits.val["TransactionDT"].min() >= 30
        assert splits.val["TransactionDT"].max() < 60
        assert splits.test["TransactionDT"].min() >= 60

    def test_split_uses_settings_defaults(self, mock_settings: Settings) -> None:
        # Build a frame whose TransactionDT values straddle the default
        # 121-day and 151-day boundaries.
        dts = np.array(
            [
                86400 * 10,  # train
                86400 * 100,  # train
                86400 * 130,  # val (>=121, <151)
                86400 * 160,  # test
            ],
            dtype=np.int64,
        )
        df = pd.DataFrame(
            {
                "TransactionID": np.arange(len(dts)),
                "TransactionDT": dts,
                "isFraud": [0, 1, 0, 1],
            }
        )
        splits = temporal_split(df, settings=mock_settings)
        assert len(splits.train) == 2
        assert len(splits.val) == 1
        assert len(splits.test) == 1

    def test_split_empty_val_raises(self, mock_settings: Settings) -> None:
        df = _make_frame(n_rows=50, dt_start=0, dt_end=100)
        with pytest.raises(ValueError, match="val_end_dt=30.*greater than train_end_dt=30"):
            temporal_split(
                df,
                train_end_dt=30,
                val_end_dt=30,
                settings=mock_settings,
            )

    def test_split_empty_test_raises(self, mock_settings: Settings) -> None:
        df = _make_frame(n_rows=50, dt_start=0, dt_end=100)
        with pytest.raises(ValueError, match="empty 'test'"):
            temporal_split(
                df,
                train_end_dt=50,
                val_end_dt=200,
                settings=mock_settings,
            )

    def test_split_empty_train_raises(self, mock_settings: Settings) -> None:
        df = _make_frame(n_rows=50, dt_start=1000, dt_end=2000)
        with pytest.raises(ValueError, match="empty 'train'"):
            temporal_split(
                df,
                train_end_dt=500,
                val_end_dt=1500,
                settings=mock_settings,
            )

    def test_split_missing_column_raises(self, mock_settings: Settings) -> None:
        df = pd.DataFrame({"TransactionID": [1, 2], "TransactionDT": [10, 20]})
        with pytest.raises(KeyError, match="isFraud"):
            temporal_split(df, settings=mock_settings)

    def test_split_kwargs_override_settings(self, mock_settings: Settings) -> None:
        df = _make_frame(n_rows=60, dt_start=0, dt_end=60)
        splits = temporal_split(
            df,
            train_end_dt=10,
            val_end_dt=20,
            settings=mock_settings,
        )
        # With these overrides, 10/10/40 is the split.
        assert len(splits.train) == 10
        assert len(splits.val) == 10
        assert len(splits.test) == 40

    def test_manifest_has_expected_fields(self, mock_settings: Settings) -> None:
        df = _make_frame(n_rows=100, dt_start=0, dt_end=100)
        splits = temporal_split(df, train_end_dt=30, val_end_dt=60, settings=mock_settings)
        manifest = splits.manifest
        for key in (
            "schema_version",
            "transaction_dt_anchor_iso",
            "train_end_dt",
            "val_end_dt",
            "seed",
            "n_original",
            "n_train",
            "n_val",
            "n_test",
            "fraud_rate_overall",
            "fraud_rate_train",
            "fraud_rate_val",
            "fraud_rate_test",
            "min_transaction_dt",
            "max_transaction_dt",
        ):
            assert key in manifest, f"missing manifest key {key!r}"
        assert manifest["schema_version"] == 1
        assert manifest["train_end_dt"] == 30
        assert manifest["val_end_dt"] == 60


class TestSplitFrames:
    """Contract tests for the `SplitFrames` dataclass itself."""

    def test_frozen(self) -> None:
        frame = pd.DataFrame({"a": [1]})
        splits = SplitFrames(train=frame, val=frame, test=frame, manifest={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            splits.train = frame  # type: ignore[misc]


class TestValidateNoOverlap:
    """Contract tests for `validate_no_overlap`."""

    def test_accepts_clean_split(self, mock_settings: Settings) -> None:
        df = _make_frame(n_rows=90, dt_start=0, dt_end=90)
        splits = temporal_split(df, train_end_dt=30, val_end_dt=60, settings=mock_settings)
        validate_no_overlap(splits)  # must not raise

    def test_rejects_transaction_id_overlap(self) -> None:
        # Hand-build a bad SplitFrames to exercise the disjointness
        # check — bypassing the splitter is the easiest way to
        # simulate a bug upstream.
        frame_a = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3],
                "TransactionDT": [10, 20, 30],
                "isFraud": [0, 0, 0],
            }
        )
        frame_b = pd.DataFrame(
            {
                "TransactionID": [3, 4],  # 3 collides with frame_a
                "TransactionDT": [40, 50],
                "isFraud": [0, 0],
            }
        )
        frame_c = pd.DataFrame(
            {
                "TransactionID": [5, 6],
                "TransactionDT": [60, 70],
                "isFraud": [0, 0],
            }
        )
        manifest = {"n_train": 3, "n_val": 2, "n_test": 2}
        splits = SplitFrames(train=frame_a, val=frame_b, test=frame_c, manifest=manifest)
        with pytest.raises(ValueError, match="TransactionID overlap between train and val"):
            validate_no_overlap(splits)

    def test_rejects_temporal_overlap(self) -> None:
        frame_a = pd.DataFrame(
            {
                "TransactionID": [1, 2],
                "TransactionDT": [10, 40],  # max=40
                "isFraud": [0, 0],
            }
        )
        frame_b = pd.DataFrame(
            {
                "TransactionID": [3, 4],
                "TransactionDT": [35, 50],  # min=35 < 40
                "isFraud": [0, 0],
            }
        )
        frame_c = pd.DataFrame(
            {
                "TransactionID": [5],
                "TransactionDT": [60],
                "isFraud": [0],
            }
        )
        manifest = {"n_train": 2, "n_val": 2, "n_test": 1}
        splits = SplitFrames(train=frame_a, val=frame_b, test=frame_c, manifest=manifest)
        with pytest.raises(ValueError, match="Temporal overlap.*train"):
            validate_no_overlap(splits)

    def test_rejects_val_test_temporal_overlap(self) -> None:
        """Hand-built SplitFrames with val.max >= test.min hits the second
        contiguity branch (between val and test). Existing
        `test_rejects_temporal_overlap` covers the train↔val branch only.
        """
        frame_a = pd.DataFrame(
            {
                "TransactionID": [1, 2],
                "TransactionDT": [10, 20],
                "isFraud": [0, 0],
            }
        )
        frame_b = pd.DataFrame(
            {
                "TransactionID": [3, 4],
                "TransactionDT": [30, 60],  # max=60
                "isFraud": [0, 0],
            }
        )
        frame_c = pd.DataFrame(
            {
                "TransactionID": [5, 6],
                "TransactionDT": [55, 70],  # min=55 < 60 — overlaps val
                "isFraud": [0, 0],
            }
        )
        manifest = {"n_train": 2, "n_val": 2, "n_test": 2}
        splits = SplitFrames(train=frame_a, val=frame_b, test=frame_c, manifest=manifest)
        with pytest.raises(ValueError, match="Temporal overlap.*val"):
            validate_no_overlap(splits)

    def test_rejects_split_size_mismatch(self) -> None:
        """Hand-built SplitFrames with `len(...)` totals not matching the
        manifest's `n_train + n_val + n_test` hits the size-mismatch
        branch. Defensive against a caller bypassing `temporal_split` and
        constructing a malformed `SplitFrames` directly.
        """
        frame_a = pd.DataFrame(
            {
                "TransactionID": [1, 2],
                "TransactionDT": [10, 20],
                "isFraud": [0, 0],
            }
        )
        frame_b = pd.DataFrame(
            {
                "TransactionID": [3],
                "TransactionDT": [30],
                "isFraud": [0],
            }
        )
        frame_c = pd.DataFrame(
            {
                "TransactionID": [4],
                "TransactionDT": [40],
                "isFraud": [0],
            }
        )
        # Manifest claims 5 total rows; the frames have only 4. The
        # ID-set sum (4) does not equal the manifest sum (5), so the
        # mismatch branch must raise.
        manifest = {"n_train": 2, "n_val": 2, "n_test": 1}
        splits = SplitFrames(train=frame_a, val=frame_b, test=frame_c, manifest=manifest)
        with pytest.raises(ValueError, match="Split size mismatch"):
            validate_no_overlap(splits)


class TestWriteSplitManifest:
    """Contract tests for `write_split_manifest`."""

    def test_round_trip(self, mock_settings: Settings, tmp_path: Path) -> None:
        df = _make_frame(n_rows=90, dt_start=0, dt_end=90)
        splits = temporal_split(df, train_end_dt=30, val_end_dt=60, settings=mock_settings)
        out = tmp_path / "sub" / "splits_manifest.json"
        written = write_split_manifest(splits, out)
        assert written == out
        assert out.is_file()

        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == splits.manifest

    def test_creates_parent_dir(self, mock_settings: Settings, tmp_path: Path) -> None:
        df = _make_frame(n_rows=60, dt_start=0, dt_end=60)
        splits = temporal_split(df, train_end_dt=20, val_end_dt=40, settings=mock_settings)
        deep = tmp_path / "a" / "b" / "c" / "manifest.json"
        write_split_manifest(splits, deep)
        assert deep.is_file()
