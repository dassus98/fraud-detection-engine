"""Materialise the interim data layer: load → clean → split → write.

This script is the canonical entry point that produces
``data/interim/{train,val,test}.parquet`` plus
``data/interim/splits_manifest.json``. It runs under a `Run` so the
artefacts and parameters are captured under
``logs/runs/{run_id}/``, and every transformation is wrapped with
``@lineage_step`` so a parallel JSONL trail under
``logs/lineage/{run_id}/lineage.jsonl`` records the full structural
journey from raw CSV to the three persisted partitions.

Business rationale:
    Sprint 2 onwards reads only the interim parquets — re-running the
    raw load + cleaner + split on every feature regen would cost
    minutes per iteration and risk silent divergence between
    notebooks and the canonical pipeline. Persisting once, with full
    lineage, means every later sprint scores against bit-identical
    inputs to the baseline. The lineage trail is the audit surface
    CLAUDE.md §7.2 mandates: any prediction in Sprint 5's API is
    traceable back through ``split_train``, ``interim_clean``, and
    ``load_merged`` to the raw CSV that produced it.

Trade-offs considered:
    - The lineage decorator is defined for ``DataFrame → DataFrame``
      callables, but the load step has no DataFrame input (it reads a
      CSV). Rather than promote a private helper or special-case the
      record, we wrap the loader's call in a thin function that
      accepts an empty seed DataFrame. The seed's schema fingerprint
      (``sha256("{}")[:16]``) is well-defined and the resulting
      record has the standard 8-field shape.
    - Each split slice gets its own lineage record (``split_train``,
      ``split_val``, ``split_test``) rather than one ``temporal_split``
      record. This costs three JSONL lines instead of one but lets
      ``verify_lineage.py`` cross-check every parquet's row count
      against a dedicated record without re-deriving the partition.
    - Run-level parameters (output dir, train/val cut points) and
      metrics (per-split row counts, per-split fraud rates) are
      captured via ``run.log_param`` / ``run.log_metric`` so the
      ``run.json`` payload is self-contained for post-hoc audit.
    - The script does not delete pre-existing parquets; pandas'
      ``to_parquet`` overwrites in place. A future iteration that
      keeps run-stamped copies under ``data/interim/runs/{run_id}/``
      would be additive to this contract.

Usage:
    uv run python scripts/build_interim.py
    uv run python scripts/build_interim.py --output-dir custom/path
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final

import click
import pandas as pd

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.data.cleaner import TransactionCleaner
from fraud_engine.data.lineage import lineage_step
from fraud_engine.data.loader import RawDataLoader
from fraud_engine.data.splits import (
    SplitFrames,
    temporal_split,
    validate_no_overlap,
    write_split_manifest,
)
from fraud_engine.utils.logging import get_logger
from fraud_engine.utils.tracing import Run

# Filenames are pinned constants — Sprint 2's feature pipeline and
# Sprint 4's evaluator both glob by these names. Renaming requires a
# coordinated update there.
_TRAIN_FILENAME: Final[str] = "train.parquet"
_VAL_FILENAME: Final[str] = "val.parquet"
_TEST_FILENAME: Final[str] = "test.parquet"
_MANIFEST_FILENAME: Final[str] = "splits_manifest.json"

# Step names for the JSONL lineage records. `verify_lineage.py` pins
# the same five strings independently — duplication is intentional so
# the verifier is a regression detector, not a derivative of this file.
_STEP_LOAD: Final[str] = "load_merged"
_STEP_CLEAN: Final[str] = "interim_clean"
_STEP_SPLIT_TRAIN: Final[str] = "split_train"
_STEP_SPLIT_VAL: Final[str] = "split_val"
_STEP_SPLIT_TEST: Final[str] = "split_test"


def _make_load_step(loader: RawDataLoader) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return a `DataFrame → DataFrame` adapter wrapping `loader.load_merged`.

    The lineage decorator scans positional args for a `pd.DataFrame`;
    `load_merged` has none of its own, so the adapter accepts an empty
    seed frame purely to satisfy the contract. The seed's empty
    schema hashes to ``sha256("{}")[:16]`` deterministically, which
    is the right fingerprint for "no input schema" — load is the
    boundary where data enters the pipeline.

    Args:
        loader: The configured `RawDataLoader` to call.

    Returns:
        A function that ignores its DataFrame argument and returns the
        merged train frame.
    """

    @lineage_step(_STEP_LOAD)
    def _load(_seed: pd.DataFrame) -> pd.DataFrame:
        # `optimize=False`: the loader's `_optimize` downcasts to
        # int32/float32/category, but `InterimTransactionSchema`
        # (which `cleaner.clean` validates against) inherits
        # `MergedSchema`'s strict int64/float64/object dtype contract.
        # The optimised frame trips that validation; the un-optimised
        # frame passes through cleanly and the parquet writer below
        # re-applies pyarrow's column compression at write time.
        return loader.load_merged(optimize=False)

    return _load


def _make_split_step(
    slice_name: str, slice_df: pd.DataFrame
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return a `DataFrame → DataFrame` passthrough that emits one slice.

    The wrapper takes the full pre-split frame as input (so
    `input_rows`/`input_schema_hash` reflect the source) and returns
    the captured slice (so `output_rows`/`output_schema_hash`
    reflect the partition). The closure pattern keeps each slice's
    lineage record self-contained without a public split-record API.

    Args:
        slice_name: Step name for the lineage record
            (one of `split_train` / `split_val` / `split_test`).
        slice_df: The post-split partition to record on the output
            side of the lineage step.

    Returns:
        A decorated function ready to call with the merged interim
        frame as input.
    """

    @lineage_step(slice_name)
    def _passthrough(_full: pd.DataFrame) -> pd.DataFrame:
        return slice_df

    return _passthrough


def _emit_split_records(splits: SplitFrames, interim: pd.DataFrame) -> None:
    """Append one lineage record per split slice via the passthrough trick.

    Args:
        splits: Output of `temporal_split` on `interim`.
        interim: The full pre-split frame; used as the lineage record's
            input shape so each split's `input_rows` / input fingerprint
            agrees across the three records.
    """
    for slice_name, slice_df in (
        (_STEP_SPLIT_TRAIN, splits.train),
        (_STEP_SPLIT_VAL, splits.val),
        (_STEP_SPLIT_TEST, splits.test),
    ):
        decorated = _make_split_step(slice_name, slice_df)
        decorated(interim)


def _write_parquets(splits: SplitFrames, target_dir: Path) -> tuple[Path, Path, Path]:
    """Persist the three split frames as parquet under `target_dir`.

    Args:
        splits: Output of `temporal_split`.
        target_dir: Directory to write into. Must already exist.

    Returns:
        A tuple of `(train_path, val_path, test_path)` for the
        caller to log / attach.
    """
    train_path = target_dir / _TRAIN_FILENAME
    val_path = target_dir / _VAL_FILENAME
    test_path = target_dir / _TEST_FILENAME
    splits.train.to_parquet(train_path)
    splits.val.to_parquet(val_path)
    splits.test.to_parquet(test_path)
    return train_path, val_path, test_path


def _build_interim(*, settings: Settings, target_dir: Path, run: Run) -> SplitFrames:
    """Run the full load → clean → split pipeline under `run`.

    Kept as a private helper so the Click entry point reads as a
    sequence of bookkeeping steps without the data plumbing inline.

    Args:
        settings: Active `Settings` (drives raw_dir, anchor, split
            cut points).
        target_dir: Where to write the parquet outputs and manifest.
        run: The active `Run` whose params/metrics/artifacts are
            mutated as the pipeline progresses.

    Returns:
        The `SplitFrames` produced by `temporal_split`. The caller
        uses its `manifest` to populate run metrics.
    """
    logger = get_logger(__name__)

    loader = RawDataLoader(settings=settings)
    load_step = _make_load_step(loader)
    # Empty DataFrame seed: see `_make_load_step` docstring. The seed's
    # schema fingerprint is the canonical sha256("{}") prefix.
    merged = load_step(pd.DataFrame())
    logger.info(
        "build_interim.loaded",
        rows=int(merged.shape[0]),
        cols=int(merged.shape[1]),
    )

    cleaner = TransactionCleaner(settings=settings)
    clean_step = lineage_step(_STEP_CLEAN)(cleaner.clean)
    interim = clean_step(merged)
    logger.info(
        "build_interim.cleaned",
        rows_in=int(merged.shape[0]),
        rows_out=int(interim.shape[0]),
    )

    splits = temporal_split(interim, settings=settings)
    validate_no_overlap(splits)
    _emit_split_records(splits, interim)
    logger.info(
        "build_interim.split",
        n_train=splits.manifest["n_train"],
        n_val=splits.manifest["n_val"],
        n_test=splits.manifest["n_test"],
    )

    train_path, val_path, test_path = _write_parquets(splits, target_dir)
    manifest_path = target_dir / _MANIFEST_FILENAME
    write_split_manifest(splits, manifest_path)
    logger.info(
        "build_interim.persisted",
        train=str(train_path),
        val=str(val_path),
        test=str(test_path),
        manifest=str(manifest_path),
    )

    run.log_metric("n_train", float(splits.manifest["n_train"]))
    run.log_metric("n_val", float(splits.manifest["n_val"]))
    run.log_metric("n_test", float(splits.manifest["n_test"]))
    run.log_metric("fraud_rate_train", float(splits.manifest["fraud_rate_train"]))
    run.log_metric("fraud_rate_val", float(splits.manifest["fraud_rate_val"]))
    run.log_metric("fraud_rate_test", float(splits.manifest["fraud_rate_test"]))
    run.attach_artifact(_MANIFEST_FILENAME, manifest_path)

    return splits


@click.command()
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Override interim output directory. Defaults to settings.interim_dir.",
)
def main(output_dir: str | None) -> None:
    """Build the interim layer: train/val/test parquets + splits manifest."""
    settings = get_settings()
    settings.ensure_directories()
    target_dir = Path(output_dir) if output_dir is not None else settings.interim_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    with Run(
        "build_interim",
        settings=settings,
        metadata={"output_dir": str(target_dir)},
    ) as run:
        run.log_param("output_dir", str(target_dir))
        run.log_param("train_end_dt", settings.train_end_dt)
        run.log_param("val_end_dt", settings.val_end_dt)
        run.log_param("transaction_dt_anchor_iso", settings.transaction_dt_anchor_iso)

        splits = _build_interim(settings=settings, target_dir=target_dir, run=run)

        click.echo(click.style("build_interim: GREEN", fg="green", bold=True))
        click.echo(f"  run_id: {run.run_id}")
        click.echo(f"  output: {target_dir}")
        click.echo(
            f"  rows:   train={splits.manifest['n_train']:,} "
            f"val={splits.manifest['n_val']:,} "
            f"test={splits.manifest['n_test']:,}"
        )


if __name__ == "__main__":
    main()
