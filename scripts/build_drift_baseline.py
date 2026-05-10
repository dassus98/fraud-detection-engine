"""Build the per-feature drift baseline parquet from a training-data slice.

Sprint 6 prompt 6.1.b: one-shot CLI that materialises
`data/baselines/distributions.parquet` for the runtime `DriftMonitor`
(see `src/fraud_engine/monitoring/drift.py`).

Operator workflow:
    1. Train the LightGBM model (Sprint 3.3.b) — this writes
       `models/sprint3/lightgbm_model_manifest.json` with the canonical
       743-feature input list.
    2. Run the feature pipeline through Tier-5 — this writes
       `data/processed/tier5_train.parquet`.
    3. Run THIS script. Result: `data/baselines/distributions.parquet`
       with one row per (feature, bin), ~7,430 rows × 6 cols ≈ 250 KB.
    4. Subsequent `DriftMonitor()` instantiations load this parquet
       once at construction.

The script is a thin wrapper around `DriftBaselineBuilder.build()` so
the underlying math + I/O is unit-testable in-process (tests pass
synthetic DataFrames directly to the builder; this script only adds
parquet I/O + Click argument parsing).

Business rationale:
    Bin edges + baseline percentages are deterministic functions of
    the training slice; recomputing them on every drift check would
    burn ~37 s per feature (np.quantile on ~400K rows × 743
    features). Persisting once after each model retrain reduces the
    runtime cost to a 5 ms parquet read.

Trade-offs considered:
    - **Click CLI thin wrapper around `DriftBaselineBuilder.build()`.**
      Same split as `warmup_redis.py` (Sprint 5.1.g): the math lives
      in the library, the script handles I/O + CLI. Tests call the
      builder directly; the script is only invoked from the operator's
      shell.
    - **Read feature_names from the model manifest, not a separate
      configs/ file.** The manifest is the single source of truth for
      what the model expects (`feature_service.py:250-270` already
      uses the same source). Drift baseline must align column-wise
      with the model's input; reading from the same file keeps them
      in sync by construction.
    - **One-shot, no incremental update.** Re-running the script
      overwrites the previous parquet entirely. Incremental updates
      (e.g., "extend the baseline window by 7 more days") are a
      Sprint 6.x concern; for now, every model retrain triggers a
      full rebuild.
    - **`temporal_split` integration.** The script accepts a
      `--train-parquet` of pre-split training data directly; it does
      NOT re-run `temporal_split` internally. Operators are expected
      to pass `data/processed/tier5_train.parquet` (the pre-split
      train slice from Sprint 2/3 pipelines), not the merged
      `data/processed/tier5_features.parquet`. This matches how
      Sprint 4's economic-evaluation script treats split data.

Cross-references:
    - `src/fraud_engine/monitoring/drift.py:DriftBaselineBuilder.build`
      — the underlying math.
    - `scripts/warmup_redis.py` — Click-CLI conventions this script
      mirrors.
    - `models/sprint3/lightgbm_model_manifest.json:feature_names` —
      the canonical 743-feature list.
    - `src/fraud_engine/data/splits.py:79-177` — `temporal_split` for
      the train-slice contract operators are expected to honour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import click
import pandas as pd

from fraud_engine.monitoring.drift import DriftBaselineBuilder
from fraud_engine.utils.logging import get_logger

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_DEFAULT_TRAIN_PARQUET: Final[Path] = _PROJECT_ROOT / "data" / "processed" / "tier5_train.parquet"
_DEFAULT_MANIFEST: Final[Path] = (
    _PROJECT_ROOT / "models" / "sprint3" / "lightgbm_model_manifest.json"
)
_DEFAULT_OUTPUT: Final[Path] = _PROJECT_ROOT / "data" / "baselines" / "distributions.parquet"
_DEFAULT_BINS: Final[int] = 10

_logger = get_logger(__name__)


def _load_feature_names(manifest_path: Path) -> list[str]:
    """Read the model manifest and return its `feature_names` array.

    Mirrors `warmup_redis._load_feature_names` (Sprint 5.1.g): single
    pattern across all scripts that need to align with the model's
    input list.

    Raises:
        FileNotFoundError: If the manifest is missing.
        ValueError: If `feature_names` is missing or not a list of
            strings.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"build_drift_baseline: model manifest not found at "
            f"{manifest_path} — train the LightGBM model first via "
            f"Sprint 3.3.b."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_names = manifest.get("feature_names")
    if not isinstance(feature_names, list) or not all(isinstance(n, str) for n in feature_names):
        raise ValueError(
            f"build_drift_baseline: manifest at {manifest_path} missing "
            f"or malformed `feature_names` (expected list[str])"
        )
    return feature_names


@click.command(help="Build per-feature drift baseline parquet from a training slice.")
@click.option(
    "--train-parquet",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_TRAIN_PARQUET,
    show_default=True,
    help=(
        "Pre-split training data parquet. Use the Sprint 2/3 "
        "feature-pipeline output (e.g. tier5_train.parquet); do NOT "
        "pass merged feature data — temporal_split contract requires "
        "no val/test leakage into the production-drift baseline."
    ),
)
@click.option(
    "--manifest",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_MANIFEST,
    show_default=True,
    help="LightGBM model manifest with `feature_names` array.",
)
@click.option(
    "--output",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_OUTPUT,
    show_default=True,
    help="Output parquet path. Parent directories are created on demand.",
)
@click.option(
    "--bins",
    type=int,
    default=_DEFAULT_BINS,
    show_default=True,
    help="Equal-frequency quantile-bin count. Industry standard: 10.",
)
def main(
    train_parquet: Path,
    manifest: Path,
    output: Path,
    bins: int,
) -> None:
    """CLI entrypoint — load → build → persist."""
    feature_names = _load_feature_names(manifest)
    _logger.info(
        "build_drift_baseline.start",
        train_parquet=str(train_parquet),
        manifest=str(manifest),
        output=str(output),
        n_features=len(feature_names),
        n_bins=bins,
    )

    train_df = pd.read_parquet(train_parquet)
    _logger.info(
        "build_drift_baseline.train_loaded",
        n_rows=len(train_df),
        n_cols=len(train_df.columns),
    )

    baseline_df = DriftBaselineBuilder.build(
        train_df=train_df,
        feature_names=feature_names,
        n_bins=bins,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    baseline_df.to_parquet(output, index=False)

    _logger.info(
        "build_drift_baseline.complete",
        output=str(output),
        n_rows=len(baseline_df),
        n_kept_features=baseline_df["feature_name"].nunique(),
    )


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover — Click entrypoint
