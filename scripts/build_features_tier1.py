"""Materialise the Tier-1 feature layer: load interim → fit → transform → write.

This script is the canonical entry point that produces
``data/processed/tier1_{train,val,test}.parquet`` plus
``models/pipelines/tier1_pipeline.joblib`` and
``models/pipelines/feature_manifest.json``. It runs under a `Run` so
the artefacts and parameters land under ``logs/runs/{run_id}/`` for
post-hoc audit, and concludes with a quick LightGBM retrain so the
report can quote a Tier-1 val AUC alongside Sprint 1's 0.9247
baseline.

Business rationale:
    Sprint 3's LightGBM tuning, Sprint 4's economic-cost evaluation,
    and Sprint 5's serving layer all read the processed parquets and
    reuse the saved pipeline. Persisting the fit-on-train pipeline
    once means every later sprint scores against bit-identical
    transformations; the alternative — re-fitting features in every
    consumer — would silently drift across runs.

Trade-offs considered:
    - **Fit on train only.** The pipeline's `fit_transform(train)`
      learns generator state (qcut bin edges, missing-indicator
      target columns, etc.) on training rows; `transform(val)` and
      `transform(test)` then apply that state. Mirrors Sprint 1's
      baseline temporal-split discipline — no leakage from val / test
      into fitted parameters.
    - **Schema validation post-transform.** Each split is validated
      against `TierOneFeaturesSchema` *before* parquet write. A drift
      in a generator's output dtype surfaces here loudly, instead of
      contaminating the parquet and breaking Sprint 3's training.
    - **Quick LightGBM retrain in the build script.** The report's
      headline is the Tier-1 val AUC (Sprint 1 baseline: 0.9247);
      computing it inline keeps the verification self-contained.
      Object / string columns (provider, tld) are dropped from the
      retrain — they're for Sprint 3's target encoder. Numeric +
      `category`-dtype columns survive.

Usage:
    uv run python scripts/build_features_tier1.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import click
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.features import FeaturePipeline
from fraud_engine.features.tier1_basic import (
    AmountTransformer,
    EmailDomainExtractor,
    MissingIndicatorGenerator,
    TimeFeatureGenerator,
)
from fraud_engine.schemas.features import TierOneFeaturesSchema
from fraud_engine.utils.logging import get_logger
from fraud_engine.utils.tracing import Run

# Filenames pinned at module scope. Sprint 3+ loaders glob by these
# names — renaming requires a coordinated update there.
_PIPELINE_FILENAME: Final[str] = "tier1_pipeline.joblib"
_PIPELINE_SUBDIR: Final[str] = "pipelines"
_PROCESSED_PREFIX: Final[str] = "tier1"

# Columns dropped from the LightGBM retrain feature set. Mirrors
# `baseline.py:_NON_FEATURE_COLUMNS` plus `timestamp` (the cleaner's
# tz-aware datetime column, derived from TransactionDT — pure
# leakage if fed to the model).
_NON_FEATURE_COLS: Final[frozenset[str]] = frozenset(
    {"TransactionID", "TransactionDT", "isFraud", "timestamp"}
)


def _load_interim_split(interim_dir: Path, name: str) -> pd.DataFrame:
    """Read one interim parquet (`train` / `val` / `test`)."""
    path = interim_dir / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected interim parquet at {path} — run "
            f"`uv run python scripts/build_interim.py` first."
        )
    return pd.read_parquet(path)


def _build_pipeline() -> FeaturePipeline:
    """Construct the Tier-1 `FeaturePipeline` with the four generators."""
    return FeaturePipeline(
        generators=[
            AmountTransformer(),
            TimeFeatureGenerator(),
            EmailDomainExtractor(),
            MissingIndicatorGenerator(),
        ]
    )


def _validate_split(name: str, df: pd.DataFrame) -> None:
    """Validate one split against `TierOneFeaturesSchema`."""
    TierOneFeaturesSchema.validate(df, lazy=True)
    get_logger(__name__).info(
        "build_features_tier1.validated",
        split=name,
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
    )


def _write_processed_parquets(
    splits: dict[str, pd.DataFrame], processed_dir: Path
) -> dict[str, Path]:
    """Write each split to ``processed_dir/tier1_{name}.parquet``.

    Returns a mapping `name → path` for caller logging.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, df in splits.items():
        path = processed_dir / f"{_PROCESSED_PREFIX}_{name}.parquet"
        df.to_parquet(path)
        paths[name] = path
    return paths


def _select_lgbm_features(df: pd.DataFrame) -> list[str]:
    """Return the subset of columns LightGBM's sklearn API can ingest.

    Drops the non-feature columns and any object / string-dtype
    columns (provider / tld go to Sprint 3's target encoder; the
    sklearn LightGBM API can't ingest object columns directly).
    Categorical-dtype columns (card4 / ProductCD / etc.) survive.
    """
    return [
        col
        for col in df.columns
        if col not in _NON_FEATURE_COLS
        and not pd.api.types.is_object_dtype(df[col])
        and not pd.api.types.is_string_dtype(df[col])
    ]


def _quick_lgbm_val_auc(train: pd.DataFrame, val: pd.DataFrame, settings: Settings) -> float:
    """Fit a LightGBM on train, score val, return ROC-AUC.

    Mirrors `baseline.train_baseline` shape but inlined here because
    we want the Tier-1 features (which baseline.py doesn't know about)
    in the feature set without re-implementing baseline.py's MLflow /
    artefact wiring. Sprint 3 will replace this with a proper tuned
    model; this is the smoke-test number for the completion report.
    """
    feature_cols = _select_lgbm_features(train)
    clf = LGBMClassifier(
        **settings.lgbm_defaults,
        random_state=settings.seed,
        verbose=-1,
    )
    clf.fit(
        train[feature_cols],
        train["isFraud"],
        categorical_feature="auto",
    )
    val_proba = clf.predict_proba(val[feature_cols])[:, 1]
    return float(roc_auc_score(val["isFraud"], val_proba))


@click.command()
def main() -> None:
    """Build the Tier-1 feature layer and report a quick val AUC."""
    settings = get_settings()
    settings.ensure_directories()
    logger = get_logger(__name__)

    with Run("build_features_tier1", settings=settings) as run:
        run.log_param("interim_dir", str(settings.interim_dir))
        run.log_param("processed_dir", str(settings.processed_dir))
        run.log_param("seed", settings.seed)

        # 1. Load interim splits.
        train = _load_interim_split(settings.interim_dir, "train")
        val = _load_interim_split(settings.interim_dir, "val")
        test = _load_interim_split(settings.interim_dir, "test")
        logger.info(
            "build_features_tier1.loaded",
            n_train=len(train),
            n_val=len(val),
            n_test=len(test),
        )

        # 2. Build pipeline; fit on train only; transform all three.
        pipeline = _build_pipeline()
        train_out = pipeline.fit_transform(train)
        val_out = pipeline.transform(val)
        test_out = pipeline.transform(test)

        # 3. Validate each split against the Tier-1 schema.
        _validate_split("train", train_out)
        _validate_split("val", val_out)
        _validate_split("test", test_out)

        # 4. Write processed parquets.
        splits = {"train": train_out, "val": val_out, "test": test_out}
        processed_paths = _write_processed_parquets(splits, settings.processed_dir)

        # 5. Save pipeline + manifest with the tier-1 filename.
        pipeline_dir = settings.models_dir / _PIPELINE_SUBDIR
        pipeline_path, manifest_path = pipeline.save(
            pipeline_dir, pipeline_filename=_PIPELINE_FILENAME
        )
        logger.info(
            "build_features_tier1.persisted",
            pipeline=str(pipeline_path),
            manifest=str(manifest_path),
        )
        run.attach_artifact("feature_manifest.json", manifest_path)

        # 6. Quick LightGBM retrain → val AUC for the report headline.
        val_auc = _quick_lgbm_val_auc(train_out, val_out, settings)
        run.log_metric("tier1_val_auc", val_auc)
        run.log_metric("tier1_n_train", float(len(train_out)))
        run.log_metric("tier1_n_val", float(len(val_out)))
        run.log_metric("tier1_n_test", float(len(test_out)))
        run.log_metric("tier1_n_features_total", float(train_out.shape[1]))

        # 7. Echo a summary table.
        click.echo(click.style("build_features_tier1: GREEN", fg="green", bold=True))
        click.echo(f"  run_id: {run.run_id}")
        click.echo(f"  pipeline: {pipeline_path}")
        click.echo(f"  manifest: {manifest_path}")
        for name, path in processed_paths.items():
            click.echo(f"  {name}.parquet: {path}  ({len(splits[name]):,} rows)")
        click.echo(f"  Tier-1 val AUC: {val_auc:.4f}  " f"(Sprint 1 baseline temporal AUC: 0.9247)")


if __name__ == "__main__":
    main()
