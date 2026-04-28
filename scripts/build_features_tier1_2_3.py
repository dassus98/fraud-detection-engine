"""Materialise the Tier-1 + Tier-2 + Tier-3 feature layer.

This script is the canonical entry point that produces
``data/processed/tier3_{train,val,test}.parquet`` plus
``models/pipelines/tier3_pipeline.joblib`` and
``models/pipelines/feature_manifest.json``. Runs under a `Run` so
the artefacts and parameters land under ``logs/runs/{run_id}/`` for
post-hoc audit, and concludes with a quick LightGBM retrain so the
report can quote a Tier-3 val AUC alongside Sprint 1's 0.9247
baseline, 2.1.d's Tier-1 (0.9165), and 2.2.e's Tier-2 (0.9143).

The Tier-3 build chains all four Tier-1 generators
(`AmountTransformer`, `TimeFeatureGenerator`, `EmailDomainExtractor`,
`MissingIndicatorGenerator`), the three Tier-2 generators
(`VelocityCounter`, `HistoricalStats`, `TargetEncoder`), and the
three Tier-3 generators (`BehavioralDeviation`, `ColdStartHandler`,
`NanGroupReducer`) into a single fitted `FeaturePipeline`.

Pipeline ordering puts `NanGroupReducer` LAST so it sees the full
column set and so no downstream generator references the dropped
V columns (per the class docstring).

Business rationale:
    Sprint 3's LightGBM tuning, Sprint 4's economic-cost evaluation,
    and Sprint 5's serving layer all read the processed parquets and
    reuse the saved pipeline. Persisting the fit-on-train pipeline
    once means every later sprint scores against bit-identical
    transformations; the alternative — re-fitting features in every
    consumer — would silently drift across runs.

Trade-offs considered:
    - **Fit on train only.** `pipeline.fit_transform(train)` runs
      OOF target encoding on training rows AND fits the full-train
      encoder; later `transform(val)` and `transform(test)` apply the
      full-train encoder. Same train-only discipline as Tier-1/2.
    - **Schema validation against `TierThreeFeaturesSchema`.** Each
      split's output is validated *before* parquet write. Missing V
      columns (dropped by `NanGroupReducer`) pass through via
      `strict=False` inherited from `MergedSchema`.
    - **Quick LightGBM retrain.** Spec acceptance Val AUC ≥ 0.91.
      Object / string columns (provider, tld) are dropped; the
      `*_target_enc` columns are float and survive.

Usage:
    uv run python scripts/build_features_tier1_2_3.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import click
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.features import (
    BehavioralDeviation,
    ColdStartHandler,
    FeaturePipeline,
    HistoricalStats,
    NanGroupReducer,
    TargetEncoder,
    VelocityCounter,
)
from fraud_engine.features.tier1_basic import (
    AmountTransformer,
    EmailDomainExtractor,
    MissingIndicatorGenerator,
    TimeFeatureGenerator,
)
from fraud_engine.schemas.features import TierThreeFeaturesSchema
from fraud_engine.utils.logging import get_logger
from fraud_engine.utils.tracing import Run

# Filenames pinned at module scope. Sprint 3+ loaders glob by these
# names — renaming requires a coordinated update there.
_PIPELINE_FILENAME: Final[str] = "tier3_pipeline.joblib"
_PIPELINE_SUBDIR: Final[str] = "pipelines"
_PROCESSED_PREFIX: Final[str] = "tier3"

# Columns dropped from the LightGBM retrain feature set. Identical
# to the Tier-1/2 builds — Tier-3 introduces no new non-feature
# columns, only deterministic feature additions and V drops.
_NON_FEATURE_COLS: Final[frozenset[str]] = frozenset(
    {"TransactionID", "TransactionDT", "isFraud", "timestamp"}
)

# Comparison anchors for the echoed summary line.
_SPRINT1_BASELINE_VAL_AUC: Final[float] = 0.9247
_TIER1_VAL_AUC: Final[float] = 0.9165
_TIER2_VAL_AUC: Final[float] = 0.9143


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
    """Construct the canonical Tier-1 + Tier-2 + Tier-3 `FeaturePipeline`.

    Order matters:
        1.  AmountTransformer            — log_amount / amount_decile
        2.  TimeFeatureGenerator         — hour / weekend / sin/cos
        3.  EmailDomainExtractor         — provider / tld / is_free / is_disposable
        4.  MissingIndicatorGenerator    — is_null_* (data-dependent)
        5.  VelocityCounter              — per-entity counts in 1h/24h/7d windows
        6.  HistoricalStats              — rolling mean/std/max
        7.  TargetEncoder                — OOF on training; full-train encoder for val/test
        8.  BehavioralDeviation          — per-card1 amt_z, time_z, addr/device change, hour deviation
        9.  ColdStartHandler             — is_coldstart_{entity} flags
        10. NanGroupReducer              — drop redundant V columns (LAST; see class docstring)
    """
    return FeaturePipeline(
        generators=[
            AmountTransformer(),
            TimeFeatureGenerator(),
            EmailDomainExtractor(),
            MissingIndicatorGenerator(),
            VelocityCounter(),
            HistoricalStats(),
            TargetEncoder(),
            BehavioralDeviation(),
            ColdStartHandler(),
            NanGroupReducer(),
        ]
    )


def _validate_split(name: str, df: pd.DataFrame) -> None:
    """Validate one split against `TierThreeFeaturesSchema`."""
    TierThreeFeaturesSchema.validate(df, lazy=True)
    get_logger(__name__).info(
        "build_features_tier3.validated",
        split=name,
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
    )


def _write_processed_parquets(
    splits: dict[str, pd.DataFrame], processed_dir: Path
) -> dict[str, Path]:
    """Write each split to ``processed_dir/tier3_{name}.parquet``."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, df in splits.items():
        path = processed_dir / f"{_PROCESSED_PREFIX}_{name}.parquet"
        df.to_parquet(path)
        paths[name] = path
    return paths


def _select_lgbm_features(df: pd.DataFrame) -> list[str]:
    """Return the subset of columns LightGBM's sklearn API can ingest."""
    return [
        col
        for col in df.columns
        if col not in _NON_FEATURE_COLS
        and not pd.api.types.is_object_dtype(df[col])
        and not pd.api.types.is_string_dtype(df[col])
    ]


def _quick_lgbm_val_auc(train: pd.DataFrame, val: pd.DataFrame, settings: Settings) -> float:
    """Fit a LightGBM on train; score val; return ROC-AUC."""
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
    """Build the Tier-1 + Tier-2 + Tier-3 feature layer and report a quick val AUC."""
    settings = get_settings()
    settings.ensure_directories()
    logger = get_logger(__name__)

    with Run("build_features_tier3", settings=settings) as run:
        run.log_param("interim_dir", str(settings.interim_dir))
        run.log_param("processed_dir", str(settings.processed_dir))
        run.log_param("seed", settings.seed)

        # 1. Load interim splits.
        train = _load_interim_split(settings.interim_dir, "train")
        val = _load_interim_split(settings.interim_dir, "val")
        test = _load_interim_split(settings.interim_dir, "test")
        logger.info(
            "build_features_tier3.loaded",
            n_train=len(train),
            n_val=len(val),
            n_test=len(test),
        )

        # 2. Build pipeline; fit on train only; transform all three.
        pipeline = _build_pipeline()
        train_out = pipeline.fit_transform(train)
        val_out = pipeline.transform(val)
        test_out = pipeline.transform(test)

        # 3. Validate each split against the Tier-3 schema.
        _validate_split("train", train_out)
        _validate_split("val", val_out)
        _validate_split("test", test_out)

        # 4. Write processed parquets.
        splits = {"train": train_out, "val": val_out, "test": test_out}
        processed_paths = _write_processed_parquets(splits, settings.processed_dir)

        # 5. Save pipeline + manifest with the tier-3 filename.
        pipeline_dir = settings.models_dir / _PIPELINE_SUBDIR
        pipeline_path, manifest_path = pipeline.save(
            pipeline_dir, pipeline_filename=_PIPELINE_FILENAME
        )
        logger.info(
            "build_features_tier3.persisted",
            pipeline=str(pipeline_path),
            manifest=str(manifest_path),
        )
        run.attach_artifact("feature_manifest.json", manifest_path)

        # 6. Quick LightGBM retrain → val AUC for the report headline.
        val_auc = _quick_lgbm_val_auc(train_out, val_out, settings)
        run.log_metric("tier3_val_auc", val_auc)
        run.log_metric("tier3_n_train", float(len(train_out)))
        run.log_metric("tier3_n_val", float(len(val_out)))
        run.log_metric("tier3_n_test", float(len(test_out)))
        run.log_metric("tier3_n_features_total", float(train_out.shape[1]))

        # 7. Echo a summary table.
        click.echo(click.style("build_features_tier3: GREEN", fg="green", bold=True))
        click.echo(f"  run_id: {run.run_id}")
        click.echo(f"  pipeline: {pipeline_path}")
        click.echo(f"  manifest: {manifest_path}")
        for name, path in processed_paths.items():
            click.echo(f"  {name}.parquet: {path}  ({len(splits[name]):,} rows)")
        click.echo(
            f"  Tier-3 val AUC: {val_auc:.4f}  "
            f"(Tier-2: {_TIER2_VAL_AUC:.4f}; "
            f"Tier-1: {_TIER1_VAL_AUC:.4f}; "
            f"Sprint 1 baseline: {_SPRINT1_BASELINE_VAL_AUC:.4f})"
        )


if __name__ == "__main__":
    main()
