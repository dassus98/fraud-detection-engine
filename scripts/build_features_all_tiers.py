"""Materialise the full Tier-1 + Tier-2 + Tier-3 + Tier-4 + Tier-5 feature layer.

This script is the canonical entry point that produces
``data/processed/tier5_{train,val,test}.parquet`` plus
``models/pipelines/tier5_pipeline.joblib`` and
``models/pipelines/feature_manifest.json``. Runs under a `Run` so the
artefacts and parameters land under ``logs/runs/{run_id}/`` for
post-hoc audit, and concludes with a quick LightGBM retrain so the
report can quote a Tier-5 val AUC alongside Sprint 1's 0.9247
baseline, 2.1.d's Tier-1 (0.9165), 2.2.e's Tier-2 (0.9143), 2.3.c's
Tier-3 (0.9063), and 3.1.b's Tier-4 (0.7932). Spec target for Tier-5:
0.93-0.94.

Replaces ``scripts/build_features_tier1_2_3_4.py``: it chains all
four Tier-1 generators (`AmountTransformer`, `TimeFeatureGenerator`,
`EmailDomainExtractor`, `MissingIndicatorGenerator`), the three
Tier-2 generators (`VelocityCounter`, `HistoricalStats`,
`TargetEncoder`), the three Tier-3 generators (`BehavioralDeviation`,
`ColdStartHandler`, `NanGroupReducer`), the Tier-4 generator
(`ExponentialDecayVelocity`), AND the new Tier-5 generator
(`GraphFeatureExtractor`) into a single fitted `FeaturePipeline`.

Pipeline ordering puts `GraphFeatureExtractor` at position 11
(between `ExponentialDecayVelocity` and `NanGroupReducer`).
`NanGroupReducer` must stay last per its class docstring — it
removes V columns; no downstream generator may reference them. The
8 graph columns (`connected_component_size`, `entity_degree_*`,
`fraud_neighbor_rate`, `pagerank_score`, `clustering_coefficient`)
do not match `NanGroupReducer`'s `_detect_v_columns` regex, so they
survive the reduction step intact.

Business rationale:
    Sprint 3's LightGBM tuning, Sprint 4's economic-cost evaluation,
    and Sprint 5's serving layer all read the processed parquets and
    reuse the saved pipeline. Persisting the fit-on-train pipeline
    once means every later sprint scores against bit-identical
    transformations; the alternative — re-fitting features in every
    consumer — would silently drift across runs. Adding Tier-5 is the
    last feature-engineering step before hyperparameter tuning, so
    `tier5_*.parquet` is the surface every downstream stage will hit.

Trade-offs considered:
    - **Canonical replacement, not parallel script.** We rename the
      Tier-4 build to "all tiers" rather than keeping both. The
      project's downstream consumers all want the latest feature
      surface; maintaining two scripts in lock-step would invite
      drift. Tier-4 parquets remain on disk from prior runs as a
      historical reference.
    - **Schema validation against `TierFiveFeaturesSchema`.** Each
      split's output is validated *before* parquet write. The 8 new
      Tier-5 columns are all `nullable=True` because val/test rows
      legitimately produce NaN for txn-level features (the val txn
      is not in the training graph by temporal-safety contract).
    - **GraphFeatureExtractor's clustering gate fires on full data.**
      Above 50,000 txn nodes, `clustering_coefficient` falls back to
      0.0 with a structlog WARNING (the "last resort" fallback
      documented in 3.2.b). The schema accepts this — the column is
      preserved at 0.0 rather than dropped, so downstream LightGBM
      sees a constant which it then ignores.
    - **Quick LightGBM retrain.** Spec target val AUC 0.93-0.94. The
      Tier-4 build saw a regression to 0.7932 at default
      hyperparameters; we report whatever Tier-5 produces and let the
      hyperparameter-tuning prompt that follows recover the gap.

Usage:
    uv run python scripts/build_features_all_tiers.py
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
    ExponentialDecayVelocity,
    FeaturePipeline,
    GraphFeatureExtractor,
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
from fraud_engine.schemas.features import TierFiveFeaturesSchema
from fraud_engine.utils.logging import get_logger
from fraud_engine.utils.tracing import Run

# Filenames pinned at module scope. Sprint 3+ loaders glob by these
# names — renaming requires a coordinated update there.
_PIPELINE_FILENAME: Final[str] = "tier5_pipeline.joblib"
_PIPELINE_SUBDIR: Final[str] = "pipelines"
_PROCESSED_PREFIX: Final[str] = "tier5"

# Columns dropped from the LightGBM retrain feature set. Identical
# to the Tier-1/2/3/4 builds — Tier-5 introduces no new non-feature
# columns, only deterministic feature additions.
_NON_FEATURE_COLS: Final[frozenset[str]] = frozenset(
    {"TransactionID", "TransactionDT", "isFraud", "timestamp"}
)

# Comparison anchors for the echoed summary line.
_SPRINT1_BASELINE_VAL_AUC: Final[float] = 0.9247
_TIER1_VAL_AUC: Final[float] = 0.9165
_TIER2_VAL_AUC: Final[float] = 0.9143
_TIER3_VAL_AUC: Final[float] = 0.9063
_TIER4_VAL_AUC: Final[float] = 0.7932


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
    """Construct the canonical T1+T2+T3+T4+T5 `FeaturePipeline`.

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
        10. ExponentialDecayVelocity     — per-(entity, λ) EWM (with OOF-safe fraud_v_ewm)
        11. GraphFeatureExtractor        — bipartite-graph features (CC, degree, fraud-neighbour, pagerank, clustering)
        12. NanGroupReducer              — drop redundant V columns (LAST; see class docstring)
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
            ExponentialDecayVelocity(),
            GraphFeatureExtractor(),
            NanGroupReducer(),
        ]
    )


def _validate_split(name: str, df: pd.DataFrame) -> None:
    """Validate one split against `TierFiveFeaturesSchema`."""
    TierFiveFeaturesSchema.validate(df, lazy=True)
    get_logger(__name__).info(
        "build_features_all_tiers.validated",
        split=name,
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
    )


def _write_processed_parquets(
    splits: dict[str, pd.DataFrame], processed_dir: Path
) -> dict[str, Path]:
    """Write each split to ``processed_dir/tier5_{name}.parquet``."""
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
    """Build the full T1+T2+T3+T4+T5 feature layer; report a quick val AUC."""
    settings = get_settings()
    settings.ensure_directories()
    logger = get_logger(__name__)

    with Run("build_features_all_tiers", settings=settings) as run:
        run.log_param("interim_dir", str(settings.interim_dir))
        run.log_param("processed_dir", str(settings.processed_dir))
        run.log_param("seed", settings.seed)

        # 1. Load interim splits.
        train = _load_interim_split(settings.interim_dir, "train")
        val = _load_interim_split(settings.interim_dir, "val")
        test = _load_interim_split(settings.interim_dir, "test")
        logger.info(
            "build_features_all_tiers.loaded",
            n_train=len(train),
            n_val=len(val),
            n_test=len(test),
        )

        # 2. Build pipeline; fit on train only; transform all three.
        pipeline = _build_pipeline()
        train_out = pipeline.fit_transform(train)
        val_out = pipeline.transform(val)
        test_out = pipeline.transform(test)

        # 3. Validate each split against the Tier-5 schema.
        _validate_split("train", train_out)
        _validate_split("val", val_out)
        _validate_split("test", test_out)

        # 4. Write processed parquets.
        splits = {"train": train_out, "val": val_out, "test": test_out}
        processed_paths = _write_processed_parquets(splits, settings.processed_dir)

        # 5. Save pipeline + manifest with the tier-5 filename.
        pipeline_dir = settings.models_dir / _PIPELINE_SUBDIR
        pipeline_path, manifest_path = pipeline.save(
            pipeline_dir, pipeline_filename=_PIPELINE_FILENAME
        )
        logger.info(
            "build_features_all_tiers.persisted",
            pipeline=str(pipeline_path),
            manifest=str(manifest_path),
        )
        run.attach_artifact("feature_manifest.json", manifest_path)

        # 6. Quick LightGBM retrain → val AUC for the report headline.
        val_auc = _quick_lgbm_val_auc(train_out, val_out, settings)
        run.log_metric("tier5_val_auc", val_auc)
        run.log_metric("tier5_n_train", float(len(train_out)))
        run.log_metric("tier5_n_val", float(len(val_out)))
        run.log_metric("tier5_n_test", float(len(test_out)))
        run.log_metric("tier5_n_features_total", float(train_out.shape[1]))

        # 7. Echo a summary table.
        click.echo(click.style("build_features_all_tiers: GREEN", fg="green", bold=True))
        click.echo(f"  run_id: {run.run_id}")
        click.echo(f"  pipeline: {pipeline_path}")
        click.echo(f"  manifest: {manifest_path}")
        for name, path in processed_paths.items():
            click.echo(f"  {name}.parquet: {path}  ({len(splits[name]):,} rows)")
        click.echo(
            f"  Tier-5 val AUC: {val_auc:.4f}  "
            f"(Tier-4: {_TIER4_VAL_AUC:.4f}; "
            f"Tier-3: {_TIER3_VAL_AUC:.4f}; "
            f"Tier-2: {_TIER2_VAL_AUC:.4f}; "
            f"Tier-1: {_TIER1_VAL_AUC:.4f}; "
            f"Sprint 1 baseline: {_SPRINT1_BASELINE_VAL_AUC:.4f})"
        )


if __name__ == "__main__":
    main()
