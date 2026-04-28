"""Profile `NanGroupReducer` on the Tier-2 processed parquets.

Runs the V-reducer end-to-end against `data/processed/tier2_*.parquet`
and produces:

- `models/pipelines/v_reduction_manifest.json` — the per-group drop /
  PCA manifest from `NanGroupReducer.get_manifest()`.
- A printed summary: kept / dropped column counts, before-vs-after
  val AUC from a quick LightGBM retrain.

Source provenance for the per-prompt-2.3.b
`reports/v_feature_reduction_report.md` and Sprint-3's tuning sweep,
which needs to know exactly which V-columns were discarded and why.

Usage:
    uv run python scripts/profile_v_reduction.py
"""

from __future__ import annotations

import json
from typing import Final

import click
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.features.v_reduction import NanGroupReducer

_NON_FEATURE_COLS: Final[frozenset[str]] = frozenset(
    {"TransactionID", "TransactionDT", "isFraud", "timestamp"}
)
_MANIFEST_FILENAME: Final[str] = "v_reduction_manifest.json"


def _select_lgbm_features(df: pd.DataFrame) -> list[str]:
    """Mirror the build-script feature selection (drops object/string + non-feature)."""
    return [
        col
        for col in df.columns
        if col not in _NON_FEATURE_COLS
        and not pd.api.types.is_object_dtype(df[col])
        and not pd.api.types.is_string_dtype(df[col])
    ]


def _quick_val_auc(train: pd.DataFrame, val: pd.DataFrame, settings: Settings) -> float:
    """Fit a LightGBM on train; score val; return ROC-AUC."""
    feat = _select_lgbm_features(train)
    clf = LGBMClassifier(
        **settings.lgbm_defaults,
        random_state=settings.seed,
        verbose=-1,
    )
    clf.fit(train[feat], train["isFraud"], categorical_feature="auto")
    proba = clf.predict_proba(val[feat])[:, 1]
    return float(roc_auc_score(val["isFraud"], proba))


@click.command()
def main() -> None:
    """Profile the V-reducer on Tier-2 parquets."""
    settings = get_settings()
    train = pd.read_parquet(settings.processed_dir / "tier2_train.parquet")
    val = pd.read_parquet(settings.processed_dir / "tier2_val.parquet")

    v_cols_before = [c for c in train.columns if c.startswith("V") and c[1:].isdigit()]
    click.echo(f"V columns in train (pre-reduction): {len(v_cols_before)}")

    # 1. Fit reducer on train only.
    reducer = NanGroupReducer()
    reducer.fit(train)

    # 2. Write manifest.
    pipeline_dir = settings.models_dir / "pipelines"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    manifest = reducer.get_manifest()
    manifest_path = pipeline_dir / _MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    click.echo(f"Manifest written: {manifest_path}")
    click.echo(
        f"  n_groups={manifest['n_groups']} "
        f"n_input={manifest['n_columns_input']} "
        f"n_kept={manifest['n_columns_output']} "
        f"n_dropped={manifest['n_columns_dropped']}"
    )

    # 3. Before AUC.
    click.echo("Training LightGBM on full feature set (pre-reduction) ...")
    auc_before = _quick_val_auc(train, val, settings)
    click.echo(f"  val_auc_before = {auc_before:.4f}")

    # 4. After AUC.
    train_reduced = reducer.transform(train)
    val_reduced = reducer.transform(val)
    click.echo(
        f"After reduction: train_cols={train_reduced.shape[1]} " f"val_cols={val_reduced.shape[1]}"
    )
    click.echo("Training LightGBM on reduced feature set ...")
    auc_after = _quick_val_auc(train_reduced, val_reduced, settings)
    click.echo(f"  val_auc_after  = {auc_after:.4f}")

    delta = auc_after - auc_before
    click.echo(f"  delta = {delta:+.4f}")

    summary = {
        "v_columns_before": len(v_cols_before),
        "n_groups": manifest["n_groups"],
        "n_kept": manifest["n_columns_output"],
        "n_dropped": manifest["n_columns_dropped"],
        "val_auc_before": auc_before,
        "val_auc_after": auc_after,
        "delta": delta,
        "method": manifest["method"],
        "correlation_threshold": manifest["correlation_threshold"],
    }
    summary_path = pipeline_dir / "v_reduction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    click.echo(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()
