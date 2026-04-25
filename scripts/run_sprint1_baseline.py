"""Sprint 1 baseline runner — end-to-end LightGBM on the real dataset.

Loads the merged IEEE-CIS frame, carves train/val/test via
`temporal_split`, persists the split manifest, fits the LightGBM
baseline under both the random and temporal variants, and logs
every run to MLflow + the per-run directory under `logs/runs/`.

Business rationale:
    The integration test fits on 10k rows to keep CI fast; the real
    full-dataset numbers live here. This script is what the sprint
    report quotes, what the hiring reviewer re-runs if they want to
    verify a claim, and what the MLflow UI populates as the "before"
    row of every later sprint's leaderboard.

Trade-offs considered:
    - Click `--random/--no-random` + `--temporal/--no-temporal`
      flags let the reviewer skip either half if they want to check
      only one variant. Defaults run both; that is what the acceptance
      checklist requires.
    - Runs under one `run_context` wrapper so stdout/stderr + the
      result dicts + the split manifest all land in a single
      `logs/runs/{run_id}/` directory — makes the post-hoc
      inspection trivially `cat`-able.

Usage:
    uv run python scripts/run_sprint1_baseline.py
    uv run python scripts/run_sprint1_baseline.py --no-random
    uv run python scripts/run_sprint1_baseline.py --log-level DEBUG
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from typing import Final

import click
import pandas as pd

from fraud_engine.config.settings import get_settings
from fraud_engine.data.loader import RawDataLoader
from fraud_engine.data.splits import (
    temporal_split,
    validate_no_overlap,
    write_split_manifest,
)
from fraud_engine.models.baseline import BaselineResult, Variant, train_baseline
from fraud_engine.utils.logging import configure_logging, get_logger
from fraud_engine.utils.tracing import Run, attach_artifact, run_context

_SPLITS_MANIFEST_FILENAME: Final[str] = "splits_manifest.json"


def _run_variant(
    merged: pd.DataFrame,
    *,
    variant: Variant,
    run: Run,
) -> BaselineResult:
    """Fit a single variant, attach its result dict to the run directory.

    Private helper so the main function reads as a sequence of
    conditional variant fits without duplicated plumbing.
    """
    logger = get_logger(__name__)
    logger.info("baseline.variant_start", variant=variant)
    result = train_baseline(merged, variant=variant)
    attach_artifact(
        run,
        dataclasses.asdict(result),
        name=f"baseline_{variant}_result",
    )
    logger.info(
        "baseline.variant_done",
        variant=variant,
        auc=result.auc,
        model_path=str(result.model_path),
        content_hash=result.content_hash,
    )
    return result


@click.command()
@click.option(
    "--random/--no-random",
    default=True,
    help="Run the random-split variant (stratified 80/20).",
)
@click.option(
    "--temporal/--no-temporal",
    default=True,
    help="Run the temporal-split variant using Settings boundaries.",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        case_sensitive=False,
    ),
    help="Override the structlog level for this run.",
)
def main(random: bool, temporal: bool, log_level: str) -> None:
    """Fit the Sprint 1 LightGBM baseline on the full merged dataset."""
    # Propagate the CLI log level into Settings so `configure_logging`
    # picks it up. `get_settings` is lru_cached; clear before reading.
    os.environ["LOG_LEVEL"] = log_level.upper()
    get_settings.cache_clear()
    settings = get_settings()
    settings.ensure_directories()

    if not (random or temporal):
        click.echo(
            click.style(
                "Both --no-random and --no-temporal were set — nothing to do.",
                fg="red",
            ),
            err=True,
        )
        sys.exit(2)

    configure_logging(pipeline_name="sprint1_baseline")
    logger = get_logger(__name__)

    with run_context(
        "sprint1_baseline",
        metadata={"variants_requested": {"random": random, "temporal": temporal}},
    ) as run:
        logger.info("sprint1.load_start")
        loader = RawDataLoader()
        merged = loader.load_merged(optimize=True)
        logger.info(
            "sprint1.load_done",
            rows=int(merged.shape[0]),
            cols=int(merged.shape[1]),
        )

        # Always carve + persist the temporal partition — even if only
        # the random variant runs, Sprint 2 reads the manifest to line
        # up its feature windows.
        splits = temporal_split(merged, settings=settings)
        validate_no_overlap(splits)
        manifest_path = settings.interim_dir / _SPLITS_MANIFEST_FILENAME
        write_split_manifest(splits, manifest_path)
        attach_artifact(run, splits.manifest, name="splits_manifest")
        logger.info(
            "sprint1.splits_written",
            path=str(manifest_path),
            n_train=splits.manifest["n_train"],
            n_val=splits.manifest["n_val"],
            n_test=splits.manifest["n_test"],
        )

        results: dict[str, BaselineResult] = {}
        if random:
            results["random"] = _run_variant(merged, variant="random", run=run)
        if temporal:
            results["temporal"] = _run_variant(merged, variant="temporal", run=run)

        summary_lines = [
            "=" * 60,
            "Sprint 1 baseline — AUC summary",
            "=" * 60,
        ]
        for variant, result in results.items():
            summary_lines.append(
                f"  {variant:8s}  AUC={result.auc:.4f}  " f"model={Path(result.model_path).name}"
            )
        summary_lines.append("=" * 60)
        logger.info("sprint1.summary", lines=summary_lines)
        for line in summary_lines:
            click.echo(line)


if __name__ == "__main__":
    main()
