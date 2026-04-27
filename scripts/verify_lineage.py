"""Validate lineage and interim-data invariants for a run.

Reads the most recent (or `--run-id`-specified) lineage JSONL trail and
the interim parquet outputs, then asserts every contract that links
them. Failures print a red status table and exit non-zero; a clean run
prints a green summary. This script is the Sprint 1+ replacement for
``scripts/verify_bootstrap.py`` referenced by CLAUDE.md §11.

Business rationale:
    A lineage record is only as trustworthy as the cross-checks
    against the artefacts it claims to describe. ``verify_lineage.py``
    is the gate that closes that loop: it confirms the JSONL trail
    matches the on-disk parquets, the row-count chain is internally
    consistent, and the manifest schema_version has not regressed.
    Re-runs of this script (one per sprint, per CI invocation) are
    the contractual evidence that the data layer is reproducible.

Trade-offs considered:
    - "Most recent run" is identified by mtime on
      ``logs/lineage/*/lineage.jsonl``. The alternative — scanning
      ``logs/runs/*/run.json`` for the latest ``build_interim`` and
      cross-referencing — is more "correct" but adds a join across
      two trees for no behavioural improvement when this script runs
      directly after `build_interim.py`.
    - The script collects every failure rather than failing fast on
      the first one. A reviewer running it after a build wants to
      see the whole picture, not iterate one fix at a time.
    - Manifest `schema_version` is pinned to a literal in this script
      rather than imported from `splits._MANIFEST_SCHEMA_VERSION`.
      This is intentional: the script's whole job is to detect
      regressions; importing the live constant would mask a silent
      bump that this gate is supposed to catch loudly.
    - We read parquet row counts via `pyarrow.parquet.read_metadata`
      rather than a full `pd.read_parquet`. The metadata read is
      O(1) and avoids materialising the (potentially hundreds of MB)
      frame for a row-count check.

Usage:
    uv run python scripts/verify_lineage.py
    uv run python scripts/verify_lineage.py --run-id <hex>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

import click
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.data.lineage import LineageLog, LineageStep
from fraud_engine.utils.logging import configure_logging, get_logger

# Step names the build pipeline emits, in chronological order. Pinned
# here independently of `scripts/build_interim.py`'s private `_STEP_*`
# constants so this verifier remains the *contract* — a future refactor
# that renames a step in the build script will fail this gate until
# both sides are updated together.
_EXPECTED_STEPS: Final[tuple[str, ...]] = (
    "load_merged",
    "interim_clean",
    "split_train",
    "split_val",
    "split_test",
)

# Filenames the build pipeline persists. Mirrored from
# `scripts/build_interim.py`; intentional duplication so the verifier
# is an independent check, not a derivative of the producer.
_TRAIN_FILENAME: Final[str] = "train.parquet"
_VAL_FILENAME: Final[str] = "val.parquet"
_TEST_FILENAME: Final[str] = "test.parquet"
_MANIFEST_FILENAME: Final[str] = "splits_manifest.json"

# Map split-step name → expected parquet filename. Used by the parquet
# row-count vs lineage-row-count cross-check.
_SPLIT_PARQUETS: Final[tuple[tuple[str, str], ...]] = (
    ("split_train", _TRAIN_FILENAME),
    ("split_val", _VAL_FILENAME),
    ("split_test", _TEST_FILENAME),
)

# Pinned manifest schema version. See module docstring for why this is
# a literal rather than an import. Bump together with
# `splits._MANIFEST_SCHEMA_VERSION` when the manifest shape changes.
_EXPECTED_MANIFEST_SCHEMA_VERSION: Final[int] = 1


def _find_latest_run_id(logs_dir: Path) -> str:
    """Return the run_id whose `lineage.jsonl` has the newest mtime.

    Args:
        logs_dir: The configured `Settings.logs_dir`. The lineage
            tree lives at `logs_dir / "lineage" / *`.

    Returns:
        The directory name of the most recently modified lineage run.

    Raises:
        FileNotFoundError: If no `lineage.jsonl` exists under
            `logs_dir/lineage/`.
    """
    candidates = list((logs_dir / "lineage").glob("*/lineage.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No lineage.jsonl found under {logs_dir / 'lineage'}; "
            "run `uv run python scripts/build_interim.py` first."
        )
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.parent.name


def _check_steps_present(steps_by_name: dict[str, LineageStep], failures: list[str]) -> None:
    """Assert every step in `_EXPECTED_STEPS` appears in the lineage trail."""
    for expected in _EXPECTED_STEPS:
        if expected not in steps_by_name:
            failures.append(f"Missing lineage step: {expected!r}")


def _check_load_clean_chain(steps_by_name: dict[str, LineageStep], failures: list[str]) -> None:
    """Assert the cleaner's input rows equal the loader's output rows."""
    if "load_merged" not in steps_by_name or "interim_clean" not in steps_by_name:
        return
    load_out = steps_by_name["load_merged"].output_rows
    clean_in = steps_by_name["interim_clean"].input_rows
    if load_out != clean_in:
        failures.append(
            f"Row chain broken: load_merged.output_rows={load_out:,} "
            f"!= interim_clean.input_rows={clean_in:,}"
        )


def _check_clean_drop_invariant(steps_by_name: dict[str, LineageStep], failures: list[str]) -> None:
    """Assert `cleaner` only ever drops rows (never invents them)."""
    if "interim_clean" not in steps_by_name:
        return
    step = steps_by_name["interim_clean"]
    if step.output_rows > step.input_rows:
        failures.append(
            f"Cleaner invented rows: interim_clean.output_rows="
            f"{step.output_rows:,} > input_rows={step.input_rows:,}"
        )


def _check_splits_sum(steps_by_name: dict[str, LineageStep], failures: list[str]) -> None:
    """Assert split partitions sum to the cleaner's output."""
    needed = ("interim_clean", "split_train", "split_val", "split_test")
    if not all(name in steps_by_name for name in needed):
        return
    cleaned_out = steps_by_name["interim_clean"].output_rows
    split_sum = sum(
        steps_by_name[name].output_rows for name in ("split_train", "split_val", "split_test")
    )
    if cleaned_out != split_sum:
        failures.append(
            f"Partition sum mismatch: interim_clean.output_rows={cleaned_out:,} "
            f"!= split_train+val+test={split_sum:,}"
        )


def _check_parquets_match_lineage(
    steps_by_name: dict[str, LineageStep],
    interim_dir: Path,
    failures: list[str],
) -> None:
    """Assert each parquet's row count matches its split-step record."""
    for slice_name, filename in _SPLIT_PARQUETS:
        parquet_path = interim_dir / filename
        if not parquet_path.is_file():
            failures.append(f"Missing interim parquet: {parquet_path}")
            continue
        n_parquet = int(pq.read_metadata(parquet_path).num_rows)
        if slice_name not in steps_by_name:
            continue
        n_lineage = steps_by_name[slice_name].output_rows
        if n_parquet != n_lineage:
            failures.append(
                f"{filename}: parquet has {n_parquet:,} rows, "
                f"lineage records {n_lineage:,} for {slice_name!r}"
            )


def _check_manifest_schema_version(interim_dir: Path, failures: list[str]) -> None:
    """Assert the manifest's `schema_version` matches the pinned expectation."""
    manifest_path = interim_dir / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        failures.append(f"Missing manifest: {manifest_path}")
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = payload.get("schema_version")
    if actual != _EXPECTED_MANIFEST_SCHEMA_VERSION:
        failures.append(
            f"{_MANIFEST_FILENAME}: schema_version={actual!r}, "
            f"expected {_EXPECTED_MANIFEST_SCHEMA_VERSION}"
        )


def _run_checks(steps: list[LineageStep], settings: Settings) -> list[str]:
    """Run every contract check and return the list of failure messages.

    Args:
        steps: The full lineage trail for the run, in append order.
        settings: Active settings (drives `interim_dir`).

    Returns:
        An empty list on full pass; otherwise one human-readable
        failure string per broken invariant.
    """
    by_name: dict[str, LineageStep] = {step.step_name: step for step in steps}
    failures: list[str] = []
    _check_steps_present(by_name, failures)
    _check_load_clean_chain(by_name, failures)
    _check_clean_drop_invariant(by_name, failures)
    _check_splits_sum(by_name, failures)
    _check_parquets_match_lineage(by_name, settings.interim_dir, failures)
    _check_manifest_schema_version(settings.interim_dir, failures)
    return failures


def _print_green(run_id: str, steps: list[LineageStep]) -> None:
    """Render the all-checks-pass status block."""
    click.echo(click.style("Lineage verification: GREEN", fg="green", bold=True))
    click.echo(f"  run_id: {run_id}")
    click.echo(f"  steps:  {len(steps)} ({', '.join(s.step_name for s in steps)})")


def _print_red(run_id: str, failures: list[str]) -> None:
    """Render the failure block for non-zero exit."""
    click.echo(click.style("Lineage verification: RED", fg="red", bold=True))
    click.echo(f"  run_id: {run_id}")
    for failure in failures:
        click.echo(click.style(f"  - {failure}", fg="red"))


@click.command()
@click.option(
    "--run-id",
    "run_id",
    default=None,
    help="Run id to verify; defaults to the most recently modified lineage trail.",
)
def verify(run_id: str | None) -> None:
    """Validate lineage + parquet invariants. Exits 0 on green, 1 on red."""
    settings = get_settings()
    configure_logging(pipeline_name="verify_lineage")
    logger = get_logger(__name__)

    effective_run_id = run_id or _find_latest_run_id(settings.logs_dir)
    logger.info("verify.start", run_id=effective_run_id)

    steps = LineageLog.read(effective_run_id, settings=settings)
    if not steps:
        click.echo(
            click.style(
                f"Lineage verification: RED — no records under run_id={effective_run_id}",
                fg="red",
                bold=True,
            ),
            err=True,
        )
        logger.error("verify.empty", run_id=effective_run_id)
        sys.exit(1)

    failures = _run_checks(steps, settings)
    if failures:
        _print_red(effective_run_id, failures)
        logger.error("verify.failed", run_id=effective_run_id, n_failures=len(failures))
        sys.exit(1)

    _print_green(effective_run_id, steps)
    logger.info("verify.passed", run_id=effective_run_id, n_steps=len(steps))


if __name__ == "__main__":
    verify()
