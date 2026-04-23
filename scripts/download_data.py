"""Download the IEEE-CIS Fraud Detection dataset from Kaggle.

Authenticates with the Kaggle API using `KAGGLE_USERNAME` / `KAGGLE_KEY`
from the project `.env`, fetches the `ieee-fraud-detection` competition
archive, extracts the five CSVs into `data/raw/`, and writes a
`MANIFEST.json` keyed by filename with SHA256 + byte count.

Business rationale:
    A deterministic fingerprint over the raw data is the only way to
    guarantee that every sprint works from bit-identical inputs.
    Without a manifest, schema drifts, missing rows, or re-releases
    from the upstream Kaggle host are invisible until features shift
    by a percent and nobody can reproduce yesterday's AUC. The
    MANIFEST.json is committed to the repo (via a negated gitignore
    rule) so the reviewer can verify the exact snapshot used.

Trade-offs considered:
    - We do not download the full ZIP once per process: if a matching
      MANIFEST.json is present AND every file's byte count + SHA256
      matches, the download is skipped. `--force` bypasses the check
      for forced re-fetches.
    - SHA256 is streamed in 1 MiB chunks so the ~1.3 GB transaction
      CSV never lives fully in memory.
    - Kaggle credentials are sourced from `Settings`; we export them
      as environment variables only for the duration of the download
      and never log them.

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --force
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import click

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.schemas.raw import SCHEMA_VERSION
from fraud_engine.utils.logging import configure_logging, get_logger, log_call

_COMPETITION_SLUG: Final[str] = "ieee-fraud-detection"
_EXPECTED_FILES: Final[tuple[str, ...]] = (
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
    "sample_submission.csv",
)
_MANIFEST_FILENAME: Final[str] = "MANIFEST.json"
_HASH_CHUNK_BYTES: Final[int] = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class FileFingerprint:
    """Single row in the manifest."""

    sha256: str
    bytes_: int


def _sha256(path: Path) -> str:
    """Stream `path` through SHA-256 in 1 MiB chunks.

    Args:
        path: Absolute path to the file.

    Returns:
        Hex-encoded digest string (64 chars).
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_dir(raw_dir: Path) -> dict[str, FileFingerprint]:
    """Hash every expected CSV in `raw_dir`.

    Missing files are simply absent from the returned dict; the caller
    compares to the expected set to decide if a re-download is needed.

    Args:
        raw_dir: The data/raw directory.

    Returns:
        Mapping of filename → FileFingerprint.
    """
    out: dict[str, FileFingerprint] = {}
    for name in _EXPECTED_FILES:
        path = raw_dir / name
        if not path.is_file():
            continue
        out[name] = FileFingerprint(
            sha256=_sha256(path),
            bytes_=path.stat().st_size,
        )
    return out


def _manifest_matches(manifest_path: Path, raw_dir: Path) -> bool:
    """Return True iff `MANIFEST.json` exists and every file matches it.

    Args:
        manifest_path: Path to the candidate manifest.
        raw_dir: The directory where the files should live.

    Returns:
        True if every expected file is present and matches the manifest
        on both byte count and SHA256. False otherwise (including the
        manifest-missing case).
    """
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.get("files", {})
    for name in _EXPECTED_FILES:
        if name not in recorded:
            return False
        entry = recorded[name]
        path = raw_dir / name
        if not path.is_file():
            return False
        if path.stat().st_size != entry["bytes"]:
            return False
        if _sha256(path) != entry["sha256"]:
            return False
    return True


def _configure_kaggle_env(settings: Settings) -> None:
    """Export `KAGGLE_USERNAME` / `KAGGLE_KEY` into the process env.

    Kaggle's Python client reads them at import time via
    `os.environ`. We set them before the first `import kaggle` so
    authentication succeeds without touching `~/.kaggle/kaggle.json`.

    Args:
        settings: The Settings instance carrying the credentials.

    Raises:
        RuntimeError: If either credential is missing.
    """
    if not settings.kaggle_username or settings.kaggle_key is None:
        raise RuntimeError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in .env before running "
            "scripts/download_data.py. See .env.example."
        )
    os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
    os.environ["KAGGLE_KEY"] = settings.kaggle_key.get_secret_value()


def _download_competition(raw_dir: Path) -> None:
    """Call the Kaggle API to download + extract the competition archive.

    Import is local so `_configure_kaggle_env` runs first (the kaggle
    module reads credentials at import time).

    Args:
        raw_dir: Destination directory; must already exist.
    """
    # Local import so the env-var configuration above is honoured;
    # the kaggle module reads KAGGLE_USERNAME / KAGGLE_KEY at import.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(
        _COMPETITION_SLUG,
        path=str(raw_dir),
        quiet=False,
    )
    archive = raw_dir / f"{_COMPETITION_SLUG}.zip"
    if not archive.is_file():
        raise RuntimeError(f"Kaggle download produced no archive at {archive}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)
    archive.unlink()


def _write_manifest(raw_dir: Path, fingerprints: dict[str, FileFingerprint]) -> Path:
    """Serialise the manifest to `raw_dir/MANIFEST.json`.

    Args:
        raw_dir: Destination directory.
        fingerprints: The {filename: FileFingerprint} mapping.

    Returns:
        The path the manifest was written to.
    """
    manifest = {
        "source": f"kaggle:{_COMPETITION_SLUG}",
        "downloaded_at": datetime.now(UTC).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "files": {
            name: {"sha256": fp.sha256, "bytes": fp.bytes_}
            for name, fp in sorted(fingerprints.items())
        },
    }
    path = raw_dir / _MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


@click.command()
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Ignore manifest and re-download even if hashes already match.",
)
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Override settings.raw_dir. Created if missing.",
)
@log_call
def main(force: bool, output_dir: str | None) -> None:
    """Acquire the IEEE-CIS raw CSVs and write a SHA256 manifest."""
    settings = get_settings()
    settings.ensure_directories()
    raw_dir = Path(output_dir) if output_dir is not None else settings.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    run_id = configure_logging(pipeline_name="data_download")
    logger = get_logger(__name__)
    logger.info(
        "download.start",
        competition=_COMPETITION_SLUG,
        raw_dir=str(raw_dir),
        force=force,
        run_id=run_id,
    )

    manifest_path = raw_dir / _MANIFEST_FILENAME
    if not force and _manifest_matches(manifest_path, raw_dir):
        click.echo(
            click.style(
                f"Manifest intact at {manifest_path} — skipping download.",
                fg="green",
            )
        )
        logger.info("download.skip", reason="manifest matches on all files")
        return

    try:
        _configure_kaggle_env(settings)
    except RuntimeError as exc:
        click.echo(click.style(str(exc), fg="red"), err=True)
        logger.error("download.auth_missing", message=str(exc))
        sys.exit(2)

    logger.info("download.fetching", competition=_COMPETITION_SLUG)
    _download_competition(raw_dir)

    fingerprints = _fingerprint_dir(raw_dir)
    missing = [name for name in _EXPECTED_FILES if name not in fingerprints]
    if missing:
        logger.error("download.incomplete", missing=missing)
        click.echo(
            click.style(f"Missing after extract: {missing}", fg="red"),
            err=True,
        )
        sys.exit(1)

    path = _write_manifest(raw_dir, fingerprints)
    click.echo(click.style(f"Wrote manifest {path}", fg="green"))
    for name, fp in sorted(fingerprints.items()):
        mb = fp.bytes_ / (1024**2)
        click.echo(f"  {name}  {fp.sha256[:12]}…  {mb:7.2f} MB")
    logger.info(
        "download.done",
        files=len(fingerprints),
        manifest=str(path),
    )


if __name__ == "__main__":
    main()
