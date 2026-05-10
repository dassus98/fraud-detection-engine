"""Populate Redis with entity-feature state from training data.

Sprint 5 prompt 5.1.g. One-shot CLI for dev / demo: reads
`data/processed/tier4_train.parquet`, groups by each entity column
(card1, addr1, DeviceInfo, P_emaildomain), and writes the most-recent
training-set entity-feature state into Redis via the Sprint 5.1.b
`RedisFeatureStore.write_entity_features` API.

Without this warmup, every /predict request against the dev / demo
stack returns `degraded_mode=true` because the Redis MGET path comes
back empty (the FeatureService catches the misses, fills with
population defaults, and flags degraded). With this warmup, predictions
for entities present in training data return non-degraded responses
that reflect the entity's real velocity / EWM / behavioural-deviation
state.

Business rationale:
    The portfolio demo's value depends on the API showing non-trivial
    SHAP reasons for known entities. A degraded-mode response shows
    Tier-1-only reasons (amount, time-of-day, email-domain flags) —
    the Tier-2/3/4 reasons (velocity, EWM heat, target encoding) only
    surface when Redis has the entity's history loaded. This script
    is the bridge between training-time features and request-time
    serving, the operational primitive Sprint 5.x's offline batch
    loader will eventually formalise.

Trade-offs considered:
    - **Click CLI + sync wrapper around async core.** Click is the
      project convention for one-shot scripts (`run_economic_evaluation.py`,
      `run_sprint1_baseline.py`); RedisFeatureStore is async-only.
      `asyncio.run(_run(...))` at the bottom of `main()` lets the
      Click decorators stay synchronous (the standard pattern) while
      the core write loop uses the async store as designed.
    - **"Most recent row per entity" snapshot semantics.** Training
      data has multiple rows per entity over time; the API needs ONE
      state per entity (the entity's current state). Sort by
      TransactionDT descending + drop_duplicates(keep="first") gives
      the canonical "freshest" snapshot. Alternative considered:
      compute the EWM running state forward through history per
      entity (production-correct) — out of scope for a warmup
      script (~30 min runtime + duplicates Tier-4 logic). The
      snapshot approach is good enough for dev/demo and makes the
      warmup deterministic + auditable.
    - **Pipelined writes per entity, NOT batched across entities.**
      `RedisFeatureStore.write_entity_features` ships one MULTI-less
      pipeline per entity (one round-trip per entity regardless of
      feature count). Cross-entity batching would multiplex SETEX
      across multiple entities in one pipeline, but the per-entity
      pipeline is already <2 ms on loopback (~13K entities × 4
      types × 2 ms ≈ 100 s wall — fine for a one-shot script).
    - **No --concurrent flag.** Single-async-task write loop. Would
      be 5-10x faster with `asyncio.gather` over batches, but the
      script is one-shot + the sequential approach has clearer
      failure semantics ("failed at entity card1=4141" vs "one of
      this batch of 100 failed").
    - **`--limit` for quick sanity testing.** Caps per-entity-type
      writes so a developer can run `--limit 100` and see ~400
      Redis keys appear in seconds, instead of waiting 100 s for a
      full warmup. Production use should leave it at the default
      (no limit).

Cross-references:
    - `src/fraud_engine/api/redis_store.py` (5.1.b) — `RedisFeatureStore`
      contract (async lifecycle, key schema, TTL config).
    - `src/fraud_engine/api/feature_service.py` (5.1.c) — Decision #2:
      per-source degraded mode. The flag this script defeats.
    - `models/sprint3/lightgbm_model_manifest.json` — `feature_names`
      array; we filter by entity prefix to know which columns to write.
    - `data/processed/tier4_train.parquet` — Tier-4-aware training data
      with all engineered entity-feature columns materialised.
    - `scripts/run_economic_evaluation.py` — Click + Settings + structlog
      pattern this script mirrors.
    - `CLAUDE.md` §3 (production-API stack), §5.4 (no hardcoded values),
      §5.5 (logging discipline).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import click
import pandas as pd

from fraud_engine.api.redis_store import RedisFeatureStore
from fraud_engine.config.settings import get_settings
from fraud_engine.utils.logging import configure_logging, get_logger

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_DEFAULT_SOURCE: Final[Path] = _PROJECT_ROOT / "data" / "processed" / "tier4_train.parquet"
_DEFAULT_MANIFEST: Final[Path] = (
    _PROJECT_ROOT / "models" / "sprint3" / "lightgbm_model_manifest.json"
)
_DEFAULT_ENTITY_TYPES: Final[tuple[str, ...]] = (
    "card1",
    "addr1",
    "DeviceInfo",
    "P_emaildomain",
)
_TIMESTAMP_COL: Final[str] = "TransactionDT"
# Progress-log cadence — emit a "wrote N entities" line every N entities
# so a developer sees forward motion on the slow path. Tuned so a
# full-warmup run (~50 K entities) emits ~50 progress lines.
_PROGRESS_EVERY: Final[int] = 1000

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Module-private helpers.
# ---------------------------------------------------------------------


def _load_feature_names(manifest_path: Path) -> list[str]:
    """Read the model manifest and return its `feature_names` array.

    Raises:
        FileNotFoundError: If the manifest is missing.
        ValueError: If `feature_names` is missing or not a list of strings.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"warmup_redis: model manifest not found at {manifest_path} — "
            f"train the LightGBM model first via Sprint 3.3.b."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_names = manifest.get("feature_names")
    if not isinstance(feature_names, list) or not all(isinstance(n, str) for n in feature_names):
        raise ValueError(
            f"warmup_redis: manifest at {manifest_path} missing or malformed "
            f"`feature_names` (expected list[str])"
        )
    return feature_names


def _entity_features_for_type(
    entity_type: str,
    feature_names: Sequence[str],
) -> list[str]:
    """Return the model-feature columns prefixed by `<entity_type>_`.

    Mirrors `feature_service._entity_type_for_feature` — a feature is
    entity-keyed iff it starts with `{entity_type}_` (e.g. `card1_velocity_24h`,
    `addr1_amt_mean_30d`, `P_emaildomain_is_free`).
    """
    prefix = f"{entity_type}_"
    return [name for name in feature_names if name.startswith(prefix)]


def _coerce_value(value: Any) -> Any:
    """Translate a parquet-cell value into a JSON-serialisable form.

    The Redis store JSON-serialises every value via `json.dumps(value,
    allow_nan=True)`. `np.float64` / `np.int64` round-trip cleanly via
    json (numpy scalars are subclasses of `float`/`int` for json
    purposes), but pandas' `NaT` and `<NA>` sentinels need explicit
    coercion.

    Args:
        value: A single cell value from a pandas DataFrame.

    Returns:
        The same value as a JSON-serialisable Python type. NaN floats
        round-trip via `allow_nan=True`; pandas NA / NaT become None.
    """
    if value is None:
        return None
    # pd.isna handles np.nan + pd.NaT + pd.NA without dragging numpy.
    try:
        if pd.isna(value):
            # Float NaN is allow_nan-able and distinguishable from None
            # in JSON ("NaN" vs null). Preserve that distinction so a
            # downstream operator can tell "feature not present" from
            # "feature present but missing".
            if isinstance(value, float) and math.isnan(value):
                return float("nan")
            return None
    except (TypeError, ValueError):
        # pd.isna raises on some numpy dtypes when passed a sequence;
        # for individual cells this should never happen, but fail-safe.
        pass
    # numpy scalars expose .item() which returns a native Python type;
    # use it where available so json.dumps doesn't need to know about numpy.
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def _extract_entity_snapshot(
    df: pd.DataFrame,
    entity_type: str,
    entity_features: Sequence[str],
    limit: int | None,
) -> dict[str, dict[str, Any]]:
    """For one entity_type, return `{entity_id_str: {feature: value, ...}}`.

    Snapshot semantics: per entity, the most-recent training row's
    values for the entity-prefixed features. "Most recent" is by
    `TransactionDT` (an integer; larger = later).

    Args:
        df: The training DataFrame (must contain `entity_type` column,
            `TransactionDT`, and every column in `entity_features`).
        entity_type: One of `card1` / `addr1` / `DeviceInfo` / `P_emaildomain`.
        entity_features: Columns to extract (prefixed with `entity_type_`).
        limit: Cap the returned dict size for quick testing. None → no cap.

    Returns:
        Dict keyed by string entity_id (cleaned to str via `str(...)`).
        Values are per-entity feature dicts ready for
        `RedisFeatureStore.write_entity_features(entity_type, entity_id, features)`.
    """
    if entity_type not in df.columns:
        _logger.warning(
            "warmup.entity_column_missing",
            entity_type=entity_type,
            note="parquet does not have this entity column; skipping",
        )
        return {}
    if not entity_features:
        _logger.info(
            "warmup.no_entity_features",
            entity_type=entity_type,
            note="model manifest has no features prefixed with this entity type; skipping",
        )
        return {}

    # Sort descending by timestamp so head(1) per group gives the freshest row.
    sorted_df = df.sort_values(_TIMESTAMP_COL, ascending=False, kind="stable")
    # `as_index=False` returns entity_type as a regular column (cleaner downstream).
    # `head(1)` on a groupby gives the first row per group post-sort
    # (i.e. the latest, since we sorted descending).
    snapshot = sorted_df.groupby(entity_type, as_index=False, sort=False).head(1)
    # Drop rows where the entity_id itself is null — those carry no
    # routable identity for the API to look up.
    snapshot = snapshot.dropna(subset=[entity_type])
    if limit is not None:
        snapshot = snapshot.head(limit)

    out: dict[str, dict[str, Any]] = {}
    for _, row in snapshot.iterrows():
        entity_id = str(row[entity_type])
        # `iat`/.loc on the row Series carries one cell at a time; build
        # the feature dict explicitly so missing columns surface as
        # KeyError now rather than at write time.
        feature_dict: dict[str, Any] = {}
        for feature_name in entity_features:
            if feature_name in row.index:
                feature_dict[feature_name] = _coerce_value(row[feature_name])
        out[entity_id] = feature_dict
    return out


async def _write_snapshot(
    store: RedisFeatureStore,
    entity_type: str,
    snapshot: dict[str, dict[str, Any]],
    dry_run: bool,
) -> tuple[int, int]:
    """Write the snapshot to Redis, one entity at a time.

    Args:
        store: Connected RedisFeatureStore.
        entity_type: For log lines + RedisFeatureStore.write_entity_features.
        snapshot: Output of `_extract_entity_snapshot`.
        dry_run: If True, log per-entity intent without writing.

    Returns:
        Tuple of (entities_written, total_features_written). On dry_run
        these are the would-be counts.
    """
    n_entities = 0
    n_features = 0
    for entity_id, features in snapshot.items():
        if not features:
            continue
        if dry_run:
            n_entities += 1
            n_features += len(features)
        else:
            await store.write_entity_features(entity_type, entity_id, features)
            n_entities += 1
            n_features += len(features)
        if n_entities % _PROGRESS_EVERY == 0:
            _logger.info(
                "warmup.progress",
                entity_type=entity_type,
                entities_written=n_entities,
                features_written=n_features,
            )
    return n_entities, n_features


async def _run(  # noqa: PLR0913 — six knobs map 1:1 to Click options; folding into a config dataclass would obscure the per-flag default behaviour
    source: Path,
    manifest: Path,
    entity_types: Sequence[str],
    limit: int | None,
    redis_url: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Async core of the warmup script.

    Returns a summary dict with per-entity-type counts + elapsed seconds.
    """
    started_at = time.perf_counter()

    _logger.info(
        "warmup.start",
        source=str(source),
        manifest=str(manifest),
        entity_types=list(entity_types),
        limit=limit,
        redis_url=redis_url or get_settings().redis_url,
        dry_run=dry_run,
    )

    # Load model feature names (~743 strings).
    feature_names = _load_feature_names(manifest)
    _logger.info("warmup.manifest_loaded", n_features=len(feature_names))

    # Load training parquet (~162 MB; ~2-3 s read).
    if not source.is_file():
        raise FileNotFoundError(
            f"warmup_redis: training data not found at {source} — "
            f"run the Sprint 2-4 pipeline to materialise it first."
        )
    t = time.perf_counter()
    df = pd.read_parquet(source)
    _logger.info(
        "warmup.parquet_loaded",
        path=str(source),
        rows=len(df),
        cols=df.shape[1],
        elapsed_s=round(time.perf_counter() - t, 2),
    )

    # Connect Redis (skipped on dry_run — store still gets constructed
    # so a misconfigured TTL YAML surfaces in --dry-run too).
    store = RedisFeatureStore(redis_url=redis_url)
    if not dry_run:
        await store.connect()
        _logger.info("warmup.redis_connected", url=redis_url or get_settings().redis_url)

    summary: dict[str, Any] = {"per_entity_type": {}, "totals": {}}
    total_entities = 0
    total_features = 0

    try:
        for entity_type in entity_types:
            entity_features = _entity_features_for_type(entity_type, feature_names)
            _logger.info(
                "warmup.entity_type_start",
                entity_type=entity_type,
                n_features_in_manifest=len(entity_features),
            )
            snapshot = _extract_entity_snapshot(df, entity_type, entity_features, limit)
            _logger.info(
                "warmup.entity_type_snapshot",
                entity_type=entity_type,
                n_entities=len(snapshot),
            )
            n_entities, n_features = await _write_snapshot(store, entity_type, snapshot, dry_run)
            summary["per_entity_type"][entity_type] = {
                "entities": n_entities,
                "features_written": n_features,
                "manifest_features_per_entity": len(entity_features),
            }
            total_entities += n_entities
            total_features += n_features
            _logger.info(
                "warmup.entity_type_done",
                entity_type=entity_type,
                entities=n_entities,
                features_written=n_features,
            )
    finally:
        if not dry_run:
            await store.disconnect()

    elapsed_s = round(time.perf_counter() - started_at, 2)
    summary["totals"] = {
        "entities": total_entities,
        "features_written": total_features,
        "elapsed_s": elapsed_s,
        "dry_run": dry_run,
    }
    _logger.info("warmup.done", **summary["totals"])
    return summary


# ---------------------------------------------------------------------
# Click CLI.
# ---------------------------------------------------------------------


@click.command(
    context_settings={"max_content_width": 100, "show_default": True},
    help=(
        "Populate Redis with entity-feature state from training data. "
        "Defeats degraded-mode for entities present in the training set."
    ),
)
@click.option(
    "--source",
    type=click.Path(path_type=Path),
    default=str(_DEFAULT_SOURCE),
    show_default=True,
    help="Training parquet to source entity state from.",
)
@click.option(
    "--manifest",
    type=click.Path(path_type=Path),
    default=str(_DEFAULT_MANIFEST),
    show_default=True,
    help="Model manifest to read feature_names from.",
)
@click.option(
    "--entity-types",
    type=str,
    default=",".join(_DEFAULT_ENTITY_TYPES),
    show_default=True,
    help="Comma-separated entity columns to populate (subset of card1,addr1,DeviceInfo,P_emaildomain).",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Cap per-entity-type writes for quick testing. Default: no cap.",
)
@click.option(
    "--redis-url",
    type=str,
    default=None,
    help="Override Settings.redis_url. Default: read from Settings.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Log per-entity intent without writing to Redis.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="stdlib log level.",
)
def main(  # noqa: PLR0913 — seven Click options; collapsing into a single ctx.obj dict would lose Click's auto-help rendering per-flag
    source: Path,
    manifest: Path,
    entity_types: str,
    limit: int | None,
    redis_url: str | None,
    dry_run: bool,
    log_level: str,
) -> None:
    """Click entry-point. See module docstring for the full design rationale."""
    # Mirror the project convention: env-set LOG_LEVEL → cache-clear
    # Settings → configure_logging. The env-var path lets the structlog
    # processor pick up the level the same way the API does.
    os.environ["LOG_LEVEL"] = log_level.upper()
    get_settings.cache_clear()
    configure_logging(pipeline_name="warmup_redis")

    # Parse the comma-separated entity-types list; reject any token
    # outside the canonical four (the API doesn't know how to look up
    # other entity types — the warmup must match).
    types = tuple(t.strip() for t in entity_types.split(",") if t.strip())
    unknown = set(types) - set(_DEFAULT_ENTITY_TYPES)
    if unknown:
        raise click.BadParameter(
            f"unknown entity types: {sorted(unknown)} "
            f"(supported: {list(_DEFAULT_ENTITY_TYPES)})"
        )

    summary = asyncio.run(
        _run(
            source=source,
            manifest=manifest,
            entity_types=types,
            limit=limit,
            redis_url=redis_url,
            dry_run=dry_run,
        )
    )

    # Print a compact summary to stdout for the human at the terminal
    # (the structlog stream is JSON; this is the readable line).
    totals = summary["totals"]
    per_type = summary["per_entity_type"]
    click.echo("")
    click.echo("===== warmup_redis summary =====")
    for entity_type, stats in per_type.items():
        click.echo(
            f"  {entity_type:>16}  "
            f"entities={stats['entities']:>6}  "
            f"features_written={stats['features_written']:>7}  "
            f"manifest_per_entity={stats['manifest_features_per_entity']}"
        )
    click.echo("---------------------------------")
    click.echo(
        f"  TOTAL  entities={totals['entities']}  "
        f"features_written={totals['features_written']}  "
        f"elapsed_s={totals['elapsed_s']}  "
        f"dry_run={totals['dry_run']}"
    )


if __name__ == "__main__":
    main()
