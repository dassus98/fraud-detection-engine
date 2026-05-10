"""Online feature orchestrator for the fraud-scoring API.

Sprint 5 prompt 5.1.c: the request-time pipeline that combines three
feature sources into one DataFrame ready for `LightGBMFraudModel.predict_proba`:

1. **Real-time (inline)** — Tier-1 generators (`AmountTransformer`,
   `TimeFeatureGenerator`, `EmailDomainExtractor`,
   `MissingIndicatorGenerator`) computed inline from the request payload.
2. **Entity (Redis)** — Tier-2 velocity + amount stats, Tier-3
   behavioural deviations, Tier-4 EWM decay state — looked up via the
   Sprint 5.1.b `RedisFeatureStore` MGET path.
3. **Batch (Postgres)** — Tier-5 graph features (`pagerank_score`,
   `connected_component_size`, `entity_degree_*`,
   `fraud_neighbor_rate`, `clustering_coefficient`). 5.1.c stubs the
   actual `SELECT` (the table doesn't exist yet — Sprint 5.x's batch
   loader builds the schema). The connection lifecycle IS real: a
   per-call `SELECT 1` health probe drives the Postgres source's
   degraded-mode flag.

When Redis or Postgres is unreachable, missing features are filled
from `configs/feature_defaults.yaml` and `degraded_mode=True` is set
on the returned `FeatureVector` — the load-bearing fault-tolerance
contract behind `PredictionResponse.degraded_mode` (Sprint 5.1.a).

Business rationale:
    Stateful features (velocity, EWM, behavioural baselines, graph
    metrics) cannot be recomputed at request time from the inbound
    transaction alone — they depend on history. Redis at <5ms MGET +
    Postgres at <10ms SELECT keeps the project's <100ms P95 budget
    intact (CLAUDE.md §3) while letting Model A consume the full
    ~743-feature surface.

    Without this orchestrator, the FastAPI route (Sprint 5.1.d) has
    nothing to call into and falls back to Tier-1-only inference for
    every request, gutting the AUC the model relies on. The route
    that lands in Sprint 5.1.d will be a thin shim around
    `FeatureService.get_features(...)`.

Trade-offs considered:
    - **Postgres as connection-real / SELECT-stub.** The Postgres
      schema for batch graph features doesn't exist yet — Sprint 5.x's
      batch loader will design it. Designing it inside this prompt
      would balloon scope past the "Risk: High" budget. The compromise
      is to ship a real connection lifecycle + per-call `SELECT 1`
      health probe; on success, return population defaults; on
      failure, set degraded-mode. When 5.x lands the real query
      (`SELECT pagerank_score, ... FROM feature_batch_graph WHERE
      entity_id IN (...)`), the probe is replaced; the failure-mode
      contract is unchanged.
    - **Population defaults in YAML, not hardcoded.** Mirrors the
      `tier4_config.yaml` / `redis_feature_store.yaml` / `economic_defaults.yaml`
      runtime-consumed YAML pattern. Sprint 5.x batch loader will
      regenerate from training data; the hand-crafted values in
      `configs/feature_defaults.yaml` are conservative-by-design (zero
      velocity, IEEE-CIS population fraud rate, ~1/N PageRank). Other
      options considered: hardcoded zeros (loses missing-vs-zero
      signal), startup-time computation from training parquet
      (multi-second cold-start). Rejected per CLAUDE.md §5.4 (no
      hardcoded values) and §11 (cold-start budget).
    - **Per-source degraded flags, OR-ed.** Each external source has
      its own try/except + fallback path; the orchestrator tracks
      `source_status` per source and OR-s the booleans for the final
      `degraded_mode` flag. `PredictionResponse.degraded_mode` (5.1.a)
      stays a single bool; the per-source dict is logged but not on
      the wire. Sprint 5.x can promote it if operations needs the
      finer granularity.
    - **Output is `pd.DataFrame` of shape `(1, N_features)`.** Mirrors
      `LightGBMFraudModel.predict_proba(X: pd.DataFrame)`'s contract;
      column-name validation is built in. ndarray would lose names;
      dict would defer validation to the consumer.
    - **`@log_call` decorator on every public async method.** Async-aware
      via `inspect.iscoroutinefunction` in `utils.logging:444-458`.
      Emits `<qualname>.start` / `.done` / `.failed` events with
      `duration_ms`; the FastAPI route (5.1.d) will inherit per-request
      tracing from these events.
    - **`asyncpg` pool sized 2-10.** At 1650 RPS theoretical max for
      Redis (50-slot pool × 33 RPS/slot — see 5.1.b), Postgres needs
      a smaller pool (the `SELECT 1` probe + future graph queries are
      <10ms each, so 10 slots × 100 RPS/slot = 1000 RPS). Tunable via
      `__init__` kwarg.
    - **Single-attempt + fail-loud, not retry.** The <100ms P95 budget
      doesn't accommodate retries (a single retry doubles tail latency).
      `tenacity` and similar retry libraries are deliberately not used;
      the right answer for sub-100ms-P95 is fail-fast + degraded-mode
      fallback.

Module surface (re-exported from `fraud_engine.api`):
    - FeatureService
    - FeatureVector

Cross-references:
    - `src/fraud_engine/api/redis_store.py` (Sprint 5.1.b) — async
      lifecycle pattern + `@log_call` discipline this class mirrors.
    - `src/fraud_engine/api/schemas.py` (Sprint 5.1.a) —
      `TransactionRequest` field set this service consumes.
    - `src/fraud_engine/features/pipeline.py:84-272` —
      `FeaturePipeline.transform(df)` contract.
    - `src/fraud_engine/data/cleaner.py:273-276` — canonical
      `TransactionDT → timestamp` conversion mirrored here.
    - `models/sprint3/lightgbm_model_manifest.json` —
      `feature_names` (the 743-column canonical model input).
    - `models/pipelines/tier1_pipeline.joblib` — fitted Tier-1
      pipeline artefact loaded at startup.
    - `configs/feature_defaults.yaml` — population-default values.
    - `CLAUDE.md` §3 (production-API stack), §5.5 (logging), §8 (cost
      defaults + latency budget).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Literal

import asyncpg  # type: ignore[import-untyped]  # asyncpg ships no type stubs (PEP-561 absent)
import numpy as np
import pandas as pd
import yaml
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    RedisError,
    TimeoutError as RedisTimeoutError,
)

from fraud_engine.api.redis_store import RedisFeatureStore
from fraud_engine.api.schemas import TransactionRequest
from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.features.pipeline import FeaturePipeline
from fraud_engine.utils.logging import get_logger, log_call

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

_DEFAULT_PIPELINE_DIR: Final[Path] = Path("models/pipelines")
_DEFAULT_PIPELINE_FILENAME: Final[str] = "tier1_pipeline.joblib"
_DEFAULT_MANIFEST_PATH: Final[Path] = Path("models/sprint3/lightgbm_model_manifest.json")
_DEFAULT_DEFAULTS_FILENAME: Final[str] = "feature_defaults.yaml"

# Postgres pool sizing. ~10ms per query × 10 slots = 1000 RPS.
_PG_POOL_MIN: Final[int] = 2
_PG_POOL_MAX: Final[int] = 10
# Per-call SELECT 1 timeout. Aggressive — failures should surface fast.
_PG_PROBE_TIMEOUT_S: Final[float] = 1.0

# Source-status sentinel labels.
_SOURCE_OK: Final[str] = "ok"
_SOURCE_REDIS_DOWN: Final[str] = "redis_down"
_SOURCE_POSTGRES_DOWN: Final[str] = "postgres_down"

# Entity types known to be entity-keyed in Redis. Mirrors the entity
# set documented in Sprint 5.1.b's exploration and `tier4_config.yaml`.
_ENTITY_TYPES: Final[tuple[str, ...]] = (
    "card1",
    "addr1",
    "DeviceInfo",
    "P_emaildomain",
)

# Tier-1 generators expect these columns to exist on the input frame
# (preserve-all-input-columns contract per `BaseFeatureGenerator`).
# Source: Sprint 5.1.a's TransactionRequest field set + cleaner.py:273-276
# `timestamp` derivation.
_TIER1_INPUT_COLS: Final[tuple[str, ...]] = (
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
)

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# FeatureVector — the public output type.
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureVector:
    """Feature vector ready for `model.predict_proba(...)` + degraded-mode metadata.

    Attributes:
        df: pandas DataFrame of shape (1, N_features) with columns in
            the model's canonical order (`feature_names_in_`). Suitable
            for direct `LightGBMFraudModel.predict_proba(df)` call.
        degraded_mode: True iff any source returned a degraded-mode
            fallback (Redis unreachable, Postgres unreachable). Mirrors
            `PredictionResponse.degraded_mode` semantics.
        source_status: Per-source status: keys in {"redis", "postgres"};
            values in {"ok", "redis_down", "postgres_down"}. Logged for
            observability; not surfaced on the API response unless
            Sprint 5.x extends the contract.
    """

    df: pd.DataFrame
    degraded_mode: bool
    source_status: dict[str, str]


# ---------------------------------------------------------------------
# Module-private helpers.
# ---------------------------------------------------------------------


def _load_feature_defaults(path: Path) -> dict[str, Any]:
    """Read and parse the population-defaults YAML.

    Args:
        path: Absolute path to `configs/feature_defaults.yaml`.

    Returns:
        The parsed YAML root mapping with keys `entity_features` and
        `batch_features`.

    Raises:
        FileNotFoundError: If `path` does not exist (loud-fail; better
            than silently using zeros for everything).
        TypeError: If the YAML root is not a mapping.
        ValueError: If required top-level keys are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"FeatureService defaults config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected top-level mapping in {path}, got {type(loaded).__name__}")
    for required in ("entity_features", "batch_features"):
        if required not in loaded or not isinstance(loaded[required], dict):
            raise ValueError(f"feature_defaults.yaml missing or malformed key: {required!r}")
    return loaded


def _load_model_feature_names(path: Path) -> list[str]:
    """Read the model manifest's `feature_names` array (the 743-column list).

    Args:
        path: Absolute path to the LightGBM model manifest JSON.

    Returns:
        The ordered list of feature column names.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the manifest is missing `feature_names` or it
            is not a list of strings.
    """
    if not path.exists():
        raise FileNotFoundError(f"FeatureService model manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    feature_names = manifest.get("feature_names")
    if not isinstance(feature_names, list) or not all(isinstance(n, str) for n in feature_names):
        raise ValueError(f"manifest at {path} missing or malformed 'feature_names' (list[str])")
    return feature_names


def _request_to_dataframe(
    request: TransactionRequest,
    transaction_dt_anchor_iso: str,
) -> pd.DataFrame:
    """Convert a `TransactionRequest` to a single-row DataFrame.

    The output frame carries every column the Tier-1 pipeline expects
    (preserve-input-columns contract), with the group-dicts
    (`vesta_v/c/d/m`, `identity`) flattened into individual columns
    (`V1`, `V2`, ..., `id_01`, etc.). The `timestamp` column is derived
    from `TransactionDT` via the anchor + delta convention used in
    `data.cleaner` (`cleaner.py:273-276`).

    Args:
        request: Validated TransactionRequest from the API ingress.
        transaction_dt_anchor_iso: `Settings.transaction_dt_anchor_iso`
            (default `"2017-12-01T00:00:00+00:00"`).

    Returns:
        Single-row DataFrame ready to feed into
        `tier1_pipeline.transform(...)`.
    """
    # Pydantic v2 model_dump returns the explicit + group-dict fields
    # at the top level. Flatten the group-dicts into individual columns
    # so the cleaner / Tier-1 generators see the same shape they saw at
    # training time.
    raw = request.model_dump()
    # The `metadata` sub-model is not a feature input — drop it.
    raw.pop("metadata", None)
    # Flatten the five group-dicts. Each value is a dict[str, Any];
    # promote its keys into the parent. Empty group-dicts contribute
    # no columns (consistent with cold-start frames).
    flattened: dict[str, Any] = {}
    for key, value in raw.items():
        if key in {"vesta_v", "vesta_c", "vesta_d", "vesta_m", "identity"} and isinstance(
            value, dict
        ):
            flattened.update(value)
        else:
            flattened[key] = value

    df = pd.DataFrame([flattened])

    # Derive `timestamp` per cleaner.py:273-276. The Tier-1
    # `TimeFeatureGenerator` reads `timestamp` (tz-aware UTC), not
    # raw `TransactionDT`.
    anchor = pd.Timestamp(transaction_dt_anchor_iso)
    delta = pd.to_timedelta(df["TransactionDT"], unit="s")
    df["timestamp"] = anchor + delta

    # `isFraud` is the training target. The Tier-1 generators don't
    # consume it but the pipeline's `last_output_dtypes` includes it
    # (training-time artefact). Add as NaN so column presence matches.
    if "isFraud" not in df.columns:
        df["isFraud"] = pd.NA

    return df


# ---------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------


class FeatureService:
    """Orchestrate Tier-1 inline + Redis entity + Postgres batch features.

    Public API:
        - `connect()` / `disconnect()` — pool lifecycle (Redis +
          Postgres).
        - `__aenter__` / `__aexit__` — `async with` support.
        - `get_features(request)` — the hot path; returns
          `FeatureVector` with feature DataFrame + degraded-mode flag.
        - `health_check()` — per-source readiness for Sprint 5.1.d's
          `/ready` endpoint.

    Lifecycle:
        Constructor is cheap and side-effect-free: loads the fitted
        Tier-1 pipeline (joblib), loads the model manifest (for the
        canonical column list), loads the population-defaults YAML.
        `connect()` opens the Redis + Postgres pools and verifies both.
        `disconnect()` closes them.

    Business rationale + trade-offs considered: see module docstring.
    """

    def __init__(  # noqa: PLR0913 — six DI knobs on the constructor are the test surface; folding into a config dataclass would obscure the per-knob default semantics
        self,
        pipeline_dir: Path | None = None,
        model_manifest_path: Path | None = None,
        defaults_config_path: Path | None = None,
        redis_store: RedisFeatureStore | None = None,
        postgres_url: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Configure the service; does NOT open Redis/Postgres pools.

        Args:
            pipeline_dir: Override the path to the fitted Tier-1
                pipeline directory. None → `models/pipelines/`.
            model_manifest_path: Override the model manifest path.
                None → `models/sprint3/lightgbm_model_manifest.json`.
            defaults_config_path: Override the
                `configs/feature_defaults.yaml` path. None → project
                default.
            redis_store: Inject a pre-constructed `RedisFeatureStore`
                (e.g. for tests with fakeredis). None → construct one
                with `Settings`-resolved defaults.
            postgres_url: Override `Settings.postgres_url`. None → use
                Settings.
            settings: Inject a Settings instance (for tests). None →
                `get_settings()`.

        Raises:
            FileNotFoundError: If pipeline / manifest / defaults file
                is missing.
            TypeError / ValueError: On malformed manifest or defaults.
        """
        self._settings: Settings = settings if settings is not None else get_settings()

        # Load fitted Tier-1 pipeline (cheap: joblib of ~4 generators).
        pipeline_dir_resolved = pipeline_dir if pipeline_dir is not None else _DEFAULT_PIPELINE_DIR
        self._pipeline: FeaturePipeline = FeaturePipeline.load(
            pipeline_dir_resolved, _DEFAULT_PIPELINE_FILENAME
        )

        # Load model's canonical feature-name list (the 743-column
        # canonical input order LightGBM was trained on).
        manifest_path = (
            model_manifest_path if model_manifest_path is not None else _DEFAULT_MANIFEST_PATH
        )
        self._feature_names: list[str] = _load_model_feature_names(manifest_path)

        # Load population defaults.
        defaults_path = (
            defaults_config_path
            if defaults_config_path is not None
            else _resolve_config_path(_DEFAULT_DEFAULTS_FILENAME)
        )
        defaults_cfg = _load_feature_defaults(defaults_path)
        self._entity_defaults: dict[str, float] = {
            k: float(v) for k, v in defaults_cfg["entity_features"].items()
        }
        self._batch_defaults: dict[str, float] = {
            k: float(v) for k, v in defaults_cfg["batch_features"].items()
        }

        # Inject-or-construct the Redis store. The injected version is
        # used by tests (fakeredis-backed); production constructs the
        # default which reads from Settings.redis_url.
        self._redis_store: RedisFeatureStore = (
            redis_store if redis_store is not None else RedisFeatureStore()
        )
        self._owns_redis: bool = redis_store is None

        # Postgres URL — resolved at connect() time, not now.
        self._postgres_url: str = (
            postgres_url if postgres_url is not None else self._settings.postgres_url
        )
        self._pg_pool: asyncpg.Pool | None = None

    # ---------- lifecycle ----------------------------------------------

    @log_call
    async def connect(self) -> None:
        """Open the Redis + Postgres pools and verify both.

        Idempotent: a second call while already connected is a no-op
        for whichever sources are already up.

        Raises:
            redis.exceptions.ConnectionError / OSError: If Redis is
                unreachable AND we own the store (i.e. caller didn't
                inject a connected one).
            asyncpg.PostgresError / OSError: If Postgres is unreachable.
        """
        # Connect Redis only if we own the store (the injected variant
        # is presumed already connected by the test fixture).
        if self._owns_redis:
            await self._redis_store.connect()

        # Open Postgres pool. asyncpg.create_pool() runs an immediate
        # connection probe by default — failure raises here.
        if self._pg_pool is None:
            self._pg_pool = await asyncpg.create_pool(
                self._postgres_url,
                min_size=_PG_POOL_MIN,
                max_size=_PG_POOL_MAX,
            )

    @log_call
    async def disconnect(self) -> None:
        """Close pools gracefully. Idempotent."""
        if self._pg_pool is not None:
            await self._pg_pool.close()
            self._pg_pool = None
        # Only disconnect Redis if we own it. The injected store is
        # the test fixture's responsibility.
        if self._owns_redis:
            await self._redis_store.disconnect()

    async def __aenter__(self) -> FeatureService:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.disconnect()

    # ---------- the hot path -------------------------------------------

    @log_call
    async def get_features(self, request: TransactionRequest) -> FeatureVector:
        """Build the feature vector for one transaction.

        Orchestrates: TransactionRequest → single-row DataFrame →
        Tier-1 inline → Redis MGET → Postgres SELECT 1 (stub) →
        defaults fill → reorder to model's canonical column list.

        Args:
            request: Validated TransactionRequest from the API ingress.

        Returns:
            FeatureVector with `df` shaped (1, N_features), `degraded_mode`
            OR-ed across all source flags, and `source_status` per source.
        """
        source_status: dict[str, str] = {"redis": _SOURCE_OK, "postgres": _SOURCE_OK}

        # Step 1: marshall to DataFrame (no I/O).
        df = _request_to_dataframe(request, self._settings.transaction_dt_anchor_iso)

        # Step 2: Tier-1 inline. Always succeeds (no external deps).
        df = self._pipeline.transform(df)

        # Step 3: Entity features from Redis. On connection failure,
        # fill with population defaults + flag.
        entity_values, redis_ok = await self._fetch_entity_features(df)
        if not redis_ok:
            source_status["redis"] = _SOURCE_REDIS_DOWN

        # Step 4: Batch features from Postgres (stubbed: per-call
        # SELECT 1 health probe + return defaults).
        batch_values, postgres_ok = await self._fetch_batch_features()
        if not postgres_ok:
            source_status["postgres"] = _SOURCE_POSTGRES_DOWN

        # Step 5: Assemble final feature dict + reorder to model's
        # canonical column list. Anything still missing falls back to
        # the entity-default (which is 0.0 for typical entity features).
        feat_dict = self._build_feature_dict(df, entity_values, batch_values)
        out_df = self._to_model_dataframe(feat_dict)

        degraded = source_status["redis"] != _SOURCE_OK or source_status["postgres"] != _SOURCE_OK
        return FeatureVector(df=out_df, degraded_mode=degraded, source_status=source_status)

    # ---------- per-source helpers -------------------------------------

    async def _fetch_entity_features(
        self,
        df: pd.DataFrame,
    ) -> tuple[dict[str, Any], bool]:
        """Look up entity features from Redis.

        For each entity type in `_ENTITY_TYPES` that has a non-null
        value in the request frame, build a list of `feat:<entity>:<id>:<feature>`
        keys (one per known entity-feature). Issue a single MGET via
        `RedisFeatureStore.get_multi`. On `ConnectionError` / `TimeoutError` /
        `RedisError` / `OSError`, return `({}, False)` — the caller
        applies defaults.

        Args:
            df: The post-Tier-1 DataFrame (carries entity columns).

        Returns:
            Tuple of (values_by_feature_name, redis_ok).
            `values_by_feature_name` is `dict[str, Any]` keyed by the
            feature name (e.g. `card1_velocity_24h`); `redis_ok` is False
            iff Redis was unreachable.
        """
        # Build the (entity_type, entity_id, feature_name) tuples for
        # the known entity-feature surface. The `feature_names_in_` list
        # carries entries like `card1_velocity_24h` / `addr1_amt_mean_30d`
        # / `card1_v_ewm_lambda_0.05`. We iterate model-feature-names
        # and route by entity-prefix to reduce explicit knowledge.
        keys_to_features: dict[str, str] = {}  # redis_key → feature_name
        for feature_name in self._feature_names:
            entity_type = _entity_type_for_feature(feature_name)
            if entity_type is None:
                continue
            entity_id_value = df[entity_type].iloc[0]
            if pd.isna(entity_id_value):
                # Entity is null on this request — skip (defaults will
                # fill at assembly time, and that's correct: no entity →
                # no Redis state to look up).
                continue
            entity_id = str(entity_id_value)
            redis_key = self._redis_store.make_key(entity_type, entity_id, feature_name)
            keys_to_features[redis_key] = feature_name

        if not keys_to_features:
            return {}, True

        try:
            raw = await self._redis_store.get_multi(list(keys_to_features.keys()))
        except (RedisConnectionError, RedisTimeoutError, RedisError, OSError, RuntimeError) as exc:
            # `RuntimeError` covers the "store not connected" state —
            # equivalent to Redis-unreachable from the orchestrator's
            # perspective (the production failure mode "connect()
            # raised at startup, store is in a half-initialised state"
            # surfaces here as RuntimeError, not a Redis exception).
            _logger.warning(
                "feature_service.redis_down",
                error=type(exc).__name__,
                detail=str(exc),
            )
            return {}, False

        # Map redis_key → feature_name → value, dropping None values
        # (cold-start: defaults will fill those naturally).
        values: dict[str, Any] = {}
        for redis_key, feature_name in keys_to_features.items():
            value = raw.get(redis_key)
            if value is not None:
                values[feature_name] = value
        return values, True

    async def _fetch_batch_features(self) -> tuple[dict[str, Any], bool]:
        """Probe Postgres + return batch features (stubbed for 5.1.c).

        Sprint 5.1.c stubs the actual `SELECT pagerank_score, ... FROM
        feature_batch_graph WHERE entity_id IN (...)` — the table
        doesn't exist yet, Sprint 5.x's batch loader builds it.

        What this method DOES do: a per-call `SELECT 1` health probe
        against the Postgres pool. On success, returns the population
        defaults (no actual graph data). On failure (`asyncpg.PostgresError`
        or `OSError`), returns the same defaults BUT with `postgres_ok=False`,
        which propagates `degraded_mode=True` to the caller.

        This makes the degraded-mode contract testable now (Postgres
        up vs down distinguishable) without designing the schema. When
        5.x replaces the probe with the real query, the failure-mode
        contract is unchanged.

        Returns:
            Tuple of (batch_defaults, postgres_ok). `batch_defaults`
            is the population-default dict from `feature_defaults.yaml`'s
            `batch_features` section, expanded to per-feature-column
            (e.g. `entity_degree_card1` → `entity_degree_default`).
        """
        # Build the batch-default dict. Hand-crafted entity-degree
        # columns (one per entity type) all read from the
        # `entity_degree_default` value.
        batch_values: dict[str, Any] = {}
        for feature_name in self._feature_names:
            if feature_name in self._batch_defaults:
                batch_values[feature_name] = self._batch_defaults[feature_name]
            elif feature_name.startswith("entity_degree_"):
                batch_values[feature_name] = self._batch_defaults["entity_degree_default"]

        # Probe the Postgres pool. A None pool means we're not connected;
        # treat as down (caller's `connect()` should have run first).
        if self._pg_pool is None:
            return batch_values, False

        try:
            async with self._pg_pool.acquire(timeout=_PG_PROBE_TIMEOUT_S) as conn:
                await conn.fetchval("SELECT 1")
        except (asyncpg.PostgresError, OSError, TimeoutError) as exc:
            _logger.warning(
                "feature_service.postgres_down",
                error=type(exc).__name__,
                detail=str(exc),
            )
            return batch_values, False

        return batch_values, True

    # ---------- assembly + defaults ------------------------------------

    def _build_feature_dict(
        self,
        df: pd.DataFrame,
        entity_values: dict[str, Any],
        batch_values: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine Tier-1 frame + Redis values + Postgres values into one dict.

        Order of precedence (later overrides earlier):
            1. Tier-1 columns from the post-pipeline DataFrame.
            2. Entity values from Redis (for keys that came back).
            3. Batch values from Postgres / defaults.

        Anything still missing after this is filled by `_to_model_dataframe`
        with the entity-feature default (0.0 by convention).
        """
        feat: dict[str, Any] = {}
        # Step 1: Tier-1 columns. The `iloc[0]` extracts the single
        # row's value for each column.
        for col in df.columns:
            feat[col] = df[col].iloc[0]
        # Step 2: entity values override anything from Tier-1.
        feat.update(entity_values)
        # Step 3: batch values.
        feat.update(batch_values)
        return feat

    def _to_model_dataframe(self, feat_dict: dict[str, Any]) -> pd.DataFrame:
        """Build the (1, N_features) float64 DataFrame in the model's column order.

        Missing features (not in `feat_dict` after the assembly path)
        are filled with the entity-features default (0.0 by convention).
        Extra columns in `feat_dict` are dropped silently — the model
        only consumes its known feature names.

        Builds the row as a single contiguous `np.float64` array, NOT
        a `pd.DataFrame.apply(pd.to_numeric)` over 743 columns. The
        per-column apply adds ~50–80 ms of overhead at request time;
        the direct float-array build is ~1 ms. LightGBM rejects
        `object`-dtype columns at predict time (`ValueError: pandas
        dtypes must be int, float or bool`); the request payload's
        `None` values for nullable fields (e.g. `dist1=None`,
        `V137=None`, `R_emaildomain=None`) propagate as `None` through
        the assembly path. Coercing inline translates None → NaN
        (LightGBM's native missing-value sentinel); any non-numeric
        residue (a stray string from a misbehaving feature generator)
        also becomes NaN with `try/except float(...)` rather than a
        deep-stack error.

        Trade-offs:
            - Building the full float64 row directly is ~80× faster
              than `df.apply(pd.to_numeric, errors="coerce")` for the
              743-column request-time path, while preserving the same
              "None → NaN, non-numeric → NaN" semantic. Tested under
              Sprint 5.1.f's p95-budget gate.
            - Booleans coerce to 0.0/1.0 via `float(True)` / `float(False)`
              (Python language guarantee). LightGBM accepts both bool
              and float dtype, so we'd be free to keep bool here, but
              homogenising to float64 lets us use a single contiguous
              numpy array — no per-column dtype dance.
        """
        default = self._entity_defaults["default"]
        row = np.empty(len(self._feature_names), dtype=np.float64)
        for i, name in enumerate(self._feature_names):
            value = feat_dict.get(name, default)
            if value is None:
                row[i] = np.nan
                continue
            try:
                row[i] = float(value)
            except (TypeError, ValueError):
                # Non-numeric residue (rare; e.g. a stray string from a
                # feature generator). LightGBM treats NaN as missing.
                row[i] = np.nan
        return pd.DataFrame(row.reshape(1, -1), columns=list(self._feature_names))

    # ---------- health check -------------------------------------------

    async def health_check(self) -> dict[str, Literal["ok", "degraded", "unreachable"]]:
        """Per-source readiness for Sprint 5.1.d's `/ready` endpoint.

        Probes Redis (`PING`) and Postgres (`SELECT 1`); returns
        `"ok"` for each reachable source, `"unreachable"` otherwise.
        Never raises.
        """
        result: dict[str, Literal["ok", "degraded", "unreachable"]] = {}

        # Redis probe.
        try:
            # Use the underlying client's ping() directly to avoid
            # dragging the get_multi path's exception surface into
            # the readiness probe.
            client = self._redis_store._client  # noqa: SLF001 — readiness needs the live client
            if client is None:
                result["redis"] = "unreachable"
            else:
                await client.ping()
                result["redis"] = "ok"
        except (RedisConnectionError, RedisTimeoutError, RedisError, OSError):
            result["redis"] = "unreachable"

        # Postgres probe.
        if self._pg_pool is None:
            result["postgres"] = "unreachable"
        else:
            try:
                async with self._pg_pool.acquire(timeout=_PG_PROBE_TIMEOUT_S) as conn:
                    await conn.fetchval("SELECT 1")
                result["postgres"] = "ok"
            except (asyncpg.PostgresError, OSError, TimeoutError):
                result["postgres"] = "unreachable"

        return result


# ---------------------------------------------------------------------
# Module-private path resolution.
# ---------------------------------------------------------------------


def _resolve_config_path(filename: str) -> Path:
    """Resolve `configs/{filename}` relative to the repo root.

    Mirrors the pattern in `features.tier4_decay._resolve_config_path`
    and `api.redis_store._resolve_config_path` — repo root is four
    `parents[]` up from this file.
    """
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "configs" / filename


def _entity_type_for_feature(feature_name: str) -> str | None:
    """Return the entity type prefix for a feature name, or None.

    Examples:
        `card1_velocity_24h` → `"card1"`
        `addr1_amt_mean_30d` → `"addr1"`
        `card1_v_ewm_lambda_0.05` → `"card1"`
        `pagerank_score` → None (graph feature, not entity-keyed)
        `log_amount` → None (Tier-1 feature, not entity-keyed)

    The check is a simple prefix match against `_ENTITY_TYPES`. We
    deliberately don't try to parse the feature name's structure
    (entity_velocity_window vs entity_v_ewm_lambda_λ) — the prefix is
    sufficient signal.
    """
    for entity_type in _ENTITY_TYPES:
        if feature_name.startswith(f"{entity_type}_"):
            return entity_type
    return None


__all__ = ["FeatureService", "FeatureVector"]
