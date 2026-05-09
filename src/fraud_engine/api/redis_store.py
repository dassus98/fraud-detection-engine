"""Redis-backed online feature store for the fraud-scoring API.

Sprint 5 prompt 5.1.b: the async client over Redis that the FastAPI
route will call at request time to fetch entity-keyed features
(Tier-2 velocity counters, Tier-3 behavioural deviations, Tier-4 EWM
running state, Tier-5 graph metrics). This module ships the
*primitive*; Sprint 5.1.c wires it into the route, Sprint 5.x adds the
offline batch loader that populates Redis from training-set state.

Business rationale:
    Stateful features cannot be recomputed at request time from the
    inbound transaction alone — they depend on the entity's full
    history (e.g. `card1`'s last 24 hours of transactions, or its
    EWM-decay state under three different λ values). Reading those
    values from Redis at <5ms MGET keeps the project's <100ms P95
    end-to-end budget intact (CLAUDE.md §3) while letting Model A
    consume the full feature set instead of Tier-1 fallback.

    Without this primitive, the API has only two options:
      (a) Recompute from the request body alone — impossible by
          construction; the request carries one transaction, not the
          entity's history.
      (b) Read state from parquet at request time — ~50ms minimum
          before any inference runs; blows the budget and forces
          Tier-1 fallback for every request.

    The Redis store closes that gap. Sprint 5.1.c builds the
    request-side consumer, Sprint 5.x ships the offline producer,
    Sprint 5.2 wires the Redis-unreachable fallback that drives the
    `degraded_mode=True` flag on `PredictionResponse`.

Trade-offs considered:
    - **JSON serialisation over msgpack.** ~3× larger on the wire
      (~50 B vs ~15 B for the canonical `_DecayState`-shaped value)
      but human-readable under `redis-cli MONITOR` for incident
      triage, zero new runtime deps. Worst-case Redis footprint at
      full population is ~30-40 MB (4 entity types × ~13.5K unique
      IDs × ~14 features × ~50 B/value); the size penalty is
      invisible. Round-trip cost ~25 µs per 50-key MGET — <0.1% of
      the post-MGET budget. Flagged as a Sprint 5.x optimisation if
      profiling shows `json.loads` is non-trivial.
    - **`fakeredis` for unit tests, real Redis for integration.**
      Unit tests run with no Docker dep (deterministic, CI-friendly).
      Bonus integration test connects to `Settings.redis_url` and
      `pytest.skip`s if unreachable; it catches any fakeredis vs
      real-Redis behavioural drift. The known fakeredis gaps (Lua
      subtleties, pubsub edge cases, sub-ms expiry precision) don't
      touch the four operations this class uses (PING, MGET, SETEX,
      aclose). `unittest.mock.AsyncMock` was rejected — it would
      mock the very contract the test is meant to verify.
    - **Glob-pattern TTL config (first-match-wins) over exact-match
      dict or regex.** Exact-match would enumerate all 54 features
      verbatim and break when Sprint 4.x adds features; regex is
      overkill. `fnmatch` handles every realistic case naturally
      and reads cleanly in YAML. The TTL math (7d = ~12 half-lives
      at λ=0.05/h → state ≈ 0) is documented in three places: the
      YAML's per-line comments, this docstring, and the completion
      report.
    - **Connection pool size = 50.** At ~30ms per MGET, 50 slots ≈
      1650 RPS theoretical max — comfortable for the project's
      economic-eval baseline (1M txns/month ≈ 0.4 RPS sustained).
      Tunable via `__init__(max_connections=...)`.
    - **`decode_responses=False` on the pool.** Bytes in, bytes out.
      Keep JSON decoding explicit in code so the wire payload is
      unambiguous and test fixtures don't need to coerce string vs
      bytes. The alternative (`decode_responses=True`) would give
      `str` from `MGET` but would conflict with binary value formats
      (e.g. msgpack) if a future migration switches serialisation.
    - **`allow_nan=True` on `json.dumps`.** EWM math is bounded so
      true NaN should be a code-level invariant violation, not a
      payload condition. Pinning `allow_nan=True` lets a bug-NaN
      round-trip through Redis and surface in `redis-cli MONITOR`
      for triage rather than silently failing the write.
    - **Atomicity per key, not across keys.** `write_entity_features`
      pipelines per-feature SETEX commands but does NOT wrap them
      in MULTI/EXEC. The offline batch loader (its only Sprint
      5.1.b consumer) is read-then-overwrite per entity; partial
      writes on crash are recoverable on the next batch run.
      Online per-request EWM updates (read-decay-write within a
      single round-trip) require atomicity that this method does
      not provide; that path is a separate Sprint 5.x Lua-script
      primitive.
    - **`__aenter__` / `__aexit__` context-manager.** Pool leaks
      block process shutdown. `async with RedisFeatureStore() as
      store: ...` guarantees `disconnect()` runs on every exit
      path, including exception. Sprint 5.1.c will use FastAPI's
      `lifespan` context which honours the same protocol.

Module surface (re-exported from `fraud_engine.api`):
    - RedisFeatureStore

Cross-references:
    - `configs/redis_feature_store.yaml` — TTL pattern map; load-bearing
      rationale per pattern.
    - `src/fraud_engine/features/tier4_decay.py:332-358` — `_DecayState`
      dataclass; the most operationally critical value this store
      will eventually carry (`{"last_t": int, "v": float, "fraud_v":
      float}` shape).
    - `src/fraud_engine/utils/logging.py:444-458` — `@log_call` async
      support; emits `<qualname>.start` / `.done` / `.failed` with
      `duration_ms`.
    - `src/fraud_engine/config/settings.py:redis_url` — the URL this
      module reads when no override is supplied.
    - `CLAUDE.md` §3 (production-API stack), §5.5 (logging discipline).
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Final

import redis.asyncio as aioredis
import yaml
from redis.asyncio.client import Redis
from redis.asyncio.connection import ConnectionPool

from fraud_engine.config.settings import get_settings
from fraud_engine.utils.logging import get_logger, log_call

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Pool sizing. 50 slots × ~30 ms per MGET ≈ 1650 RPS theoretical max.
_DEFAULT_MAX_CONNECTIONS: Final[int] = 50

# Fallback default TTL if the YAML omits `default_ttl_seconds`.
# Mirrors the YAML's documented default.
_DEFAULT_TTL_FALLBACK_SECONDS: Final[int] = 604800  # 7 days

# Filename of the per-feature TTL YAML in `configs/`.
_TTL_CONFIG_FILENAME: Final[str] = "redis_feature_store.yaml"

# Validator regex for `entity_type` and `feature_name`. Colons are
# reserved by the key schema (`feat:{entity_type}:{entity_id}:{feature_name}`);
# any colon in a component would let a malformed input collide with
# a real key. Letters / digits / underscore / dot / dash only.
# Length cap defensive (longest realistic feature name is
# `card1_fraud_v_ewm_lambda_0.05` at 28 chars; 128 is generous).
_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")

# Key-schema prefix. Mirrored verbatim in `make_key`.
_KEY_PREFIX: Final[str] = "feat"

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Module-private helpers (YAML loader; mirrors the
# `features.tier4_decay._resolve_config_path` / `_load_yaml` pattern).
# ---------------------------------------------------------------------


def _resolve_config_path(filename: str) -> Path:
    """Resolve `configs/{filename}` relative to the repo root.

    Mirrors the pattern in `features.tier4_decay._resolve_config_path`.
    The repo root is four `parents[]` up from this file
    (`api/redis_store.py` → `api/` → `fraud_engine/` → `src/` → root).
    """
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "configs" / filename


def _load_ttl_config(path: Path) -> dict[str, Any]:
    """Read and parse the per-feature TTL YAML.

    Args:
        path: Absolute path to the YAML file.

    Returns:
        The parsed YAML root mapping.

    Raises:
        FileNotFoundError: If `path` does not exist (loud-fail; better
            than silently using defaults that may not match the
            deployment's intent).
        TypeError: If the YAML root is not a mapping.
    """
    if not path.exists():
        raise FileNotFoundError(f"RedisFeatureStore TTL config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected top-level mapping in {path}, got {type(loaded).__name__}")
    return loaded


def _parse_ttl_patterns(raw: object) -> list[tuple[str, int]]:
    """Normalise the YAML's `ttl_by_pattern` list into (glob, ttl) tuples.

    Args:
        raw: The value of the YAML's `ttl_by_pattern` key (expected
            to be a list of `{pattern, ttl_seconds}` mappings).

    Returns:
        List of `(pattern, ttl_seconds)` tuples in declaration order.
        Empty list if `raw` is an empty list.

    Raises:
        ValueError: If the list shape is malformed (non-list, non-dict
            entry, missing key, empty pattern, non-positive ttl).
    """
    if not isinstance(raw, list):
        raise ValueError(f"ttl_by_pattern must be a list, got {type(raw).__name__}")
    parsed: list[tuple[str, int]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"ttl_by_pattern entry must be a mapping, got {entry!r}")
        try:
            pattern = entry["pattern"]
            ttl = entry["ttl_seconds"]
        except KeyError as exc:
            raise ValueError(f"ttl_by_pattern entry missing key: {exc}") from exc
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"ttl_by_pattern.pattern must be non-empty str, got {pattern!r}")
        if not isinstance(ttl, int) or ttl <= 0:
            raise ValueError(f"ttl_by_pattern.ttl_seconds must be positive int, got {ttl!r}")
        parsed.append((pattern, ttl))
    return parsed


# ---------------------------------------------------------------------
# Public class.
# ---------------------------------------------------------------------


class RedisFeatureStore:
    """Async Redis client for online entity-feature lookup.

    Public API:
        - `connect()` / `disconnect()` — pool lifecycle.
        - `__aenter__` / `__aexit__` — `async with` support.
        - `make_key(entity_type, entity_id, feature_name)` — canonical
          schema `feat:{entity_type}:{entity_id}:{feature_name}`.
        - `ttl_for(feature_name)` — glob-pattern lookup against the
          YAML config.
        - `get_multi(keys)` — MGET in one round-trip; returns a dict
          keyed by input keys, with `None` for missing keys.
        - `write_entity_features(entity_type, entity_id, features)` —
          pipelined SETEX with per-feature TTL.

    Lifecycle:
        Constructor is cheap and side-effect-free. `connect()` opens
        the pool and runs `PING`. `disconnect()` closes the pool.
        `async with` wraps both. The FastAPI route (Sprint 5.1.c)
        will hold a single instance for the process lifetime, opened
        in `lifespan` startup, closed in `lifespan` shutdown.

    Key schema:
        `feat:{entity_type}:{entity_id}:{feature_name}`
        Example: `feat:card1:13926:velocity_24h`
        Example: `feat:DeviceInfo:samsung_sm_g930v:v_ewm_lambda_0.05`

    Business rationale + trade-offs considered: see module docstring.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_config_path: Path | None = None,
        max_connections: int = _DEFAULT_MAX_CONNECTIONS,
        default_ttl_seconds: int | None = None,
    ) -> None:
        """Configure the store; does NOT open the pool.

        Args:
            redis_url: Override `Settings.redis_url`. None → use Settings.
            ttl_config_path: Override the canonical YAML at
                `configs/redis_feature_store.yaml`. None → use the
                project default. Raises if the file is missing.
            max_connections: Pool size cap. Default 50.
            default_ttl_seconds: Override the YAML's
                `default_ttl_seconds`. None → use YAML.

        Raises:
            FileNotFoundError: If the resolved TTL YAML is missing.
            TypeError: If the YAML root is not a mapping.
            ValueError: If `ttl_by_pattern` entries are malformed.
        """
        settings = get_settings()
        self._redis_url: str = redis_url if redis_url is not None else settings.redis_url
        self._max_connections: int = max_connections

        cfg_path = (
            ttl_config_path
            if ttl_config_path is not None
            else _resolve_config_path(_TTL_CONFIG_FILENAME)
        )
        ttl_cfg = _load_ttl_config(cfg_path)
        self._default_ttl_seconds: int = (
            default_ttl_seconds
            if default_ttl_seconds is not None
            else int(ttl_cfg.get("default_ttl_seconds", _DEFAULT_TTL_FALLBACK_SECONDS))
        )
        # First-match-wins evaluation in `ttl_for`; declaration order
        # is the priority order.
        self._ttl_by_pattern: list[tuple[str, int]] = _parse_ttl_patterns(
            ttl_cfg.get("ttl_by_pattern", [])
        )

        # Pool + client are populated by `connect()`. None here so a
        # method called pre-connect raises a clear RuntimeError instead
        # of an obscure AttributeError.
        self._pool: ConnectionPool | None = None
        self._client: Redis | None = None

    # ---------- lifecycle ----------------------------------------------

    @log_call
    async def connect(self) -> None:
        """Open the connection pool and verify connectivity via PING.

        Idempotent: a second call while already connected is a no-op.

        Raises:
            redis.exceptions.ConnectionError: If Redis is unreachable.
                Lets Sprint 5.1.c's readiness probe surface this as
                `ReadyResponse.checks["redis"] = "unreachable"`.
        """
        if self._pool is not None:
            return
        self._pool = ConnectionPool.from_url(
            self._redis_url,
            max_connections=self._max_connections,
            decode_responses=False,  # bytes in, bytes out — JSON-decode in code
        )
        self._client = aioredis.Redis(connection_pool=self._pool)
        await self._client.ping()

    @log_call
    async def disconnect(self) -> None:
        """Close the pool gracefully.

        Idempotent — second call is a no-op. `__aexit__` always calls
        this so a leaked pool can't survive an exception path.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._pool is not None:
            await self._pool.aclose()
            self._pool = None

    async def __aenter__(self) -> RedisFeatureStore:
        """Open the pool on context entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the pool on context exit, even if an exception escapes."""
        await self.disconnect()

    # ---------- key schema + TTL helpers --------------------------------

    def make_key(
        self,
        entity_type: str,
        entity_id: str | int,
        feature_name: str,
    ) -> str:
        """Build the canonical Redis key for an entity-feature pair.

        Schema: `feat:{entity_type}:{entity_id}:{feature_name}`.

        Args:
            entity_type: One of `card1`, `addr1`, `DeviceInfo`,
                `P_emaildomain` (validated against the no-colon
                character class — `_NAME_RE`).
            entity_id: Entity value. Coerced to `str` via `str(...)`.
                The entity_id is NOT validated against `_NAME_RE`
                because user-data entity IDs (e.g. email domains
                containing dots, device strings containing spaces)
                may legitimately carry characters outside the
                feature-name character class. The key-prefix itself
                (`feat:`) and the static `entity_type` + `feature_name`
                slots are validated; entity_id is the only free-form
                slot, and its content is opaque to Redis.
            feature_name: Feature column name (e.g. `velocity_24h`,
                `v_ewm_lambda_0.05`). Validated against `_NAME_RE`.

        Returns:
            The fully-qualified Redis key.

        Raises:
            ValueError: If `entity_type` or `feature_name` contain a
                colon or any character outside `[A-Za-z0-9_.-]`, or
                are empty / over 128 chars.
        """
        self._validate_name("entity_type", entity_type)
        self._validate_name("feature_name", feature_name)
        return f"{_KEY_PREFIX}:{entity_type}:{entity_id}:{feature_name}"

    def ttl_for(self, feature_name: str) -> int:
        """Resolve the TTL (in seconds) for a feature name.

        Walks the YAML's `ttl_by_pattern` list in declaration order,
        applying `fnmatch.fnmatch` against each pattern. The first
        matching pattern's `ttl_seconds` is returned. Falls through to
        `self._default_ttl_seconds` if no pattern matches.

        Args:
            feature_name: e.g. `velocity_24h`, `v_ewm_lambda_0.05`,
                `pagerank_score`.

        Returns:
            TTL in seconds. Always > 0 (the YAML loader rejects
            non-positive values at construction time).
        """
        for pattern, ttl in self._ttl_by_pattern:
            if fnmatch.fnmatch(feature_name, pattern):
                return ttl
        return self._default_ttl_seconds

    @staticmethod
    def _validate_name(field: str, value: str) -> None:
        """Enforce the no-colon character class on schema components."""
        if not _NAME_RE.fullmatch(value):
            raise ValueError(
                f"RedisFeatureStore: {field}={value!r} must match "
                f"{_NAME_RE.pattern!r} (no colons; alnum/_/./- only; 1-128 chars)"
            )

    # ---------- batch read ----------------------------------------------

    @log_call
    async def get_multi(self, keys: Sequence[str]) -> dict[str, Any]:
        """Read multiple keys in a single `MGET` round-trip.

        Args:
            keys: Sequence of fully-qualified Redis keys (build via
                `make_key`).

        Returns:
            Dict keyed by the input keys (insertion order preserved).
            Value is the JSON-decoded payload, or `None` for keys not
            present in Redis.

        Raises:
            RuntimeError: If `connect()` has not been called.
            json.JSONDecodeError: If a stored value is not valid JSON
                (would indicate a writer-side bug — fail loud).
        """
        if self._client is None:
            raise RuntimeError("RedisFeatureStore: call connect() before get_multi()")
        if not keys:
            return {}
        raw_values: list[bytes | None] = await self._client.mget(*keys)
        result: dict[str, Any] = {}
        for key, raw in zip(keys, raw_values, strict=True):
            result[key] = None if raw is None else json.loads(raw)
        return result

    # ---------- entity-feature write ------------------------------------

    @log_call
    async def write_entity_features(
        self,
        entity_type: str,
        entity_id: str | int,
        features: Mapping[str, Any],
    ) -> None:
        """Write a batch of features for one entity, each with its own TTL.

        Pipelined SETEX so all features for the entity ship in one
        round-trip. Per-feature TTL is resolved via `ttl_for`. Atomicity
        is per-key, NOT across keys — partial-write recovery is the
        caller's responsibility (acceptable for the offline batch
        loader; online per-request EWM updates need a separate Lua
        primitive that's out of scope for 5.1.b).

        Intended caller: the offline batch loader (Sprint 5.x), called
        once per entity at training-data ingestion time.

        Args:
            entity_type: `card1` / `addr1` / `DeviceInfo` /
                `P_emaildomain`.
            entity_id: Entity value (coerced to `str`).
            features: Map of `feature_name -> JSON-serialisable value`.
                Mixed types fine (int, float, str, None, dict, list).
                NaN / Infinity round-trip via `allow_nan=True`.

        Raises:
            RuntimeError: If `connect()` has not been called.
            ValueError: If `entity_type` or any feature name fails
                `_validate_name`.
            TypeError: If any feature value is not JSON-serialisable
                (e.g. a `set`).
        """
        if self._client is None:
            raise RuntimeError("RedisFeatureStore: call connect() before write_entity_features()")
        if not features:
            return
        async with self._client.pipeline(transaction=False) as pipe:
            for feature_name, value in features.items():
                key = self.make_key(entity_type, entity_id, feature_name)
                ttl = self.ttl_for(feature_name)
                payload = json.dumps(value, allow_nan=True)
                pipe.setex(key, ttl, payload)
            await pipe.execute()


__all__ = ["RedisFeatureStore"]
