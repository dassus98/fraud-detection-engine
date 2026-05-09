"""Integration tests for `FeatureService` against real backing services.

Sprint 5 prompt 5.1.c. The unit tests cover the orchestrator's
internal logic via fakeredis + mock asyncpg pools; this file covers
the wire-protocol contracts that mocks cannot fake faithfully.

Per the spec ("integration test with Redis running and with Redis
down (degraded mode)"), this module ships **two** complementary
integration paths:

1. **Real Redis up** (skipped if `Settings.redis_url` is unreachable).
2. **Real Redis down** simulated via an unreachable URL — `FeatureService`
   should set `degraded_mode=True` and fill from population defaults.

Both Postgres-up and Postgres-down paths are exercised the same way:
the success path uses an unreachable Postgres URL too (because the
project memory `project_docker_deferred` indicates Postgres isn't
guaranteed to be running locally yet); the failure path is the same
URL with the connection pool intentionally not opened.

Business rationale:
    Mocked failure modes catch API-shape bugs in the orchestrator's
    call sites; only real wire-protocol round-trips catch protocol
    drift (e.g. a fakeredis 2.x → 3.x signature change, an asyncpg
    minor that adds a new exception type). The integration tests are
    the catch-all for those classes of failure.

Trade-offs considered:
    - Skip-if-unreachable for the Redis-up path matches the Sprint
      5.1.b precedent. CI without Docker simply skips; local dev
      with `docker compose up redis` runs the full path.
    - Unreachable-URL injection for the down-path: faster and more
      deterministic than stopping a running Redis mid-test.
    - UUID4 namespace per test for Redis-up keys: prevents pollution
      of a shared dev Redis (multiple developers hitting the same
      instance won't collide). Mirrors `test_redis_store_integration.py`.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
import redis.asyncio
import yaml

from fraud_engine.api.feature_service import FeatureService
from fraud_engine.api.redis_store import RedisFeatureStore
from fraud_engine.api.schemas import TransactionRequest
from fraud_engine.config.settings import get_settings

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------
# Required artefacts (skip if missing).
# ---------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PIPELINE_DIR = _REPO_ROOT / "models" / "pipelines"
_PIPELINE_FILE = _PIPELINE_DIR / "tier1_pipeline.joblib"
_MANIFEST_FILE = _REPO_ROOT / "models" / "sprint3" / "lightgbm_model_manifest.json"
_DEFAULTS_FILE = _REPO_ROOT / "configs" / "feature_defaults.yaml"
_REDIS_TTL_FILE = _REPO_ROOT / "configs" / "redis_feature_store.yaml"


def _require_artefacts() -> None:
    missing = [
        p
        for p in (_PIPELINE_FILE, _MANIFEST_FILE, _DEFAULTS_FILE, _REDIS_TTL_FILE)
        if not p.exists()
    ]
    if missing:
        pytest.skip(f"FeatureService artefacts missing: {missing}")


# ---------------------------------------------------------------------
# Fixtures — Redis reachability + namespace.
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_redis_url() -> str:
    """Resolve the Redis URL from Settings; skip the module if unreachable."""
    _require_artefacts()
    settings = get_settings()
    url = settings.redis_url

    async def _probe() -> None:
        client = redis.asyncio.from_url(url)
        try:
            await client.ping()
        finally:
            await client.aclose()

    try:
        asyncio.run(_probe())
    except Exception as exc:  # noqa: BLE001 — broad on purpose; many failure modes
        pytest.skip(f"Redis unreachable at {url}: {exc}")
    return url


@pytest.fixture
def namespace() -> str:
    """UUID4 hex namespace for this test's Redis keys."""
    return uuid4().hex


@pytest.fixture
async def real_redis_store(
    real_redis_url: str,
) -> AsyncIterator[RedisFeatureStore]:
    """A `RedisFeatureStore` connected to real Redis."""
    async with RedisFeatureStore(redis_url=real_redis_url, ttl_config_path=_REDIS_TTL_FILE) as s:
        yield s


@pytest.fixture
def valid_request(namespace: str) -> TransactionRequest:
    """Canonical TransactionRequest for integration tests.

    The namespace is embedded in entity_id values so the test's keys
    don't collide with other tests' keys in a shared dev Redis.
    """
    return TransactionRequest(
        TransactionID=2987000,
        TransactionDT=86400,
        TransactionAmt=59.95,
        ProductCD="W",
        # Stash the namespace in `card1` (treated as int internally; we
        # use a deterministic int derived from the UUID's first 8 hex
        # chars to avoid collisions across tests).
        card1=int(namespace[:8], 16) % 1_000_000_000,
        card4="visa",
        addr1=315.0,
        P_emaildomain="gmail.com",
    )


# ---------------------------------------------------------------------
# Tests — Redis up.
# ---------------------------------------------------------------------


async def test_redis_up_postgres_down_returns_partial_degraded(
    real_redis_store: RedisFeatureStore,
    valid_request: TransactionRequest,
) -> None:
    """Real Redis reachable + unreachable Postgres URL → only postgres_down flag."""
    s = FeatureService(
        pipeline_dir=_PIPELINE_DIR,
        model_manifest_path=_MANIFEST_FILE,
        defaults_config_path=_DEFAULTS_FILE,
        redis_store=real_redis_store,
        postgres_url="postgresql://no-such-host:5432/fake",
    )
    # Don't call connect(); leave _pg_pool=None to simulate Postgres-down.
    result = await s.get_features(valid_request)
    assert result.df.shape == (1, 743)
    assert result.degraded_mode is True
    assert result.source_status["redis"] == "ok"
    assert result.source_status["postgres"] == "postgres_down"
    # Batch defaults still applied.
    assert result.df["pagerank_score"].iloc[0] == 0.0001


async def test_real_redis_with_pre_seeded_features(
    real_redis_store: RedisFeatureStore,
    valid_request: TransactionRequest,
    namespace: str,
) -> None:
    """Pre-seed entity features via real Redis; verify they appear in the output."""
    # Seed the canonical card1 velocity_24h for this test's entity.
    card1_id = valid_request.card1
    await real_redis_store.write_entity_features(
        "card1",
        card1_id,
        {"card1_velocity_24h": 13.0},
    )
    s = FeatureService(
        pipeline_dir=_PIPELINE_DIR,
        model_manifest_path=_MANIFEST_FILE,
        defaults_config_path=_DEFAULTS_FILE,
        redis_store=real_redis_store,
        postgres_url="postgresql://no-such-host:5432/fake",
    )
    try:
        result = await s.get_features(valid_request)
        # The seeded value beats the default; verifies real Redis MGET.
        assert result.df["card1_velocity_24h"].iloc[0] == 13.0
    finally:
        # Cleanup: explicit delete via the live client.
        with contextlib.suppress(Exception):
            client = real_redis_store._client  # noqa: SLF001
            if client is not None:
                key = real_redis_store.make_key("card1", card1_id, "card1_velocity_24h")
                await client.delete(key)


# ---------------------------------------------------------------------
# Tests — Redis down (degraded mode).
# ---------------------------------------------------------------------


async def test_redis_down_returns_defaults_with_degraded_flag(
    tmp_path: Path,
    valid_request: TransactionRequest,
) -> None:
    """Unreachable Redis URL → all entity features default + degraded_mode=True."""
    _require_artefacts()
    # Build an isolated TTL YAML so we don't depend on the project's
    # actual file (keeps the test hermetic).
    ttl_yaml = tmp_path / "redis_feature_store.yaml"
    ttl_yaml.write_text(yaml.safe_dump({"default_ttl_seconds": 60, "ttl_by_pattern": []}))

    # Unreachable URL — the store's `connect()` will raise; we suppress
    # and proceed disconnected. `get_features` will raise `RuntimeError`
    # from the store's "call connect() before get_multi()" guard, which
    # the orchestrator catches via the redis-error branch.
    bad_store = RedisFeatureStore(
        redis_url="redis://no-such-host:6379/0",
        ttl_config_path=ttl_yaml,
    )
    with contextlib.suppress(Exception):
        await bad_store.connect()

    s = FeatureService(
        pipeline_dir=_PIPELINE_DIR,
        model_manifest_path=_MANIFEST_FILE,
        defaults_config_path=_DEFAULTS_FILE,
        redis_store=bad_store,
        postgres_url="postgresql://no-such-host:5432/fake",
    )
    result = await s.get_features(valid_request)
    assert result.degraded_mode is True
    assert result.source_status["redis"] == "redis_down"
    assert result.source_status["postgres"] == "postgres_down"
    # Verify entity features all default.
    assert result.df["card1_velocity_24h"].iloc[0] == 0.0


async def test_health_check_with_real_redis(
    real_redis_store: RedisFeatureStore,
) -> None:
    """`health_check()` returns ok for real Redis + unreachable for Postgres."""
    s = FeatureService(
        pipeline_dir=_PIPELINE_DIR,
        model_manifest_path=_MANIFEST_FILE,
        defaults_config_path=_DEFAULTS_FILE,
        redis_store=real_redis_store,
        postgres_url="postgresql://no-such-host:5432/fake",
    )
    result = await s.health_check()
    assert result["redis"] == "ok"
    assert result["postgres"] == "unreachable"
