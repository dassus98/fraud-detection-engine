"""Integration tests for `RedisFeatureStore` against real Redis.

Sprint 5 prompt 5.1.b. Skipped when Redis is unreachable so CI without
Docker still passes; runs when `docker compose -f docker-compose.dev.yml
up -d redis` has been executed.

Business rationale:
    fakeredis (used in the unit tests) covers the API-shape contract
    correctly for the four operations this class uses (PING, MGET,
    SETEX, aclose), but cannot detect behavioural drift if a future
    fakeredis or real-Redis upgrade changes wire-protocol semantics.
    The integration test is the catch-all: it round-trips a write, an
    MGET-with-missing, and a TTL-expiry against the real Redis server
    that production will use.

    Skipping when unreachable matches the repo's `@pytest.mark.integration`
    convention — local dev without Docker (or CI matrices that don't
    spin up the docker-compose stack) get a green test-fast run.

Trade-offs considered:
    - UUID4 prefix on every key + explicit teardown deletion: avoids
      polluting a developer's shared dev Redis with leftover test keys.
      The prefix lives in `entity_id` (the only free-form slot in the
      key schema) so it doesn't violate `make_key`'s validation.
    - One module-scoped reachability probe + per-test fixture: probing
      once amortises the connection-cost across the whole module.
    - `pytest.skip` instead of `xfail` on unreachable Redis: skip is the
      right semantic — there's no failing assertion, just an absent
      dependency. xfail would falsely advertise a known-broken test.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
import redis.asyncio
import yaml

from fraud_engine.api.redis_store import RedisFeatureStore
from fraud_engine.config.settings import get_settings

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def redis_url() -> str:
    """Resolve the Redis URL from Settings; skip the module if unreachable.

    Probes once per module so the connection cost amortises across
    every test. The probe uses the redis-py client directly (bypasses
    `RedisFeatureStore` so a bug in the store's `connect()` doesn't
    cause the whole module to skip silently).
    """
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
def ttl_yaml(tmp_path: Path) -> Path:
    """Minimal TTL YAML; tests that need richer patterns build their own."""
    cfg = {
        "default_ttl_seconds": 60,
        "ttl_by_pattern": [
            {"pattern": "ephemeral_*", "ttl_seconds": 1},
            {"pattern": "*_velocity_24h", "ttl_seconds": 86400},
        ],
    }
    path = tmp_path / "redis_feature_store.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture
async def real_store(
    redis_url: str,
    ttl_yaml: Path,
) -> AsyncIterator[tuple[RedisFeatureStore, str]]:
    """A connected store + a UUID4 namespace for this test's keys.

    The namespace is embedded in `entity_id` so every key the test
    writes carries `feat:<entity_type>:<test-uuid>:<feature_name>`.
    The teardown deletes every key under `feat:*:<test-uuid>:*` so the
    real Redis instance is left exactly as it was found.
    """
    async with RedisFeatureStore(redis_url=redis_url, ttl_config_path=ttl_yaml) as s:
        namespace = uuid4().hex
        yield s, namespace
        # Teardown: scan + delete every key carrying our namespace.
        # (`s._client` is the live Redis handle; this is internal access
        # used only for test cleanup.)
        if s._client is not None:
            async for key in s._client.scan_iter(match=f"feat:*:{namespace}:*"):
                await s._client.delete(key)


# ---------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------


async def test_real_redis_round_trip(
    real_store: tuple[RedisFeatureStore, str],
) -> None:
    """Write three features, MGET them all, every value round-trips."""
    store, ns = real_store
    await store.write_entity_features(
        "card1",
        ns,
        {
            "card1_velocity_24h": 7,
            "card1_v_ewm_lambda_0.05": {"last_t": 13046400, "v": 1.25, "fraud_v": 0.0},
            "card1_amt_mean_30d": 145.67,
        },
    )
    keys = [
        store.make_key("card1", ns, "card1_velocity_24h"),
        store.make_key("card1", ns, "card1_v_ewm_lambda_0.05"),
        store.make_key("card1", ns, "card1_amt_mean_30d"),
    ]
    result = await store.get_multi(keys)
    assert result[keys[0]] == 7
    assert result[keys[1]] == {"last_t": 13046400, "v": 1.25, "fraud_v": 0.0}
    assert result[keys[2]] == 145.67


async def test_real_redis_mget_with_missing(
    real_store: tuple[RedisFeatureStore, str],
) -> None:
    """Missing keys come back as None; surrounding present keys round-trip."""
    store, ns = real_store
    await store.write_entity_features("card1", ns, {"card1_velocity_24h": 5})
    keys = [
        store.make_key("card1", ns, "card1_velocity_24h"),
        store.make_key("card1", ns, "card1_velocity_1h"),  # never written
    ]
    result = await store.get_multi(keys)
    assert result[keys[0]] == 5
    assert result[keys[1]] is None


async def test_real_redis_ttl_expiry(
    real_store: tuple[RedisFeatureStore, str],
) -> None:
    """A 1s-TTL key expires after sleep; verify against real Redis clock."""
    store, ns = real_store
    await store.write_entity_features("card1", ns, {"ephemeral_one": "x"})
    key = store.make_key("card1", ns, "ephemeral_one")
    # Pre-expiry: present.
    before = await store.get_multi([key])
    assert before[key] == "x"
    # Post-expiry: gone. 1.5s buffer to absorb scheduler jitter.
    await asyncio.sleep(1.5)
    after = await store.get_multi([key])
    assert after[key] is None
