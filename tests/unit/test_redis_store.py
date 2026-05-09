"""Tests for `fraud_engine.api.redis_store.RedisFeatureStore`.

Sprint 5 prompt 5.1.b verification surface.

Business rationale:
    The RedisFeatureStore is the load-bearing primitive for online
    feature lookup at <100ms P95. A regression here — a wrong TTL, a
    silently-dropped key, a leaked pool, a malformed key-schema —
    leaks into the production prediction path and either (a) blows
    the latency budget or (b) corrupts the model's feature inputs
    via stale state. The one place those contracts are pinned is
    here.

Trade-offs considered:
    - `fakeredis.aioredis.FakeRedis` over `unittest.mock.AsyncMock`:
      mocks the very contract the test is meant to verify; a typo in
      the MGET call site or SETEX arg position would pass against a
      mock and fail in production. fakeredis runs the real client
      code paths against an in-process emulator, catching at least
      the API-shape bugs.
    - Function-scoped fixture per test (matches
      `asyncio_default_fixture_loop_scope = "function"` in
      `pyproject.toml`). Module scope would let one test's
      surviving keys pollute the next; the per-test cost is ~1 ms.
    - Monkeypatching `_pool` and `_client` directly rather than
      patching `ConnectionPool.from_url`: the public-API
      interception is cleaner and doesn't rely on internal
      implementation details of redis-py's pool factory.
    - Known fakeredis gaps (Lua subtleties, pubsub edge cases,
      sub-ms expiry precision) don't touch the four operations
      this class uses (PING, MGET, SETEX, aclose). The
      integration test against real Redis (a separate file) is
      the catch-all for any drift.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import fakeredis.aioredis
import pytest
import yaml

from fraud_engine.api.redis_store import RedisFeatureStore

# ---------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------


@pytest.fixture
async def fake_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    """A clean fakeredis instance per test."""
    server = fakeredis.aioredis.FakeRedis()
    yield server
    await server.aclose()


@pytest.fixture
def ttl_yaml(tmp_path: Path) -> Path:
    """Write a representative TTL YAML to tmp_path; return its path.

    Mirrors the canonical `configs/redis_feature_store.yaml` shape but
    is local to the test so deletions in `configs/` cannot break the
    suite.
    """
    cfg = {
        "default_ttl_seconds": 604800,
        "ttl_by_pattern": [
            {"pattern": "*_velocity_1h", "ttl_seconds": 3600},
            {"pattern": "*_velocity_24h", "ttl_seconds": 86400},
            {"pattern": "*_velocity_7d", "ttl_seconds": 604800},
            {"pattern": "*_amt_*_30d", "ttl_seconds": 2592000},
            {"pattern": "*_v_ewm_lambda_*", "ttl_seconds": 604800},
            {"pattern": "*_fraud_v_ewm_lambda_*", "ttl_seconds": 604800},
            {"pattern": "is_coldstart_*", "ttl_seconds": 604800},
            {"pattern": "pagerank_*", "ttl_seconds": 86400},
            {"pattern": "connected_component_*", "ttl_seconds": 86400},
            {"pattern": "entity_degree_*", "ttl_seconds": 86400},
            {"pattern": "*_target_enc_*", "ttl_seconds": 86400},
        ],
    }
    path = tmp_path / "redis_feature_store.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture
async def store(
    monkeypatch: pytest.MonkeyPatch,
    fake_redis: fakeredis.aioredis.FakeRedis,
    ttl_yaml: Path,
) -> AsyncIterator[RedisFeatureStore]:
    """A connected RedisFeatureStore whose pool returns fake_redis."""
    s = RedisFeatureStore(
        redis_url="redis://localhost:6379/0",
        ttl_config_path=ttl_yaml,
    )
    # Patch `_pool` and `_client` directly. The pool object is unused
    # by fakeredis (it carries its own internal state), but `_pool is
    # not None` is the signal `connect()` uses to detect "already
    # connected"; setting both keeps the lifecycle invariants intact.
    monkeypatch.setattr(s, "_pool", fake_redis.connection_pool)
    monkeypatch.setattr(s, "_client", fake_redis)
    yield s
    # `disconnect()` walks both `_client` and `_pool`; suppress errors
    # to keep teardown idempotent if a test already cleaned up.
    with contextlib.suppress(Exception):
        await s.disconnect()


# ---------------------------------------------------------------------
# TestInit — construction + YAML loading.
# ---------------------------------------------------------------------


class TestInit:
    """Constructor reads Settings + loads the TTL YAML."""

    def test_default_redis_url_from_settings(
        self,
        ttl_yaml: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`redis_url=None` resolves to `Settings.redis_url`."""
        monkeypatch.setenv("REDIS_URL", "redis://example:6379/0")
        # Re-load Settings (the lru_cache means we need a fresh import).
        from fraud_engine.config.settings import get_settings

        get_settings.cache_clear()
        s = RedisFeatureStore(ttl_config_path=ttl_yaml)
        assert s._redis_url == "redis://example:6379/0"
        get_settings.cache_clear()

    def test_explicit_redis_url_override(self, ttl_yaml: Path) -> None:
        """An explicit `redis_url` kwarg wins over Settings."""
        s = RedisFeatureStore(
            redis_url="redis://override:9999/1",
            ttl_config_path=ttl_yaml,
        )
        assert s._redis_url == "redis://override:9999/1"

    def test_missing_ttl_config_raises(self, tmp_path: Path) -> None:
        """A non-existent YAML path raises `FileNotFoundError`."""
        bogus = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            RedisFeatureStore(ttl_config_path=bogus)

    def test_non_mapping_yaml_root_raises(self, tmp_path: Path) -> None:
        """YAML root that isn't a mapping raises `TypeError`."""
        bad = tmp_path / "list_root.yaml"
        bad.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(TypeError):
            RedisFeatureStore(ttl_config_path=bad)

    def test_malformed_ttl_pattern_raises(self, tmp_path: Path) -> None:
        """A `ttl_by_pattern` entry missing `ttl_seconds` raises `ValueError`."""
        bad = tmp_path / "malformed.yaml"
        bad.write_text(
            yaml.safe_dump(
                {
                    "default_ttl_seconds": 100,
                    "ttl_by_pattern": [{"pattern": "*", "tll_seconds_typo": 1}],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            RedisFeatureStore(ttl_config_path=bad)


# ---------------------------------------------------------------------
# TestMakeKey — schema correctness + validation.
# ---------------------------------------------------------------------


class TestMakeKey:
    """Canonical key schema + character-class validation."""

    def test_schema_card1_int_id(self, store: RedisFeatureStore) -> None:
        assert store.make_key("card1", 13926, "velocity_24h") == "feat:card1:13926:velocity_24h"

    def test_schema_string_entity_id(self, store: RedisFeatureStore) -> None:
        """Free-form entity_id (e.g. email domain) flows through verbatim."""
        assert (
            store.make_key("P_emaildomain", "gmail.com", "v_ewm_lambda_0.05")
            == "feat:P_emaildomain:gmail.com:v_ewm_lambda_0.05"
        )

    def test_int_entity_id_coerced_to_str(self, store: RedisFeatureStore) -> None:
        """Plain `str(...)` coercion; no `repr` decoration."""
        assert ":42:" in store.make_key("addr1", 42, "velocity_1h")

    @pytest.mark.parametrize("entity_type", ["card:1", "with space", "tab\t", "", "x" * 129])
    def test_invalid_entity_type_raises(
        self,
        store: RedisFeatureStore,
        entity_type: str,
    ) -> None:
        with pytest.raises(ValueError, match="entity_type"):
            store.make_key(entity_type, 42, "velocity_1h")

    @pytest.mark.parametrize("feature_name", ["with:colon", "with space", "tab\t", "", "x" * 129])
    def test_invalid_feature_name_raises(
        self,
        store: RedisFeatureStore,
        feature_name: str,
    ) -> None:
        with pytest.raises(ValueError, match="feature_name"):
            store.make_key("card1", 42, feature_name)

    def test_underscore_dot_dash_allowed(self, store: RedisFeatureStore) -> None:
        """Realistic feature names like `v_ewm_lambda_0.05` validate."""
        assert store.make_key("card1", 1, "v_ewm_lambda_0.05").endswith("v_ewm_lambda_0.05")
        assert store.make_key("card1", 1, "x-y_z.a").endswith("x-y_z.a")


# ---------------------------------------------------------------------
# TestTTLFor — glob-pattern lookup.
# ---------------------------------------------------------------------


class TestTTLFor:
    """First-match-wins glob lookup over the YAML's `ttl_by_pattern`."""

    @pytest.mark.parametrize(
        ("feature_name", "expected_ttl"),
        [
            ("card1_velocity_1h", 3600),
            ("addr1_velocity_24h", 86400),
            ("DeviceInfo_velocity_7d", 604800),
            ("card1_amt_mean_30d", 2592000),
            ("card1_v_ewm_lambda_0.05", 604800),
            ("card1_fraud_v_ewm_lambda_0.5", 604800),
            ("is_coldstart_card1", 604800),
            ("pagerank_score", 86400),
            ("connected_component_size", 86400),
            ("entity_degree_card1", 86400),
            ("card1_target_enc_pcd", 86400),
        ],
    )
    def test_pattern_matches_expected_ttl(
        self,
        store: RedisFeatureStore,
        feature_name: str,
        expected_ttl: int,
    ) -> None:
        assert store.ttl_for(feature_name) == expected_ttl

    def test_unknown_feature_falls_through_to_default(
        self,
        store: RedisFeatureStore,
    ) -> None:
        """A feature matching no pattern falls through to `default_ttl_seconds`."""
        assert store.ttl_for("brand_new_feature_unmatched") == 604800

    def test_default_ttl_override_via_kwarg(self, ttl_yaml: Path) -> None:
        """`default_ttl_seconds` kwarg overrides the YAML's value."""
        s = RedisFeatureStore(ttl_config_path=ttl_yaml, default_ttl_seconds=42)
        assert s.ttl_for("brand_new_feature_unmatched") == 42

    def test_first_match_wins(self, tmp_path: Path) -> None:
        """Earlier patterns take priority over later ones for the same feature."""
        cfg = tmp_path / "priority.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "default_ttl_seconds": 99,
                    "ttl_by_pattern": [
                        {"pattern": "abc_*", "ttl_seconds": 100},
                        {"pattern": "*_def", "ttl_seconds": 200},
                    ],
                }
            ),
            encoding="utf-8",
        )
        s = RedisFeatureStore(ttl_config_path=cfg)
        # `abc_def` matches both patterns; the earlier one wins.
        assert s.ttl_for("abc_def") == 100


# ---------------------------------------------------------------------
# TestGetMulti — MGET round-trip.
# ---------------------------------------------------------------------


class TestGetMulti:
    """Batch read via MGET; preserves input order; missing keys → None."""

    async def test_empty_keys_returns_empty_dict(self, store: RedisFeatureStore) -> None:
        assert await store.get_multi([]) == {}

    async def test_all_present_round_trip(self, store: RedisFeatureStore) -> None:
        """Write three features, MGET them all, every value round-trips."""
        await store.write_entity_features(
            "card1", 13926, {"velocity_24h": 5, "amt_mean_30d": 100.5, "missing_meta": None}
        )
        keys = [
            store.make_key("card1", 13926, "velocity_24h"),
            store.make_key("card1", 13926, "amt_mean_30d"),
            store.make_key("card1", 13926, "missing_meta"),
        ]
        result = await store.get_multi(keys)
        assert result[keys[0]] == 5
        assert result[keys[1]] == 100.5
        assert result[keys[2]] is None  # JSON null round-trips

    async def test_some_missing_keys_yield_none(self, store: RedisFeatureStore) -> None:
        """Keys not in Redis come back as None (not absent from the dict)."""
        await store.write_entity_features("card1", 1, {"velocity_24h": 5})
        keys = [
            store.make_key("card1", 1, "velocity_24h"),
            store.make_key("card1", 999, "velocity_24h"),  # not written
        ]
        result = await store.get_multi(keys)
        assert result[keys[0]] == 5
        assert result[keys[1]] is None
        assert len(result) == 2

    async def test_preserves_input_key_order(self, store: RedisFeatureStore) -> None:
        """Dict insertion order matches input key order (debugability)."""
        await store.write_entity_features("card1", 1, {"a": 1, "b": 2, "c": 3})
        keys = [
            store.make_key("card1", 1, "c"),
            store.make_key("card1", 1, "a"),
            store.make_key("card1", 1, "b"),
        ]
        result = await store.get_multi(keys)
        assert list(result.keys()) == keys

    async def test_get_multi_before_connect_raises(self, ttl_yaml: Path) -> None:
        """Calling `get_multi` before `connect()` raises `RuntimeError`."""
        s = RedisFeatureStore(ttl_config_path=ttl_yaml)
        with pytest.raises(RuntimeError, match="connect"):
            await s.get_multi(["feat:card1:1:foo"])


# ---------------------------------------------------------------------
# TestWriteEntityFeatures — pipelined SETEX with per-feature TTL.
# ---------------------------------------------------------------------


class TestWriteEntityFeatures:
    """Round-trip writes; TTL applied per feature; mixed value types."""

    async def test_round_trip_after_write(self, store: RedisFeatureStore) -> None:
        await store.write_entity_features(
            "card1", 13926, {"velocity_24h": 5, "v_ewm_lambda_0.05": {"v": 1.5, "fraud_v": 0.0}}
        )
        v24h = await store.get_multi([store.make_key("card1", 13926, "velocity_24h")])
        ewm = await store.get_multi([store.make_key("card1", 13926, "v_ewm_lambda_0.05")])
        assert next(iter(v24h.values())) == 5
        assert next(iter(ewm.values())) == {"v": 1.5, "fraud_v": 0.0}

    async def test_per_feature_ttl_applied(
        self,
        store: RedisFeatureStore,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        """Each feature's TTL matches `ttl_for(...)` exactly.

        Feature names carry the entity prefix verbatim (`card1_velocity_1h`,
        not bare `velocity_1h`) — matches the trained model's column-name
        convention from Tier-2..Tier-5. The glob patterns in the YAML
        (`*_velocity_1h`, `*_v_ewm_lambda_*`) require this prefix.
        """
        await store.write_entity_features(
            "card1",
            13926,
            {
                "card1_velocity_1h": 1,
                "card1_velocity_24h": 1,
                "card1_v_ewm_lambda_0.05": 1,
            },
        )
        # fakeredis exposes `ttl(key)` as an async coroutine.
        ttl_1h = await fake_redis.ttl(store.make_key("card1", 13926, "card1_velocity_1h"))
        ttl_24h = await fake_redis.ttl(store.make_key("card1", 13926, "card1_velocity_24h"))
        ttl_ewm = await fake_redis.ttl(store.make_key("card1", 13926, "card1_v_ewm_lambda_0.05"))
        # TTLs may have ticked down by 1s between SETEX and TTL probes;
        # accept anything within 1s of the configured value.
        assert 3599 <= ttl_1h <= 3600
        assert 86399 <= ttl_24h <= 86400
        assert 604799 <= ttl_ewm <= 604800

    async def test_empty_features_is_noop(self, store: RedisFeatureStore) -> None:
        """`features={}` returns without raising and without writing."""
        await store.write_entity_features("card1", 13926, {})
        # Nothing in Redis matching the entity prefix.
        result = await store.get_multi([store.make_key("card1", 13926, "anything")])
        assert next(iter(result.values())) is None

    @pytest.mark.parametrize(
        "value",
        [
            42,
            3.14,
            "string",
            None,
            {"nested": {"v": 1.0, "fraud_v": 0.0}},
            [1, 2, 3],
            True,
            False,
        ],
    )
    async def test_mixed_value_types_round_trip(
        self,
        store: RedisFeatureStore,
        value: Any,
    ) -> None:
        await store.write_entity_features("card1", 1, {"f": value})
        result = await store.get_multi([store.make_key("card1", 1, "f")])
        assert next(iter(result.values())) == value

    async def test_invalid_feature_name_raises(self, store: RedisFeatureStore) -> None:
        """A bad feature name raises before any write reaches Redis."""
        with pytest.raises(ValueError, match="feature_name"):
            await store.write_entity_features("card1", 1, {"bad:name": 1})

    async def test_write_before_connect_raises(self, ttl_yaml: Path) -> None:
        s = RedisFeatureStore(ttl_config_path=ttl_yaml)
        with pytest.raises(RuntimeError, match="connect"):
            await s.write_entity_features("card1", 1, {"f": 1})


# ---------------------------------------------------------------------
# TestTTLExpiry — verify keys actually expire.
# ---------------------------------------------------------------------


class TestTTLExpiry:
    """Sanity-check that SETEX'd keys disappear after their TTL."""

    async def test_short_ttl_key_expires(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_redis: fakeredis.aioredis.FakeRedis,
        ttl_yaml: Path,
    ) -> None:
        """Write with TTL=1s, sleep 1.2s, key gone."""
        s = RedisFeatureStore(
            ttl_config_path=ttl_yaml,
            default_ttl_seconds=1,  # default 1s — every unmatched feature uses this
        )
        monkeypatch.setattr(s, "_pool", fake_redis.connection_pool)
        monkeypatch.setattr(s, "_client", fake_redis)

        # `unmatched_feature` → falls through to default 1s.
        await s.write_entity_features("card1", 1, {"unmatched_feature": 42})
        key = s.make_key("card1", 1, "unmatched_feature")
        # Pre-expiry: present.
        before = await s.get_multi([key])
        assert before[key] == 42
        # Post-expiry: gone.
        await asyncio.sleep(1.2)
        after = await s.get_multi([key])
        assert after[key] is None

    async def test_explicit_short_ttl_overrides_pattern(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_redis: fakeredis.aioredis.FakeRedis,
        tmp_path: Path,
    ) -> None:
        """A YAML with a 1s pattern overrides the long defaults."""
        cfg = tmp_path / "fast.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "default_ttl_seconds": 604800,
                    "ttl_by_pattern": [
                        {"pattern": "ephemeral_*", "ttl_seconds": 1},
                    ],
                }
            ),
            encoding="utf-8",
        )
        s = RedisFeatureStore(ttl_config_path=cfg)
        monkeypatch.setattr(s, "_pool", fake_redis.connection_pool)
        monkeypatch.setattr(s, "_client", fake_redis)
        await s.write_entity_features("card1", 1, {"ephemeral_one": "x"})
        key = s.make_key("card1", 1, "ephemeral_one")
        await asyncio.sleep(1.2)
        result = await s.get_multi([key])
        assert result[key] is None


# ---------------------------------------------------------------------
# TestSerialisation — JSON round-trip for representative shapes.
# ---------------------------------------------------------------------


class TestSerialisation:
    """JSON round-trip for the canonical value shapes."""

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("int", 42),
            ("float", 3.14),
            ("str", "samsung browser 6.2"),
            ("none", None),
            (
                "decay_state",
                {"last_t": 13046400, "v": 1.5, "fraud_v": 0.0},
            ),
        ],
    )
    async def test_round_trip_value(
        self,
        store: RedisFeatureStore,
        name: str,
        value: Any,
    ) -> None:
        await store.write_entity_features("card1", 1, {name: value})
        result = await store.get_multi([store.make_key("card1", 1, name)])
        assert next(iter(result.values())) == value


# ---------------------------------------------------------------------
# TestErrorHandling — connection lifecycle + non-serialisable values.
# ---------------------------------------------------------------------


class TestErrorHandling:
    """Lifecycle + boundary errors raise loudly."""

    async def test_get_multi_after_disconnect_raises(self, store: RedisFeatureStore) -> None:
        await store.disconnect()
        with pytest.raises(RuntimeError, match="connect"):
            await store.get_multi(["feat:card1:1:foo"])

    async def test_write_after_disconnect_raises(self, store: RedisFeatureStore) -> None:
        await store.disconnect()
        with pytest.raises(RuntimeError, match="connect"):
            await store.write_entity_features("card1", 1, {"f": 1})

    async def test_non_json_serialisable_value_raises(self, store: RedisFeatureStore) -> None:
        """A `set` is not JSON-serialisable; raises `TypeError`."""
        with pytest.raises(TypeError):
            await store.write_entity_features("card1", 1, {"bad": {"unserialisable_set"}})


# ---------------------------------------------------------------------
# TestContextManager — `async with` lifecycle.
# ---------------------------------------------------------------------


class TestContextManager:
    """`async with` opens the pool on entry, closes it on exit."""

    async def test_disconnect_is_idempotent(self, store: RedisFeatureStore) -> None:
        """Two `disconnect()` calls in a row don't raise."""
        await store.disconnect()
        await store.disconnect()  # idempotent

    async def test_context_manager_rebinds_after_exit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_redis: fakeredis.aioredis.FakeRedis,
        ttl_yaml: Path,
    ) -> None:
        """After `async with` exits, `_client` is None.

        We cannot exercise the real `connect()` -> `PING` path with
        fakeredis injected via monkeypatch (the pool isn't a fakeredis
        pool). Instead, test the `disconnect` half of the lifecycle:
        construct, monkeypatch, exit the context, verify cleanup.
        """
        s = RedisFeatureStore(ttl_config_path=ttl_yaml)
        monkeypatch.setattr(s, "_pool", fake_redis.connection_pool)
        monkeypatch.setattr(s, "_client", fake_redis)
        async with s:
            assert s._client is not None
        assert s._client is None
        assert s._pool is None
