"""Unit tests for `fraud_engine.api.feature_service.FeatureService`.

Sprint 5 prompt 5.1.c verification surface.

Business rationale:
    The FeatureService is the load-bearing orchestrator at request
    time. A regression here — a wrong default, a silent column-drop, a
    Redis-failure handler that swallows real bugs, a Postgres probe
    that masks degraded mode — leaks into the production prediction
    path and either (a) blows the latency budget, (b) corrupts the
    feature vector, or (c) hides a true outage. The one place those
    contracts are pinned is here.

Trade-offs considered:
    - **fakeredis-backed `RedisFeatureStore` + mock asyncpg pool.**
      Unit tests run with no Docker dep. The fakeredis path catches
      API-shape bugs in the Redis call site; the mock-pool path
      catches PostgresError-handling bugs in the Postgres path. Real
      wire-protocol drift is the integration test's job
      (`tests/integration/test_feature_service.py`).
    - **Skip-if-artefacts-missing.** The fitted Tier-1 pipeline and
      LightGBM manifest live in `models/` (gitignored). When those
      files are absent (clean CI without model artefacts), the fixture
      `pytest.skip`s so the suite stays green. The skip is explicit so
      a CI-runner with the artefacts WILL run these tests. Mirrors the
      Sprint 4 `test_run_economic_evaluation.py` pattern.
    - **`AsyncMock` for asyncpg fault injection.** The Postgres pool's
      `acquire()` returns an async context manager whose `fetchval()`
      either succeeds (returns 1) or raises `asyncpg.PostgresError` /
      `OSError`. Both branches are exercised. Mocking the pool object
      is cleaner than monkeypatching `asyncpg.create_pool`, which has
      module-level visibility surprises.
    - **Real Tier-1 pipeline (not mocked).** The pipeline loads in
      <100 ms; mocking it would lose coverage of the actual generators
      that run inline at request time. The Tier-1 generators have
      their own tests (Sprint 2) — these tests verify the orchestrator
      composes them correctly, not the generators themselves.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import asyncpg  # type: ignore[import-untyped]
import fakeredis.aioredis
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from fraud_engine.api.feature_service import (
    FeatureService,
    FeatureVector,
    _entity_type_for_feature,
    _request_to_dataframe,
)
from fraud_engine.api.redis_store import RedisFeatureStore
from fraud_engine.api.schemas import TransactionRequest

# ---------------------------------------------------------------------
# Required artefacts. The fixtures below `pytest.skip` if any are
# missing so this module runs cleanly in CI without model artefacts.
# ---------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PIPELINE_DIR = _REPO_ROOT / "models" / "pipelines"
_PIPELINE_FILE = _PIPELINE_DIR / "tier1_pipeline.joblib"
_MANIFEST_FILE = _REPO_ROOT / "models" / "sprint3" / "lightgbm_model_manifest.json"
_DEFAULTS_FILE = _REPO_ROOT / "configs" / "feature_defaults.yaml"


def _require_artefacts() -> None:
    """Skip the test if any required artefact is absent."""
    missing = [p for p in (_PIPELINE_FILE, _MANIFEST_FILE, _DEFAULTS_FILE) if not p.exists()]
    if missing:
        pytest.skip(f"FeatureService artefacts missing: {missing}")


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
async def fake_redis_store(
    monkeypatch: pytest.MonkeyPatch,
    fake_redis: fakeredis.aioredis.FakeRedis,
) -> AsyncIterator[RedisFeatureStore]:
    """A `RedisFeatureStore` whose pool/client point at fake_redis."""
    _require_artefacts()
    store = RedisFeatureStore(
        ttl_config_path=_REPO_ROOT / "configs" / "redis_feature_store.yaml",
    )
    monkeypatch.setattr(store, "_pool", fake_redis.connection_pool)
    monkeypatch.setattr(store, "_client", fake_redis)
    yield store


def _make_mock_pg_pool(*, fail: bool = False) -> MagicMock:
    """Build a mock asyncpg.Pool whose `SELECT 1` succeeds or raises.

    Args:
        fail: If True, the connection's `fetchval` raises
            `asyncpg.PostgresError`. Otherwise returns `1`.
    """
    mock_conn = AsyncMock()
    if fail:
        mock_conn.fetchval = AsyncMock(side_effect=asyncpg.PostgresError("simulated"))
    else:
        mock_conn.fetchval = AsyncMock(return_value=1)

    # `acquire()` returns an async context manager whose __aenter__
    # yields the connection. AsyncMock doesn't natively act as an
    # async-cm; we build one explicitly.
    acquire_ctx = AsyncMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_ctx)
    pool.close = AsyncMock()
    return pool


@pytest.fixture
async def service(
    monkeypatch: pytest.MonkeyPatch,
    fake_redis_store: RedisFeatureStore,
) -> AsyncIterator[FeatureService]:
    """A FeatureService with fake_redis + mocked Postgres pool (success)."""
    _require_artefacts()
    s = FeatureService(
        pipeline_dir=_PIPELINE_DIR,
        model_manifest_path=_MANIFEST_FILE,
        defaults_config_path=_DEFAULTS_FILE,
        redis_store=fake_redis_store,
        postgres_url="postgresql://fake/fake",
    )
    # Inject a successful mock Postgres pool. Tests that want a
    # failing pool override this attribute.
    monkeypatch.setattr(s, "_pg_pool", _make_mock_pg_pool(fail=False))
    yield s
    with contextlib.suppress(Exception):
        await s.disconnect()


@pytest.fixture
def valid_request() -> TransactionRequest:
    """A canonical TransactionRequest fixture used across tests."""
    return TransactionRequest(
        TransactionID=2987000,
        TransactionDT=86400,
        TransactionAmt=59.95,
        ProductCD="W",
        card1=13926,
        card2=490.0,
        card4="visa",
        card6="credit",
        addr1=315.0,
        P_emaildomain="gmail.com",
        DeviceType="mobile",
    )


# ---------------------------------------------------------------------
# TestInit — construction + artefact loading.
# ---------------------------------------------------------------------


class TestInit:
    """Constructor loads pipeline + manifest + defaults; raises loudly on missing."""

    def test_default_construction(self, fake_redis_store: RedisFeatureStore) -> None:
        """Default kwargs resolve to the project's canonical paths."""
        _require_artefacts()
        s = FeatureService(
            pipeline_dir=_PIPELINE_DIR,
            model_manifest_path=_MANIFEST_FILE,
            defaults_config_path=_DEFAULTS_FILE,
            redis_store=fake_redis_store,
        )
        assert len(s._feature_names) == 743
        assert s._entity_defaults["default"] == 0.0
        assert s._batch_defaults["pagerank_score"] == 0.0001

    def test_missing_pipeline_raises(
        self,
        tmp_path: Path,
        fake_redis_store: RedisFeatureStore,
    ) -> None:
        """A non-existent pipeline directory raises (FeaturePipeline.load)."""
        _require_artefacts()
        with pytest.raises(FileNotFoundError):
            FeatureService(
                pipeline_dir=tmp_path,
                model_manifest_path=_MANIFEST_FILE,
                defaults_config_path=_DEFAULTS_FILE,
                redis_store=fake_redis_store,
            )

    def test_missing_manifest_raises(
        self,
        tmp_path: Path,
        fake_redis_store: RedisFeatureStore,
    ) -> None:
        """A non-existent model-manifest path raises `FileNotFoundError`."""
        _require_artefacts()
        bogus = tmp_path / "missing.json"
        with pytest.raises(FileNotFoundError):
            FeatureService(
                pipeline_dir=_PIPELINE_DIR,
                model_manifest_path=bogus,
                defaults_config_path=_DEFAULTS_FILE,
                redis_store=fake_redis_store,
            )

    def test_missing_defaults_raises(
        self,
        tmp_path: Path,
        fake_redis_store: RedisFeatureStore,
    ) -> None:
        """A non-existent defaults YAML raises `FileNotFoundError`."""
        _require_artefacts()
        bogus = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError):
            FeatureService(
                pipeline_dir=_PIPELINE_DIR,
                model_manifest_path=_MANIFEST_FILE,
                defaults_config_path=bogus,
                redis_store=fake_redis_store,
            )

    def test_malformed_defaults_yaml_raises(
        self,
        tmp_path: Path,
        fake_redis_store: RedisFeatureStore,
    ) -> None:
        """A defaults YAML missing required keys raises `ValueError`."""
        _require_artefacts()
        bad = tmp_path / "bad.yaml"
        bad.write_text("entity_features: {}\n", encoding="utf-8")  # missing batch_features
        with pytest.raises(ValueError, match="batch_features"):
            FeatureService(
                pipeline_dir=_PIPELINE_DIR,
                model_manifest_path=_MANIFEST_FILE,
                defaults_config_path=bad,
                redis_store=fake_redis_store,
            )


# ---------------------------------------------------------------------
# TestRequestToDataFrame — TransactionRequest → DataFrame conversion.
# ---------------------------------------------------------------------


class TestRequestToDataFrame:
    """The pure-data marshalling helper (no I/O)."""

    def test_minimum_required(self, valid_request: TransactionRequest) -> None:
        """Single-row DataFrame with explicit columns and timestamp."""
        df = _request_to_dataframe(valid_request, "2017-12-01T00:00:00+00:00")
        assert df.shape[0] == 1
        assert "TransactionID" in df.columns
        assert "TransactionDT" in df.columns
        assert "timestamp" in df.columns

    def test_timestamp_derivation(self, valid_request: TransactionRequest) -> None:
        """`timestamp` = anchor + TransactionDT seconds."""
        df = _request_to_dataframe(valid_request, "2017-12-01T00:00:00+00:00")
        # TransactionDT=86400 → 1 day after anchor → 2017-12-02
        ts = df["timestamp"].iloc[0]
        assert ts.year == 2017
        assert ts.month == 12
        assert ts.day == 2
        assert ts.hour == 0

    def test_group_dicts_flattened(self) -> None:
        """`vesta_v={"V1": 1.0}` becomes column `V1`, not column `vesta_v`."""
        req = TransactionRequest(
            TransactionID=1,
            TransactionDT=0,
            TransactionAmt=10.0,
            ProductCD="W",
            card1=1,
            vesta_v={"V1": 1.5, "V137": -0.5},
            vesta_c={"C1": 7.0},
            identity={"id_01": -5.0},
        )
        df = _request_to_dataframe(req, "2017-12-01T00:00:00+00:00")
        assert "V1" in df.columns
        assert df["V1"].iloc[0] == 1.5
        assert "V137" in df.columns
        assert "C1" in df.columns
        assert "id_01" in df.columns
        # Group-dict keys themselves are not columns.
        assert "vesta_v" not in df.columns
        assert "vesta_c" not in df.columns
        assert "identity" not in df.columns

    def test_metadata_dropped(self, valid_request: TransactionRequest) -> None:
        """The `metadata` sub-model is not a feature input."""
        df = _request_to_dataframe(valid_request, "2017-12-01T00:00:00+00:00")
        assert "metadata" not in df.columns

    def test_is_fraud_added_as_na(self, valid_request: TransactionRequest) -> None:
        """`isFraud` (training target) added as NaN for column-presence parity."""
        df = _request_to_dataframe(valid_request, "2017-12-01T00:00:00+00:00")
        assert "isFraud" in df.columns


# ---------------------------------------------------------------------
# TestEntityTypeRouting — feature-name → entity_type prefix lookup.
# ---------------------------------------------------------------------


class TestEntityTypeRouting:
    """`_entity_type_for_feature` correctly routes feature names."""

    @pytest.mark.parametrize(
        ("feature_name", "expected"),
        [
            ("card1_velocity_24h", "card1"),
            ("card1_v_ewm_lambda_0.05", "card1"),
            ("addr1_amt_mean_30d", "addr1"),
            ("DeviceInfo_velocity_1h", "DeviceInfo"),
            ("P_emaildomain_velocity_7d", "P_emaildomain"),
            ("pagerank_score", None),
            ("connected_component_size", None),
            ("log_amount", None),
            ("hour_of_day", None),
            ("is_null_DeviceType", None),
        ],
    )
    def test_routing(self, feature_name: str, expected: str | None) -> None:
        assert _entity_type_for_feature(feature_name) == expected


# ---------------------------------------------------------------------
# TestGetFeaturesHappyPath — end-to-end with all sources up.
# ---------------------------------------------------------------------


class TestGetFeaturesHappyPath:
    """All sources up → DataFrame shape (1, 743) + degraded=False."""

    async def test_returns_feature_vector(
        self,
        service: FeatureService,
        valid_request: TransactionRequest,
    ) -> None:
        result = await service.get_features(valid_request)
        assert isinstance(result, FeatureVector)
        assert result.df.shape == (1, 743)
        assert result.degraded_mode is False
        assert result.source_status == {"redis": "ok", "postgres": "ok"}

    async def test_column_order_matches_model(
        self,
        service: FeatureService,
        valid_request: TransactionRequest,
    ) -> None:
        """The DataFrame columns are in the canonical model order."""
        result = await service.get_features(valid_request)
        assert list(result.df.columns) == service._feature_names

    async def test_pre_seeded_redis_values_used(
        self,
        service: FeatureService,
        valid_request: TransactionRequest,
    ) -> None:
        """Pre-seeded Redis values appear in the output (not defaults)."""
        # Seed a velocity feature; verify it lands in the right cell.
        await service._redis_store.write_entity_features(
            "card1",
            13926,
            {"card1_velocity_24h": 7},
        )
        result = await service.get_features(valid_request)
        assert result.df["card1_velocity_24h"].iloc[0] == 7


# ---------------------------------------------------------------------
# TestGetFeaturesDegradedMode — fault injection per source.
# ---------------------------------------------------------------------


class TestGetFeaturesDegradedMode:
    """Each source's failure flips degraded_mode and uses defaults."""

    async def test_redis_down_sets_degraded_and_defaults(
        self,
        monkeypatch: pytest.MonkeyPatch,
        service: FeatureService,
        valid_request: TransactionRequest,
    ) -> None:
        """`redis.exceptions.ConnectionError` → degraded + all entity = 0.0."""
        # Force the store's get_multi to raise ConnectionError.
        monkeypatch.setattr(
            service._redis_store,
            "get_multi",
            AsyncMock(side_effect=RedisConnectionError("simulated")),
        )
        result = await service.get_features(valid_request)
        assert result.degraded_mode is True
        assert result.source_status["redis"] == "redis_down"
        # Verify a representative entity feature was filled with the
        # default (0.0).
        assert result.df["card1_velocity_24h"].iloc[0] == 0.0

    async def test_postgres_down_sets_degraded_and_defaults(
        self,
        monkeypatch: pytest.MonkeyPatch,
        service: FeatureService,
        valid_request: TransactionRequest,
    ) -> None:
        """`asyncpg.PostgresError` → degraded; batch defaults still applied."""
        # Replace the success pool with a failing one.
        monkeypatch.setattr(service, "_pg_pool", _make_mock_pg_pool(fail=True))
        result = await service.get_features(valid_request)
        assert result.degraded_mode is True
        assert result.source_status["postgres"] == "postgres_down"
        # Batch defaults are applied either way (5.1.c stub).
        assert result.df["pagerank_score"].iloc[0] == 0.0001

    async def test_both_down_sets_both_flags(
        self,
        monkeypatch: pytest.MonkeyPatch,
        service: FeatureService,
        valid_request: TransactionRequest,
    ) -> None:
        """Both sources unreachable → both flags + all defaults."""
        monkeypatch.setattr(
            service._redis_store,
            "get_multi",
            AsyncMock(side_effect=RedisConnectionError("simulated")),
        )
        monkeypatch.setattr(service, "_pg_pool", _make_mock_pg_pool(fail=True))
        result = await service.get_features(valid_request)
        assert result.degraded_mode is True
        assert result.source_status == {
            "redis": "redis_down",
            "postgres": "postgres_down",
        }

    async def test_postgres_pool_none_treated_as_down(
        self,
        monkeypatch: pytest.MonkeyPatch,
        service: FeatureService,
        valid_request: TransactionRequest,
    ) -> None:
        """If `_pg_pool is None` (no connect()), Postgres treated as down."""
        monkeypatch.setattr(service, "_pg_pool", None)
        result = await service.get_features(valid_request)
        assert result.source_status["postgres"] == "postgres_down"


# ---------------------------------------------------------------------
# TestDefaults — population defaults applied correctly.
# ---------------------------------------------------------------------


class TestDefaults:
    """Population defaults from feature_defaults.yaml."""

    async def test_default_for_unknown_entity_feature(
        self,
        service: FeatureService,
        valid_request: TransactionRequest,
    ) -> None:
        """Entity features not in Redis fall back to the entity default 0.0."""
        # No Redis seeding; happy path returns mostly defaults for cold-start.
        result = await service.get_features(valid_request)
        # Every entity-routed feature column where the entity is null
        # in the request OR Redis returned None should be 0.0.
        assert result.df["card1_velocity_24h"].iloc[0] == 0.0

    async def test_batch_defaults_pagerank(
        self,
        service: FeatureService,
        valid_request: TransactionRequest,
    ) -> None:
        """`pagerank_score` defaults to 0.0001 from YAML."""
        result = await service.get_features(valid_request)
        assert result.df["pagerank_score"].iloc[0] == 0.0001

    async def test_batch_defaults_fraud_neighbor_rate(
        self,
        service: FeatureService,
        valid_request: TransactionRequest,
    ) -> None:
        """`fraud_neighbor_rate` defaults to 0.035 (IEEE-CIS pop rate)."""
        result = await service.get_features(valid_request)
        assert result.df["fraud_neighbor_rate"].iloc[0] == 0.035


# ---------------------------------------------------------------------
# TestHealthCheck — per-source readiness.
# ---------------------------------------------------------------------


class TestHealthCheck:
    """`health_check()` returns per-source status."""

    async def test_both_up(self, service: FeatureService) -> None:
        result = await service.health_check()
        assert result == {"redis": "ok", "postgres": "ok"}

    async def test_redis_down(
        self,
        monkeypatch: pytest.MonkeyPatch,
        service: FeatureService,
    ) -> None:
        # Replace the underlying client's ping with a raiser.
        client = service._redis_store._client
        assert client is not None
        monkeypatch.setattr(
            client, "ping", AsyncMock(side_effect=RedisConnectionError("simulated"))
        )
        result = await service.health_check()
        assert result["redis"] == "unreachable"
        assert result["postgres"] == "ok"

    async def test_postgres_down(
        self,
        monkeypatch: pytest.MonkeyPatch,
        service: FeatureService,
    ) -> None:
        monkeypatch.setattr(service, "_pg_pool", _make_mock_pg_pool(fail=True))
        result = await service.health_check()
        assert result["redis"] == "ok"
        assert result["postgres"] == "unreachable"


# ---------------------------------------------------------------------
# TestContextManager — async-with lifecycle.
# ---------------------------------------------------------------------


class TestContextManager:
    """`async with FeatureService(...) as svc:` opens + closes both pools."""

    async def test_disconnect_is_idempotent(
        self,
        service: FeatureService,
    ) -> None:
        """Two `disconnect()` calls in a row don't raise."""
        await service.disconnect()
        await service.disconnect()  # idempotent

    async def test_aenter_aexit_routes_through_disconnect(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_redis_store: RedisFeatureStore,
    ) -> None:
        """`async with` opens (stub) and closes both pools."""
        _require_artefacts()
        s = FeatureService(
            pipeline_dir=_PIPELINE_DIR,
            model_manifest_path=_MANIFEST_FILE,
            defaults_config_path=_DEFAULTS_FILE,
            redis_store=fake_redis_store,
            postgres_url="postgresql://fake/fake",
        )
        # Mock connect() so the test doesn't need a real Postgres.
        monkeypatch.setattr(s, "connect", AsyncMock())
        monkeypatch.setattr(s, "disconnect", AsyncMock())
        async with s as ctx_service:
            assert ctx_service is s
        s.connect.assert_awaited_once()
        s.disconnect.assert_awaited_once()


# ---------------------------------------------------------------------
# TestFeatureVector — the dataclass itself.
# ---------------------------------------------------------------------


class TestFeatureVector:
    """`FeatureVector` is frozen + slots."""

    def test_construction(self) -> None:
        import pandas as pd

        df = pd.DataFrame([{"a": 1.0}])
        fv = FeatureVector(df=df, degraded_mode=False, source_status={"redis": "ok"})
        assert fv.degraded_mode is False
        assert fv.df is df

    def test_frozen(self) -> None:
        import pandas as pd

        fv = FeatureVector(
            df=pd.DataFrame([{"a": 1.0}]),
            degraded_mode=False,
            source_status={"redis": "ok"},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            fv.degraded_mode = True  # type: ignore[misc]
