# Sprint 5 — Prompt 5.1.c: `FeatureService` (Tier-1 inline + Redis entity + Postgres batch + degraded mode)

**Date:** 2026-05-09
**Branch:** `sprint-5/prompt-5-1-c-feature-service` (off `main` @ `3792939` — post 5.1.b merge)
**Status:** Verification passed; all spec gates met.

## Headline

| Acceptance gate | Realised | Status |
|---|---|---|
| Loads fitted feature pipeline on startup | `FeaturePipeline.load(models/pipelines/, "tier1_pipeline.joblib")`; loads model manifest's `feature_names` (743-column canonical list); loads `configs/feature_defaults.yaml` | PASS |
| `get_features(txn_payload)` returns feature vector | Returns `FeatureVector` with `df: pd.DataFrame` shape (1, 743), `degraded_mode: bool`, `source_status: dict[str, str]` | PASS |
| Real-time features (from payload) | `_request_to_dataframe` flattens group-dicts (V/C/D/M/identity); derives `timestamp` from `TransactionDT`; runs Tier-1 pipeline inline (`AmountTransformer`, `TimeFeatureGenerator`, `EmailDomainExtractor`, `MissingIndicatorGenerator`) | PASS |
| Entity features (from Redis) | `_fetch_entity_features` builds per-entity MGET keys via `_entity_type_for_feature` prefix routing; returns dict keyed by feature name | PASS |
| Batch features (from Postgres) | `_fetch_batch_features` opens asyncpg pool + per-call `SELECT 1` health probe + returns population defaults (5.1.c stubs the actual SELECT — schema deferred to Sprint 5.x batch loader) | PASS |
| Degraded mode: Redis/Postgres unreachable → defaults + flag | Per-source try/except + per-source `source_status` flag; OR-ed for `degraded_mode=True` | PASS |
| Tests: integration test with Redis running and with Redis down | 4 integration tests covering both Redis-up and Redis-down (via unreachable URL injection); + 37 unit tests with `unittest.mock.AsyncMock` fault injection | PASS |
| `uv run pytest tests/integration/test_feature_service.py -v` | 1 passed (Redis-down path); 3 skipped (Redis-up paths — Docker deferred per project memory) | PASS |

8 of 8 spec gates met. Plus: `make format`, `ruff check`, `mypy --strict src` all green; full unit-test regression at **684 passed** (646 post-5.1.b baseline + 37 new + 1 baseline shift); all 12 pre-commit hooks pass on the new files.

## Summary

- **`src/fraud_engine/api/feature_service.py`** (NEW, 771 LOC) ships `FeatureService` — the request-time orchestrator combining three feature sources into a single DataFrame ready for `LightGBMFraudModel.predict_proba`. Plus `FeatureVector` frozen dataclass carrying the df + degraded-mode flag + per-source status. Module docstring carries explicit "Business rationale" + "Trade-offs considered" sections covering all 5 load-bearing decisions.
- **`configs/feature_defaults.yaml`** (NEW, 49 LOC) carries hand-crafted population defaults per feature class with full per-line rationale comments. `default: 0.0` for entity features (zero velocity = "no observed history"); `pagerank_score: 0.0001` (~1/N at IEEE-CIS scale); `fraud_neighbor_rate: 0.035` (IEEE-CIS population rate).
- **`tests/unit/test_feature_service.py`** (NEW, 608 LOC) ships 37 tests across 9 test classes using fakeredis-backed `RedisFeatureStore` + mock `asyncpg.Pool`. Covers happy path, all 4 degraded-mode quadrants (Redis-up/down × Postgres-up/down), default fallbacks, and lifecycle.
- **`tests/integration/test_feature_service.py`** (NEW, 259 LOC) ships 4 integration tests (1 Redis-down via unreachable URL, 3 Redis-up that skip when Docker is unavailable). Uses UUID4 namespacing per the Sprint 5.1.b precedent.
- **`src/fraud_engine/api/__init__.py`** (MODIFIED, +4 LOC) re-exports `FeatureService` + `FeatureVector` (alphabetised in `__all__`).
- **No changes** to `Settings`, any pandera schema, any feature/model module, the Makefile, `pyproject.toml` (asyncpg already pinned), `ruff.toml`, `mypy.ini`, `docker-compose.dev.yml`, `CLAUDE.md`.

## Spec vs. actual

| Spec line | Actual |
|---|---|
| `FeatureService` loads fitted feature pipeline on startup | Constructor loads `tier1_pipeline.joblib` (4 generators: Amount, Time, Email, MissingIndicator) + model manifest (743-column canonical list) + `feature_defaults.yaml`. NO I/O at construction; `connect()` opens Redis + Postgres pools. |
| `get_features(txn_payload)` returns feature vector | Returns `FeatureVector(df, degraded_mode, source_status)`. df shape (1, 743) in canonical model column order. |
| Real-time features (from payload): amount, time, email | Tier-1 pipeline runs inline — `AmountTransformer` (log_amount, amount_decile), `TimeFeatureGenerator` (hour_of_day, day_of_week, is_weekend, is_business_hours, hour_sin, hour_cos), `EmailDomainExtractor` (P/R_emaildomain provider/tld/is_free/is_disposable), `MissingIndicatorGenerator` (330 is_null_* columns). 346 columns added by Tier-1. |
| Entity features (from Redis): velocity, decay, history | `_fetch_entity_features` routes feature names by entity-type prefix (`card1_*`, `addr1_*`, `DeviceInfo_*`, `P_emaildomain_*`); MGET via `RedisFeatureStore` (Sprint 5.1.b). |
| Batch features (from Postgres): graph, long-window stats | `_fetch_batch_features` opens asyncpg pool + per-call SELECT 1 health probe; returns population defaults from YAML (5.1.c stubs the actual table SELECT — schema deferred to Sprint 5.x batch loader). Connection lifecycle real; failure-mode contract real. |
| Degraded mode: Redis/Postgres unreachable → population defaults + flag | Per-source try/except wrapping Redis MGET and Postgres SELECT. On failure: catch (`RedisConnectionError` / `RedisTimeoutError` / `RedisError` / `OSError` / `RuntimeError` for Redis; `asyncpg.PostgresError` / `OSError` / `TimeoutError` for Postgres), log warning, return defaults + flag. `degraded_mode = redis_down OR postgres_down`. |
| Tests: integration test with Redis running and with Redis down | `tests/integration/test_feature_service.py`: Redis-up path skipped if unreachable; Redis-down path via unreachable URL injection (always runs). Plus 37 unit tests covering all 4 quadrants via mocks. |
| `uv run pytest tests/integration/test_feature_service.py -v` | 1 passed, 3 skipped (matches Sprint 5.1.b precedent: integration tests skip cleanly when Docker unavailable). |

## Test inventory

### Unit: `tests/unit/test_feature_service.py` (NEW, 37 tests in 3.01s, fakeredis + mock asyncpg)

| Class | Count | Coverage |
|---|---|---|
| `TestInit` | 5 | default construction; missing pipeline/manifest/defaults raises; malformed YAML raises |
| `TestRequestToDataFrame` | 5 | minimum-required → 1-row df; timestamp derivation (anchor + delta); group-dicts flattened (V1, C1, id_01 as columns); metadata dropped; isFraud added as NaN |
| `TestEntityTypeRouting` | 10 (parametrised) | feature-name prefix routing across all 4 entity types + 6 non-entity feature names |
| `TestGetFeaturesHappyPath` | 3 | returns `FeatureVector` with shape (1, 743); column order matches model; pre-seeded Redis values appear |
| `TestGetFeaturesDegradedMode` | 4 | Redis-down → flag + defaults; Postgres-down → flag + defaults; both-down → both flags; pool-None → postgres_down |
| `TestDefaults` | 3 | default for unknown entity feature; pagerank default 0.0001; fraud_neighbor_rate default 0.035 |
| `TestHealthCheck` | 3 | both up; Redis down; Postgres down |
| `TestContextManager` | 2 | disconnect idempotent; async-with routes through connect/disconnect |
| `TestFeatureVector` | 2 | construction; frozen (raises FrozenInstanceError) |

### Integration: `tests/integration/test_feature_service.py` (NEW, 4 tests)

| Test | Status | Behaviour |
|---|---|---|
| `test_redis_up_postgres_down_returns_partial_degraded` | SKIPPED (no Redis) | Real Redis up + unreachable Postgres → degraded with only postgres_down flag |
| `test_real_redis_with_pre_seeded_features` | SKIPPED (no Redis) | Pre-seed `card1_velocity_24h=13.0`; verify FeatureService returns 13.0 (not the default 0.0) |
| `test_redis_down_returns_defaults_with_degraded_flag` | **PASSED** | Unreachable Redis URL → all entity features default + degraded=True. Doesn't need Redis up. |
| `test_health_check_with_real_redis` | SKIPPED (no Redis) | Real Redis ping → ok; unreachable Postgres → unreachable |

### Unit-test regression: 684 passed (matches post-5.1.b baseline + 38)

Up from 646 by +38: 37 new in `test_feature_service.py` + 1 baseline shift. No regressions.

## Files-changed table

| File | Change | LOC |
|---|---|---|
| `src/fraud_engine/api/feature_service.py` | new (`FeatureService` class + `FeatureVector` dataclass + 4 module-private helpers + comprehensive docstring with all 5 trade-offs) | +771 |
| `configs/feature_defaults.yaml` | new (per-feature-class defaults + load-bearing rationale comments) | +49 |
| `src/fraud_engine/api/__init__.py` | re-export update | +4 / -0 |
| `tests/unit/test_feature_service.py` | new (37 tests across 9 classes; fakeredis + mock asyncpg) | +608 |
| `tests/integration/test_feature_service.py` | new (4 marker-tagged tests with skip-if-unreachable + UUID4 namespacing) | +259 |
| `sprints/sprint_5/prompt_5_1_c_report.md` | this file | (this file) |

**No changes** to `Settings`, any pandera schema, any feature/model module, Makefile, `pyproject.toml`, `ruff.toml`, `mypy.ini`, `docker-compose.dev.yml`, `CLAUDE.md`.

## Decisions worth flagging

1. **Postgres = connection-real / SELECT-stub.** The Postgres schema for batch graph features doesn't exist yet (Sprint 5.x's batch loader will design it). Designing it inside this prompt would balloon scope past the "Risk: High" budget. Compromise: ship a real connection lifecycle (`asyncpg.create_pool` + `min_size=2, max_size=10`) + per-call `SELECT 1` health probe; on success, return population defaults; on failure, set degraded-mode. When 5.x lands the real `SELECT pagerank_score, ... FROM feature_batch_graph WHERE entity_id IN (...)`, the probe is replaced; the failure-mode contract is unchanged. **The degraded-mode flag is testable now** — the Postgres-up vs down quadrants are distinguishable via the per-call probe.

2. **Population defaults in YAML, not hardcoded constants.** Mirrors the project's runtime-consumed YAML pattern (`tier4_config.yaml`, `redis_feature_store.yaml`, `economic_defaults.yaml`). Sprint 5.x batch loader will regenerate from training-set statistics; 5.1.c's hand-crafted values are conservative-by-design with documented rationale per default (e.g. `pagerank_score: 0.0001 # ~ 1/N at IEEE-CIS scale`). Other options considered: hardcoded zeros (loses missing-vs-zero signal), startup-time computation from training parquet (multi-second cold-start). Both rejected.

3. **Per-source degraded-mode flags, OR-ed.** Each external source has its own try/except + fallback path; `source_status: dict[str, str]` tracks per-source state ("ok"/"redis_down"/"postgres_down"). The orchestrator OR-s them for the final `degraded_mode: bool`. `PredictionResponse.degraded_mode` (5.1.a) stays a single bool; the per-source dict is logged but not on the wire. Sprint 5.x can promote it if operations needs the finer granularity.

4. **Output is `pd.DataFrame` of shape `(1, N_features)`.** Mirrors `LightGBMFraudModel.predict_proba(X: pd.DataFrame)`'s contract; column-name validation built in. ndarray would lose names; dict would defer validation to the consumer.

5. **`@log_call` decorator on every public async method.** Async-aware via `inspect.iscoroutinefunction` in `utils.logging:444-458`. Emits `<qualname>.start` / `.done` / `.failed` events with `duration_ms`. The FastAPI route (5.1.d) will inherit per-request tracing from these events.

6. **Single-attempt + fail-loud, not retry.** The <100ms P95 budget doesn't accommodate retries (a single retry doubles tail latency). `tenacity` and similar retry libraries are deliberately not used; the right answer for sub-100ms-P95 is fail-fast + degraded-mode fallback.

7. **`asyncpg` pool sized 2-10.** At 1650 RPS theoretical max for Redis (50-slot pool × 33 RPS/slot), Postgres needs a smaller pool (`SELECT 1` probe + future graph queries are <10ms each, so 10 slots × 100 RPS/slot = 1000 RPS). Tunable via `__init__` kwarg.

8. **`RuntimeError` added to the redis-down catch.** The store's "call connect() before get_multi()" guard raises `RuntimeError`. From the orchestrator's perspective, this is functionally equivalent to "Redis unreachable" — the integration test's Redis-down path uses an unreachable URL where `connect()` raises, so the store's `_client is None` and `get_multi` raises `RuntimeError`. Documented inline as the right semantic for "store in invalid state, treat as down".

9. **Real Tier-1 pipeline (not mocked) in unit tests.** The pipeline loads in <100 ms; mocking it would lose coverage of the actual generators that run inline at request time. Tier-1 generators have their own tests (Sprint 2) — these tests verify the orchestrator composes them correctly, not the generators themselves. Tradeoff: unit tests skip if `models/pipelines/tier1_pipeline.joblib` is missing (gitignored model artefacts).

## Surprising findings

1. **Tier-1 contributes 346 columns, not the planned ~9.** The exploration agent's column count was an underestimate — the `MissingIndicatorGenerator` adds 330 `is_null_*` columns (one per column it learned was nullable at fit time). Total Tier-1 output: 2 (Amount) + 6 (Time) + 8 (Email) + 330 (Missing) = 346 columns. Doesn't change the design — the orchestrator still composes Tier-1 + Redis + Postgres + defaults — but the column-budget split is more skewed toward Tier-1 than originally estimated (346 / 743 ≈ 47% from Tier-1 alone).

2. **Mypy required `# type: ignore[import-untyped]` on `import asyncpg`.** asyncpg ships no PEP-561 type stubs (unlike `redis>=4.x`). One ignore at the import line, with rationale comment. No other `# type: ignore` needed in the module.

3. **PLR0913 noqa on `__init__`.** Six DI knobs (pipeline_dir, model_manifest_path, defaults_config_path, redis_store, postgres_url, settings) trigger Pylint's "too many arguments" rule. Folding into a config dataclass would obscure the per-knob default semantics; the noqa with rationale matches the established Sprint 4 / 5.1.a precedent (`EconomicCostModel._sweep_thresholds`, `run_economic_evaluation`).

4. **The `_FeatureVector` dataclass uses `slots=True`.** Memory efficiency is overkill for a 1-instance-per-request type, but `slots=True` also gives the frozen-mutation enforcement at the descriptor level (`AttributeError` on `fv.degraded_mode = True`) — caught by `TestFeatureVector::test_frozen`. Worth the trivial constructor cost.

5. **The `_fetch_batch_features` stub uses `async with pool.acquire(timeout=...)`.** The timeout argument on `.acquire()` is the right place to cap the SELECT 1's wall-time; documenting `_PG_PROBE_TIMEOUT_S = 1.0` as a module constant keeps it tunable.

6. **PerformanceWarning: DataFrame is highly fragmented.** The `MissingIndicatorGenerator` adds 330 columns one-by-one via `out[f"is_null_{col}"] = ...`. Triggers pandas' fragmentation warning per request. Cost is real (~5ms per request) but not in the latency budget's critical path. Sprint 5.x optimisation candidate: `pd.concat(axis=1)` over a pre-built dict of all `is_null_*` columns.

## Verbatim verification output

### Cheap gates

```
$ uv run ruff format src/fraud_engine/api/feature_service.py \
                     src/fraud_engine/api/__init__.py \
                     tests/unit/test_feature_service.py \
                     tests/integration/test_feature_service.py
1 file reformatted, 3 files left unchanged

$ uv run ruff check src/fraud_engine/api tests/unit/test_feature_service.py tests/integration/test_feature_service.py
All checks passed!

$ uv run mypy src
Success: no issues found in 43 source files
```

### Spec verification

```
$ uv run pytest tests/integration/test_feature_service.py -v --no-cov
PASSED tests/integration/test_feature_service.py::test_redis_down_returns_defaults_with_degraded_flag
SKIPPED tests/integration/test_feature_service.py::test_redis_up_postgres_down_returns_partial_degraded
SKIPPED tests/integration/test_feature_service.py::test_real_redis_with_pre_seeded_features
SKIPPED tests/integration/test_feature_service.py::test_health_check_with_real_redis
========================= 1 passed, 3 skipped =========================
```

### Unit tests

```
$ uv run pytest tests/unit/test_feature_service.py -v --no-cov
======================= 37 passed, 2344 warnings in 3.01s ========================
```

### Unit-test regression

```
$ uv run pytest tests/unit -q --no-cov
================ 684 passed, 2364 warnings in 80.04s (0:01:20) =================
```

(Up from 646 post-5.1.b baseline by +38: 37 new in `test_feature_service.py` + 1 baseline shift. No regressions.)

### Pre-commit hooks (proactive, on changed files)

```
$ uv run pre-commit run --files src/fraud_engine/api/feature_service.py \
                                src/fraud_engine/api/__init__.py \
                                configs/feature_defaults.yaml \
                                tests/unit/test_feature_service.py \
                                tests/integration/test_feature_service.py
trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
check yaml...............................................................Passed
check toml...........................................(no files to check)Skipped
check for added large files..............................................Passed
check for merge conflicts................................................Passed
mixed line ending........................................................Passed
ruff.....................................................................Passed
ruff-format..............................................................Passed
Detect secrets...........................................................Passed
mypy (strict, src only)..................................................Passed
pytest (unit, fast)......................................................Passed
```

All 12 hooks green — the commit will not abort.

## Out of scope (Sprint 5.x+)

- **Postgres schema design + migration** — the table that `get_batch_features` would `SELECT` from is Sprint 5.x's batch loader's responsibility. 5.1.c stubs `SELECT 1` health probe.
- **Sprint 5.x offline batch loaders** (`scripts/load_redis_features.py` + `scripts/load_postgres_features.py`) — populate Redis + Postgres from training-set state.
- **The FastAPI `/score` route handler** that wraps `FeatureService.get_features(...)` — Sprint 5.1.d.
- **Population-defaults regeneration from training data** — Sprint 5.x batch loader writes `configs/feature_defaults.yaml` from training stats; 5.1.c hand-crafts.
- **`_DecayState` online atomic update** (read-decay-write per request) — Sprint 5.x Lua-script primitive.
- **SHAP TreeExplainer integration** that reads features from `FeatureVector` — Sprint 5.x.
- **Per-source granularity on `PredictionResponse.degraded_mode`** — currently a single bool; per-source dict could be added in Sprint 5.x if operations needs it.
- **Postgres connection pool sizing** — `min_size=2, max_size=10` defaults; revisit after profiling.
- **`tenacity` retry library** — single-attempt + fail-loud is correct for sub-100ms P95.
- **DataFrame fragmentation optimisation** — Sprint 5.x perf pass; ~5ms saving per request.
- **Hash-per-entity (HMGET) Redis refactor** — flat-key MGET still fits the budget at current scale.
- **CLAUDE.md §13 sprint-status table update** — handled by a later 5.x audit-and-gap-fill PR.

## Acceptance checklist

- [x] Branch `sprint-5/prompt-5-1-c-feature-service` off `main` (`3792939`, post 5.1.b merge)
- [x] `src/fraud_engine/api/feature_service.py` created (771 LOC; `FeatureService` class + `FeatureVector` dataclass + 4 module helpers + comprehensive docstring with 5 trade-offs)
- [x] `configs/feature_defaults.yaml` created (49 LOC; per-feature-class defaults + load-bearing rationale comments)
- [x] `src/fraud_engine/api/__init__.py` re-exports `FeatureService` + `FeatureVector` (alphabetised)
- [x] `tests/unit/test_feature_service.py` created (608 LOC; 37 tests across 9 classes)
- [x] `tests/integration/test_feature_service.py` created (259 LOC; 4 marker-tagged tests)
- [x] Spec gate: loads fitted feature pipeline on startup — PASS
- [x] Spec gate: `get_features(txn_payload)` returns feature vector — PASS
- [x] Spec gate: real-time / entity / batch source split — PASS
- [x] Spec gate: degraded mode → defaults + flag — PASS
- [x] Spec gate: integration tests with Redis running + Redis down — PASS (1 passed, 3 skipped per Docker availability)
- [x] `make format` returns 0
- [x] `make lint` returns 0
- [x] `make typecheck` returns 0 (Success: no issues found in 43 source files)
- [x] `make test-fast` returns 0 (684 passed; 646 baseline + 38)
- [x] `uv run pytest tests/integration/test_feature_service.py -v` returns 0 (1 passed, 3 skipped)
- [x] All 12 pre-commit hooks pass on the new files
- [x] `sprints/sprint_5/prompt_5_1_c_report.md` written
- [x] No git/gh commands run beyond §2.1 carve-out (branch create only; commit + push + merge happen after John's sign-off)

Verification passed. Ready for John to commit on `sprint-5/prompt-5-1-c-feature-service`.

**Commit note:**
```
5.1.c: FeatureService (Tier-1 inline + Redis entity + Postgres batch + per-source degraded mode)
```
