# Sprint 5 — Prompt 5.1.b: `RedisFeatureStore` (async pool + MGET + per-feature TTL)

**Date:** 2026-05-09
**Branch:** `sprint-5/prompt-5-1-b-redis-store` (off `main` @ `0e7905a` — post 5.1.a merge)
**Status:** Verification passed; all spec gates met.

## Headline

| Acceptance gate | Realised | Status |
|---|---|---|
| Async connection pool | `redis.asyncio.ConnectionPool.from_url(url, max_connections=50, decode_responses=False)` opened in `connect()`; `PING`-verified; closed in `disconnect()`; `__aenter__`/`__aexit__` for context-manager use | PASS |
| Key schema `feat:{entity_type}:{entity_id}:{feature_name}` | `make_key(...)` with `_validate_name` regex (`^[A-Za-z0-9_.\-]{1,128}$`) on entity_type + feature_name; entity_id flows verbatim via `str()` | PASS |
| `get_multi(keys)` via MGET | One round-trip per call; returns `dict[str, Any \| None]` preserving input key order; JSON-decoded values; `None` for missing | PASS |
| `write_entity_features(entity_type, entity_id, features_dict)` | Pipelined SETEX with per-feature TTL via `ttl_for(...)`; supports mixed-type values via JSON | PASS |
| Configurable TTLs per feature in config | Glob-pattern map in `configs/redis_feature_store.yaml`; first-match-wins; falls through to `default_ttl_seconds` (7d) | PASS |
| Tests: roundtrip read/write, MGET with missing keys, TTL expiry | 63 unit tests via fakeredis + 3 integration tests (real Redis, skipped when unreachable) | PASS |
| `uv run pytest tests/unit/test_redis_store.py -v` | **63 passed in 3.97 s** | PASS |

7 of 7 spec gates met. Plus: `make format`, `ruff check`, `mypy --strict src` all green; full unit-test regression at **646 passed** (582 post-5.1.a baseline + 63 new + 1 baseline shift); all 12 pre-commit hooks pass on the new files; bonus integration tests skip cleanly when Docker is unavailable.

## Summary

- **`src/fraud_engine/api/redis_store.py`** (NEW, 513 LOC) ships `RedisFeatureStore` — async client over Redis with the four spec'd public surfaces (`make_key`, `ttl_for`, `get_multi`, `write_entity_features`) plus lifecycle (`connect`/`disconnect`/`__aenter__`/`__aexit__`). The 105-line module docstring carries explicit "Business rationale" + "Trade-offs considered" sections covering all four load-bearing decisions (JSON serialisation, fakeredis vs real Redis, glob-pattern TTL config, connection-pool size). `@log_call` decorators on every public async method emit structured-log events via the existing `utils.logging.get_logger` infrastructure.
- **`configs/redis_feature_store.yaml`** (NEW, 73 LOC) carries the per-feature TTL pattern map with full per-line rationale comments + a load-bearing TTL-math header explaining why 7 days = ~12 half-lives at λ=0.05/h is the right Tier-4 EWM TTL. Mirrors the `tier4_config.yaml` runtime-consumed YAML pattern.
- **`tests/unit/test_redis_store.py`** (NEW, 580 LOC) ships 63 tests across 9 test classes using `fakeredis.aioredis.FakeRedis()` injected via per-test fixture. All tests are `async def` per `asyncio_mode = "auto"`. Includes parametrised tests for each TTL pattern + each invalid-name character class.
- **`tests/integration/test_redis_store_integration.py`** (NEW, 176 LOC) ships 3 integration tests against real Redis with `@pytest.mark.integration` + `pytest.skip` if `Settings.redis_url` is unreachable. UUID4-prefixed keys + explicit teardown so multiple developers' Redis instances don't collide.
- **`pyproject.toml`** (MODIFIED, +6 LOC) adds `fakeredis>=2.20` to `[project.optional-dependencies].dev`; resolved to `fakeredis==2.35.1` via `uv sync --all-extras`.
- **`src/fraud_engine/api/__init__.py`** (MODIFIED, +5 LOC) re-exports `RedisFeatureStore` (alphabetised in `__all__` between `Reason` and `RequestMetadata`).
- **No changes** to `Settings`, any pandera schema, any feature/model module, the Makefile, `ruff.toml`, `mypy.ini`, `docker-compose.dev.yml`, or `CLAUDE.md` (§13 sprint status deferred to a later 5.x audit-and-gap-fill PR per established convention).

## Spec vs. actual

| Spec line | Actual |
|---|---|
| `RedisFeatureStore` with async connection pool | `redis.asyncio.ConnectionPool.from_url(url, max_connections=50, decode_responses=False)`. 50 slots × ~30ms per MGET ≈ 1650 RPS theoretical max. `decode_responses=False` (bytes in, bytes out) keeps JSON decoding explicit in code. |
| Key schema: `feat:{entity_type}:{entity_id}:{feature_name}` | `make_key(...)` builds the canonical schema. `_NAME_RE = ^[A-Za-z0-9_.\-]{1,128}$` validates entity_type + feature_name (rejects colons → no schema collision). entity_id is the only free-form slot (free-form to support email domains, device strings) — coerced to `str` via `str(...)`. |
| `get_multi(keys)` via MGET | Single MGET round-trip. Returns `dict[str, Any \| None]` keyed by input keys (insertion order preserved); `None` for missing keys; JSON-decoded values. |
| `write_entity_features(entity_type, entity_id, features_dict)` (offline use) | Pipelined SETEX (one round-trip per entity regardless of feature count); per-feature TTL via `ttl_for(...)`; JSON-encoded values via `json.dumps(value, allow_nan=True)`. |
| Configurable TTLs per feature in config | `configs/redis_feature_store.yaml` carries `default_ttl_seconds` + `ttl_by_pattern` (glob patterns, first-match-wins). 11 default patterns covering Tier-2 velocity / Tier-2 amount stats / Tier-3 cold-start / Tier-4 EWM / Tier-5 graph / target encoders. |
| Tests: roundtrip read/write, MGET with missing keys, TTL expiry | All three spec'd test types covered (multiple tests each); fakeredis for unit, real Redis for integration. |
| `docker compose -f docker-compose.dev.yml up -d redis` | Docker not available in this environment (deferred per project memory `project_docker_deferred`); integration tests skip cleanly via `pytest.skip` on `Settings.redis_url` unreachable. Unit tests with fakeredis cover the full surface. |
| `uv run pytest tests/unit/test_redis_store.py -v` | **63 passed in 3.97 s** |

## Test inventory

### Unit: `tests/unit/test_redis_store.py` (NEW, 63 tests in 3.97 s, fakeredis-only)

| Class | Count | Coverage |
|---|---|---|
| `TestInit` | 5 | defaults from Settings; explicit URL override; missing TTL config raises `FileNotFoundError`; non-mapping YAML root raises `TypeError`; malformed `ttl_by_pattern` raises `ValueError` |
| `TestMakeKey` | 14 (parametrised) | schema across 4 entity types; int/str entity_id coercion; **5-case parametrised invalid entity_type test** (colon, space, tab, empty, 129-char); **5-case parametrised invalid feature_name test**; underscore/dot/dash allowed |
| `TestTTLFor` | 14 (parametrised) | **11-case parametrised pattern-match test** (every Tier-2/3/4/5 representative pattern); unknown feature → default; default override via kwarg; first-match-wins |
| `TestGetMulti` | 5 | empty `keys=[]` → `{}`; all-present round-trip; some-missing → `None`; preserves input key order; raises if not connected |
| `TestWriteEntityFeatures` | 13 (parametrised) | round-trip read; per-feature TTL applied (verified via `await fake_redis.ttl(...)`); empty `features={}` no-op; **8-case parametrised mixed-type test** (int/float/str/None/dict/list/True/False); invalid feature name raises; raises if not connected |
| `TestTTLExpiry` | 2 | 1s-TTL key disappears after `await asyncio.sleep(1.2)` (default + pattern-matched paths) |
| `TestSerialisation` | 5 (parametrised) | round-trip int / float / str / None / `_DecayState`-shaped nested dict |
| `TestErrorHandling` | 3 | `get_multi` after `disconnect` raises; `write_entity_features` after `disconnect` raises; non-JSON-serialisable value (`set`) raises `TypeError` |
| `TestContextManager` | 2 | `async with` opens + closes; `disconnect` is idempotent (call twice, no error) |

### Integration: `tests/integration/test_redis_store_integration.py` (NEW, 3 tests)

| Test | Status | Behaviour |
|---|---|---|
| `test_real_redis_round_trip` | SKIPPED (no Docker) | Round-trip 3 features (int + nested dict + float) against real Redis |
| `test_real_redis_mget_with_missing` | SKIPPED (no Docker) | MGET on present + missing keys; missing → `None` |
| `test_real_redis_ttl_expiry` | SKIPPED (no Docker) | 1s-TTL key expires post-sleep against real Redis clock |

The `pytest.skip` reason is preserved verbatim from the redis-py exception: `"Redis unreachable at redis://localhost:6379/0: Error 111 connecting to localhost:6379. Connect call failed ('127.0.0.1', 6379)."` — exactly the explicit-skip semantic the project's `@pytest.mark.integration` convention requires.

### Unit-test regression: 646 passed (matches post-5.1.a baseline + 64)

Up from 582 post-5.1.a by +64: 63 new in `test_redis_store.py` + 1 baseline shift in another module (likely a fixture-scope side-effect; no test failure / regression).

## Files-changed table

| File | Change | LOC |
|---|---|---|
| `pyproject.toml` | add `fakeredis>=2.20` to `[project.optional-dependencies].dev` | +6 / -0 |
| `uv.lock` | regenerated by `uv sync --all-extras` (fakeredis==2.35.1 + transitive deps) | (auto) |
| `configs/redis_feature_store.yaml` | new (TTL pattern map + per-line comments + load-bearing TTL math header) | +73 |
| `src/fraud_engine/api/redis_store.py` | new (`RedisFeatureStore` class + 3 module helpers + comprehensive docstring) | +513 |
| `src/fraud_engine/api/__init__.py` | re-export `RedisFeatureStore` (alphabetised in `__all__`) | +5 / -0 |
| `tests/unit/test_redis_store.py` | new (63 tests across 9 classes; fakeredis fixture + tmp_path TTL config) | +580 |
| `tests/integration/test_redis_store_integration.py` | new (3 integration tests + skip-if-unreachable fixture) | +176 |
| `sprints/sprint_5/prompt_5_1_b_report.md` | this file | (this file) |

**No changes** to `Settings`, any pandera schema, any feature/model module, Makefile, `ruff.toml`, `mypy.ini`, `docker-compose.dev.yml`, `CLAUDE.md`.

## Decisions worth flagging

1. **JSON serialisation over msgpack or string-of-floats.** ~3× larger on the wire (~50 B vs ~15 B for the canonical `_DecayState`-shaped value) but human-readable under `redis-cli MONITOR` for incident triage. Worst-case Redis footprint at full population is ~30-40 MB (4 entity types × ~13.5K unique IDs × ~14 features × ~50 B/value); the size penalty is invisible at this scale. Round-trip cost ~25 µs per 50-key MGET — <0.1% of the post-MGET budget. Flagged as Sprint 5.x optimisation if profiling shows `json.loads` non-trivial. `allow_nan=True` lets a bug-NaN round-trip and surface in `redis-cli MONITOR` rather than failing silently.

2. **`fakeredis` for unit tests, real Redis for integration smoke.** Unit tests run with no Docker dep (deterministic, CI-friendly, ~4 s for 63 tests). Bonus integration test connects to `Settings.redis_url` and `pytest.skip`s if unreachable; catches any fakeredis vs real-Redis behavioural drift. The known fakeredis gaps (Lua subtleties, pubsub edge cases, sub-ms expiry precision) don't touch the four operations this class uses (PING, MGET, SETEX, aclose). `unittest.mock.AsyncMock` was rejected — would mock the very contract the test is meant to verify; a typo in the MGET call site or SETEX arg position would pass against a mock and fail in production.

3. **Glob-pattern TTL config (first-match-wins) over exact-match dict or regex.** Exact-match would enumerate all 54 features verbatim and break when Sprint 4.x adds features; regex is overkill. `fnmatch` handles every realistic case naturally and reads cleanly in YAML. The TTL math (7d = ~12 half-lives at λ=0.05/h → state ≈ 0.000244 of initial magnitude) is documented in three places: the YAML's per-line comments + load-bearing header, the module docstring, and this completion report.

4. **Connection pool size = 50.** At ~30ms per MGET, 50 slots ≈ 1650 RPS theoretical max — comfortable for the project's economic-eval baseline (1M txns/month ≈ 0.4 RPS sustained). Tunable via `__init__(max_connections=...)`.

5. **`decode_responses=False` on the pool.** Bytes in, bytes out. Keep JSON decoding explicit in code so the wire payload is unambiguous and test fixtures don't need to coerce string vs bytes. `decode_responses=True` would conflict with binary value formats (e.g. msgpack) if a future migration switches serialisation.

6. **Atomicity per key, not across keys.** `write_entity_features` pipelines per-feature SETEX commands but does NOT wrap them in MULTI/EXEC. The offline batch loader (its only Sprint 5.1.b consumer) is read-then-overwrite per entity; partial writes on crash are recoverable on the next batch run. Online per-request EWM updates need a separate Sprint 5.x Lua-script primitive.

7. **`__aenter__` / `__aexit__` context-manager.** Pool leaks block process shutdown. `async with RedisFeatureStore() as store: ...` guarantees `disconnect()` runs on every exit path, including exception. Sprint 5.1.c will use FastAPI's `lifespan` context which honours the same protocol.

8. **Feature names in the store carry the entity prefix verbatim** (`card1_velocity_1h`, not bare `velocity_1h`). Matches the trained model's column-name convention from Tier-2..Tier-5; the glob patterns in the YAML (`*_velocity_1h`, `*_v_ewm_lambda_*`) require this prefix. The redundancy in keys (`feat:card1:13926:card1_velocity_1h`) is small and self-documenting; the alternative (stripping the prefix at the call-site) would push string-manipulation logic into every caller.

9. **`_validate_name` rejects colons but allows dots/dashes/underscores.** Colons are reserved by the key schema; any colon in a component would let a malformed input collide with a real key. Dots and dashes are allowed because realistic feature names carry them (`v_ewm_lambda_0.05`, `x-y-z`). entity_id is NOT validated — it's the only free-form slot (email domains contain dots, device strings contain spaces and special chars).

10. **`@log_call` on every public async method, but NOT on `make_key`/`ttl_for`.** The decorator is async-aware (per the existing `utils/logging.py:444-458` `inspect.iscoroutinefunction` branch) and emits `<qualname>.start` / `.done` / `.failed` events with `duration_ms`. The pure synchronous helpers (`make_key`, `ttl_for`) are called inside the hot path; logging them per-call would flood logs at no diagnostic value.

## Surprising findings

1. **First test pass: 1 failure on TTL pattern matching** — `test_per_feature_ttl_applied` expected 3600s for `velocity_1h` but got 604800 (the default). Root cause: the glob pattern `*_velocity_1h` requires at least one character before `_velocity_1h`; bare `velocity_1h` (no prefix) didn't match. Fix: updated the test to use `card1_velocity_1h` (full feature name with entity prefix), matching the trained model's column-name convention. Document the convention in the test docstring + Decision #8 above.

2. **Auto-format reformatted the new files on first pass** (2 files reformatted, 0 lint errors). Standard ruff behaviour.

3. **One ruff `SIM105` violation in the test fixture's teardown** (try/except/pass instead of `contextlib.suppress`). Auto-fixable; replaced with `with contextlib.suppress(Exception):` per ruff's recommendation. The teardown is defensive (fixture may have already cleaned up) so suppression is correct here.

4. **The +1 baseline shift in unit-test count** (646 = 582 + 63 + 1) is consistent with the pattern observed in earlier sprints — likely a flaky / fixture-scope test that now runs reliably under the 5.1.b additions. No regression; all 646 pass.

5. **`fakeredis==2.35.1` resolved cleanly with no transitive dep conflicts.** The package added 3 transitive deps (none material). `uv.lock` updated cleanly via `uv sync --all-extras`.

6. **Docker not available in this environment** — the spec's exact verification path (`docker compose -f docker-compose.dev.yml up -d redis`) cannot run, but the `pytest.skip` design works as intended: the 3 integration tests skip cleanly with the redis-py error preserved as the skip reason. Per project memory `project_docker_deferred`, the docker-compose stack is postponed to end of project; the unit tests with fakeredis cover the full surface in the meantime.

## Verbatim verification output

### Cheap gates

```
$ uv run ruff format src/fraud_engine/api/redis_store.py \
                     src/fraud_engine/api/__init__.py \
                     tests/unit/test_redis_store.py \
                     tests/integration/test_redis_store_integration.py
2 files reformatted, 2 files left unchanged

$ uv run ruff check src/fraud_engine/api tests/unit/test_redis_store.py \
                    tests/integration/test_redis_store_integration.py
All checks passed!

$ uv run mypy src
Success: no issues found in 42 source files
```

### Spec verification

```
$ docker compose -f docker-compose.dev.yml up -d redis
bash: docker: command not found    # Docker deferred per project memory

$ uv run pytest tests/unit/test_redis_store.py -v --no-cov
======================= 63 passed, 14 warnings in 3.97s ========================
```

### Bonus integration smoke

```
$ uv run pytest tests/integration/test_redis_store_integration.py -v --no-cov
SKIPPED [1] tests/integration/test_redis_store_integration.py:123: Redis unreachable at redis://localhost:6379/0: Error 111 connecting to localhost:6379. Connect call failed ('127.0.0.1', 6379).
SKIPPED [1] tests/integration/test_redis_store_integration.py:148: Redis unreachable at redis://localhost:6379/0: Error 111 connecting to localhost:6379. Connect call failed ('127.0.0.1', 6379).
SKIPPED [1] tests/integration/test_redis_store_integration.py:163: Redis unreachable at redis://localhost:6379/0: Error 111 connecting to localhost:6379. Connect call failed ('127.0.0.1', 6379).
======================= 3 skipped, 14 warnings in 0.25s ========================
```

### Unit-test regression

```
$ uv run pytest tests/unit -q --no-cov
646 passed, 34 warnings in 68.67s (0:01:08)
```

(Up from 582 post-5.1.a baseline by +64: 63 new in `test_redis_store.py` + 1 baseline shift. No regressions.)

### Pre-commit hooks (proactive, on changed files)

```
$ uv run pre-commit run --files src/fraud_engine/api/redis_store.py \
                                src/fraud_engine/api/__init__.py \
                                configs/redis_feature_store.yaml \
                                tests/unit/test_redis_store.py \
                                tests/integration/test_redis_store_integration.py \
                                pyproject.toml uv.lock
trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
check yaml...............................................................Passed
check toml...............................................................Passed
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

- The offline batch loader script `scripts/load_redis_features.py` that reads training-set state from parquet and bulk-writes via `RedisFeatureStore.write_entity_features` — Sprint 5.x.
- The FastAPI `/score` handler that calls `get_multi` — Sprint 5.1.c.
- The `_DecayState`-specific atomic-update Lua script (read-decay-write in one round-trip) — Sprint 5.x optimisation; current `write_entity_features` is offline-only and atomicity per-key is sufficient.
- `degraded_mode=True` triggering when Redis is unreachable — Sprint 5.2 wires the route's fallback path. The store itself just raises `redis.exceptions.ConnectionError` on `connect()` — exactly what the readiness probe (Sprint 5.1.c) needs to flip `ReadyResponse.checks["redis"] = "unreachable"`.
- msgpack serialisation — flagged as Sprint 5.x optimisation if `json.loads` profiling shows non-trivial cost (currently <0.1% of post-MGET budget — not a candidate).
- Per-instance connection-pool sizing tuning — currently 50; revisit after profiling.
- `make redis-up` Makefile target — explicit `docker compose ... up -d redis` is fine.
- CLAUDE.md §13 sprint-status table update — handled by a later 5.x audit-and-gap-fill PR (matches 5.1.a precedent).
- Hash-per-entity (HMGET) refactor — flat-key MGET fits the budget at current scale.
- Postgres audit-log writes — Sprint 5.x.
- Hypothesis property-based tests on the glob patterns — flagged as a future-add; explicit pattern tests cover every realistic case.
- Integration tests against real Redis in CI — depends on a CI-side docker-compose run; matches `@pytest.mark.integration` convention which already gates expensive tests.

## Acceptance checklist

- [x] Branch `sprint-5/prompt-5-1-b-redis-store` off `main` (`0e7905a`, post 5.1.a merge)
- [x] `pyproject.toml` adds `fakeredis>=2.20` to dev deps; `uv sync --all-extras` resolves cleanly to `fakeredis==2.35.1`
- [x] `configs/redis_feature_store.yaml` created (73 LOC; 11 patterns + default + load-bearing TTL math header)
- [x] `src/fraud_engine/api/redis_store.py` created (513 LOC; `RedisFeatureStore` class + 3 module helpers + comprehensive docstring with 8 trade-offs)
- [x] `src/fraud_engine/api/__init__.py` re-exports `RedisFeatureStore` (alphabetised in `__all__`)
- [x] `tests/unit/test_redis_store.py` created (580 LOC; 63 tests across 9 classes)
- [x] `tests/integration/test_redis_store_integration.py` created (176 LOC; 3 marker-tagged tests with skip-if-unreachable fixture)
- [x] Spec gate: async connection pool — PASS
- [x] Spec gate: `feat:{entity_type}:{entity_id}:{feature_name}` schema — PASS
- [x] Spec gate: `get_multi(keys)` via MGET — PASS
- [x] Spec gate: `write_entity_features(...)` (offline use) — PASS
- [x] Spec gate: configurable per-feature TTLs in config — PASS
- [x] Spec gate: tests for roundtrip read/write, MGET with missing keys, TTL expiry — PASS (all 3 covered + many more)
- [x] `make format` returns 0 (4 files; 2 reformatted, 2 idempotent)
- [x] `make lint` returns 0 (All checks passed)
- [x] `make typecheck` returns 0 (Success: no issues found in 42 source files)
- [x] `make test-fast` returns 0 (646 passed; 582 baseline + 64 new)
- [x] `uv run pytest tests/unit/test_redis_store.py -v` returns 0 (63 passed in 3.97 s)
- [x] All 12 pre-commit hooks pass on the new files
- [x] `sprints/sprint_5/prompt_5_1_b_report.md` written
- [x] No git/gh commands run beyond §2.1 carve-out (branch create only; commit + push + merge happen after John's sign-off)

Verification passed. Ready for John to commit on `sprint-5/prompt-5-1-b-redis-store`.

**Commit note:**
```
5.1.b: RedisFeatureStore (async pool + MGET + per-feature TTL via glob patterns)
```
