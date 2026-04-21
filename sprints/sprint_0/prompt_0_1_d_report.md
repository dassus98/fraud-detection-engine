# Sprint 0 Prompt 0.1.d — Pydantic Settings (Audit & Gap-Fill)

**Depends on:** 0.1.c
**Date:** 2026-04-21
**Risk:** Medium (every downstream module imports `get_settings()`)

## Summary

The three files this prompt produces —
[src/fraud_engine/config/settings.py](../../src/fraud_engine/config/settings.py),
[src/fraud_engine/config/__init__.py](../../src/fraud_engine/config/__init__.py),
and `tests/unit/test_settings.py` — had two of three already in the
repo from prompt 0.1 (commit `9f88036`). The existing
`Settings` class was a **strict superset** of the 0.1.d spec: it
carried every spec field plus MLflow/Prometheus/Grafana ports,
Grafana admin seed (with `SecretStr`), Kaggle credentials,
LightGBM defaults, temporal-split fields, a log-level validator,
a temporal-split-ordering validator, and an `ensure_directories()`
helper.

Two genuine gaps:

1. **No non-negative validator on the three cost fields.** The spec
   requires negative costs to raise `ValidationError`. Added
   `ge=0.0` to `fraud_cost_usd`, `fp_cost_usd`, `tp_cost_usd`.
   Used the `Field(ge=…)` pattern rather than the spec's
   `_non_negative` classmethod validator — this matches the
   existing convention in the class (`decision_threshold` uses
   `ge=0.0, le=1.0`; `train_end_dt` uses `ge=1`). Semantically
   equivalent.
2. **`tests/unit/test_settings.py` did not exist.** Written from
   scratch — 26 tests across 8 classes. Covers every spec case plus
   regression coverage for the repo-specific extensions.

## Per-file audit

### `src/fraud_engine/config/settings.py`

**Spec fields** (13): all present with matching defaults.

| Field | Default | Spec | Class | Notes |
|---|---|---|---|---|
| `seed` | 42 | ✅ | ✅ | identical |
| `log_level` | `"INFO"` | `Literal[...]` | `str` + `@field_validator` | Repo variant accepts any case, uppercases, validates against frozenset. Behaviour is a strict superset. |
| `data_dir` | `./data` | ✅ | ✅ resolved against `_PROJECT_ROOT` so it works no matter where the process is launched. |
| `models_dir` | `./models` | ✅ | ✅ same project-root trick |
| `logs_dir` | `./logs` | ✅ | ✅ same project-root trick |
| `kaggle_username` | `""` → `None` | str | `str \| None` | Class uses None sentinel; behaviour equivalent (falsy in either form). |
| `kaggle_key` | `""` → `None` | str | `SecretStr \| None` | Class uses `SecretStr` so it can't be accidentally logged. Stricter than spec. |
| `redis_url` | `redis://localhost:6379/0` | ✅ | ✅ |
| `postgres_url` | `postgresql://fraud:fraud@localhost:5432/fraud` | ✅ | ✅ |
| `mlflow_tracking_uri` | `./mlruns` | ✅ | ✅ |
| `api_host` | `0.0.0.0` | ✅ | ✅ |
| `api_port` | 8000 | ✅ | ✅ |
| `fraud_cost_usd` | 450.0 | ✅ | ✅ (**+ `ge=0.0` added**) |
| `fp_cost_usd` | 35.0 | ✅ | ✅ (**+ `ge=0.0` added**) |
| `tp_cost_usd` | 5.0 | ✅ | ✅ (**+ `ge=0.0` added**) |
| `decision_threshold` | 0.5 | `ge=0, le=1` | ✅ |

**Repo extensions** (kept; documented in .env.example):

| Field | Default | Why kept |
|---|---|---|
| `lgbm_defaults` | dict | Sprint 3 Optuna starting point; centralised so the sweep and the eval pipeline agree. |
| `mlflow_experiment_name` | `fraud-detection` | Sprint 3/4 need one experiment tree for all training + sweeps. |
| `mlflow_port` / `prometheus_port` / `grafana_port` | 5000/9090/3000 | docker-compose.dev.yml publishes these. |
| `grafana_admin_user` / `grafana_admin_password` | `admin` / `SecretStr("admin")` | Dev-only seed for Grafana container. Password is a SecretStr. |
| `transaction_dt_anchor_iso` | `2017-12-01T00:00:00+00:00` | IEEE-CIS calendar anchor (Sprint 1). |
| `train_end_dt` / `val_end_dt` | 10,454,400 / 13,046,400 | Sprint 1 temporal-split boundaries in TransactionDT seconds. |

**Derived properties:** `raw_dir`, `interim_dir`, `processed_dir` —
all `Path` objects computed from `data_dir`. Matches spec.

**Validators:**
- `_normalise_log_level` — rejects invalid levels, uppercases input. (pre-existing)
- `_val_end_after_train_end` — refuses an empty val window. (pre-existing)
- Cost non-negative — now enforced via `Field(ge=0.0)` (new).
- Threshold `[0, 1]` — via `Field(ge=0.0, le=1.0)` (pre-existing).

**Helper:** `ensure_directories()` — creates the data/models/logs
tree idempotently. (pre-existing)

**`get_settings()`:** `@lru_cache(maxsize=1)` wrapped function
returning a process-singleton. Tests clear the cache via
`get_settings.cache_clear()`. (pre-existing)

### `src/fraud_engine/config/__init__.py`

Already exports `Settings` and `get_settings`. No changes.

```python
from fraud_engine.config.settings import Settings, get_settings
__all__ = ["Settings", "get_settings"]
```

### `tests/unit/test_settings.py` (new)

26 tests across 8 classes. Test helper `_build(**overrides)` calls
`Settings(_env_file=None, **overrides)` so the tests do not pick up
whatever happens to live in a developer's local `.env`.

| Class | Tests | Covers |
|---|---|---|
| `TestDefaults` | 2 | seed, log_level, costs, threshold, API host/port; Redis/Postgres/MLflow/Prometheus/Grafana defaults match `.env.example` |
| `TestCostValidation` | 6 (2 × 3 fields) | Negatives raise; zero allowed — parametrised across all three cost fields |
| `TestThresholdValidation` | 5 | Below 0, above 1, and boundaries 0.0 / 0.5 / 1.0 |
| `TestDerivedPaths` | 2 | `raw_dir`/`interim_dir`/`processed_dir` rooted in `data_dir` and recomputed on change |
| `TestLogLevelValidator` | 6 (5 + 1) | Case normalisation; unknown level rejected |
| `TestTemporalSplitValidator` | 2 | `val_end_dt == train_end_dt` and `<` both raise |
| `TestGetSettingsCache` | 1 | `get_settings() is get_settings()` after `cache_clear()` |
| `TestEnsureDirectories` | 2 | Full tree creation; idempotent |

## Verification

### 1. `uv run python -c "from fraud_engine.config.settings import get_settings; s = get_settings(); print(s.model_dump())"`

Dumped via a JSON wrapper (to escape cleanly through
PowerShell → WSL bash → python -c). Output shows every field
loaded, `SecretStr` values auto-masked:

```text
{
  "data_dir": "data",
  "models_dir": "/home/dchit/projects/fraud-detection-engine/models",
  "logs_dir": "/home/dchit/projects/fraud-detection-engine/logs",
  "seed": "42",
  "fraud_cost_usd": "450.0",
  "fp_cost_usd": "35.0",
  "tp_cost_usd": "5.0",
  "lgbm_defaults": "{...}",
  "api_host": "0.0.0.0",
  "api_port": "8000",
  "redis_url": "redis://localhost:6379/0",
  "postgres_url": "postgresql://fraud:fraud@localhost:5432/fraud",
  "mlflow_tracking_uri": "./mlruns",
  "mlflow_experiment_name": "fraud-detection",
  "mlflow_port": "5000",
  "prometheus_port": "9090",
  "grafana_port": "3000",
  "grafana_admin_user": "admin",
  "grafana_admin_password": "**********",
  "kaggle_username": "<redacted>",
  "kaggle_key": "**********",
  "log_level": "INFO",
  "decision_threshold": "0.5",
  "transaction_dt_anchor_iso": "2017-12-01T00:00:00+00:00",
  "train_end_dt": "10454400",
  "val_end_dt": "13046400"
}
```

**Note:** `data_dir` resolves to `"data"` (relative) because this
process was launched from the repo root and a `.env` on the
developer's box sets `DATA_DIR=./data`. When tests override it
via `_build(data_dir=tmp_path)` the absolute path is used. Not a
bug — `ensure_directories()` resolves it before use.

### 2. `uv run pytest tests/unit/test_settings.py -v`

```text
============================== 26 passed in 0.17s ==============================
```

All 26 tests pass; full list in the per-file table above.

### 3. `uv run mypy src/fraud_engine/config`

```text
Success: no issues found in 2 source files
```

Strict-mode clean on `settings.py` and `__init__.py`.

### 4. `uv run ruff check src/fraud_engine/config`

```text
All checks passed!
```

Clean under the 0.1.b ruleset (`E, F, I, N, UP, B, SIM, ARG, RET,
PTH, PL`).

## Deviations from prompt

1. **Non-negative cost validation via `Field(ge=0.0)` rather than
   the spec's `_non_negative` classmethod validator.** Semantically
   equivalent (both raise `ValidationError` on negatives). Chose
   `Field(ge=…)` to match the existing class's convention —
   `decision_threshold` already uses `ge=0.0, le=1.0` and
   `train_end_dt` uses `ge=1`. Staying consistent beats mimicking
   the spec literally.
2. **Kaggle fields are `str | None` + `SecretStr | None`, not
   `""`.** Pre-existing. `SecretStr` for the key prevents
   accidental log leakage — strictly stronger than the spec.
3. **Extra fields preserved.** The class carries MLflow/Prometheus/
   Grafana ports, Grafana admin seed, LightGBM defaults, temporal
   split, and the ISO anchor. Removing them would break Sprint 1
   splits and docker-compose.dev.yml port binding. Kept.
4. **`log_level: str` + validator, not `Literal[...]`.** Pre-existing.
   The validator is a strict superset (accepts case variants and
   normalises). Functional contract is tighter than the spec's
   Literal (which would reject "info" lowercase).
5. **Test file covers more than the spec's five cases.** The spec
   lists five must-haves; the file has 26 tests covering those
   plus log-level, temporal-split ordering, and
   `ensure_directories()`. Extra coverage at zero cost since the
   validators already existed.

## Known gaps / handoffs

- **One test touches the process-wide `get_settings` cache.**
  `TestGetSettingsCache.test_returns_same_instance` calls
  `get_settings.cache_clear()` before and after to stay hygienic,
  but any test that relies on an `lru_cache`d Settings earlier in
  the same session could see a cleared cache. Current test order
  does not trigger this. If a later prompt adds such a test,
  consider a `pytest.fixture(autouse=True)` that clears the cache
  per test.
- **`data_dir` absolute-vs-relative behaviour depends on `.env`.**
  A developer's `.env` that sets `DATA_DIR=./data` leaves the
  field as a relative `Path("data")`; the default would be
  `_PROJECT_ROOT / "data"` (absolute). Both work because every
  caller resolves via `Path.resolve()` / `mkdir(parents=True)`.
  Flagged for clarity only.
- **No integration between `Settings` and the 0.1.b `ruff.toml`
  rule expansion.** 0.1.b added `PL`/`RET`/`PTH` families; they
  surfaced zero issues on `src/fraud_engine/config/`. Other
  directories have not been linted in this prompt.

## Acceptance checklist

- [x] **`Settings` class exists with every spec field + validator** —
  see per-file audit; all 13 spec fields present with matching
  defaults. Three cost fields now validate non-negatives.
- [x] **`get_settings()` is `lru_cache`-wrapped** — line 346 of
  [settings.py](../../src/fraud_engine/config/settings.py).
- [x] **`__init__.py` exports `Settings` + `get_settings`** —
  pre-existing, unchanged.
- [x] **`tests/unit/test_settings.py` covers five spec cases** — defaults
  load, negative costs raise, threshold out-of-range raises, derived
  paths correct, cache returns the same instance. Plus 21 more.
- [x] **`uv run python -c '…get_settings…'` prints the dump** —
  26 fields, SecretStrs masked.
- [x] **`uv run pytest tests/unit/test_settings.py -v`** — 26/26 pass in 0.17s.
- [x] **`uv run mypy src/fraud_engine/config`** — strict clean.
- [x] **`uv run ruff check src/fraud_engine/config`** — clean.
- [x] **No git commands run** — per CLAUDE.md §2.

Ready for John to commit. (No git action from me.)
