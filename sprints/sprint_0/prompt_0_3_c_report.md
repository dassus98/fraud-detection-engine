# Sprint 0 — Prompt 0.3.c Completion Report

**Prompt:** MLflow helpers at `src/fraud_engine/utils/mlflow_setup.py` covering `setup_experiment`, `log_dataframe_stats`, `log_economic_metrics`, with a matching `tests/unit/test_mlflow_setup.py`. Single source of truth for MLflow experiment wiring + run-level DataFrame fingerprints + economic-cost metric logging, consumed by Sprint 3 (training + hyperparameter sweep) and Sprint 4 (threshold optimisation).
**Date completed:** 2026-04-23

---

## 1. Summary

Three helpers delivered in one module, each documented with business rationale + trade-offs per CLAUDE §5.2, and each pinned by tests that isolate MLflow to a `tmp_path`-backed tracking URI (no pollution of the repo's `./mlruns`):

| Helper | What it is | Where it gets called |
|---|---|---|
| `setup_experiment(name: str \| None = None, settings: Settings \| None = None) -> str` | Sets MLflow tracking URI from `Settings`, calls `mlflow.get_experiment_by_name` + create-if-missing, returns `str(experiment_id)`. DI-friendly: tests inject a `Settings` pointing at a tmp URI. | Sprint 3 training scripts (one call before opening runs); Sprint 4 threshold-sweep driver (separate experiment for sweeps). |
| `log_dataframe_stats(df: pd.DataFrame, prefix: str) -> None` | Logs 4 params (rows/cols/memory_mb/dtypes-JSON) + 2 metrics (n_missing/n_duplicates), prefix-scoped so a single run can fingerprint train/val/test. | Sprint 3 training (train + val fingerprint on each model run); Sprint 6 drift baseline (logs production batch shapes). |
| `log_economic_metrics(fn_count, fp_count, tp_count, tn_count, fraud_cost, fp_cost, tp_cost=0.0) -> None` | Logs 6 metrics: 4 confusion-matrix counts + `total_cost_usd` + `cost_per_txn` (zero-guard on empty population). Signature mirrors `metrics.economic_cost` so call sites pass through by keyword. Raises `RuntimeError` if no active run. | Sprint 4 threshold sweep (one call per candidate threshold — MLflow UI plots the cost curve directly). |

The pre-existing `configure_mlflow()` entry point is preserved for backwards compat with the observability-demo notebook and `docs/OBSERVABILITY.md`; both set the tracking URI idempotently and coexist.

**12 tests, 100% coverage on `mlflow_setup.py` (42/42 statements, 4/4 branches).** Ruff, ruff-format, and mypy all clean.

---

## 2. Audit — Pre-Existing State

### `src/fraud_engine/utils/mlflow_setup.py` (pre-this-turn, pre-compaction)

A prior version of the module existed with `configure_mlflow()` + a stale `log_economic_metrics(fn_rate, fp_rate, total_cost_usd)` signature — three-parameter, rate-based, and incompatible with the 0.3.c spec's count-based 7-parameter signature. `setup_experiment` existed but set the tracking URI only implicitly (required a prior `configure_mlflow()` call), and had no `settings` DI parameter — tests could not inject a custom Settings to redirect the tracking URI cleanly.

### `tests/unit/test_mlflow_setup.py` (pre-this-turn)

A thin pre-compaction test file existed but targeted the stale rate-based `log_economic_metrics` and the pre-DI `setup_experiment`. Did not cover the spec-required cases (DI-style settings injection, outside-run RuntimeError, empty-population zero-guard).

### Notebook caller

`notebooks/00_observability_demo.ipynb` — §4 *MLflow* cell (id `mlflow-demo`) used the old signature: `log_economic_metrics(fn_rate=0.08, fp_rate=0.02, total_cost_usd=125.40)`. Broke under the new count-based signature.

### Gaps vs spec

| Gap | Severity |
|---|---|
| `log_economic_metrics(fn_rate, fp_rate, total_cost_usd)` — rate-based, 3 args | **Spec-blocker** — spec requires `(fn_count, fp_count, tp_count, tn_count, fraud_cost, fp_cost, tp_cost=0.0)` |
| `setup_experiment` missing `settings: Settings \| None = None` DI parameter | **Spec-blocker** — spec signature is `(name, settings=None)` |
| `setup_experiment` relies on `configure_mlflow()` being called first to set URI | **Spec-blocker** — spec wording is "Set tracking URI from settings, create-or-get experiment, return experiment_id" in one call |
| Spec tests missing (DI injection, outside-run raise, empty-population guard) | **Spec-blocker** |
| Notebook caller uses pre-refactor API | **Downstream-breakage** |

---

## 3. Gap-Fill — Edits This Turn

### `src/fraud_engine/utils/mlflow_setup.py` (267 lines post-ruff-format)

**`setup_experiment`** — refactored to accept `settings: Settings | None = None` and to set the tracking URI inside the function:

```python
def setup_experiment(name: str | None = None, settings: Settings | None = None) -> str:
    effective_settings = settings if settings is not None else get_settings()
    effective_name = name or effective_settings.mlflow_experiment_name

    # Idempotent global side effect — ensures the URI is set even if
    # the caller forgot to invoke `configure_mlflow()` first. Tests
    # rely on this so the Settings-injection override actually
    # reaches MLflow.
    mlflow.set_tracking_uri(effective_settings.mlflow_tracking_uri)

    existing = mlflow.get_experiment_by_name(effective_name)
    if existing is not None:
        experiment_id = existing.experiment_id
    else:
        experiment_id = mlflow.create_experiment(effective_name)

    get_logger(__name__).info(
        "mlflow.experiment_ready",
        name=effective_name,
        experiment_id=experiment_id,
        tracking_uri=effective_settings.mlflow_tracking_uri,
    )
    return str(experiment_id)
```

`None` defaults on `name` and `settings` fall back to `get_settings().mlflow_experiment_name` and `get_settings()` respectively (CLAUDE §5.4 — "no hardcoded values outside config"). The in-function `set_tracking_uri` is idempotent with the pre-existing `configure_mlflow()` — MLflow silently overwrites the URI on each call, so the observability-demo notebook's existing `configure_mlflow(); setup_experiment(...)` flow still works.

**`log_dataframe_stats`** — positional-or-keyword `prefix` (no `*,` separator) matching the spec signature:

```python
def log_dataframe_stats(df: pd.DataFrame, prefix: str) -> None:
    dtype_counts = {str(dt): int(n) for dt, n in df.dtypes.value_counts().items()}
    memory_mb = float(df.memory_usage(deep=True).sum()) / (1024 * 1024)

    mlflow.log_param(f"{prefix}_rows", int(df.shape[0]))
    mlflow.log_param(f"{prefix}_cols", int(df.shape[1]))
    mlflow.log_param(f"{prefix}_memory_mb", round(memory_mb, 4))
    mlflow.log_param(f"{prefix}_dtypes", json.dumps(dtype_counts))

    mlflow.log_metric(f"{prefix}_n_missing", int(df.isna().sum().sum()))
    mlflow.log_metric(f"{prefix}_n_duplicates", int(df.duplicated().sum()))
```

Params-vs-metrics split is the MLflow schema choice: shape/structure is a write-once fact (params are immutable once set); observed null/duplicate counts could theoretically be re-emitted as a DataFrame is re-cleaned (metrics support overwrite). The JSON-encoded `dtypes` param stays well under MLflow ≤2.x's 500-char limit because it's a dtype-to-count histogram.

**`log_economic_metrics`** — rewritten to the count-based 7-parameter signature:

```python
def log_economic_metrics(  # noqa: PLR0913 — the four confusion-matrix counts and three USD costs are the business contract mirroring `metrics.economic_cost`; collapsing into a dict would hide the cost-model semantics at every call site.
    fn_count: int,
    fp_count: int,
    tp_count: int,
    tn_count: int,
    fraud_cost: float,
    fp_cost: float,
    tp_cost: float = 0.0,
) -> None:
    if mlflow.active_run() is None:
        raise RuntimeError(
            "log_economic_metrics requires an active MLflow run. "
            "Wrap the call in `with mlflow.start_run(): ...`."
        )

    total_cost_usd = fn_count * fraud_cost + fp_count * fp_cost + tp_count * tp_cost
    n_total = fn_count + fp_count + tp_count + tn_count
    cost_per_txn = total_cost_usd / n_total if n_total > 0 else 0.0

    mlflow.log_metric("fn_count", float(fn_count))
    mlflow.log_metric("fp_count", float(fp_count))
    mlflow.log_metric("tp_count", float(tp_count))
    mlflow.log_metric("tn_count", float(tn_count))
    mlflow.log_metric("total_cost_usd", float(total_cost_usd))
    mlflow.log_metric("cost_per_txn", float(cost_per_txn))
```

TN cost is zero by convention in fraud ML (no observable event), so it's not a parameter. The `active_run()` check is a loud failure mode — silently opening a nested run would hide misconfiguration and pollute the root experiment.

### `tests/unit/test_mlflow_setup.py` (285 lines post-ruff-format)

12 tests across 4 test classes, all sharing an `mlflow_tmp` fixture that redirects MLflow to a `tmp_path`-backed `file:` URI and tears down any leaked active run:

| Class | Test count | Highlights |
|---|---|---|
| `TestConfigureMLflow` | 1 | `test_sets_tracking_uri` — process-global URI is set |
| `TestSetupExperiment` | 4 | `test_creates_and_returns_id` (idempotent on second call), `test_defaults_to_settings_name` (None → Settings), **`test_sets_tracking_uri_from_settings_when_not_pre_configured`** (spec: one-stop entry point), **`test_accepts_injected_settings`** (DI — constructs `Settings(mlflow_tracking_uri=injected_uri)`, confirms MLflow follows) |
| `TestLogDataframeStats` | 3 | `test_records_params_and_metrics` (all 4 params + 2 metrics), `test_counts_nulls_and_dupes` (2 None values + 1 dup row), `test_accepts_positional_prefix` (spec signature is `(df, prefix)`) |
| `TestLogEconomicMetrics` | 4 | **`test_raises_outside_run`** (spec: loud failure), `test_records_counts_and_costs` (hand-computed: fn=2, fp=3, tp=5, tn=90, fraud=450, fp=35, tp=5 → `total=1030`, `cost_per_txn=10.30`), `test_tp_cost_defaults_to_zero` (spec default), `test_zero_population_cost_per_txn_is_zero` (empty-input zero-guard) |

The `mlflow_tmp` fixture also teardown-clears `get_settings.cache_clear()` so sibling tests don't inherit a cached Settings with the tmp URI.

### `notebooks/00_observability_demo.ipynb`

One-line change inside the `mlflow-demo` cell:

```diff
- log_economic_metrics(fn_rate=0.08, fp_rate=0.02, total_cost_usd=125.40)
+ log_economic_metrics(fn_count=8, fp_count=2, tp_count=5, tn_count=85, fraud_cost=450.0, fp_cost=35.0, tp_cost=5.0)
```

Matches the new count-based signature and uses the standard fraud/fp/tp cost constants from `Settings`. JSON validity of the notebook confirmed by round-tripping through the Read tool (all cells preserved).

---

## 4. Deviations from Spec

### (a) `setup_experiment` accepts `name: str | None = None` (spec says `name: str`)

**Spec phrasing:** `setup_experiment(name: str, settings: Settings | None = None) -> str`.

**What exists:** `setup_experiment(name: str | None = None, settings: Settings | None = None) -> str`.

**Justification:** CLAUDE §5.4 — "no hardcoded values outside config". Forcing every caller to thread the experiment name through would either (i) hardcode the default name at each call site (direct violation), or (ii) force every caller to `from fraud_engine.config.settings import get_settings` + `settings.mlflow_experiment_name` themselves. The `None`-fallback pattern is idiomatic for Pydantic-Settings-backed projects and mirrors the approach used in `utils/metrics.py::economic_cost` (0.3.b). The test `test_defaults_to_settings_name` pins the behaviour; the test `test_creates_and_returns_id` still exercises the explicit-name path.

### (b) `configure_mlflow()` kept alongside `setup_experiment()`

**Spec implication:** The spec lists only `setup_experiment`, `log_dataframe_stats`, `log_economic_metrics` as the three helpers — implicitly silent on `configure_mlflow()`.

**What exists:** Both `configure_mlflow()` and `setup_experiment()` coexist.

**Justification:** The observability-demo notebook and `docs/OBSERVABILITY.md` already document the explicit `configure_mlflow(); setup_experiment(...)` two-step pattern, and `baseline.py` uses the same convention. Removing `configure_mlflow()` would be an unjustified breakage with no upside — both functions call `mlflow.set_tracking_uri()`, which is idempotent. Test `test_sets_tracking_uri_from_settings_when_not_pre_configured` confirms `setup_experiment()` works standalone (the spec's "one-stop entry point" path); test `test_sets_tracking_uri` confirms `configure_mlflow()` still works on its own.

### (c) `log_economic_metrics` raises rather than silently nesting

**Spec phrasing:** Doesn't explicitly cover the no-active-run case.

**What exists:** `raise RuntimeError("log_economic_metrics requires an active MLflow run. ...")`.

**Justification:** Silently opening a nested run would hide misconfiguration — the Sprint 4 threshold-sweep driver that forgets to wrap in `mlflow.start_run()` would see its metrics land in orphan runs, which pollute the experiment UI and break cost-curve plotting. The loud `RuntimeError` forces early failure. Test `test_raises_outside_run` locks this in.

### (d) `log_economic_metrics` has 7 parameters — `PLR0913` noqa'd with justification

Ruff's default `max-args=5` triggers `PLR0913` on the 7-parameter signature. The `# noqa: PLR0913` is annotated inline with the rationale: the four confusion-matrix counts + three USD costs are the business contract mirroring `metrics.economic_cost`; collapsing into a dict would hide the cost-model semantics at every call site. This matches CLAUDE §9 rule 3 ("Silencing linters with `# noqa` without justification" — justification is provided) and is the same pattern 0.3.b used for `economic_cost`.

### (e) `log_dataframe_stats` logs 6 MLflow keys (4 params + 2 metrics), spec says "shape, dtypes, memory, and null counts"

**Spec bullets:** shape, dtypes, memory, null counts.

**What exists:** `{prefix}_rows`, `{prefix}_cols`, `{prefix}_memory_mb`, `{prefix}_dtypes` (params); `{prefix}_n_missing`, `{prefix}_n_duplicates` (metrics).

**Justification:** Spec covers shape (rows+cols), dtypes, memory, null counts — all four present. `n_duplicates` is an additional, standard pandas fingerprint; Sprint 3 will want to catch train/val duplicates that slipped past the cleaner, and this is the natural place to capture it. Not a deviation per se, but worth calling out.

### (f) Params vs metrics boundary

**Business judgment:** Shape/structure (rows, cols, memory_mb, dtypes-JSON) → params (write-once, string-typed, immutable once set). Observed counts (n_missing, n_duplicates) → metrics (numeric, plottable, overwritable). This split is documented in the module docstring's "Trade-offs considered" section and matches MLflow community convention.

---

## 5. Files Changed

| File | Status | Lines | Role |
|---|---|---|---|
| `src/fraud_engine/utils/mlflow_setup.py` | Rewritten (pre-compaction + formatter) | 267 | `configure_mlflow` + three spec helpers |
| `tests/unit/test_mlflow_setup.py` | Rewritten (pre-compaction + formatter) | 285 | Contract tests: 12 tests across 4 classes |
| `notebooks/00_observability_demo.ipynb` | Edited | — | §4 `mlflow-demo` cell: rate-based → count-based `log_economic_metrics` call |
| `sprints/sprint_0/prompt_0_3_c_report.md` | **NEW** | — | This report |

No other files modified.

---

## 6. Verification

### Ruff

```
$ uv run ruff check src/fraud_engine/utils/mlflow_setup.py tests/unit/test_mlflow_setup.py
All checks passed!
```

### Ruff format

```
$ uv run ruff format --check src/fraud_engine/utils/mlflow_setup.py tests/unit/test_mlflow_setup.py
2 files already formatted
```

### Mypy (strict)

```
$ uv run mypy src/fraud_engine/utils/mlflow_setup.py
Success: no issues found in 1 source file
```

### Pytest — coverage

```
$ uv run pytest tests/unit/test_mlflow_setup.py -v --cov=src/fraud_engine/utils/mlflow_setup --cov-report=term-missing
...
tests/unit/test_mlflow_setup.py::TestConfigureMLflow::test_sets_tracking_uri PASSED                                   [  8%]
tests/unit/test_mlflow_setup.py::TestSetupExperiment::test_creates_and_returns_id PASSED                              [ 16%]
tests/unit/test_mlflow_setup.py::TestSetupExperiment::test_defaults_to_settings_name PASSED                           [ 25%]
tests/unit/test_mlflow_setup.py::TestSetupExperiment::test_sets_tracking_uri_from_settings_when_not_pre_configured PASSED [ 33%]
tests/unit/test_mlflow_setup.py::TestSetupExperiment::test_accepts_injected_settings PASSED                           [ 41%]
tests/unit/test_mlflow_setup.py::TestLogDataframeStats::test_records_params_and_metrics PASSED                        [ 50%]
tests/unit/test_mlflow_setup.py::TestLogDataframeStats::test_counts_nulls_and_dupes PASSED                            [ 58%]
tests/unit/test_mlflow_setup.py::TestLogDataframeStats::test_accepts_positional_prefix PASSED                         [ 66%]
tests/unit/test_mlflow_setup.py::TestLogEconomicMetrics::test_raises_outside_run PASSED                               [ 75%]
tests/unit/test_mlflow_setup.py::TestLogEconomicMetrics::test_records_counts_and_costs PASSED                         [ 83%]
tests/unit/test_mlflow_setup.py::TestLogEconomicMetrics::test_tp_cost_defaults_to_zero PASSED                         [ 91%]
tests/unit/test_mlflow_setup.py::TestLogEconomicMetrics::test_zero_population_cost_per_txn_is_zero PASSED             [100%]

---------- coverage: platform linux, python 3.11.15-final-0 ----------
Name                                      Stmts   Miss Branch BrPart  Cover   Missing
-------------------------------------------------------------------------------------
src/fraud_engine/utils/mlflow_setup.py       42      0      4      0   100%
-------------------------------------------------------------------------------------
(other modules omitted)

======================= 12 passed, 24 warnings in 4.12s ========================
```

**`src/fraud_engine/utils/mlflow_setup.py`: 100% line coverage, 100% branch coverage.** The 24 warnings are matplotlib/pyparsing deprecation noise + one MLflow FutureWarning about the deprecated filesystem tracking backend (the repo uses `file:` URIs per `docker-compose.dev.yml` — migration to SQLite is a Sprint 5+ concern, not this prompt's).

---

## 7. Acceptance Checklist

From the 0.3.c spec:

- [x] `src/fraud_engine/utils/mlflow_setup.py` exports `setup_experiment`, `log_dataframe_stats`, `log_economic_metrics`
- [x] `setup_experiment(name, settings=None) -> str` sets tracking URI from settings, creates-or-gets experiment, returns `experiment_id`
- [x] `log_dataframe_stats(df, prefix)` logs DataFrame shape, dtypes, memory, and null counts as params + metrics
- [x] `log_economic_metrics(fn_count, fp_count, tp_count, tn_count, fraud_cost, fp_cost, tp_cost=0.0)` logs confusion counts + `total_cost_usd` + `cost_per_txn`
- [x] Tests isolate MLflow via `tmp_path`-backed tracking URI (no pollution of real `mlruns/`)
- [x] Test confirms `setup_experiment` accepts injected `Settings` (DI path)
- [x] Test confirms `log_economic_metrics` raises outside an active run
- [x] Test confirms hand-computed cost (`fn=2, fp=3, tp=5, tn=90, fraud=450, fp=35, tp=5 → total=1030, cpt=10.30`)
- [x] 100% coverage on `src/fraud_engine/utils/mlflow_setup.py`
- [x] Ruff clean
- [x] Ruff format clean
- [x] Mypy clean (strict)
- [x] Notebook caller updated to new signature
- [x] No git commands executed (CLAUDE §2)

---

## 8. Non-Goals

- **MLflow server bring-up:** Deferred. `docker-compose.dev.yml` runs an MLflow server at `http://localhost:5000`, but per the user's standing note ("Docker stack deferred") the compose bring-up is postponed to end of project. Until then, `file:` URIs under `./mlruns` are the backend, which is what the spec assumes for unit tests.
- **SQLite migration:** MLflow's FutureWarning about filesystem backends flags a 2026-era migration path. Out of scope for 0.3.c; revisit in Sprint 5 (API productionisation) or Sprint 6 (monitoring).
- **Autolog / model flavour wrappers:** `mlflow.sklearn.autolog` / `mlflow.lightgbm.autolog` are Sprint 3 concerns. The 0.3.c helpers sit one level below — they are the glue Sprint 3 calls before autolog kicks in.
- **Sprint 4 cost-curve driver:** `log_economic_metrics` is the primitive; the threshold-sweep loop that calls it per candidate threshold is a Sprint 4 prompt.
- **Per-feature PSI logging into MLflow:** Deferred to Sprint 6. The MLflow-side helper for drift metrics does not yet exist; `compute_psi` (0.3.b) is the primitive; Sprint 6 will add the wrapper.
- **Git action:** CLAUDE §2 — no stage, commit, push, or branch from Claude Code.

---

Verification passed. Ready for John to commit. No git action from me.
