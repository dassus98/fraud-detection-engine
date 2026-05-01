# Sprint 3 — Prompt 3.3.b: Optuna tuning harness for `LightGBMFraudModel`

**Date:** 2026-05-01
**Branch:** `sprint-3/prompt-3-3-b-optuna-tuning-harness` (off `main` @ `b7e004c`)
**Status:** Verification passed.

## Summary

- **`run_tuning(...)`** in `src/fraud_engine/models/tuning.py` — Optuna study harness over `LightGBMFraudModel` (the wrapper from 3.3.a). TPE sampler + MedianPruner; in-memory storage; each trial logs to MLflow as a nested run under a single parent study run; best params + study metadata written to a YAML config.
- **9-knob LightGBM search space** mirroring the canonical Section-3.3 surface (num_leaves, learning_rate log-uniform, max_depth, min_child_samples, reg_alpha log-uniform, reg_lambda log-uniform, feature_fraction, bagging_fraction, bagging_freq).
- **`configs/model_best_params.yaml`** seeded with a 5-trial smoke run on synthetic data (option (b) per user direction: traceable result over empty placeholder). Will be overwritten by the full 100-trial sweep in 3.3.d.
- **5 integration tests** covering smoke completion, YAML round-trip, MLflow nested-run logging, n_trials validation, and search-space coverage.
- **Spec-named verification green:** `pytest tests/integration/test_tuning.py -v` → 5 passed in 7.06 s.

## Spec vs. actual

| Spec line | Actual |
|---|---|
| Optuna study, 100 trials | ✅ `_DEFAULT_N_TRIALS = 100`; tests override to 5 (smoke); 3.3.d will run the full sweep |
| Search space per project plan §3.3 | ✅ 9 knobs (LightGBM canonical). PROJECT_PLAN.md not yet checked in; documented inline as the standard fraud-ML LightGBM tuning surface, with a docstring note that this module is the one place to update if the doc lands different bounds |
| Every trial logged to MLflow | ✅ `mlflow.start_run(nested=True)` per trial under a parent study run; trial logs sampled params + `val_auc` metric |
| Best params saved to config | ✅ YAML payload at `configs/model_best_params.yaml`: `schema_version`, `study_name`, `n_trials`, `best_value`, `best_trial_number`, `random_state`, `best_params` |
| Tests: 5-trial smoke on tiny data | ✅ `test_run_tuning_5_trial_smoke_completes` (synthetic 600 rows × 6 features, 5 trials in <2 s) |
| Tests: assert study completes | ✅ Returned dict carries `n_trials==5`, `best_params` non-empty, `best_value` ∈ [0, 1] |
| Tests: best_params saved | ✅ `test_best_params_written_to_yaml_and_round_trips` |
| Verification: `pytest tests/integration/test_tuning.py -v` | ✅ 5 passed in 7.06 s |
| Do NOT run full 100-trial tune here | ✅ Deferred to 3.3.d |

## Test inventory

**`tests/integration/test_tuning.py`** — 5 tests:

| Test | Coverage |
|---|---|
| `test_run_tuning_5_trial_smoke_completes` | 5 trials on synthetic 600-row frame; returned dict has expected keys + types |
| `test_best_params_written_to_yaml_and_round_trips` | YAML at `output_path`; `yaml.safe_load` → identical content; `schema_version=1` |
| `test_mlflow_logs_one_parent_and_n_trial_runs` | Via `MlflowClient.search_runs(experiment_ids=[...])`: 1 parent + 5 nested children; each trial carries `val_auc` ∈ [0, 1] and `parentRunId` pointing at the parent |
| `test_n_trials_zero_raises` | `n_trials=0` → `ValueError` (fail-fast before any Optuna setup) |
| `test_search_space_keys_all_appear_in_best_params` | `set(best_params.keys()) == set(SEARCH_SPACE_KEYS)` — catches drift between `_suggest_params` and the public tuple |

**Wall: 7.06 s** for the 5-test suite.

## Files-changed table

| File | Change | LOC |
|---|---|---|
| `src/fraud_engine/models/tuning.py` | new (Optuna harness + MLflow nesting + YAML output) | +394 |
| `src/fraud_engine/models/__init__.py` | re-export `run_tuning` + `SEARCH_SPACE_KEYS` (alphabetised) | +5 |
| `tests/integration/test_tuning.py` | new (5 tests + synthetic-data fixture + MLflow-isolated settings fixture) | +220 |
| `scripts/_seed_best_params_smoke.py` | new (one-shot seeder for the YAML; deterministic synthetic data) | +89 |
| `configs/model_best_params.yaml` | new (5-trial smoke result; option (b) — traceable) | +27 |
| `sprints/sprint_3/prompt_3_3_b_report.md` | this file | (this file) |

## Decisions worth flagging

1. **TPE sampler + MedianPruner with `n_startup_trials=5`.** Optuna defaults; consistently outperforms random search at moderate trial counts. The pruner has a 5-trial warmup so early trials run to completion (the median needs comparison baselines); from trial 6+ aggressive pruning kicks in. Trade-off: pruning can kill a slow-starting good trial, but with 100 trials the TPE sampler revisits promising regions reliably.

2. **In-memory Optuna storage.** No SQLite file, no parallel workers, no resume. Sufficient for the 100-trial sweep (~5-10 min on Tier-5 features). 3.3.d may switch to SQLite if the sweep grows or needs resume-on-failure semantics.

3. **MLflow `set_experiment(experiment_id=...)` is required.** Without it, `mlflow.start_run(nested=True)` inside the trial closure defaults to MLflow's "Default" experiment (id=0), not the one returned by `setup_experiment()`. The trial runs become invisible to `client.search_runs(experiment_ids=[experiment_id])`. This was the bug that surfaced in test debugging — fixed by adding `mlflow.set_experiment(experiment_id=experiment_id)` immediately after `setup_experiment()`. Documented inline.

4. **Test uses `MlflowClient.search_runs`, not `mlflow.search_runs`.** The high-level `mlflow.search_runs` returned only 1 row (the parent) on the filesystem-backed tracking store even with the right experiment specified. The lower-level `MlflowClient.search_runs(experiment_ids=[...])` returns all 6 runs (1 parent + 5 children) reliably. Mirrors a pattern that may need to apply to other MLflow-aware tests in the future.

5. **`scale_pos_weight` not in the search space.** Inherited from `LightGBMFraudModel`'s constructor default (`(neg/pos)` from y_train). The class-imbalance weight is a property of the data, not a hyperparameter to tune; tuning over it would conflate imbalance correction with model bias. If a future audit shows benefit, the search space can add it as a `FloatDistribution(0.5, 50.0, log=True)`.

6. **`PROJECT_PLAN.md` not yet checked into the repo.** CLAUDE.md §14 references it; the spec referenced "Section 3.3". The 9-knob search space here is the canonical fraud-ML LightGBM tuning surface; if/when PROJECT_PLAN.md ships with different bounds, `_suggest_params` is the one place to update.

7. **YAML output contains `best_params` mirror of the Optuna `study.best_params` exactly.** Downstream consumers (3.3.d's full-sweep wrapper, 3.3.c calibration, 3.3.d threshold optimisation) load the YAML and feed `best_params` to `LightGBMFraudModel(params=...)` unchanged. The harness deliberately does NOT fold in non-search-space defaults (objective, metric, etc.) — those come from `Settings.lgbm_defaults` at `LightGBMFraudModel.__init__`. Keeping the two layers separate avoids YAML drift and leaves `Settings` as the single source of truth for non-tuned defaults.

## Verbatim verification output

### Cheap gates
```
$ make format && make lint && make typecheck
uv run ruff format src tests scripts
89 files left unchanged
uv run ruff check src tests scripts
All checks passed!
uv run mypy src
Success: no issues found in 35 source files
```

### Spec-named verification
```
$ uv run pytest tests/integration/test_tuning.py -v --no-cov
============================= test session starts ==============================
collected 5 items

tests/integration/test_tuning.py::test_run_tuning_5_trial_smoke_completes PASSED
tests/integration/test_tuning.py::test_best_params_written_to_yaml_and_round_trips PASSED
tests/integration/test_tuning.py::test_mlflow_logs_one_parent_and_n_trial_runs PASSED
tests/integration/test_tuning.py::test_n_trials_zero_raises PASSED
tests/integration/test_tuning.py::test_search_space_keys_all_appear_in_best_params PASSED

======================== 5 passed, 18 warnings in 7.06s ========================
```

### Smoke seed (option b — traceable result)
```
$ MLFLOW_TRACKING_URI=/tmp/seed_mlruns uv run python scripts/_seed_best_params_smoke.py
[I 2026-05-01 12:04:45] Trial 0 finished with value: 0.5423
[I 2026-05-01 12:04:45] Trial 1 finished with value: 0.6194  ← BEST
[I 2026-05-01 12:04:45] Trial 2 finished with value: 0.5968
[I 2026-05-01 12:04:45] Trial 3 finished with value: 0.5277
[I 2026-05-01 12:04:45] Trial 4 finished with value: 0.6071
best_value     = 0.619398
best_params    = {'num_leaves': 185, 'learning_rate': 0.005439667429522981, 'max_depth': 12,
                  'min_child_samples': 84, 'reg_alpha': 8.148e-07, 'reg_lambda': 4.329e-07,
                  'feature_fraction': 0.5917, 'bagging_fraction': 0.6521, 'bagging_freq': 4}
output_path    = configs/model_best_params.yaml
```

### Regression: `make test-fast`
```
421 passed, 34 warnings in 69.89s (0:01:09)
```
Up from 420 (3.3.a baseline) — integration tests don't run in test-fast, so this is +1 unit test (likely a baseline.py adjustment). No regressions.

## Surprising findings

1. **MLflow's `start_run(nested=True)` does NOT inherit `experiment_id` from the parent.** When `set_experiment` hasn't been called, nested runs default to the global "Default" experiment (id=0), not the parent's experiment. This is undocumented behaviour that surfaced as a 1-row `search_runs` result during test debugging. Fix: always call `mlflow.set_experiment(experiment_id=...)` before opening any nested-run-bearing context.

2. **`mlflow.search_runs` (high-level) silently filtered nested runs.** Even with the correct experiment_id, the high-level API returned only the parent run on the filesystem-backed tracking store. Switching to `MlflowClient.search_runs(experiment_ids=[...])` returned all 6 runs as expected. There may be a default `parentRunId IS NULL` filter applied by the high-level API that the client API skips.

3. **Optuna trial wall-time on 600-row synthetic ≈ 200 ms.** The full 5-trial smoke completes in <2 seconds on this synthetic frame. On the 414k-row Tier-5 train split (3.3.d), per-trial wall-time will scale roughly linearly to ~30-60 s, putting the 100-trial sweep at ~50-100 minutes. Manageable.

## Out of scope (Sprint 3 follow-on)

- **Full 100-trial sweep on Tier-5 features** (prompt 3.3.d) — uses this harness, runs against `data/processed/tier5_*.parquet`, overwrites `configs/model_best_params.yaml` with the production result.
- **SQLite-backed Optuna storage** for resume-on-failure (deferred until needed).
- **Per-trial calibration** (Sprint 4 territory).
- **Threshold optimisation** (Sprint 4).
- **`scale_pos_weight` as a tuned knob** (currently fixed at `(neg/pos)`; revisit if audit shows benefit).

## Acceptance checklist

- [x] Branch `sprint-3/prompt-3-3-b-optuna-tuning-harness` off `main` (`b7e004c`)
- [x] `src/fraud_engine/models/tuning.py` created (Optuna harness + MLflow nesting + YAML output, ~394 LOC)
- [x] `src/fraud_engine/models/__init__.py` re-exports `run_tuning` + `SEARCH_SPACE_KEYS`
- [x] `tests/integration/test_tuning.py` created (5 tests across smoke / YAML / MLflow / arg validation / search-space coverage)
- [x] `scripts/_seed_best_params_smoke.py` created (one-shot seeder for the YAML)
- [x] `configs/model_best_params.yaml` created with 5-trial smoke result (option b — traceable per user direction)
- [x] 5-trial smoke completes; best_params populated; YAML round-trips
- [x] MLflow logs 1 parent run + 5 nested trial runs; each trial carries `val_auc`
- [x] `make format && make lint && make typecheck` all return 0
- [x] `make test-fast` returns 0 (421 unit tests pass)
- [x] `uv run pytest tests/integration/test_tuning.py -v` returns 0 (5 passed in 7.06 s)
- [x] `sprints/sprint_3/prompt_3_3_b_report.md` written
- [x] Did NOT run the full 100-trial tune (deferred to 3.3.d per spec)
- [x] No git/gh commands run beyond §2.1 carve-out (branch create only)

Verification passed. Ready for John to commit on `sprint-3/prompt-3-3-b-optuna-tuning-harness`.

**Commit note:**
```
3.3.b: Optuna tuning harness for LightGBMFraudModel (TPE+MedianPruner, MLflow nesting, YAML output)
```
