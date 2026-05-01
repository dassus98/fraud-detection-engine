# Sprint 3 — Prompt 3.3.a: `LightGBMFraudModel` (native Booster wrapper)

**Date:** 2026-05-01
**Branch:** `sprint-3/prompt-3-3-a-lightgbm-model-wrapper` (off `main` @ `7d317bf`)
**Status:** Verification passed.

## Summary

- **`LightGBMFraudModel`** wraps `lgb.Booster` via the native `lgb.train` API (NOT the sklearn `LGBMClassifier` API used by Sprint 1's `train_baseline`). Provides explicit early stopping, joblib + JSON-manifest persistence, `feature_importance` returning a sorted DataFrame, and a sklearn-shaped `predict_proba` (n × 2).
- **`scale_pos_weight` per spec** — computed from `y_train` as `(neg / pos)` when not supplied at construction, override-friendly via the `scale_pos_weight` kwarg.
- **24 unit tests** across 5 contract surfaces: training, predict_proba, save/load round-trip, feature importance, error handling.
- This is the production-realistic model surface for Sprint 3's hyperparameter-tuning prompt, Sprint 4's economic-cost evaluator, and Sprint 5's serving stack. Sprint 1's `train_baseline` stays — it remains the temporal-split / random-split AUC anchor and the only MLflow consumer.

## Spec vs. actual

| Spec line | Actual |
|---|---|
| `LightGBMFraudModel` wrapping `lgb.Booster` | ✅ `src/fraud_engine/models/lightgbm_model.py` (516 LOC) |
| `fit(X_train, y_train, X_val, y_val)` with early stopping | ✅ Native `lgb.train` + `lgb.early_stopping(N)` + `lgb.log_evaluation(0)` callbacks |
| `predict_proba(X)` → probabilities | ✅ Returns `(n, 2)` float64; rows sum to 1; sklearn convention |
| `save(path)` / `load(path)` — joblib + metadata sidecar | ✅ `lightgbm_model.joblib` + `lightgbm_model_manifest.json`; `load` is `@classmethod` with `TypeError`/`FileNotFoundError` guards |
| Metadata: params, feature names, schema hash, content hash | ✅ All 4 + `best_iteration`, `best_score`, `scale_pos_weight`, `random_state`, `num_boost_round`, `early_stopping_rounds`, `schema_version` (12 keys total) |
| `feature_importance(importance_type="gain")` → DataFrame | ✅ Returns `pd.DataFrame[feature, importance]`, sorted descending; supports `"gain"` and `"split"` |
| Uses `scale_pos_weight` (not `is_unbalance`) | ✅ Default `(neg / pos)` from y_train; override via `scale_pos_weight` kwarg |
| Tests: fit on tiny data | ✅ `TestFit` (5 tests, ~600-row synthetic) |
| Tests: save/load identity check | ✅ `TestSaveLoad::test_load_round_trip_predicts_identically` (`np.testing.assert_array_equal`) |
| Tests: predict_proba shape correct | ✅ `TestPredictProba::test_predict_proba_shape_and_invariants` (n×2; rows sum to 1) |

## Test inventory

**`tests/unit/test_lightgbm_model.py`** — 24 tests across 5 classes:

| Class | Count | Coverage |
|---|---|---|
| `TestFit` | 5 | Fitted state populated; scale_pos_weight from data; explicit override; column-mismatch raises; one-class y_train raises |
| `TestPredictProba` | 5 | Shape `(n, 2)`; pre-fit raises; missing columns raise; empty input → `(0, 2)`; reordered columns → identical predictions |
| `TestSaveLoad` | 6 | Both files written; manifest has expected keys + types; round-trip predicts bit-identically; wrong type → `TypeError`; missing file → `FileNotFoundError`; pre-fit save → `AttributeError` |
| `TestFeatureImportance` | 5 | DataFrame with `feature` + `importance`; sorted descending; both `gain` and `split`; invalid type → `ValueError`; pre-fit raises |
| `TestErrorHandling` | 3 | `num_boost_round=0` raises; `early_stopping_rounds=0` raises; caller `params` override settings defaults |

**Wall: 3.46s** for the 24-test suite (synthetic data; small `num_boost_round`).

## Files-changed table

| File | Change | LOC |
|---|---|---|
| `src/fraud_engine/models/lightgbm_model.py` | new | +516 |
| `src/fraud_engine/models/__init__.py` | add `LightGBMFraudModel` re-export; updated module docstring | +9 |
| `tests/unit/test_lightgbm_model.py` | new | +312 |
| `sprints/sprint_3/prompt_3_3_a_report.md` | this file | (this file) |

## Decisions worth flagging

1. **Native `lgb.Booster` API (NOT sklearn `LGBMClassifier`).** Per spec. Diverges from Sprint 1's baseline which used `LGBMClassifier`. The native API gives explicit early stopping (the sklearn wrapper handles it implicitly), first-class `feature_importance` (no `.booster_` indirection), and a smaller serialised payload. Sprint 1's baseline stays unchanged — it remains the AUC anchor and the only MLflow consumer.

2. **`predict_proba` returns `(n, 2)` not `(n,)`.** The Booster's native `predict()` returns 1-D `P[class=1]`. We stack `[1 - p, p]` to match sklearn's convention so downstream `roc_auc_score(y, proba[:, 1])` calls work identically across wrappers. Cost: 2× memory in the prediction array; negligible.

3. **`scale_pos_weight` defaults to `(neg / pos)` computed from `y_train`.** Mirrors the standard imbalanced-classification practice. Override at construction via the `scale_pos_weight` kwarg. Spec mandates `scale_pos_weight` over `is_unbalance` — the latter is a coin-flip flag that loses the severity gradient.

4. **Filename stable, content hash in manifest.** Sprint 1's baseline embeds `content_hash[:12]` in the filename (`baseline_temporal_a3f...joblib`); we keep `lightgbm_model.joblib` stable and surface the hash via the manifest. Reason: downstream callers should target a well-known path (e.g. `models/pipelines/lightgbm_model.joblib`); the manifest is where provenance lives.

5. **Manifest schema version starts at 1.** Bump on non-backward-compatible JSON shape changes. Mirrors `_FEATURE_MANIFEST_SCHEMA_VERSION` in `pipeline.py` and `_GRAPH_MANIFEST_SCHEMA_VERSION` in `tier5_graph.py`.

6. **Schema hash truncated to 16 hex chars; content hash full 64.** Schema hash mirrors `data/lineage.py:_schema_fingerprint` (16 chars = 64 bits, sufficient for collision-resistant fingerprinting); content hash mirrors `models/baseline.py:_sha256_joblib` (full SHA-256). Two different concerns: schema hash detects "the input frame I'm about to score doesn't match what was trained on"; content hash detects bit-level drift across re-saves.

7. **`# noqa: N803` on `X_train` / `X_val`.** Project lint rules forbid uppercase argument names, but the spec mandates the sklearn-convention `X_train, y_train, X_val, y_val` signature. Sprint 1's `baseline.py` uses lowercase `x_train`; the new wrapper follows spec. Per-line `noqa` with justification.

## Verbatim verification output

### Cheap gates
```
$ make format && make lint && make typecheck
uv run ruff format src tests scripts
90 files left unchanged
uv run ruff check src tests scripts
All checks passed!
uv run mypy src
Success: no issues found in 34 source files
```

### Spec-named verification
```
$ uv run pytest tests/unit/test_lightgbm_model.py -v
============================= test session starts ==============================
collected 24 items

tests/unit/test_lightgbm_model.py::TestFit::test_fit_populates_fitted_state PASSED
tests/unit/test_lightgbm_model.py::TestFit::test_fit_computes_scale_pos_weight_from_data PASSED
tests/unit/test_lightgbm_model.py::TestFit::test_fit_uses_explicit_scale_pos_weight_override PASSED
tests/unit/test_lightgbm_model.py::TestFit::test_fit_column_mismatch_raises PASSED
tests/unit/test_lightgbm_model.py::TestFit::test_fit_one_class_y_train_raises PASSED
tests/unit/test_lightgbm_model.py::TestPredictProba::test_predict_proba_shape_and_invariants PASSED
tests/unit/test_lightgbm_model.py::TestPredictProba::test_predict_proba_before_fit_raises PASSED
tests/unit/test_lightgbm_model.py::TestPredictProba::test_predict_proba_missing_columns_raises PASSED
tests/unit/test_lightgbm_model.py::TestPredictProba::test_predict_proba_empty_dataframe_returns_empty_2d PASSED
tests/unit/test_lightgbm_model.py::TestPredictProba::test_predict_proba_handles_reordered_columns PASSED
tests/unit/test_lightgbm_model.py::TestSaveLoad::test_save_writes_model_and_manifest PASSED
tests/unit/test_lightgbm_model.py::TestSaveLoad::test_manifest_has_expected_shape PASSED
tests/unit/test_lightgbm_model.py::TestSaveLoad::test_load_round_trip_predicts_identically PASSED
tests/unit/test_lightgbm_model.py::TestSaveLoad::test_load_rejects_wrong_object_type PASSED
tests/unit/test_lightgbm_model.py::TestSaveLoad::test_load_missing_file_raises PASSED
tests/unit/test_lightgbm_model.py::TestSaveLoad::test_save_before_fit_raises PASSED
tests/unit/test_lightgbm_model.py::TestFeatureImportance::test_returns_dataframe_with_correct_columns_and_length PASSED
tests/unit/test_lightgbm_model.py::TestFeatureImportance::test_sorted_descending_by_importance PASSED
tests/unit/test_lightgbm_model.py::TestFeatureImportance::test_supports_both_gain_and_split PASSED
tests/unit/test_lightgbm_model.py::TestFeatureImportance::test_invalid_importance_type_raises PASSED
tests/unit/test_lightgbm_model.py::TestFeatureImportance::test_before_fit_raises PASSED
tests/unit/test_lightgbm_model.py::TestErrorHandling::test_invalid_num_boost_round_raises PASSED
tests/unit/test_lightgbm_model.py::TestErrorHandling::test_invalid_early_stopping_rounds_raises PASSED
tests/unit/test_lightgbm_model.py::TestErrorHandling::test_params_override_settings_defaults PASSED

======================= 24 passed, 14 warnings in 3.46s ========================
```

### Regression: `make test-fast`
```
420 passed, 34 warnings in 83.32s (0:01:23)
```
Up from 395 (Tier-5 baseline) → +24 new tests + 1 misc adjustment. No regressions.

## Surprising findings

1. **`lgb.train` accepts `n_estimators` only via the params dict.** The native API uses `num_boost_round` as the explicit kwarg, and `n_estimators` (sklearn-API name) lands in the params dict where lgb logs a warning about an unknown param. The wrapper pops `n_estimators` from the inherited `Settings.lgbm_defaults` dict and routes it to `num_boost_round` instead, suppressing the warning. Documented inline.

2. **`booster.best_score` shape is `{val_name: {metric_name: score}}`** — a nested dict. The wrapper extracts the metric value via the params-derived metric name (defaults to `"auc"`), with a fallback to the first available metric if the configured one isn't found. Edge case: `lgb.early_stopping` may emit warnings about the metric being None; passing `verbose=False` to the callback suppresses those.

3. **Sprint 1's `baseline.py` uses lowercase `x_train`.** Project lint rule N803 ("argument name should be lowercase") enforces this. The 3.3.a spec mandates uppercase `X_train` (sklearn convention). Resolution: per-line `# noqa: N803` with justification. Could've been a project-wide config to allow uppercase X (sklearn convention), but per-line keeps the rule tight elsewhere.

## Out of scope (Sprint 3 follow-on)

- **Hyperparameter tuning sweep** (next prompt) using this wrapper — Optuna over the 12-generator Tier-5 features to recover val AUC toward the 0.93-0.94 envelope.
- **Calibration** (Sprint 4 territory) — sigmoid / isotonic on `predict_proba(X)[:, 1]`.
- **MLflow integration** for the wrapper. Sprint 1's `train_baseline` does this for its baseline; the new wrapper deliberately omits it to keep the surface minimal. Future sprints can layer MLflow on top.
- **Multi-class support.** The wrapper is binary-only (`objective="binary"`); IEEE-CIS is binary fraud, no multi-class need.

## Acceptance checklist

- [x] Branch `sprint-3/prompt-3-3-a-lightgbm-model-wrapper` off `main` (`7d317bf`)
- [x] `src/fraud_engine/models/lightgbm_model.py` created (516 LOC; `LightGBMFraudModel` + 7-trade-off docstring)
- [x] `src/fraud_engine/models/__init__.py` re-exports `LightGBMFraudModel` (alphabetised)
- [x] `tests/unit/test_lightgbm_model.py` created (24 tests across 5 classes)
- [x] Fit on tiny data passes (synthetic 600 rows; populates booster_, feature_names_, etc.)
- [x] `predict_proba` shape `(n, 2)` correct; rows sum to 1
- [x] save/load identity check passes (bit-identical predictions after round-trip)
- [x] Manifest carries: schema_version, params, feature_names, n_features, best_iteration, best_score, scale_pos_weight, num_boost_round, early_stopping_rounds, random_state, schema_hash (16 hex), content_hash (64 hex)
- [x] `scale_pos_weight` used (not `is_unbalance`); default `(neg / pos)` from y_train
- [x] `feature_importance(importance_type="gain")` returns sorted DataFrame
- [x] `make format && make lint && make typecheck` all return 0
- [x] `make test-fast` returns 0 (420 tests; +24 new, no regressions)
- [x] `uv run pytest tests/unit/test_lightgbm_model.py -v` returns 0 (24 passed in 3.46s)
- [x] `sprints/sprint_3/prompt_3_3_a_report.md` written
- [x] No git/gh commands run beyond §2.1 carve-out (branch create only)

Verification passed. Ready for John to commit on `sprint-3/prompt-3-3-a-lightgbm-model-wrapper`.

**Commit note:**
```
3.3.a: LightGBMFraudModel (native Booster wrapper with early stopping + joblib/manifest persistence)
```
