# Sprint 5 — Prompt 5.1.e: `ShapExplainer` (TreeExplainer + reason-code mapping)

**Date:** 2026-05-09
**Branch:** `sprint-5/prompt-5-1-e-shap-explainer` (off `main` @ `f78f287` — post 5.1.d merge)
**Status:** Verification passed; all spec gates met.

## Headline

| Acceptance gate | Realised | Status |
|---|---|---|
| Precomputed TreeExplainer on startup | `shap.TreeExplainer(model.booster_)` constructed in `__init__`; cached for process lifetime; rebuilt only on `reload(model)` | PASS |
| `top_k_contributions(features, k=3) → list[(feature_name, shap_value, direction)]` | Returns `Contribution(NamedTuple)` list — sorted by |shap_value| descending; zero entries dropped; default k=3 | PASS |
| `map_to_reasons(contributions) → list[str]` | Translates via YAML; null-text drops; unmapped features drop silently; preserves contribution order | PASS |
| `configs/reason_codes.yaml` — 20-30 entries | **24 entries** across Tier-1 (5) + Tier-2 (3) + Tier-3 (6) + Tier-4 (5) + Tier-5 (5) | PASS |
| Tests: SHAP values sum correctly (base + contributions == prediction) | `TestSumCheckInvariant` — 2 tests (zeros + random input); `np.isclose(base + sum(shap[0]), raw_logit, atol=1e-5)` | PASS |
| Tests: reason mapping returns clean strings | `TestMapToReasons` — 6 tests covering high/low/null/unmapped/order/empty | PASS |
| `uv run pytest tests/unit/test_shap_explainer.py -v` | **32 passed in 4.66 s** | PASS |

7 of 7 spec gates met. Plus: `make format`, `ruff check`, `mypy --strict src` all green; full unit-test regression at **751 passed** (718 post-5.1.d baseline + 32 new + 1 baseline shift); all 12 pre-commit hooks pass on the new files.

## Summary

- **`src/fraud_engine/api/shap_explainer.py`** (NEW, 511 LOC) ships `ShapExplainer` — wraps `shap.TreeExplainer` against the production LightGBM booster + the reason-codes YAML. Plus `Contribution` NamedTuple matching the spec's `(feature_name, shap_value, direction)` shape with attribute access. Module docstring carries explicit "Business rationale" + "Trade-offs considered" sections covering all 6 load-bearing decisions. Plus 4 module-private helpers (`_load_reason_codes`, `_validate_reason_codes_shape`, `_expected_value_to_scalar`, `_shap_values_to_row_array`) with defensive coercion for SHAP-version variation.
- **`configs/reason_codes.yaml`** (NEW, 134 LOC) ships the hand-curated 24-entry feature-name → user-facing-text mapping. Each entry carries `high` (text for SHAP > 0) and `low` (text for SHAP < 0; often `null`). Vesta-anonymised V/C/D/M and `is_null_*` indicators deliberately excluded — no honest reason text we can publish. Documented per-tier with rationale comments.
- **`src/fraud_engine/api/__init__.py`** (MODIFIED, +4 LOC) re-exports `ShapExplainer` + `Contribution` (alphabetised in `__all__`).
- **`tests/unit/test_shap_explainer.py`** (NEW, 529 LOC) ships 32 tests across 8 test classes including the load-bearing sum-check invariant (2 tests against the real booster).
- **No changes** to `Settings`, any pandera schema, any feature/model module, the Makefile, `pyproject.toml` (`shap==0.46.0` already pinned), `ruff.toml`, `mypy.ini`, `docker-compose.dev.yml`, `CLAUDE.md`. No new dependencies.

## Spec vs. actual

| Spec line | Actual |
|---|---|
| Precomputed `TreeExplainer` on startup | `__init__` constructs once via `shap.TreeExplainer(model.booster_)`. ~50–200ms construction cost paid at startup; per-call `shap_values(X)` is ~30–50ms. |
| `top_k_contributions(features, k=3) → list of (feature_name, shap_value, direction)` | Returns `list[Contribution]` where `Contribution` is a `NamedTuple` (still a tuple — `isinstance(c, tuple)` is True). Sorted by `abs(shap_value)` descending; zero entries dropped. Default `k=3`. Negative `k` raises `ValueError`. |
| `map_to_reasons(contributions) → list of human-readable strings` | Looks up each contribution's `feature_name` in YAML; uses `entry["high"]` if direction == "increases_risk" else `entry["low"]`. Drops contributions whose feature is unmapped OR whose direction has `null` text. Preserves order. |
| `configs/reason_codes.yaml` 20-30 entries | 24 entries. `high` + `low` keys per entry; one or both may be `null`. |
| Spec example: `amt_zscore_vs_card1_history: {high: "Transaction amount unusual...", low: null}` | Verbatim shape replicated; the YAML's actual entry uses spec text. |
| Tests: SHAP values sum correctly | `TestSumCheckInvariant::test_zeros_input_sum_check` + `test_random_input_sum_check` — both verify `np.isclose(expected_value + sum(shap[0]), booster.predict(X, raw_score=True)[0], atol=1e-5)`. |
| Tests: reason mapping returns clean strings | `TestMapToReasons` (6 tests) + `TestReasonCodesYaml` (4 tests) — covers high/low/null/unmapped/order/empty + the actual YAML's structural invariants (every key in manifest, no shell injection, etc.). |

## Test inventory

32 tests across 8 contract surfaces:

| Class | Count | Coverage |
|---|---|---|
| `TestExplainerInit` | 4 | default construction succeeds; missing reason_codes raises `FileNotFoundError`; non-mapping YAML root raises `TypeError`; missing `low` key raises `ValueError` |
| `TestTopKContributions` | 6 | default k=3; respects custom k; negative k raises; sorted by abs(shap) descending; direction matches sign; **zero-shap entries dropped** (mock the explainer to return a row with one zero, assert it's absent from output) |
| `TestMapToReasons` | 6 | high text used for increases_risk; low text used for decreases_risk; null text drops; unmapped feature drops silently; mixed mapped+unmapped preserves order; empty input → empty output |
| `TestSumCheckInvariant` | 2 | **the load-bearing test** — `expected_value + sum(shap[0]) ≈ booster.predict(X, raw_score=True)[0]` to atol=1e-5 on (a) zeros input + (b) random Gaussian input |
| `TestReasonCodesYaml` | 4 | the actual YAML loads cleanly + has ≥20 entries; every entry has at least one non-null direction; all keys exist in model manifest's `feature_names`; no `<` `>` `${` or backticks in any text (defensive XSS / shell-injection check) |
| `TestReload` | 2 | reload-with-same-model is idempotent (same shap values to atol=1e-9); reload-with-unfitted-model raises `RuntimeError` |
| `TestEndToEnd` | 1 | full chain smoke: features → top_k_contributions → map_to_reasons → list of non-empty strings |
| `TestPrivateHelpers` | 7 | `_expected_value_to_scalar` handles ndarray + scalar; `_shap_values_to_row_array` handles 2D ndarray + legacy 2-element list; bad shape raises; `_validate_reason_codes_shape` rejects non-dict root + empty-string text |

### Unit-test regression: 751 passed

Up from 718 post-5.1.d baseline by +33: 32 new in `test_shap_explainer.py` + 1 baseline shift. No regressions.

## Files-changed table

| File | Change | LOC |
|---|---|---|
| `src/fraud_engine/api/shap_explainer.py` | new (`ShapExplainer` class + `Contribution` NamedTuple + 4 module helpers + comprehensive docstring with 6 trade-offs) | +511 |
| `configs/reason_codes.yaml` | new (24 entries with `high`/`low` keys + per-tier rationale comments) | +134 |
| `src/fraud_engine/api/__init__.py` | re-export `ShapExplainer` + `Contribution` (alphabetised) | +4 / -0 |
| `tests/unit/test_shap_explainer.py` | new (32 tests across 8 classes including the sum-check invariant) | +529 |
| `sprints/sprint_5/prompt_5_1_e_report.md` | this file | (this file) |

**No changes** to `Settings`, any pandera schema, any feature/model module, Makefile, `pyproject.toml`, `ruff.toml`, `mypy.ini`, `docker-compose.dev.yml`, `CLAUDE.md`. No new dependencies.

## Decisions worth flagging

1. **`Contribution(NamedTuple)` over raw tuple, frozen dataclass, or Pydantic `Reason`.** The spec says "list of (feature_name, shap_value, direction)" — NamedTuple keeps that shape (`isinstance(c, tuple)` is True; positional unpacking works) while adding attribute access (`c.feature_name`) that makes consumer code readable. Pydantic `Reason` (5.1.a) would couple the explainer to the API schema; the route handler (5.1.f) converts at the boundary instead.

2. **Sort by `abs(shap_value)` descending; drop zero entries.** A top-3 list should surface the strongest drivers regardless of direction — a +0.4 alongside a -0.3 is more informative than two +0.4s. Zero contributions are noise (the booster never split on that feature for this row); including them would generate "transaction blocked because [V137 contributed nothing]" — meaningless.

3. **Silent drop on unmapped features in `map_to_reasons`.** The YAML covers ~24 of 743 features by design (Vesta-anonymised V/C/D/M and `is_null_*` indicators are deliberately excluded because there's no honest reason text we can publish). Raising on every unmapped feature would force every test fixture to populate the YAML with the test's specific feature set; silent drop matches the production semantic (we couldn't explain this feature; show the others).

4. **`shap.TreeExplainer` constructed at startup, not per-request.** Construction is ~50–200ms for a 743-feature LightGBM; per-call `shap_values(X)` is ~30–50ms. Caching the explainer cuts the per-request budget to just the per-call cost. `reload(model)` is the only path that re-runs the constructor — same atomic-swap semantic as Sprint 5.1.d's `InferenceService.reload`.

5. **SHAP runs against the raw booster, NOT the calibrator.** TreeExplainer is a tree-model technique; the isotonic calibrator (Sprint 3.3.c) is not a tree, so it can't be explained the same way. Contributions are in **log-odds space** (the model's raw output), which is the natural additive scale where the sum-check invariant holds. The route handler (5.1.f) knows the calibrated probability for the API surface; the per-feature contributions are about the raw model's reasoning. Documented inline + in the module docstring + reaffirmed by the `TestSumCheckInvariant` test using `booster.predict(..., raw_score=True)`.

6. **`expected_value` defensively coerced to scalar float.** SHAP 0.46.0 returns `expected_value` as a numpy `ndarray` of shape `(1,)` for LightGBM binary classifiers (verified empirically before coding — see "Surprising findings" #1). `np.asarray(ev).reshape(-1)[0]` extracts the scalar regardless of the underlying shape; future SHAP versions changing the shape (back to scalar, or to a list) don't break the contract. Same defensive pattern in `_shap_values_to_row_array` for the older SHAP API's 2-element list shape.

7. **YAML reason text is defensively HTML/shell-safe.** `TestReasonCodesYaml::test_no_shell_injection_or_html` rejects any text containing `<`, `>`, `${`, or backticks. Reason text is downstream-serialised to JSON by FastAPI in 5.1.f; while FastAPI's JSON encoder handles this correctly, the YAML-side check catches a future text edit that introduces unsafe characters at authoring time.

8. **`_REQUIRED_DIRECTION_KEYS` validator catches typos.** If a YAML edit writes `hihg: "..."` instead of `high: "..."`, `entry.get("high")` returns None silently and the contribution drops. The validator at load time enforces both `high` and `low` keys are present (one or both can be `null`), turning the typo into a loud `ValueError("missing required key 'high'")`.

## Surprising findings

1. **SHAP 0.46.0's `expected_value` is a numpy `ndarray` of shape `(1,)`, not a scalar float** — verified empirically before coding via a `python3 -c` script. SHAP's UserWarning "shap values output has changed to a list of ndarray" fires but the actual return is a single 2D array `(1, 743)`. This forced two defensive helpers (`_expected_value_to_scalar` + `_shap_values_to_row_array`) handling all three shapes (scalar, ndarray-shape-(1,), 2-element list) so future SHAP version drifts don't break the explainer.

2. **The deprecation warning fires inside SHAP's own code paths** during `shap_values(X)` calls. Tests emit ~900 warnings per session from this single deprecation. Acceptable noise — would require pinning a SHAP version range or filtering warnings module-wide; both are over-engineering for 5.1.e.

3. **Two ruff PLR2004 magic-value warnings on `2`** in the SHAP-shape-coercion helper. Resolved with module constants `_BINARY_CLASSES = 2` and `_NDIM_2D = 2` plus rationale comments.

4. **Test for `test_shap_values_bad_shape_raises` initially regex-mismatched** — used `match="shape"` but the error message contains `"n_features"` not `"shape"`. Fixed by updating the test's match pattern.

5. **The ruff auto-fix sorted imports differently than expected** — `from fraud_engine.api.shap_explainer import ...` ended up after the `from fraud_engine.api.schemas import (...)` block in `__init__.py`. Ruff's import-sort logic treats the parenthesised import as one group; the ordering is correct alphabetically by module path. No semantic impact.

## Verbatim verification output

### Cheap gates

```
$ uv run ruff format src/fraud_engine/api/shap_explainer.py \
                     src/fraud_engine/api/__init__.py \
                     configs/reason_codes.yaml \
                     tests/unit/test_shap_explainer.py
2 files reformatted, 2 files left unchanged

$ uv run ruff check src/fraud_engine/api tests/unit/test_shap_explainer.py
All checks passed!

$ uv run mypy src
Success: no issues found in 45 source files
```

### Spec verification

```
$ uv run pytest tests/unit/test_shap_explainer.py -v --no-cov
======================= 32 passed, 932 warnings in 4.66s ========================
```

### Unit-test regression

```
$ uv run pytest tests/unit -q --no-cov
================ 751 passed, 3282 warnings in 74.82s (0:01:14) =================
```

(Up from 718 post-5.1.d baseline by +33: 32 new in `test_shap_explainer.py` + 1 baseline shift. No regressions.)

### Pre-commit hooks (proactive, on changed files)

```
$ uv run pre-commit run --files src/fraud_engine/api/shap_explainer.py \
                                src/fraud_engine/api/__init__.py \
                                configs/reason_codes.yaml \
                                tests/unit/test_shap_explainer.py
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

- **The FastAPI `/score` route handler** that calls `ShapExplainer.top_k_contributions(...)` + `map_to_reasons(...)` and packages as `PredictionResponse.top_reasons` — Sprint 5.1.f.
- **Glob/regex pattern matching in reason_codes.yaml** — currently exact-match only; pattern support if Sprint 5.x finds the literal-name approach insufficient.
- **SHAP integration with the calibrator.** TreeExplainer works on **raw** model log-odds; calibrated probabilities aren't tree-explainable. Documented in module docstring + reaffirmed by the sum-check test using `raw_score=True`.
- **Async `top_k_contributions`** — kept synchronous; the route can `run_in_executor` if blocking the event loop becomes a problem.
- **SHAP background dataset** (`TreeExplainer(model, data=X_background)`) — uses booster's average prediction by default; sufficient for binary classification per SHAP 0.46 docs.
- **Reason-code regeneration from the manifest at runtime** — YAML is hand-curated per the spec.
- **Reason-code i18n / per-locale text** — Sprint 6 territory.
- **Vesta-anonymised feature reason codes** — would require fabricated text; deferred indefinitely.
- **`top_reasons` length cap of 10** (per `PredictionResponse.top_reasons.max_length=10` from 5.1.a) — `top_k_contributions` defaults to 3 and is bounded by k; the route handler caps at 10 if needed.
- **Concurrent-reload race test** — the same GIL-atomic-swap pattern Sprint 5.1.d's race test verified for `InferenceService.reload`. Not duplicated for `ShapExplainer.reload` — same primitive (single-attribute rebind).
- **CLAUDE.md §13 sprint-status table update** — handled by a later 5.x audit-and-gap-fill PR.

## Acceptance checklist

- [x] Branch `sprint-5/prompt-5-1-e-shap-explainer` off `main` (`f78f287`, post 5.1.d merge)
- [x] `src/fraud_engine/api/shap_explainer.py` created (511 LOC; `ShapExplainer` + `Contribution` + 4 module helpers + comprehensive docstring with 6 trade-offs)
- [x] `configs/reason_codes.yaml` created (134 LOC; 24 entries across 5 tiers with rationale comments)
- [x] `src/fraud_engine/api/__init__.py` re-exports `ShapExplainer` + `Contribution` (alphabetised)
- [x] `tests/unit/test_shap_explainer.py` created (529 LOC; 32 tests across 8 classes including sum-check invariant)
- [x] Spec gate: TreeExplainer precomputed on startup — PASS
- [x] Spec gate: `top_k_contributions(features, k=3)` returns list of `(feature_name, shap_value, direction)` — PASS
- [x] Spec gate: `map_to_reasons(contributions)` returns list of human-readable strings — PASS
- [x] Spec gate: 20-30 entries in `configs/reason_codes.yaml` — PASS (24 entries)
- [x] Spec gate: SHAP values sum correctly (base + contributions == prediction) — PASS (2 tests; atol=1e-5)
- [x] Spec gate: reason mapping returns clean strings — PASS (6 + 4 tests)
- [x] `make format` returns 0
- [x] `make lint` returns 0
- [x] `make typecheck` returns 0 (Success: no issues found in 45 source files)
- [x] `make test-fast` returns 0 (751 passed; 718 baseline + 33)
- [x] `uv run pytest tests/unit/test_shap_explainer.py -v` returns 0 (32 passed in 4.66 s)
- [x] All 12 pre-commit hooks pass on the new files
- [x] `sprints/sprint_5/prompt_5_1_e_report.md` written
- [x] No git/gh commands run beyond §2.1 carve-out (branch create only; commit + push + merge happen after John's sign-off)

Verification passed. Ready for John to commit on `sprint-5/prompt-5-1-e-shap-explainer`.

**Commit note:**
```
5.1.e: ShapExplainer (TreeExplainer + reason_codes.yaml + sum-check invariant)
```
