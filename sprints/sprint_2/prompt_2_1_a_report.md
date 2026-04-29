# Sprint 2 — Prompt 2.1.a Report: BaseFeatureGenerator + FeaturePipeline foundation

**Date:** 2026-04-27
**Branch:** `sprint-2/prompt-2-1-a-feature-base-and-pipeline` (off `main` at `a7c121b`, post-Sprint-1-audit)
**Status:** all verification gates green — `make format` (50 files unchanged), `make lint` (All checks passed), `make typecheck` (25 source files, was 23 — +2 for `base.py` + `pipeline.py`), `uv run pytest tests/unit/test_feature_base.py -v` (8 passed in 2.83 s), `make test-fast` (223 passed).

## Summary

Prompt 2.1.a is the **first prompt of Sprint 2**. It lays the abstract foundation for every subsequent feature module (Tier-1 basic, Tier-2 aggregations, Tier-3 behavioural, Tier-4 EWM, Tier-5 graph). Risk: medium-high — every later feature module hangs off this contract, so the API needed to be right the first time.

Three production modules + one test file:

- **`src/fraud_engine/features/base.py`** — `BaseFeatureGenerator(ABC)` with 4 abstract methods (`fit`, `transform`, `get_feature_names`, `get_business_rationale`) and 1 concrete method (`fit_transform`, default composition). Uses `Self` (PEP 673) as `fit`'s return type so chained calls (`MyGen().fit(df).transform(df2)`) mypy-narrow correctly to the concrete subclass. Contract documented in the class docstring; trade-offs (`ABC` vs `Protocol`, `Self` vs `BaseFeatureGenerator`, no `BaseEstimator`) documented in the module docstring.
- **`src/fraud_engine/features/pipeline.py`** — `FeaturePipeline` `@dataclass` with sequential composition. `fit_transform` and `transform` are both decorated with `@log_call` + `@lineage_step` (one lineage record per pipeline call, not per generator — see Decision 1 below). `save(path)` writes `pipeline.joblib` + `feature_manifest.json` sidecar; `load(path)` is a `@classmethod` returning `FeaturePipeline` (mirrors the post-audit `LineageLog.read` shape). Private `_build_manifest` renders the feature × generator × rationale × dtype dict.
- **`src/fraud_engine/features/__init__.py`** — replaced the one-line stub (`"""Feature engineering. Populated in Sprint 2."""`) with explicit re-exports for `BaseFeatureGenerator` and `FeaturePipeline`.
- **`tests/unit/test_feature_base.py`** — 8 unit tests across 3 contract surfaces (ABC enforcement, fit_transform chaining, save/load round-trip). Two inline stub generators (`_MeanCenter`, `_LogPlusOne`) exercise the contract without depending on Sprint 2's not-yet-built real generators.

## Spec vs. actual

| Spec element | Implementation | Status |
|---|---|---|
| `BaseFeatureGenerator(ABC)` with 4 abstract methods + 1 concrete `fit_transform` | Implemented in `base.py`; `fit` returns `Self` (PEP 673) for chained-call type-narrowing | ✓ |
| `FeaturePipeline` composing generators sequentially | `@dataclass FeaturePipeline` with `generators: list[BaseFeatureGenerator]` | ✓ |
| `save(path)` / `load(path)` via joblib | `save` writes joblib + manifest sidecar; `load` is `@classmethod` with isinstance guard | ✓ (intentional: classmethod load + tuple return on save) |
| `feature_manifest.json` listing every feature × generator × rationale × output dtype | Rendered by private `_build_manifest`; `dtype` falls back to `"unknown"` when `last_output_dtypes` is None | ✓ |
| Test: instantiating concrete subclass without abstract methods fails | `test_subclass_missing_method_cannot_instantiate` | ✓ |
| Test: pipeline fit→transform produces expected columns | `test_pipeline_fit_transform_chains_generators` | ✓ |
| Test: pipeline save→load→transform produces identical output | `test_load_round_trip_produces_identical_output` (uses `pd.testing.assert_frame_equal` for bit-exact comparison) | ✓ |

**Gap analysis: zero substantive gaps.** Two intentional design improvements over the literal spec:

1. **`Self` instead of `BaseFeatureGenerator` as `fit`'s return type** (Decision 2 in the plan). Behaviourally identical; mypy strict enforcement is stricter at chained-call sites.
2. **`load` as `@classmethod`** rather than free function (mirrors the post-audit `LineageLog.read` shape). Stateless reader; no instance needed.

## Decisions worth flagging

### Decision 1 — Lineage at the pipeline level only

CLAUDE.md §7.2 mandates `@lineage_step` on every transformation. Two interpretations:

- **(A)** Per-generator: each subclass's `transform` is decorated → 5+ lineage records per pipeline run (with 5+ generators planned for Sprint 2).
- **(B)** Pipeline-level: only `FeaturePipeline.fit_transform` and `transform` are decorated → 1 lineage record per pipeline run.

**Adopted (B).** Mirrors how `interim_clean` is one record (not per-step inside the cleaner). The pipeline is the boundary; per-generator visibility is recoverable from `@log_call`'s `feature_pipeline.start` / `.done` events. If Sprint 5 needs per-generator lineage for compliance, lift `@lineage_step` onto each subclass's `transform` at that point. Documented in `pipeline.py`'s module docstring.

### Decision 2 — `Self` instead of `BaseFeatureGenerator` as `fit`'s return type

The spec wrote `def fit(self, df) -> BaseFeatureGenerator`. PEP 673's `Self` (Python 3.11+) preserves the concrete subclass return type, so `MyGen().fit(df).transform(df2)` mypy-narrows to `MyGen.transform`, not the abstract method. Behaviourally identical; strict-mode mypy is stricter. Documented in `base.py` trade-offs.

## Test inventory

8 unit tests, all in `tests/unit/test_feature_base.py`, all green:

| # | Class | Name | Asserts |
|---|---|---|---|
| 1 | `TestBaseFeatureGeneratorContract` | `test_cannot_instantiate_abstract_base` | `BaseFeatureGenerator()` raises `TypeError` matching `/abstract/` |
| 2 | `TestBaseFeatureGeneratorContract` | `test_subclass_missing_method_cannot_instantiate` | A subclass that defines `fit` + `transform` only (forgets `get_feature_names` + `get_business_rationale`) raises `TypeError` matching `/abstract/` |
| 3 | `TestBaseFeatureGeneratorContract` | `test_fit_transform_default_composition` | `_MeanCenter("x").fit_transform(df)` produces `feat_x_centred` column with values `[-1, 0, 1]` for input `[1, 2, 3]` |
| 4 | `TestFeaturePipelineFitTransform` | `test_pipeline_fit_transform_chains_generators` | Two-generator pipeline produces both `feat_x_centred` + `feat_y_log`; original `x` and `y` columns preserved |
| 5 | `TestFeaturePipelineFitTransform` | `test_pipeline_records_last_output_dtypes` | `last_output_dtypes` is `None` before any call; populated after `fit_transform`; contains `feat_x_centred` |
| 6 | `TestFeaturePipelineSaveLoad` | `test_save_writes_pipeline_and_manifest` | Both `pipeline.joblib` + `feature_manifest.json` written; manifest has `schema_version=1`, correct `n_features`, `name`, `generator` (`_MeanCenter`), `dtype` (`float64`), and rationale snippet |
| 7 | `TestFeaturePipelineSaveLoad` | `test_load_round_trip_produces_identical_output` | `pd.testing.assert_frame_equal(reloaded.transform(df), original_out)` — bit-exact equality |
| 8 | `TestFeaturePipelineSaveLoad` | `test_load_rejects_wrong_object_type` | Saving a `dict` at the pipeline path; `load` raises `TypeError` matching `/expected FeaturePipeline/` |

Two inline stub generators (`_MeanCenter` for the stateful path, `_LogPlusOne` for the stateless path) exercise the contract without depending on Sprint 2's not-yet-built real generators.

## Files changed

| File | Type | LOC | Purpose |
|---|---|---|---|
| `src/fraud_engine/features/base.py` | new | 138 | `BaseFeatureGenerator(ABC)` + 4 abstract methods + 1 concrete `fit_transform` |
| `src/fraud_engine/features/pipeline.py` | new | 224 | `FeaturePipeline` dataclass + `fit_transform` / `transform` (both `@log_call` + `@lineage_step`-decorated) + `save` + `load` (`@classmethod`) + private `_build_manifest` |
| `src/fraud_engine/features/__init__.py` | modified | +6 lines | Re-export `BaseFeatureGenerator` + `FeaturePipeline` |
| `tests/unit/test_feature_base.py` | new | 168 | 8 unit tests + 2 inline stub generators |
| `sprints/sprint_2/prompt_2_1_a_report.md` | new | this file | Completion report |

Total source diff: ~536 LOC (production + tests + report). The production code (base.py + pipeline.py) is 362 LOC; the test code is 168 LOC.

## Verification — verbatim output

### 1. `make format`
```
uv run ruff format src tests scripts
50 files left unchanged
```

### 2. `make lint`
```
uv run ruff check src tests scripts
All checks passed!
```

### 3. `make typecheck`
```
uv run mypy src
Success: no issues found in 25 source files
```

(Was 23 source files before this prompt; +2 for `features/base.py` and `features/pipeline.py`.)

### 4. `uv run pytest tests/unit/test_feature_base.py -v`
```
8 passed, 14 warnings in 2.83s
```

### 5. `make test-fast`
```
223 passed, 34 warnings in 6.83s
```

The +8 from this prompt sums with prior counts — pytest collection has historically drifted by a small number across sessions (documented in 1.2.b's report); the meaningful invariant is the green status, not the exact total.

## Surprising findings

1. **`make format` reported "50 files left unchanged" on the first run** — confirming the new `base.py`, `pipeline.py`, `test_feature_base.py`, and `__init__.py` were written in formatter-compatible style on the first pass. No re-staging dance during pre-commit, which is the failure mode `feedback_run_ruff_format.md` was created to prevent.
2. **mypy strict accepted `Self` cleanly** at the `fit` return type and at every chained-call site in the test stubs. The earlier audit had me worried about `# type: ignore` proliferation; in practice, modern mypy + Python 3.11+ types this correctly without any suppressions.
3. **The `@dataclass` decoration on `FeaturePipeline` was a deliberate deviation from cleaner / loader's plain-class shape.** Those modules carry per-call mutable state (`last_report`, `_log`); the pipeline only carries `generators` + `last_output_dtypes`, both of which fit the dataclass model. Trade-off: `@dataclass` gives a free `__repr__` and `__eq__` (useful in debugging) at the cost of locking the constructor signature; the module docstring documents this.
4. **`@lineage_step` accepted being layered under `@log_call` on the same method** (decorator order: `@log_call` outermost, `@lineage_step` inner). This is the standard Python decorator stacking pattern but worth confirming since the lineage decorator scans `args` for the first `pd.DataFrame` and `@log_call` doesn't intercept arguments.
5. **`load` as `@classmethod` mirrors the Sprint 1 audit's `LineageLog.read` change.** Both are stateless readers; constructing an instance just to call `.load()` would be unnecessary boilerplate. The audit-driven precedent paid off here — no instance-vs-classmethod ambiguity to debate.

## Deviations from the spec

1. **`Self` instead of `BaseFeatureGenerator` as `fit`'s return type.** Spec wrote `-> BaseFeatureGenerator`; implementation uses `Self`. Behaviourally identical for runtime; strictly better for mypy at chained-call sites. Documented in `base.py` trade-offs.
2. **`load` as `@classmethod`.** Spec showed it as a method on the class without specifying static / classmethod / instance. The classmethod choice mirrors `LineageLog.read` (stateless reader) and avoids the "construct an instance just to call load" pattern.
3. **`save` returns `tuple[Path, Path]`** (pipeline path + manifest path) for caller logging. Spec didn't specify a return type; the tuple is non-breaking and useful.

## Acceptance checklist

- [x] Branch `sprint-2/prompt-2-1-a-feature-base-and-pipeline` created off `main` (`a7c121b`) **before any edits** (per `feedback_branch_first.md`)
- [x] `src/fraud_engine/features/base.py` created (`BaseFeatureGenerator(ABC)` with 4 abstract methods + 1 concrete `fit_transform`)
- [x] `src/fraud_engine/features/pipeline.py` created (`FeaturePipeline` dataclass with `fit_transform` / `transform` / `save` / `load` / `_build_manifest`)
- [x] `src/fraud_engine/features/__init__.py` updated (re-exports `BaseFeatureGenerator` + `FeaturePipeline`)
- [x] `tests/unit/test_feature_base.py` created (8 tests + 2 stub generators)
- [x] `make format` returns 0 (run BEFORE lint per `feedback_run_ruff_format.md`)
- [x] `make lint` returns 0
- [x] `make typecheck` returns 0 (25 source files, was 23 — +2)
- [x] `make test-fast` returns 0 (223 passed)
- [x] `uv run pytest tests/unit/test_feature_base.py -v` returns 0 (8 passed)
- [x] `sprints/sprint_2/prompt_2_1_a_report.md` written (this file)
- [x] No source files outside the four declared above are modified
- [x] Per the user directive: agent will push + open PR after John commits, then ask permission for squash-merge

Verification passed. Ready for John to commit on `sprint-2/prompt-2-1-a-feature-base-and-pipeline`.

---

## Audit (2026-04-28)

Re-audit on branch `sprint-2/audit-and-gap-fill` (off `main` at `106f321`, post-Sprint-2 original audit). Goal: re-verify the 2.1.a deliverables against the spec and gap-fill anything missing.

### Findings

- **Spec coverage: complete.** All 3 spec-required tests are present (concrete-subclass-missing-method failure, pipeline fit→transform produces expected columns, pipeline save→load→transform produces identical output). Six additional contract tests exceed the spec (ABC base instantiation guard, default `fit_transform` composition, `last_output_dtypes` population, manifest shape assertions, custom `pipeline_filename` round-trip, `load` type-guard).
- **No `TODO` / `FIXME` / `XXX` / `HACK` markers** in `base.py`, `pipeline.py`, `__init__.py`, or `test_feature_base.py`.
- **No skipped or `xfail`-marked tests.**
- **One minor doc drift:** the 2026-04-27 report records **8 unit tests**; the current `tests/unit/test_feature_base.py` holds **9 tests**. The extra test is `test_save_and_load_with_custom_filename`, added during prompt 2.1.d when tier-specific pipeline filenames (`tier1_pipeline.joblib`, etc.) were introduced so `models/pipelines/` could host multiple tiers. Headline green status is unchanged. The test inventory table above pre-dates the addition.
- **Pipeline polymorphism fix from 2.2.d is in place** (`pipeline.py` lines 138–147 — `current = gen.fit_transform(current)` instead of `gen.fit(current).transform(current)`). Identity-preserving for stateless generators; engages `TargetEncoder`'s OOF override under composition.

### Verification (audit run)

```
$ uv run pytest tests/unit/test_feature_base.py -v
9 passed, 14 warnings in 3.57s
```

### Conclusion

No code changes required. The 2.1.a deliverables are spec-complete and audit-clean.
