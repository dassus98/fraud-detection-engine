# Sprint 3 — Prompt 3.4.c: Model A/B/C side-by-side comparison

**Date:** 2026-05-02
**Branch:** `sprint-3/prompt-3-4-c-model-comparison` (off `main` @ `f7a0afa`)
**Status:** Verification passed. Documentation-only; no new code or model artefacts.

## Headline

| Acceptance gate | Status |
|---|---|
| Side-by-side comparison covers all 5 spec dimensions (ROC-AUC, AUC-PR, p95 latency, training time, interpretability) | PASS |
| All 3 models present in every dimension | PASS |
| Decision (A=production / B=shadow / C=batch-only) stated and justified per spec | PASS |
| Methodology caveat present (training-set sizes differ — A/414K, B/50K, C/5K) | PASS |
| Sprint 4 follow-ons listed | PASS |

5 of 5 unique acceptance gates met. The deliverable is the durable in-repo artefact `reports/model_comparison.md`.

## Summary

- **`reports/model_comparison.md`** is the single document that puts the three Sprint-3 models side by side along the spec dimensions and locks in the production / shadow / batch-only role assignment that CLAUDE.md §3 declared at architecture-design time. Numbers come straight from the per-model training reports (`model_{a,b,c}_training_report.md`). Sections: decision summary, methodology + caveat, side-by-side comparison table, per-model deep dive (architecture / strengths / constraints / role), decision rationale per role, Sprint 4 follow-ons, cross-references.
- **`.gitignore`** whitelists the new `reports/model_comparison.md` (mirrors the pattern used for the per-model training reports).
- **No new code, no new tests, no model retraining.** Cheap gates trivially green; `make test-fast` matches the 3.4.b post-merge baseline of 447 unit tests passing.

## Spec vs. actual

| Spec line | Actual |
|---|---|
| Side-by-side: AUC, AUC-PR, inference p95, training time, interpretability | All five dimensions present in the comparison table at `reports/model_comparison.md`. Supporting context table adds training rows, params, calibration, persistence, and inductive-scoring support. |
| Decision: A is production, B is shadow candidate, C is batch-only | Stated as the report's headline + restated in the per-role rationale section with metric-backed justifications. |
| `Review the report for completeness.` | All 5 spec dimensions covered; all 3 models present in each row; methodology caveat prominent; Sprint 4 follow-ons enumerated; cross-references to per-model reports + Sprint-3 prompt reports + CLAUDE.md §3. |
| Completion report: `sprints/sprint_3/prompt_3_4_c_report.md` | This file. |

## The decision (verbatim from the report)

| Model | Role | One-line rationale |
|---|---|---|
| **A — LightGBM** | **Production** | p95 = 3.29 ms (under the 15 ms hot-path budget); SHAP-explainable; calibrated probabilities (val log loss 0.291 → 0.109, 62 % better) |
| **B — FraudNet** | **Shadow candidate** | Diverse signal (entity embeddings + focal loss); p95 = 60 ms (above budget — production-blocking); ensemble candidate, not primary |
| **C — FraudGNN** | **Batch-only** | Captures graph topology no other model can; cached-logits design gives 0.07 ms p95; transductive contract (no inductive scoring); feeds Model A as features in Sprint 5 |

## Headline numbers (from the comparison table)

| Dimension | A — LightGBM | B — FraudNet | C — FraudGNN |
|---|---|---|---|
| ROC-AUC val / test | **0.8281 / 0.8070** | 0.8183 / 0.8229 | 0.7778 / 0.7929 |
| PR-AUC val / test | **0.3814 / 0.4220** | 0.3351 / 0.3259 | 0.2099 / 0.2108 |
| Inference p95 | 3.29 ms | 59.94 ms | **0.072 ms** |
| Training time | ~22 min (full + 100 trials) | ~13 s (50K + 7 ep) | ~1 s (5K + 5 ep) |
| Interpretability | SHAP TreeExplainer | gradient × input / IG | GNNExplainer (out of scope) |

## Files-changed table

| File | Change | LOC |
|---|---|---|
| `reports/model_comparison.md` | new | +205 |
| `.gitignore` | whitelist 1 path | +1 |
| `sprints/sprint_3/prompt_3_4_c_report.md` | this file | (this file) |

## Decisions worth flagging

1. **Methodology caveat is prominent, not a footnote.** Model A trained on 414K, B on 50K, C on 5K — direct AUC comparisons are misleading. The caveat sits **before** the comparison table so a reviewer reads it before they read any numbers. Sprint 4 will retrain B + C on full data + calibrate; until then the comparison report frames AUC as "evidence supporting roles," not "horse race."

2. **No fourth row for the Sprint 1 Tier-1-only baseline (val AUC 0.9247).** The spec said "side-by-side A/B/C", and adding a fourth column would dilute the comparison. The Sprint 1 baseline is mentioned in Model A's deep-dive narrative (it's the anchor that makes the Tier-2-5 regression visible) but doesn't appear in the table.

3. **No quantitative ensemble forecast.** Without actually running an ensemble, any number would be speculation. The report mentions "expected based on error decorrelation visible in Model B's cleaner test/val parity" but declines to put a specific lift estimate. Sprint 4 will produce the actual number.

4. **Per-model "deep dive" is structured (architecture / strengths / constraints / role), not narrative.** Makes the report scannable for a recruiter who reads it in 90 seconds, while still carrying the trade-offs a senior reviewer would want.

## Surprising findings

1. **Model B's test/val parity is cleaner than Model A's.** FraudNet: val 0.8183 / test 0.8229 (test > val). Model A: val 0.8281 / test 0.8070 (val > test). On a temporal split, that's an interesting hint — Model B's entity embeddings appear to generalise across the time gap better than LightGBM's tree splits. It's the strongest piece of evidence in the report that the ensemble has lift.

2. **Model C's training was still climbing at epoch 5** (val AUC 0.4546 → 0.6081 → 0.7645 → 0.7736 → 0.7778, no early stop fired). The 5K + 5-epoch smoke under-runs the model's actual capacity; the production-scale Sprint 4 retrain is expected to land materially higher than the 0.7778 reported.

3. **Latency span is two orders of magnitude.** A: 3.29 ms; B: 60 ms; C: 0.07 ms. The cached-logits design gives Model C a free-lunch latency profile that no other model can match — but ONLY for transactions in the persisted graph. That's the architectural reason batch-only fits.

## Verbatim verification output

### Cheap gates

```
$ make format && make lint && make typecheck
uv run ruff format src tests scripts
103 files left unchanged
uv run ruff check src tests scripts
All checks passed!
uv run mypy src
Success: no issues found in 38 source files
```

### Unit-test regression (`make test-fast`)

```
447 passed, 34 warnings in 75.33s (0:01:15)
```

(Same baseline as 3.4.b post-merge; no Python files were modified in this prompt.)

### Report-completeness gate (per spec: "Review the report for completeness.")

Content checklist (all confirmed against `reports/model_comparison.md`):

- [x] Decision summary section with role assignments
- [x] Methodology + caveat (training-set sizes differ)
- [x] Side-by-side comparison table covering all 5 spec dimensions
- [x] Supporting-context table (training rows, params, calibration, persistence, inductive support)
- [x] Per-model deep dive — Model A
- [x] Per-model deep dive — Model B
- [x] Per-model deep dive — Model C
- [x] Decision rationale — Why A is production
- [x] Decision rationale — Why B is shadow candidate
- [x] Decision rationale — Why C is batch-only
- [x] Sprint 4 follow-ons enumerated (retrain, calibrate, ensemble, cost-curve, stratified, inductive, interpretability)
- [x] Cross-references to per-model reports + Sprint-3 prompt reports + CLAUDE.md §3

## Out of scope (Sprint 4+ territory)

- Retraining Models B + C on full IEEE-CIS for AUC-parity comparison (the parity caveat above goes away then).
- Calibration of B + C via the 3.3.c toolkit.
- Cost-curve evaluation + economic threshold optimisation (Sprint 4's headline deliverable).
- Stratified metrics by amount / `ProductCD` / time bucket.
- Ensemble blending experiments (the actual code, not the prediction in the report).
- Interpretability implementations (SHAP TreeExplainer for A in Sprint 5; integrated gradients for B and GNNExplainer for C are deferred).
- Updating CLAUDE.md §3 — the report cites and reinforces §3 but doesn't modify it (the architectural decision is unchanged).

## Acceptance checklist

- [x] Branch `sprint-3/prompt-3-4-c-model-comparison` off `main` (`f7a0afa`)
- [x] `reports/model_comparison.md` created with all 5 spec dimensions, comparison table, per-model deep dive, decision rationale, Sprint 4 follow-ons
- [x] `sprints/sprint_3/prompt_3_4_c_report.md` written
- [x] `.gitignore` whitelists `reports/model_comparison.md`
- [x] `make format && make lint && make typecheck` all return 0
- [x] `make test-fast` returns 0 (447 unit tests pass; no regressions)
- [x] No git/gh commands run beyond §2.1 carve-out (branch create only)

Verification passed. Ready for John to commit on `sprint-3/prompt-3-4-c-model-comparison`.

**Commit note:**
```
3.4.c: A/B/C side-by-side comparison + production/shadow/batch role assignment
```
