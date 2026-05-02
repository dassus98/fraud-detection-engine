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

---

## Audit — sprint-3-complete sweep (2026-05-02) — FINAL Sprint 3 audit

Audit branch: `sprint-3/audit-and-gap-fill` off `main` @ `ad266e5`.
Goal: confirm spec deliverables and report completeness before
tagging `sprint-3-complete`. This is the **final prompt audit**
in the Sprint 3 sweep — after this, the audit branch consolidates
into one commit + PR + merge, and `sprint-3-complete` gets tagged.

### 1. Files verified

| File | Status | Notes |
|---|---|---|
| `reports/model_comparison.md` | ✅ | 16,224 bytes; 166 lines; covers all 5 spec dimensions |
| `.gitignore` | ✅ | Line 49: `!/reports/model_comparison.md` whitelist confirmed; `git check-ignore` returns nothing for the path |
| `sprints/sprint_3/prompt_3_4_c_report.md` | ✅ | this file |

All cross-referenced files in `model_comparison.md` exist on
disk:

| Cross-reference | Size | Status |
|---|---|---|
| `reports/model_a_training_report.md` | 5,147 B | ✅ |
| `reports/model_b_training_report.md` | 2,123 B | ✅ |
| `reports/model_c_training_report.md` | 2,386 B | ✅ |
| `sprints/sprint_3/prompt_3_3_d_report.md` | 41,493 B | ✅ |
| `sprints/sprint_3/prompt_3_4_a_report.md` | 47,450 B | ✅ |
| `sprints/sprint_3/prompt_3_4_b_report.md` | 48,077 B | ✅ |
| `CLAUDE.md` §3 | (root) | ✅ |

### 2. Loading verification

This is a documentation-only prompt — no integration tests, no
training runs to verify. The "review the report for
completeness" gate is verified in §3 below.

`make format && make lint && make typecheck` (post-audit) →
all green:

```
$ uv run ruff check src tests scripts
All checks passed!
$ uv run ruff format --check src tests scripts
103 files already formatted
$ uv run mypy src
Success: no issues found in 38 source files
```

The `mypy scripts/` issue that the prior three audits found and
fixed (`train_test_split` → `cast(pd.DataFrame, ...)` in
`_stratified_subsample`) does NOT recur here because no script
files are modified by 3.4.c.

### 3. Report completeness — spec gate

The spec verification command is **"Review the report for
completeness."** Cross-checked against `reports/model_comparison.md`:

| Required content | Present? | Location |
|---|---|---|
| All 5 spec dimensions: ROC-AUC, PR-AUC, p95 latency, training time, interpretability | ✅ | "Side-by-side comparison" table, lines 39-45 |
| All 3 models compared in each dimension | ✅ | Each row has columns for A, B, C |
| Decision (A=production / B=shadow / C=batch-only) stated | ✅ | "Decision summary" table, lines 13-17 |
| Decision justified per spec | ✅ | "Decision rationale" section, lines 116-136 (3 sub-sections, one per role) |
| Methodology caveat present (training-set sizes differ) | ✅ | "Methodology + caveat" section, lines 23-33; placed BEFORE the comparison table per design |
| Sprint 4 follow-ons listed | ✅ | "Sprint 4 follow-ons" section, lines 140-153 (8 explicit items) |
| Per-model deep dive for each model | ✅ | Lines 62-113 (3 sub-sections, structured architecture / strengths / constraints / role) |
| Cross-references to per-model reports + sprint reports + CLAUDE.md | ✅ | "Cross-references" section, lines 159-165 |

### 4. Number consistency — cross-checked against per-model reports

Verified that every metric in `model_comparison.md` traces back
to its source training report:

| Claim in comparison | Source report | Source value | Match? |
|---|---|---|---|
| Model A val ROC-AUC = 0.8281 | `model_a_training_report.md` line 13 | 0.8281 | ✅ |
| Model A test ROC-AUC = 0.8070 | `model_a_training_report.md` line 13 | 0.8070 | ✅ |
| Model A val log loss 0.291 → 0.109 (62 % better) | `model_a_training_report.md` lines 15-16 | 0.291436 → 0.108958 | ✅ |
| Model A val ECE 0.193 → 0.000 | `model_a_training_report.md` lines 19-20 | 0.192569 → 0.000000 | ✅ |
| Model A latency p50/p95/p99 = 2.28/3.29/7.21 ms | `model_a_training_report.md` lines 36-38 | 2.28 / 3.29 / 7.21 | ✅ |
| Model A train rows = 414,542 | `model_a_training_report.md` line 4 | 414,542 | ✅ |
| Calibration method = isotonic | `model_a_training_report.md` line 22 | isotonic | ✅ |
| Best Optuna trial AUC = 0.8281 over 100 trials | `model_a_training_report.md` line 44 | 0.8281, 100 trials | ✅ |
| Model B val ROC-AUC = 0.8183 | original 3.4.a report header | 0.8183 | ✅ |
| Model B test ROC-AUC = 0.8229 | 3.4.a report headline | 0.8229 | ✅ |
| Model B latency p95 = 59.94 ms | 3.4.a report headline | 59.94 ms | ✅ |
| Model B param count = 304,779 | 3.4.a `--quick` console output | 304,779 | ✅ |
| Model B trained on 50K rows × 7 epochs | 3.4.a report headline | 50K × 7 | ✅ |
| Model C val ROC-AUC = 0.7778 | 3.4.b report headline | 0.7778 | ✅ |
| Model C test ROC-AUC = 0.7929 | 3.4.b report headline | 0.7929 | ✅ |
| Model C latency p50/p95/p99 = 0.037/0.072/0.127 ms | 3.4.b report headline | 0.037 / 0.072 / 0.127 ms | ✅ |
| Model C trained on 5K × 5 epochs | 3.4.b report headline | 5K × 5 | ✅ |
| Model C bundle size ~1.2 GB | 3.4.b "Decisions worth flagging" #4 | 1.2-1.5 GB | ✅ |

**All 18 cross-checks pass.** No drift between the comparison
report and the per-model reports.

### 5. Lint / format / typecheck / logging / comments

- `ruff check src tests scripts` → **clean** (103 files)
- `ruff format --check src tests scripts` → **103 files already formatted**
- `mypy src` → **no issues found in 38 source files**
- No code changes in this prompt → no logging or comment audit
  applies. The markdown is well-structured with explicit section
  headers and no Lorem-ipsum placeholders.

### 6. Design rationale (the deep dive)

This is a documentation-only prompt with a clear scope ("compare
A/B/C; commit role assignments"). The design rationale is
correspondingly compact.

#### 6.1 Justifications

- **Single document at `reports/model_comparison.md`.** Per
  spec; matches the project pattern of "one markdown report per
  cross-cutting deliverable" (mirrors `model_a/b/c_training_report.md`).
- **Methodology caveat BEFORE the metrics table.** Reviewers
  who read top-down see "AUC numbers are not directly comparable"
  before they see the AUC numbers. Defensive against the obvious
  misread.
- **Per-model deep dive structured (architecture / strengths /
  constraints / role), not narrative.** A senior reviewer scans
  the strengths-vs-constraints contrast in 30 seconds; a
  recruiter reads it in 90 seconds; both extract the same
  decision-relevant signal. Narrative prose buries the trade-offs.
- **Decision rationale as 3 separate sub-sections** (one per
  role), not a single decision table. Each role is defended
  on its own merits + on what disqualifies the other two
  models from filling that role. Easier to challenge / verify
  per-role.
- **Sprint 4 follow-ons explicit (8 items).** Frames the report
  as the start of the Sprint 4 work-list, not as a Sprint 3
  victory lap. Each item is a tractable next step
  (retrain B+C on full data, calibrate B+C, ensemble blend,
  cost-curve, stratified metrics, inductive scoring for C,
  interpretability tooling per model).

#### 6.2 Consequences (positive + negative)

| Choice | Positive | Negative |
|---|---|---|
| Single comparison document | One file, easy to find | Could grow stale if Sprint 4 retrains B+C and the AUCs diverge from this snapshot |
| Methodology caveat first | Defends against AUC-only misread | Adds vertical space before the table |
| Structured deep dive | Scannable; trade-offs visible | Less narrative voice |
| Per-role rationale | Each decision defensible | Some repetition across the three sub-sections |
| 8 Sprint 4 follow-ons | Hands the next sprint a clear work-list | Implies Sprint 4's scope (calibration, ensemble, cost-curve) |
| No quantitative ensemble forecast | Honest about not having run the numbers | A reader looking for "how much will the ensemble help?" gets "Sprint 4 will tell you" |
| No fourth row for Sprint 1 baseline | Spec says "A/B/C, side-by-side" | The Sprint 1 baseline (val AUC 0.9247 on Tier-1 only) is the anchor that makes the regression visible — only mentioned in Model A's deep dive, not in the table |

#### 6.3 Alternatives considered and rejected

These were settled at plan-mode time (per the plan file
`going-forward-give-me-glittery-manatee.md`); restating for
audit completeness:

- **Add a Sprint 1 baseline row to the comparison table.**
  Rejected: the spec says "side-by-side A/B/C", not "A/B/C +
  baseline". The baseline is mentioned in Model A's deep-dive
  narrative as the anchor that makes the Tier-2-5 regression
  visible (val AUC 0.9247 → 0.8281 with full Tier-5).
- **Quantitative ensemble lift forecast.** Rejected: any number
  without actually running the ensemble would be speculation.
  The report mentions "expected based on error decorrelation
  visible in Model B's cleaner test/val parity vs A's" but
  declines to put a specific lift estimate.
- **Methodology caveat as a footnote at the bottom.** Rejected:
  reviewers who read top-down would see the AUC table first
  and risk drawing a horse-race conclusion before seeing the
  caveat. Top-of-document placement is the right defensive
  posture.
- **Pinning specific numbers in CLAUDE.md §3.** Rejected: the
  comparison report cites and reinforces §3 but doesn't modify
  it. The architectural decision (A=production, B=shadow,
  C=batch-only) is unchanged; only the metric backing is new.

#### 6.4 Trade-offs

- Document scope: comparison only vs comparison + ensemble
  experiments → **chose comparison only** (Sprint 3 spec).
- Caveat placement: top vs footnote → **chose top** (defensive
  vs misread).
- Deep-dive style: structured vs narrative → **chose structured**
  (scannable).
- Sprint 4 forecast: speculate vs decline → **chose decline**
  (honest engineering).
- Sprint 1 baseline in table: yes vs no → **chose no** (spec
  scope).

#### 6.5 Potential issues + mitigations

- **Report numbers will drift if B+C are retrained on full data
  in Sprint 4.** This is by design — the report is the snapshot
  at Sprint 3 close. Sprint 4 should produce a follow-up
  comparison report (`reports/model_comparison_v2.md` or
  similar) rather than overwriting this one. Mitigation:
  documented in the Sprint 4 follow-ons.
- **The AUC headline could mislead a reviewer who skips the
  methodology caveat.** Mitigation: the caveat is before the
  table and bolded; the per-model deep dive's "Constraints"
  sections explicitly call out training-set scale.
- **No direct ensemble blend math in the report.** A reader
  looking for "Sprint 3 also delivered an ensemble" would be
  disappointed; that's Sprint 4. Mitigation: explicit in
  "Sprint 4 follow-ons".
- **Cross-references are textual, not anchor-linked.** GitHub's
  markdown rendering doesn't have automatic cross-anchor
  validation. Mitigation: this audit cross-checked all 7
  cross-references resolve to files that exist on disk; future
  reorganisations will need to update the report's
  cross-references manually.

#### 6.6 Scalability

- **Document size: 16 KB; 166 lines.** Renders cleanly on GitHub
  and locally; no scrolling fatigue.
- **Number of models compared: 3.** Sprint 4 ensemble work may
  add a fourth row (the ensemble itself); the table layout is
  already wide enough to accommodate.

#### 6.7 Reproducibility

- **All numbers traceable to per-model reports** (verified in §4
  above). A reader can re-derive every claim by reading
  `model_{a,b,c}_training_report.md` directly.
- **Per-model reports are auto-generated by their training
  scripts.** Re-running `train_lightgbm.py / train_neural.py /
  train_gnn.py` regenerates the source reports; this comparison
  document is hand-authored and would need a manual edit to
  reflect re-runs (acceptable for a comparison-of-reports
  document; automating it would couple to the per-model report
  format and make maintenance brittle).

### 7. Gap-fixes applied on the audit branch

**None.** This is a documentation-only prompt; the report is
internally consistent and the numbers cross-check perfectly
against the per-model reports. No real findings.

### 8. Sprint 4+ follow-ons (out of scope for the audit)

The Sprint 4 follow-ons are already enumerated in the report
itself (lines 140-153). For completeness, restated here:

- Retrain B and C on full IEEE-CIS for apples-to-apples AUC
  comparison.
- Calibrate B and C via the 3.3.c toolkit
  (`select_calibration_method`).
- Cost-curve evaluation + threshold optimisation per model.
- Ensemble blend (linear stacking with `LogisticRegression`
  meta-features).
- Stratified metrics (amount bucket, `ProductCD`, time bucket)
  per model.
- Inductive scoring path for Model C (Sprint 5 prerequisite for
  the batch-feature pipeline).
- Interpretability tooling per model (SHAP for A, integrated
  gradients for B, GNNExplainer for C).

### Verbatim audit verification

```
$ uv run ruff check src tests scripts
All checks passed!

$ uv run ruff format --check src tests scripts
103 files already formatted

$ uv run mypy src
Success: no issues found in 38 source files

$ git check-ignore -v reports/model_comparison.md
(returns nothing — not ignored, whitelist confirmed)
```

### Audit verdict

**3.4.c is sound. No gap-fixes applied — none needed.** This is
the cleanest documentation-only prompt audited in the Sprint 3
sweep. All five spec dimensions present; all three models
compared in each dimension; decision (A=production,
B=shadow, C=batch-only) stated and justified per role; methodology
caveat prominent; Sprint 4 follow-ons enumerated. **All 18
metric cross-checks against per-model reports pass with no
drift.**

---

## Sprint 3 audit-and-gap-fill — END-OF-SWEEP SUMMARY

This is the final prompt in the Sprint 3 audit sweep. Closing
notes on the cumulative audit work:

### What the audit branch carries

**12 prompts audited** (3.1.a, 3.1.b, 3.2.a, 3.2.b, 3.2.c,
3.3.a, 3.3.b, 3.3.c, 3.3.d, 3.4.a, 3.4.b, 3.4.c).

**Real findings + fixes (in chronological audit order):**

| Prompt | Finding | Fix | LOC delta |
|---|---|---|---|
| 3.1.a | (none — clean) | — | 0 |
| 3.1.b | (none — clean) | — | 0 |
| 3.2.a | (none — clean) | — | 0 |
| 3.2.b | Pagerank docstring drift (claimed `max_iter=50, tol=1e-4`; actual constants `max_iter=20, tol=1e-3`) | Updated docstring trade-off #5 in `tier5_graph.py` to match constants | -3 +6 |
| 3.2.c | `reports/graph_feature_analysis.md` was gitignored (matched `/reports/*` deny rule) | Added `!/reports/graph_feature_analysis.md` to `.gitignore` allow-list; the file (9.7 KB) is now committable | +1 |
| 3.3.a | Dead helpers `_schema_fingerprint` + `_sha256_joblib` in `lightgbm_model.py` (defined, never called); module docstring trade-off #4 + cross-references referenced a fictional `data/lineage.py:_schema_fingerprint` (real name is `_fingerprint_dataframe`); inline comment in `_build_manifest` claimed dtype-aware hash but code hashes column names only | Deleted both dead helpers + unused `tempfile` import; rewrote trade-off #4 + cross-references to match reality; updated inline comment; updated dangling reference in `neural_model.py:468` | -28 +12 |
| 3.3.b | `MedianPruner` configured but inert (Optuna pruners require `trial.report` calls; project-wide grep found zero) — the trade-off #2 docstring claim was aspirational | Rewrote trade-off #2 in `tuning.py` to honestly document the inertness; added inline note at the `MedianPruner(...)` construction site; preserved the config as design seam for Sprint 4 activation | -7 +25 |
| 3.3.c | (none — cleanest module audited) | — | 0 |
| 3.3.d | mypy `no-any-return` in `_stratified_subsample` (not caught because `make typecheck` runs `mypy src`, not `mypy scripts/`); fixture scope mismatch (docstring promised "share across tests" but function-scoped → 6× pipeline runs); misleading "rename" comment at calibrator save site | Added `cast(pd.DataFrame, ...)` + import; module-scoped `isolated_settings` + `smoke_result` via `tmp_path_factory` + `MonkeyPatch.context()`; replaced misleading comment | -32 +52; **integration suite 70.26 s → 15.41 s (4.5× speedup)** |
| 3.4.a | Same `train_test_split` mypy pattern as 3.3.d (second occurrence) | Same `cast(pd.DataFrame, ...)` fix in `train_neural.py` | -1 +5 |
| 3.4.b | Same `train_test_split` mypy pattern as 3.3.d / 3.4.a (third occurrence) | Same `cast(pd.DataFrame, ...)` fix in `train_gnn.py` | -1 +5 |
| 3.4.c | (none — final report cross-checks clean against per-model reports) | — | 0 |

**Cumulative LOC delta on production code:** -72 LOC, +105 LOC,
net **+33 LOC** of corrections / clarifications. The audit
branch removes more dead/wrong than it adds.

**Per-prompt completion-report appends:** 12 prompts × ~150-700
LOC of audit notes per report, totalling ~3,500-4,500 LOC of
new audit documentation across the sprint reports. Every audit
section follows the same 9-section structure (files / loading /
business-logic / spec-vs-realised / coverage / lint+logging /
**design rationale (7 dimensions)** / gap-fixes / Sprint 4
follow-ons).

### Cross-cutting Sprint 6 follow-ons surfaced by the audit

These appeared multiple times across the per-prompt audits and
should be acted on as project-level infrastructure work:

1. **Extend `make typecheck` to cover `scripts/`.** Surfaced
   in 3.3.d, 3.4.a, and 3.4.b — three sibling scripts with the
   identical `train_test_split` mypy defect that the project's
   current `make typecheck` (which runs `mypy src` only) couldn't
   catch. A fourth occurrence is virtually guaranteed without
   this fix.
2. **Document the module-scoped expensive-fixture pattern as a
   project testing convention.** Both 3.3.d and 3.4.a's first
   revisions had function-scoped fixtures over expensive setup
   (15-70 s pipeline runs), causing 4-19× test suite slowdowns.
   Codifying the pattern in `docs/CONTRIBUTING.md` (or a new
   `docs/TESTING.md`) would prevent the third occurrence.
3. **MLflow filesystem-backend deprecation** (Feb 2026 warning)
   — surfaced in 3.3.b, 3.3.d, 3.4.a, 3.4.b. Sprint 6
   monitoring concern; migrate to SQLite or Postgres backend.
4. **Cross-platform `pyg-lib` wheel pinning.** The
   `[tool.uv.sources]` URL pin is Linux x86_64 + CPython 3.11
   specific. macOS / arm64 contributors would need a different
   wheel; parametric source pin for cross-platform CI is the
   fix.

### Verification gates after the entire audit sweep

`make test-fast` (final state on audit branch):

```
447 passed, 34 warnings in ~67 s
```

Same baseline as 3.4.b post-merge (no unit-test count change
across the audit; gap-fixes only touch script + docstring +
fixture-scope code). All 33 lineage tests pass. All 55
integration tests pass.

`make format && make lint && make typecheck` (final state):

```
$ uv run ruff format --check src tests scripts → 103 files already formatted
$ uv run ruff check src tests scripts → All checks passed!
$ uv run mypy src → Success: no issues found in 38 source files
```

Plus the three audit-driven mypy-fixes mean
`mypy scripts/train_lightgbm.py scripts/train_neural.py
scripts/train_gnn.py` is now also clean (was 1 error per script
pre-audit).

### Ready for `sprint-3-complete` tag

After the audit branch is committed, pushed, PR-ed, and squash-
merged to `main`, John can tag `sprint-3-complete` against the
merge commit. The tag pins:

- **Sprint 3 deliverables:** Tier-4 EWM + Tier-5 graph features
  + Models A/B/C + per-model training pipelines + comparison
  report.
- **Sprint 3 metrics:** Model A val AUC 0.8281 (calibrated log
  loss 0.109; p95 3.29 ms); Model B val AUC 0.8183
  (uncalibrated; p95 60 ms; trained on 50K subsample); Model C
  val AUC 0.7778 (uncalibrated; p95 0.07 ms; trained on 5K
  subsample).
- **Sprint 3 architecture:** Three diversity models with the
  decorrelation property visible in test/val gap behaviour —
  the precondition for Sprint 4's ensemble blend, which is the
  most important architectural payoff Sprint 3 delivers.

Audit edits will be consolidated into a single commit at the end
of the Sprint 3 audit-and-gap-fill sweep, per John's instruction.

**End of Sprint 3 audit sweep.**
