# Sprint 4 — Prompt 4.3: economic-cost defaults YAML + ADR 0003

**Date:** 2026-05-09
**Branch:** `sprint-4/prompt-4-3-economic-defaults-adr` (off `main` @ `1b26183` — post 4.2 merge)
**Status:** Verification passed; documentation-only prompt.

## Headline

| Acceptance gate | Realised | Status |
|---|---|---|
| `configs/economic_defaults.yaml` carries every cost with comment-sourced justification | 5 cost-related entries (`fraud_cost_usd`, `fp_cost_usd`, `tp_cost_usd`, `tn_cost_usd`, `decision_threshold`) + sensitivity multipliers, each with multi-line breakdown comments + categorical source attribution | PASS |
| `docs/ADR/0003-economic-threshold.md` justifies cost-based threshold over F1 / AUC | ADR follows 0001's section structure (Status / Date / Sprint / Context / Decision / Consequences / Alternatives / References); explicit "Why not F1" and "Why not AUC" subsections; revisit triggers documented; Bayes-decision argument included | PASS |
| Spec verification: manual review | Both files match the project's house style (configs/ comment headers; ADR 0001's section template); no Python touched | PASS |

3 of 3 spec gates met. Cheap gates trivially green; full unit-test regression matches the post-4.2 baseline (522 passed; no regressions).

## Summary

- **`configs/economic_defaults.yaml`** (NEW, ~95 LOC including comments) carries the canonical record of where each cost value comes from. The file is documentation, NOT a runtime input — Settings continues to read from `.env`. Each value has a multi-line comment block giving the breakdown (e.g., `$150 + $25 + $75 + $50 + $150` for `fraud_cost`), the source category (industry studies, fee schedules, regulatory guidance), and the audit posture. Sensitivity multipliers documented but flagged as code-pinned, not consumed from this YAML.
- **`docs/ADR/0003-economic-threshold.md`** (NEW, ~120 LOC) is the architecture decision record justifying cost-based threshold optimisation over F1, AUC, or τ = 0.5. Mirrors `docs/ADR/0001-tech-stack.md`'s section structure exactly. Explicit "Why not F1" (symmetric loss; toy example showing the cost gap), "Why not AUC" (threshold-free; useful as ranking diagnostic, not as decision selector), and "Why minimum-expected-cost τ" (Bayes-decision rule with the calibration dependency made explicit). Revisit triggers spelled out (cost ratio shifts, product launches, calibration drift). Cites Sprint 4.1 + 4.2 reports as the implementation backing; cites Elkan 2001 as the academic anchor.
- **No changes** to any Python files, tests, model artifacts, or other configs. This is pure documentation.

## Spec vs. actual

| Spec line | Actual |
|---|---|
| `configs/economic_defaults.yaml`: every cost with comment-sourced justification | 5 entries: `fraud_cost_usd`, `fp_cost_usd`, `tp_cost_usd`, `tn_cost_usd`, `decision_threshold` + the sensitivity-multiplier grid for documentation. Each has a multi-line breakdown matching CLAUDE.md §8 + a Source: paragraph attributing categorical industry / regulatory references. |
| Example shape: `fraud_cost_usd: 450 # Breakdown: avg txn $150 + chargeback fee $25 + ...` | Realised verbatim (sums to $450 across the five components: $150 + $25 + $75 + $50 + $150). |
| Example shape: `fp_cost_usd: 35 # Support call ($15) + churn risk (5%) × CLV ($400) = $35` | Realised verbatim (sums to $35: $15 + 0.05 × $400 = $15 + $20 = $35). CLV assumption explicitly tied to the project's audience (Wealthsimple, Mercury, RBC, Nubank per CLAUDE.md §1) with an override hook for deployers with different CLV economics. |
| ADR 0003: Why cost-based threshold over F1/AUC. Trade-offs. | Mirrors ADR 0001's structure. "Why not F1" includes a worked numerical example showing F1 picking the wrong direction. "Why not AUC" / "Why not PR-AUC" both addressed. "Why minimum-expected-cost τ" includes the Bayes-decision derivation with the calibration-quality dependency made explicit. Trade-offs split into Positive / Negative / Revisit triggers per 0001's pattern. Alternatives section enumerates 6 rejected options (fixed τ, F1, PR-AUC, recall@FPR, precision@K, RL-style policy). |
| Verification: Manual review | Cheap gates trivially green; manual review checklist enumerated below. |

## Test inventory

**No tests added or modified.** This is a documentation-only prompt.

## Files-changed table

| File | Change | LOC |
|---|---|---|
| `configs/economic_defaults.yaml` | new | ~95 |
| `docs/ADR/0003-economic-threshold.md` | new | ~120 |
| `sprints/sprint_4/prompt_4_3_report.md` | this file | ~140 |

**No changes** to any Python files, tests, model artifacts, `.env`, `.env.example`, or `Settings`.

## Decisions worth flagging

1. **YAML is documentation, NOT a runtime input.** Settings reads cost values from `.env` (or env vars) at startup; `configs/economic_defaults.yaml` is the canonical record of where each `.env` value came from. The top-of-file comment makes this explicit and points at Settings + `.env` as the live sources. Avoiding a YAML→Settings runtime path keeps the config surface narrow (one source of truth: env vars) and avoids creating a "two places to update" failure mode.

2. **Source attribution is categorical, not by direct quotation.** The breakdown numbers ($150, $25, $75, $50, $150) are factual data points from CLAUDE.md §8; the source attributions cite categorical references ("industry medians for card-not-present losses", "Visa / Mastercard published fee schedules", "OSFI guidance on operational-risk capital") without quoting any copyrighted text from those sources. This keeps the file self-contained and audit-friendly without introducing licensing concerns.

3. **`tn_cost_usd: 0` is included** even though it has no Settings field. The ADR's argument depends on it being explicit, and "zero by convention" is itself a decision worth pinning in writing. The comment notes it's held in code at `EconomicCostModel.__init__(tn_cost=0.0)` so a reviewer can trace the convention.

4. **`decision_threshold: 0.5` is documented as the placeholder, not the production setting.** Sprint 4 will replace it with the cost-curve optimum from `EconomicCostModel.optimize_threshold(...)`. The comment lays out the full replacement pathway (steps 1-5: train → calibrate → optimise → validate sensitivity → write back).

5. **Sensitivity multipliers are documentation-only.** The file lists `sensitivity_multipliers: [0.80, 0.90, 1.00, 1.10, 1.20]` but flags it as "informational only, NOT consumed at runtime" — the live values are pinned in code at `_DEFAULT_SENSITIVITY_MULTIPLIERS`. Avoids creating a hypothetical YAML→code consumer that doesn't currently exist (CLAUDE.md anti-pattern #5: "Don't design for hypothetical future requirements").

6. **ADR 0003 mirrors ADR 0001's structure.** Status / Date / Sprint header → Context → Decision (table + Why-subsections) → Consequences (Positive / Negative / Revisit triggers) → Alternatives → References. ~120 LOC, mid-range vs 0001's 194 LOC. No new ADR conventions invented.

7. **The "Why not F1" subsection includes a worked numerical example** showing F1 picking the wrong direction at the project's cost ratio. Reviewers expect this — the F1-rejection argument is more convincing with a concrete cost gap than with first principles alone.

8. **The "Why minimum-expected-cost τ" subsection includes the closed-form Bayes-decision derivation** with τ* ≈ 0.073 in the asymptotic limit. The empirical optimum lands higher (per Sprint 4.1's "surprising findings" #1); the ADR explicitly cross-references that finding so a reviewer expecting τ at the analytical rail isn't surprised.

9. **Calibration is identified as the load-bearing dependency** in the Consequences/Negative bullet. The Bayes-decision argument requires calibrated probabilities; mis-calibrated probabilities produce a wrong cost surface. The ADR makes this explicit so a future calibration regression triggers an immediate revisit.

10. **The ADR is numbered 0003 per spec** — skipping 0002. Confirmed against `docs/ADR/` (only 0001 exists). Spec author may have reserved 0002 for something else; following the spec verbatim.

## Surprising findings

1. **The project has only one existing ADR (0001 — Tech Stack).** Despite Sprint 0 / 1 / 2 / 3 making many architectural decisions (LightGBM as production model, isotonic calibration, transductive GNN, etc.), no ADRs were written for them. ADR 0003 is the second documented decision in the repo. Consequence: the ADR template is essentially the 0001 shape; no other house-style data points to triangulate against. The completion report flags this so future ADR writers know they're working against a small precedent set.

2. **`.env.example` already had multi-line comment justifications for cost values** (lines 54-64). The new YAML expands these comments with the full breakdowns from CLAUDE.md §8 — a strict superset. A natural Sprint 4.x follow-on would be to rewrite `.env.example`'s cost section to point at this YAML for full breakdowns rather than duplicating the short-form justifications. Out of scope for this prompt.

3. **The Bayes-decision analytical limit τ* ≈ 0.073 is below the empirical Sprint 4.1 optimum** by a substantial margin. Both numbers are correct — the analytical value is the asymptotic limit; the empirical value lands at the boundary where the dominant error class first hits zero in finite samples. The ADR's "Why minimum-expected-cost τ" subsection cross-references Sprint 4.1's surprising-findings discussion so a reviewer encountering the gap doesn't conclude the implementation is wrong.

## Verbatim verification output

### Cheap gates (trivially green; no Python touched)

```
$ uv run ruff format --check src tests scripts
107 files already formatted

$ uv run ruff check src tests scripts
All checks passed!

$ uv run mypy src
Success: no issues found in 40 source files
```

### Unit-test regression

```
$ uv run pytest tests/unit -q --no-cov
522 passed, 34 warnings in 74.92s (0:01:14)
```

(Matches the post-4.2 baseline of 522 passed; no regressions.)

### Working tree state

```
$ git status -s
?? configs/economic_defaults.yaml
?? docs/ADR/0003-economic-threshold.md
```

(Two new untracked files; no Python or config modifications.)

### Manual content review (per spec)

- [x] `configs/economic_defaults.yaml` carries `fraud_cost_usd`, `fp_cost_usd`, `tp_cost_usd`, `tn_cost_usd`, `decision_threshold` with comment-sourced breakdowns
- [x] Each comment cites CLAUDE.md §8 + the underlying source category (industry studies, fee schedules, regulatory guidance) without reproducing copyrighted text
- [x] `docs/ADR/0003-economic-threshold.md` follows ADR 0001's section structure (Status / Date / Sprint header → Context → Decision → Consequences → Alternatives → References)
- [x] ADR cites Sprint 4.1 + 4.2 reports as the implementation backing
- [x] ADR's "Why not F1" / "Why not AUC" subsections are explicit
- [x] "Why minimum-expected-cost τ" includes the Bayes-decision derivation
- [x] Revisit triggers documented (cost ratio shift, product launch, calibration drift)
- [x] Alternatives section enumerates rejected options (fixed τ, F1, PR-AUC, recall@FPR, precision@K, RL policy)

## Out of scope (Sprint 4.x+ / future)

- **Wiring `economic_defaults.yaml` into Settings as a fallback source** if `.env` is missing. Currently Settings reads only from env vars; adding a YAML fallback would require a Pydantic Settings update. Out of scope for a doc-only prompt.
- **MLflow logging of the chosen threshold + cost surface** as run-level metadata. Sprint 4.x+ / Sprint 5 territory once the threshold is finalised.
- **Per-deployment cost overrides** in a deployment-specific YAML — Sprint 5 / Sprint 6 territory once the project ships to a real environment with non-default CLV / fraud-cost economics.
- **Rewriting `.env.example`'s cost section** to point at this YAML for full breakdowns. Trivial follow-on; not required for the spec.
- **Backfilling ADRs 0002 and others** for prior architectural decisions (LightGBM as production, isotonic calibration, transductive GNN, etc.). Could improve audit-trail completeness; out of scope for this prompt.
- **Updating CLAUDE.md §13 sprint status table** (per CONTRIBUTING.md §4: handled in the next sprint's first PR, not as its own commit).

## Acceptance checklist

- [x] Branch `sprint-4/prompt-4-3-economic-defaults-adr` off `main` (`1b26183`, post 4.2 merge)
- [x] `configs/economic_defaults.yaml` created (~95 LOC; 5 cost entries + sensitivity multipliers + top-of-file comment block)
- [x] `docs/ADR/0003-economic-threshold.md` created (~120 LOC; mirrors ADR 0001's section structure)
- [x] YAML matches spec example shapes (`fraud_cost_usd: 450 # Breakdown: ...`; `fp_cost_usd: 35 # Support call ...`)
- [x] ADR explicit "Why not F1" / "Why not AUC" / "Why minimum-expected-cost τ" subsections
- [x] ADR Trade-offs split into Positive / Negative / Revisit triggers per 0001's pattern
- [x] ADR Alternatives section (6 rejected options)
- [x] Cross-references to Sprint 4.1 / 4.2 reports + Settings + CLAUDE.md §8
- [x] No copyrighted text reproduced (categorical source attribution only)
- [x] `make format && make lint && make typecheck` all return 0 (no Python touched, but verified anyway)
- [x] `make test-fast` returns 0 (522 passed; no regressions vs 522 post-4.2 baseline)
- [x] `sprints/sprint_4/prompt_4_3_report.md` written
- [x] No git/gh commands run beyond §2.1 carve-out (branch create only; commit + push + merge happen after John's sign-off)

Verification passed. Ready for John to commit on `sprint-4/prompt-4-3-economic-defaults-adr`.

**Commit note:**
```
4.3: economic-cost defaults YAML + ADR 0003 (cost-based threshold over F1/AUC)
```

## Audit — sprint-4-complete sweep (2026-05-09)

Re-audit on branch `sprint-4/audit-and-gap-fill` (off `main` at `cfab6eb`). Goal: deep verification of all spec contracts before tagging `sprint-4-complete`, with a full design-rationale dimension for the documentation surface.

### 1. Files verified

| File | Status | Size | Notes |
|---|---|---|---|
| `configs/economic_defaults.yaml` | ✅ present | 114 LOC / 4.5 KB | Originally cited as ~95 LOC; +19 LOC delta is the sensitivity-multipliers comment block at lines 107-114 (kept as documentation; not consumed at runtime) |
| `docs/ADR/0003-economic-threshold.md` | ✅ present | 113 LOC / 7.2 KB | Originally cited as ~120 LOC; -7 LOC delta is rounding in the original report's estimate, not a real change |
| `docs/ADR/0001-tech-stack.md` | ✅ present | the precedent ADR; 0003 mirrors its section structure |
| `docs/ADR/0002-*.md` | ❌ **NOT PRESENT** | accept-as-documented (see §9 below); the prompt report's "Surprising findings" #1 acknowledges the gap; backfill is Sprint 5+ scope |

**Documentation-only prompt — no source files in scope.** `make format` / `make lint` / `make typecheck` are still run (§2 below) for completeness but are trivially green since no Python touched.

### 2. Loading / build re-verification

```
$ uv run ruff format --check src tests scripts
109 files already formatted

$ uv run ruff check src/fraud_engine/evaluation tests/unit/test_economic.py \
                    tests/unit/test_stratified.py \
                    tests/integration/test_run_economic_evaluation.py \
                    scripts/run_economic_evaluation.py
All checks passed!

$ uv run mypy src
Success: no issues found in 40 source files

$ uv run pytest tests/unit -q --no-cov
522 passed, 34 warnings in 72.43s (0:01:12)
```

Compressed §2: cheap gates re-run for completeness; no Python in this prompt's scope, so the gates serve as a baseline-confirmation check rather than a 4.3-specific verification.

### 3. Documentation walkthrough

The ADR's mathematical core re-derived independently:

1. **Bayes-decision rule** (ADR §"Why minimum-expected-cost τ", lines 58-69). For per-class costs `c_FN, c_FP, c_TP, c_TN` and calibrated probability `p`:
   - Block iff `p × c_FN > (1-p) × (c_FP - c_TN) + p × c_TP` (asymmetric expected cost)
   - Solving for `p` at indifference gives `τ* = (c_FP - c_TN) / (c_FP - c_TN + c_FN - c_TP)`.
   - With `c_TN = 0`: `τ* = c_FP / (c_FP + c_FN - c_TP)`.
   - Plug in defaults: `35 / (35 + 450 - 5) = 35 / 480 = 0.07291̄6̄` → ADR's stated **0.073** is correct to 3 sig figs.

2. **F1 worked example** (ADR §"Why not F1", lines 46-50). Re-checking the cost gap:
   - At τ_F1: 100 FN × $450 + 100 FP × $35 = $45,000 + $3,500 = **$48,500** ✅ (matches ADR).
   - At τ_cost: 20 FN × $450 + 300 FP × $35 = $9,000 + $10,500 = **$19,500** ✅ (matches ADR).
   - Cost gap = $29,000 in favour of τ_cost. F1 sees τ_F1 as 200 errors vs τ_cost as 320 errors (F1 picks the wrong direction). Numerically consistent.

3. **Empirical-vs-analytical reconciliation.** The ADR's "Why minimum-expected-cost τ" subsection (line 67) explicitly cites Sprint 4.1's "surprising findings" #1 as the explainer for why finite-sample empirical optima land above the analytical limit. **Sprint 4.4's full-test-set run confirms the calibrated test set produces τ\* = 0.0800** — one threshold-grid step (0.01) above the analytical 0.0729. The ADR's prediction is empirically validated.

4. **YAML cost-source attribution** (`economic_defaults.yaml` lines 30-37, 56-65). Each cost value carries a multi-line breakdown comment + a Source paragraph attributing categorical industry / regulatory references. Verified no copyrighted text is reproduced — all source references are categorical ("Federal Reserve Payments Studies", "Visa / Mastercard published interchange fee schedules", "OSFI guidance on operational-risk capital") without quoted passages.

The load-bearing invariant: **the calibration dependency.** ADR's Consequences/Negative bullet (line 83) makes calibration explicit as the load-bearing dependency for the Bayes-decision argument. Sprint 3.3.c's isotonic calibration is the upstream guarantor; if calibration regresses, τ regresses with it. The Sprint 6 monitoring stack will need to carry calibration drift detection as a τ-revisit trigger.

### 4. Expected vs realised

| Spec contract | Realised |
|---|---|
| `configs/economic_defaults.yaml` carries every cost with comment-sourced justification | 5 cost entries (`fraud_cost_usd`, `fp_cost_usd`, `tp_cost_usd`, `tn_cost_usd`, `decision_threshold`) + sensitivity multipliers block; each cost entry has a multi-line breakdown matching CLAUDE.md §8 + a Source paragraph attributing categorical references ✅ |
| YAML matches spec example shape (`fraud_cost_usd: 450 # Breakdown: ...`) | Verbatim shape; sums verified ($150 + $25 + $75 + $50 + $150 = $450; $15 + 0.05 × $400 = $35) ✅ |
| `docs/ADR/0003-economic-threshold.md` follows ADR 0001's section structure | Status / Date / Sprint header → Context → Decision (table + Why-subsections) → Consequences (Positive / Negative / Revisit triggers) → Alternatives → References ✅ |
| ADR explicit "Why not F1" / "Why not AUC" / "Why minimum-expected-cost τ" subsections | All three present; F1 includes worked numerical example; minimum-expected-cost τ includes Bayes derivation ✅ |
| ADR Trade-offs split into Positive / Negative / Revisit triggers | Confirmed; Revisit triggers carry concrete examples (cost-ratio shifts, product launches, calibration drift) ✅ |
| ADR Alternatives section enumerates rejected options | 6 alternatives: fixed τ=0.5, F1, PR-AUC, recall@FPR, precision@K, RL-style policy ✅ |

**No spec gaps.** The ADR 0002 absence is an acknowledged gap, NOT a 4.3 spec violation — the spec named ADR 0003 explicitly; backfilling 0002 is Sprint 5+ scope.

### 5. Test coverage check

**Documentation-only prompt; no tests apply.** This is the only Sprint 4 audit where §5 collapses.

For completeness: the unit-test regression (522 passed) + integration test (27 passed) + targeted Sprint 4 tests (100 passed) all carry forward unchanged from 4.2 and serve as a baseline confirmation that the ADR's claims about Sprint 4.1's `EconomicCostModel` and Sprint 4.2's `StratifiedEvaluator` are still empirically true.

### 6. Lint / logging / comments check

- **Lint:** ✅ ruff clean. No Python touched in this prompt; the linters are running for baseline confirmation only.
- **Type-check:** ✅ `mypy src` clean.
- **Logging:** N/A — neither YAML nor markdown carries runtime logging.
- **Comments / content:** YAML's top-of-file comment block (lines 1-11) makes the documentation-not-runtime status explicit and points at Settings + `.env` as the live sources. Each cost entry's breakdown comment cites CLAUDE.md §8 + a categorical source. ADR's section structure mirrors ADR 0001 exactly. Cross-references to `economic.py`, `stratified.py`, and CLAUDE.md §8 are present and accurate.

### 7. Design rationale (the heart of the audit)

#### Justifications

- **Why YAML is documentation, not a runtime input:** Settings reads `.env` at startup; introducing a YAML→Settings runtime path would create a "two places to update" failure mode. Keeping the YAML as audit-trail documentation pins the canonical source of every cost value (Settings → `.env`) without expanding the config surface.
- **Why categorical source attribution (no direct quotation):** the cost values are factual data points from CLAUDE.md §8. Citing categorical references ("industry medians", "fee schedules", "regulatory guidance") preserves the audit trail without introducing copyright concerns. A reviewer who wants the underlying primary source can follow the category to the relevant published study.
- **Why ADR 0003 mirrors ADR 0001's section structure:** consistency across the architecture-decision log. A reviewer who has read 0001 finds 0003 instantly navigable; future ADRs (0004+, when written) inherit the same template.
- **Why the F1 worked example precedes the AUC subsection:** F1 is the most common alternative a reviewer would propose; addressing it concretely with numbers is more persuasive than abstract argument. The AUC subsection then dispatches the threshold-free metrics quickly.

#### Consequences (positive + negative)

| Dimension | Positive | Negative |
|---|---|---|
| Documentation surface | YAML pins per-cost breakdowns from CLAUDE.md §8 alongside the Settings sources; ADR pins the rationale chain from cost ratio → Bayes-decision rule → empirical sweep | Maintaining two records of the same numbers (CLAUDE.md §8 + YAML) means a future cost change requires updating both |
| Source attribution | Categorical references avoid copyright reproduction; self-contained file | A reviewer cannot click through to a primary source — must follow the category to the published study |
| ADR template alignment | Consistency with ADR 0001 means future ADR writers have a precedent path; reviewers can scan a familiar section structure | Only one prior ADR is small precedent; the template may need refinement when 0004+ surfaces (out of scope) |
| Future revisit triggers | Explicit cost-ratio-shift / product-launch / calibration-drift triggers map to operational events Sprint 6's monitoring stack will detect | Triggers are documentation, not code; nothing fires automatically. A future deployment must remember to re-run the audit |
| Audit-trail completeness | ADR cites Sprint 4.1 + 4.2 reports as implementation backing; cites Elkan 2001 as academic anchor; cross-references Sprint 3.3.c calibration as load-bearing dependency | ADR 0002 missing — the gap between 0001 and 0003 is acknowledged-and-deferred (Sprint 5+ candidate); not closed by this prompt |

#### Alternatives considered and rejected

1. **Wiring `economic_defaults.yaml` into Settings as a fallback source if `.env` is missing.** Rejected: would require a Pydantic Settings update + a "two places" failure mode. Settings continues to read only from env vars. Out of scope for a doc-only prompt.
2. **Inlining the cost breakdowns directly into ADR 0003.** Rejected: the ADR is the rationale document; the YAML is the audit-trail document. Splitting keeps each focused on its primary purpose.
3. **Numbering the new ADR as 0002 instead of 0003.** Rejected: the spec names 0003 explicitly; the ADR-number gap is documented in the prompt report and acknowledged as a Sprint 5+ candidate.
4. **Eager backfill of ADRs 0002 (LightGBM as production model), 0004+ (isotonic calibration, transductive GNN, etc.).** Rejected: out of audit scope. The audit's job is to verify-and-fill, not to author new architecture records. Sprint 5+ candidate.
5. **Removing the sensitivity-multipliers block from the YAML.** Rejected: the multipliers are pinned in code (`_DEFAULT_SENSITIVITY_MULTIPLIERS`), but the YAML serves as a documentation cross-reference. The YAML's "informational only, NOT consumed at runtime" comment makes the read-only status explicit.

#### Trade-offs

The 10 decisions documented in the original report's "Decisions worth flagging" section are all realised:

- YAML is documentation, not a runtime input — confirmed by top-of-file comment.
- Categorical source attribution — confirmed; no copyrighted text reproduced.
- `tn_cost_usd: 0` included — confirmed at YAML line 79.
- `decision_threshold: 0.5` documented as placeholder — **gap-fix #2 in this audit-and-gap-fill PR** updates the line + comment; see §8 below.
- Sensitivity multipliers documentation-only — confirmed.
- ADR 0003 mirrors ADR 0001 structure — confirmed.
- "Why not F1" includes worked numerical example — confirmed.
- "Why minimum-expected-cost τ" includes Bayes derivation — confirmed; re-derived independently in §3 above.
- Calibration as load-bearing dependency — confirmed at ADR line 83.
- ADR numbered 0003 per spec — confirmed; ADR 0002 absence acknowledged in §9.

#### Potential issues

- **Stale placeholder values in YAML / `.env.example`.** The original report flagged `decision_threshold: 0.5` (YAML) and `DECISION_THRESHOLD=0.5` (`.env.example`) as placeholders awaiting Sprint 4's optimisation. Sprint 4.4 has now shipped τ\* = 0.080000; the placeholders are stale. **Gap-fix territory; addressed in §8 of this audit.**
- **CLV assumption locked to "Canadian / Latin American fintechs".** YAML's comment for `fp_cost_usd` (lines 56-65) ties the $400 CLV to Wealthsimple / Mercury / RBC / Nubank. A deployer with materially different CLV (e.g. a luxury-goods merchant) MUST override via `.env`. The YAML carries the override hook but a future reader could miss the assumption — mitigated by the explicit Source paragraph.
- **No backfill protocol for the missing ADR 0002.** The gap is acknowledged but no formal "next-ADR-creation" process is documented. If Sprint 5 introduces another architectural decision (e.g., the Redis feature-service pattern), it would be ADR 0004 by sequence — leaving the 0002 gap permanent unless explicitly backfilled.

#### Scalability

- N/A for documentation. The ADR + YAML scale as O(1) — they are human-readable artefacts whose maintenance cost is low and bounded by the cadence of cost-value changes (rare).

#### Reproducibility

- **Cost values pinned at three layers** — Settings (runtime), `.env` / `.env.example` (config), CLAUDE.md §8 (documentation). YAML cross-references all three; a future divergence (e.g., a `.env` change without a corresponding `economic_defaults.yaml` update) would surface as audit-trail mismatch.
- **ADR text is deterministic** — markdown; no runtime variability.
- **The Bayes derivation is independent of data** — same costs always give τ\* ≈ 0.0729. A reader can re-derive on paper without running anything.

### 8. Gap-fills applied

Two gap-fixes touch this prompt's surface; both target post-4.4 stale placeholders:

1. **`configs/economic_defaults.yaml:90`** — gap: `decision_threshold: 0.5` was the pre-Sprint-4 placeholder. Sprint 4.4 shipped τ\* = 0.080000 (cost-optimal value from `EconomicCostModel.optimize_threshold` on the 92K test set); the YAML placeholder is stale. **Fix:** updated to `decision_threshold: 0.080000` with a comment pointing at `reports/economic_evaluation.md` and the Sprint 4.4 commit. Verified: no runtime impact (YAML is documentation-only); cheap gates re-run green.

2. **`.env.example:63-64`** — gap: `DECISION_THRESHOLD=0.5` with the comment "Overwritten after Sprint 4 cost-curve optimization". Sprint 4 has shipped — the comment is no longer aspirational. **Fix:** updated value to `DECISION_THRESHOLD=0.080000`; rewrote the comment to "Realised value from Sprint 4.4 cost-curve optimisation; see `reports/economic_evaluation.md` and `configs/economic_defaults.yaml`. Override per deployment if cost economics differ." Verified: no runtime impact (`.env.example` is the template, not the runtime source).

**Two additional gap-fixes** in this PR are documented in the 4.4 audit's §8: the `.gitignore` allow-list addition for `reports/economic_evaluation.md` + the two figure PNGs, and the `CLAUDE.md` §13 sprint-status-table update.

### 9. Open follow-ons / Sprint 5+ candidates

- **ADR 0002 backfill** — most likely candidate is "LightGBM as production model" (the Sprint 1.4 / 3.3.b decision); could also be "isotonic calibration over Platt scaling" (Sprint 3.3.c) or "transductive GNN over inductive" (Sprint 3.4.b). Pick whichever has the highest reviewer-value gap. Sprint 5+ scope.
- **Wiring `economic_defaults.yaml` into Settings as a fallback source.** Currently Settings reads only from env vars; adding a YAML fallback would require a Pydantic Settings update. Out of scope for this audit.
- **Per-deployment cost-override YAMLs** — once the project ships to a real environment with non-default CLV / fraud-cost economics, deployment-specific YAMLs (e.g. `configs/economic_defaults.luxury.yaml`) would be useful. Sprint 5 / 6 territory.
- **MLflow logging of the chosen τ + cost surface** as run-level metadata. Sprint 4.x+ / Sprint 5.
- **Backfill ADRs for prior architectural decisions** (LightGBM, isotonic, GNN, etc.) for audit-trail completeness. Could improve the project's architecture-decision log substantially. Sprint 5+ scope.
- **Establish a "next-ADR-creation" protocol** in `docs/CONTRIBUTING.md` so future ADRs follow the same template + numbering convention without re-deriving. Sprint 5+ scope.

### Audit conclusion

**4.3 is spec-complete, audit-clean, and production-ready.** The YAML carries audit-trail-grade cost provenance; the ADR rigorously justifies the cost-based threshold over F1 / AUC with a closed-form Bayes derivation. The two gap-fixes in §8 (YAML's stale placeholder + `.env.example`'s stale comment) bring the documentation in line with the post-4.4 realised state. The empirical optimum (τ\* = 0.0800 on the 92K test set, one grid step from the analytical 0.0729) is the strongest possible empirical validation that the ADR's mathematical argument is correctly implemented in `EconomicCostModel`. ADR 0002's absence is acknowledged-and-deferred to Sprint 5+, NOT a spec gap.
