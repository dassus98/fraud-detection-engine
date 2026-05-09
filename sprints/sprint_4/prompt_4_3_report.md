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
