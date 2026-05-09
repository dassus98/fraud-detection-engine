# ADR 0003: Economic-cost threshold over F1 / AUC

- **Status:** Accepted
- **Date:** 2026-05-09
- **Sprint:** 4 (prompt 4.3)

## Context

The fraud-detection model produces calibrated probabilities (Model A, post-Sprint 3.3.c isotonic calibration; val log loss 0.291 → 0.109). Translating those probabilities into a binary block / allow decision requires a threshold τ. The conventional ML choices for picking τ are:

1. **Fixed τ = 0.5** — the field default; assumes a symmetric loss.
2. **F1-optimal τ** — picks the threshold maximising the harmonic mean of precision and recall.
3. **ROC-AUC** — measures ranking quality; threshold-free.
4. **PR-AUC** — measures precision-recall curve area; also threshold-free.

None of these select τ for the loss function the business actually pays. Per CLAUDE.md §8 a missed fraud (false negative) costs ~$450 and a blocked legit transaction (false positive) costs ~$35 — a 13× ratio. F1 weighs the two error classes equally; AUC and PR-AUC are decision-free; τ = 0.5 is arbitrary. All four are misaligned with the deployed system's real loss surface.

The loss surface that *does* matter is the expected USD cost across a transaction stream:

```
total_cost = FN × fraud_cost + FP × fp_cost + TP × tp_cost + TN × tn_cost
```

with the cost values pinned in `configs/economic_defaults.yaml` and live-read via `Settings`.

## Decision

Pick τ via `EconomicCostModel.optimize_threshold(y_true, y_scores)` (Sprint 4.1):

1. Sweep `np.linspace(0.01, 0.99, 99)` over candidate thresholds.
2. At each τ, threshold the calibrated probabilities to `y_pred`, compute the expected USD cost via `economic_cost(...)`.
3. Pick the τ that minimises `total_cost`. Tie-break favours the larger τ (block fewer transactions on equal cost — documented decision, not silent default).
4. Validate stability via `EconomicCostModel.sensitivity_analysis(...)` — a ±20 % symmetric grid across the three cost axes (CLAUDE.md §8: "decisions are stable under ±20 % variation"). Optimal τ should cluster within a small range.
5. Validate per-segment robustness via `StratifiedEvaluator.evaluate(...)` (Sprint 4.2) — surface any stratum where the global-cost-optimal τ underperforms (low-amount transactions, no-identity rows, mobile sessions, etc.).

| Aspect | Choice | Source |
|---|---|---|
| Loss surface | Expected USD cost | CLAUDE.md §8; values in `configs/economic_defaults.yaml` |
| Sweep grid | `linspace(0.01, 0.99, 99)` | `economic.py:_DEFAULT_*_THRESHOLD*` |
| Tie-break | Larger τ on equal cost | `economic.py:_sweep_thresholds` |
| Sensitivity | ±20 % symmetric per axis | `economic.py:_DEFAULT_SENSITIVITY_MULTIPLIERS` |
| Per-segment validation | `StratifiedEvaluator.evaluate(...)` | Sprint 4.2 |

### Why not F1

F1 is the harmonic mean of precision and recall — it weighs the two error classes equally. With a 13× FN/FP cost ratio, this systematically picks τ too high: the F1-optimal τ trades fraud catches for precision because a missed fraud and a blocked legit transaction count the same in the score. They do not count the same in dollars.

A toy example: at τ_F1, suppose 100 missed frauds and 100 blocked legits per day. F1 sees 200 errors. Cost: 100 × $450 + 100 × $35 = $48,500. At a lower τ_cost, suppose 20 missed frauds and 300 blocked legits per day. F1 sees 320 errors (worse). Cost: 20 × $450 + 300 × $35 = $19,500 (much better). F1 picks the wrong direction.

The right metric is not "balanced precision-recall"; it is "minimise dollars".

### Why not AUC

ROC-AUC is threshold-free. It measures the probability that a randomly-chosen positive scores higher than a randomly-chosen negative. That is a useful diagnostic of ranking quality (Sprint 3.3.d's training report uses it as the headline; Sprint 4.2's stratified heatmap uses it as a per-segment quality metric). It does not tell you where to operate.

PR-AUC has the same property: an area-under-curve summary, not a decision. We use both as ranking-quality diagnostics, never as threshold selectors.

### Why minimum-expected-cost τ

For calibrated probabilities and per-class costs, the expected-cost-minimising rule is the standard Bayes-decision argument: block iff the expected cost of allowing exceeds the expected cost of blocking. With `c_FN = fraud_cost`, `c_FP = fp_cost`, `c_TP = tp_cost`, `c_TN = 0`, the threshold is:

```
τ* = (c_FP - c_TN) / (c_FP - c_TN + c_FN - c_TP)
   = fp_cost / (fp_cost + fraud_cost - tp_cost)
```

With the project defaults that gives τ* ≈ 35 / (35 + 450 − 5) ≈ 0.073 in the asymptotic limit. The empirical optimum from `optimize_threshold` lands higher than this analytical limit because the swept grid operates on finite-sample empirical FN/FP rates rather than the theoretical decision boundary — see Sprint 4.1's "surprising findings" #1 for the full discussion of why high `fraud_cost` ratios don't push τ all the way to the rail in finite samples.

The Bayes-decision result requires *calibrated* probabilities. Sprint 3.3.c's isotonic calibration is what makes the rule applicable; without it, the cost surface is read off mis-scaled probabilities and the chosen τ is wrong even with correct cost values.

## Consequences

**Positive:**

- Decisions are dollar-aligned. The unit a fraud team's manager actually cares about is reported USD savings; this decision rule optimises that unit directly.
- Sensitivity-tested per CLAUDE.md §8 (stability under ±20 % cost variation), so cost-input uncertainty does not silently shift the operating point. The `sensitivity_analysis` grid is the audit trail.
- Per-segment validation via Sprint 4.2's `StratifiedEvaluator` surfaces strata where the global-cost-optimal τ underperforms. The heatmap visualises segment-level skew that would otherwise hide behind the aggregate metric.
- Captures decision in code (`EconomicCostModel`), in config (`economic_defaults.yaml`), and in writing (this ADR). Three layers of audit trail; reviewers can trace a production threshold back to the dollar values that produced it.

**Negative:**

- Cost values are uncertain. `fraud_cost_usd = 450` is an industry-median estimate, not a per-deployment measurement. The sensitivity grid bounds the impact (decisions stable under ±20 % variation), but a deployer with materially different cost structure (e.g. a luxury-goods merchant with much higher per-fraud loss; a high-CLV bank with much higher fp_cost) MUST override the defaults via `.env` and re-run `sensitivity_analysis`.
- Cost-based τ depends on calibration quality. Mis-calibrated probabilities produce a wrong cost surface, and the threshold derived from it is misaligned. Sprint 3.3.c's isotonic calibration is the load-bearing dependency; if calibration regresses, τ regresses with it.
- Adds the `optimize_threshold` step to the training pipeline (~2 s on val) and the `sensitivity_analysis` grid (~1 s on val for the default 125-cell grid). Trivial cost.
- The ADR's mathematical argument (Bayes-decision, theoretical τ ≈ 0.073) is internally consistent but not directly enforceable; the empirical sweep is what the production threshold is read from. Reviewers expecting τ at the analytical rail will be surprised; Sprint 4.1's surprising-findings discussion is the explainer.

**Revisit triggers:**

- **Cost ratio shifts beyond ±20 %.** Examples: regulatory penalty changes (OSFI / FCA / OCC tightening operational-risk capital expectations), CLV revaluation (fintech repricing customer LTV after a market event), scheme fee restructuring (Visa or Mastercard adjusting interchange / chargeback fees). Re-run `sensitivity_analysis`; if the ±20 % grid no longer brackets the new cost, re-derive τ.
- **New product launch with materially different fraud profile.** Example: card-present launch (fraud_cost shape changes — chargeback rules differ). Treat as a new deployment with overridden costs.
- **Calibration drift detected** by Sprint 6's monitoring. If isotonic calibration's ECE on production scores rises beyond the ECE budget, the cost surface degrades; re-fit calibration before re-deriving τ.
- **Test/val temporal gap widens.** If the IEEE-CIS-style temporal split assumption breaks (e.g. seasonal fraud pattern), per-segment τ via `StratifiedEvaluator` is the natural extension. Out of scope for this ADR.

## Alternatives considered

- **Fixed τ = 0.5.** The field default; ignores the cost ratio entirely. Rejected on first principles: the cost ratio is the entire reason the threshold matters.
- **F1-optimal τ.** Symmetric loss; rejected per "Why not F1" above. Cited in fraud-detection tutorials as a reasonable default; it is not for production fraud systems.
- **PR-AUC-optimal operating point.** PR-AUC is threshold-free; "operating point" requires a separate decision rule. Rejected per "Why not AUC" above.
- **Recall@FPR < X %.** An operational target (e.g. "catch 90 % of fraud while keeping false positives under 2 %") rather than an economic optimum. Useful as a *constraint* (Sprint 5 might want it as a bound on the cost-optimal τ), not as the primary objective. Tracked separately in `utils/metrics.py:recall_at_fpr`.
- **Precision@K.** Analyst-capacity constrained — picks τ such that the top K-fraction of scored transactions is flagged. Useful when the analyst-review queue has a fixed daily capacity, not for an open-ended block / allow decision. Tracked separately in `utils/metrics.py:precision_recall_at_k`.
- **Reinforcement-learning-style policy with reward = cost.** Considered for completeness; rejected as overengineered. The Bayes-decision rule on calibrated probabilities is the closed-form answer; RL would need a much richer environment (multi-step user flows, churn dynamics) to add value.

## References

- `CLAUDE.md` §8 — Business-Logic Constants You Must Know (cost defaults).
- `configs/economic_defaults.yaml` — cost values + comment-sourced breakdowns (Sprint 4.3).
- `src/fraud_engine/evaluation/economic.py` — `EconomicCostModel`, `optimize_threshold`, `sensitivity_analysis` (Sprint 4.1).
- `src/fraud_engine/evaluation/stratified.py` — `StratifiedEvaluator` (Sprint 4.2).
- `src/fraud_engine/evaluation/calibration.py` — isotonic calibration; the load-bearing dependency for cost-surface fidelity (Sprint 3.3.c).
- `src/fraud_engine/utils/metrics.py:68-161` — `economic_cost` per-call primitive.
- `sprints/sprint_4/prompt_4_1_report.md` — `EconomicCostModel` design rationale + asymptotic-gate findings ("surprising findings" #1 explains the empirical-vs-theoretical τ gap).
- `sprints/sprint_4/prompt_4_2_report.md` — `StratifiedEvaluator` per-segment validation; the heatmap is the audit artefact.
- Elkan, C. (2001). "The foundations of cost-sensitive learning." IJCAI 2001 — the classical reference for cost-sensitive classification thresholds via Bayes-decision theory. Cited as the academic anchor; not quoted.
