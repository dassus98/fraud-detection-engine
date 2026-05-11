# ADR 0005: LightGBM as the production champion (LightGBM A, FraudNet B shadow, FraudGNN C batch-only)

- **Status:** Accepted
- **Date:** 2026-05-02
- **Sprint:** 0 (initial pick); 3 (validated via Sprint 3.4 cross-model comparison)

## Context

The fraud-detection-engine project trains three distinct models, each with different strengths:

- **Model A — LightGBM (GBDT)** — tabular gradient-boosted trees + isotonic calibration.
- **Model B — FraudNet (NN)** — PyTorch entity-embedding neural net with focal loss.
- **Model C — FraudGNN (Graph)** — PyTorch Geometric GraphSAGE on the (card / address / device / email) ↔ transaction bipartite graph.

Exactly one of these can be served on the production `/predict` path. Which?

[ADR-0001](0001-tech-stack.md) named LightGBM as the production-tier candidate at Sprint 0, on hand-wave grounds (latency + interpretability + tabular dominance). This ADR documents the **post-implementation validation** of that choice using Sprint 3.4's cross-model comparison + Sprint 5.1.f's measured latency + Sprint 5.2.c's promotion criteria.

The deciding metrics for "is this model the right champion?":

1. **AUC + AUC-PR** — discrimination quality.
2. **Brier score + ECE** — calibration quality (drives the cost-optimal threshold reliability).
3. **Inference latency P99** — must clear the 100 ms P95 budget with headroom.
4. **SHAP-compatibility** — interpretability requirement for analyst review.
5. **Operator-side cost** — joblib serialisation simplicity, dependency footprint, hot-reload semantics.

Alternatives considered:

1. **Deploy FraudNet as champion.** Best-test-AUC by ~0.016 over LightGBM; lower train AUC; ~60 ms `predict_proba` vs LightGBM's ~2 ms.
2. **Deploy FraudGNN as champion.** Best in-graph signal capture; but stateful (needs the full graph in memory at inference time) and 0.078 ms `predict_proba` on its 50K-node sample — at full 590K-node graph would be ~10-100x slower + memory-intensive.
3. **Voting ensemble (A + B + C).** Combines all three at inference time; ~70 ms aggregate latency; loses the SHAP-per-prediction story (no clean attribution).
4. **Stacked ensemble (A + B + C → linear meta-learner).** Same latency issues + adds a separate meta-model to train + calibrate. Trip the budget.
5. **Choose-best-per-segment routing.** Use FraudNet on slices where it wins, LightGBM elsewhere. Adds runtime branching + segment-detection latency; loses calibration consistency.

## Decision

| Model | Role | Reasoning |
|---|---|---|
| **Model A — LightGBM** | **Production champion** | Wins on calibration (Brier 0.0254 / ECE 0.0000 val); fastest inference (3.29 ms p95 isolated; ~1-2 ms predict_proba in-loop); SHAP-compatible (TreeExplainer); native joblib. |
| Model B — FraudNet | Shadow challenger (deployable; `Settings.shadow_enabled` gates) | Higher test AUC but 18× slower predict_proba; per-(card1, addr1, DeviceInfo) entity embeddings (dim=32) capture interactions LightGBM doesn't. See [ADR-0004](0004-shadow-mode.md) for the shadow architecture. |
| Model C — FraudGNN | **Batch-only feature provider** (NOT deployed) | Its outputs feed Tier-5 graph features into Model A. Building the graph at inference time is impossible under the latency budget; see [ADR-0006](0006-graph-features-batch.md). |

The decision is re-evaluated at every retraining cycle via Sprint 5.2.c's shadow-comparison report. Promotion criteria (all three must pass simultaneously):

- `cost_improvement > 2%` (Model B beats Model A by > 2% lower total economic cost over a labelled comparison window).
- `p_value < 0.05` (bootstrap significance test on the cost difference, two-sided).
- `agreement_rate > 85%` (champion and challenger agree on > 85% of decisions; below this the swap is too disruptive).

Until all three pass, Model A remains the champion.

## Rationale

1. **Calibration is load-bearing for the cost-optimal threshold.** [ADR-0003](0003-economic-threshold.md) picks τ = 0.080 from the calibrated probability surface. Brier 0.0254 (Model A) vs 0.0355 (Model B) means Model A's threshold is more reliable; a ~28% lower Brier translates to a measurably more stable τ. ECE 0.0000 (Model A) vs 0.0882 (Model B) is the same story at higher fidelity.
2. **Latency budget non-negotiable.** Sprint 5.1.f measured 70.98 ms P95 with LightGBM in-loop. FraudNet's ~60 ms predict_proba (Sprint 3.4 measurement) would push P95 past 130 ms — blowing the 100 ms budget by 30%. The shadow path's `asyncio.to_thread` workaround works because shadow is fire-and-forget; the production path can't use it without losing the SHAP-on-the-prediction story.
3. **SHAP interpretability is a hard requirement.** Sprint 5.1.e's `ShapExplainer` uses `shap.TreeExplainer`, which is LightGBM-native + O(n_features) per prediction. SHAP for FraudNet requires DeepExplainer or KernelExplainer (the former is ~10x slower; the latter is ~100x slower). The analyst-review workflow expects top reasons within the response — non-negotiable.
4. **AUC gap is small enough to NOT promote on AUC alone.** Test AUC 0.8070 (Model A) vs 0.8229 (Model B) — a ~0.016 gap. Sprint 5.2.c's promotion criteria require cost-based evidence + statistical significance + decision-agreement; small AUC wins don't clear that bar. The promotion gate prevents Goodhart's-Law cherry-picking.
5. **Joblib serialisation is simpler than PyTorch state_dicts** for atomic hot-reload. Sprint 5.1.d's `InferenceService.reload()` swaps the artefact bundle by a single GIL-atomic attribute rebind; equivalent for FraudNet requires loading the model + the entity-vocab + the calibrator + matching the version triple — more failure surface.
6. **FraudGNN's batch-only role is the right shape for graph features.** [ADR-0006](0006-graph-features-batch.md) documents the latency + memory reasoning. Putting graph signal into the model via batch-pre-computed features captures most of the value at inference cost of O(0 ms).

## Consequences

- **New diversity models join as shadow first.** Any candidate champion must pass the shadow-mode integration + Sprint 5.2.c's promotion gate before going live. A future deep-learning model (e.g., a tabular transformer) can be shadow-tested without touching the production path.
- **Calibration regression in Model A is the highest-leverage operational risk.** Sprint 6.1.b's PSI drift monitoring catches input drift; Sprint 6.1.c's PerformanceMonitor catches output regression. Both feed the [`docs/RUNBOOK.md`](../RUNBOOK.md) procedures.
- **The 18× latency gap to FraudNet's path is intrinsic** — the shadow path's `asyncio.to_thread` workaround works only because shadow is fire-and-forget. Promoting FraudNet to champion would require either (a) accepting a higher P95 budget (e.g., 200 ms) or (b) a model compression / distillation pass that brings predict_proba under 5 ms.
- **The decision is re-evaluated every retraining cycle.** [`docs/RUNBOOK.md#how-to-trigger-retraining`](../RUNBOOK.md#how-to-trigger-retraining) Step 5 calls for re-running `scripts/run_economic_evaluation.py`; Step 8 (validate) includes "compare champion + challenger on the new validation slice via Sprint 5.2.c's shadow_compare_report.py".
- **Stratified evaluation is the safety net for slice-specific regressions.** [`src/fraud_engine/evaluation/stratified.py`](../../src/fraud_engine/evaluation/stratified.py) (Sprint 4.2) gates each retrain on per-slice AUC + cost; if a future LightGBM retrain regresses on a slice where FraudNet performs better, the report surfaces it.
- **The model-card surface stays clean** — only one model has `model_version` on its `PredictionResponse`; only one model's calibration + Brier + ECE numbers appear in MODEL_CARD's headline metrics. Sprint 6.2.a's model card documents the cross-model comparison as a table in §Quantitative Analyses.

## Revisit triggers

- **Sprint 5.2.c's promotion criteria pass.** All three (cost_improvement > 2%, p_value < 0.05, agreement_rate > 85%) green on a 10K+-row comparison window. Trigger a deployment review.
- **Model A regresses on the no-identity slice** (Sprint 4.2 stratified) below the 0.75 AUC floor. If Model B holds that slice better, segment-aware routing becomes attractive.
- **Latency budget loosens** (e.g., the deploying institution's SLA allows 200 ms P95). FraudNet's 60-ms predict_proba becomes acceptable in-loop; SHAP for FraudNet (via DeepExplainer) becomes a more interesting trade-off.
- **A new candidate model joins** (e.g., a tabular transformer or quantile-regression model). Run it as shadow per [ADR-0004](0004-shadow-mode.md); re-evaluate this ADR if it passes the promotion gate.
- **Production fraud patterns shift toward graph-detected fraud rings** (organised crime with shared infrastructure). FraudGNN's role expands from feature-provider to potential champion; this ADR's premise needs revisiting.
- **The interpretability requirement changes** — e.g., a regulator accepts post-hoc explanations (LIME, integrated gradients) instead of SHAP. Opens the door to non-TreeExplainer-compatible models.

## References

- [`src/fraud_engine/api/inference.py`](../../src/fraud_engine/api/inference.py) — `InferenceService` (Sprint 5.1.d).
- [`src/fraud_engine/api/shap_explainer.py`](../../src/fraud_engine/api/shap_explainer.py) — `ShapExplainer` (Sprint 5.1.e).
- [`models/sprint3/lightgbm_model_manifest.json`](../../models/sprint3/lightgbm_model_manifest.json) — Model A artefact manifest with `best_score = 0.8281`.
- [`sprints/sprint_3/prompt_3_3_d_report.md`](../../sprints/sprint_3/prompt_3_3_d_report.md) — Model A training report.
- [`sprints/sprint_3/prompt_3_4_a_report.md`](../../sprints/sprint_3/prompt_3_4_a_report.md) — Model B (FraudNet) training report.
- [`sprints/sprint_3/prompt_3_4_b_report.md`](../../sprints/sprint_3/prompt_3_4_b_report.md) — Model C (FraudGNN) training report.
- [`sprints/sprint_3/prompt_3_4_c_report.md`](../../sprints/sprint_3/prompt_3_4_c_report.md) — cross-model comparison.
- [`sprints/sprint_5/prompt_5_1_f_report.md`](../../sprints/sprint_5/prompt_5_1_f_report.md) — Sprint 5.1.f P95 = 70.98 ms (with Model A).
- [`sprints/sprint_5/prompt_5_2_c_report.md`](../../sprints/sprint_5/prompt_5_2_c_report.md) — shadow comparison + promotion criteria.
- [`docs/MODEL_CARD.md#quantitative-analyses`](../MODEL_CARD.md#quantitative-analyses) — cross-model comparison table.
- Related: [ADR-0001 — Tech stack](0001-tech-stack.md) (initial pick); [ADR-0004 — Shadow mode](0004-shadow-mode.md) (architecture for non-champion candidates); [ADR-0006 — Graph features batch](0006-graph-features-batch.md) (Model C's role as feature provider).
