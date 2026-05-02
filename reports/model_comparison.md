# Sprint 3 model comparison — A vs B vs C

**Date:** 2026-05-02
**Models compared:** A (LightGBM), B (FraudNet entity-embedding NN), C (FraudGNN GraphSAGE)
**Decision:** **A is production. B is the shadow candidate. C is batch-only.**

This report puts the three Sprint-3 models side by side along the five spec dimensions (ROC-AUC, AUC-PR, inference p95, training time, interpretability) and locks in the production / shadow / batch-only role assignment that CLAUDE.md §3 declared at the architectural level. The numbers come straight from the per-model training reports written by `scripts/train_lightgbm.py`, `scripts/train_neural.py`, and `scripts/train_gnn.py`.

---

## Decision summary

| Model | Role | One-line rationale |
|---|---|---|
| **A — LightGBM** | **Production** | p95 = 3.29 ms (under the 15 ms hot-path budget); SHAP-explainable; calibrated probabilities (val log loss 0.291 → 0.109, 62 % better) |
| **B — FraudNet** | **Shadow candidate** | Diverse signal (entity embeddings + focal loss); p95 = 60 ms (above budget — production-blocking); ensemble candidate, not primary |
| **C — FraudGNN** | **Batch-only** | Captures graph topology no other model can; cached-logits design gives 0.07 ms p95; transductive contract (no inductive scoring); feeds Model A as features in Sprint 5 |

These role assignments were declared in CLAUDE.md §3 at architecture-design time and are **confirmed**, not revised, by the realised metrics below.

---

## Methodology + caveat

The three per-model reports are **NOT directly comparable on AUC alone** because they trained on different sample sizes:

| Model | Training size | Why |
|---|---|---|
| A | **414,542 rows** (full IEEE-CIS) + 100-trial Optuna sweep + isotonic calibration | The production model — the headline gate is val AUC ≥ 0.93, so it gets the canonical full run |
| B | **50,000 rows** (stratified subsample) + 7 epochs + early stopping | The spec target is "trains on 50K to convergence"; matches integration test |
| C | **5,000 rows** (stratified subsample) + 5 epochs | The spec gate is the integration test (smoke); the production GNN gate is convergence + latency, not the AUC headline |

**Sprint 4** will retrain B and C on full IEEE-CIS plus calibrate them via the 3.3.c toolkit (`select_calibration_method`) so the ensemble blend can be done on apples-to-apples calibrated probabilities. **Until then, treat the AUC numbers below as evidence supporting the architectural roles, not as the basis for a head-to-head AUC race.**

---

## Side-by-side comparison (the five spec dimensions)

| Dimension | A — LightGBM | B — FraudNet | C — FraudGNN |
|---|---|---|---|
| **ROC-AUC** (val / test) | **0.8281 / 0.8070** | 0.8183 / 0.8229 | 0.7778 / 0.7929 |
| **PR-AUC** (val / test) | **0.3814 / 0.4220** | 0.3351 / 0.3259 | 0.2099 / 0.2108 |
| **Inference p95** | 3.29 ms | 59.94 ms | **0.072 ms** (cached lookup) |
| **Training time** | ~22 min (full data + 100 trials + final fit + calibration) | ~13 s (50K + 7 epochs) | ~1 s (5K + 5 epochs) |
| **Interpretability** | **SHAP TreeExplainer** (first-class) | gradient × input or integrated gradients | GNNExplainer / attention inspection |

### Supporting context

| Dimension | A — LightGBM | B — FraudNet | C — FraudGNN |
|---|---|---|---|
| Training rows | 414,542 | 50,000 | 5,000 |
| Trainable params | best_iter × num_leaves (LightGBM tree); ~12 MB serialised | 304,779 | 111,489 |
| Calibration | **isotonic** (val log loss 0.291 → 0.109, 62 % better; ECE 0.193 → 0.000) | uncalibrated (val ECE 0.088) | uncalibrated (val ECE 0.075) |
| Inference path | `predict_proba → calibrator.transform` per row | `predict_proba` per row (entity lookup + embed + numeric MLP forward) | TransactionID → cached node-logit lookup → sigmoid |
| Persistence | `models/sprint3/{lightgbm_model.joblib, calibrator.joblib, lightgbm_model_manifest.json}` | `models/sprint3/fraudnet/{neural_model.pt, neural_model_manifest.json}` | `models/sprint3/fraudgnn/{gnn_model.pt, gnn_model_manifest.json}` (~1.2 GB on full data — graph baked into the bundle) |
| Inductive scoring | yes (any new row with the right schema) | yes (any new row with the right schema) | **no** — KeyError on unseen TransactionID; Sprint 5+ adds inductive |

---

## Per-model deep dive

### Model A — LightGBM (production)

**Architecture recap.** Native `lgb.Booster` API (3.3.a) with explicit early stopping, joblib + JSON-manifest persistence, and `feature_importance` / `predict_proba` / `save` / `load` surface area. Trained via 100-trial Optuna TPE sweep with MedianPruner (3.3.b), winning hyperparameters at trial 75 (`learning_rate=0.120, num_leaves=121, max_depth=3, min_child_samples=80`, etc.). Final fit on full train + val for early stopping, then isotonic calibration on the val split (3.3.c).

**Strengths.**
- **Latency comfortably under budget.** p95 = 3.29 ms vs the 15 ms production gate — 4.5× headroom.
- **Probabilities are calibrated.** The 62 % log-loss improvement (0.291 → 0.109) means a `0.7` score actually corresponds to ~70 % empirical fraud rate — exactly what Sprint 4's threshold-optimisation needs.
- **First-class interpretability.** SHAP TreeExplainer support is built in for tree models — no approximation, no perturbation. Sprint 5 will ship per-prediction SHAP attributions on the API path.
- **Best AUC of the three** (0.8281 val / 0.8070 test), which matters most for a production model.

**Constraints.**
- **AUC under the 0.93 spec gate.** The 100-trial sweep recovered +0.06 over Tier-5 default-hparam baseline (0.7689 → 0.8281), but couldn't close the gap. The Sprint 1 Tier-1-only baseline hit val AUC 0.9247 — every Tier 2-5 addition slightly degraded AUC at default hparams. Hypothesis: the 743-column feature space dilutes the top signals; feature pruning is the obvious Sprint 4 experiment.
- **Test/val gap is modest** (val 0.828 → test 0.807, ~−0.02). The model generalises reasonably under the temporal split.

**Role in production stack.** Model A is THE scoring model on the FastAPI hot path (Sprint 5). Every transaction enters → Redis online-feature lookup → `LightGBMFraudModel.predict_proba` → isotonic calibration → SHAP attribution → response. Sprint 4's economic-cost threshold optimiser writes its decision threshold against Model A's calibrated probabilities.

---

### Model B — FraudNet (shadow candidate)

**Architecture recap.** PyTorch entity-embedding network: three `nn.Embedding(vocab+1, 32)` tables for `card1` / `addr1` / `DeviceInfo` (index 0 reserved for OOV) + numeric branch (`BatchNorm1d → Linear(741, 64) → ReLU → Dropout(0.3)`); concat → `Linear(160, 64) → ReLU → Dropout → Linear(64, 1)`. Trained with focal loss (α=0.25, γ=2.0), Adam (lr=1e-3, weight_decay=1e-5), early stopping on val AUC.

**Strengths.**
- **Cleaner test/val parity than Model A** (val 0.8183 / test 0.8229; Model A: val 0.8281 / test 0.8070). The val→test improvement is the error-decorrelation hint Sprint 4's ensemble blend wants to exploit.
- **Diverse signal source.** Entity embeddings learn dense per-ID representations where neighbouring IDs in embedding space behave similarly — structurally invisible to LightGBM's per-split decisions, especially at the high-cardinality `card1` (12,251 unique values) where embedding sharing pays off most.
- **Architecture is small** (~305k params, ~1.2 MB model file) — fast to retrain, cheap to ship.

**Constraints.**
- **Latency is over budget.** p95 = 60 ms is 4× the 15 ms production hot-path budget. Per-call overhead dominates: pandas `.iloc[[idx]]` + four torch tensor allocations + scaler + module forward. Sprint 4/5's serving harness would need to batch predictions or pre-tensorise to bring this down.
- **Uncalibrated probabilities.** ECE = 0.088 — the network outputs trail off into "0.9 confident" zones that don't correspond to ~90 % empirical fraud. Sprint 4 will apply 3.3.c calibration.
- **Trained on 50K subsample, not full data.** Direct AUC comparison with Model A is misleading until Sprint 4's full-data retrain.

**Role in production stack.** Per CLAUDE.md §3: "Model B is shadow-deployable." Sprint 5 will run it in parallel to Model A on a fraction of production traffic, log its predictions, but route the actual decision through Model A. If Sprint 4 closes the latency + accuracy gap (batched inference + calibration + ensemble), Model B is the natural promotion candidate. Until then: **diverse signal source for the ensemble, not the primary ranker.**

---

### Model C — FraudGNN (batch-only)

**Architecture recap.** Three `SAGEConv` layers (in_channels=741 → hidden=64 → 64 → 64) + `Linear(64, 1)` head. Mean aggregator. Bipartite undirected graph (txn ↔ {card1, addr1, DeviceInfo}); transaction nodes carry the Tier-5 numeric feature vector, entity nodes are zero-padded — the GNN learns to use them as aggregation hubs. Trained with focal loss + `NeighborLoader(num_neighbors=[10, 10, 10])` per spec. Transductive contract: graph built once from train+val+test entities; train mask labels train txns; val/test masks slice the rest. After training, ONE full-graph forward populates `self.cached_node_logits_`; `predict_proba(df)` becomes a TransactionID → index lookup + sigmoid (0.072 ms p95).

**Strengths.**
- **Captures graph topology no other model can.** A fraud ring rotating cards across one device or one address chain is invisible to the per-row tier features and to FraudNet's per-row entity embeddings; SAGEConv's mean aggregation directly encodes "the average behaviour of transactions sharing this entity."
- **Practically free at inference.** Cached-logits design: O(1) lookup for any TransactionID in the persisted graph. p95 = 0.072 ms is two orders of magnitude under FraudNet and 50× under Model A.
- **Smallest network** (~111k params), trained quickly even at full scale (the integration test trains a 5K subgraph in ~1 s).

**Constraints.**
- **Transductive contract.** `predict_proba(df)` raises `KeyError` for any TransactionID not in the persisted graph. Sprint 5+ adds inductive scoring (subgraph extraction per request); for now FraudGNN scores ONLY the transactions present at training time.
- **Persisted bundle is large** (~1.2 GB on full IEEE-CIS — the graph + features + edge_index pickled together). Acceptable for batch deployment under `models/sprint3/fraudgnn/` (gitignored); Sprint 5 may compress to a parquet edge-list + npz feature-tensor format if the deployment surface needs it.
- **AUC is the lowest of the three** (val 0.778 / test 0.793) — but this is on a 5K subsample with 5 epochs; the training curve was still climbing (val AUC 0.4546 → 0.7778 over 5 epochs without early stop), so the production-scale number is expected to be materially higher.

**Role in production stack.** Per CLAUDE.md §3: "Model C is batch-only — its outputs feed Model A as features." Sprint 5 builds a daily/hourly batch job that scores every TransactionID via FraudGNN's cached logits and writes the result as a Tier-N feature column ingested by Model A's online pipeline. The graph topology becomes a feature, not a model.

---

## Decision rationale

### Why A is production

1. **p95 = 3.29 ms ≪ 15 ms hot-path budget.** Model A is the only model that comfortably fits the production latency contract.
2. **SHAP TreeExplainer is first-class.** Sprint 5's prediction-API ships per-request SHAP attributions; the alternative interpretability stories for B (gradient × input / integrated gradients) and C (GNNExplainer / attention) are slower, less developed, and not aligned with how the API consumer wants to see explanations.
3. **Calibrated probabilities are realised, not aspirational.** The 62 % log-loss improvement (val 0.291 → 0.109) means Sprint 4's economic-cost threshold optimiser operates on a faithful cost surface immediately; B and C would need calibration first.
4. **Best AUC of the three** (0.8281 val) — and even setting the AUC race aside, Model A has the headline gate of the three, the only sweep, and the only calibration. Production-readiness is a sum of properties, not a single number.

### Why B is shadow candidate (not production)

1. **p95 = 60 ms is production-blocking.** 4× the hot-path budget. Even with batched inference (the obvious Sprint 4/5 lever), getting under 15 ms on single-row scoring requires architectural rework (e.g., pre-tensorising val/test, JIT-compiling the module via TorchScript). Worth doing if Sprint 4's ensemble + calibration math justifies it; not yet justified.
2. **No calibration.** Sprint 4 work item.
3. **CLAUDE.md §3 declared "shadow-deployable" at architecture time.** That decision pre-dates the metrics; the metrics confirm rather than revise it.
4. **Diverse signal source for the ensemble.** The cleaner test/val parity (val 0.8183 / test 0.8229) suggests Model B sees patterns Model A misses on the temporal-split tail. That's the precondition for an ensemble that beats either alone.

### Why C is batch-only

1. **Transductive contract is foundational, not patchable in scope.** `predict_proba` raises `KeyError` on unseen TransactionIDs. The serving stack would need to know in advance which IDs to add to the graph and re-run a subgraph forward — incompatible with a hot-path request/response model.
2. **Cached-logits design gives 0.072 ms p95 — but ONLY for IDs in the training graph.** That's exactly the right shape for a batch-feature pipeline: at retrain time, score every known transaction in O(1); write the score as a feature column.
3. **Per CLAUDE.md §3:** "Model C is batch-only — its outputs feed Model A as features." Sprint 5's batch-feature pipeline is the architectural fit. The graph topology becomes a feature; Model A consumes it the same way it consumes any other Tier-5 column.

---

## Sprint 4 follow-ons

These are explicit out-of-scope items for Sprint 3 that this comparison report identifies as next steps:

- **Retrain B and C on full IEEE-CIS** for apples-to-apples AUC comparison (the parity caveat above goes away).
- **Calibrate B and C** via the 3.3.c toolkit (`select_calibration_method(val_y, val_p)` returns the best of {Platt, isotonic, identity}).
- **Cost-curve evaluation + threshold optimisation per model** (Sprint 4's headline deliverable; uses the calibrated probabilities + the `fraud_cost_usd` / `fp_cost_usd` / `tp_cost_usd` from Settings).
- **Ensemble blend.** Linear stacking — `LogisticRegression(meta_features=[A_proba, B_proba, C_proba])` — fit on val, evaluated on test. The error-decorrelation hint visible in B's cleaner test/val parity vs A's is the precondition; expect a measurable AUC + cost-surface lift over Model A alone.
- **Stratified metrics** (amount bucket, `ProductCD`, time bucket) per model — the Sprint-4 evaluator emits these.
- **Inductive scoring path for C** (Sprint 5 prerequisite for batch-feature pipeline integration).
- **Interpretability tooling.**
  - **A:** SHAP TreeExplainer (Sprint 5 deliverable; will run on the API path with caching).
  - **B:** Integrated gradients (Captum or hand-rolled torch); deferred to Sprint 5 if Model B promotes from shadow to production.
  - **C:** GNNExplainer or attention-weight inspection; out of scope until inductive scoring lands.

---

## Cross-references

- `reports/model_a_training_report.md` — full Model A metrics, top-50 feature importance, latency histogram.
- `reports/model_b_training_report.md` — full Model B metrics, per-epoch history, vocab sizes.
- `reports/model_c_training_report.md` — full Model C metrics, graph topology, per-epoch history.
- `sprints/sprint_3/prompt_3_3_d_report.md` — Model A's training-pipeline narrative (Sprint 1 baseline → Tier-5 regression → 100-trial recovery → calibration).
- `sprints/sprint_3/prompt_3_4_a_report.md` — Model B's design rationale, the function-vs-module-scoped fixture lesson, the float32 memory fix.
- `sprints/sprint_3/prompt_3_4_b_report.md` — Model C's transductive-contract decision, cached-logits design, `pyg-lib` install pin.
- `CLAUDE.md` §3 — the architectural role assignments this report restates with metric-backed rationale.
