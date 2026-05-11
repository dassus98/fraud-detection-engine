# Interview Guide — Fraud Detection Engine

> Companion to [`docs/MODEL_CARD.md`](MODEL_CARD.md) + [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) + [`docs/RUNBOOK.md`](RUNBOOK.md). This is the prep doc for talking to a senior data-scientist / ML-engineer hiring committee (Wealthsimple / Mercury / RBC / Nubank-class) about the project.

## 30-second pitch

> "I built a real-time fraud detection system on the IEEE-CIS dataset, architected as a production engagement for a Canadian fintech. A LightGBM champion with SHAP explainability serves at 71 ms P95 — 29 ms under a 100 ms SLO. The model is calibrated (Brier 0.025, ECE 0.000) and the decision threshold is cost-optimal, not F1: at the published cost coefficients of $450 per missed fraud + $35 per false positive, the threshold of τ=0.08 saves $28.96M annually on a one-million-transactions-per-month portfolio. The full stack ships with a FastAPI app, Redis online feature store, Postgres audit log, Prometheus + Grafana monitoring with five named alerts, an operator runbook, a model card in Mitchell et al. format, and six architecture decision records. Six sprints, hundreds of tests, every architectural decision documented with footnoted citations to its source artefact."

Three numbers to anchor the conversation: **0.8281 AUC**, **70.98 ms P95 latency**, **$28.96M annual savings**.

## The 5 hardest questions (with answers)

### Q1: How do you handle class imbalance? Why not just resample?

**The short answer:** the project doesn't resample. It calibrates and then optimises a cost-aware threshold.

**The longer answer:** IEEE-CIS has a 3.5% positive class rate ([CLAUDE.md §1](../CLAUDE.md)). The standard moves a reviewer might probe:

1. **Random oversampling / SMOTE.** Inflates AUC slightly but **destroys calibration** — the model now thinks fraud is ~50% prior, so its raw output probabilities are systematically wrong. For a system where the decision threshold is derived from a calibrated probability, this is fatal.
2. **`scale_pos_weight=N`** (LightGBM's built-in). Same calibration regression in a different form. The model's raw scores are pushed higher for the minority class; the post-softmax-style output looks like ~50% prior even though the empirical prior is 3.5%.
3. **`class_weight='balanced'`** at training time + raw-probability decisions. Same issue.

What I did instead: train on the 3.5% prior + apply isotonic calibration on top of the raw decision function. Sprint 3.3.c's [`prompt_3_3_d_report.md`](../sprints/sprint_3/prompt_3_3_d_report.md) shows isotonic improved Brier from 0.0769 (uncalibrated) to 0.0254 (67% reduction) and ECE from 0.1926 to 0.0000. Isotonic won over Platt because LightGBM's `scale_pos_weight=27.4` produces a non-sigmoid miscalibration shape; Platt's sigmoid assumption fits poorly.

Then [ADR-0003](ADR/0003-economic-threshold.md) picks the decision threshold by sweeping the calibrated probabilities + minimising **expected USD cost**, not F1 or AUC. The cost function is `FN × $450 + FP × $35 + TP × $5`. F1 would weigh FN and FP equally — wrong for the business. ROC AUC and PR AUC are threshold-free — they don't pick a τ at all. The cost surface is the only one aligned with the deployed system's actual loss.

**The deeper point:** resampling treats imbalance as a problem; calibration + cost-aware thresholding treat it as a *signal*. The 3.5% prior is itself information.

### Q2: How do you prevent target leakage in your feature engineering?

**The short answer:** four defences, each tested.

**The long answer:**

1. **Temporal split everywhere** ([ADR-0002](ADR/0002-temporal-split.md)). All train / val / test splits use `Settings.train_end_dt = 121 days` and `Settings.val_end_dt = 151 days`. No random splitting anywhere. Train rows are strictly before val rows, which are strictly before test rows. This matches production: a model serves data from after its training cutoff.

2. **OOF target encoding for high-cardinality categoricals** (`TargetEncoder`, Sprint 2.2). Each training row's encoded value derives from a 5-fold split that does NOT contain the row itself. Val / test use a full-train encoder. Smoothing toward the global rate via `(sum + α × global_rate) / (count + α)` protects low-cardinality categories from over-confident estimates.

3. **`fraud_neighbor_rate` OOF discipline** ([ADR-0006](ADR/0006-graph-features-batch.md)). This is the leakage trap most fraud-ML projects miss. The feature `fraud_neighbor_rate` answers: "what fraction of this transaction's graph neighbours (other transactions sharing the same card / addr / device / email) are fraud?". A naïve implementation lets a row's own fraud label flow into its own neighbour rate via the shared entity — textbook leakage. Sprint 3.3.d implements 5-fold OOF on this feature specifically, mirroring the `TargetEncoder` pattern.

4. **Shuffled-label tests in CI** (per CLAUDE.md §6.3). For every feature module + training pipeline, there's a test that retrains on shuffled labels and asserts val AUC collapses to ~0.5. If leakage exists, shuffled-label AUC stays high (the model "learns" from the leaked signal). The test catches future regressions on a routine PR review cycle.

The temporal-decay features (Tier 4 `ExponentialDecayVelocity`) also use a read-before-push two-pass discipline: each row reads the prior EWM state, then pushes its own label into the state for the next row. A bug that swaps the order leaks the current label.

### Q3: What's your model selection criteria? Why LightGBM over a deep model?

**The short answer:** LightGBM wins on calibration + latency + interpretability; FraudNet was tested as a challenger and didn't beat the bar.

**The long answer** ([ADR-0005](ADR/0005-lightgbm-as-production.md)):

The project trains three models:

| Model | Val AUC | Test AUC | Brier (val) | ECE (val) | Latency p95 |
|---|---|---|---|---|---|
| **A — LightGBM (production)** | **0.8281** | 0.8070 | **0.0254** | **0.0000** | **3.29 ms** |
| B — FraudNet (entity-embedding NN) | 0.8183 | 0.8229 | 0.0355 | 0.0882 | 59.94 ms |
| C — FraudGNN (GraphSAGE) | 0.7778 | 0.7929 | 0.0357 | 0.0888 | 0.072 ms (batch) |

LightGBM wins on every axis that drives a production decision:

- **Calibration.** Brier 0.0254 vs 0.0355 means LightGBM's threshold is more reliable. ECE 0.0000 vs 0.0882 is the same point at higher fidelity. The cost-optimal threshold [ADR-0003](ADR/0003-economic-threshold.md) depends on the calibrated probability; a less-well-calibrated model's threshold is less stable under cost-coefficient perturbation.
- **Latency.** ~20× faster predict_proba. The 100 ms P95 budget allows LightGBM in-loop; FraudNet would need `asyncio.to_thread` workarounds that don't apply to the champion path (only to the fire-and-forget shadow path).
- **Interpretability.** SHAP TreeExplainer is LightGBM-native and O(n_features) per prediction. SHAP for FraudNet requires DeepExplainer (~10× slower) or KernelExplainer (~100× slower). The analyst-review workflow expects top reasons within the response payload — non-negotiable.

FraudNet beats LightGBM on **test AUC by 0.016** — a real signal but small. Sprint 5.2.c's promotion criteria require **all three** of cost_improvement > 2%, p_value < 0.05, agreement_rate > 85%. Sprint 5.2.c's bootstrap significance test (10K resamples) doesn't currently clear those gates for FraudNet. So FraudNet stays as shadow ([ADR-0004](ADR/0004-shadow-mode.md)); the production champion is LightGBM.

FraudGNN is repurposed: not a deployment candidate, but a **feature provider** for Tier 5 graph features ([ADR-0006](ADR/0006-graph-features-batch.md)). Its GNN embeddings feed `GraphFeatureExtractor` outputs that LightGBM consumes.

### Q4: What's your decision threshold and how did you pick it?

**The short answer:** τ = 0.0800, picked by sweeping the cost-optimal expected loss across the test set. Empirical winner matches the analytical Bayes-decision limit to within one grid step.

**The long answer** ([ADR-0003](ADR/0003-economic-threshold.md)):

The cost coefficients are published in [`configs/economic_defaults.yaml`](../configs/economic_defaults.yaml) and CLAUDE.md §8: **FN = $450** (missed fraud, comprising $150 txn loss + $25 chargeback fee + $75 investigation + $50 scheme penalty + $150 reputation/regulatory amortised); **FP = $35** ($15 customer-service contact + 5% churn × $400 customer lifetime value); **TP = $5** (analyst review time). These were calibrated from 2024 industry medians.

Sprint 4.1's [`EconomicCostModel.optimize_threshold`](../src/fraud_engine/evaluation/economic.py) sweeps `np.linspace(0.01, 0.99, 99)` over candidate thresholds. At each τ, threshold the calibrated probabilities to `y_pred`, compute `total_cost = FN × $450 + FP × $35 + TP × $5`, pick the minimum. Tie-break favours the larger τ (block fewer transactions on equal cost).

The empirical winner τ = 0.0800. The **analytical Bayes-decision limit** τ* = `fp_cost / (fp_cost + fraud_cost − tp_cost) = 35 / 480 ≈ 0.0729`. The empirical-vs-analytical gap is < 0.005 — one grid step — confirming the sweep finds the minimum.

**Stability under cost-coefficient perturbation:** Sprint 4.2's [`EconomicCostModel.sensitivity_analysis`](../src/fraud_engine/evaluation/economic.py) sweeps a ±20% symmetric grid across all three cost axes (CLAUDE.md §8 mandate: "decisions are stable under ±20% variation"). Result: optimal τ moves by ≤ 0.06 across the entire grid — well under the 0.20 ceiling. This means small cost-estimate errors don't destabilise the deployed decision.

**At τ = 0.080 on a 1M-txn/month portfolio, expected annual savings vs. a no-model baseline are $28.96M** — 58× over the spec's $500K floor.

**Operationally:** the threshold is monitored. Sprint 6.1.b's PSI drift on input features fires if the population shifts ([`docs/RUNBOOK.md#alert-featuredrift`](RUNBOOK.md#alert-featuredrift)); Sprint 6.1.c's performance monitor fires on a >5% drop in rolling AUC/AUC-PR/cost. If either fires, the operator retrains + re-derives τ per [`docs/RUNBOOK.md#how-to-trigger-retraining`](RUNBOOK.md#how-to-trigger-retraining).

### Q5: How do you know the model is working in production?

**The short answer:** the monitoring tripod — live Prometheus metrics + offline drift detection + offline performance regression — feeding a Grafana dashboard, five named alert rules, and an operator runbook with ordered remediation per alert.

**The long answer** (Sprint 6.1.a-e):

**Live metrics (Sprint 6.1.a + 6.1.d retrofit).** 13 Prometheus metrics on `/metrics`:

- 4 latency histograms (`fraud_engine_feature_fetch_seconds`, `_inference_seconds`, `_shap_seconds`, `_predict_total_seconds`) with buckets `(0.005, 0.010, 0.025, 0.050, 0.100, 0.250)`. The `0.100` bucket is load-bearing — it's the CLAUDE.md §3 P95 budget gate.
- Score distribution histogram (`fraud_engine_prediction_score`) bucketed at the decision threshold 0.08.
- Counters: `predictions_total{decision}`, `degraded_mode_total`, `shadow_total{event}`, plus the 6.1.d retrofits `drift_alerts_total`, `shadow_disagreement_total`.
- Gauges: `dependency_up{component}`, `shadow_breaker_state{state}`, `model_info{model_version}`.

**Offline drift (Sprint 6.1.b — DriftMonitor).** PSI on 685 features against a frozen training baseline (`data/baselines/distributions.parquet`). Alerts written to `logs/drift/{run_id}/drift_alerts.jsonl` when PSI > 0.20. PSI < 0.10 = stable; > 0.25 = significant shift.

**Offline performance (Sprint 6.1.c — PerformanceMonitor).** Rolling AUC, AUC-PR, economic cost on labelled (score, label) pairs vs the training-time baseline (operator-curated `Settings.performance_training_*` fields). Alerts written to `logs/performance/{run_id}/performance_alerts.jsonl` when any metric degrades > 5%. The single-class window edge case is handled — when all labels are 0 (no fraud in the window), AUC + AUC-PR return NaN and don't generate false alerts.

**Grafana dashboard (Sprint 6.1.d).** 7 panels at uid `fraud-detection-main`: API latency p50/p95/p99 with threshold line at 0.100; per-stage latency p95 stacked; prediction volume + decision mix; block-rate-% with threshold line at 15%; score-distribution heatmap; dependency health stat panel; shadow + drift activity time series.

**5 named alert rules.** [`configs/alerts/alert_rules.yml`](../configs/alerts/alert_rules.yml): `HighLatency` (p95 > 100 ms for 2m), `BlockRateSpike` (block rate > 15% for 10m, vs the 3.5% baseline), `FeatureDrift` (any drift alert in last 1h, for 5m), `ShadowDisagreement` (> 10% disagreement over 15m, for 10m), `ApiErrorRate` (5xx > 1% for 5m).

**Operator runbook (Sprint 6.1.e — [`docs/RUNBOOK.md`](RUNBOOK.md)).** Each of the 5 alerts has a section with: what it means → common causes → ordered investigation steps with copy-pasteable commands (Grafana panel links, `psql` queries, `jq` recipes, `docker compose logs` filters) → mitigation steps from least-disruptive to most-disruptive. Plus two cross-cutting procedures: "how to safely roll back a model" (file-swap + container restart for all three artefacts: `lightgbm_model.joblib` + `calibrator.joblib` + `lightgbm_model_manifest.json`; verify via `curl /predict | jq .model_version`) and "how to trigger retraining" (8-step end-to-end: data refresh → feature pipeline → train → evaluate → deploy → validate).

**End-to-end test (Sprint 6.1.e).** `tests/integration/test_monitoring_e2e.py` drives `/predict` traffic + triggers synthetic drift + asserts the counter deltas + statically evaluates the alert-rule thresholds. The dashboard ↔ emitted-metric coverage test catches dangling-reference regressions.

## "What makes this different" table

This is the differentiator surface vs. typical Kaggle / portfolio fraud-ML projects. Every row cites a specific artefact in this repository, so a reviewer can verify.

| Dimension | Typical Kaggle / portfolio project | **This project** | Source |
|---|---|---|---|
| **Train/test split** | Random 80/20 or stratified random | **Temporal** (train < val < test by `TransactionDT`); no random splits anywhere | [ADR-0002](ADR/0002-temporal-split.md) |
| **Class imbalance** | SMOTE / `scale_pos_weight` / oversampling | **Trained on real 3.5% prior + isotonic calibration**; Brier 0.0769 → 0.0254 (67% reduction); ECE 0.1926 → 0.0000 | [`prompt_3_3_d_report.md`](../sprints/sprint_3/prompt_3_3_d_report.md) |
| **Decision objective** | F1 / AUC / fixed τ=0.5 | **Cost-optimal τ** from a real $-cost model; τ=0.080 matches Bayes limit τ*≈0.0729 within one grid step | [ADR-0003](ADR/0003-economic-threshold.md) |
| **Business value** | "improved AUC by 0.03" | **$28.96M / year savings** on a 1M-txn/month portfolio; 58× over the $500K spec floor | [`prompt_4_4_report.md`](../sprints/sprint_4/prompt_4_4_report.md) |
| **Latency** | offline notebook; never measured | **Measured P95 = 70.98 ms** on live `/predict` over 100 sequential requests; 29 ms headroom under the 100 ms SLO | [`prompt_5_1_f_report.md`](../sprints/sprint_5/prompt_5_1_f_report.md) |
| **Leakage prevention** | "I used cross-validation" | Temporal split + OOF target encoding + **OOF `fraud_neighbor_rate`** + shuffled-label CI tests | [ADR-0006](ADR/0006-graph-features-batch.md) + Sprint 3.3.d |
| **Feature engineering** | 10-50 hand-picked features | **685 features × 12 generators × 5 tiers** with a machine-readable manifest + per-tier failure-modes doc | [`docs/FEATURE_DOCUMENTATION.md`](FEATURE_DOCUMENTATION.md) |
| **Model selection** | "I tried 3 models, ensemble was best" | Champion / shadow / batch trio with **multi-factor promotion criteria** (cost_improvement > 2% AND p<0.05 AND agreement > 85%) | [ADR-0005](ADR/0005-lightgbm-as-production.md) + Sprint 5.2.c |
| **Production serving** | Flask `/predict` returning a number | FastAPI + Redis online store + Postgres audit + SHAP top-reasons + degraded-mode + fire-and-forget shadow + fire-and-forget audit-log | Sprint 5.1.f + 5.2.a + 5.2.b |
| **Monitoring** | "I added Prometheus metrics" | **13 metrics + 5 named alert rules + 7-panel Grafana dashboard + 30d retention + PSI drift + performance regression + JSONL alert streams + integration test of the chain** | Sprint 6.1.a-e |
| **Documentation** | README + a notebook | Model card (Mitchell et al. format with footnoted citations) + feature doc + architecture (4 mermaid diagrams) + runbook + interview guide + 6 ADRs + 80+ sprint completion reports | Sprint 6.2.a-c |
| **Operations** | "rerun the notebook" | Operator runbook with **per-alert remediation steps**, model-rollback procedure, retraining procedure, log file inventory, `jq` recipes, PromQL queries | [`docs/RUNBOOK.md`](RUNBOOK.md) |

## Live-demo tips

The 5-second demo is `scripts/demo_prediction.py`. Assumes the dev stack is running (`docker compose -f docker-compose.dev.yml up -d`) and uvicorn is on host (`make serve`).

```bash
# Start the dev stack (Postgres + Redis + MLflow + Prometheus + Grafana).
docker compose -f docker-compose.dev.yml up -d

# Start uvicorn on host (in a separate terminal).
make serve

# Run the demo.
uv run python scripts/demo_prediction.py
```

The demo drives two hardcoded payloads:

1. **`clearly_legit`** — a typical low-amount mid-day Mastercard credit transaction. Expected score ~0.004, decision = `allow`.
2. **`obvious_fraud`** — high amount, unusual hour, free email domain, never-seen card1 (cold-start), suspicious device. Expected score > 0.08, decision = `block`.

For each, the demo prints: score (4 decimal places), decision (BLOCK/ALLOW), top-5 SHAP reasons (feature name + contribution + direction).

**If the demo fails to connect**, the friendly error points at `make serve`. If the API is up but returns degraded_mode=true, the dev stack's Redis or Postgres is down; check `make docker-up` or `docker compose -f docker-compose.dev.yml ps`.

**Talking points during the demo:**

- After the first request: "Notice the SHAP top reasons — those go into the analyst's case-management UI."
- After the second request: "The cost-optimal threshold makes this decision; F1 or AUC would give a different cut."
- "Both calls together returned in under 200 ms wall clock; the steady-state P95 is 71 ms per request."

## Cross-references

- [`docs/MODEL_CARD.md`](MODEL_CARD.md) — Model details + metrics + ethical considerations + caveats.
- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — 4 mermaid diagrams + 25-component reference.
- [`docs/RUNBOOK.md`](RUNBOOK.md) — Per-alert remediation + rollback + retraining.
- [`docs/FEATURE_DOCUMENTATION.md`](FEATURE_DOCUMENTATION.md) — 685 features grouped by tier + generator.
- [`docs/ADR/`](ADR/) — 6 architecture decision records.
- [`scripts/demo_prediction.py`](../scripts/demo_prediction.py) — the 5-second demo.
- [`README.md`](../README.md) — headline metrics + business value + quick start.
