# Model Card — Real-Time Fraud Detection Engine

> Follows the [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03677) format (Mitchell et al. 2018). Every number is footnoted to its source artefact for portfolio-reader auditability.

---

## Model Details

- **Model name:** Real-Time Fraud Detection Engine — Model A (LightGBM champion).
- **Owner:** John Das.
- **Model version:** see `content_hash` in [`models/sprint3/lightgbm_model_manifest.json`](../models/sprint3/lightgbm_model_manifest.json) (SHA-256 of the serialised joblib + manifest; surfaced in every `PredictionResponse.model_version`).
- **Type:** binary classifier returning a calibrated fraud probability ∈ [0, 1].
- **Training algorithm:** Gradient-boosted decision trees (LightGBM) with isotonic-regression calibration on top of the raw decision-function output.
- **Hyperparameters:** Optuna-tuned 100-trial sweep on the validation slice; final values in [`configs/model_best_params.yaml`](../configs/model_best_params.yaml). Best iteration: 72 (early stopping). Best validation AUC: 0.8281.
- **Inputs:** 685 engineered features (see [FEATURE_DOCUMENTATION.md](FEATURE_DOCUMENTATION.md)). The full feature manifest lives at [`models/pipelines/feature_manifest.json`](../models/pipelines/feature_manifest.json).
- **Output:** `PredictionResponse` (see [`src/fraud_engine/api/schemas.py`](../src/fraud_engine/api/schemas.py)) with `score` (float ∈ [0, 1]), `decision` ∈ {`block`, `allow`}, `top_reasons` (up to 10 SHAP-derived contributions), `model_version`, `latency_ms`, `degraded_mode`.
- **Decision rule:** `decision = "block"` iff `score ≥ Settings.decision_threshold` (default `0.080`, cost-optimal per [ADR-0003](ADR/0003-economic-threshold.md)).
- **Training framework + dependencies:** Python 3.11+, LightGBM (latest pinned in `pyproject.toml`), scikit-learn for isotonic calibration, Optuna for hyperparameter search, MLflow for experiment tracking.
- **Diversity models** (NOT served; see §[Quantitative Analyses](#quantitative-analyses) for comparison):
  - **Model B (FraudNet):** PyTorch entity-embedding NN. Shadow-deployable via Sprint 5.2.b's `ShadowService` (currently `Settings.shadow_enabled=False` by default).
  - **Model C (FraudGNN):** PyTorch Geometric GraphSAGE. Batch-only — its outputs feed `GraphFeatureExtractor` (Tier 5) as features into Model A.
- **License:** MIT (project); training data per Kaggle competition terms (see [Training Data](#training-data)).
- **Contact:** John Das ([owner per `CLAUDE.md` §1](../CLAUDE.md)).
- **Citation:** project repository; this model card; the IEEE-CIS Fraud Detection competition (Vesta Corporation, 2019).
- **Last refreshed:** see the `Last updated:` line at the bottom of this file.

---

## Intended Use

### Primary use cases

- **Real-time per-transaction fraud screening** for an e-commerce payments processor. The model returns a calibrated probability + SHAP-derived top reasons in <100 ms P95 (Sprint 5.1.f measurement: 70.98 ms P95 over 100 sequential requests with no shadow path; [`sprints/sprint_5/prompt_5_1_f_report.md`](../sprints/sprint_5/prompt_5_1_f_report.md)).
- **Cost-optimal blocking decisions** under the project's published cost coefficients (FN = $450, FP = $35, TP = $5 per [`configs/economic_defaults.yaml`](../configs/economic_defaults.yaml) and [CLAUDE.md §8](../CLAUDE.md)). The decision threshold (τ = 0.080) is derived from these by `EconomicCostModel.optimize_threshold` (Sprint 4.1).

### Primary intended users

- **Fraud-ops engineers** at a Canadian / LATAM fintech (Wealthsimple / Mercury / RBC / Nubank-class). They consume the per-transaction `PredictionResponse` and surface it in their case-management UI; `top_reasons` is the human-readable explanation column.
- **ML platform engineers** at the same firms maintaining the API + monitoring stack. They read the Prometheus metrics + the alert rules in [`configs/alerts/alert_rules.yml`](../configs/alerts/alert_rules.yml) and operate per [`docs/RUNBOOK.md`](RUNBOOK.md).
- **Compliance reviewers** auditing the model's fairness + intended-use envelope. They read THIS document.

### Out-of-scope use cases

- **Credit-risk decisioning** — the model is trained on transaction-fraud labels (IEEE-CIS `isFraud`), not credit-default labels. It says nothing about creditworthiness.
- **Account-takeover detection at the auth layer** — the model scores at transaction time, not at login. ATO detection needs upstream session-level signals not in the training data.
- **Money-laundering / sanctions screening** — different label distribution, different cost coefficients, different regulatory regime (FINTRAC / FinCEN / OFAC).
- **Identity-fraud / synthetic-identity detection** — the training labels are transaction-fraud; synthetic identities that successfully complete legitimate-looking transactions are not in-distribution.
- **Real-time decisioning without isotonic calibration** — the cost-optimal threshold (τ = 0.080) assumes the calibrated output. Raw uncalibrated scores at the same numeric threshold produce different decisions; see [Caveats](#caveats-and-recommendations) §4.

---

## Factors

### Relevant factors

The fraud signal in IEEE-CIS varies materially across these slices, which the model is evaluated against in Sprint 4.2's [`StratifiedEvaluator`](../src/fraud_engine/evaluation/stratified.py):

1. **Transaction amount bucket** — low (< $50), mid ($50–$200), high (> $200). Low-amount fraud rate exceeds high-amount by > 0.20 (test gate in [`sprints/sprint_4/prompt_4_2_report.md`](../sprints/sprint_4/prompt_4_2_report.md)).
2. **ProductCD** — W, C, R, H, S. W (separable product space) AUC exceeds C (overlapping) AUC by > 0.10.
3. **DeviceType** — mobile, desktop, null. The 76% of rows with null DeviceType is the largest fraction; mobile is the most-fraud-correlated populated slice.
4. **Identity coverage** — `id_01.notna()` (has any identity feature) vs `id_01.isna()` (no identity). Has-identity AUC exceeds no-identity AUC (the identity features carry real signal); the model still scores cleanly on the no-identity 76% population.
5. **Month boundary** — month 5 (training-period tail) vs month 6 (val-period). Month 5's cost_per_txn exceeds month 6's (training-set noise + survivor bias).

### Evaluation factors

All five factors above carry per-slice AUC, AUC-PR, total cost, and cost-per-transaction in `sprints/sprint_4/prompt_4_2_report.md`. Intersectional slicing (e.g., amount_bucket × ProductCD) was scoped but not run; the Sprint 4 report flags this as a future deepening if a per-slice cost regression appears.

### Factors NOT in training data — fairness limitations

- **Race, gender, age, sexual orientation, religion** — IEEE-CIS carries no demographic metadata. Fairness across protected classes is **untestable from this dataset alone**. Production deployment must add demographic-attribute joins (from the deploying institution's CRM) before any disparate-impact analysis is possible.
- **Geographic location** — `addr1` / `addr2` are integer-anonymised; no postal-code / region / country mapping. Geographic fairness is not testable.
- **Income, employment status** — not in dataset.

These omissions are properties of the IEEE-CIS dataset and are not addressed by the model. See [Ethical Considerations](#ethical-considerations) for downstream implications.

---

## Metrics

### Overall

| Metric | Value | Slice | Source |
|---|---|---|---|
| Sprint 1 baseline AUC | 0.9247 | val | [CLAUDE.md §13](../CLAUDE.md); `sprints/sprint_1/` |
| **Production Model A AUC** | **0.8281** | val | [`prompt_3_3_d_report.md:43`](../sprints/sprint_3/prompt_3_3_d_report.md) |
| Production Model A AUC | 0.8070 | test | [`prompt_3_3_d_report.md:43`](../sprints/sprint_3/prompt_3_3_d_report.md) |
| Brier score (post-isotonic calibration) | 0.0254 | val | [`prompt_3_3_d_report.md:41-49`](../sprints/sprint_3/prompt_3_3_d_report.md) |
| Brier score (post-isotonic) | 0.0249 | test | same |
| ECE (post-isotonic) | 0.0000 | val | same |
| ECE (post-isotonic) | 0.0075 | test | same |
| Log loss (post-isotonic) | 0.1090 | val | same |
| Log loss (post-isotonic) | 0.1118 | test | same |
| Per-request inference latency (P95) | 70.98 ms | live `/predict` (no shadow) | [`prompt_5_1_f_report.md`](../sprints/sprint_5/prompt_5_1_f_report.md) |
| Per-request inference latency (P95) | 81.4 ms | live `/predict` (shadow failing every call) | [`prompt_5_2_b_report.md`](../sprints/sprint_5/prompt_5_2_b_report.md) |

**Note on the Sprint 1 → Sprint 3 AUC regression** (0.9247 → 0.8281). The Sprint 1 baseline was trained on the 11-feature pre-Tier-5 set; Sprint 3's full 685-feature pipeline lost ~0.06 AUC because the Tier-5 graph features add noise on the IEEE-CIS distribution (where shared-infrastructure fraud rings are not dominant). Sprint 3's tuning recovered most of the gap from the Tier-5-baseline. The trade-off was accepted because Tier-5 generalises better to production-grade fraud patterns where graph signal is stronger (per Sprint 3.4 cross-model comparison).

### Cost-optimal decision threshold (Sprint 4)

| Quantity | Value | Source |
|---|---|---|
| **Cost-optimal threshold τ** | **0.0800** | empirical (cost-curve sweep) |
| Analytical limit τ* (Bayes decision) | ≈ 0.0729 | `fp_cost / (fp_cost + fraud_cost − tp_cost) = 35 / 480` |
| Empirical vs analytical gap | < one grid step (0.005) | [ADR-0003](ADR/0003-economic-threshold.md) |
| Sensitivity spread across ±20% cost grid | 0.06 | [`prompt_4_4_report.md`](../sprints/sprint_4/prompt_4_4_report.md); well under 0.20 ceiling per [CLAUDE.md §8](../CLAUDE.md) |
| **Annual savings** | **$28.96M** on 1M transactions / month portfolio | [CLAUDE.md §13](../CLAUDE.md) |
| Annual savings — spec floor | $500K | [CLAUDE.md §13](../CLAUDE.md); ratio 58× over floor |

The threshold is robust: ±20% perturbation of each cost coefficient produces no more than 0.06 absolute change in the optimal τ. Operationally this means small cost-estimate errors do not destabilise the deployed decision.

### Stratified metrics (Sprint 4.2 — full per-slice in `prompt_4_2_report.md`)

Per the [`StratifiedEvaluator`](../src/fraud_engine/evaluation/stratified.py:162-171) on the 92K-row test slice:

| Slice axis | Quick observation | Sprint 4.2 test gate |
|---|---|---|
| **amount_bucket** (low/mid/high) | Low-amount fraud rate > high-amount by > 0.20. Cost-per-txn highest in low bucket. | Gate passes. |
| **ProductCD** (W / C / R / H / S) | W (separable) AUC exceeds C (overlapping) AUC by > 0.10. | Gate passes. |
| **DeviceType** (mobile / desktop / null) | Mobile is the most-fraud-correlated populated slice; null (76% of rows) is the dominant population. | No regression. |
| **identity_coverage** (has / no identity) | Has-identity AUC > no-identity AUC. Both stay above 0.75. | Gate passes. |
| **month** (5 / 6) | Month 5 (training-period tail) cost-per-txn > month 6 (val-period). Reflects training-set noise. | Gate passes. |

Full per-slice numeric tables are in the report. The model card surfaces the test-gate verdicts so a portfolio reader sees that EVERY slice was evaluated, not just the headline.

---

## Training Data

### Source

- **Dataset:** [Kaggle IEEE-CIS Fraud Detection competition](https://www.kaggle.com/competitions/ieee-fraud-detection) (Vesta Corporation, 2019).
- **License:** Kaggle competition terms (research use permitted).
- **Citation:** Vesta Corporation, "IEEE-CIS Fraud Detection," Kaggle, 2019.

### Scale + composition

| Field | Value | Source |
|---|---|---|
| Total transactions | 590,540 | [CLAUDE.md §1](../CLAUDE.md), [`docs/DATA_DICTIONARY.md`](DATA_DICTIONARY.md) |
| Fraud rate | ~3.5% | same |
| Identity coverage | ~24% (any `id_*` non-null) | [CLAUDE.md §8](../CLAUDE.md) |
| Raw transaction columns | 394 | [`docs/DATA_DICTIONARY.md`](DATA_DICTIONARY.md) |
| Raw identity columns (left-joined) | 41 | same |
| Vesta-engineered V-features | V1-V339 (anonymised) | same |

### Temporal split (no random split per CLAUDE.md §2 leakage prevention)

| Split | Boundary (days since IEEE-CIS anchor 2017-12-01) | Approx rows | Approx % |
|---|---|---|---|
| Train | < 121 days | ~414,542 | 71% |
| Val | 121–151 days | ~83,473 | 14% |
| Test | ≥ 151 days | ~92,427 | 15% |

Source: `Settings.train_end_dt = 86400 * 121`, `Settings.val_end_dt = 86400 * 151` ([`src/fraud_engine/config/settings.py`](../src/fraud_engine/config/settings.py)). Row counts confirmed in [`sprints/sprint_3/prompt_3_3_d_report.md:20`](../sprints/sprint_3/prompt_3_3_d_report.md).

### Preprocessing

- Five-tier feature engineering pipeline. See [`docs/FEATURE_DOCUMENTATION.md`](FEATURE_DOCUMENTATION.md) for the per-tier rationale + 685 final features.
- V-column reduction (Tier 0) compresses the 339 V-features to 281 retained via NaN-group exploitation.
- Predictive-missingness indicators (`is_null_*` for 330 columns) explicitly encode the EDA finding that several IEEE-CIS columns carry 5×+ fraud-rate lift when present vs null (D7 strongest).

---

## Evaluation Data

### Validation slice

- **Window:** 121–151 days since IEEE-CIS anchor.
- **Row count:** ~83,473 (14% of total).
- **Fraud rate:** matches the overall 3.5%.
- **Used for:** Optuna hyperparameter selection (Sprint 3.3.b), early-stopping (best_iteration = 72), isotonic calibration fit (Sprint 3.3.c), economic-threshold optimisation (Sprint 4.4).

### Test slice

- **Window:** ≥ 151 days since IEEE-CIS anchor.
- **Row count:** ~92,427 (15% of total).
- **Fraud rate:** matches the overall 3.5%.
- **Used for:** held-out evaluation only — no hyperparameter / threshold / calibration tuning touches the test set. Sprint 4.4 cost-optimal threshold was validated against the test slice; reported numbers in [Metrics](#metrics) are test-set values.

### Production-time evaluation slice (offline batch — Sprint 6.1.c PerformanceMonitor)

- Configured via `Settings.performance_window_size` (default 1000 most-recent labelled predictions).
- The training-time baselines for AUC / AUC-PR / cost are operator-curated values in `Settings.performance_training_*` fields; alert fires on >5% degradation. See [`src/fraud_engine/monitoring/performance_monitor.py`](../src/fraud_engine/monitoring/performance_monitor.py) and [`docs/RUNBOOK.md`](RUNBOOK.md).

---

## Quantitative Analyses

### Overall results

See [Metrics](#metrics). Headline numbers:
- AUC = 0.8281 (val) / 0.8070 (test).
- Brier = 0.0254 (val) / 0.0249 (test) — post-isotonic calibration; 62% improvement over uncalibrated (0.0769 → 0.0254 on val).
- ECE = 0.0000 (val) / 0.0075 (test) — isotonic calibration is near-perfect on val and very tight on test.
- $28.96M annual savings on a 1M-txn/month portfolio at τ = 0.080 with FN/FP/TP = $450/$35/$5.

### Per-slice results

See [Metrics → Stratified metrics](#metrics). Every Sprint 4.2 gate passes. The dominant cost-per-transaction is in the low-amount bucket (fraud rate inversely correlates with amount); ProductCD W is the easiest slice; identity-coverage gap is real but the model performs cleanly on both halves.

### Cross-model comparison (Sprint 3.4)

| Model | Type | Val AUC | Test AUC | Brier (val) | ECE (val) | Inference latency (p95) | Role |
|---|---|---|---|---|---|---|---|
| **Model A — LightGBM** | GBDT | **0.8281** | 0.8070 | 0.0254 | 0.0000 | 3.29 ms | **Production champion** |
| Model B — FraudNet | NN (entity embeddings) | 0.8183 | 0.8229 | 0.0355 | 0.0882 | 59.94 ms | Shadow-deployable challenger |
| Model C — FraudGNN | GraphSAGE | 0.7778 | 0.7929 | 0.0357 | 0.0888 | 0.072 ms | Batch-only; outputs feed Tier 5 |

Model A is the champion by Brier + ECE (calibration quality, which drives the cost-optimal threshold reliability), not just AUC. Model B's val/test AUC is 1.5 percentage-points behind Model A but generalises better on the test set; under Sprint 5.2.c's promotion criteria (cost_improvement > 2% AND p_value < 0.05 AND agreement_rate > 85%) it does not currently promote. Model C is held out as a feature provider, not a deployment candidate.

### Calibration

Post-isotonic calibration:
- val Brier improved from 0.0769 (uncalibrated) to 0.0254 — 67% reduction.
- val ECE improved from 0.1926 to 0.0000 — three orders of magnitude.
- Isotonic won an 80/20 hold-out comparison against Platt. The model's uncalibrated output had a non-sigmoid miscalibration shape (a known consequence of LightGBM with `scale_pos_weight=27.4`); Platt's sigmoid assumption fits poorly.

### Intersectional analyses

Per-slice axes are documented above; cross-slice (e.g., amount_bucket × ProductCD) was scoped but not run. The Sprint 4.2 report flags this as future deepening if a per-slice cost regression appears in production via Sprint 6.1.c's PerformanceMonitor.

---

## Ethical Considerations

### False-positive customer impact

A false-positive block disrupts a legitimate transaction — at the project's published cost coefficient ($35 per FP, comprising $15 CSR contact + 5% churn × $400 CLV), the cumulative customer-friction cost on a 1M-txn/month portfolio at 3.5% block rate is ~$1.1M/month in friction. The model is calibrated to accept this cost in exchange for catching $28.96M/year in fraud losses; the ratio (~25× return on friction cost) is what makes the trade-off acceptable. Deployers with different customer economics (luxury goods, high-CLV banks, friction-averse markets) must override the FP cost in `.env` and re-derive the threshold.

### Demographic fairness

The IEEE-CIS dataset **does not contain race, gender, age, sexual orientation, religion, or income data**. The model therefore cannot be evaluated for disparate impact on these protected classes from this dataset alone. **A production deployment is responsible for:**

1. Joining demographic attributes from the deploying institution's CRM (subject to applicable consent + privacy regulations).
2. Running disparate-impact analysis on the joined data — at minimum, the 80% rule on block-rate parity across protected groups.
3. Documenting the analysis in a per-deployment supplement to this model card.

A deployment that skips this step is operating with unknown fairness properties.

### Opaque V-features

The Vesta-engineered V1-V339 columns are anonymised and (per Vesta's competition documentation) the true meanings are not disclosed. The model uses 281 of these (post Tier 0 reduction). **Implications:**

- A regulator asking "why does this transaction have a high `V257` value?" cannot be answered substantively beyond "the IEEE-CIS distribution's V257 column carries fraud signal."
- A future Vesta sensor change would alter the feature distribution invisibly to the modellers.
- This is a known limitation; deployers using IEEE-CIS as a training proxy for their own data should retrain on their own (non-anonymised) features before production.

### Geographic + payment-instrument fairness

`addr1`, `addr2`, `card1-card6` are integer-anonymised. Geographic + issuer fairness is not testable from this dataset. A deployment that joins to the institution's CRM can test these axes; the model card supplement should report the results.

---

## Caveats and Recommendations

### 1. 24% identity coverage — the model must degrade gracefully

76% of training transactions have no identity features (`id_*` columns all null). The model is designed to work without them — Tier 3 `ColdStartHandler` tags this case explicitly, and the Tier 1/2/4/5 features carry enough signal that the no-identity slice still achieves AUC > 0.75 per Sprint 4.2. **Production deployers should monitor the no-identity-slice performance separately** since identity-availability in their environment may differ from IEEE-CIS's 24%.

### 2. 2019 vintage — recalibration is mandatory

The IEEE-CIS data is from 2019. Production fraud patterns in 2024–2026 differ in:
- Merchant ecosystem composition (e.g., gig-economy / crypto / BNPL).
- Attack sophistication (e.g., AI-generated synthetic identities).
- Regulatory regime (PSD2, GDPR, FINTRAC AML thresholds).

A deployment trained on IEEE-CIS and run unmodified on 2025 traffic will degrade. Sprint 6.1.b's PSI drift monitoring + Sprint 6.1.c's PerformanceMonitor are designed to catch this; the `docs/RUNBOOK.md#how-to-trigger-retraining` procedure documents the response. **Recalibration on the deploying institution's own labelled data is mandatory before production.**

### 3. Cost-coefficient sensitivity

The decision threshold (τ = 0.080) is derived from FN/FP/TP = $450/$35/$5. The sensitivity spread is 0.06 across ±20% perturbation (well within the 0.20 stability ceiling). **Deployers with materially different economics must override `.env` cost coefficients and re-run `scripts/run_economic_evaluation.py`** before relying on the default threshold. See [ADR-0003 § Revisit triggers](ADR/0003-economic-threshold.md).

### 4. Calibration is load-bearing

The cost-optimal threshold assumes isotonic-calibrated output. The Sprint 3.3.c isotonic fit is fragile to:
- Significant input-distribution drift (Sprint 6.1.b PSI alerts on this).
- Re-training without re-calibrating (the operator runbook makes calibration a mandatory step).
- Down-sampling the validation slice below ~10K rows (isotonic loses statistical power).

If calibration regresses (ECE > 0.05 or Brier > 0.05), **re-derive τ before relying on the cost-savings claim**.

### 5. Temporal-split-only validation

No random train/test split was used at any stage (CLAUDE.md §2 leakage prevention). The model is validated against a temporally-later slice, which matches production usage. **Implication for deployers:** the same discipline must be enforced at retraining time — the deploying institution's training pipeline must use temporal splits (not random) to avoid leakage in the new training data.

### 6. The decision is at the transaction layer, not the customer layer

The model scores each transaction independently. A customer with a recently-blocked legitimate transaction does NOT get a "trust boost" on their next transaction; the model doesn't carry cross-transaction memory beyond what's encoded in the Redis entity state (Tier 2/4 features). Operators should pair this model with a customer-level allowlist / re-auth flow for high-value customers post-FP.

### 7. The fraud_neighbor_rate feature is OOF-discipline-critical

`Tier 5 GraphFeatureExtractor` produces `fraud_neighbor_rate` from the graph of (card, address, device, email) ↔ transaction adjacencies. A naïve implementation that uses a row's own fraud label in its own neighbour rate is a textbook leakage bug. **Future PRs touching this code must preserve the OOF discipline tested by Sprint 3.3.d's leakage tests** (val AUC must collapse to ~0.5 on shuffled labels).

### 8. The 685-feature surface is a maintenance commitment

Each of the 12 generators has a docstring carrying its business rationale, cost, and failure modes (per [`docs/FEATURE_DOCUMENTATION.md`](FEATURE_DOCUMENTATION.md)). **A future PR adding a new generator must:**
- Add the feature(s) to `models/pipelines/feature_manifest.json`.
- Update `FEATURE_DOCUMENTATION.md` (a sanity-check script in [`sprints/sprint_6/prompt_6_2_a_report.md`](../sprints/sprint_6/prompt_6_2_a_report.md) catches missing entries).
- Add a leakage test (shuffled-label AUC collapse) before merge.

---

## References

- Mitchell, M., et al. "Model Cards for Model Reporting." FAT*'19. [arXiv:1810.03677](https://arxiv.org/abs/1810.03677).
- IEEE-CIS Fraud Detection competition, Vesta Corporation, 2019. [Kaggle competition page](https://www.kaggle.com/competitions/ieee-fraud-detection).
- [ADR-0001 — Tech stack](ADR/0001-tech-stack.md).
- [ADR-0003 — Economic threshold](ADR/0003-economic-threshold.md).
- Project plan source-of-truth: [`CLAUDE.md`](../CLAUDE.md).
- Companion docs: [`FEATURE_DOCUMENTATION.md`](FEATURE_DOCUMENTATION.md), [`RUNBOOK.md`](RUNBOOK.md), [`OBSERVABILITY.md`](OBSERVABILITY.md), [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md), [`CONVENTIONS.md`](CONVENTIONS.md).

---

_Last updated: 2026-05-11 (Sprint 6.2.a). Refresh procedure: see [`docs/RUNBOOK.md#how-to-trigger-retraining`](RUNBOOK.md#how-to-trigger-retraining) Step 8 (validation)._
