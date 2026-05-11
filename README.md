# Fraud Detection Engine

> Real-time fraud detection system on the [IEEE-CIS dataset](https://www.kaggle.com/competitions/ieee-fraud-detection), built as a production engagement for a Canadian / LATAM fintech. LightGBM champion + FraudNet shadow + FraudGNN batch features, calibrated, threshold-tuned, served behind a FastAPI app with Redis + Postgres + Prometheus + Grafana, with a full monitoring stack and an operator runbook. **Portfolio target audience:** senior data scientist / ML engineer hiring committees at Wealthsimple, Mercury, RBC, Nubank.

## Headline metrics

| Metric | Value | Source |
|---|---|---|
| **Validation AUC** (LightGBM Model A, post-isotonic calibration) | **0.8281** | [`models/sprint3/lightgbm_model_manifest.json`](models/sprint3/lightgbm_model_manifest.json), [`prompt_3_3_d_report.md`](sprints/sprint_3/prompt_3_3_d_report.md) |
| **Brier score** (val, post-isotonic) | 0.0254 (67% reduction from uncalibrated 0.0769) | [`prompt_3_3_d_report.md`](sprints/sprint_3/prompt_3_3_d_report.md) |
| **ECE** (val, post-isotonic) | 0.0000 | same |
| **P95 inference latency** (live `/predict`, no shadow) | **70.98 ms** under a 100 ms SLO budget | [`prompt_5_1_f_report.md`](sprints/sprint_5/prompt_5_1_f_report.md) |
| **Cost-optimal decision threshold τ** | **0.0800** (analytical Bayes limit τ* ≈ 0.0729, within one grid step) | [ADR-0003](docs/ADR/0003-economic-threshold.md), [`prompt_4_4_report.md`](sprints/sprint_4/prompt_4_4_report.md) |
| **Annual savings** at τ=0.080 | **$28.96M** on a 1M-transactions / month portfolio (58× over the spec floor of $500K) | [CLAUDE.md §13](CLAUDE.md), [`prompt_4_4_report.md`](sprints/sprint_4/prompt_4_4_report.md) |
| Tests in CI | **819 unit + 22 integration** all passing | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |

## Business value

**The business pays in dollars, not F1.** A missed fraud (false negative) costs ~$450 — $150 chargeback + $25 fee + $75 investigation + $50 scheme penalty + $150 reputation/regulatory amortised. A blocked legitimate transaction (false positive) costs ~$35 — $15 customer-service contact + 5% churn × $400 customer lifetime value. A true positive (analyst review) costs $5.

```
total_cost = FN × $450 + FP × $35 + TP × $5
           + TN × $0
```

The cost-optimal decision threshold is the one that minimises **this** loss surface, not F1 or AUC. Sprint 4's [`EconomicCostModel.optimize_threshold`](src/fraud_engine/evaluation/economic.py) sweeps `np.linspace(0.01, 0.99, 99)`, computes total cost at each τ on the 92K-row test slice, and picks the minimum. The empirical winner (τ = 0.0800) matches the analytical Bayes-decision limit `τ* = fp_cost / (fp_cost + fraud_cost − tp_cost) = 35 / 480 ≈ 0.0729` within one grid step.

**At τ = 0.080 on a 1M-txn/month portfolio, the model saves $28.96M / year vs. a no-model baseline** — 58× over the spec's $500K floor. The threshold is robust: a ±20% perturbation of any cost coefficient moves τ by ≤ 0.06 (well under the 0.20 stability ceiling per [CLAUDE.md §8](CLAUDE.md)).

The full derivation + sensitivity analysis lives in [ADR-0003](docs/ADR/0003-economic-threshold.md) and [`reports/economic_evaluation.md`](reports/economic_evaluation.md).

## Architecture

The system serves a calibrated fraud probability + SHAP-derived top reasons in <100 ms P95 via this runtime topology:

```mermaid
flowchart LR
    Client[HTTP Client] -->|POST /predict| API[FastAPI<br/>fraud_engine.api.main]
    API --> FS[FeatureService]
    FS -->|Tier 1 inline| FS
    FS -->|MGET Tier 2-5| Redis[(Redis<br/>online feature store)]
    FS --> PG[(Postgres<br/>audit + batch)]
    API --> Inf[InferenceService]
    Inf --> JL[(lightgbm_model.joblib<br/>+ calibrator.joblib)]
    API --> SHAP[ShapExplainer]
    API -.->|fire-and-forget| Shadow[ShadowService<br/>FraudNet challenger]
    API -.->|fire-and-forget| PL[PredictionLogger]
    PL --> PG
    API --> Metrics[/metrics endpoint]
    Metrics --> Prom[Prometheus<br/>30d retention]
    Prom --> Graf[Grafana<br/>7-panel dashboard]
```

This is one of four diagrams in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), which also has the training pipeline, the per-request inference flow, and the dev vs. prod-like compose topologies. The architectural decisions behind the topology are recorded in six ADRs ([0001-tech-stack](docs/ADR/0001-tech-stack.md), [0002-temporal-split](docs/ADR/0002-temporal-split.md), [0003-economic-threshold](docs/ADR/0003-economic-threshold.md), [0004-shadow-mode](docs/ADR/0004-shadow-mode.md), [0005-lightgbm-as-production](docs/ADR/0005-lightgbm-as-production.md), [0006-graph-features-batch](docs/ADR/0006-graph-features-batch.md)).

## Quick start

Requires Python 3.11 + [uv](https://docs.astral.sh/uv/) + Docker (for Redis + Postgres + MLflow + Prometheus + Grafana via the dev compose).

```bash
# 1. Install dependencies from the locked manifest.
uv sync --all-extras

# 2. Seed local config + install pre-commit hooks.
cp .env.example .env
make install

# 3. Bring up the dev stack (Postgres + Redis + Prometheus + Grafana + MLflow).
docker compose -f docker-compose.dev.yml up -d

# 4. Run the API (uvicorn on host, scraped by the in-container Prometheus).
make serve  # waits on http://localhost:8000

# 5. Drive the 5-second demo (one obvious-fraud + one clearly-legit transaction).
uv run python scripts/demo_prediction.py
```

The demo prints score + decision + top-5 SHAP reasons for each transaction. Expected wall-clock: ~2 seconds total across both calls.

Day-to-day commands:

```bash
make format            # ruff format (auto-fix)
make lint              # ruff check (CI gate)
make typecheck         # mypy --strict on src/
make test-fast         # pytest tests/unit (no coverage; ~2 min)
make test              # full unit + integration + lineage suite
make test-lineage      # schema + temporal-integrity contracts only
make notebooks         # rebuild + execute committable notebooks in place
```

CI runs the same gate on every push + PR — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Key results

### Champion (Model A — LightGBM, served on `/predict`)

| Metric | Validation | Test |
|---|---|---|
| ROC AUC | 0.8281 | 0.8070 |
| PR AUC (avg precision) | ~ 0.55 | 0.515 |
| Brier (post-isotonic) | **0.0254** | 0.0249 |
| ECE (post-isotonic) | **0.0000** | 0.0075 |
| Log loss (post-isotonic) | 0.1090 | 0.1118 |
| Per-row inference latency (P95, isolated) | 3.29 ms | — |

Sources: [`prompt_3_3_d_report.md`](sprints/sprint_3/prompt_3_3_d_report.md) lines 41–49.

### Cross-model comparison (Sprint 3.4)

| Model | Role | Val AUC | Test AUC | Brier (val) | Latency p95 |
|---|---|---|---|---|---|
| **Model A — LightGBM** | **Production champion** | **0.8281** | 0.8070 | **0.0254** | **3.29 ms** |
| Model B — FraudNet | Shadow challenger | 0.8183 | 0.8229 | 0.0355 | 59.94 ms |
| Model C — FraudGNN | Batch-only feature provider | 0.7778 | 0.7929 | 0.0357 | 0.072 ms |

Model A wins on calibration + latency + SHAP-compatibility; FraudNet edges Model A on test AUC by 0.016 but loses on everything else. Sprint 5.2.c's promotion criteria (cost_improvement > 2% AND p_value < 0.05 AND agreement_rate > 85%) gate any swap; see [ADR-0005](docs/ADR/0005-lightgbm-as-production.md).

### Stratified evaluation (Sprint 4.2)

All five Sprint 4.2 test gates pass on the 92K-row test slice — `amount_bucket`, `ProductCD`, `DeviceType`, `identity_coverage`, `month`. Headline observations: low-amount fraud rate exceeds high-amount by > 0.20; ProductCD W AUC > C AUC by > 0.10; has-identity AUC > no-identity AUC; both stay above 0.75. Full per-slice tables in [`prompt_4_2_report.md`](sprints/sprint_4/prompt_4_2_report.md).

### Production behaviour (Sprint 5)

- **Latency p95:** 70.98 ms / 100 ms SLO (29% headroom).
- **Degraded-mode resilience:** API returns 200 with `degraded_mode=true` when Redis or Postgres is unreachable; only Tier-1 features serve, model falls back to population defaults.
- **Shadow mode:** FraudNet runs fire-and-forget on every `/predict` when `Settings.shadow_enabled=True`. Circuit breaker (5 failures → OPEN, 30s cooldown, 2× backoff to 300s cap) isolates challenger failures.
- **Audit log:** every prediction lands in Postgres `predictions` table via async fire-and-forget `PredictionLogger`.

### Monitoring (Sprint 6)

- 13 Prometheus metrics on `/metrics` (4 latency histograms + 1 score histogram + 3 counters + 3 gauges + 2 retrofit Counters for offline drift / shadow disagreement).
- Offline `DriftMonitor` (PSI on 685 features against a frozen training baseline) writes JSONL alerts to `logs/drift/{run_id}/`.
- Offline `PerformanceMonitor` (rolling AUC / AUC-PR / cost vs training baseline) writes JSONL alerts to `logs/performance/{run_id}/`.
- 7-panel Grafana dashboard + 5 named alert rules (`HighLatency`, `BlockRateSpike`, `FeatureDrift`, `ShadowDisagreement`, `ApiErrorRate`) with 30-day Prometheus retention.
- Operator runbook with ordered remediation per alert + model rollback + retraining procedures: [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Trade-offs

The project's design constraints (latency, interpretability, observability, reproducibility) drove explicit trade-offs:

| Choice | Won | Lost |
|---|---|---|
| **LightGBM over deep NN** ([ADR-0005](docs/ADR/0005-lightgbm-as-production.md)) | ~30× faster inference; SHAP-native; better calibration | ~1-2 AUC points on test; harder to capture non-tabular structure (compensated by Tier-5 graph features fed in batch) |
| **Cost-optimal threshold over F1** ([ADR-0003](docs/ADR/0003-economic-threshold.md)) | Aligned with the business loss surface; $28.96M / year savings | Threshold depends on cost coefficients staying in their ±20% stability band; recalibration needed on cost shifts |
| **Temporal split, no random** ([ADR-0002](docs/ADR/0002-temporal-split.md)) | Production-correct evaluation; no time-leakage | Smaller val/test slices than K-fold; one-shot hyperparameter selection |
| **Isotonic calibration over Platt** ([`prompt_3_3_d_report.md`](sprints/sprint_3/prompt_3_3_d_report.md)) | 67% Brier reduction; ECE → 0; non-parametric handles LightGBM's miscalibration shape | Requires ≥ 10K val rows for statistical power; more fragile than Platt under tiny-sample regimes |
| **Fire-and-forget shadow** ([ADR-0004](docs/ADR/0004-shadow-mode.md)) | Zero added latency on champion path; 3-state breaker isolates challenger failures | Per-request log volume doubles when enabled; shutdown drain has 5s timeout |
| **Graph features batch-only** ([ADR-0006](docs/ADR/0006-graph-features-batch.md)) | Inference latency contribution = 0 ms (vs. 5-30s for live pagerank); no graph DB dependency | Graph staleness between retraining cycles; drift monitor catches population shifts |
| **Mandatory SHAP top reasons** | Analyst-actionable predictions; regulator-friendly | Forces TreeExplainer-compatible models (no transformer / GNN on `/predict` path) |
| **Pull-from-existing-reports docs over auto-gen** | Editorial control; reviewer-facing prose | Operator must update MODEL_CARD after each retrain (manual step in RUNBOOK) |

## Limitations

The model and its evaluation envelope have known limitations that any production deployment must address:

1. **24% identity coverage.** 76% of IEEE-CIS training transactions have no identity (`id_*`) features. The model is designed to work without them — Tier 3 `ColdStartHandler` tags this case explicitly, and the no-identity slice still achieves AUC > 0.75 per Sprint 4.2. **Production deployers should monitor the no-identity-slice performance separately** since identity availability in their environment may differ.
2. **2019 vintage.** The IEEE-CIS data is from 2019. Production fraud patterns in 2024-2026 differ in merchant ecosystem (gig-economy, crypto, BNPL), attack sophistication (AI-generated synthetic identities), and regulatory regime (PSD2, GDPR, FINTRAC AML). **Recalibration on the deploying institution's own labelled data is mandatory before production.**
3. **Opaque V-features.** The Vesta-engineered V1-V339 columns are anonymised; the true meanings are not disclosed by Vesta. The model uses 281 of these (post Tier-0 NaN-group reduction). Regulator interpretability of these features is limited.
4. **No demographic data in IEEE-CIS.** Race, gender, age, sexual orientation, religion, geographic-postcode are not in the dataset. The model cannot be evaluated for disparate impact on protected classes from this dataset alone; production deployment is responsible for joining demographic attributes from the deploying institution's CRM and running disparate-impact analysis (e.g., 80% rule on block-rate parity).
5. **Cost-coefficient sensitivity.** The decision threshold τ = 0.080 is derived from FN/FP/TP = $450/$35/$5. Deployers with materially different economics (luxury goods, high-CLV banks, friction-averse markets) must override `.env` cost coefficients and re-run `scripts/run_economic_evaluation.py`.
6. **Calibration is load-bearing.** The cost-optimal threshold assumes the isotonic-calibrated output. If calibration regresses (ECE > 0.05 or Brier > 0.05 — monitored by Sprint 6.1.b PSI drift + Sprint 6.1.c PerformanceMonitor), **re-derive τ before relying on the cost-savings claim**.
7. **Per-transaction decisioning, not customer-layer.** The model scores each transaction independently. A customer with a recently-blocked legitimate transaction does NOT get a "trust boost" on their next transaction. Operators should pair this model with a customer-level allowlist / re-auth flow for high-value customers post-FP.

Full limitations + recommendations are in the model card's [Caveats section](docs/MODEL_CARD.md#caveats-and-recommendations).

## Repository structure

```
fraud-detection-engine/
├── .github/workflows/          # CI configuration (quality-gate)
├── configs/                    # YAML configs + alerts + prometheus + grafana
│   ├── alerts/                 # alert_rules.yml (5 rules)
│   ├── grafana/                # dashboards.yml + fraud_dashboard.json
│   └── prometheus/             # prometheus.yml (scrape + rule_files)
├── data/                       # gitignored — raw, interim, processed parquet
├── docs/                       # operator + portfolio docs
│   ├── ADR/                    # 6 architecture decision records
│   ├── MODEL_CARD.md           # Mitchell et al. 2018 format
│   ├── FEATURE_DOCUMENTATION.md
│   ├── ARCHITECTURE.md         # 4 mermaid diagrams + component reference
│   ├── RUNBOOK.md              # per-alert remediation + rollback + retraining
│   ├── INTERVIEW_GUIDE.md      # portfolio-interview prep
│   └── OBSERVABILITY.md        # logging discipline + jq recipes
├── logs/                       # gitignored — structured JSON + per-run dirs
├── models/                     # gitignored — joblibs + manifests + baselines
├── notebooks/                  # numbered — EDA + observability demo (commit WITH outputs)
├── scripts/                    # CLI entry points (download, build_features, train, demo, ...)
├── src/fraud_engine/           # library source
│   ├── api/                    # FastAPI, FeatureService, InferenceService, ShapExplainer, ShadowService
│   ├── config/                 # Pydantic Settings
│   ├── data/                   # Loader, cleaner, temporal_split, lineage
│   ├── features/               # Tier 1-5 generators + v_reduction
│   ├── models/                 # LightGBM, FraudNet, FraudGNN, baseline
│   ├── evaluation/             # Economic cost, stratified, calibration, shadow_compare
│   ├── monitoring/             # Prometheus metrics, drift, performance
│   ├── schemas/                # Pandera schemas
│   └── utils/                  # Logging, tracing, seeding, MLflow, metrics
├── sprints/                    # per-prompt completion reports (Sprints 0-6)
└── tests/                      # unit, integration, lineage
```

Full operating procedure for each sprint + the agent / human collaboration model lives in [CLAUDE.md](CLAUDE.md).

## Documentation

| Audience | Doc | What it answers |
|---|---|---|
| **Reviewers** | [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | What does this model do? How well? What are the ethical considerations? |
| Reviewers | [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md) | 30-second pitch + 5 hardest Q&A + what makes this different |
| Reviewers | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 4 mermaid diagrams + 25-component reference table |
| **Operators** | [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Per-alert remediation + model rollback + retraining |
| Operators | [`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md) | Structured-logging discipline + `jq` recipes |
| **Engineers** | [`docs/FEATURE_DOCUMENTATION.md`](docs/FEATURE_DOCUMENTATION.md) | 685 features × 12 generators × 5 tiers, per-generator rationale |
| Engineers | [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) | Raw IEEE-CIS columns |
| Engineers | [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) | Coding standards (mirror of CLAUDE.md) |
| Engineers | [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | Branching + PR conventions (one-prompt-per-PR squash-merge) |
| Engineers | [`docs/ADR/`](docs/ADR/) | 6 architecture decision records (0001-0006) |
| **Deep dive** | [`sprints/`](sprints/) | Per-prompt completion reports across Sprints 0-6 |
| **Agent** | [CLAUDE.md](CLAUDE.md) | Source-of-truth instructions for every Claude Code session in this repo |

---

_Last updated: 2026-05-11 (Sprint 6.2.c, pre-v1.0.0)._
