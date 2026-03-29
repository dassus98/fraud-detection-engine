# Real-Time Fraud Detection Engine

## Executive Summary

Card-not-present fraud costs payment networks 3–5% of annual revenue. Blocking it too aggressively is almost as expensive: a falsely blocked transaction carries a 5% churn risk on a customer worth ~$2,500 in lifetime value, plus a $25 support call. A naive model optimising for accuracy treats these outcomes as equivalent — this one does not.

This project builds a production-grade fraud detection system trained on the IEEE-CIS Vesta dataset (590k transactions, 3.5% fraud rate). The central design choice is an **economic cost function** that converts every model decision into a dollar figure, then finds the probability threshold that minimises expected cost rather than maximising AUC. The result is a deployable FastAPI service that scores a transaction in under 50ms, returns a BLOCK/ALLOW decision with the threshold used, and surfaces the top three SHAP feature contributions explaining why.

---

## Architecture

```
raw CSV
  └─► src/train.py          — temporal split, fit pipeline, train LightGBM, evaluate threshold
        ├─► FraudPipeline   — VFeatureCleaner → feature selection → categorical encoding
        └─► FraudModel      — LightGBM wrapper; stores optimal_threshold after evaluate()

models/
  ├── pipeline.pkl           — fitted transformer (feature_names_ baked in)
  └── fraud_model.pkl        — trained booster + optimal_threshold

src/api/main.py              — FastAPI service
  ├── /health                — liveness check; confirms model, SHAP, Redis status
  └── /predict               — per-transaction scoring
        ├── Redis lookup     — per-card aggregates (txn_count, mean_amount, last_txn_time)
        ├── pipeline.transform()
        ├── model.predict()
        └── SHAP TreeExplainer → top-3 feature contributions

src/evaluation/
  ├── shadow_mode.py         — batch simulation on validation set; economic report
  └── explainability.py      — full SHAP summary + bar plots → reports/
```

The pipeline and model are trained once, serialised to `models/`, and loaded at API startup. Redis is optional — the API degrades gracefully when it is unavailable, scoring on model features alone.

---

## Key Design Decisions

### 1. Economic cost function over accuracy

Standard fraud models are tuned on AUC or F1. Both treat false negatives and false positives as symmetric, which they are not:

| Outcome | Cost | Rationale |
|---|---|---|
| False negative (fraud missed) | $525 | $500 average fraud loss + $25 chargeback fee |
| False positive (legitimate blocked) | $150 | 5% churn × $2,500 LTV + $25 support call |

`find_optimal_threshold()` scans 0.01–0.99 and returns the threshold minimising `FN × $525 + FP × $150` per transaction. On this dataset the optimal threshold is **0.16**, well below the conventional 0.50 — reflecting that the asymmetry makes it cheaper to cast a wider net.

### 2. Temporal validation split

The dataset's `TransactionDT` column is a time offset. Rows are sorted by it before splitting 80/20. This mirrors production: the model is always trained on the past and evaluated on the future. Random splits leak future patterns into training and overstate generalisation.

### 3. V-feature pruning via correlation

The dataset contains 339 anonymised Vesta device/browser features (V1–V339) with extreme collinearity. `VFeatureCleaner` computes a pairwise correlation matrix and drops any feature whose maximum correlation with another feature in the upper triangle exceeds 0.90. This reduces 339 features to **179**, cutting noise and training time without measurable AUC loss.

### 4. Inference-time feature alignment

API requests only carry named transaction fields — V-features from device fingerprinting are absent. The `FraudPipeline` stores `feature_names_` (all 214 columns in training order) and calls `reindex()` in `transform()`, filling absent columns with NaN. LightGBM handles NaN natively via its missing-value splits, so the model applies learned thresholds correctly even when device features are missing.

### 5. Redis feature store

`RedisClient` maintains per-card running aggregates (`txn_count`, `amount_sum`, `last_txn_time`) as hash keys. At prediction time the API merges these into the feature row before pipeline transform, enabling real-time velocity features without a database round-trip. All Redis calls degrade to empty dicts on connection failure — the prediction path never blocks on cache availability.

---

## Results

Evaluated on 118,108 held-out transactions (temporally later 20% of the dataset).

| Metric | Value |
|---|---|
| ROC-AUC | **0.9126** |
| PR-AUC | **0.5582** |
| Optimal threshold | **0.16** |
| Cost per transaction (with model) | **$10.73** |

**Shadow mode simulation** (full validation set, threshold = 0.16):

| | |
|---|---|
| Fraud caught | $290,637 across 2,149 transactions |
| Fraud missed | $319,298 across 1,915 transactions |
| Fraud catch rate (by dollars) | 47.7% |
| Legitimate transactions blocked | 1,743 (1.53% false positive rate) |
| Baseline cost (no model) | $2,133,600 |
| Model total cost | $1,266,825 |
| **Net revenue saved** | **$866,775** on a single validation window |

The 47.7% dollar catch rate understates the model's impact: the remaining 52.3% of missed fraud dollars tends to cluster in high-value transactions where the model has lower confidence, exactly the cases where human review would be warranted. Precision (0.55) and recall (0.53) are deliberately balanced by the cost function rather than pushed toward one extreme.

---

## How to Run

**Requirements:** Python 3.10+, 8 GB RAM recommended for full training, Redis (optional).

### Setup

```bash
make setup          # creates venv and installs requirements.txt
```

### Train

```bash
make train          # runs src/train.py — loads data, fits pipeline, trains LightGBM,
                    # computes optimal threshold, saves models/fraud_model.pkl
                    # and models/pipeline.pkl
```

Full training takes 5–20 minutes depending on hardware. Hyperparameters are loaded from `models/best_params.json` if present (produced by `python -m src.tune`); otherwise sensible defaults are used.

### Serve

```bash
make serve          # starts FastAPI on http://localhost:8000
```

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 149.99, "ProductCD": "W", "card1": 4921, "card4": "visa"}'

# Health check
curl http://localhost:8000/health
```

The response includes `fraud_probability`, `decision` (BLOCK/ALLOW), `threshold_used`, and `top_shap_features` (top 3 contributors with signed values).

### Evaluate

```bash
python -m src.evaluation.shadow_mode      # writes reports/shadow_mode_results.txt
python -m src.evaluation.explainability   # writes reports/shap_summary.png
                                          # and reports/feature_rankings.csv
```

### Docker

```bash
make up             # docker-compose: API + Redis
```
