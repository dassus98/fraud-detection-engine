# Feature Documentation

> Companion to [`docs/DATA_DICTIONARY.md`](DATA_DICTIONARY.md). This doc covers the **engineered features** the LightGBM champion (Model A) consumes; the data dictionary covers the **raw IEEE-CIS columns** they're derived from.

## Overview

The feature pipeline produces **685 engineered features** from the raw IEEE-CIS columns via **12 generator classes** organised into **5 logical tiers** (plus one V-column reduction stage that runs before Tier 1).

The full machine-readable manifest lives at [`models/pipelines/feature_manifest.json`](../models/pipelines/feature_manifest.json) — one record per feature with `name`, `dtype`, `generator`, `rationale`. This document is the human-readable view of that manifest, structured tier-by-tier with the per-generator business rationale + cost + failure modes that aren't in the JSON.

### Pipeline at a glance

```
Raw IEEE-CIS (590,540 transactions × 433 raw columns)
        │
        ▼
[Tier 0]  V-column reduction (NanGroupReducer)        — 281 features
        │
        ▼
[Tier 1]  Basic transformations                       — 346 features
        │   AmountTransformer, TimeFeatureGenerator,
        │   EmailDomainExtractor, MissingIndicatorGenerator
        ▼
[Tier 2]  Aggregations                                — 20 features
        │   VelocityCounter, HistoricalStats, TargetEncoder
        ▼
[Tier 3]  Behavioral                                  — 6 features
        │   BehavioralDeviation, ColdStartHandler
        ▼
[Tier 4]  Temporal decay                              — 24 features
        │   ExponentialDecayVelocity
        ▼
[Tier 5]  Graph                                       — 8 features
        │   GraphFeatureExtractor
        ▼
LightGBM (685 features → calibrated probability → decision @ τ=0.080)
```

### Feature-count summary

| Tier | Generator | Count | % of 685 |
|---|---|---|---|
| 0 | `NanGroupReducer` | 281 | 41.0% |
| 1 | `MissingIndicatorGenerator` | 330 | 48.2% |
| 1 | `EmailDomainExtractor` | 8 | 1.2% |
| 1 | `TimeFeatureGenerator` | 6 | 0.9% |
| 1 | `AmountTransformer` | 2 | 0.3% |
| 2 | `VelocityCounter` | 12 | 1.8% |
| 2 | `HistoricalStats` | 5 | 0.7% |
| 2 | `TargetEncoder` | 3 | 0.4% |
| 3 | `BehavioralDeviation` | 5 | 0.7% |
| 3 | `ColdStartHandler` | 1 | 0.1% |
| 4 | `ExponentialDecayVelocity` | 24 | 3.5% |
| 5 | `GraphFeatureExtractor` | 8 | 1.2% |
| **Total** | **12 generators** | **685** | **100%** |

89% of the surface (Tier 0 + `MissingIndicatorGenerator`) is **predictive-missingness boilerplate** — null indicators on sparse columns + V-column NaN-group reductions. The remaining 11% carries the substantive behavioural / velocity / graph signal.

## How to read this document

Each tier section follows the same shape:

- **Purpose** — one paragraph on what the tier exists for.
- **Latency contribution** — per-request cost share from the Sprint 5.1.f P95 latency budget (100 ms total, observed 70.98 ms; see [`sprints/sprint_5/prompt_5_1_f_report.md`](../sprints/sprint_5/prompt_5_1_f_report.md)).
- **Failure modes** — common ways the tier can degrade or produce surprises.

Per-generator subsections list **count**, **sample feature names** (first 5–10), **business rationale** (verbatim from the manifest), **cost** (compute / memory / latency), and **failure modes** (drawn from the generator's class docstring).

The full 685-row feature listing is in [Appendix A](#appendix-a--full-feature-listing-685-rows). Tier-level failure-mode quick reference is in [Appendix B](#appendix-b--failure-modes-by-tier).

---

## Tier 0 — V-column reduction

- **Purpose:** the IEEE-CIS dataset's V1-V339 columns (339 anonymised Vesta-engineered features) share NaN patterns by upstream sensor and are highly correlated within those groups. This pre-tier reduces the V-space to a subset of representative columns before downstream feature engineering can use them. Reduces input cardinality from 339 raw V-columns to 281 retained features.
- **Latency contribution:** ≈ 0 ms at inference (the reduction is offline-fit; runtime applies the persisted column list).
- **Failure modes:** an upstream Vesta sensor change that alters the NaN-group structure invalidates the reduction's group membership; the model would silently use stale group representatives until the next refit.

### Generator: `NanGroupReducer` (`src/fraud_engine/features/v_reduction.py`)

- **Count:** 281 features
- **Sample features:** `V257`, `V244`, `V246`, `V233`, `V232`, `V231`, `V217`, `V258`, `V228`, `V219` (…and 271 more)
- **Business rationale:**

  > V-feature dimensionality reduction by NaN-group exploitation. IEEE-CIS V columns share NaN patterns by upstream sensor; within a group columns are highly correlated. Reduction keeps the most target-correlated column (correlation mode) or runs PCA to a variance threshold (PCA mode). Pre-reducing wide V-spaces lets Sprint 3's LightGBM concentrate on signal-bearing columns rather than redundant siblings.

- **Cost:** offline-fit; at inference, the persisted column list is a dict lookup. Memory: 281 × 4 bytes (float column tags) ≈ 1 KB.
- **Failure modes:** the group representatives are chosen by training-time correlation with `isFraud`. If the underlying V-feature population shifts (Vesta upstream change), the correlation-based pick is stale until refit. Tracked via Sprint 6.1.b PSI drift detection on each retained V-column.

---

## Tier 1 — Basic transformations

- **Purpose:** stateless per-row feature engineering — log transforms, calendar features, email domain parsing, predictive missingness. These features carry no entity history and are computed inline from a single transaction row. They are the "Tier-1 baseline" that the API's degraded-mode path falls back to when Redis or Postgres is unreachable (see Sprint 5.1.c `FeatureService`).
- **Latency contribution:** < 5 ms total at inference (all inline on a single-row DataFrame).
- **Failure modes:**
  - The source row is missing a required raw column → the generator raises (data contract violation; should be caught upstream by Pandera schema validation).
  - The source column has an unexpected dtype (e.g., `TransactionDT` as string instead of int seconds) → silent NaN propagation downstream.
  - Email domain parser receives a malformed email → returns "unknown" provider/TLD; not an error path.

### Generator: `AmountTransformer` (`src/fraud_engine/features/tier1_basic.py:104`)

- **Count:** 2 features
- **Sample features:** `log_amount`, `amount_decile`
- **Business rationale:**

  > Monotone log-amount captures the heavy right tail of transaction values; amount_decile discretises the same signal so tree models can split at the regime boundaries the EDA flagged (mid-to-large buckets carry the most fraud). Sprint 4's cost-curve analysis keys off `amount_decile`.

- **Cost:** trivial — one `np.log1p` + one `np.digitize` per row. < 0.01 ms.
- **Failure modes:** negative or zero `TransactionAmt` would produce NaN log values; the production data has no such rows (validated by Pandera schema), but a synthetic test fixture could trip this.

### Generator: `TimeFeatureGenerator` (`src/fraud_engine/features/tier1_basic.py:242`)

- **Count:** 6 features
- **Sample features:** `hour_of_day`, `day_of_week`, `is_weekend`, `is_business_hours`, `hour_sin`, `hour_cos`
- **Business rationale:**

  > Diurnal and weekday signals surface as fraud-rate hot-spots in the EDA's Section B.4 by-hour plot and Section B.6 dow × hour heatmap. Cyclical (sin / cos) encoding lets tree splits handle the 23 → 0 wrap-around; `is_business_hours` tags the daytime trough where fraud rates sit below the overall 3.5%.

- **Cost:** trivial — datetime extraction + two `np.sin`/`np.cos` calls. < 0.01 ms.
- **Failure modes:** `TransactionDT` is integer-seconds since the anonymised IEEE-CIS anchor (2017-12-01 UTC by community convention). If the anchor is mis-set in `Settings.transaction_dt_anchor_iso`, all time features shift; the cyclical encoding makes the shift invisible to downstream features but produces wrong calendar attributions.

### Generator: `EmailDomainExtractor` (`src/fraud_engine/features/tier1_basic.py:381`)

- **Count:** 8 features
- **Sample features:** `P_emaildomain_provider`, `P_emaildomain_tld`, `P_emaildomain_is_free`, `P_emaildomain_is_disposable`, `R_emaildomain_provider`, `R_emaildomain_tld`, `R_emaildomain_is_free`, `R_emaildomain_is_disposable`
- **Business rationale:**

  > Email-domain features encode the EDA's Section B.7 finding that certain free / disposable providers carry 10–30× the baseline fraud rate. The provider / TLD split feeds Sprint 2's later target-encoding generators; the free / disposable flags give a single-bit signal Sprint 3's baseline can use directly.

- **Cost:** ≈ 0.1 ms per row (dict lookups + regex on the email string).
- **Failure modes:**
  - Provider list (`configs/email_providers.yaml`) is exhaustive only for the IEEE-CIS distribution; a 2024-era email like `proton.me` would map to "unknown" until the config is extended.
  - Both `P_emaildomain` and `R_emaildomain` are NULL in ~76% of rows; the generator emits "missing" placeholder values.

### Generator: `MissingIndicatorGenerator` (`src/fraud_engine/features/tier1_basic.py:546`)

- **Count:** 330 features
- **Sample features:** `is_null_D10`, `is_null_D11`, `is_null_D12`, `is_null_D13`, `is_null_D14`, `is_null_D15`, `is_null_D2`, `is_null_D3`, `is_null_D4`, `is_null_D5` (…and 320 more covering most numeric columns in the IEEE-CIS train table)
- **Business rationale:**

  > EDA Section C.4 showed several columns carry a 5×+ fraud-rate lift when present vs null (D7 strongest). Explicit `is_null_*` features let Sprint 3's baseline exploit that predictive-missingness signal without re-discovering it from the underlying column's NaN pattern at every split.

- **Cost:** ≈ 1 ms per row (330 `.isna()` checks, vectorised).
- **Failure modes:**
  - The 330 columns covered are fixed at training time; new IEEE-CIS-shaped columns added downstream don't gain an indicator until the manifest is regenerated.
  - The signal is correlated across columns (e.g., D2 / D3 / D4 share the same NaN pattern in 80%+ of rows); LightGBM handles the correlation gracefully but a linear model would not.

---

## Tier 2 — Aggregations

- **Purpose:** per-entity historical aggregates that need a window of prior events to compute. These are the "online feature store" features served from Redis (Sprint 5.1.b): the FastAPI request looks up the entity's pre-computed rolling stats by key.
- **Latency contribution:** ≈ 35–55 ms at inference dominated by the Redis `MGET` round-trip (4 entity types × 3 features each ≈ 12 keys per request).
- **Failure modes:**
  - Redis unreachable → FeatureService flips `degraded_mode=True`; Tier-2 features fall back to population defaults from `configs/feature_defaults.yaml`. Inference still returns 200, with `degraded_mode=true` in the response payload.
  - Entity is new (never written to Redis) → population defaults used; the `ColdStartHandler` (Tier 3) tags this case explicitly.
  - Entity feature TTL expired in Redis (per `configs/redis_feature_store.yaml`) → same as new-entity case.

### Generator: `VelocityCounter` (`src/fraud_engine/features/tier2_aggregations.py:160`)

- **Count:** 12 features
- **Sample features:** `card1_velocity_1h`, `card1_velocity_24h`, `card1_velocity_7d`, `addr1_velocity_1h`, `addr1_velocity_24h`, `addr1_velocity_7d`, `DeviceInfo_velocity_1h`, `DeviceInfo_velocity_24h`, `DeviceInfo_velocity_7d`, `P_emaildomain_velocity_1h`, `P_emaildomain_velocity_24h`, `P_emaildomain_velocity_7d`
- **Business rationale:**

  > Per-entity transaction counts over fixed lookback windows. Velocity is the canonical fraud signal — a card running N transactions in the past hour is far more likely to be fraudulent than one running 1, irrespective of amount or hour. Strictly-past semantics avoid look-ahead leakage; tied timestamps do not see each other.

- **Cost:** ≈ 15 ms at inference (Redis `MGET` across the 12 keys for the 4 entity types).
- **Failure modes:**
  - Fixed-window cliff: a card with 9 transactions exactly 60 minutes ago has `velocity_1h = 0`; the same card with 9 transactions 59 minutes ago has `velocity_1h = 9`. The Tier-4 `ExponentialDecayVelocity` smooths this.
  - Tied-timestamp behaviour: ties don't see each other (strict-past semantics). At high RPS for the same entity, this leaves some signal on the floor; not a production concern at the project's RPS today.

### Generator: `HistoricalStats` (`src/fraud_engine/features/tier2_aggregations.py:358`)

- **Count:** 5 features
- **Sample features:** `card1_amt_mean_30d`, `card1_amt_std_30d`, `card1_amt_max_30d`, `addr1_amt_mean_30d`, `addr1_amt_std_30d`
- **Business rationale:**

  > Per-entity rolling mean / std / max of the amount column over fixed lookback windows. Captures expected-spending shape — a card whose 30-day mean is $40 suddenly seeing $2000 is suspicious; the same card with a 30-day mean of $1500 is not. Strict-past semantics avoid look-ahead leakage.

- **Cost:** ≈ 5 ms (3 Redis keys per entity × 2 entity types).
- **Failure modes:**
  - Same fixed-window cliff as `VelocityCounter`; 31-day-ago activity drops out entirely.
  - First-30-days entities get `std_30d = 0` until they have at least 2 transactions; downstream models treat this as "no variance yet" which is the right semantic.

### Generator: `TargetEncoder` (`src/fraud_engine/features/tier2_aggregations.py:619`)

- **Count:** 3 features
- **Sample features:** `card4_target_enc`, `addr1_target_enc`, `P_emaildomain_target_enc`
- **Business rationale:**

  > Out-of-fold (OOF) target encoding for high-cardinality categoricals (card4, addr1, P_emaildomain). Each training row's encoded value derives from a fold that does NOT contain the row itself — the only correct defence against self-leakage. Val / test use a full-train encoder. Smoothing toward the global rate via (sum + α × global_rate) / (count + α) protects low-cardinality categories from over-confident estimates.

- **Cost:** ≈ 3 ms (3 dict lookups; encoder pre-loaded into memory at startup).
- **Failure modes:**
  - **Target leakage is the dominant risk** — handled by 5-fold OOF discipline in training + full-train encoder at inference. Tested at every retraining run (Sprint 2 + 3 integration tests assert val AUC drops when OOF discipline is violated).
  - Unseen categorical values (new `card4` issuer that wasn't in training) → fall back to the global smoothed rate. Acceptable default.

---

## Tier 3 — Behavioral

- **Purpose:** *per-cardholder anomaly signals* — "is this transaction unusual for THIS specific cardholder?" — as opposed to Tier 2's "is this entity hot in aggregate?". Tier 3 features depend on the same Redis-served per-entity state as Tier 2 but compute deviation rather than raw counts.
- **Latency contribution:** ≈ 3–5 ms at inference (shares the Redis lookups with Tier 2; the deviation arithmetic is local).
- **Failure modes:**
  - Same degraded-mode fallback as Tier 2: Redis unreachable → population defaults → all deviation features near 0.
  - First-event entities: BehavioralDeviation's z-score features fall back to 0 (= "not anomalous"); `ColdStartHandler` tags the first-event case explicitly so the model can learn the "no history" pattern.

### Generator: `BehavioralDeviation` (`src/fraud_engine/features/tier3_behavioral.py:177`)

- **Count:** 5 features
- **Sample features:** `amt_zscore_vs_card1_history`, `time_since_last_txn_zscore`, `addr_change_flag`, `device_change_flag`, `hour_deviation`
- **Business rationale:**

  > Per-card1 behavioural deviations: amount z-score, inter-arrival z-score, addr change flag, device change flag, hour deviation. Captures 'is this transaction unusual for THIS specific cardholder?' — complementing Tier 2's 'is the entity hot?'. All strictly past-only; first-event rows fall back to 0 (not anomalous).

- **Cost:** ≈ 1 ms (z-score arithmetic on the same Redis-fetched state Tier 2 used).
- **Failure modes:**
  - First-event rows get `0` for all deviations (= "not anomalous"); the model should not learn "deviation == 0 implies legit" because cold-start = uncertain. `ColdStartHandler` provides the explicit tag.
  - `std` of 0 in the historical state → z-score is undefined; falls back to 0.
  - `addr_change_flag` / `device_change_flag` are binary; a legitimate cardholder travelling / replacing a phone will trip these. Sprint 4's economic-cost framework values an FP at $35; the model is calibrated to accept some false positives in exchange for higher recall.

### Generator: `ColdStartHandler` (`src/fraud_engine/features/tier3_behavioral.py:432`)

- **Count:** 1 feature
- **Sample features:** `is_coldstart_card1`
- **Business rationale:**

  > Cold-start indicator: per-entity flag set to 1 when fewer than `min_history` strictly-prior events exist for the entity. Lets downstream models distinguish 'no history' from 'history says this is normal'; pairs naturally with BehavioralDeviation's first-event 0-fallbacks.

- **Cost:** trivial — one comparison against `min_history`. < 0.01 ms.
- **Failure modes:** the `min_history` threshold is tuned in `configs/coldstart.yaml`; a too-low threshold makes the flag fire too rarely (no signal), too-high makes it fire on most-but-not-all real cardholders (false positives on the flag itself).

---

## Tier 4 — Temporal decay

- **Purpose:** **exponentially-decayed velocity** that smooths the fixed-window cliff of Tier 2. Where `card1_velocity_1h` drops to 0 the moment activity ages past 60 minutes, `card1_v_ewm_lambda_0.05` decays gracefully over hours. Multiple λ (decay rates) span hourly through daily timescales so LightGBM can pick the right rate per fraud pattern.
- **Latency contribution:** ≈ 5–10 ms at inference (Redis `MGET` across 24 keys for the 4 entity types × 3 lambdas × 2 variants).
- **Failure modes:**
  - **Fraud-weighted variant requires OOF discipline.** Training uses a two-pass `read → push` pattern: each row reads the prior EWM state (no leakage), then pushes its own label into the state for the next row. A bug that swaps the order leaks the current label.
  - Redis state lag: the EWM state at inference time is the most-recent state Sprint 5.1.b wrote. If the warmup script hasn't run since the last training cycle, the state is stale by hours/days.

### Generator: `ExponentialDecayVelocity` (`src/fraud_engine/features/tier4_decay.py:365`)

- **Count:** 24 features
- **Sample features:** `card1_v_ewm_lambda_0.05`, `card1_fraud_v_ewm_lambda_0.05`, `card1_v_ewm_lambda_0.1`, `card1_fraud_v_ewm_lambda_0.1`, `card1_v_ewm_lambda_0.5`, `card1_fraud_v_ewm_lambda_0.5`, `addr1_v_ewm_lambda_0.05`, `addr1_fraud_v_ewm_lambda_0.05`, `addr1_v_ewm_lambda_0.1`, `addr1_fraud_v_ewm_lambda_0.1` (…and 14 more covering DeviceInfo + P_emaildomain × 3 lambdas × 2 variants)
- **Business rationale:**

  > Per-(entity, λ) exponentially-decayed velocity (EWM). Smooths the window-boundary cliff that fixed-window velocity has by construction — recent activity decays gracefully rather than dropping off at an arbitrary boundary. Multiple lambdas span hourly-through-daily timescales so the model can pick the right decay rate per fraud pattern. The fraud-weighted variant tracks EWM of past confirmed fraud for this entity, OOF-safe by the read-before-push two-pass discipline.

- **Cost:** ≈ 5–10 ms at inference (Redis `MGET` + per-row EWM update is O(1)).
- **Failure modes:**
  - **OOF-discipline regression** — a future PR that refactors the two-pass training loop could re-introduce label leakage. Sprint 2's leakage-prevention tests assert val AUC drops to ~0.5 if OOF is violated.
  - The fraud-weighted variant requires labelled history. In production with chargeback lag (30–90 days), the fraud_v_ewm features are 1–3 months stale; this is correct (you genuinely don't know the label fresh) but means the feature lags the unweighted variant in regime shifts.

---

## Tier 5 — Graph

- **Purpose:** **shared-infrastructure structure** — fraud rings rotating cards across a small pool of compromised devices and addresses. Per-card aggregations cannot see this; only a graph view across entities can. These features are computed batch-offline (the graph is fit once per training cycle); at inference, the FastAPI app reads the pre-computed per-entity graph features from the model's snapshot.
- **Latency contribution:** ≈ 0 ms at inference (graph features are baked into the entity state; the lookup is part of Tier 2's Redis `MGET`).
- **Failure modes:**
  - Graph is stale between retraining runs. Sprint 6.1.b's drift monitoring on the graph features catches a population shift; the operator triggers retraining (see [`docs/RUNBOOK.md#how-to-trigger-retraining`](RUNBOOK.md#how-to-trigger-retraining)).
  - The graph is built from the training window only; rare entities only appearing in val/test get `connected_component_size=1` and `entity_degree_*=0` (singleton).

### Generator: `GraphFeatureExtractor` (`src/fraud_engine/features/tier5_graph.py:543`)

- **Count:** 8 features
- **Sample features:** `connected_component_size`, `entity_degree_card1`, `entity_degree_addr1`, `entity_degree_DeviceInfo`, `entity_degree_P_emaildomain`, `fraud_neighbor_rate`, `pagerank_score`, `clustering_coefficient`
- **Business rationale:**

  > Tier-5 graph-derived features expose shared-infrastructure structure that per-card aggregations cannot see — fraud rings rotating cards across a small pool of compromised devices and addresses. Five distinct features per transaction (connected component size, per-entity degree, OOF-safe fraud neighbour rate, pagerank, bipartite clustering) catch organised-fraud topology that velocity and behavioural-deviation features systematically miss.

- **Cost:** offline-fit (graph construction is ~5 minutes wall on the full train slice via `scripts/train_lightgbm.py` Step 3). At inference: 0 ms (looked up).
- **Failure modes:**
  - **`fraud_neighbor_rate` is the OOF-discipline hot-spot.** A naïve implementation that uses the same row's fraud label in its own neighbour rate is a textbook leakage bug. Sprint 3.3.d's tests assert this is OOF-safe.
  - Computing pagerank + clustering on the full transaction graph (~590K nodes) is memory-intensive (~4 GB RAM); a smaller hardware footprint requires graph sampling that loses long-tail community structure.

---

## Appendix A — full feature listing (685 rows)

The complete listing of all 685 features lives in [`models/pipelines/feature_manifest.json`](../models/pipelines/feature_manifest.json). A quick way to enumerate them:

```bash
# List all features grouped by generator:
jq -r '.features | group_by(.generator) | map({generator: .[0].generator, count: length, names: [.[].name]}) | .[] | "\(.generator) (\(.count)): \(.names | join(\", \"))"' \
  models/pipelines/feature_manifest.json
```

```bash
# Count by generator:
jq '.features | group_by(.generator) | map({generator: .[0].generator, count: length}) | sort_by(-.count)' \
  models/pipelines/feature_manifest.json
```

```bash
# List all sample names for a specific generator (e.g., MissingIndicatorGenerator):
jq -r '.features[] | select(.generator == "MissingIndicatorGenerator") | .name' \
  models/pipelines/feature_manifest.json | head -30
```

For governance / audit purposes, the manifest is the source of truth and this document is a derived view. Any discrepancy between the two should be resolved in favour of the manifest.

## Appendix B — failure modes by tier

| Tier | Generator | Most-common failure mode |
|---|---|---|
| 0 | NanGroupReducer | Vesta upstream V-sensor change invalidates the persisted group structure; refit-on-retrain catches it. |
| 1 | AmountTransformer | Trivial; only fails on impossible inputs (negative amount). |
| 1 | TimeFeatureGenerator | Anchor mismatch shifts all calendar features; transparent to downstream. |
| 1 | EmailDomainExtractor | Provider list is non-exhaustive for new domains; falls back to "unknown". |
| 1 | MissingIndicatorGenerator | Coverage frozen at training time; new IEEE-CIS-shaped columns need a manifest regen. |
| 2 | VelocityCounter | Fixed-window cliff (60 min vs 61 min); Tier-4 EWM smooths it. |
| 2 | HistoricalStats | First-30-days entities get std=0; downstream treats as "no variance". |
| 2 | TargetEncoder | OOF-discipline regression risks self-leakage; Sprint 2 tests assert val AUC collapses to ~0.5 if violated. |
| 3 | BehavioralDeviation | First-event rows fall back to 0; `ColdStartHandler` tags this. Legitimate behaviour changes (travel, new phone) trip the flags; Sprint 4 cost framework accepts this trade-off. |
| 3 | ColdStartHandler | Threshold tuning trade-off — too-low under-fires, too-high false-positives on the flag itself. |
| 4 | ExponentialDecayVelocity | OOF-discipline regression risks fraud-weighted self-leakage; tests catch it. Chargeback lag (30–90 days) means `fraud_v_ewm_*` features are weeks stale in production — correct by construction but means slower response to regime shifts. |
| 5 | GraphFeatureExtractor | Graph is stale between retraining cycles; drift monitor catches population shifts. `fraud_neighbor_rate` is the highest-risk OOF-discipline site in the pipeline; tests assert it. |

## Cross-references

- [`models/pipelines/feature_manifest.json`](../models/pipelines/feature_manifest.json) — machine-readable source of truth (685 features × 4 fields).
- [`docs/DATA_DICTIONARY.md`](DATA_DICTIONARY.md) — raw IEEE-CIS columns the generators derive from.
- [`docs/MODEL_CARD.md`](MODEL_CARD.md) — model metrics + intended use built on these features.
- [`src/fraud_engine/features/`](../src/fraud_engine/features/) — generator implementations with full per-class docstrings.
- [`src/fraud_engine/api/feature_service.py`](../src/fraud_engine/api/feature_service.py) — runtime composition (Tier-1 inline + Redis MGET for Tier 2/3/4/5).
- [`sprints/sprint_2/`](../sprints/sprint_2/) + [`sprints/sprint_3/`](../sprints/sprint_3/) — completion reports with per-tier integration tests.
- [`sprints/sprint_5/prompt_5_1_f_report.md`](../sprints/sprint_5/prompt_5_1_f_report.md) — P95 latency measurement (70.98 ms with all tiers active).
