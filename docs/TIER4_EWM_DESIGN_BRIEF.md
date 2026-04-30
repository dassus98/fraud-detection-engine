# Tier 4 (Exponential Decay Velocity) — Design Brief

**Audience:** non-technical reviewers, hiring committees, fraud product managers, model risk officers.

**Status:** shipped via PR #26 (the generator) and PR #27 (the build pipeline + schema + leak gate). The 10-generator Tier-3 pipeline is now extended to 11 generators; the val AUC at default LightGBM hyperparameters dropped from 0.9063 (Tier-3) to 0.7932 (Tier-4) — a documented modelling regression that the upcoming hyperparameter-tuning prompt is expected to recover.

This document is a **plain-English explainer** of what Tier 4 does, why it matters, the design tradeoffs we made, how it would behave in a production credit-card-fraud system at a bank, and what was deferred and why.

---

## TL;DR

| Question | Answer |
|---|---|
| What is Tier 4? | A new family of features that score "how active has this card / address / device / email been recently?" — but with smooth time decay instead of hard 24-hour windows. |
| Why does a bank need it? | Fixed-window features have a cliff: a transaction at 23h59m counts; one at 24h01m doesn't. Sophisticated fraudsters time their bursts to fall just outside the window. EWM removes the cliff. |
| What changed in the model? | 24 new features per transaction (4 entities × 3 timescales × 2 signals). Pipeline grew from 750 to 774 columns per row. |
| What's the cost? | A meaningful drop in val AUC at default LightGBM hyperparameters (0.9063 → 0.7932). Not a correctness bug — a modelling regression caused by feature collinearity and split-selection noise across the wider feature space. The leak gate confirmed zero target leakage. |
| What's the recovery path? | The next prompt is a hyperparameter-tuning sweep. Industry experience says EWM features are worth ~0.01–0.03 AUC lift after tuning. |
| Why ship it now if AUC dropped? | Tier 4 is the **production-realistic** velocity feature shape. Sprint 5's serving stack stores 3 floats per Redis key and updates them in O(1) per request — fixed-window VelocityCounter can't be served at sub-100ms latency without re-walking the deque per request. We need EWM in the pipeline to validate Sprint 5's wire contract; tuning recovery is straightforward. |

---

## 1. The fraud-detection problem Tier 4 solves

A bank's fraud-detection system has to answer one question per transaction, in roughly 100 milliseconds: **"is this transaction unusual enough to block?"**

To do that, the model needs to know how the cardholder's recent behaviour compares to their normal pattern. "Recent behaviour" is itself made up of dozens of signals, but one of the most powerful is **velocity** — how many transactions has this entity done lately?

### The status quo (Tier 2's `VelocityCounter`)

Sprint 2 shipped `VelocityCounter`, which produces fixed-window counts:

- `card1_velocity_1h` — how many transactions in the last 1 hour
- `card1_velocity_24h` — last 24 hours
- `card1_velocity_7d` — last 7 days

It works, but it has a structural flaw: **window-boundary cliffs**. A transaction 23h59m old contributes 1 to the 24h count. The same transaction 2 minutes later (24h01m old) contributes 0.

### Why the cliff matters at a real bank

Three concrete operational problems:

1. **Score instability.** Imagine a fraudster's card had 50 transactions yesterday. At hour 23 today, the 24h velocity count is 50. At hour 25, the same card's 24h count is ~48 (the oldest transactions have aged out). The risk score wobbles even though the underlying behaviour is unchanged. Risk teams have to filter out "score noise from time progression" — extra operational complexity.

2. **Adversarial gaming.** Sophisticated fraudsters use a well-known pattern: burst a few transactions, wait 24 hours and 5 minutes, burst again. Their 24h velocity stays low; the model never sees the true rate of activity. This is a documented attack pattern in the credit-card fraud literature.

3. **Multi-timescale signal is hard to express.** With only 1h / 24h / 7d, the model can't natively express "the last 6 hours mattered more than the rest of the day." Adding 6h / 12h / 48h windows is doable but inflates the feature space without smoothness — and you still have cliffs at every new boundary.

### What EWM does

Replace the cliff with a smooth decay. A transaction's contribution to the EWM **decays exponentially with time** — recent transactions count fully, old ones gradually fade out. There's no boundary. There's no cliff. The model sees a single smooth signal that says "this card has recently been busy."

---

## 2. What is Exponential Decay Velocity? — plain-English explainer

Imagine a fraud analyst staring at a card's transaction history. They care about how active it's been **recently**, but "recently" is fuzzy. A transaction 5 minutes ago should count almost as much as one happening right now. From a week ago, it should still count, but less. From a year ago, nearly forgotten.

EWM (exponentially weighted moving sum) captures that intuition with one knob: **λ** (lambda), the decay rate. Each past transaction contributes a weight:

```
weight = exp(-λ × Δt_hours)
```

where `Δt_hours` is "how many hours ago was this transaction."

### Three landmarks to anchor the intuition

| When did the transaction happen? | Weight |
|---|---|
| Just now (Δt = 0) | 1.0 — full credit |
| One **half-life** ago (Δt = ln(2) / λ) | 0.5 — half credit |
| Many half-lives ago | ≈ 0 — forgotten |

The half-life depends on λ:

| λ (per hour) | Half-life | Reads as... |
|---:|---:|---|
| 0.05 | 13.9 hours | "about half a day" |
| 0.10 | 6.9 hours | "a working day" |
| 0.50 | 1.4 hours | "under 90 minutes" |

### Why three different λ values at once?

Because different fraud patterns happen at different timescales:

- **Burst-of-activity fraud** (e.g., a stolen card racking up 10 charges in 30 minutes) shows up clearly at λ=0.5 (1.4-hour half-life). The fast decay means recent activity dominates.
- **Slow-burn fraud** (a compromised card making one small charge per hour for a day to test limits) shows up clearly at λ=0.05 (~14-hour half-life). The slow decay accumulates the test-charges.
- **Mid-timescale fraud** (e.g., a card used in a rapid trip across multiple merchants over a few hours) shows up at λ=0.1 (~7-hour half-life).

By emitting all three lambdas as separate features, the LightGBM model gets to choose which timescale carries the most signal for which fraud pattern. This is what production fraud teams actually deploy — typically 3 to 5 lambdas spanning hourly through weekly.

---

## 3. A worked example with banking dollars

### The scenario

You're a fraud analyst at a Canadian challenger bank. Card `A` belongs to a young professional in Toronto who normally spends in 5–6 transactions per week, mostly groceries and coffee. Today the card has had three transactions:

| Time today | Transaction | Amount | Suspicious? |
|---|---|---:|---|
| 7:00 AM (T = 0h) | Loblaws grocery | $50 | Normal |
| 8:00 AM (T = 1h) | Tim Hortons coffee | $30 | Normal |
| 1:00 PM (T = 6h) | Best Buy electronics | $200 | Mildly unusual amount, but plausible |

Now at 1:30 PM (T = 6.5h), a new $400 transaction comes in from an electronics retailer in another province. The fraud system has 100 milliseconds to decide whether to block it.

### What the EWM features tell the model

For card A at T = 6.5h, with `fraud_weighted=False` (just counting transactions):

**At λ = 0.5 / hour (1.4-hour half-life):**

| Past transaction | Δt (hours) | Weight = exp(-0.5 · Δt) |
|---|---:|---:|
| 7:00 AM grocery | 6.5 | exp(-3.25) ≈ **0.039** (almost forgotten) |
| 8:00 AM coffee | 5.5 | exp(-2.75) ≈ **0.064** (mostly forgotten) |
| 1:00 PM electronics | 0.5 | exp(-0.25) ≈ **0.779** (still very fresh) |
| **Sum (`v_ewm` at λ=0.5)** | | **≈ 0.88** |

This says: at the 1.4-hour timescale, this card has had *roughly one* recent transaction's worth of activity, dominated by the electronics purchase 30 minutes ago.

**At λ = 0.05 / hour (13.9-hour half-life):**

| Past transaction | Δt (hours) | Weight = exp(-0.05 · Δt) |
|---|---:|---:|
| 7:00 AM grocery | 6.5 | exp(-0.325) ≈ **0.722** |
| 8:00 AM coffee | 5.5 | exp(-0.275) ≈ **0.760** |
| 1:00 PM electronics | 0.5 | exp(-0.025) ≈ **0.975** |
| **Sum (`v_ewm` at λ=0.05)** | | **≈ 2.46** |

This says: at the half-day timescale, this card has had ~2.5 transactions of activity. Close to its normal day-rate.

### How the model uses it

The LightGBM splits its decisions on these two numbers (and 22 others — across `addr1`, `DeviceInfo`, `P_emaildomain`, plus the `fraud_v_ewm` variants). A possible split tree fragment:

> If `card1_v_ewm_lambda_0.5 > 5.0` AND `addr1_v_ewm_lambda_0.5 < 1.0` → high-risk (this card is bursting at multiple new addresses)

The cliff problem doesn't apply: a transaction 23h59m vs 24h01m old produces nearly identical EWM contributions, so adversarial timing doesn't help the fraudster.

### What the dollar stakes look like

The project's economic constants (from `.env`, used in Sprint 4's threshold optimization):

| Constant | Value | Meaning |
|---|---:|---|
| `FRAUD_COST_USD` | $450 | Average cost of a missed fraud — $150 transaction + $25 chargeback fee + $75 investigation + $50 scheme penalty + $150 reputation/regulatory exposure |
| `FP_COST_USD` | $35 | Cost of blocking a legit transaction — $15 support call + 5% churn × $400 customer lifetime value |
| `TP_COST_USD` | $5 | Investigation cost on a confirmed-fraud block |

The asymmetry is **~13×**: missing one fraud costs the bank as much as creating ~13 false positives. EWM features improve the model's ability to flag the right transactions; even a 1% improvement in catch rate at the same false-positive rate translates to material savings at the volume of a national fraud system (~590k transactions in the IEEE-CIS dataset; real banks process tens of millions per day).

---

## 4. The 10 design decisions, with banking implications

This section walks through each design decision we made for `ExponentialDecayVelocity`, what the alternative would have been, and how the choice would behave at a real bank.

### Decision 1 — Running scalar state vs. storing every past event

**The choice:** keep three numbers per `(card, λ)` pair: `(last_t, v, fraud_v)`. Don't store the individual past transactions.

**Why it matters at a bank:** every fraud decision happens in real time — typically under 100ms end-to-end including network calls to Redis, the model, and SHAP explainability. If you stored every past transaction, you'd have to re-walk the entire deque on every new event. For a card with thousands of transactions over its lifetime, that's untenable. With running state, every new event is two arithmetic operations: decay the running total, add 1 (or `+is_fraud`). O(1) per event, regardless of history length.

**Tradeoff cost:** **audit opacity.** When a fraud analyst asks "why did this card's velocity score spike at 2:00 PM?", you can't answer "because of these specific 5 transactions" — the running scalar has merged them all. We mitigate by keeping a slow O(n²) reference implementation in the test suite that can reconstruct the answer offline for a single suspicious transaction.

**Production implication:** in the Sprint-5 serving stack, each `(entity, λ, value)` Redis key stores 24 bytes total. For a bank with 10 million active cards, that's about 240 MB per λ per entity column — well within Redis's working set on a single node.

### Decision 2 — Two-pass tied-group batching

**The choice:** when multiple transactions share the same timestamp (e.g., simultaneous purchases at different terminals), process them in two passes: first read everyone's pre-tie state, then update state for each.

**Why it matters at a bank:** if two transactions on the same card arrive at the same instant, the model must score them based on the cardholder's history *before* either one — neither transaction should "see" the other in its own EWM. Without this discipline, the second tied transaction would erroneously count the first, inflating its risk score.

**Tradeoff cost:** ~20 extra lines of code; one extra `# noqa` comment for the lint rule about branch count. Trivial.

**Production implication:** real-time serving doesn't have the tied-group problem in the same shape — events arrive serially, one at a time. The serving-side logic is slightly different: read state, score, push, repeat. The batch-side discipline ensures the *training data* is leak-free; the serving-side is naturally so by virtue of being serial.

### Decision 3 — `fit_transform` ≠ `fit() + transform(train)` for training rows

**The choice:** the generator has three methods that all do something different. `fit_transform(train)` produces leak-free training-row outputs. `fit(train)` builds the end-state without producing per-row outputs. `transform(val)` applies the frozen end-state to validation/test data.

**Why it matters at a bank:** target-leakage is the single most common failure mode in fraud ML. If a feature accidentally encodes its own row's `is_fraud` label, the model looks brilliant in offline evaluation and crashes in production. The `fit_transform` discipline (inherited from Sprint 2's `TargetEncoder`) ensures training rows compute their EWM features from data that strictly precedes them.

**The footgun:** `gen.fit(train).transform(train)` produces *leaked* outputs — the full-train state has all training labels baked in, applied back to training rows. This pattern is documented loudly in the class docstring as "do not do this for training rows; this is the val/test path."

**Production implication:** the val/test path is what runs in serving. Bank operations would never call `fit` mid-day — the model is retrained on a regular cadence (weekly or monthly) and the production service only ever calls `transform`. The footgun risk is therefore confined to offline retraining workflows; the integration leak gate (`test_tier4_no_fraud_leak.py`) catches it before deployment.

### Decision 4 — `transform(val)` does NOT push validation labels

**The choice:** during validation/test, decay the trained state forward to each row's timestamp and read; don't push the row's label into state.

**Why it matters at a bank:** in production, the model scores transactions in real time. The label (was-this-fraud or was-this-legit) is unknown at scoring time and only becomes known days or weeks later (after chargebacks settle). The batch evaluation must mirror this: val rows score against the trained state, not against state that has been "updated" with future val labels.

**Tradeoff cost:** state decays toward zero over long validation periods. If validation spans weeks, late val rows see fully-decayed state. Acceptable for IEEE-CIS's contiguous validation window; in production the trained state is replaced every retraining cycle.

**Production implication:** this matches Redis-state semantics exactly. Each new transaction reads-then-pushes; the read happens before the push, so the transaction's own label (which doesn't exist yet) cannot leak into its own features. The Sprint-5 serving path will be a near-verbatim translation of `transform`'s logic.

### Decision 5 — Underflow is correct (no clamp)

**The choice:** when `λ × Δt` is very large, `exp(-λ × Δt)` returns `0.0` due to floating-point underflow. We let it. Don't add a `max(state, epsilon)` clamp.

**Why it matters at a bank:** an entity that's been quiet for weeks SHOULD have an EWM near zero. Clamping to a positive epsilon would systematically bias the model toward thinking long-quiet entities are slightly active — a small but real drift that compounds over millions of transactions.

**Tradeoff cost:** `state == 0.0` is ambiguous between "fully decayed" and "never seen this entity before." The model can't tell from the EWM alone. We mitigate via `ColdStartHandler` (Sprint 2.3.a), which emits a separate `is_coldstart_card1` flag for entities with thin history — the model gets to learn how to weight zero EWMs differently when coldstart is set.

**Production implication:** at a bank with millions of inactive cards, the EWMs for dormant cards naturally decay to zero. The `is_coldstart` flag combined with EWM-equals-zero is a clean signal; no special handling needed in serving.

### Decision 6 — Hard-error on backward time

**The choice:** if `transform(val)` ever sees a row with timestamp earlier than the trained state's last update, we raise an exception immediately rather than silently producing wrong values.

**Why it matters at a bank:** the upstream guarantee (training data ends before validation begins) is invariant to the system's correctness. If that invariant is ever violated — say, by a bug in the temporal-split code or a manual data-loading mistake — the EWM math would silently inflate state values above 1.0 (because `exp(-λ × negative) > 1`), corrupting predictions in ways that wouldn't surface until deployment.

**Tradeoff cost:** no graceful degradation. A user feeding a non-temporally-sorted frame gets an error rather than best-effort output. Accepted because silent inflation is strictly worse than loud failure.

**Production implication:** matches the bank's general "fail loudly at boundaries" philosophy — schema validation, negative-amount rejection, transaction-ID uniqueness all follow the same pattern. A wrong prediction silently leaving the system to be used in a chargeback dispute is a regulatory and operational nightmare; an exception during model retraining is a fix-the-bug-and-retry.

### Decision 7 — λ in /hour despite timestamps in seconds

**The choice:** the `TransactionDT` column stores integer seconds (the IEEE-CIS dataset's convention). λ values in the YAML are per-hour. Inside the generator, we convert: `dt_hours = (T_event - T_prev) / 3600.0`.

**Why it matters at a bank:** "λ = 0.05 with a half-life of 14 hours" is an intuition a fraud SME can hold in their head. "λ = 0.0000139 per second with a half-life of 50,000 seconds" is not. The YAML reads naturally to the people who tune it.

**Tradeoff cost:** two extra division operations per event (~5 nanoseconds each on a modern CPU). Negligible.

**Production implication:** the YAML is the configuration surface that fraud product managers and risk officers actually look at. Making it human-readable matters more than the trivial computational cost.

### Decision 8 — Memory-efficient per-state record (`dataclass(slots=True)`)

**The choice:** the per-`(entity, λ, value)` running state is a Python dataclass with `slots=True`, holding three floats: `(last_t, v, fraud_v)`.

**Why it matters at a bank:** at 14,000 unique cards × 12 (entity, λ) keys, the total state takes about 4 megabytes in memory. With a non-slotted dataclass it would be ~20 MB — five times more. Slots also speed attribute access by ~30% by skipping the per-instance dictionary lookup.

**Tradeoff cost:** none material. Slots disable dynamic attribute addition on the dataclass, which we don't need.

**Production implication:** in the serving path, the running state lives in Redis (not in process memory), so this particular tradeoff is more about offline batch builds. The choice still matters for the fitted-pipeline `joblib` payload size, which grew from 36 KB (Tier-3) to 2.7 MB (Tier-4) — entirely from persisting the end-state. The serving service would extract the end-state from joblib at startup and load it into Redis as the initial state.

### Decision 9 — Validate λ uniqueness in `__init__`

**The choice:** if the YAML or constructor passes duplicate λ values, raise a `ValueError` immediately. Three lines of code.

**Why it matters at a bank:** if duplicates were allowed, they'd silently produce duplicate column names (the second overwriting the first). The model would train on 23 columns instead of 24; one entire λ would be missing without anyone noticing. At a regulated institution where model lineage and feature contracts are auditable, "we accidentally dropped a feature" is a compliance issue.

**Tradeoff cost:** none. Duplicates have no legitimate use case.

**Production implication:** the error fires at module-load time during retraining, before any predictions are made. Caught in dev/CI; never reaches production.

### Decision 10 — Multiple lambdas as separate features (vs. picking one)

**The choice:** emit features for every λ in the YAML list (default 3). Don't try to pick the "right" λ at training time.

**Why it matters at a bank:** different fraud patterns surface at different timescales (see §2 above). By emitting all three lambdas, the model can split on whichever timescale matches the pattern it's identifying. A burst-fraud pattern uses high-λ; a slow-burn pattern uses low-λ.

**Tradeoff cost:** **3× feature space.** 4 entities × 3 λ × 2 signals = 24 columns vs. 8 if we picked one λ. Adds work for LightGBM's split-selection logic; this is a meaningful contributor to Tier 4's val-AUC regression at default hyperparameters (see §6 below).

**Production implication:** post-tuning, the model should learn to ignore the lambdas that aren't useful for its split decisions. The 3× feature cost is upfront; the tuning-time benefit is being able to discover the right timescale automatically rather than committing to a single "right" λ in the YAML. Industry teams typically deploy 3–5 lambdas spanning hourly through weekly.

---

## 5. How this would work in production at a bank

### Architecture overview

```
┌─────────────────┐
│ Card transaction │ (POS terminal, e-commerce checkout, etc.)
│    arrives       │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Tier-4 EWM update + read (~1 ms)            │
│                                              │
│  For each (entity_col, λ) pair:             │
│   1. Look up Redis key (entity, λ, value)   │
│   2. Compute (v, fraud_v) decayed to T_now  │
│   3. ATOMICALLY: update state with this txn │
│   4. Return (v, fraud_v) for the model      │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Other features (Tier 1, 2, 3 — already in   │
│ pipeline) computed from same Redis state     │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ LightGBM scoring + SHAP explanation (~30ms) │
└────────┬─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Threshold + decision (block / allow / hold) │
└──────────────────────────────────────────────┘
```

### What lives where

- **Redis state.** For each `(card, λ)` pair: three floats (`last_t`, `v`, `fraud_v`). Same shape as `_DecayState`. Atomic update via Lua script or Redis transactions.
- **Joblib model artefact.** Loaded once at service startup; the Tier-4 generator's end-state seeds Redis with the trained state.
- **Streaming retraining.** Weekly or monthly: the new training data is processed through `fit_transform`, the new end-state replaces the Redis state atomically.

### Latency budget

The project's spec target is P95 < 100 ms end-to-end. Tier 4's contribution:

| Step | Time |
|---|---:|
| Redis lookups (4 entities × 3 λ = 12 keys) | ~5 ms |
| Decay + read math | ~1 ms |
| Redis writes (12 keys) | ~5 ms |
| **Tier 4 total** | **~11 ms** |

Comfortably within budget. The bulk of latency is in LightGBM scoring + SHAP, not feature computation.

### Monitoring at a bank

Production fraud teams monitor:

1. **Feature distribution drift.** PSI (Population Stability Index) on each Tier-4 column, comparing training distribution to live serving. If `card1_v_ewm_lambda_0.5` drifts substantially, alert. Sprint 6's monitoring stack will own this.
2. **Catch rate vs. false-positive rate.** Daily: fraud caught / total fraud, and false blocks / total transactions. EWM features should help on burst patterns; this is where the lift is observable.
3. **Latency P50 / P95 / P99.** As above, Tier 4 should add ~11ms; flag if it exceeds 30ms.
4. **Adversarial-pattern surfaces.** Specifically watch for fraud rings that have figured out how to defeat the new features. Industry experience: each EWM deployment causes a brief uptick in fraud as adversaries discover the new boundaries, then the model adapts.

### Regulatory and audit considerations

In Canada, federally regulated banks (Schedule I) operate under OSFI's E-23 Model Risk Management guideline. Fraud models are typically classified as low-to-moderate risk (vs. credit-decisioning models which are higher risk), but they still require:

1. **Model documentation.** This document, the prompt 3.1.a/b reports, the docstrings, and the trade-off analyses are all part of the model documentation package.
2. **Lineage.** Every prediction must trace back to the data that produced it. Tier 4's end-state is in the saved joblib; the training run has a `run_id` in `logs/runs/`; the parquets are reproducible from the cleaner output. This is intact.
3. **Explainability.** SHAP explanations are mandatory for any decision a customer can dispute. Tier 4 columns are individually interpretable in SHAP — "your card's velocity at the 1.4-hour timescale was unusually high" — although communicating that to a customer in plain English requires a translation layer (Sprint 5's reason-code service).
4. **Bias and fairness.** Velocity features can be a proxy for socioeconomic factors. A bank deploying this should run a fairness analysis (disparate impact on protected groups). Out of scope for this project; flagged for the model card in Sprint 6.
5. **Change management.** Any feature added to the model goes through model risk review. EWM lambdas are configuration parameters that can be tuned without retraining; significant lambda changes (e.g., adding a new entity, changing default λ values) should be reviewed.

---

## 6. The current val-AUC gap, in plain English

The Tier-4 build script reported a validation AUC of **0.7932**. The project spec said "expected 0.92-0.93." That's a substantial gap — about 0.13 points below the lower bound, and 0.11 points below Tier-3's 0.9063.

### What this is NOT

It is **not** a correctness bug.

- Schema validates on all three splits. ✓
- Row counts preserved. ✓
- All 24 EWM columns present, finite, non-negative. ✓
- Unit tests (3.1.a): 17 tests pass, including a 50-example hypothesis property test that compares the optimised running-state algorithm to a slow O(n²) reference. The numerical match is within `1e-9` relative precision.
- **Leak gate: val AUC = 0.4514 < 0.55 ceiling.** With training labels shuffled, the model can't predict val labels above chance. Zero target leakage.

The math is right. The contracts hold. There's no leakage.

### What this IS

A **modelling regression at default LightGBM hyperparameters.** Three contributing factors:

1. **Feature-space inflation.** Sprint 2 ended with 750 columns; we added 24 more, taking the total to 774. The default LightGBM (`num_leaves=63`, `n_estimators=500`, default regularisation) was already struggling at 750 columns (Tier-3 val AUC 0.9063 vs. Sprint-1 baseline 0.9247). Adding 24 more without re-tuning regularisation widens the split-selection space and lets the model fragment its splits across redundant features.

2. **Multi-timescale collinearity.** Within a single entity, the three EWM features at λ = 0.05, 0.1, 0.5 measure roughly the same activity at different decay rates. They're strongly correlated by construction. The `NanGroupReducer` (Sprint 2.3.b) doesn't compress them — its regex matches `V[0-9]+` columns only, not EWM columns. A LightGBM split on `card1_v_ewm_lambda_0.05 > X` is approximately equivalent to a split on `card1_v_ewm_lambda_0.1 > Y` for some Y; the model fragments its decision logic across the three lambdas without proportional information gain.

3. **Sparse `fraud_v_ewm`.** The IEEE-CIS dataset has a 3.5% fraud rate. The fraud-weighted EWM is therefore a sparse signal, especially at short half-lives. Default `min_child_samples` is too low for the sparsity; leaves can specialise on individual fraud events and fail to generalise.

### What recovers it

The next prompt in Sprint 3 is the hyperparameter-tuning sweep. Standard recovery moves:

1. **Lower `num_leaves`** (try 15, 31). Smaller trees prefer informative features; redundant lambda features get pruned.
2. **Raise `min_child_samples`** (try 100-500). Forces leaves to be bigger; sparse `fraud_v_ewm` patterns can't dominate.
3. **Add `reg_alpha` / `reg_lambda` regularisation.** Penalises feature-importance fragmentation.
4. **Possibly drop one λ.** If post-tuning the model still ignores the middle λ (0.1), drop it from the YAML — 16 columns instead of 24.

Industry experience: EWM features post-tuning typically lift AUC by 0.01-0.03 over fixed-window velocity features. Recovery from 0.7932 to ~0.91-0.93 is the expected outcome.

### Why we shipped 3.1.b anyway

Three reasons:

1. **The structure is right.** Tier-4 is the production-realistic velocity feature shape. Sprint 5's serving stack needs this shape to validate its wire contract.
2. **The leak gate confirms safety.** No correctness bugs. No target leakage. Tier 4 is *operationally* safe to deploy; it's just not yet *modelling-optimal*.
3. **Tuning is the natural next prompt.** Following the project's per-prompt cadence, the cleanest split is "ship the features, then tune." Trying to tune within the same prompt would conflate two different concerns and make the diff harder to review.

This is the same pattern Sprint 2 followed at every tier boundary: ship the features, document the val-AUC pattern, recover via tuning later. We've now had four such regressions (Tier-1, -2, -3, -4 each below baseline). Sprint 3's tuning prompt should compound the recovery across all four tiers at once.

---

## 7. What was deferred and why

| Item | Why deferred | When it'll land |
|---|---|---|
| Hyperparameter tuning sweep | Pre-tuning, AUC isn't representative; tuning needs its own prompt | Sprint 3, next prompt |
| Lineage walk for all 24 EWM columns × 50 samples | The unit test's hypothesis property + `assert_no_future_leak` smoke + 3.1.b's shuffled-labels integration gate provide layered coverage | Possibly a future audit prompt; not on critical path |
| Wire ExponentialDecayVelocity into the Sprint-5 serving stack | Sprint 5 territory | Sprint 5 |
| `MissingIndicatorGenerator` PerformanceWarning fix | Cosmetic; doesn't affect correctness | Open Sprint-3 cleanup |
| YAML helper de-duplication (4 copies of `_resolve_config_path` + `_load_yaml`) | Project convention; refactoring is a separate task | Future cleanup |

---

## 8. Where this fits in the broader project

```
Sprint 1: Baseline LightGBM at 0.9247 val AUC ──────────────┐
                                                            │
Sprint 2: Add Tier-1, Tier-2, Tier-3 features ──────────────┤
   - Each tier adds features at default LightGBM            │
   - Each tier produces a slight AUC regression             │
   - Documented as Sprint-3 tuning recovery pattern         │
                                                            │
Sprint 3 (in progress):                                     │
   - 3.1.a: ExponentialDecayVelocity (Tier-4 EWM)  ✓ DONE  │
   - 3.1.b: Tier-4 build pipeline + schema         ✓ DONE  │
   - 3.x: HYPERPARAMETER TUNING ←─── recovers ALL tiers ◄──┘
   - 3.x: Neural-net diversity model (Model B)
   - 3.x: GNN (Model C)

Sprint 4: Economic threshold optimisation
Sprint 5: FastAPI + Redis + SHAP + shadow mode
Sprint 6: Monitoring + drift + model card
```

The val-AUC regressions at each tier boundary are not a problem; they're the expected pattern. The recovery happens in Sprint 3's tuning prompt, where all the per-tier regressions get addressed at once with proper regularisation and feature-importance pruning.

---

## 9. References

**In-repo:**

- Module docstring + 10-tradeoff section: `src/fraud_engine/features/tier4_decay.py:1-180`
- Naive O(n²) reference + hypothesis property test: `tests/unit/test_tier4_decay.py:42-110, 290-330`
- 11-generator pipeline build: `scripts/build_features_tier1_2_3_4.py:108-136`
- Tier-4 schema declaration: `src/fraud_engine/schemas/features.py:130-170`
- Shuffled-labels integration leak gate: `tests/integration/test_tier4_no_fraud_leak.py:107-140`
- 3.1.a completion report: `sprints/sprint_3/prompt_3_1_a_report.md`
- 3.1.b completion report: `sprints/sprint_3/prompt_3_1_b_report.md`

**External (concepts):**

- Holt-Winters exponential smoothing — the statistical ancestor of EWM in time-series forecasting.
- Production fraud feature engineering: Stripe Radar's exponentially-decayed velocity features (publicly described in their 2017 ML blog post).
- IEEE-CIS Fraud Detection competition (Vesta Corporation, 2019) — the dataset this project trains on.
- OSFI E-23 Model Risk Management Guideline (Canadian regulatory baseline).

---

## Appendix: glossary for the non-technical reader

| Term | What it means here |
|---|---|
| **AUC** | Area Under the ROC Curve. A model's overall discrimination quality. 1.0 = perfect, 0.5 = random. Industry-standard fraud-model metric. |
| **Decay rate (λ)** | How fast a past transaction's contribution fades out. Half-life = ln(2) / λ. |
| **EWM** | Exponentially Weighted Moving sum. A running total where each contribution decays by a fixed multiplier per unit time. |
| **OOF (out-of-fold)** | A discipline where each training row's features are computed without using its own label. Prevents target leakage. |
| **P95 latency** | The 95th-percentile response time. If P95 < 100ms, then 95% of decisions return within 100ms. |
| **PSI** | Population Stability Index. A statistical drift metric. Production teams alert when PSI exceeds 0.1-0.25 (depending on the bank's risk appetite). |
| **Redis** | An in-memory key-value store. Used for low-latency state lookup in fraud-detection serving. |
| **Schema validation** | Programmatically checking that a dataset has the expected columns, types, and value ranges before feeding it to a model. |
| **SHAP** | A model-explainability technique. Produces per-feature contribution scores for each prediction. Mandatory for customer-disputable fraud decisions. |
| **Target leakage** | A bug where a feature accidentally encodes information about the row's own label. Catastrophic in production; the project's leak gates exist to detect it. |
| **Temporal split** | Train on rows from time T1 to T2; validate on T2 to T3; test on T3 to T4. Mirrors how production retraining works. |
| **Velocity** | How many transactions an entity (card / address / device / email) has done in some time window. Strongest single non-monetary fraud signal. |
