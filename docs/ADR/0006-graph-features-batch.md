# ADR 0006: Graph features computed batch (not live per-request)

- **Status:** Accepted
- **Date:** 2026-05-02
- **Sprint:** 3 (prompt 3.3.d)

## Context

Tier 5 of the feature pipeline produces 8 graph-derived features per transaction: `connected_component_size`, `entity_degree_card1`, `entity_degree_addr1`, `entity_degree_DeviceInfo`, `entity_degree_P_emaildomain`, `fraud_neighbor_rate`, `pagerank_score`, `clustering_coefficient`. These features expose shared-infrastructure structure (fraud rings rotating cards across compromised devices/addresses) that per-entity aggregations cannot see — a real signal class missing from Tier 1–4.

Computing these features requires a bipartite graph of (card / addr / device / email) ↔ transaction adjacencies. The graph on the full IEEE-CIS training slice has ~590,540 transaction nodes + ~250K entity nodes ≈ 840K total nodes; ~2.4M edges; ~4 GB memory footprint in dense form, ~600 MB sparse (NetworkX + scipy.sparse).

Two architecturally distinct approaches:

1. **Live graph per request** — maintain a continuously-updated graph in memory (or in a graph database). At inference, run `pagerank` + clustering + neighbour-rate on the live graph. Adding the new transaction to the graph requires a write lock + neighbour propagation.
2. **Batch graph at training time** — fit the graph once per retraining cycle. Compute per-entity graph features offline. Persist these features into the per-entity online state (Redis). At inference, the FastAPI app reads pre-computed per-entity graph features from Redis (already part of the Tier-2/3/4 MGET round-trip; no extra latency).

The constraints making (1) impractical:

- **Memory** — 4 GB graph in every serving process; multiplies by worker count.
- **Latency** — `pagerank` is O(E × log V) per recompute; on 2.4M edges that's ~5-30 seconds. Incremental update is theoretically possible but unstable on sparse/streaming graphs. `clustering_coefficient` is O(V × max_degree²) — worse.
- **Write coherence** — concurrent `/predict` requests need a consistent view of the graph; a write lock blocks reads.
- **Operator complexity** — a graph DB (Neo4j / TigerGraph) is a third stateful service to maintain. The project's stack is intentionally minimal (PostgreSQL + Redis + LightGBM joblib).

The constraints making (2) attractive:

- **Inference cost is O(0 ms)** — the graph features are already in the per-entity Redis state; the Tier-2/3/4 MGET fetches them in the same round-trip.
- **Memory at inference time is zero** — the serving process never holds the graph in memory.
- **Staleness is bounded** — features are as fresh as the last retraining cycle. Sprint 6.1.b's PSI drift monitoring on these features (`fraud_engine_drift_alerts_total`) catches population shift; the operator triggers retraining per [`docs/RUNBOOK.md`](../RUNBOOK.md#how-to-trigger-retraining).
- **Operator simplicity** — no new service. The graph fit happens inside `scripts/train_lightgbm.py`'s feature pipeline step.

Alternatives considered (and rejected):

1. **Live graph in serving memory.** Memory + latency + write-coherence problems above.
2. **Live graph in Neo4j / graph DB.** Adds a third stateful service; staleness still exists (the graph is only as fresh as the most-recent write); per-query latency on `pagerank` is similar to in-process.
3. **Approximate online graph features** (e.g., min-hash sketches for neighbour-rate). Loses pagerank + clustering; the approximation error is hard to quantify against a moving distribution.
4. **Stream-process the graph** (Apache Flink / Kafka Streams). Heavyweight infrastructure overhead; per-query latency similar to (2); operationally complex.
5. **Batch with daily refresh.** Better staleness than per-retraining-cycle but requires nightly orchestration; same infrastructure as (2)+(4) just less of it.

## Decision

Tier 5 graph features are **computed once per training cycle, persisted into the per-entity Redis state, and read at inference as part of the Tier 2/3/4 MGET round-trip**:

1. **Training time** — [`scripts/train_lightgbm.py`](../../scripts/train_lightgbm.py) Step 3 builds the bipartite `TransactionEntityGraph` from the training-slice rows. [`src/fraud_engine/features/tier5_graph.py:GraphFeatureExtractor`](../../src/fraud_engine/features/tier5_graph.py) computes the 8 features per transaction.
2. **Persistence** — the per-entity graph features (e.g., `card1_degree`, `card1_clustering`) are written into the entity's Redis hash by `scripts/warmup_redis.py` (Sprint 5.1.g) as part of the entity's snapshot.
3. **Inference** — the FastAPI app's `FeatureService.get_features` (Sprint 5.1.c) reads these features in the same Redis MGET round-trip as Tier 2/3/4. Per-transaction features like `fraud_neighbor_rate` are reconstructed at inference from the entity-level features via a lightweight aggregation (no graph reconstruction).
4. **`fraud_neighbor_rate` OOF discipline** — at training time, each row's `fraud_neighbor_rate` is computed using a fold that excludes the row itself (5-fold OOF, mirroring `TargetEncoder` in Tier 2). At inference, the full-train-fit values are used.

| Aspect | Value |
|---|---|
| Graph type | Bipartite (transaction ↔ {card1, addr1, DeviceInfo, P_emaildomain}) |
| Backend | NetworkX + scipy.sparse |
| Features per transaction | 8 (component size, 4 degree-features, fraud-neighbor-rate, pagerank, clustering) |
| Fit cadence | Once per retraining cycle |
| Persistence | Redis (alongside Tier 2/3/4 entity state) |
| Inference latency contribution | 0 ms (within the Tier 2-4 MGET budget) |
| OOF discipline | 5-fold; `fraud_neighbor_rate` is the OOF-critical feature |
| Drift monitoring | Sprint 6.1.b PSI on each of the 8 features |

## Rationale

1. **Inference latency budget is non-negotiable.** Live graph computation on the production graph (840K nodes, 2.4M edges) is ~5-30 seconds per request — 50× to 300× the 100 ms P95 budget. Pre-computing is the only path that fits.
2. **Operator simplicity wins on tractable trade-offs.** The "graph DB" option adds a third stateful service for a feature that contributes <2% of total feature count (8 / 685) and an estimated 1-3 AUC points (Sprint 3.4 ablation). Not worth the operational cost.
3. **OOF discipline is preservable at training time.** `fraud_neighbor_rate` is the canonical leakage trap: a row's neighbour-rate must not include its own label. Sprint 3.3.d implements 5-fold OOF on this feature, mirroring Tier 2's `TargetEncoder` pattern. The leakage test asserts val AUC collapses to ~0.5 on shuffled labels — same defence as Tier 2.
4. **Staleness is bounded + monitored.** The graph is as fresh as the last retraining cycle; in production a retraining cadence of weekly to monthly is normal. Sprint 6.1.b's drift monitor catches population shifts on the graph features; operators retrain on the `FeatureDrift` alert.
5. **Singleton fallback for new entities.** Entities never seen during training get `connected_component_size=1`, `entity_degree_*=0`, `pagerank_score≈uniform`. This is the correct "no graph signal yet" semantic; the model has been trained on similar new-entity rows from the training slice and knows what to do.
6. **The same Redis MGET amortises the cost.** Tier 2-4 already round-trip to Redis for per-entity state. Adding 8 more keys to that MGET is ~0 latency overhead at the project's RPS.

## Consequences

- **`scripts/train_lightgbm.py` Step 3 runtime is ~5-10 minutes** for the graph fit + feature extraction on the full train slice. Acceptable for a step that runs once per retraining cycle; the alternative (live computation) is impossible.
- **The graph is stale between retraining cycles.** Operators can either trigger retraining on a `FeatureDrift` alert (the canonical response) OR run a partial refresh — re-fit only the graph step using the most-recent windows, persist new entity features, leave the LightGBM model untouched. The partial-refresh path is a Sprint 6.x candidate.
- **`warmup_redis.py` must run before serving** — the per-entity state needs to be in Redis. The dev-stack instructions in `docs/RUNBOOK.md` document the warmup step; the prod-like stack runs warmup during deploy.
- **New entities get singleton defaults.** A genuinely new card never seen in training will have low entity-degree + small connected-component. The model has been trained on similar new-entity rows from the training slice; performance on these is documented in Sprint 4.2's stratified evaluation (no-identity slice).
- **Drift in graph features triggers a full retrain.** Sprint 6.1.b's PSI on these features is part of the global drift counter; an alert here means either entity-population shift (new geos, new fraud ring) or upstream pipeline regression. The RUNBOOK procedure handles both.
- **`fraud_neighbor_rate` is the highest-leverage leakage trap in the pipeline.** Future PRs touching the graph feature code must include a leakage test (val AUC must collapse to ~0.5 on shuffled labels). Sprint 3.3.d's test is the canonical gate.
- **Inference reads 8 extra Redis keys per request** — already accounted for in the Tier 2/3/4 MGET round-trip; no measurable latency add.
- **FraudGNN (Model C) is repurposed as a feature provider, not a deployment candidate.** Its GraphSAGE embedding outputs feed `GraphFeatureExtractor` (some of the 8 features are GNN-derived). FraudGNN is trained alongside LightGBM during the retraining cycle; its outputs are baked into the LightGBM training data. See [ADR-0005](0005-lightgbm-as-production.md).

## Revisit triggers

- **Production fraud-ring patterns intensify** (organised crime with shared infrastructure dominates the fraud mix). Then live graph features start providing materially more signal than batch; consider a graph DB or stream-process backend.
- **Latency budget loosens** (e.g., the deploying institution accepts 200-500 ms P95). Live graph computation on a smaller graph (e.g., 50K-node recent-windows graph) becomes feasible.
- **A graph-native infrastructure exists at the deploying institution** (e.g., Neo4j cluster already in production for adjacent use cases). The "third stateful service" argument weakens; revisit.
- **Sprint 6.1.b drift alerts on graph features become a daily occurrence.** The retraining cycle is too slow for the fraud population's churn rate. Consider partial-refresh (graph features only) on a sub-weekly cadence.
- **An approximate-online-graph algorithm matures** to within ~1% of the offline-batch values (e.g., a streaming pagerank with bounded error). Trade-off equation changes.
- **`fraud_neighbor_rate` leakage regression in a future PR.** Sprint 3.3.d's leakage test catches it; the response is to revert the bad refactor, not to re-architect.

## References

- [`src/fraud_engine/features/tier5_graph.py`](../../src/fraud_engine/features/tier5_graph.py) — `GraphFeatureExtractor` + `TransactionEntityGraph`.
- [`scripts/train_lightgbm.py`](../../scripts/train_lightgbm.py) — training pipeline; graph fit + feature extraction is Step 3.
- [`scripts/warmup_redis.py`](../../scripts/warmup_redis.py) (Sprint 5.1.g) — persists per-entity features (incl. graph features) into Redis.
- [`src/fraud_engine/api/feature_service.py`](../../src/fraud_engine/api/feature_service.py) — inference-time MGET that reads graph features from Redis.
- [`sprints/sprint_3/prompt_3_3_d_report.md`](../../sprints/sprint_3/prompt_3_3_d_report.md) — Tier 5 implementation + leakage tests.
- [`sprints/sprint_3/prompt_3_4_b_report.md`](../../sprints/sprint_3/prompt_3_4_b_report.md) — FraudGNN (Model C) training; outputs feed Tier 5.
- [`docs/FEATURE_DOCUMENTATION.md#tier-5--graph`](../FEATURE_DOCUMENTATION.md#tier-5--graph) — feature-level docs for the 8 graph features.
- [`docs/MODEL_CARD.md#caveats-and-recommendations`](../MODEL_CARD.md#caveats-and-recommendations) — caveat §7 on `fraud_neighbor_rate` OOF discipline.
- [`docs/RUNBOOK.md#alert-featuredrift`](../RUNBOOK.md#alert-featuredrift) — alert operationalisation for graph-feature drift.
- Related: [ADR-0002 — Temporal split](0002-temporal-split.md) (defines the OOF-safe fold structure); [ADR-0005 — LightGBM as production](0005-lightgbm-as-production.md) (FraudGNN's role as feature provider, not champion).
