# ADR 0004: Shadow mode for champion-vs-challenger evaluation

- **Status:** Accepted
- **Date:** 2026-05-09
- **Sprint:** 5 (prompt 5.2.b)

## Context

The fraud-detection-engine carries two scoring models with comparable AUC: Model A (LightGBM, the production champion per [ADR-0005](0005-lightgbm-as-production.md)) and Model B (FraudNet, an entity-embedding NN). Sprint 3.4's cross-model comparison showed FraudNet's val AUC (0.8183) lags Model A's (0.8281) by ~1 percentage-point but its test AUC (0.8229) edges Model A's (0.8070). The question for production: **should we ever switch?**

Two ways to answer it:

1. **Offline replay** — re-score the held-out test slice with both models and compare costs. Cheap but misses the live-feature-distribution + per-request entity-state details that production exposes.
2. **Shadow mode** — every production `/predict` fires the challenger asynchronously alongside the champion. The challenger's prediction is logged but never affects the served decision. Offline analysis joins champion + challenger by `request_id` and computes the same cost / agreement metrics on real-traffic data.

Shadow is the gold standard for production model evaluation. The constraints making it tricky in this project:

- **Latency budget** — the project's 100 ms P95 budget (CLAUDE.md §3) is non-negotiable. A synchronous shadow path adds 2-60 ms (FraudNet's `predict_proba` is ~60 ms vs LightGBM's ~2 ms); even the lower end blows the budget.
- **Challenger failure isolation** — a buggy challenger model that throws on every call would burn CPU + log volume indefinitely. The breaker must trip and stop calling.
- **Feature-shape mismatch** — FraudNet expects different feature columns than LightGBM. The shadow path needs a way to adapt the same `FeatureVector` to FraudNet's input contract without coupling the production request path.
- **Promotion criteria** — when does the challenger become the champion? Need an explicit gate, not "the engineer prefers it".

Alternatives considered:

1. **Synchronous shadow** — adds challenger latency to the request. Blows the budget.
2. **Background process scoring offline logs** — decouples from the request path but loses the live-feature-distribution + per-request state.
3. **Async fire-and-forget on every /predict** with a circuit breaker for failure isolation. Standard production shadow pattern.
4. **Sample-rate shadow** (e.g., 1% of traffic) — reduces log volume but biases the comparison towards the sampled slice.
5. **A/B routing** — actually serve some fraction of traffic from the challenger. Riskier (challenger's decisions affect users); deferred to a future ramp-up phase if Sprint 5.2.c's promotion criteria pass.

## Decision

Shadow scoring is **fire-and-forget asynchronous** with a **three-state circuit breaker**:

1. On every `/predict`, when `Settings.shadow_enabled=True` AND `app.state.app_state.shadow is not None`, the route schedules `asyncio.create_task(shadow.score(...))` AFTER building the `PredictionResponse`. The task does NOT block the response.
2. Inside `_score_one`, `predict_proba` runs in `asyncio.to_thread` so the CPU-bound torch call doesn't stall the event loop.
3. `CircuitBreaker` (5 failures → OPEN, 30 s initial cooldown, exponential 2× backoff to 300 s cap) wraps the call path. OPEN means subsequent calls log `shadow.breaker_open_skip` and skip without invoking the model.
4. Outputs are structured-log events (`shadow.scored` / `shadow.failed` / `shadow.breaker_open_skip`) joinable by `request_id` against the champion's `PredictionResponse`.
5. Shadow disagreement (champion + challenger decisions differ) increments the `fraud_engine_shadow_disagreement_total` Prometheus counter (Sprint 6.1.d retrofit) so [`ShadowDisagreement`](../RUNBOOK.md#alert-shadowdisagreement) alert can fire on sustained > 10% disagreement rate.

| Component | Value |
|---|---|
| Enable gate | `Settings.shadow_enabled` (default `False`) |
| Failure threshold | 5 consecutive failures → OPEN |
| Initial cooldown | 30 s |
| Backoff factor | 2× per HALF_OPEN failure |
| Max cooldown | 300 s |
| Disagreement signal | `fraud_engine_shadow_disagreement_total` counter |
| Promotion criteria | `cost_improvement > 2% AND p_value < 0.05 AND agreement_rate > 85%` (Sprint 5.2.c) |

The full implementation lives in [`src/fraud_engine/api/shadow.py`](../../src/fraud_engine/api/shadow.py) + [`src/fraud_engine/api/circuit_breaker.py`](../../src/fraud_engine/api/circuit_breaker.py).

## Rationale

1. **Fire-and-forget is the only pattern that fits the latency budget.** Sprint 5.2.b measured P95 = 81.4 ms with shadow failing on every request (worst case: breaker hasn't tripped yet). With the breaker tripped, shadow is a no-op (~0 ms added). The "shadow failing" case is the upper bound; the breaker ensures it doesn't persist past 5 calls.
2. **`asyncio.to_thread` for `predict_proba` is mandatory.** FraudNet's per-row predict is ~60 ms on CPU. Running it on the event loop blocks the loop for that duration, breaking the latency budget for ANY concurrent `/predict` request. Offloading to a worker thread keeps the loop responsive.
3. **The breaker isolates sustained challenger failures.** A buggy deploy of FraudNet that throws on every request would, without the breaker, generate 60 ms of wasted CPU per `/predict` + a `shadow.failed` log line. The breaker trips at 5 failures (30 s window), drops the wasted work to ~0, and probes after the cooldown. Exponential backoff prevents flapping.
4. **Structured-log output joins cleanly by `request_id`.** Every `/predict` already binds a `request_id` via middleware (Sprint 5.1.f); the shadow event reuses it. Offline `jq` joins are one-liners (see [`docs/RUNBOOK.md` Appendix B](../RUNBOOK.md#appendix-b--jq-recipes)).
5. **Disagreement counter is the Grafana / alerting hook.** Sprint 6.1.d added `fraud_engine_shadow_disagreement_total` (incremented on every `agree_decision is False`). The `ShadowDisagreement` alert rule fires at > 10% sustained disagreement; operators investigate via the shadow-vs-champion comparison report (Sprint 5.2.c) before considering promotion.
6. **Promotion criteria are explicit + multi-factor.** Sprint 5.2.c's report defines cost_improvement > 2% AND p_value < 0.05 AND agreement_rate > 85% — all three must pass simultaneously. The conjunction guards against the common failure mode of cherry-picking a single metric.

## Consequences

- **Per-request log volume roughly doubles when `shadow_enabled=True`.** Every `/predict` emits both a `prediction.logged` + a `shadow.scored` (or `shadow.failed` / `shadow.breaker_open_skip`) event. Operators should size their log-aggregation pipeline accordingly.
- **The breaker's OPEN state hides real challenger regressions.** If FraudNet is genuinely degraded and the breaker is tripped, the dashboard shows "no shadow data" — a soft signal that needs operator interpretation. Sprint 6.1.d's `fraud_engine_shadow_breaker_state{state="open"} == 1` gauge is the explicit detection.
- **`asyncio.to_thread` consumes the default thread-pool executor's slots.** At sufficiently high RPS (hundreds per second per worker), the pool saturates and shadow scoring slows down (but doesn't block `/predict`). The current project's RPS doesn't hit this; future scaling needs an explicit thread-pool sizing decision.
- **The shutdown drain has a timeout.** `ShadowService.disconnect()` waits up to 5 s for pending shadow tasks to complete on lifespan shutdown; tasks still running after 5 s are cancelled. A graceful shutdown with no in-flight shadow tasks completes instantly; the worst case is 5 s added to container restart time.
- **Promotion criteria are non-trivial to validate** — Sprint 5.2.c's bootstrap significance test (10K resamples, two-sided p-value) takes ~30 seconds per offline comparison run. The promotion procedure is an operator command, not an automated trigger.
- **Shadow doesn't observe Postgres writes.** The audit-log Postgres `predictions` table records only champion predictions. A future Sprint 6.x can add a `shadow_predictions` table or a `model_role` discriminator column if shadow audit becomes a compliance requirement.

## Revisit triggers

- **Promotion criteria pass for FraudNet** (Sprint 5.2.c's report indicates a green verdict). Then re-evaluate: actually deploy FraudNet as champion (potentially via blue/green or A/B), demote LightGBM to shadow.
- **A third model joins the candidate set.** Current architecture assumes single-shadow; multi-shadow support (load 2+ challengers in parallel) is a Sprint 6.x extension.
- **Latency budget tightens** (e.g., to 50 ms P95). The thread-pool overhead might become noticeable; consider a dedicated executor or moving shadow to a sidecar process.
- **Compliance audit requires per-shadow-prediction Postgres durability.** Promote shadow output from structlog stream to a dedicated table.
- **Sustained `ShadowDisagreement` alerts despite no champion regression.** Likely indicates the shadow's calibration is mis-aligned to the champion's threshold; investigate before any promotion.
- **Process-restart latency dominates the SLO** (current 5 s drain timeout). Tune `_DRAIN_TIMEOUT_S` down if the worst case becomes the common case.

## References

- [`src/fraud_engine/api/shadow.py`](../../src/fraud_engine/api/shadow.py) — `ShadowService` implementation.
- [`src/fraud_engine/api/circuit_breaker.py`](../../src/fraud_engine/api/circuit_breaker.py) — three-state breaker.
- [`src/fraud_engine/evaluation/shadow_compare.py`](../../src/fraud_engine/evaluation/shadow_compare.py) (Sprint 5.2.c) — bootstrap significance test + promotion criteria.
- [`sprints/sprint_5/prompt_5_2_b_report.md`](../../sprints/sprint_5/prompt_5_2_b_report.md) — original implementation + 81.4 ms P95 measurement with shadow failing.
- [`sprints/sprint_5/prompt_5_2_c_report.md`](../../sprints/sprint_5/prompt_5_2_c_report.md) — promotion criteria + bootstrap testing.
- [`configs/alerts/alert_rules.yml`](../../configs/alerts/alert_rules.yml) — `ShadowDisagreement` rule definition.
- [`docs/RUNBOOK.md#alert-shadowdisagreement`](../RUNBOOK.md#alert-shadowdisagreement) — operator remediation.
- [`docs/MODEL_CARD.md`](../MODEL_CARD.md) — cross-model comparison table.
- Related: [ADR-0005 — LightGBM as production champion](0005-lightgbm-as-production.md) (the decision that promotes one specific candidate to champion).
