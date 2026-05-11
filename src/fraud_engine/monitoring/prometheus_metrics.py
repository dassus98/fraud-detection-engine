"""Single source of truth for every Prometheus metric the fraud-engine API emits.

Sprint 6 prompt 6.1.a: this module centralises the API's monitoring surface
so an operator running `curl localhost:8000/metrics` sees a consistent,
documented set of signals — and so the next Sprint-6 prompts (Grafana
dashboards, drift detection, alerting rules) can reference metric names by
constant rather than free-form string.

Pre-Sprint-6.1.a: 4 inline `Histogram`s in `api/main.py` covered per-stage
latency.  Post-PR: the same 4 histograms live here (names + buckets
unchanged for backward compatibility), plus 7 new metrics covering
prediction volume, score distribution, dependency health, model identity,
and shadow-mode signals.

Public surface (re-exported by `fraud_engine.monitoring`):

    Histograms (latency — seconds):
        - FEATURE_FETCH_SECONDS
        - INFERENCE_SECONDS
        - SHAP_SECONDS
        - PREDICT_TOTAL_SECONDS

    Histogram (calibrated probability — [0, 1]):
        - PREDICTION_SCORE

    Counters (monotonic):
        - PREDICTIONS_TOTAL{decision}
        - DEGRADED_MODE_TOTAL
        - SHADOW_TOTAL{event}

    Gauges (instantaneous):
        - DEPENDENCY_UP{component}
        - SHADOW_BREAKER_STATE{state}
        - MODEL_INFO{model_version}

    Constants (bucket tuples):
        - LATENCY_BUCKETS
        - SCORE_BUCKETS

    Helpers:
        - set_shadow_breaker_state(state)

Business rationale:
    Production fraud APIs need a comprehensive monitoring surface so an
    on-call engineer can answer four questions from `/metrics` alone:
    "what's the prediction volume?", "what's the latency?", "what's the
    score distribution (drift seed)?", and "which dependencies are
    healthy?".  Centralising the metric definitions in one module is the
    standard prometheus-client layout — it lets operators grep one file
    for "what does this app emit?", and lets tests import the metrics
    directly without booting the FastAPI app.

Trade-offs considered:
    - **Centralise here vs leave 4 histograms inline in main.py.** Inline
      worked for 4 metrics; doesn't scale to ~10 (visual clutter, harder
      to test).  Centralising mirrors the prometheus-client README's
      recommended layout and is the convention in production codebases
      with mature monitoring.

    - **Three metric types: Counter / Histogram / Gauge.**  Counter for
      monotonic volumes (predictions, degraded incidents, shadow events).
      Histogram for distributions (latencies + score) — bucket counts
      aggregate cleanly across instances; Summaries can't.  Gauge for
      instantaneous state (dependency health, breaker state, model
      identity).  Rejected: a native `info` metric type — Prometheus has
      no such type; the labelled-gauge-set-to-1 pattern (used by
      `MODEL_INFO`) is the established convention.

    - **Latency buckets reused verbatim from Sprint 5.1.f**
      `[0.005, 0.010, 0.025, 0.050, 0.100, 0.250]`.  0.005 covers typical
      sub-5ms inference / SHAP per-call cost; 0.025–0.050 covers the
      typical feature-fetch path; **0.100** is the load-bearing bucket
      for the CLAUDE.md §3 P95 budget alert; 0.250 covers the tail.  The
      bucket boundary at exactly the budget value makes the alert
      `histogram_quantile(0.95, ...) > 0.100` a clean read.

      *Math: bucket choice for p95 reads.*  Prometheus's
      `histogram_quantile(0.95, ...)` linearly interpolates within the
      bucket containing the quantile.  With these buckets, p95 ≈ 70 ms
      falls in [0.050, 0.100] (interpolation accuracy ~5 ms — adequate
      for "is p95 < 100ms?" alerting); p95 ≈ 110 ms falls in [0.100,
      0.250] (the alert has already fired).

    - **Score buckets `[0.0, 0.01, 0.05, 0.08, 0.1, 0.2, 0.5, 0.8, 1.0]`.**
      0.08 = `Settings.decision_threshold` (post Sprint 4.4 cost-optimal
      value). Fine resolution at the low end where the decision boundary
      lives; coarser at the high end (rare high-confidence-fraud cases
      don't need fine bucketing).  0.0 and 1.0 are boundary anchors so
      "fraction of scores ≤ X" reads cleanly.

    - **Bounded-cardinality labels only.**  Per Prometheus best practice
      — high-cardinality labels (request_id, txn_id, card1, entity_id)
      explode the time-series storage.  All label values are drawn from
      finite enumerations: decision ∈ {block, allow}; event ∈ {scored,
      failed, breaker_open_skip}; component ∈ {redis, postgres, model};
      state ∈ {closed, open, half_open}; model_version is a SHA-256 hex
      that changes only on artefact swap (≤2 series in the lifetime of
      a typical process).

      *Math: cardinality estimate.*  predictions_total: 2 series.
      degraded_mode_total: 1.  shadow_total: 3.  prediction_score: ~12
      internal series (10 buckets + count + sum).  dependency_up: 3.
      shadow_breaker_state: 3.  model_info: ≤2.  Plus 4 latency
      histograms × ~8 internal series each = 32.  **Total ~57 series**
      — Prometheus instances handle 10⁶+ comfortably.

    - **Module is import-side-effecting.**  Defining `Counter("foo")` at
      module scope auto-registers it against `prometheus_client.REGISTRY`
      on import.  Tests must NOT use `importlib.reload` on this module
      (would raise `ValueError: Duplicated timeseries`).  The
      `__init__.py` re-exports references rather than re-importing.

Cross-references:
    - `src/fraud_engine/api/main.py` — imports the metric constants and
      observes / increments inline in the `/predict` path + lifespan.
    - `src/fraud_engine/api/shadow.py` — increments SHADOW_TOTAL +
      SHADOW_BREAKER_STATE inline at the success / failure / skip /
      breaker-state-change sites.
    - `CLAUDE.md` §3 (latency budget that drives bucket choice), §4
      (monitoring/ module home), §8 (decision_threshold = 0.08 that
      drives score bucket choice).
"""

from __future__ import annotations

from typing import Final

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------
# Bucket constants — exposed so callers / tests can introspect.
# ---------------------------------------------------------------------

# Latency buckets (seconds) — moved verbatim from `api/main.py:200-207`
# (Sprint 5.1.f).  Same buckets ensure histogram_quantile() reads from
# `/metrics` are byte-identical pre/post move.  See the module docstring
# for the design discussion.
LATENCY_BUCKETS: Final[tuple[float, ...]] = (
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.250,
)

# Score buckets (calibrated probability ∈ [0, 1]).  0.08 anchors at the
# post-Sprint-4.4 cost-optimal decision threshold (CLAUDE.md §8).  Fine
# resolution at the low end where the decision boundary lives; coarser at
# the high end where high-confidence-fraud is rare.
SCORE_BUCKETS: Final[tuple[float, ...]] = (
    0.0,
    0.01,
    0.05,
    0.08,
    0.1,
    0.2,
    0.5,
    0.8,
    1.0,
)


# ---------------------------------------------------------------------
# Histograms — per-stage latency (4 existing, moved verbatim).
# ---------------------------------------------------------------------

FEATURE_FETCH_SECONDS: Final[Histogram] = Histogram(
    "fraud_engine_feature_fetch_seconds",
    "Time to fetch features (Tier-1 inline + Redis MGET + Postgres probe).",
    buckets=LATENCY_BUCKETS,
)
INFERENCE_SECONDS: Final[Histogram] = Histogram(
    "fraud_engine_inference_seconds",
    "Time for LightGBM predict_proba + isotonic calibration.",
    buckets=LATENCY_BUCKETS,
)
SHAP_SECONDS: Final[Histogram] = Histogram(
    "fraud_engine_shap_seconds",
    "Time for SHAP top-k contributions + reason mapping.",
    buckets=LATENCY_BUCKETS,
)
PREDICT_TOTAL_SECONDS: Final[Histogram] = Histogram(
    "fraud_engine_predict_total_seconds",
    "End-to-end /predict latency (excludes network round-trip).",
    buckets=LATENCY_BUCKETS,
)


# ---------------------------------------------------------------------
# Histogram — calibrated probability (drift-detection seed).
# ---------------------------------------------------------------------

PREDICTION_SCORE: Final[Histogram] = Histogram(
    "fraud_engine_prediction_score",
    "Distribution of calibrated fraud probabilities returned to clients.",
    buckets=SCORE_BUCKETS,
)


# ---------------------------------------------------------------------
# Counters — monotonic volumes.
# ---------------------------------------------------------------------

PREDICTIONS_TOTAL: Final[Counter] = Counter(
    "fraud_engine_predictions_total",
    "Predictions served, labelled by decision.",
    ["decision"],
)
DEGRADED_MODE_TOTAL: Final[Counter] = Counter(
    "fraud_engine_degraded_mode_total",
    "Predictions served while a feature source (Redis/Postgres) was unreachable.",
)
SHADOW_TOTAL: Final[Counter] = Counter(
    "fraud_engine_shadow_total",
    "Shadow-mode events: scored (success), failed, breaker_open_skip.",
    ["event"],
)

# Sprint 6.1.d retrofit — surface offline drift + shadow-disagreement
# signals on the Prometheus scrape so alert rules can fire on them
# without a log-scraper sidecar.  Both unlabelled (1 series each) —
# per-feature breakdown stays in the JSONL stream (high-cardinality risk
# if labelled by feature_name across 743 features).
DRIFT_ALERTS_TOTAL: Final[Counter] = Counter(
    "fraud_engine_drift_alerts_total",
    "Cumulative drift alerts written by DriftMonitor.check_and_alert. "
    "One increment per JSONL line written to "
    "logs/drift/{run_id}/drift_alerts.jsonl.",
)
SHADOW_DISAGREEMENT_TOTAL: Final[Counter] = Counter(
    "fraud_engine_shadow_disagreement_total",
    "Cumulative shadow-vs-champion decision disagreements. "
    "One increment per `shadow.scored` event with agree_decision=False.",
)


# ---------------------------------------------------------------------
# Gauges — instantaneous state.
# ---------------------------------------------------------------------

DEPENDENCY_UP: Final[Gauge] = Gauge(
    "fraud_engine_dependency_up",
    "1 if the named component is healthy, 0 otherwise.",
    ["component"],
)
SHADOW_BREAKER_STATE: Final[Gauge] = Gauge(
    "fraud_engine_shadow_breaker_state",
    "Shadow circuit-breaker state — 1 hot for the active state, 0 for the others.",
    ["state"],
)
MODEL_INFO: Final[Gauge] = Gauge(
    "fraud_engine_model_info",
    "Info-style gauge: always 1 with model_version on the label.",
    ["model_version"],
)


# ---------------------------------------------------------------------
# Documented label-value sets — for tests, dashboards, and operators.
# Not used at runtime; the metrics accept any label value.  Kept
# in-sync with the call sites' actual usage by
# `tests/unit/test_prometheus_metrics.py::test_label_value_sets_are_finite`.
# ---------------------------------------------------------------------

DECISION_LABELS: Final[tuple[str, ...]] = ("block", "allow")
SHADOW_EVENT_LABELS: Final[tuple[str, ...]] = (
    "scored",
    "failed",
    "breaker_open_skip",
)
DEPENDENCY_LABELS: Final[tuple[str, ...]] = ("redis", "postgres", "model")
SHADOW_BREAKER_STATE_LABELS: Final[tuple[str, ...]] = (
    "closed",
    "open",
    "half_open",
)


# ---------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------


def set_shadow_breaker_state(state: str) -> None:
    """Set the active shadow-breaker state to 1, all others to 0.

    Prometheus convention for representing a categorical instantaneous
    value: one labelled gauge per possible state; the active one is 1,
    the others are 0.  This helper enforces the invariant so callers
    can't accidentally leave a stale state hot.

    Business rationale:
        The shadow circuit breaker has three states (closed, open,
        half_open) and an on-call operator querying Grafana wants a
        single panel showing "what state is the breaker in right now?".
        Emitting one Gauge per state with exactly one of them at 1
        renders cleanly as a stacked bar / sum-by-state heatmap.

    Trade-offs considered:
        - **Three labelled series with one hot vs a single Gauge with
          numeric encoding (e.g. closed=0, open=1, half_open=2).**  The
          numeric encoding loses semantic meaning in a dashboard query
          ("what does state=2 mean?") and prevents `sum by (state)`
          aggregation.  The labelled-with-one-hot pattern is the
          standard Prometheus convention for categorical state.

        - **Helper function vs inline three .set calls at every call
          site.**  Helper centralises the invariant ("exactly one is 1,
          all others are 0") so a future caller adding a new state
          can't accidentally leave stale state hot.

    Args:
        state: One of `SHADOW_BREAKER_STATE_LABELS` ("closed" / "open" /
            "half_open").  An unknown value silently zeroes all known
            states (no hot one) — defensive: a caller passing an
            unexpected literal shouldn't crash the request path.
    """
    for known_state in SHADOW_BREAKER_STATE_LABELS:
        SHADOW_BREAKER_STATE.labels(state=known_state).set(1.0 if known_state == state else 0.0)


__all__ = [
    "DECISION_LABELS",
    "DEGRADED_MODE_TOTAL",
    "DEPENDENCY_LABELS",
    "DEPENDENCY_UP",
    "DRIFT_ALERTS_TOTAL",
    "FEATURE_FETCH_SECONDS",
    "INFERENCE_SECONDS",
    "LATENCY_BUCKETS",
    "MODEL_INFO",
    "PREDICTIONS_TOTAL",
    "PREDICTION_SCORE",
    "PREDICT_TOTAL_SECONDS",
    "SCORE_BUCKETS",
    "SHADOW_BREAKER_STATE",
    "SHADOW_BREAKER_STATE_LABELS",
    "SHADOW_DISAGREEMENT_TOTAL",
    "SHADOW_EVENT_LABELS",
    "SHADOW_TOTAL",
    "SHAP_SECONDS",
    "set_shadow_breaker_state",
]
