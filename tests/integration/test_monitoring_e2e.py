"""End-to-end monitoring integration tests for Sprint 6.

Sprint 6 prompt 6.1.e — closes the loop on the monitoring stack built
across Sprints 6.1.a-d:

    - 6.1.a delivered 11 Prometheus metrics + the /metrics scrape endpoint.
    - 6.1.b delivered the offline DriftMonitor (PSI on 743 features).
    - 6.1.c delivered the offline PerformanceMonitor (rolling AUC/cost).
    - 6.1.d delivered the 7-panel Grafana dashboard + 5 alert rules +
      2 retrofit Counters (drift_alerts_total + shadow_disagreement_total).

This file's tests run against the live FastAPI app (via
`httpx.ASGITransport(app=app)` + `LifespanManager`) — same pattern as
`test_api_e2e.py` (Sprint 5.1.f). Each test scrapes /metrics over HTTP
and parses the prometheus text format via
`prometheus_client.parser.text_string_to_metric_families` to assert
counter VALUES (not just presence), going one level deeper than 5.1.f's
substring-grep test.

Test scenarios (per the spec):
    1. test_predict_volume_reflected_in_metrics_counters — drive 5
       /predict; assert fraud_engine_predictions_total{decision} delta
       sums to 5.
    2. test_predict_latency_histograms_observe_each_request — drive 5
       /predict; assert fraud_engine_predict_total_seconds_count delta
       == 5; at least one bucket has cumulative count >= 5.
    3. test_synthetic_drift_increments_drift_alerts_counter — build a
       synthetic baseline parquet in tmp_path, run
       DriftMonitor.check_and_alert(shifted_recent_window), scrape
       /metrics, assert fraud_engine_drift_alerts_total delta >= 1.
    4. test_drift_alert_rule_threshold_would_fire_after_synthetic_drift
       — load configs/alerts/alert_rules.yml, find the FeatureDrift
       rule, regex out its threshold (`> 0`), assert post-drift counter
       delta exceeds it. Static-evaluates the alert-rule semantics
       without needing a live Prometheus.
    5. test_dashboard_and_alerts_reference_emitted_metrics — regex out
       every fraud_engine_* metric name from the dashboard JSON's
       `expr` fields + the alert rules' `expr` fields; scrape /metrics;
       assert each referenced name appears in the live scrape (suffix-
       tolerant for histograms' _bucket / _count / _sum / _created
       families). Catches dashboard/alert-references-deleted-metric
       regressions across all future PRs.

Trade-offs considered:
    - **HTTP scrape via the live client (not in-process REGISTRY).**
      Tests exercise the actual scrape path the operator's Prometheus
      hits — catches bugs in the prometheus-fastapi-instrumentator
      wiring that an in-process REGISTRY check would miss.
    - **Delta pattern (capture pre-value, do work, capture post-value)**
      so tests are order-independent against the global REGISTRY
      singleton — same convention as test_prometheus_metrics.py.
    - **DriftMonitor isolated to tmp_path.** check_and_alert writes a
      JSONL line to logs/drift/{run_id}/...; we pass alert_log_dir=
      tmp_path/"drift" so writes never touch the real logs/ directory.
    - **Test 4 uses regex extraction of the alert threshold instead
      of hardcoding `> 0`.** Couples the assertion to alert_rules.yml
      so a future threshold tweak in the rule file automatically
      updates the assertion. Narrow regex (matches `> NUM` suffix
      only); skips with clear message if a future rule uses a complex
      expression.
    - **Test 5 walks dashboard panels recursively** because Grafana
      JSON nests `targets[*].expr` under `panels[*]` (and deeper for
      row-grouped layouts). Suffix-tolerance on metric-name matching
      means `fraud_engine_predict_total_seconds_bucket` matches
      `fraud_engine_predict_total_seconds` family root.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
import redis.asyncio
import yaml
from asgi_lifespan import LifespanManager
from prometheus_client.parser import text_string_to_metric_families

from fraud_engine.api.main import create_app
from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.monitoring.drift import DriftBaselineBuilder, DriftMonitor

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_SAMPLE_TXN_PATH = _FIXTURES_DIR / "sample_txn.json"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ALERT_RULES_PATH = _PROJECT_ROOT / "configs" / "alerts" / "alert_rules.yml"
_DASHBOARD_PATH = _PROJECT_ROOT / "configs" / "grafana" / "fraud_dashboard.json"

# Drive-volume sample size for tests 1 + 2. 5 is large enough that any
# off-by-one in bucket cumulation would surface; small enough that the
# total wall is <500 ms for fast iteration.
_DRIVE_REQUEST_COUNT = 5

# Metric-name regex: matches any fraud_engine_<word> token. Greedy on
# trailing _\w+ so it captures the full series name (including suffixes
# like _bucket, _count, _sum, _created, _total).
_METRIC_NAME_RE = re.compile(r"\bfraud_engine_\w+\b")

# Histogram + Counter suffixes that prometheus_client appends to the
# user-defined metric name. Test 5 strips these to match the family
# root against /metrics output.
_PROMETHEUS_SUFFIXES: tuple[str, ...] = (
    "_bucket",
    "_count",
    "_sum",
    "_created",
    "_total",
)

# Synthetic-drift baseline + recent-window sample sizes.  Small (1k +
# 200) so the test is fast (<1s for the drift call) but large enough
# that the +1.5σ shift produces stable PSI > the 0.05 threshold the
# test uses.
_BASELINE_N = 1_000
_RECENT_N = 200


# ---------------------------------------------------------------------
# Fixtures (mirror test_api_e2e.py).
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_request_payload() -> dict[str, object]:
    """Load the canonical sample TransactionRequest payload."""
    return json.loads(_SAMPLE_TXN_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def deps_reachable() -> None:
    """Probe Redis + Postgres; skip the module if either is unreachable.

    Mirrors test_api_e2e.py's deps_reachable verbatim — we need both
    dependencies for the live /predict path that drives the metrics.
    """
    settings = get_settings()

    async def _probe_redis() -> None:
        client = redis.asyncio.from_url(settings.redis_url)
        try:
            await client.ping()
        finally:
            await client.aclose()

    async def _probe_postgres() -> None:
        # Lazy import — asyncpg is heavy.
        import asyncpg  # type: ignore[import-untyped]  # noqa: PLC0415 — test-local

        conn = await asyncpg.connect(settings.postgres_url, timeout=2.0)
        try:
            await conn.fetchval("SELECT 1")
        finally:
            await conn.close()

    try:
        asyncio.run(_probe_redis())
    except Exception as exc:  # noqa: BLE001 — many failure modes
        pytest.skip(f"Redis unreachable at {settings.redis_url}: {exc}")
    try:
        asyncio.run(_probe_postgres())
    except Exception as exc:  # noqa: BLE001 — many failure modes
        pytest.skip(f"Postgres unreachable at {settings.postgres_url}: {exc}")


@pytest.fixture
async def client(
    deps_reachable: None,  # noqa: ARG001 — module-scope dep
) -> AsyncIterator[httpx.AsyncClient]:
    """Lifespan-managed AsyncClient against a fresh app."""
    app = create_app()
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


# ---------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------


def _scrape_metrics_text(
    client: httpx.AsyncClient,
) -> dict[str, dict[tuple[tuple[str, str], ...], float]]:
    """Synchronous wrapper to scrape /metrics and parse into a dict.

    Returns {metric_name: {label_tuple: value}} where label_tuple is a
    sorted tuple of (label_name, label_value) pairs (so unordered labels
    compare equal). Includes ALL samples from each family (so for a
    histogram, bucket / count / sum / created samples each appear with
    their own metric name).

    Tests call this synchronously from inside an `async def` test body
    via `await` — see the actual call sites below.
    """
    raise NotImplementedError("Use _scrape_metrics(client) instead — async helper.")


async def _scrape_metrics(
    client: httpx.AsyncClient,
) -> dict[str, dict[tuple[tuple[str, str], ...], float]]:
    """Hit /metrics and parse into {metric_name: {labels_tuple: value}}.

    The label_tuple key is a sorted tuple of (k, v) pairs so dict
    lookups don't depend on label-ordering in the prometheus text
    format output.
    """
    response = await client.get("/metrics")
    assert response.status_code == 200
    parsed: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            label_tuple = tuple(sorted(sample.labels.items()))
            parsed.setdefault(sample.name, {})[label_tuple] = float(sample.value)
    return parsed


def _sample_value(
    parsed: dict[str, dict[tuple[tuple[str, str], ...], float]],
    name: str,
    labels: dict[str, str] | None = None,
) -> float:
    """Read one sample's value from a parsed-metrics dict, or 0.0 if absent.

    Mirrors `prometheus_client.REGISTRY.get_sample_value` semantics over
    the scraped (rather than in-process) data.
    """
    series = parsed.get(name, {})
    label_tuple = tuple(sorted((labels or {}).items()))
    return series.get(label_tuple, 0.0)


def _drift_setup(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """Build a tiny baseline parquet + a strongly-shifted recent window.

    Returns (baseline_path, recent_window) — caller passes these to a
    DriftMonitor configured with `alert_log_dir=tmp_path/"drift"` so
    the JSONL writes stay in tmp.
    """
    rng = np.random.default_rng(42)
    baseline_values = rng.normal(loc=0.0, scale=1.0, size=_BASELINE_N)
    train_df = pd.DataFrame({"feature_a": baseline_values})
    baseline_df = DriftBaselineBuilder.build(
        train_df=train_df,
        feature_names=["feature_a"],
        n_bins=10,
    )
    baseline_path = tmp_path / "baseline.parquet"
    baseline_df.to_parquet(baseline_path, index=False)

    # +1.5σ shift produces PSI well above the 0.05 alert threshold we
    # configure on the per-test Settings below.
    recent_window = pd.DataFrame({"feature_a": rng.normal(loc=1.5, scale=1.0, size=_RECENT_N)})
    return baseline_path, recent_window


def _walk_dashboard_for_metric_names(panels: list[dict[str, Any]]) -> set[str]:
    """Recursively walk dashboard panels; return every fraud_engine_* token.

    Panels can carry `targets[*].expr`, and `panels[*]` may nest under a
    `row` panel's own `panels[]` list. We walk both levels.
    """
    found: set[str] = set()
    for panel in panels:
        for target in panel.get("targets", []):
            expr = target.get("expr")
            if isinstance(expr, str):
                found.update(_METRIC_NAME_RE.findall(expr))
        # Nested panels under a row panel.
        nested = panel.get("panels")
        if isinstance(nested, list):
            found.update(_walk_dashboard_for_metric_names(nested))
    return found


def _strip_prometheus_suffix(name: str) -> str:
    """Strip a Prometheus auto-appended suffix to recover the family root.

    A reference to `fraud_engine_predict_total_seconds_bucket` should
    match the histogram family `fraud_engine_predict_total_seconds`
    in the parsed scrape (the parser keeps the suffix on the sample
    name; we tolerate either form via this strip).
    """
    for suffix in _PROMETHEUS_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


# ---------------------------------------------------------------------
# Test 1 — predict-volume reflected in counter values.
# ---------------------------------------------------------------------


async def test_predict_volume_reflected_in_metrics_counters(
    client: httpx.AsyncClient,
    sample_request_payload: dict[str, object],
) -> None:
    """Drive N /predict; assert counter delta sums to N across decisions."""
    pre = await _scrape_metrics(client)
    pre_block = _sample_value(pre, "fraud_engine_predictions_total", {"decision": "block"})
    pre_allow = _sample_value(pre, "fraud_engine_predictions_total", {"decision": "allow"})

    for _ in range(_DRIVE_REQUEST_COUNT):
        response = await client.post("/predict", json=sample_request_payload)
        assert response.status_code == 200, response.text

    post = await _scrape_metrics(client)
    post_block = _sample_value(post, "fraud_engine_predictions_total", {"decision": "block"})
    post_allow = _sample_value(post, "fraud_engine_predictions_total", {"decision": "allow"})

    delta = (post_block - pre_block) + (post_allow - pre_allow)
    assert delta == pytest.approx(_DRIVE_REQUEST_COUNT, abs=1e-6), (
        f"counter delta = {delta}, expected {_DRIVE_REQUEST_COUNT}. "
        f"Block delta = {post_block - pre_block}, allow delta = {post_allow - pre_allow}."
    )


# ---------------------------------------------------------------------
# Test 2 — latency histograms observe each request.
# ---------------------------------------------------------------------


async def test_predict_latency_histograms_observe_each_request(
    client: httpx.AsyncClient,
    sample_request_payload: dict[str, object],
) -> None:
    """Drive N /predict; assert _count delta == N + at least one bucket has cumulative count >= N."""
    pre = await _scrape_metrics(client)
    pre_count = _sample_value(pre, "fraud_engine_predict_total_seconds_count")

    for _ in range(_DRIVE_REQUEST_COUNT):
        response = await client.post("/predict", json=sample_request_payload)
        assert response.status_code == 200, response.text

    post = await _scrape_metrics(client)
    post_count = _sample_value(post, "fraud_engine_predict_total_seconds_count")

    assert post_count - pre_count == pytest.approx(
        _DRIVE_REQUEST_COUNT, abs=1e-6
    ), f"_count delta = {post_count - pre_count}, expected {_DRIVE_REQUEST_COUNT}"

    # At least the +Inf bucket should have observed every request.  The
    # parser exposes each bucket with `le` label; we sum the +Inf bucket
    # delta as a sanity bound.
    pre_inf_bucket = _sample_value(pre, "fraud_engine_predict_total_seconds_bucket", {"le": "+Inf"})
    post_inf_bucket = _sample_value(
        post, "fraud_engine_predict_total_seconds_bucket", {"le": "+Inf"}
    )
    assert post_inf_bucket - pre_inf_bucket >= _DRIVE_REQUEST_COUNT, (
        f"+Inf bucket delta = {post_inf_bucket - pre_inf_bucket}, "
        f"expected >= {_DRIVE_REQUEST_COUNT}"
    )


# ---------------------------------------------------------------------
# Test 3 — synthetic drift increments the drift_alerts counter.
# ---------------------------------------------------------------------


async def test_synthetic_drift_increments_drift_alerts_counter(
    client: httpx.AsyncClient,
    tmp_path: Path,
) -> None:
    """Trigger DriftMonitor.check_and_alert on synthetic drift; scrape /metrics; assert delta >= 1."""
    pre = await _scrape_metrics(client)
    pre_drift = _sample_value(pre, "fraud_engine_drift_alerts_total")

    baseline_path, recent_window = _drift_setup(tmp_path)
    # Low alert threshold so the +1.5σ shift trips the alert — we want
    # to verify the counter wires through to /metrics, not the threshold
    # tuning (covered by test_drift.py).
    settings = Settings(
        drift_baseline_path=baseline_path,
        drift_alert_log_dir=tmp_path / "drift",
        psi_alert_threshold=0.05,
    )
    monitor = DriftMonitor(baseline_path=baseline_path, settings=settings)
    n_alerts = monitor.check_and_alert(recent_window, run_id="e2e-drift-001")
    assert n_alerts >= 1, f"expected >=1 drift alert, got {n_alerts}"

    post = await _scrape_metrics(client)
    post_drift = _sample_value(post, "fraud_engine_drift_alerts_total")

    assert post_drift - pre_drift >= 1.0, (
        f"drift counter delta = {post_drift - pre_drift}, expected >=1. "
        f"DriftMonitor.check_and_alert returned {n_alerts} but the counter "
        f"increment didn't surface on the /metrics scrape."
    )


# ---------------------------------------------------------------------
# Test 4 — alert rule threshold would fire after synthetic drift.
# ---------------------------------------------------------------------


async def test_drift_alert_rule_threshold_would_fire_after_synthetic_drift(
    client: httpx.AsyncClient,
    tmp_path: Path,
) -> None:
    """Static-eval: load FeatureDrift rule's threshold; assert post-drift counter delta exceeds it.

    Without a live Prometheus we can't evaluate `increase(...[1h]) > 0`
    literally — but we can extract the literal threshold from the rule's
    expr string and assert our counter delta exceeds it. That's what
    Prometheus would do internally on the next evaluation tick.
    """
    rules_yaml = yaml.safe_load(_ALERT_RULES_PATH.read_text(encoding="utf-8"))
    feature_drift_rule = next(
        r for group in rules_yaml["groups"] for r in group["rules"] if r["alert"] == "FeatureDrift"
    )
    threshold_match = re.search(r">\s*([\d.]+)\s*$", feature_drift_rule["expr"])
    assert threshold_match, (
        f"FeatureDrift expr does not match the narrow `> NUM` suffix pattern; "
        f"got {feature_drift_rule['expr']!r}. Update test 4 if the rule's "
        f"expression shape changes."
    )
    threshold = float(threshold_match.group(1))

    pre = await _scrape_metrics(client)
    pre_drift = _sample_value(pre, "fraud_engine_drift_alerts_total")

    baseline_path, recent_window = _drift_setup(tmp_path)
    settings = Settings(
        drift_baseline_path=baseline_path,
        drift_alert_log_dir=tmp_path / "drift",
        psi_alert_threshold=0.05,
    )
    monitor = DriftMonitor(baseline_path=baseline_path, settings=settings)
    monitor.check_and_alert(recent_window, run_id="e2e-drift-002")

    post = await _scrape_metrics(client)
    post_drift = _sample_value(post, "fraud_engine_drift_alerts_total")

    delta = post_drift - pre_drift
    assert delta > threshold, (
        f"FeatureDrift alert WOULD NOT fire: counter delta {delta} <= threshold {threshold}. "
        f"The synthetic drift trigger didn't generate enough alerts to trip the rule."
    )


# ---------------------------------------------------------------------
# Test 5 — dashboard + alerts reference only metrics the app emits.
# ---------------------------------------------------------------------


async def test_dashboard_and_alerts_reference_emitted_metrics(
    client: httpx.AsyncClient,
    sample_request_payload: dict[str, object],
) -> None:
    """Every fraud_engine_* metric referenced by the dashboard or alerts must be registered.

    Catches the dangling-reference class of bugs: if someone deletes
    or renames a metric in prometheus_metrics.py, the dashboard panel
    silently shows "no data" and the alert rule silently never fires.

    Source of truth for "registered" is `family.name` in the parsed
    scrape — NOT individual sample names — because a Counter with no
    observed labels (e.g. SHADOW_TOTAL when shadow is disabled) emits
    a HELP/TYPE block in the scrape but no sample lines.  The family
    name is the root that dashboard / alert expressions reference;
    sample-name suffixes (`_bucket` / `_count` / etc.) are appended at
    serialisation time.
    """
    # Drive at least one /predict so request-path metrics have samples.
    await client.post("/predict", json=sample_request_payload)

    dashboard_doc = json.loads(_DASHBOARD_PATH.read_text(encoding="utf-8"))
    alerts_doc = yaml.safe_load(_ALERT_RULES_PATH.read_text(encoding="utf-8"))

    referenced: set[str] = set()
    referenced.update(_walk_dashboard_for_metric_names(dashboard_doc.get("panels", [])))
    for group in alerts_doc.get("groups", []):
        for rule in group.get("rules", []):
            expr = rule.get("expr", "")
            referenced.update(_METRIC_NAME_RE.findall(expr))

    assert referenced, (
        "no fraud_engine_* metric names extracted from dashboard/alerts — "
        "verify the dashboard JSON and alert rules YAML actually carry expressions"
    )

    # Build the registered-family-root set from family.name (which is
    # the root regardless of whether any sample has been observed).
    response = await client.get("/metrics")
    assert response.status_code == 200
    emitted_family_roots = {family.name for family in text_string_to_metric_families(response.text)}

    missing: list[str] = []
    for ref in referenced:
        root = _strip_prometheus_suffix(ref)
        if root not in emitted_family_roots and ref not in emitted_family_roots:
            missing.append(ref)

    assert not missing, (
        f"Dashboard or alerts reference metrics not registered on /metrics: {sorted(missing)}. "
        f"Either the metric was renamed/deleted in prometheus_metrics.py without updating "
        f"the dashboard/alerts, or the metric name regex needs adjustment. "
        f"Registered family roots: "
        f"{sorted(name for name in emitted_family_roots if name.startswith('fraud_engine_'))}"
    )
