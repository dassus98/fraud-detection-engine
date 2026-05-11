"""5-second portfolio demo — drive two hardcoded transactions through the live API.

Sprint 6 prompt 6.2.c. Lets a portfolio reviewer experience the system
without spinning up the full stack from scratch: one curl-level command
runs both an "obvious fraud" + a "clearly legit" payload through the
live `/predict` route and prints score + decision + top-5 SHAP reasons
in human-readable form.

Operator workflow:
    1. Start the dev stack: `docker compose -f docker-compose.dev.yml up -d`.
    2. Start uvicorn on host: `make serve`.
    3. Run: `uv run python scripts/demo_prediction.py`.

The script is intentionally self-contained — payloads are hardcoded
inline so the demo doesn't depend on `tests/fixtures/` being present in
a copy-of-the-repo download.

Business rationale:
    The portfolio target audience (Wealthsimple / Mercury / RBC / Nubank
    hiring committees per CLAUDE.md §1) often wants to TRY the system
    rather than read about it. A 5-second copy-pasteable demo lowers
    the activation energy from "clone + read 5 docs" to "clone + run
    one script + see two predictions with explanations".

Trade-offs considered:
    - **Hardcoded payloads vs. fixture files.** Hardcoded keeps the
      demo self-contained — the reviewer can run it from a freshly-
      cloned shallow checkout without ever opening `tests/fixtures/`.
    - **Click CLI vs. plain script.** Click matches `warmup_redis.py`
      + `build_drift_baseline.py` conventions + gives a `--help` for
      the API URL override.
    - **httpx (sync) vs. requests.** httpx is already a project dep
      (used by the API tests); avoids adding `requests`.
    - **Two payloads, not more.** Spec says two. More payloads risks
      visual clutter on the demo's stdout output.
"""

from __future__ import annotations

import sys
from typing import Any

import click
import httpx

from fraud_engine.utils.logging import get_logger

_logger = get_logger(__name__)

_HTTP_OK: int = 200

# ---------------------------------------------------------------------
# Hardcoded payloads.
# ---------------------------------------------------------------------

# "Clearly legit" — copy of tests/fixtures/sample_txn.json.  Empirically
# scores ~0.004 against the production model (decision=allow).  A typical
# mid-day Mastercard credit transaction with normal velocity + a known
# email domain.
_LEGIT_PAYLOAD: dict[str, Any] = {
    "TransactionID": 3485113,
    "TransactionDT": 13046456,
    "TransactionAmt": 58.95,
    "ProductCD": "W",
    "card1": 4141,
    "card2": 404.0,
    "card3": 150.0,
    "card4": "mastercard",
    "card5": 102.0,
    "card6": "credit",
    "addr1": 441.0,
    "addr2": 87.0,
    "dist1": None,
    "dist2": None,
    "P_emaildomain": "hotmail.com",
    "R_emaildomain": None,
    "DeviceType": None,
    "DeviceInfo": None,
    "vesta_v": {"V1": None, "V2": None, "V3": None, "V137": 0.0, "V339": None},
    "vesta_c": {"C1": 1.0, "C2": 1.0, "C13": 3.0, "C14": 1.0},
    "vesta_d": {"D1": 20.0, "D15": 20.0},
    "vesta_m": {"M1": None, "M2": None, "M4": None, "M9": None},
    "identity": {"id_01": None, "id_31": None, "id_38": None},
}

# "Obvious fraud" — uses the SAME card1 + addr1 as the legit payload (so
# the entity-history features have meaningful Redis-warmed state rather
# than cold-start fallbacks) but flips the fraud-signal columns: amount
# > $500, ProductCD=C (overlapping-class — harder to discriminate),
# DeviceType=mobile (most-fraud-correlated slice per Sprint 4.2),
# free-email-domain on both P_ + R_emaildomain, elevated velocity proxy
# C1/C2/C13/C14, dist1 set to a far-from-baseline value, missing identity.
#
# Note: the actual decision depends on the deployed model + threshold +
# Redis state at demo time. The demo prints whatever the API returns
# rather than asserting; the label "obvious_fraud" describes the SIGNAL
# composition, not a guaranteed block.
_FRAUD_PAYLOAD: dict[str, Any] = {
    "TransactionID": 9999999,
    "TransactionDT": 13050000,
    "TransactionAmt": 999.99,
    "ProductCD": "C",
    "card1": 4141,
    "card2": 404.0,
    "card3": 150.0,
    "card4": "mastercard",
    "card5": 102.0,
    "card6": "credit",
    "addr1": 441.0,
    "addr2": 87.0,
    "dist1": 9999.0,
    "dist2": None,
    "P_emaildomain": "anonymous.com",
    "R_emaildomain": "anonymous.com",
    "DeviceType": "mobile",
    "DeviceInfo": "rare-device-fingerprint",
    "vesta_v": {"V1": None, "V2": None, "V3": None, "V137": 0.0, "V339": None},
    "vesta_c": {"C1": 30.0, "C2": 30.0, "C13": 50.0, "C14": 25.0},
    "vesta_d": {"D1": 0.0, "D15": 0.0},
    "vesta_m": {"M1": None, "M2": None, "M4": None, "M9": None},
    "identity": {"id_01": None, "id_31": None, "id_38": None},
}

_PAYLOADS: dict[str, dict[str, Any]] = {
    "clearly_legit": _LEGIT_PAYLOAD,
    "obvious_fraud": _FRAUD_PAYLOAD,
}


def _print_response(label: str, body: dict[str, Any]) -> None:
    """Pretty-print one PredictionResponse to stdout.

    Highlights: score (4 decimal places), decision (BLOCK / ALLOW in
    caps), latency, model_version short prefix, degraded_mode flag if
    set, top-5 SHAP reasons with the feature name + contribution +
    direction.
    """
    score = float(body["score"])
    decision = str(body["decision"]).upper()
    latency_ms = float(body["latency_ms"])
    model_version = str(body["model_version"])[:12]
    degraded = bool(body.get("degraded_mode", False))
    top_reasons = body.get("top_reasons", [])

    click.echo(f"\n=== {label} ===")
    click.echo(f"  txn_id:       {body['txn_id']}")
    click.echo(f"  Score:        {score:.4f}")
    click.echo(f"  Decision:     {decision}")
    click.echo(f"  Latency:      {latency_ms:.1f} ms")
    click.echo(f"  Model:        {model_version}...")
    if degraded:
        click.echo("  Mode:         DEGRADED (Redis / Postgres unreachable)")
    click.echo("  Top reasons:")
    for reason in top_reasons[:5]:
        feature = str(reason["feature_name"])[:32]
        contrib = float(reason["contribution"])
        direction = str(reason["direction"])
        click.echo(f"    - {feature:32s}  {contrib:+.4f}  {direction}")


@click.command(help="Drive two demo transactions through the live /predict API.")
@click.option(
    "--api-url",
    default="http://localhost:8000",
    show_default=True,
    help="Base URL of the running fraud-engine API.",
)
@click.option(
    "--timeout",
    default=5.0,
    show_default=True,
    type=float,
    help="HTTP request timeout in seconds.",
)
def main(api_url: str, timeout: float) -> None:
    """5-second demo entry point."""
    click.echo(f"Fraud Detection Engine — demo against {api_url}")
    click.echo("=" * 60)

    for label, payload in _PAYLOADS.items():
        try:
            response = httpx.post(f"{api_url}/predict", json=payload, timeout=timeout)
        except httpx.ConnectError as exc:
            click.echo(
                f"\nERROR: API not reachable at {api_url}. "
                f"Start it with `make serve` (uvicorn on host) and "
                f"ensure Redis + Postgres are up via "
                f"`docker compose -f docker-compose.dev.yml up -d`.\n"
                f"Detail: {exc}",
                err=True,
            )
            sys.exit(1)
        except httpx.TimeoutException:
            click.echo(
                f"\nERROR: API timed out after {timeout}s on payload {label!r}. "
                f"The model artefacts may still be loading; retry in a moment.",
                err=True,
            )
            sys.exit(1)

        if response.status_code != _HTTP_OK:
            click.echo(
                f"\nERROR: API returned HTTP {response.status_code} on payload {label!r}.\n"
                f"Response body: {response.text}",
                err=True,
            )
            sys.exit(1)

        _print_response(label, response.json())

    click.echo("\n" + "=" * 60)
    click.echo("Demo complete. Both predictions returned successfully.")


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover — Click entrypoint
