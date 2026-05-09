"""Tests for `fraud_engine.api.schemas`.

Sprint 5 prompt 5.1.a verification surface.

Business rationale:
    The API schemas are the single typed contract between Sprint 5's
    serving layer and every client (real, shadow, batch). A regression
    here — a relaxed type, a dropped validator, a missing OpenAPI
    metadata field — leaks into the production contract silently and
    forces every later 5.x prompt to absorb the breakage. The one
    place the contract is pinned is here.

Trade-offs considered:
    - Inline payload dicts over a parametrised fixture: the four
      schemas have asymmetric shapes (request vs. response vs. health
      vs. ready) so per-test dict literals read cleaner than a
      one-fixture-per-payload approach. Each test starts from
      `_VALID_TRANSACTION_PAYLOAD | {…override…}` to spell out which
      field is being exercised.
    - Hypothesis property-based tests on the group-dict regexes are
      tempting but out of scope for 5.1.a — the regex contract is
      already exercised by the negative-key tests below, and a
      Hypothesis sweep would inflate the test-fast runtime for
      marginal coverage gain.
    - The `TestOpenAPIMetadata` meta-test (every field has
      `description` and at least one `examples` entry) is the
      mechanical gate behind the spec's "every field has example
      value and description" mandate. Without it, drift on a future
      PR is invisible until a reviewer notices an empty cell on
      `/docs`.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from fraud_engine.api.schemas import (
    HealthResponse,
    PredictionResponse,
    ReadyResponse,
    Reason,
    RequestMetadata,
    TransactionRequest,
)

# ---------------------------------------------------------------------
# Module-level payload constants. Plain dict literals — never mutated
# in-place; tests construct overrides via `payload | {…}`.
# ---------------------------------------------------------------------


_VALID_TRANSACTION_PAYLOAD: dict[str, Any] = {
    "TransactionID": 2987000,
    "TransactionDT": 86400,
    "TransactionAmt": 59.95,
    "ProductCD": "W",
    "card1": 13926,
    "card2": 490.0,
    "card3": 150.0,
    "card4": "visa",
    "card5": 226.0,
    "card6": "credit",
    "addr1": 315.0,
    "addr2": 87.0,
    "dist1": 19.0,
    "dist2": None,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": None,
    "DeviceType": "mobile",
    "DeviceInfo": "SAMSUNG SM-G930V Build/NRD90M",
    "vesta_v": {"V1": 1.0, "V2": 1.0, "V137": -0.5, "V339": None},
    "vesta_c": {"C1": 1.0, "C13": 0.0},
    "vesta_d": {"D1": 14.0, "D15": None},
    "vesta_m": {"M1": "T", "M4": "M0", "M9": None},
    "identity": {"id_01": -5.0, "id_31": "samsung browser 6.2", "id_38": "T"},
}

_MIN_TRANSACTION_PAYLOAD: dict[str, Any] = {
    "TransactionID": 2987000,
    "TransactionDT": 86400,
    "TransactionAmt": 59.95,
    "ProductCD": "W",
    "card1": 13926,
}

_VALID_PREDICTION_PAYLOAD: dict[str, Any] = {
    "txn_id": 2987000,
    "request_id": str(uuid4()),
    "score": 0.0273,
    "decision": "allow",
    "top_reasons": [
        {
            "feature_name": "card1_fraud_v_ewm_lambda_0.05",
            "contribution": 0.42,
            "direction": "increases_risk",
        },
        {
            "feature_name": "tier1_amount_log",
            "contribution": -0.15,
            "direction": "decreases_risk",
        },
    ],
    "latency_ms": 3.456,
    # Synthetic SHA-256-shaped fake; flagged for detect-secrets per
    # the same convention as schemas.py's `model_version` example.
    "model_version": "a3f8c2d9b1e7c5d4f8a2b6e9c1d5f7a3b8e9c2d4f6a8b1e3c5d7f9a2b4c6d8e1",  # pragma: allowlist secret
    "degraded_mode": False,
}


# ---------------------------------------------------------------------
# TransactionRequest — valid payloads.
# ---------------------------------------------------------------------


class TestTransactionRequestValid:
    """Every `TransactionRequest` happy path."""

    def test_minimum_required_payload_validates(self) -> None:
        """The 5 core fields alone are enough; other fields default."""
        req = TransactionRequest(**_MIN_TRANSACTION_PAYLOAD)
        assert req.TransactionID == _MIN_TRANSACTION_PAYLOAD["TransactionID"]
        assert req.card2 is None
        assert req.vesta_v == {}
        assert req.metadata.request_id is None

    def test_full_payload_validates(self) -> None:
        """`_VALID_TRANSACTION_PAYLOAD` round-trips cleanly."""
        req = TransactionRequest(**_VALID_TRANSACTION_PAYLOAD)
        assert req.TransactionAmt == _VALID_TRANSACTION_PAYLOAD["TransactionAmt"]
        assert req.card4 == "visa"
        assert req.vesta_v["V137"] == -0.5
        assert req.identity["id_31"] == "samsung browser 6.2"

    def test_email_domains_lowercased(self) -> None:
        """`P_emaildomain` / `R_emaildomain` come back lowercased."""
        payload = _VALID_TRANSACTION_PAYLOAD | {
            "P_emaildomain": "GMAIL.COM",
            "R_emaildomain": "Yahoo.COM",
        }
        req = TransactionRequest(**payload)
        assert req.P_emaildomain == "gmail.com"
        assert req.R_emaildomain == "yahoo.com"

    def test_extras_silently_dropped(self) -> None:
        """`extra='ignore'` drops unknown top-level keys; no raise."""
        payload = _VALID_TRANSACTION_PAYLOAD | {"_debug_field": "x"}
        req = TransactionRequest(**payload)
        # Round-trip via `model_dump()` excludes the unknown key.
        assert "_debug_field" not in req.model_dump()

    def test_immutable_after_construction(self) -> None:
        """`frozen=True` blocks any post-construction mutation."""
        req = TransactionRequest(**_VALID_TRANSACTION_PAYLOAD)
        with pytest.raises(ValidationError):
            req.TransactionAmt = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------
# TransactionRequest — invalid payloads.
# ---------------------------------------------------------------------


class TestTransactionRequestInvalid:
    """Negative cases: every Field constraint + every regex gate."""

    @pytest.mark.parametrize("amount", [-1.0, 0.0])
    def test_non_positive_transaction_amount_raises(self, amount: float) -> None:
        """`TransactionAmt` must be strictly positive (`gt=0`)."""
        payload = _VALID_TRANSACTION_PAYLOAD | {"TransactionAmt": amount}
        with pytest.raises(ValidationError):
            TransactionRequest(**payload)

    def test_negative_transaction_dt_raises(self) -> None:
        """`TransactionDT` must be `ge=0` (seconds since anchor)."""
        payload = _VALID_TRANSACTION_PAYLOAD | {"TransactionDT": -1}
        with pytest.raises(ValidationError):
            TransactionRequest(**payload)

    def test_unknown_product_code_raises(self) -> None:
        """`ProductCD` is a closed Literal — 'Z' fails."""
        payload = _VALID_TRANSACTION_PAYLOAD | {"ProductCD": "Z"}
        with pytest.raises(ValidationError):
            TransactionRequest(**payload)

    def test_unknown_card4_brand_raises(self) -> None:
        """`card4='amex'` fails — canonical value is 'american express'."""
        payload = _VALID_TRANSACTION_PAYLOAD | {"card4": "amex"}
        with pytest.raises(ValidationError):
            TransactionRequest(**payload)

    def test_unknown_card6_type_raises(self) -> None:
        """`card6='prepaid'` is not in the closed Literal set."""
        payload = _VALID_TRANSACTION_PAYLOAD | {"card6": "prepaid"}
        with pytest.raises(ValidationError):
            TransactionRequest(**payload)

    def test_missing_required_field_raises(self) -> None:
        """Omitting `TransactionID` (a required field) raises."""
        payload = {k: v for k, v in _MIN_TRANSACTION_PAYLOAD.items() if k != "TransactionID"}
        with pytest.raises(ValidationError):
            TransactionRequest(**payload)

    @pytest.mark.parametrize(
        ("field", "bad_payload"),
        [
            ("vesta_v", {"V0": 1.0}),  # V is 1-indexed
            ("vesta_v", {"V340": 1.0}),  # numeric cap
            ("vesta_c", {"C99": 1.0}),  # exceeds C14
            ("vesta_c", {"C0": 1.0}),  # 1-indexed
            ("vesta_d", {"D99": 1.0}),  # exceeds D15
            ("vesta_m", {"M0": "T"}),  # 1-indexed (regex `^M[1-9]$`)
            ("identity", {"id_99": 1.0}),  # exceeds id_38
            ("identity", {"id_00": 1.0}),  # 1-indexed
        ],
    )
    def test_group_dict_bad_keys_raise(self, field: str, bad_payload: dict[str, Any]) -> None:
        """Every group-dict regex + numeric cap fails on malformed keys."""
        payload = _VALID_TRANSACTION_PAYLOAD | {field: bad_payload}
        with pytest.raises(ValidationError):
            TransactionRequest(**payload)


# ---------------------------------------------------------------------
# RequestMetadata.
# ---------------------------------------------------------------------


class TestRequestMetadata:
    """Per-request bookkeeping sub-model contract."""

    def test_metadata_defaults_to_empty(self) -> None:
        """Both fields default to None when nothing is supplied."""
        meta = RequestMetadata()
        assert meta.request_id is None
        assert meta.client_id is None

    def test_metadata_extra_forbid_raises(self) -> None:
        """`extra='forbid'`: extra key on metadata raises."""
        with pytest.raises(ValidationError):
            RequestMetadata(_unknown_field="x")  # type: ignore[call-arg]

    def test_metadata_client_id_max_length_64(self) -> None:
        """65-char `client_id` triggers `max_length=64`."""
        with pytest.raises(ValidationError):
            RequestMetadata(client_id="x" * 65)


# ---------------------------------------------------------------------
# Reason.
# ---------------------------------------------------------------------


class TestReason:
    """SHAP-derived reason sub-model contract."""

    def test_valid_reason_validates(self) -> None:
        reason = Reason(
            feature_name="tier1_amount_log",
            contribution=0.42,
            direction="increases_risk",
        )
        assert reason.contribution == 0.42

    def test_unknown_direction_raises(self) -> None:
        """`direction` is a closed Literal — 'neutral' fails."""
        with pytest.raises(ValidationError):
            Reason(
                feature_name="tier1_amount_log",
                contribution=0.0,
                direction="neutral",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------
# PredictionResponse — valid payloads.
# ---------------------------------------------------------------------


class TestPredictionResponseValid:
    """Every `PredictionResponse` happy path."""

    def test_full_payload_validates(self) -> None:
        resp = PredictionResponse(**_VALID_PREDICTION_PAYLOAD)
        assert resp.score == 0.0273
        assert resp.decision == "allow"
        assert len(resp.top_reasons) == 2

    def test_top_reasons_default_empty(self) -> None:
        """Omitting `top_reasons` yields `[]`."""
        payload = {k: v for k, v in _VALID_PREDICTION_PAYLOAD.items() if k != "top_reasons"}
        resp = PredictionResponse(**payload)
        assert resp.top_reasons == []

    def test_degraded_mode_default_false(self) -> None:
        """Omitting `degraded_mode` yields `False`."""
        payload = {k: v for k, v in _VALID_PREDICTION_PAYLOAD.items() if k != "degraded_mode"}
        resp = PredictionResponse(**payload)
        assert resp.degraded_mode is False


# ---------------------------------------------------------------------
# PredictionResponse — invalid payloads.
# ---------------------------------------------------------------------


class TestPredictionResponseInvalid:
    """Negative cases on every Field constraint."""

    @pytest.mark.parametrize("score", [-0.01, 1.01])
    def test_score_out_of_range_raises(self, score: float) -> None:
        """`score` must be in [0, 1]."""
        payload = _VALID_PREDICTION_PAYLOAD | {"score": score}
        with pytest.raises(ValidationError):
            PredictionResponse(**payload)

    def test_unknown_decision_raises(self) -> None:
        """`decision='review'` is not a 5.1.a Literal value."""
        payload = _VALID_PREDICTION_PAYLOAD | {"decision": "review"}
        with pytest.raises(ValidationError):
            PredictionResponse(**payload)

    def test_negative_latency_ms_raises(self) -> None:
        """`latency_ms` must be `ge=0`."""
        payload = _VALID_PREDICTION_PAYLOAD | {"latency_ms": -1.0}
        with pytest.raises(ValidationError):
            PredictionResponse(**payload)

    def test_top_reasons_overflow_raises(self) -> None:
        """`top_reasons` capped at 10; 11 entries fail `max_length`."""
        one_reason = _VALID_PREDICTION_PAYLOAD["top_reasons"][0]
        payload = _VALID_PREDICTION_PAYLOAD | {"top_reasons": [one_reason] * 11}
        with pytest.raises(ValidationError):
            PredictionResponse(**payload)

    def test_invalid_request_id_raises(self) -> None:
        """`request_id` must parse as a UUID."""
        payload = _VALID_PREDICTION_PAYLOAD | {"request_id": "not-a-uuid"}
        with pytest.raises(ValidationError):
            PredictionResponse(**payload)

    def test_missing_model_version_raises(self) -> None:
        """`model_version` is required; omitting it raises."""
        payload = {k: v for k, v in _VALID_PREDICTION_PAYLOAD.items() if k != "model_version"}
        with pytest.raises(ValidationError):
            PredictionResponse(**payload)


# ---------------------------------------------------------------------
# HealthResponse.
# ---------------------------------------------------------------------


class TestHealthResponse:
    """Liveness probe payload contract."""

    def test_default_status_is_ok(self) -> None:
        """`status` defaults to 'ok'; only `version` is required."""
        resp = HealthResponse(version="0.1.0")
        assert resp.status == "ok"
        assert resp.service_name == "fraud-engine-api"

    def test_status_must_be_literal_ok(self) -> None:
        """Any value other than 'ok' for `status` raises."""
        with pytest.raises(ValidationError):
            HealthResponse(status="degraded", version="0.1.0")  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# ReadyResponse.
# ---------------------------------------------------------------------


class TestReadyResponse:
    """Readiness probe payload + status-checks consistency rule."""

    def test_all_ok_yields_ready(self) -> None:
        resp = ReadyResponse(
            status="ready",
            checks={"redis": "ok", "postgres": "ok", "model": "ok"},
        )
        assert resp.status == "ready"
        assert resp.details == {}

    def test_one_unreachable_yields_not_ready(self) -> None:
        resp = ReadyResponse(
            status="not_ready",
            checks={"redis": "unreachable", "postgres": "ok", "model": "ok"},
            details={"redis": "connection refused after 3 retries"},
        )
        assert resp.status == "not_ready"
        assert resp.details["redis"].startswith("connection refused")

    def test_status_ready_with_unreachable_check_raises(self) -> None:
        """Model-validator catches inconsistent payload (ready + bad check)."""
        with pytest.raises(ValidationError):
            ReadyResponse(
                status="ready",
                checks={"redis": "unreachable", "postgres": "ok", "model": "ok"},
            )

    def test_status_not_ready_with_all_ok_raises(self) -> None:
        """Model-validator catches the inverse (not_ready + all-ok checks)."""
        with pytest.raises(ValidationError):
            ReadyResponse(
                status="not_ready",
                checks={"redis": "ok", "postgres": "ok", "model": "ok"},
            )

    def test_details_optional_when_all_ok(self) -> None:
        """Empty `details` is the canonical shape when status='ready'."""
        resp = ReadyResponse(
            status="ready",
            checks={"redis": "ok", "postgres": "ok"},
        )
        assert resp.details == {}


# ---------------------------------------------------------------------
# Round-trip (model_dump -> reconstruct).
# ---------------------------------------------------------------------


class TestRoundTrip:
    """Every public schema must be model_dump → __init__ stable."""

    def test_transaction_request_dump_load_stable(self) -> None:
        original = TransactionRequest(**_VALID_TRANSACTION_PAYLOAD)
        roundtrip = TransactionRequest(**original.model_dump())
        assert roundtrip == original

    def test_prediction_response_dump_load_stable(self) -> None:
        original = PredictionResponse(**_VALID_PREDICTION_PAYLOAD)
        roundtrip = PredictionResponse(**original.model_dump())
        assert roundtrip == original

    def test_health_response_dump_load_stable(self) -> None:
        original = HealthResponse(version="0.1.0")
        roundtrip = HealthResponse(**original.model_dump())
        assert roundtrip == original

    def test_ready_response_dump_load_stable(self) -> None:
        original = ReadyResponse(
            status="ready",
            checks={"redis": "ok", "postgres": "ok", "model": "ok"},
        )
        roundtrip = ReadyResponse(**original.model_dump())
        assert roundtrip == original


# ---------------------------------------------------------------------
# OpenAPI metadata — the spec-enforcement meta-test.
# ---------------------------------------------------------------------

# Every public schema + sub-model. Add new schemas here AND to the
# `__all__` block in `src/fraud_engine/api/schemas.py` whenever Sprint
# 5.x extends the contract.
_PUBLIC_SCHEMAS: tuple[type[BaseModel], ...] = (
    TransactionRequest,
    PredictionResponse,
    HealthResponse,
    ReadyResponse,
    Reason,
    RequestMetadata,
)


class TestOpenAPIMetadata:
    """Mechanical gate: every field has `description=` + `examples=[...]`.

    The spec mandate ('every field has example value and description
    for OpenAPI') becomes a no-op without a programmatic check —
    drift on a future PR would be invisible until a reviewer
    notices an empty cell on `/docs`. The two parametrised tests
    below catch that drift at test-fast time.
    """

    @pytest.mark.parametrize("model_cls", _PUBLIC_SCHEMAS)
    def test_every_field_has_description(self, model_cls: type[BaseModel]) -> None:
        """Every field on every public schema must carry a non-empty description."""
        for name, field_info in model_cls.model_fields.items():
            description = field_info.description
            assert (
                description is not None and len(description) > 0
            ), f"{model_cls.__name__}.{name} missing `description`"

    @pytest.mark.parametrize("model_cls", _PUBLIC_SCHEMAS)
    def test_every_field_has_at_least_one_example(self, model_cls: type[BaseModel]) -> None:
        """Every field must carry `examples=[...]` with at least one entry."""
        for name, field_info in model_cls.model_fields.items():
            examples = field_info.examples
            assert (
                examples is not None and len(examples) >= 1
            ), f"{model_cls.__name__}.{name} missing `examples`"

    def test_request_id_example_is_uuid_parseable(self) -> None:
        """The example for `RequestMetadata.request_id` must be a valid UUID."""
        examples = RequestMetadata.model_fields["request_id"].examples or []
        non_null = [e for e in examples if e is not None]
        assert non_null, "request_id should have at least one non-None UUID example"
        # The first non-None example must round-trip through UUID().
        UUID(str(non_null[0]))
