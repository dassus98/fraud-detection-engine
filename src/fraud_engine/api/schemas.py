"""Pydantic v2 request and response models for the fraud-scoring API.

Sprint 5 prompt 5.1.a: the typed contract between API clients and the
fraud-detection service. Every field carries `description=` and
`examples=[...]` so the generated OpenAPI spec at `/docs` is
self-documenting and stable across Sprint 5 prompts.

Business rationale:
    Sprint 5 stands up a single high-stakes endpoint (POST /score)
    that accepts a transaction and returns a block/allow decision plus
    an interpretability trail. The contract must be (a) explicit
    enough for a senior reviewer reading the OpenAPI page to
    understand the business intent of every field, and (b) generous
    enough that Vesta-anonymised columns (V1..V339, C1..C14, D1..D15,
    M1..M9, id_01..id_38) flow through without ~431 per-column
    declarations. The hybrid shape below — 18 explicit business-value
    fields plus five regex-validated group-dicts — is the only
    defensible answer at this scale.

    Pinning the contract before any HTTP route, Redis wiring, or
    SHAP integration ships means every later 5.x prompt builds
    against a stable typed surface. The four schemas (`TransactionRequest`,
    `PredictionResponse`, `HealthResponse`, `ReadyResponse`) plus the
    two sub-models (`Reason`, `RequestMetadata`) are the public API.

Trade-offs considered:
    - **Hybrid (18 explicit + 5 group-dicts) over explicit-for-all
      (~431 fields) or pure permissive dict.** Explicit-for-all would
      produce ~700 LOC of boilerplate where every Vesta description is
      necessarily fabricated (Vesta does not publish per-column
      semantics) and every later 5.x change would touch hundreds of
      fields. Pure permissive dict (`transaction_data: dict[str, Any]`)
      would yield an empty OpenAPI surface — the spec's "every field
      has example value and description" mandate becomes vacuously
      true for a single catch-all. The hybrid keeps the schema
      human-readable (~360 LOC) while preserving full functionality:
      the 18 fields a fraud reviewer actually inspects (amount, card,
      address, email, device) get explicit `description` + `examples`;
      the V/C/D/M/identity columns flow through five typed dicts with
      regex-validated keys.

    - **`extra="ignore"` on `TransactionRequest` (NOT `"forbid"`).**
      Mirrors pandera's `strict=False` posture at the data-ingest
      boundary. A future V340 column or a transient debug header
      flows through harmlessly. `"forbid"` would force 422s on the
      kind of additive drift this contract is explicitly designed
      to absorb. `"allow"` would let arbitrary keys leak into
      `model_dump()` and pollute downstream feature computation.
      `"ignore"` drops unknowns silently — the right behaviour for
      a permissive ingest boundary.

    - **`extra="forbid"` on every response model.** Outputs are
      audit-traceable; surprises in a response body are the kind of
      bug that erodes trust in the API. Forbidding extras forces
      every Sprint 5.x change that adds a response field to update
      the schema explicitly.

    - **`frozen=True` everywhere.** Both requests and responses are
      conceptually immutable post-construction. Accidental mutation
      in the feature-pipeline / SHAP layers would silently change
      the value of a logged record; freezing the models surfaces the
      bug at the assignment site instead.

    - **`TransactionDT: int` (seconds since the IEEE-CIS anchor)
      rather than ISO datetime.** Every downstream module
      (`data.splits`, `features.tier1_basic`, the Tier-2 velocity
      window logic) already speaks integer seconds. Converting at
      the API boundary would require round-tripping through the
      anchor (`Settings.transaction_dt_anchor_iso`) to recover the
      native representation — strictly worse than carrying the
      native form. Gateway can convert ISO ↔ int if a client wants
      to send ISO.

    - **`decision: Literal["block", "allow"]` (binary, no
      "review").** A three-way decision is a deliberate Sprint 5.x
      feature (it requires an analyst-queue surface and a separate
      threshold band); 5.1.a stays binary so the
      "score >= Settings.decision_threshold" rule is unambiguous.

    - **`model_version: str` left opaque.** No project convention
      yet for whether the version is an MLflow run_id, a SHA-256
      `content_hash` of the joblib bytes, or a semantic identifier.
      Documented inline as "treat as opaque"; later 5.x prompt
      picks one canonical form.

    - **`top_reasons: list[Reason]` capped at `max_length=10`.** The
      SHAP TreeExplainer typically returns one contribution per
      feature (~743 contributions for Model A). Capping to top-10
      keeps the response body tight and matches the analyst-review
      ergonomic — beyond ~10 reasons, the audit trail becomes noise.

    - **Five group-dicts (V/C/D/M/identity) over one mega-dict.**
      The IEEE-CIS schema (`schemas/raw.py`) already groups V vs C
      vs D vs M vs identity at the data-pipeline boundary via regex
      column-groups. Mirroring that grouping at the API layer keeps
      the contract aligned with upstream truth — a consumer browsing
      `/docs` sees the same five buckets they'd see in any pandera
      schema dump. The alternative (one `engineered: dict[str, Any]`
      catch-all) loses the per-group type discipline (V/C/D are
      float, M is string T/F or M0/M1/M2, identity is mixed) and
      makes regex validation noisier.

    - **No pandera schema duplication.** Pandera remains the strict
      DataFrame-validation boundary at the data layer; the Pydantic
      schema is the API-layer contract. Each carries the constraints
      relevant to its layer; the two are kept in sync by mirroring
      the closed-set Literal aliases (ProductCD, card4, card6) but
      do NOT duplicate per-column nullability or numeric ranges.

Module surface (re-exported from `fraud_engine.api`):
    - TransactionRequest
    - PredictionResponse
    - HealthResponse
    - ReadyResponse
    - Reason (sub-model of PredictionResponse)
    - RequestMetadata (sub-model of TransactionRequest)
    - DecisionLiteral, DependencyStatusLiteral, ProductCodeLiteral,
      Card4Literal, Card6Literal, ReasonDirectionLiteral,
      HealthStatusLiteral, ReadyStatusLiteral (Literal aliases)

Cross-references:
    - `src/fraud_engine/schemas/raw.py` — IEEE-CIS raw column groups
      and closed-set value lists this schema mirrors.
    - `src/fraud_engine/config/settings.py:88-115` — cost defaults +
      `decision_threshold` (post-Sprint-4.4: 0.080) the response's
      `decision` field is computed against (server-side, not on the
      schema itself).
    - `src/fraud_engine/utils/logging.py` — `@log_call` decorator's
      `time.perf_counter() * 1000.0` pattern; `latency_ms` mirrors it.
    - `CLAUDE.md` §1 (24% identity coverage), §3 (FastAPI + Redis +
      Postgres + SHAP + shadow + logging stack), §8 (cost defaults +
      sensitivity stability rule).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final, Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------
# Literal aliases — give Pydantic + OpenAPI a tidy `enum` surface.
# Mirror the closed-set values from src/fraud_engine/schemas/raw.py so
# the API contract is grounded in the same upstream truth as the
# pandera DataFrame schema.
# ---------------------------------------------------------------------

ProductCodeLiteral = Literal["C", "H", "R", "S", "W"]
Card4Literal = Literal["american express", "discover", "mastercard", "visa"]
Card6Literal = Literal["charge card", "credit", "debit", "debit or credit"]
DecisionLiteral = Literal["block", "allow"]
DependencyStatusLiteral = Literal["ok", "degraded", "unreachable"]
ReasonDirectionLiteral = Literal["increases_risk", "decreases_risk"]
HealthStatusLiteral = Literal["ok"]
ReadyStatusLiteral = Literal["ready", "not_ready"]


# ---------------------------------------------------------------------
# Group-dict key regex patterns. These mirror the column-group regex
# conventions in `src/fraud_engine/schemas/raw.py` so a malformed key
# (e.g. V0 — V is 1-indexed; C99 — max is C14) fails loudly at the
# API boundary instead of silently flowing into feature computation.
# ---------------------------------------------------------------------

_VESTA_V_KEY_RE: Final[str] = r"^V\d{1,3}$"
_VESTA_C_KEY_RE: Final[str] = r"^C\d{1,2}$"
_VESTA_D_KEY_RE: Final[str] = r"^D\d{1,2}$"
_VESTA_M_KEY_RE: Final[str] = r"^M[1-9]$"
_IDENTITY_KEY_RE: Final[str] = r"^id_(0[1-9]|[1-3][0-9])$"

# Numeric upper bounds for V/C/D groups. Validating `^V\d{1,3}$` only
# catches V0 (1-indexed) and V1000+; V340..V999 would still pass the
# regex. The model-validator rejects values strictly above these caps
# so the contract matches the IEEE-CIS column count exactly.
_VESTA_V_MAX: Final[int] = 339
_VESTA_C_MAX: Final[int] = 14
_VESTA_D_MAX: Final[int] = 15
_IDENTITY_MAX: Final[int] = 38

# Compact `top_reasons` cap. SHAP TreeExplainer typically returns one
# contribution per feature (~743 for Model A); top-10 keeps the
# response body tight and matches the analyst-review ergonomic.
_TOP_REASONS_MAX: Final[int] = 10

# String length caps, defensively bounded so a malformed payload can't
# blow up the body size. Mirrors raw.py's nullability-and-shape posture.
_CLIENT_ID_MAX_LEN: Final[int] = 64
_EMAIL_DOMAIN_MAX_LEN: Final[int] = 64
_DEVICE_TYPE_MAX_LEN: Final[int] = 32
_DEVICE_INFO_MAX_LEN: Final[int] = 128
_FEATURE_NAME_MAX_LEN: Final[int] = 128
_MODEL_VERSION_MAX_LEN: Final[int] = 128
_SERVICE_NAME_MAX_LEN: Final[int] = 64
_SERVICE_VERSION_MAX_LEN: Final[int] = 32


# ---------------------------------------------------------------------
# Sub-models.
# ---------------------------------------------------------------------


class RequestMetadata(BaseModel):
    """Per-request bookkeeping flowed through the scoring pipeline.

    Business rationale:
        Sprint 5's API will bind `request_id` to a structlog ContextVar
        (`utils.logging.bind_request_id`) so every log record from
        feature lookup → inference → SHAP → Postgres write carries
        the same correlation id. Clients may forward an upstream
        gateway's `X-Request-ID`; if absent, the API generates a UUID4
        at ingest. The metadata block keeps that surface explicit on
        the request body rather than relying on header magic only —
        synchronous batch callers can construct a fully-deterministic
        request without HTTP plumbing, and test fixtures need not
        simulate header injection.

    Trade-offs considered:
        - Header-only `X-Request-ID` was the alternative. Including
          it on the body lets test fixtures and batch callers reuse
          the same scoring path without HTTP plumbing.
        - `client_id` is plain `str`, not `SecretStr` — it's an
          opaque identifier (e.g. "wealthsimple-prod"), not a
          credential. Authorisation lives at the gateway.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    request_id: UUID | None = Field(
        default=None,
        description=(
            "Optional client-supplied correlation id. If None, the API "
            "generates a fresh UUID4 at ingest time and binds it to "
            "the structlog request_id contextvar. Echoed back as "
            "PredictionResponse.request_id so a caller can correlate "
            "request → response → server-side audit log."
        ),
        examples=[
            "11111111-2222-4333-8444-555555555555",
            None,
        ],
    )
    client_id: str | None = Field(
        default=None,
        max_length=_CLIENT_ID_MAX_LEN,
        description=(
            "Opaque client identifier (e.g. 'wealthsimple-prod' or "
            "'internal-shadow'). Logged on every prediction; never "
            "used for authorisation (auth is handled at the gateway). "
            "Capped at 64 chars defensively."
        ),
        examples=["wealthsimple-prod", "internal-shadow", None],
    )


class Reason(BaseModel):
    """One row of the SHAP-derived `top_reasons` array.

    Business rationale:
        Production fraud decisions need an interpretability trail.
        The reviewer who blocks a customer's transaction must be able
        to cite "high amount + new device + uncommon merchant" — not
        "the model said 0.94". Each `Reason` is one feature's
        contribution to the model's log-odds. Sprint 5.x populates
        these via TreeExplainer; this prompt only fixes the shape so
        callers can deserialise the response unchanged.

    Trade-offs considered:
        - `feature_name` is the raw feature name (e.g.
          'card1_fraud_v_ewm_lambda_0.05'). A separate human-readable
          mapping is Sprint 5.x's reason-code work; keeping the raw
          name here lets a consumer cross-reference MLflow's
          feature-importance plot without a second lookup.
        - `direction: Literal["increases_risk", "decreases_risk"]` is
          easier to read than the signed `contribution` alone. The
          float magnitude is preserved for callers who want to rank
          reasons by absolute contribution.
        - No "neutral" direction Literal — a SHAP contribution of
          exactly 0.0 is vanishingly rare in practice and would be
          uninformative as a reason; if it occurs, the populating
          code in Sprint 5.x should drop the row rather than emit
          an ambiguous direction.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_name: str = Field(
        min_length=1,
        max_length=_FEATURE_NAME_MAX_LEN,
        description=(
            "Raw feature name from the trained pipeline (e.g. "
            "'tier1_amount_log', 'card1_fraud_v_ewm_lambda_0.05'). "
            "Maps to the column names emitted by the Tier-1..Tier-5 "
            "feature pipeline. Capped at 128 chars defensively."
        ),
        examples=[
            "tier1_amount_log",
            "card1_fraud_v_ewm_lambda_0.05",
            "P_emaildomain_is_free",
        ],
    )
    contribution: float = Field(
        description=(
            "Signed SHAP contribution to the model's log-odds. "
            "Positive raises the score; negative lowers it. Magnitude "
            "is on the log-odds scale, NOT a probability — consumers "
            "comparing reasons across requests should rank by "
            "absolute value, not raw value."
        ),
        examples=[0.42, -0.15, 1.08],
    )
    direction: ReasonDirectionLiteral = Field(
        description=(
            "Plain-English sign of `contribution`. 'increases_risk' "
            "iff contribution > 0; 'decreases_risk' iff contribution "
            "< 0. Sprint 5.x's SHAP-population code derives this from "
            "the SHAP value's sign (with contribution == 0.0 dropped "
            "rather than emitted as a 'neutral' row)."
        ),
        examples=["increases_risk", "decreases_risk"],
    )


# ---------------------------------------------------------------------
# TransactionRequest — the inbound contract.
# ---------------------------------------------------------------------


class TransactionRequest(BaseModel):
    """Request body for POST /score.

    Hybrid shape: 18 explicit business-meaningful fields + five typed
    group-dicts (V/C/D/M/identity) carrying Vesta-anonymised columns.
    See module docstring for the rejected alternatives and the full
    trade-off discussion.

    The schema mirrors the IEEE-CIS raw column conventions defined in
    `src/fraud_engine/schemas/raw.py` (closed-set values for ProductCD,
    card4, card6; nullability matching pandera; regex key constraints
    on the group-dicts) but is **NOT** a duplicate of pandera's
    DataFrameSchema. Pandera remains the strict-validation boundary at
    the data layer; this schema is the API-layer contract.
    """

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        str_strip_whitespace=True,
    )

    # --- request bookkeeping ----------------------------------------
    metadata: RequestMetadata = Field(
        default_factory=RequestMetadata,
        description=(
            "Per-request correlation id and client identifier. See "
            "RequestMetadata for the contract. Defaulted so simple "
            "callers can omit it entirely; the API will generate a "
            "fresh request_id at ingest."
        ),
        examples=[
            {"request_id": None, "client_id": "wealthsimple-prod"},
            {"request_id": None, "client_id": None},
        ],
    )

    # --- core transaction (5) ---------------------------------------
    TransactionID: int = Field(
        ge=0,
        description=(
            "IEEE-CIS transaction id (unique per row in training "
            "data). For production scoring this is the upstream "
            "system's txn primary key. Echoed back as `txn_id` on "
            "PredictionResponse so callers can correlate."
        ),
        examples=[2987000, 3577540],
    )
    TransactionDT: int = Field(
        ge=0,
        description=(
            "Seconds since the IEEE-CIS anchor "
            "(Settings.transaction_dt_anchor_iso = "
            "'2017-12-01T00:00:00+00:00'). Integer seconds, NOT an "
            "ISO datetime — every downstream module "
            "(data.splits, features.tier1_basic, the Tier-2 velocity "
            "window logic) speaks this representation natively. "
            "Convert at the gateway if a client wants to send ISO."
        ),
        examples=[86400, 13046400],
    )
    TransactionAmt: float = Field(
        gt=0.0,
        description=(
            "Transaction amount in USD. Strictly positive — refunds "
            "and chargebacks are out of scope for the scoring API. "
            "Tier-1's `log_amount` and `amount_decile` consume this "
            "field directly."
        ),
        examples=[59.95, 1499.00, 19.99],
    )
    ProductCD: ProductCodeLiteral = Field(
        description=(
            "IEEE-CIS product category. Closed set: C, H, R, S, W. "
            "Mirrors the constraint declared in pandera's "
            "TransactionSchema. Drives Tier-3's per-product target "
            "encoding."
        ),
        examples=["W", "C", "H"],
    )
    card1: int = Field(
        ge=0,
        description=(
            "Primary card identifier (always populated per IEEE-CIS "
            "docs). Used as the entity key for Tier-2 velocity, "
            "Tier-3 behavioural, Tier-4 EWM-decay, and Tier-5 graph "
            "features."
        ),
        examples=[13926, 2755],
    )

    # --- card metadata (5) ------------------------------------------
    card2: float | None = Field(
        default=None,
        description=(
            "IEEE-CIS card2 (anonymised numeric card attribute). "
            "Nullable — pandera reports ~1.5% missing on the training "
            "set."
        ),
        examples=[490.0, None],
    )
    card3: float | None = Field(
        default=None,
        description="IEEE-CIS card3 (anonymised numeric card attribute). Nullable.",
        examples=[150.0, None],
    )
    card4: Card4Literal | None = Field(
        default=None,
        description=(
            "Card brand. Closed set when present: 'american express', "
            "'discover', 'mastercard', 'visa'. Nullable to match "
            "pandera's TransactionSchema."
        ),
        examples=["visa", "mastercard", None],
    )
    card5: float | None = Field(
        default=None,
        description="IEEE-CIS card5 (anonymised numeric card attribute). Nullable.",
        examples=[226.0, None],
    )
    card6: Card6Literal | None = Field(
        default=None,
        description=(
            "Card type. Closed set when present: 'charge card', "
            "'credit', 'debit', 'debit or credit'. Nullable."
        ),
        examples=["credit", "debit", None],
    )

    # --- address / distance (4) -------------------------------------
    addr1: float | None = Field(
        default=None,
        description=(
            "Billing address region identifier (anonymised numeric). "
            "Used as an entity key in Tier-2 / Tier-5 features. "
            "Nullable."
        ),
        examples=[315.0, None],
    )
    addr2: float | None = Field(
        default=None,
        description="Billing address country identifier (anonymised numeric). Nullable.",
        examples=[87.0, None],
    )
    dist1: float | None = Field(
        default=None,
        description=(
            "IEEE-CIS dist1 (distance metric, anonymised). Nullable. "
            "Often missing — the model is trained to handle absence "
            "via the matching `is_null_dist1` indicator feature."
        ),
        examples=[19.0, None],
    )
    dist2: float | None = Field(
        default=None,
        description="IEEE-CIS dist2 (distance metric, anonymised). Frequently null.",
        examples=[None, 87.5],
    )

    # --- email + identity (4) ---------------------------------------
    P_emaildomain: str | None = Field(
        default=None,
        max_length=_EMAIL_DOMAIN_MAX_LEN,
        description=(
            "Purchaser email domain (free-form, e.g. 'gmail.com'). "
            "Lowercased on ingest via `_lowercase_email_domains` to "
            "match the Tier-1 EmailDomainExtractor contract. Nullable."
        ),
        examples=["gmail.com", "yahoo.com", None],
    )
    R_emaildomain: str | None = Field(
        default=None,
        max_length=_EMAIL_DOMAIN_MAX_LEN,
        description=(
            "Recipient email domain (free-form, often null per "
            "IEEE-CIS coverage). Lowercased on ingest."
        ),
        examples=["gmail.com", None],
    )
    DeviceType: str | None = Field(
        default=None,
        max_length=_DEVICE_TYPE_MAX_LEN,
        description=(
            "Identity-table DeviceType. ~24% coverage on IEEE-CIS "
            "(CLAUDE.md §1) — the model is trained to function "
            "without it via the `is_null_DeviceType` indicator. "
            "Free-form string; common values: 'mobile', 'desktop'."
        ),
        examples=["mobile", "desktop", None],
    )
    DeviceInfo: str | None = Field(
        default=None,
        max_length=_DEVICE_INFO_MAX_LEN,
        description=(
            "Identity-table DeviceInfo (free-form device string). "
            "~24% coverage. Capped at 128 chars defensively — "
            "user-agent-style strings rarely exceed this."
        ),
        examples=["SAMSUNG SM-G930V Build/NRD90M", "Windows", None],
    )

    # --- Vesta-anonymised group-dicts (4) + identity dict (1) ------
    vesta_v: dict[str, float | None] = Field(
        default_factory=dict,
        description=(
            "Vesta engineered V-features. Keys must match "
            r"`^V\d{1,3}$` and resolve to V1..V339 (Vesta does not "
            "publish per-column semantics — the model consumes them "
            "as anonymised numeric features). Values are floats or "
            "null."
        ),
        examples=[{"V1": 1.0, "V2": 1.0, "V137": -0.5, "V339": None}],
    )
    vesta_c: dict[str, float | None] = Field(
        default_factory=dict,
        description=(
            "Vesta engineered C-features (count-like). Keys must "
            r"match `^C\d{1,2}$` and resolve to C1..C14."
        ),
        examples=[{"C1": 1.0, "C2": 1.0, "C13": 0.0}],
    )
    vesta_d: dict[str, float | None] = Field(
        default_factory=dict,
        description=(
            "Vesta engineered D-features (delta-like). Keys must "
            r"match `^D\d{1,2}$` and resolve to D1..D15."
        ),
        examples=[{"D1": 14.0, "D15": None}],
    )
    vesta_m: dict[str, str | None] = Field(
        default_factory=dict,
        description=(
            "Vesta engineered M-features (match flags). Keys must "
            r"match `^M[1-9]$` and resolve to M1..M9. Values are "
            "'T'/'F' for the binary flags; M4 is the three-way "
            "match column with values 'M0'/'M1'/'M2'."
        ),
        examples=[{"M1": "T", "M2": "T", "M4": "M0", "M9": None}],
    )
    identity: dict[str, float | str | None] = Field(
        default_factory=dict,
        description=(
            "IEEE-CIS identity columns id_01..id_38. Keys must match "
            r"`^id_(0[1-9]|[1-3][0-9])$`. Values are mixed numeric or "
            "categorical per `schemas.raw._IDENTITY_NUMERIC_COLS` / "
            "`_IDENTITY_OBJECT_COLS`. Nullable; ~24% coverage on "
            "IEEE-CIS (CLAUDE.md §1)."
        ),
        examples=[
            {
                "id_01": -5.0,
                "id_31": "samsung browser 6.2",
                "id_38": "T",
            }
        ],
    )

    # ---------------------------------------------------------------
    # Validators.
    # ---------------------------------------------------------------

    @field_validator("P_emaildomain", "R_emaildomain")
    @classmethod
    def _lowercase_email_domains(cls, value: str | None) -> str | None:
        """Lowercase email domains on ingest.

        Tier-1's `EmailDomainExtractor` (`features.tier1_basic`) expects
        lowercased domains. Coercing at the API boundary means
        downstream code never has to think about case. Pure
        normalisation, not validation — an empty string or a
        whitespace-only string would still pass; that's the cleaner's
        job, not the schema's.
        """
        return value.lower() if value is not None else None

    @model_validator(mode="after")
    def _validate_group_dict_keys(self) -> TransactionRequest:
        """Enforce key-shape regex + numeric-cap on every group-dict.

        Mirrors the regex column groups from `schemas.raw.py`. A key
        like `V0` (V is 1-indexed), `C99` (max is C14), or `id_99`
        (max is id_38) is a malformed payload that should fail loudly
        at the API boundary, not silently flow into feature
        computation.

        The regex catches the obvious shape errors (V0, V1000+,
        C100+, M0); a separate numeric-cap check covers the case
        where the regex matches but the numeric component exceeds
        the IEEE-CIS column count (e.g. V340 passes `^V\\d{1,3}$`
        but is not a real Vesta column).

        Raises:
            ValueError: If any key in any group-dict fails its regex
                or exceeds its numeric cap.
        """
        _check_group_dict_keys("vesta_v", self.vesta_v, _VESTA_V_KEY_RE, _VESTA_V_MAX)
        _check_group_dict_keys("vesta_c", self.vesta_c, _VESTA_C_KEY_RE, _VESTA_C_MAX)
        _check_group_dict_keys("vesta_d", self.vesta_d, _VESTA_D_KEY_RE, _VESTA_D_MAX)
        # M is M1..M9 only — single digit; the regex already caps at
        # 9. No numeric-suffix check needed (the regex IS the cap).
        _check_group_dict_keys("vesta_m", self.vesta_m, _VESTA_M_KEY_RE, None)
        _check_group_dict_keys("identity", self.identity, _IDENTITY_KEY_RE, _IDENTITY_MAX)
        return self


def _check_group_dict_keys(
    field_name: str,
    payload: Mapping[str, object],
    pattern: str,
    numeric_max: int | None,
) -> None:
    """Validate a group-dict's keys against a regex and optional cap.

    Args:
        field_name: The schema field name, used in the error message.
        payload: The dict whose keys we're validating.
        pattern: The regex every key must match (e.g. `^V\\d{1,3}$`).
        numeric_max: Optional inclusive upper bound on the numeric
            suffix (e.g. 339 for V columns). Pass None to skip the
            numeric-cap check (used for M which is single-digit).

    Raises:
        ValueError: With the offending key embedded in the message.
    """
    compiled = re.compile(pattern)
    digit_re = re.compile(r"\d+")
    for key in payload:
        if not compiled.fullmatch(key):
            raise ValueError(
                f"TransactionRequest.{field_name}: key {key!r} does not " f"match {pattern!r}"
            )
        if numeric_max is not None:
            digits = digit_re.search(key)
            # The regex above guarantees at least one digit run; a
            # missing match would be a code-level invariant violation.
            if digits is None:  # pragma: no cover — defensive
                raise ValueError(
                    f"TransactionRequest.{field_name}: key {key!r} " f"missing numeric component"
                )
            value = int(digits.group())
            if value < 1 or value > numeric_max:
                raise ValueError(
                    f"TransactionRequest.{field_name}: key {key!r} "
                    f"numeric component must be in [1, {numeric_max}]"
                )


# ---------------------------------------------------------------------
# PredictionResponse — the outbound contract.
# ---------------------------------------------------------------------


class PredictionResponse(BaseModel):
    """Response body for POST /score.

    Eight fields, all with explicit types and constraints. `decision`
    is derived server-side as `score >= Settings.decision_threshold`
    (post-Sprint-4.4: 0.080); the threshold itself is intentionally
    NOT carried on the response — consumers should not pin off it.
    `degraded_mode=True` signals Tier-1-only fallback (Sprint 5.2
    wires the Redis-fallback path that produces this state).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    txn_id: int = Field(
        ge=0,
        description=(
            "Echoed back from TransactionRequest.TransactionID. Lets "
            "callers correlate the response with their own txn "
            "primary key without re-parsing the request body."
        ),
        examples=[2987000],
    )
    request_id: UUID = Field(
        default_factory=uuid4,
        description=(
            "The correlation id bound for this request — either "
            "client-supplied via metadata.request_id or generated "
            "server-side at ingest. Logged on every record in the "
            "request's trail (feature lookup → inference → SHAP → "
            "Postgres write) so an operator can `grep` a UUID and "
            "see the full lineage of a single prediction."
        ),
        examples=["11111111-2222-4333-8444-555555555555"],
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated fraud probability in [0, 1]. Calibrated via "
            "isotonic regression in Sprint 3.3.c — interpret as a "
            "probability, not just a ranking score. The Bayes-decision "
            "argument in ADR 0003 depends on this calibration."
        ),
        examples=[0.0273, 0.9412, 0.5001],
    )
    decision: DecisionLiteral = Field(
        description=(
            "Server-side decision: 'block' iff score >= "
            "Settings.decision_threshold (post-Sprint-4.4: 0.080), "
            "else 'allow'. The threshold is NOT on the response — "
            "consumers must not pin off it. A three-way "
            "{'block','review','allow'} decision is a deliberate "
            "Sprint 5.x feature requiring an analyst-queue surface."
        ),
        examples=["allow", "block"],
    )
    top_reasons: list[Reason] = Field(
        default_factory=list,
        max_length=_TOP_REASONS_MAX,
        description=(
            "SHAP-derived top contributing features, ordered by "
            "abs(contribution) descending. Capped at 10 — beyond "
            "that, the audit trail becomes noise. Empty list iff "
            "`degraded_mode=True` and SHAP is unavailable, or if the "
            "score is at the population mean (no salient drivers)."
        ),
        examples=[
            [
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
            ]
        ],
    )
    latency_ms: float = Field(
        ge=0.0,
        description=(
            "End-to-end server-side latency for this request, "
            "measured via `time.perf_counter() * 1000.0` (matches "
            "the @log_call decorator pattern in utils.logging). "
            "Includes feature lookup, inference, SHAP, and "
            "serialisation. NOT the network round-trip — gateway "
            "timing is observed externally. Budget: <100ms P95 per "
            "CLAUDE.md §3."
        ),
        examples=[3.456, 47.21, 12.0],
    )
    model_version: str = Field(
        min_length=1,
        max_length=_MODEL_VERSION_MAX_LEN,
        description=(
            "Identifier of the model that produced this score. The "
            "serving layer (Sprint 5.x) populates this from the "
            "loaded joblib's manifest content_hash (SHA-256 hex of "
            "the joblib bytes — see "
            "`models.lightgbm_model._build_manifest`). MLflow run_ids "
            "and semantic-ish identifiers are valid alternatives at "
            "deploy-time discretion. Format is NOT yet fixed at the "
            "API contract level — clients should treat it as opaque."
        ),
        examples=[
            "a3f8c2d9b1e7c5d4f8a2b6e9c1d5f7a3b8e9c2d4f6a8b1e3c5d7f9a2b4c6d8e1",  # pragma: allowlist secret — synthetic SHA-256-shaped fake for OpenAPI docs
            "sprint3_a_calib_isotonic",
            "mlflow:run/9b1e7f3c8d2a",
        ],
    )
    degraded_mode: bool = Field(
        default=False,
        description=(
            "True iff the API fell back to Tier-1-only features "
            "because Redis was unreachable (Sprint 5.2 wires the "
            "fallback path). When True, the score still respects "
            "the threshold but `top_reasons` may be empty and the "
            "risk estimate is necessarily noisier — a downstream "
            "operator may want to manually review predictions in "
            "this state."
        ),
        examples=[False, True],
    )


# ---------------------------------------------------------------------
# HealthResponse — liveness probe payload.
# ---------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for GET /health (liveness).

    Liveness is binary: "the process is up and the import graph is
    intact". Readiness lives on a separate endpoint (ReadyResponse) so
    Kubernetes (or any orchestrator) can route on the right signal:
    liveness drives restart, readiness drives traffic admission.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: HealthStatusLiteral = Field(
        default="ok",
        description=(
            "Liveness sentinel. Always 'ok' if this handler runs. "
            "The HTTP status code (200 vs 503) is what an "
            "orchestrator probes; this body is for human inspection."
        ),
        examples=["ok"],
    )
    service_name: str = Field(
        default="fraud-engine-api",
        min_length=1,
        max_length=_SERVICE_NAME_MAX_LEN,
        description=(
            "Static service identifier — same value across all "
            "deployments of this image. Useful when multiple "
            "FastAPI services share a gateway."
        ),
        examples=["fraud-engine-api"],
    )
    version: str = Field(
        min_length=1,
        max_length=_SERVICE_VERSION_MAX_LEN,
        description=(
            "Service version string. Sprint 5.1.b populates from "
            "the package version (pyproject.toml: 0.1.0). NOT the "
            "model version — that ships on "
            "PredictionResponse.model_version."
        ),
        examples=["0.1.0"],
    )


# ---------------------------------------------------------------------
# ReadyResponse — readiness probe payload.
# ---------------------------------------------------------------------


class ReadyResponse(BaseModel):
    """Response body for GET /ready (readiness).

    Readiness is per-dependency. The `checks` map keys each runtime
    dependency to its current status; `details` carries an optional
    human-readable error message per failed/degraded dependency. The
    overall `status` is 'ready' iff every check is 'ok' — enforced
    by the model_validator below.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ReadyStatusLiteral = Field(
        description=(
            "Aggregate readiness. 'ready' iff every value in "
            "`checks` is 'ok'; 'not_ready' otherwise. The "
            "orchestrator should stop routing traffic on "
            "'not_ready' (handler returns 503)."
        ),
        examples=["ready", "not_ready"],
    )
    checks: dict[str, DependencyStatusLiteral] = Field(
        description=(
            "Per-dependency status. Canonical keys for Sprint 5: "
            "'redis' (online feature store), 'postgres' (audit log), "
            "'model' (joblib loaded + warmed). Sprint 5.x may add "
            "'mlflow' if model lookup happens at runtime."
        ),
        examples=[
            {"redis": "ok", "postgres": "ok", "model": "ok"},
            {"redis": "unreachable", "postgres": "ok", "model": "ok"},
        ],
    )
    details: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional human-readable error messages per failed or "
            "degraded check. Keys are a subset of `checks`. Typical "
            "Sprint 5.x usage: "
            "`{'redis': 'connection refused', 'postgres': 'pool timeout'}`. "
            "Empty when all checks are 'ok'."
        ),
        examples=[
            {},
            {"redis": "connection refused after 3 retries"},
        ],
    )

    @model_validator(mode="after")
    def _status_consistent_with_checks(self) -> ReadyResponse:
        """Enforce status ↔ checks consistency.

        The whole point of the readiness probe is the aggregate
        signal. Allowing `status='ready'` with a failed check (or
        `status='not_ready'` with all-ok checks) would let an
        operator misconfigure the response and silently route
        traffic to a broken backend. The model-validator catches
        this at construction time so a bug in the response-builder
        can't escape into production.

        Raises:
            ValueError: If status='ready' but any check is not 'ok',
                or if status='not_ready' but every check is 'ok'.
        """
        all_ok = all(v == "ok" for v in self.checks.values())
        if self.status == "ready" and not all_ok:
            raise ValueError(
                "ReadyResponse: status='ready' but at least one check "
                "is not 'ok' — inconsistent payload"
            )
        if self.status == "not_ready" and all_ok and len(self.checks) > 0:
            raise ValueError(
                "ReadyResponse: status='not_ready' but every check is "
                "'ok' — inconsistent payload"
            )
        return self


__all__ = [
    "Card4Literal",
    "Card6Literal",
    "DecisionLiteral",
    "DependencyStatusLiteral",
    "HealthResponse",
    "HealthStatusLiteral",
    "PredictionResponse",
    "ProductCodeLiteral",
    "ReadyResponse",
    "ReadyStatusLiteral",
    "Reason",
    "ReasonDirectionLiteral",
    "RequestMetadata",
    "TransactionRequest",
]
