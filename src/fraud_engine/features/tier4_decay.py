"""Tier-4 exponential-decay velocity (EWM): O(1) per-event running-state generator.

Smooths the window-boundary cliffs that Tier-2 `VelocityCounter` has
by construction. Where Tier-2 produces hard counts in fixed lookback
windows (1h / 24h / 7d), Tier-4 produces an exponentially-weighted
sum that decays smoothly with time — a transaction 23h59m old still
contributes nearly the same as one 24h01m old, eliminating the
"window-boundary cliff" problem that confounds fixed-window features.

What EWM means in plain English
-------------------------------

Imagine you're a fraud analyst watching a credit card in real time.
You care about how active that card has been **recently** — but
"recently" is fuzzy. A transaction 5 minutes ago should weigh almost
as much as one happening right now. A transaction from a week ago
should still count, but less. A transaction from a year ago should
be nearly forgotten.

EWM (exponentially weighted moving sum) captures that intuition with
one knob: λ (lambda), the decay rate. Each past transaction's
contribution to the current EWM is::

    weight = exp(-λ · Δt_hours)

where Δt_hours is how many hours ago the transaction happened.

Three landmarks to anchor the intuition:

- Just-now (Δt = 0): weight = 1.0 (full credit)
- One half-life ago (Δt = ln(2) / λ): weight = 0.5 (half credit)
- Many half-lives ago: weight ≈ 0 (forgotten)

Half-lives at our default lambdas:

- λ = 0.05 / hour → half-life ≈ 13.9 hours (about half a day)
- λ = 0.10 / hour → half-life ≈  6.9 hours (a working day)
- λ = 0.50 / hour → half-life ≈  1.4 hours (under 90 minutes)

The "velocity" is the **sum of these weights** across every past
transaction for a given entity (card / address / device / email).
High velocity = the entity has been busy recently. Low velocity =
it's been quiet. By emitting features at multiple lambdas, the
model gets to choose which timescale carries the most signal:
burst-of-activity fraud surfaces at high λ; slow-burn fraud surfaces
at low λ.

The math, demystified — one worked example
------------------------------------------

Card A has three prior transactions today: $50 at T=0h, $30 at T=1h,
$200 at T=6h. A new transaction now arrives at T=6.5h. What does
the EWM see?

At λ = 0.5 / hour (1.4-hour half-life):

- Δt = 6.5h: weight = exp(-3.25) ≈ 0.039 (almost forgotten)
- Δt = 5.5h: weight = exp(-2.75) ≈ 0.064 (mostly forgotten)
- Δt = 0.5h: weight = exp(-0.25) ≈ 0.779 (still very fresh)
- Sum (v_ewm at λ=0.5) ≈ 0.88 — dominated by the 30-min-old event

At λ = 0.05 / hour (13.9-hour half-life):

- Δt = 6.5h: weight = exp(-0.325) ≈ 0.722
- Δt = 5.5h: weight = exp(-0.275) ≈ 0.760
- Δt = 0.5h: weight = exp(-0.025) ≈ 0.975
- Sum (v_ewm at λ=0.05) ≈ 2.46 — close to a full count of 3

The fraud-weighted variant `fraud_v_ewm` multiplies each weight by
the past transaction's `is_fraud` indicator (0 or 1). It's the
"recent fraud activity for this entity" signal, OOF-safe by the
read-before-push discipline (see "OOF-safety contract" below).

How the algorithm avoids re-summing — O(1) per event
----------------------------------------------------

Naive recomputation is O(n) per row, so O(n²) total. Infeasible at
590k rows. The trick: keep a running scalar `state.v` per (entity, λ)
and decay it forward whenever time advances::

    state(T_new) = state(T_old) · exp(-λ · Δt_hours)

Because exp is multiplicative, every term in the sum decays by the
same factor, so we just multiply the running scalar. To push a new
event, add 1.0 (or `+is_fraud`)::

    state.v = state.v · exp(-λ · Δt) + 1.0

That's two float operations + one `math.exp` per event per (entity, λ)
pair. The 100k-row benchmark runs in single-digit seconds.

Why EWM in fraud detection (vs NOT having it)
---------------------------------------------

Tier-2 already has `VelocityCounter` (fixed 1h / 24h / 7d windows).
Why add EWM on top?

The problem EWM solves: the **window-boundary cliff**. A 24-hour
counter treats a transaction 23h59m old as 1 (counted) and one
24h01m old as 0 (not counted). That cliff is operationally weird
in three ways:

1. The same fraud pattern produces a different score depending on
   when you measure (the counter ages out events as time advances).
2. Sophisticated fraudsters can time their transactions to fall
   just outside the window boundary ("burst, wait 24h+ε, burst").
3. Multi-timescale signal is hard to capture without proliferating
   windows. With only 1h / 24h / 7d, the model can't natively
   express "the last 6 hours mattered more than the rest of the day."

EWM replaces the cliff with a smooth decay. Activity 23h vs 25h old
produces almost the same EWM. No boundary to game.

Cost of NOT having EWM (status quo before this prompt):

1. Model can't smoothly track "still-hot" entities — only sees them
   through the 1h / 24h / 7d snapshots, with discontinuities.
2. Production-realistic latency requires EWM. `VelocityCounter`'s
   deque-based approach replays events on every read; at Sprint-5's
   serving budget (P95 < 100ms end-to-end), per-request replays
   are infeasible. EWM stores 3 floats per (entity, λ, value) Redis
   key and updates them in O(1).
3. Portfolio signal is weaker. Production fraud teams uniformly
   use EWM (or its close cousin, Holt-Winters smoothing).

Cost of HAVING EWM:

1. Feature space inflation. 4 entities × 3 λ × 2 signals = 24 new
   columns. Sprint 2 already produces 750 columns per row; +3%.
2. OOF discipline cost. `fraud_v_ewm` reads labels at training
   time. Get the read-vs-push order wrong and the model leaks
   training labels into training rows — catastrophic in production.
   The pass-1/pass-2 tied-group discipline (mirroring `VelocityCounter`)
   makes it inherently OOF-safe: reads happen before pushes.
3. State opacity. EWM compresses all history into a single running
   scalar, so debugging "which event drove this value?" is hard.
   The naive O(n²) reference (in tests) is the audit-time fallback.
4. λ is a hyperparameter. The default 0.05 / 0.1 / 0.5 trio spans
   ~14h / ~7h / ~1.4h half-lives. Sprint-3 hyperparameter sweep
   should re-evaluate.
5. Underflow is silent. For long quiet periods, `state.v` decays
   to floating-point zero. That's correct behaviour ("forgotten")
   but indistinguishable from "never seen." `ColdStartHandler`
   (Sprint 2.3.a) covers the latter case via a separate flag.

Net judgement: cliff problem + production-latency requirement
together justify the cost. OOF risk is mitigable. Portfolio signal
is meaningfully stronger.

Trade-offs considered (10 design decisions; both sides documented)
------------------------------------------------------------------

1. **Running scalar state vs deque-of-events recompute.** Chose
   running scalar: O(1) per event, ~4 MB state at 14k entities,
   serving-stack-friendly. Cost: audit opacity (which event drove
   this value? — answer requires re-running the naive reference).

2. **Two-pass tied-group batching.** Pass-1 reads state for every
   tied row first, pass-2 advances state for every tied row.
   Mirrors `VelocityCounter`. Gain: strict-past semantics + OOF
   safety for `fraud_v_ewm` at no extra cost. Cost: ~20 LOC and a
   `# noqa: PLR0912` for branch count; serving path will need a
   slightly different shape (events arrive one at a time).

3. **`fit_transform` ≠ `fit() + transform(train)` for training.**
   Mirrors `TargetEncoder`'s OOF override. Gain: BaseFeatureGenerator
   API compatibility + pipeline-fix-friendly (see 2.2.d). Cost:
   three modes with non-trivial behaviour each; `fit().transform(train)`
   silently leaks (documented footgun, not enforced).

4. **`transform(val)` does NOT push val labels.** Decay state
   forward and read; never push. Gain: production-semantics match
   (val labels unknown in serving), idempotent transforms,
   zero-leakage val path. Cost: state decays toward 0 over long
   val periods (irrelevant for IEEE-CIS's contiguous val window).

5. **Underflow is correct (no clamp).** `math.exp(-large)` returns
   `0.0`; let it. Gain: no bias toward "slightly active" for
   long-quiet entities. Cost: `state.v == 0.0` is ambiguous between
   "decayed away" and "never seen" — covered by `ColdStartHandler`.

6. **Hard-error on backward time** (`T < state.last_t` in transform).
   Raise `ValueError` with full diagnostic. Gain: loud failure on
   broken upstream invariants. Cost: no graceful degradation on
   user error — accepted (silent inflation is strictly worse).

7. **λ in /hour despite `TransactionDT` in seconds.** Divide by
   3600 in every state-update site. Gain: single source of truth
   for `TransactionDT` units across tiers; λ values match how
   fraud SMEs think. Cost: two extra divisions per event (~5 ns).

8. **`_DecayState` as `dataclass(slots=True)`.** Gain: ~5x memory
   reduction at 14k+ entries, ~30% faster attribute access, type
   safety. Cost: no dynamic attributes (acceptable; not needed).

9. **λ uniqueness validation in `__init__`.** Raise `ValueError`
   on duplicates. Gain: fail-fast on misconfiguration; duplicate
   λ values would silently produce duplicate columns. Cost: nil.

10. **Multiple lambdas as separate features (not picking one).**
    Emit features for every λ in YAML. Gain: multi-timescale
    signal; LightGBM picks the right λ per fraud pattern; aligns
    with industry practice. Cost: 3x feature space (4 × 3 × 2 = 24
    cols); 3x state memory (still ~4 MB).

OOF-safety contract
-------------------

Training row R reads the decayed state at `T_R` BEFORE its own
`is_fraud` is pushed. The pass-1/pass-2 batching enforces this at
the algorithmic level. Val/test rows decay-and-read but never push
(production semantics: val labels unknown in serving). The single-row
training case `is_fraud=1` produces `fraud_v_ewm = 0` exactly — the
"single-row OOF leak gate" in tests verifies this directly.

Performance contract
--------------------

100k rows × 4 entities × 3 λ × `fraud_weighted=True`: < 30 s wall
(matches `VelocityCounter`'s spec ceiling). Per-event cost is one
`math.exp` + one float-mul + one float-add per (entity, λ) pair.
Empirically expected sub-5-second wall on a modern machine.

Cross-references
----------------

- `temporal_guards.py:236` — `TemporalSafeGenerator` (naive-reference base).
- `tier2_aggregations.py:238` — `VelocityCounter.transform` (tied-group two-pass template).
- `tier2_aggregations.py:802` — `TargetEncoder.fit_transform` (OOF override pattern).
- Sprint-2 prompt 2.2.b report — design rationale for the deque template.
- Sprint-2 prompt 2.2.d report — the polymorphism fix that lets
  `FeaturePipeline.fit_transform` engage this class's `fit_transform`
  override.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Self

import numpy as np
import pandas as pd
import yaml

from fraud_engine.features.base import BaseFeatureGenerator

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Default temporal column. Mirrors `data/splits.py:_TIME_COLUMN` and
# `temporal_guards.py:_DEFAULT_TIMESTAMP_COL`; the project orders all
# temporal operations by `TransactionDT` (integer seconds).
_DEFAULT_TIMESTAMP_COL: Final[str] = "TransactionDT"

# Default target column for the fraud-weighted variant.
_DEFAULT_TARGET_COL: Final[str] = "isFraud"

# Config filename. Resolver helper below mirrors the per-tier
# duplication convention (4th copy across the project).
_TIER4_CONFIG_FILENAME: Final[str] = "tier4_config.yaml"

# Hours-per-second factor. λ is per-hour, `TransactionDT` is integer
# seconds; we divide by this in every state-update site to convert.
_SECONDS_PER_HOUR: Final[float] = 3600.0

# Format spec for λ values in column names. Python's `:g` produces
# clean output for the YAML defaults (0.05, 0.1, 0.5) without
# trailing zeros and without scientific notation. Both the optimised
# and naive-reference implementations use the same spec, so column
# names always agree by construction.
_LAMBDA_FORMAT_SPEC: Final[str] = "g"

# Output column name suffixes. Pinned constants so a future rename
# (e.g. `decay_velocity` instead of `v_ewm`) updates exactly two
# places (here + the test file's expected names).
_V_EWM_SUFFIX: Final[str] = "v_ewm_lambda"
_FRAUD_V_EWM_SUFFIX: Final[str] = "fraud_v_ewm_lambda"


# ---------------------------------------------------------------------
# YAML helpers (private).
# ---------------------------------------------------------------------


def _resolve_config_path(filename: str) -> Path:
    """Resolve `configs/{filename}` relative to the repo root.

    Mirrors `tier2_aggregations._resolve_config_path` and
    `tier3_behavioral._resolve_config_path` (4th copy in the project).
    Per-tier duplication is the established convention; refactoring
    to a shared helper is a separate cleanup task.

    Args:
        filename: Bare filename under `configs/` (e.g. `tier4_config.yaml`).

    Returns:
        Absolute path to the requested YAML.
    """
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "configs" / filename


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a YAML config file.

    Args:
        path: Absolute path to a YAML file.

    Returns:
        The parsed top-level mapping.

    Raises:
        FileNotFoundError: If `path` does not exist.
        TypeError: If the YAML root is not a mapping.
    """
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected top-level mapping in {path}, got {type(loaded).__name__}")
    return loaded


# ---------------------------------------------------------------------
# Internal running-state record.
# ---------------------------------------------------------------------


@dataclass(slots=True)
class _DecayState:
    """Per-(entity_col, λ, entity_value) running-state record.

    Three floats:

    - `last_t`: timestamp at which `v` and `fraud_v` are "as of".
    - `v`: EWM of all past events for this entity at this λ.
    - `fraud_v`: EWM weighted by past `is_fraud` (always tracked, even
      when `fraud_weighted=False`; the cost is a few extra float ops
      per push and the readability gain is worth it).

    Why `slots=True`:

    - At ~14k unique `card1` values × 12 (entity, λ) keys, a
      non-slotted dataclass costs ~5x more memory than a slotted
      one (~4 MB vs ~20 MB total state).
    - Slots speed attribute access by ~30% (no `__dict__` lookup).
    - Total state remains an order of magnitude smaller than the
      input frame, so memory isn't the actual bottleneck — but the
      access-speed win matters in the hot loop.
    """

    last_t: int
    v: float
    fraud_v: float


# ---------------------------------------------------------------------
# `ExponentialDecayVelocity`.
# ---------------------------------------------------------------------


class ExponentialDecayVelocity(BaseFeatureGenerator):
    """Per-(entity, λ) exponentially-decayed velocity (EWM).

    For each `(entity, λ)` pair from config, emits one feature column
    `{entity}_v_ewm_lambda_{λ}` whose value at row R is::

        Σ exp(-λ · Δt_i_hours)

    summed over **strictly-past** events for the same entity, where
    `Δt_i_hours` is hours from event_i to row R's timestamp.

    When `fraud_weighted=True`, also emits
    `{entity}_fraud_v_ewm_lambda_{λ}` (each term multiplied by the
    past row's `is_fraud` indicator). OOF-safe by the read-before-push
    pass-1/pass-2 batching (see module docstring).

    Business rationale:
        Velocity that decays gracefully captures "still-hot" entities
        without the boundary-cliff problem of fixed-window counts.
        Sprint-5's serving stack stores `(last_t, v_ewm, fraud_v_ewm)`
        per `(entity_col, λ, value)` Redis key with O(1) updates;
        this class is the batch-side counterpart.

    The OOF safety footgun (mirroring `TargetEncoder`):
        `fit_transform(train)` produces leak-free training-row outputs
        via the pass-1/pass-2 discipline. `fit(train)` builds the
        end-state without producing per-row outputs, intended for
        the val/test path: `gen.fit(train); gen.transform(val)`.
        **`gen.fit(train); gen.transform(train)` produces leaked
        outputs** — the full-train state has all training labels
        baked in, applied back to training rows. Use `fit_transform`
        for training; `fit` + `transform` for val/test only.

    Attributes:
        entity_cols: tuple of entity column names.
        lambdas: tuple of decay rates (per hour). Order preserved
            from input (YAML or constructor); pinned at construction.
        fraud_weighted: whether to emit `fraud_v_ewm` columns.
        target_col: name of the binary target column for the
            fraud-weighted variant.
        timestamp_col: temporal ordering column.
        _end_state_: post-fit running-state snapshot;
            `dict[(entity_col, λ), dict[entity_value, _DecayState]]`.
            None pre-fit. `transform(val)` decays this forward but
            does not mutate it.
    """

    def __init__(  # noqa: PLR0913 — six explicit kwargs mirror the YAML-override surface; same pattern as TargetEncoder.__init__.
        self,
        entity_cols: Sequence[str] | None = None,
        lambdas: Sequence[float] | None = None,
        fraud_weighted: bool | None = None,
        target_col: str | None = None,
        timestamp_col: str = _DEFAULT_TIMESTAMP_COL,
        config_path: Path | None = None,
    ) -> None:
        """Construct the EWM generator.

        Each value kwarg falls back to the YAML default when `None`.
        `timestamp_col` does not have a YAML override (it's a project
        invariant; only test code overrides it).

        Args:
            entity_cols: Column names to compute EWM over. If `None`,
                read from YAML's `entities` list.
            lambdas: Per-hour decay rates. If `None`, read from
                YAML's `lambdas_per_hour`. Must be unique (validated).
            fraud_weighted: When True, also emit `fraud_v_ewm`
                columns. If `None`, read from YAML.
            target_col: Target column name for the fraud-weighted
                variant. If `None`, read from YAML (default `isFraud`).
            timestamp_col: Temporal ordering column. Default
                `"TransactionDT"`.
            config_path: Override the default YAML path. Tests use
                this; production code uses the default.

        Raises:
            ValueError: If `lambdas` contains duplicates.
        """
        if entity_cols is None or lambdas is None or fraud_weighted is None or target_col is None:
            cfg = _load_yaml(config_path or _resolve_config_path(_TIER4_CONFIG_FILENAME))
        else:
            cfg = {}

        self.entity_cols: tuple[str, ...] = tuple(
            entity_cols if entity_cols is not None else cfg.get("entities", [])
        )
        chosen_lambdas: Sequence[float] = (
            lambdas if lambdas is not None else cfg.get("lambdas_per_hour", [])
        )
        # Validate λ uniqueness BEFORE storing — fail-fast on misconfiguration.
        # Duplicate λ values would silently produce duplicate column
        # names (the second overwriting the first), wasting feature
        # budget; raising here surfaces the error at construction.
        if len(set(chosen_lambdas)) != len(chosen_lambdas):
            raise ValueError(
                f"ExponentialDecayVelocity: lambdas must be unique; got {list(chosen_lambdas)!r}"
            )
        self.lambdas: tuple[float, ...] = tuple(float(lam) for lam in chosen_lambdas)
        self.fraud_weighted: bool = (
            bool(fraud_weighted)
            if fraud_weighted is not None
            else bool(cfg.get("fraud_weighted", False))
        )
        self.target_col: str = (
            target_col if target_col is not None else cfg.get("target_col", _DEFAULT_TARGET_COL)
        )
        self.timestamp_col: str = timestamp_col

        # Fitted state. Populated by `fit` or `fit_transform`. The
        # `transform(val)` path raises `AttributeError` if this is
        # still None.
        self._end_state_: dict[tuple[str, float], dict[Any, _DecayState]] | None = None

    # -----------------------------------------------------------------
    # Hot-loop helpers.
    # -----------------------------------------------------------------

    def _decay_and_read(self, st: _DecayState, t_event: int, lam: float) -> tuple[float, float]:
        """Return `(v, fraud_v)` decayed forward to `t_event` WITHOUT mutating `st`.

        Hard-errors on backward time. Why hard-error: a backward
        `t_event` means the upstream `temporal_split` invariant has
        been violated, and silently inflating the state above 1.0
        (which `exp(-λ * negative)` does) would corrupt downstream
        model predictions in ways unobservable without this assertion.
        Matches the project's fail-loudly-at-boundaries philosophy
        (negative-amount rejection in `AmountTransformer`, schema
        validation at every tier boundary).

        Args:
            st: The per-entity running-state record.
            t_event: Current row's timestamp (seconds).
            lam: Decay rate per hour.

        Returns:
            Tuple of `(v_decayed, fraud_v_decayed)`. `st` is unchanged.

        Raises:
            ValueError: If `t_event < st.last_t` (broken temporal
                invariant).
        """
        if t_event < st.last_t:
            raise ValueError(
                f"ExponentialDecayVelocity: backward time at "
                f"t_event={t_event} vs state.last_t={st.last_t}. "
                f"Upstream temporal_split should guarantee monotonic "
                f"timestamps."
            )
        dt_hours = (t_event - st.last_t) / _SECONDS_PER_HOUR
        decay = math.exp(-lam * dt_hours)
        return st.v * decay, st.fraud_v * decay

    def _push(self, st: _DecayState, t_event: int, lam: float, fraud_label: float) -> None:
        """Decay `st` to `t_event`, then add the new event's contribution.

        Why we recompute `decay` here instead of caching it from a
        prior `_decay_and_read` call: caching would require an extra
        per-(entity, λ) dict entry within each tied group, and the
        cost of allocating + populating that cache exceeds the saved
        `math.exp` call (~50 ns). One redundant `exp` per push is the
        right trade-off for code clarity in the hot loop.

        Args:
            st: The per-entity running-state record. **Mutated in place.**
            t_event: Current event's timestamp (seconds).
            lam: Decay rate per hour.
            fraud_label: 1.0 for fraud, 0.0 for not-fraud, 0.0 for
                unknown / NaN. Defensive: caller should clean NaN
                upstream, but treating NaN-as-0 here is the safe
                fallback (matches the naive reference's policy).
        """
        dt_hours = (t_event - st.last_t) / _SECONDS_PER_HOUR
        decay = math.exp(-lam * dt_hours)
        st.v = st.v * decay + 1.0
        st.fraud_v = st.fraud_v * decay + fraud_label
        st.last_t = t_event

    # -----------------------------------------------------------------
    # `BaseFeatureGenerator` contract.
    # -----------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> Self:
        """Build the end-of-training state without producing per-row outputs.

        Use only when training-row outputs are not needed (the
        canonical training path is `fit_transform`). The val/test
        path is `gen.fit(train); gen.transform(val)`.

        **Footgun warning:** `gen.fit(train); gen.transform(train)`
        produces *leaked* outputs — the full-train state has all
        training labels baked in, applied back to training rows.
        Use `fit_transform` for training; `fit` + `transform(val)`
        for val/test only. This mirrors `TargetEncoder.fit`'s
        documented misuse path.

        Args:
            df: Training frame. Must contain `target_col` if
                `self.fraud_weighted=True`.

        Returns:
            self.

        Raises:
            KeyError: If a required column is missing.
        """
        self._run_passes(df, write_results=False)
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and produce OOF-safe training-row outputs in one pass.

        Pass-1/pass-2 tied-group batching: pass-1 reads decayed state
        for every tied row at timestamp T; pass-2 advances state for
        every tied row. The pass-1 reads happen BEFORE any pass-2
        push, so each row's `fraud_v_ewm` reflects only events
        strictly earlier than its own (OOF-safe).

        Args:
            df: Training frame. Must contain `timestamp_col`, every
                `entity_cols` column, and `target_col` if
                `fraud_weighted=True`.

        Returns:
            `df.copy()` with one new column per (entity, λ) pair
            (× 2 if `fraud_weighted=True`).

        Raises:
            KeyError: If a required column is missing.
        """
        results = self._run_passes(df, write_results=True)
        out = df.copy()
        # `_run_passes` returns a dict; iterating preserves insertion
        # order which matches `get_feature_names()` ordering.
        if results is not None:
            for name, vals in results.items():
                out[name] = vals
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted end-state to `df`. Single-pass; no pushes.

        Each row's features are computed by decaying the persisted
        end-state forward to the row's timestamp and reading. State
        is NOT mutated — calling `transform` multiple times produces
        identical output (idempotent), and val/test rows do not
        contribute to subsequent rows' features.

        Production-semantics rationale: in Sprint-5 serving, val
        rows are real-time predictions where labels are unknown.
        Not pushing labels means the batch and serving paths
        produce identical features for the same input.

        Args:
            df: Frame to transform. Must contain `timestamp_col`
                and every `entity_cols` column. Does NOT need
                `target_col` (val/test labels not used).

        Returns:
            `df.copy()` with one new column per (entity, λ) pair
            (× 2 if `fraud_weighted=True`). Unseen entity values
            (not present in fit data) emit 0.

        Raises:
            AttributeError: If `fit` / `fit_transform` has not run.
            KeyError: If a required column is missing.
            ValueError: If a row's timestamp is earlier than the
                fitted state's `last_t` (broken temporal invariant).
        """
        if self._end_state_ is None:
            raise AttributeError("ExponentialDecayVelocity must be fit before transform")
        self._validate_columns(df, require_target=False)

        feature_names = self.get_feature_names()
        n = len(df)

        timestamps = df[self.timestamp_col].to_numpy()
        sort_idx = np.argsort(timestamps, kind="stable")
        sorted_timestamps = timestamps[sort_idx]
        sorted_entities: dict[str, np.ndarray[Any, Any]] = {
            ec: df[ec].to_numpy()[sort_idx] for ec in self.entity_cols
        }

        results: dict[str, list[float]] = {name: [0.0] * n for name in feature_names}

        # Single-pass: no pushes, so no tied-group batching needed —
        # tied rows all read from the same frozen end-state and
        # produce the same value (correct).
        for k in range(n):
            ts_value = int(sorted_timestamps[k])
            orig_pos = int(sort_idx[k])
            for ec in self.entity_cols:
                entity_val = sorted_entities[ec][k]
                if pd.isna(entity_val):
                    continue  # default 0.0 stays
                for lam in self.lambdas:
                    inner = self._end_state_.get((ec, lam))
                    if inner is None:
                        continue  # defensive; should never happen post-fit
                    st = inner.get(entity_val)
                    if st is None:
                        continue  # unseen entity → 0.0 fallback
                    v_val, fraud_v_val = self._decay_and_read(st, ts_value, lam)
                    lam_str = f"{lam:{_LAMBDA_FORMAT_SPEC}}"
                    results[f"{ec}_{_V_EWM_SUFFIX}_{lam_str}"][orig_pos] = v_val
                    if self.fraud_weighted:
                        results[f"{ec}_{_FRAUD_V_EWM_SUFFIX}_{lam_str}"][orig_pos] = fraud_v_val

        out = df.copy()
        for name, vals in results.items():
            out[name] = vals
        return out

    def get_feature_names(self) -> list[str]:
        """Return the deterministic list of generated column names.

        Order: outer entity_cols (declaration order); inner lambdas
        (declaration order); within each (entity, λ) the `v_ewm`
        column comes before `fraud_v_ewm`. All three orderings are
        pinned at construction so the manifest is stable across
        re-runs.

        Returns:
            List of column names. Length is
            `len(entity_cols) * len(lambdas) * (2 if fraud_weighted else 1)`.
        """
        names: list[str] = []
        for ec in self.entity_cols:
            for lam in self.lambdas:
                lam_str = f"{lam:{_LAMBDA_FORMAT_SPEC}}"
                names.append(f"{ec}_{_V_EWM_SUFFIX}_{lam_str}")
                if self.fraud_weighted:
                    names.append(f"{ec}_{_FRAUD_V_EWM_SUFFIX}_{lam_str}")
        return names

    def get_business_rationale(self) -> str:
        """Return the manifest-rendered business rationale."""
        return (
            "Per-(entity, λ) exponentially-decayed velocity (EWM). "
            "Smooths the window-boundary cliff that fixed-window "
            "velocity has by construction — recent activity decays "
            "gracefully rather than dropping off at an arbitrary "
            "boundary. Multiple lambdas span hourly-through-daily "
            "timescales so the model can pick the right decay rate "
            "per fraud pattern. The fraud-weighted variant tracks "
            "EWM of past confirmed fraud for this entity, OOF-safe "
            "by the read-before-push two-pass discipline."
        )

    # -----------------------------------------------------------------
    # Internal: shared fit / fit_transform pass machinery.
    # -----------------------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame, *, require_target: bool) -> None:
        """Raise `KeyError` if any required column is missing from `df`.

        `require_target` lets the same validator serve both the
        training path (where the target must be present for
        `fraud_weighted=True`) and the val/test path (where it
        must NOT be required, since serving doesn't have labels).
        """
        required = [self.timestamp_col, *self.entity_cols]
        if require_target and self.fraud_weighted:
            required.append(self.target_col)
        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise KeyError(f"ExponentialDecayVelocity: missing required column(s) {missing}")

    def _run_passes(  # noqa: PLR0912, PLR0915 — tied-group two-pass × per-entity × per-λ × fraud-weighted branch is one tightly-coupled algorithm; splitting across helpers would lose locality and hurt the hot-loop's clarity.
        self, df: pd.DataFrame, *, write_results: bool
    ) -> dict[str, list[float]] | None:
        """Sweep `df` in temporal order; build (and optionally write) outputs.

        Shared by `fit` (write_results=False) and `fit_transform`
        (write_results=True). Both modes:

        1. Validate columns (target required iff `fraud_weighted`).
        2. Sort by `timestamp_col`; pre-extract entity arrays.
        3. Iterate tied-timestamp groups in order:
           a. Pass-1: for every tied row × entity × λ, read decayed
              state into the results dict (if `write_results`).
           b. Pass-2: for every tied row × entity × λ, advance state
              with the row's contribution.
        4. Persist `self._end_state_`.

        Args:
            df: Training frame.
            write_results: If True, allocate and populate the
                results dict; if False, skip per-row writes
                (the `fit`-only path).

        Returns:
            Results dict if `write_results=True`, else `None`.
        """
        self._validate_columns(df, require_target=True)

        feature_names = self.get_feature_names()
        n = len(df)

        timestamps = df[self.timestamp_col].to_numpy()
        sort_idx = np.argsort(timestamps, kind="stable")
        sorted_timestamps = timestamps[sort_idx]
        sorted_entities: dict[str, np.ndarray[Any, Any]] = {
            ec: df[ec].to_numpy()[sort_idx] for ec in self.entity_cols
        }
        if self.fraud_weighted:
            sorted_fraud: np.ndarray[Any, Any] | None = (
                df[self.target_col].to_numpy()[sort_idx].astype(float)
            )
        else:
            sorted_fraud = None

        # Pre-allocate results only when needed (saves ~24 × n floats
        # in the `fit`-only path).
        results: dict[str, list[float]] | None = (
            {name: [0.0] * n for name in feature_names} if write_results else None
        )

        # State container. Plain dict (not defaultdict) — defaultdict
        # would create empty `_DecayState` records on every NaN read,
        # polluting state with phantom entities.
        state: dict[tuple[str, float], dict[Any, _DecayState]] = {
            (ec, lam): {} for ec in self.entity_cols for lam in self.lambdas
        }

        i = 0
        while i < n:
            # Identify the run of tied timestamps starting at i.
            j = i
            tie_value = sorted_timestamps[i]
            while j < n and sorted_timestamps[j] == tie_value:
                j += 1
            ts_value = int(tie_value)

            # PASS 1: read-only. Every tied row at timestamp T sees
            # the same pre-tie state (no push has happened yet for
            # this tied group).
            if write_results:
                assert results is not None  # noqa: S101 — invariant; mypy needs the narrow.
                for k in range(i, j):
                    orig_pos = int(sort_idx[k])
                    for ec in self.entity_cols:
                        entity_val = sorted_entities[ec][k]
                        if pd.isna(entity_val):
                            continue  # default 0.0 stays in place
                        for lam in self.lambdas:
                            st = state[(ec, lam)].get(entity_val)
                            if st is None:
                                continue  # never-seen → 0.0 stays
                            v_val, fraud_v_val = self._decay_and_read(st, ts_value, lam)
                            lam_str = f"{lam:{_LAMBDA_FORMAT_SPEC}}"
                            results[f"{ec}_{_V_EWM_SUFFIX}_{lam_str}"][orig_pos] = v_val
                            if self.fraud_weighted:
                                results[f"{ec}_{_FRAUD_V_EWM_SUFFIX}_{lam_str}"][orig_pos] = (
                                    fraud_v_val
                                )

            # PASS 2: push. Tied rows now contribute to subsequent
            # groups' reads. Order within the tied group doesn't
            # matter for correctness — at Δt=0 within the group,
            # the decay factor is 1 for every push.
            for k in range(i, j):
                for ec in self.entity_cols:
                    entity_val = sorted_entities[ec][k]
                    if pd.isna(entity_val):
                        continue
                    # NaN-fraud policy: cleaner forbids null `is_fraud`
                    # but defensively treat NaN as 0 (matches the
                    # naive reference's `np.nan_to_num`).
                    if self.fraud_weighted and sorted_fraud is not None:
                        raw_fraud = sorted_fraud[k]
                        fraud_label = 0.0 if math.isnan(raw_fraud) else float(raw_fraud)
                    else:
                        fraud_label = 0.0
                    for lam in self.lambdas:
                        inner = state[(ec, lam)]
                        st = inner.get(entity_val)
                        if st is None:
                            # Lazy-insert: equivalent to starting from
                            # `(ts_value, 0, 0)` then `_push` (decay=1
                            # at Δt=0; v=1, fraud_v=fraud_label).
                            inner[entity_val] = _DecayState(
                                last_t=ts_value, v=1.0, fraud_v=fraud_label
                            )
                        else:
                            self._push(st, ts_value, lam, fraud_label)

            i = j

        self._end_state_ = state
        return results


__all__ = ["ExponentialDecayVelocity"]
