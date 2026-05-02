"""FraudNet entity-embedding neural network (Sprint 3 Model B).

`FraudNet` is the project's diversity model: a small PyTorch network
that ingests `card1`, `addr1`, `DeviceInfo` as embedded categoricals
and the rest of the Tier-5 numerics through an MLP head, trained with
focal loss against the 3.5 % fraud base rate. Per CLAUDE.md §3 it is
shadow-deployable — Model A (LightGBM) bears the production-serving
contract, FraudNet earns its keep through ensemble diversity in
Sprint 4.

Business rationale:
    Tabular fraud data is dominated by high-cardinality entity IDs
    (cards, addresses, devices). LightGBM handles those via integer
    splits but cannot share signal across IDs that look similar in
    behaviour. Entity embeddings let the model learn a dense
    representation per ID where neighbouring IDs in embedding space
    behave similarly — exactly the structure a graph-aware fraud
    system needs. Pairing embeddings with focal loss (which
    down-weights the easy-negative majority) gets us a competitive
    standalone model and, more importantly, a model whose *errors*
    are decorrelated from LightGBM's — the prerequisite for an
    ensemble that beats either alone.

Trade-offs considered:
    - **Three embeddings (card1, addr1, DeviceInfo) vs more.** Per
      project plan §3.1 Model B. Adding card2-6, addr2,
      P/R_emaildomain would expand the embedding parameter count
      and the OOV-handling surface without a commensurate diversity
      gain — those are mostly redundant with what LightGBM already
      captures. The three chosen are the structurally important
      identity anchors (account / billing / device).
    - **Focal loss vs `BCEWithLogitsLoss(pos_weight=...)`.** Focal
      down-weights *easy* examples (regardless of class) by
      `(1-p_t)^γ`; pos_weight scales the positive-class loss by a
      constant. Both address imbalance, but they conflict if used
      together (focal already balances; pos_weight then over-weights
      the minority). Per spec we use focal alone.
    - **Embedding vocabularies built transductively from
      train+val+test combined.** IEEE-CIS is a fixed dataset; we
      know every ID we'll score. Building train-only vocab would
      hash 30-50 % of val/test IDs into the OOV bucket needlessly,
      destroying signal. The OOV bucket (index 0) remains for true
      production deployment where new IDs arrive at predict time.
    - **NaN handling: standardize-then-fillna(0).** Two-column-per-
      feature missing-indicator schemes double the BatchNorm noise
      surface. Filling NaNs with the post-standardization mean (=0)
      preserves the column statistics LightGBM relies on for
      tree-split signalling. Documented as a deliberate trade-off.
    - **Joblib full-instance persistence.** PyTorch state_dict-only
      saves are mechanically correct but force the loader to
      reconstruct the vocab + scaler manually. Joblib-pickle of
      the full `FraudNetModel` instance is what `LightGBMFraudModel`
      already does (PR #32) and matches downstream consumer
      expectations. Trade-off: forward-incompatible if `FraudNet`'s
      __init__ signature changes; the manifest's `schema_version`
      lets us bump for migrations.
    - **CPU-only training.** torch.cuda.is_available() is False on
      the dev machine; the network is small enough (<200k params)
      that CPU is fine. Adding CUDA paths would be untestable here.
      A `device=` knob is exposed so the production deployer can
      flip to GPU without API change.
    - **Per-epoch MLflow metrics, not per-batch.** ~200 batches/epoch
      × 50 epochs = 10k metric rows per run if logged per-batch;
      MLflow's UI sluggish past ~1k. Per-epoch is the right cadence
      for diagnostic value.
    - **`predict_proba` returns (n, 2).** Mirrors `LightGBMFraudModel`
      exactly so Sprint 4's evaluator and Sprint 5's serving stack
      can substitute models at the same call sites.

Cross-references:
    - `src/fraud_engine/models/lightgbm_model.py` — save/load + manifest pattern this mirrors
    - `src/fraud_engine/evaluation/calibration.py` — Sprint 4 will calibrate FraudNet outputs (out of scope here)
    - `docs/ADR/0001-tech-stack.md:62-74` — PyTorch scoped to diversity models
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, Self

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fraud_engine.config.settings import get_settings
from fraud_engine.utils.logging import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Persistence filenames; mirrors LightGBMFraudModel naming convention.
_MODEL_FILENAME: Final[str] = "neural_model.pt"
_MANIFEST_FILENAME: Final[str] = "neural_model_manifest.json"

# Manifest schema version; bump on incompatible JSON shape changes.
_MANIFEST_SCHEMA_VERSION: Final[int] = 1

# Schema-fingerprint truncation length, mirrors `data/lineage.py`.
_SCHEMA_FINGERPRINT_HEX_CHARS: Final[int] = 16

# Number of class probability columns returned by `predict_proba`.
_N_CLASS_PROBS: Final[int] = 2

# OOV / null index in every embedding table. The vocabulary maps real
# values to indices 1..vocab_size; index 0 is reserved.
_OOV_INDEX: Final[int] = 0

# The three categorical entities embedded by FraudNet. Order matters —
# the forward pass expects (card1_ids, addr1_ids, deviceinfo_ids).
ENTITY_COLUMNS: Final[tuple[str, ...]] = ("card1", "addr1", "DeviceInfo")

# Default architecture hyperparameters.
_DEFAULT_EMBED_DIM: Final[int] = 32
_DEFAULT_HIDDEN_DIM: Final[int] = 64
_DEFAULT_DROPOUT: Final[float] = 0.3

# Default training hyperparameters.
_DEFAULT_BATCH_SIZE: Final[int] = 2048
_DEFAULT_MAX_EPOCHS: Final[int] = 50
_DEFAULT_LR: Final[float] = 1e-3
_DEFAULT_WEIGHT_DECAY: Final[float] = 1e-5
_DEFAULT_EARLY_STOPPING_PATIENCE: Final[int] = 5

# Default focal-loss hyperparameters (Lin et al. 2017 conventions).
_DEFAULT_FOCAL_ALPHA: Final[float] = 0.25
_DEFAULT_FOCAL_GAMMA: Final[float] = 2.0


# ---------------------------------------------------------------------
# Vocabulary helpers.
# ---------------------------------------------------------------------


def _build_vocab(values: pd.Series[Any]) -> dict[Any, int]:
    """Build a value->index dict reserving index 0 for OOV/null.

    Business rationale:
        Embedding tables need a dense integer index per known value
        and a sentinel for everything else. Reserving index 0 (rather
        than `vocab_size`) means downstream `clip(0, vocab_size)` calls
        on unseen IDs are safe, and the OOV embedding sits at a
        predictable location for inspection.

    Trade-offs considered:
        - Sorting the unique values gives deterministic indices across
          re-runs of the same data, which makes manifest fingerprints
          stable. Cost: O(N log N) at vocab-build time, paid once.
        - Pandas NaN is treated specially because `pd.Series.unique`
          returns NaN as a value, and dict-keying NaN is undefined
          (NaN != NaN). We drop NaN here and rely on
          `_map_to_indices` to map missing values to index 0.

    Args:
        values: A Series of categorical IDs (numeric or string).

    Returns:
        Dict mapping each non-null unique value to a 1-based index.
        Empty dict if the series has no non-null values.
    """
    uniques = values.dropna().unique()
    # Sorting a heterogeneous-typed array via tolist() keeps numerics
    # and strings each in their natural order; mixed-type columns
    # don't appear in IEEE-CIS so we don't need a stable mixed sort.
    try:
        sorted_uniques = sorted(uniques.tolist())
    except TypeError:
        # Fallback for un-orderable mixed types: keep the discovery order.
        sorted_uniques = list(uniques.tolist())
    return {value: idx + 1 for idx, value in enumerate(sorted_uniques)}


def _map_to_indices(values: pd.Series[Any], vocab: dict[Any, int]) -> np.ndarray[Any, Any]:
    """Map a Series of values to int64 indices using `vocab`.

    Unknown / null values map to `_OOV_INDEX` (0). The result is an
    int64 array suitable for `torch.from_numpy().long()`.

    Args:
        values: A Series of categorical IDs.
        vocab: Output of `_build_vocab` (or a merged transductive
            vocab built from train+val+test).

    Returns:
        int64 numpy array of shape `(len(values),)` with values in
        `[0, len(vocab)]`.
    """
    # Pandas `.map` returns NaN for misses; fillna(0).astype(int64)
    # is the single cleanest path that handles both unknown values
    # AND original NaNs in one step.
    mapped = values.map(vocab).fillna(_OOV_INDEX).astype(np.int64)
    return mapped.to_numpy()


# ---------------------------------------------------------------------
# Focal loss.
# ---------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Focal loss for binary classification on raw logits.

    Implements Lin et al. (2017): `FL(p_t) = -α (1 - p_t)^γ log(p_t)`,
    where `p_t = p` if `y == 1` else `1 - p`. Operating on logits
    (rather than probabilities) gives `BCEWithLogits`-style numerical
    stability for very small / very large logits.

    Business rationale:
        IEEE-CIS has a 3.5 % positive rate. Plain BCE under-trains on
        positives because each easy negative contributes meaningful
        gradient. Focal loss attenuates easy-example loss by
        `(1-p_t)^γ`, which pushes optimization toward the hard,
        misclassified examples — exactly the fraud cases the model
        most needs to learn from.

    Trade-offs considered:
        - **`reduction="mean"` over per-row.** Mean keeps the loss
          magnitude scale-invariant to batch size; downstream
          schedulers (ReduceLROnPlateau) compare across epochs
          without rescaling. Sum would diverge for large batches.
        - **α defaults to 0.25** (the spec's "down-weight common
          class" sense — i.e., negatives, the majority here, get
          weight `1 - α = 0.75`; positives get α = 0.25). This is
          the Lin et al. inversion that confuses readers; we
          document it explicitly. The right α for IEEE-CIS at default
          γ is empirically near 0.25–0.5; user can override.
        - **Numerical stability via `torch.clamp(logp, min=-100)`.**
          For very negative logits the cross-entropy log term can
          underflow to -inf which then nans the loss. Clamping at
          -100 caps the loss without touching the gradient direction.
    """

    def __init__(
        self,
        alpha: float = _DEFAULT_FOCAL_ALPHA,
        gamma: float = _DEFAULT_FOCAL_GAMMA,
    ) -> None:
        """Construct the focal-loss module.

        Args:
            alpha: Class-balance weight for the *positive* class.
                Negatives get weight `1 - alpha`. Default 0.25 (Lin
                et al.).
            gamma: Focusing parameter. `gamma == 0` reduces to plain
                weighted BCE; larger values down-weight easy examples
                more aggressively. Default 2.0.

        Raises:
            ValueError: If `alpha` is outside [0, 1] or `gamma < 0`.
        """
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"FocalLoss: alpha must be in [0, 1], got {alpha}")
        if gamma < 0.0:
            raise ValueError(f"FocalLoss: gamma must be >= 0, got {gamma}")
        self.alpha: float = float(alpha)
        self.gamma: float = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the mean focal loss over the batch.

        Args:
            logits: Pre-sigmoid raw outputs, shape `(N,)` or `(N, 1)`.
            targets: Binary {0, 1} labels, shape `(N,)` or `(N, 1)`.
                Cast to `logits.dtype` internally.

        Returns:
            Scalar tensor (mean loss across the batch).
        """
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).to(logits.dtype)
        # `F.binary_cross_entropy_with_logits(reduction="none")` would
        # work but doesn't expose `p_t` cleanly. We compute it manually
        # for numerical stability and clarity.
        log_p = torch.nn.functional.logsigmoid(logits)
        log_1_minus_p = torch.nn.functional.logsigmoid(-logits)
        # Bound very negative log-probs so a saturated misclassification
        # doesn't NaN the loss; the gradient direction is preserved.
        log_p = torch.clamp(log_p, min=-100.0)
        log_1_minus_p = torch.clamp(log_1_minus_p, min=-100.0)
        # log_p_t and (1 - p_t) for the focal weighting.
        log_p_t = targets * log_p + (1.0 - targets) * log_1_minus_p
        p_t = log_p_t.exp()
        focal_weight = (1.0 - p_t) ** self.gamma
        # Class-balance: alpha on positives, 1-alpha on negatives.
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss: torch.Tensor = -alpha_t * focal_weight * log_p_t
        return loss.mean()


# ---------------------------------------------------------------------
# Network module.
# ---------------------------------------------------------------------


class FraudNet(nn.Module):
    """Entity-embedding tabular network for fraud detection.

    Architecture (forward pass):
        card1_emb = Embedding[card1_vocab + 1, embed_dim](card1_ids)
        addr1_emb = Embedding[addr1_vocab + 1, embed_dim](addr1_ids)
        device_emb = Embedding[device_vocab + 1, embed_dim](device_ids)
        x_num = BN(X_numeric) -> Linear(n_numeric, hidden) -> ReLU -> Drop
        h = Concat([card1_emb, addr1_emb, device_emb, x_num])
        h = Linear(3*embed_dim + hidden, hidden) -> ReLU -> Drop
        logit = Linear(hidden, 1)
        return logit  # (N,)

    Business rationale:
        Three embedding heads + numeric branch is the smallest
        architecture that exercises both modalities (entity-ID +
        numeric) of the IEEE-CIS feature space. A larger network
        (deeper or wider) is unjustified at this dataset scale and
        would slow ensemble experimentation in Sprint 4. The
        symmetric `embed_dim` across entities is per spec; differing
        per-entity dims add tuning surface without obvious payoff.

    Trade-offs considered:
        - **BatchNorm on numerics, not LayerNorm.** Numerics in
          IEEE-CIS have heterogeneous scales (cents vs counts vs
          ratios); BatchNorm normalizes per-batch with running
          stats, which adapts to the empirical distribution
          throughout training. LayerNorm would normalize per-row
          across features that have wildly different meanings,
          which is wrong here.
        - **No BN on embedding output.** Embeddings are already
          unit-scale init; running BN over them couples training
          stability across IDs that have nothing to do with each
          other.
        - **Embedding init ~ N(0, 1/sqrt(embed_dim)).** Standard
          Glorot-equivalent for embedding tables; PyTorch's default
          (N(0, 1)) makes the early forward passes saturate the
          ReLU. We explicitly seed.
        - **Single hidden layer in the head.** Adding a second adds
          parameters faster than capacity; spot-check on val AUC
          showed no improvement during planning.
    """

    def __init__(  # noqa: PLR0913 — three vocab sizes + n_numeric + three architecture knobs is the minimal honest surface; collapsing into a config dict would hide the contract from every call site
        self,
        card1_vocab_size: int,
        addr1_vocab_size: int,
        deviceinfo_vocab_size: int,
        n_numeric: int,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        dropout: float = _DEFAULT_DROPOUT,
    ) -> None:
        """Construct the FraudNet module.

        Args:
            card1_vocab_size: Number of distinct card1 values seen at
                vocab-build time. The embedding table is sized
                `card1_vocab_size + 1` to reserve index 0 for OOV.
            addr1_vocab_size: As above for addr1.
            deviceinfo_vocab_size: As above for DeviceInfo.
            n_numeric: Number of numeric input features.
            embed_dim: Embedding dimension per entity. Default 32.
            hidden_dim: Hidden width for both the numeric branch and
                the post-concat head. Default 64.
            dropout: Dropout probability applied after each ReLU.
                Default 0.3.

        Raises:
            ValueError: If any vocab size or `n_numeric` is negative,
                or `embed_dim`/`hidden_dim` are non-positive, or
                `dropout` is outside [0, 1).
        """
        super().__init__()
        if (
            card1_vocab_size < 0
            or addr1_vocab_size < 0
            or deviceinfo_vocab_size < 0
            or n_numeric < 0
        ):
            raise ValueError("FraudNet: vocab sizes and n_numeric must be >= 0")
        if embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError("FraudNet: embed_dim and hidden_dim must be > 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"FraudNet: dropout must be in [0, 1), got {dropout}")

        self.embed_dim: int = int(embed_dim)
        self.hidden_dim: int = int(hidden_dim)
        self.dropout_p: float = float(dropout)
        self.n_numeric: int = int(n_numeric)
        self.card1_vocab_size: int = int(card1_vocab_size)
        self.addr1_vocab_size: int = int(addr1_vocab_size)
        self.deviceinfo_vocab_size: int = int(deviceinfo_vocab_size)

        # +1 on each vocab for the OOV slot at index 0.
        self.card1_emb = nn.Embedding(card1_vocab_size + 1, embed_dim)
        self.addr1_emb = nn.Embedding(addr1_vocab_size + 1, embed_dim)
        self.deviceinfo_emb = nn.Embedding(deviceinfo_vocab_size + 1, embed_dim)
        # Glorot-equivalent init for embedding tables.
        std = 1.0 / math.sqrt(embed_dim)
        for emb in (self.card1_emb, self.addr1_emb, self.deviceinfo_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=std)

        # Numeric branch: BatchNorm -> Linear -> ReLU -> Dropout.
        # `n_numeric == 0` is supported (degenerate test case) by
        # using `nn.Identity` placeholders; production paths pass >0.
        if n_numeric > 0:
            self.numeric_bn: nn.Module = nn.BatchNorm1d(n_numeric)
            self.numeric_fc: nn.Module = nn.Linear(n_numeric, hidden_dim)
        else:
            self.numeric_bn = nn.Identity()
            self.numeric_fc = nn.Identity()

        # Post-concat head: Linear -> ReLU -> Dropout -> Linear.
        concat_dim = 3 * embed_dim + (hidden_dim if n_numeric > 0 else 0)
        self.head_fc1 = nn.Linear(concat_dim, hidden_dim)
        self.head_fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(
        self,
        card1_ids: torch.Tensor,
        addr1_ids: torch.Tensor,
        deviceinfo_ids: torch.Tensor,
        x_numeric: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning per-row logits.

        Args:
            card1_ids: Long tensor, shape `(N,)`. Values in
                `[0, card1_vocab_size]`.
            addr1_ids: As above for addr1.
            deviceinfo_ids: As above for DeviceInfo.
            x_numeric: Float tensor, shape `(N, n_numeric)`. NaNs are
                expected to be pre-filled by the caller (the
                `FraudNetModel` wrapper does this before tensorising).

        Returns:
            Float tensor of shape `(N,)` containing pre-sigmoid logits.
        """
        c = self.card1_emb(card1_ids)
        a = self.addr1_emb(addr1_ids)
        d = self.deviceinfo_emb(deviceinfo_ids)
        if self.n_numeric > 0:
            n = self.relu(self.numeric_fc(self.numeric_bn(x_numeric)))
            n = self.dropout(n)
            h = torch.cat([c, a, d, n], dim=1)
        else:
            h = torch.cat([c, a, d], dim=1)
        h = self.relu(self.head_fc1(h))
        h = self.dropout(h)
        return self.head_fc2(h).reshape(-1)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------
# FraudNetModel — sklearn-style wrapper.
# ---------------------------------------------------------------------


def _schema_fingerprint(columns: list[str]) -> str:
    """SHA-256 of a sorted column list, hex-truncated.

    Same column-name-only recipe used inline by
    `LightGBMFraudModel._build_manifest`. (The canonical
    dtype-aware fingerprint at `data.lineage._fingerprint_dataframe`
    is deliberately not used here — neural-model numeric features
    go through StandardScaler before the model sees them, so
    per-column dtype is always float32 by construction and dtype
    drift is impossible.)
    """
    schema_str = json.dumps(sorted(columns), separators=(",", ":"))
    return hashlib.sha256(schema_str.encode("utf-8")).hexdigest()[:_SCHEMA_FINGERPRINT_HEX_CHARS]


class FraudNetModel:
    """Sklearn-style wrapper owning vocab + scaler + the `FraudNet` module.

    Public API mirrors `LightGBMFraudModel` so Sprint 4's evaluator
    and Sprint 5's serving harness can swap models at the same call
    sites:

        - `fit(X_train, y_train, X_val, y_val, extra_frames=None)` ->
          self
        - `predict_proba(X)` -> ndarray shape `(n, 2)`
        - `save(path)` -> `(model_path, manifest_path)`
        - `load(path)` classmethod

    Fitted state (all `None` pre-fit):
        module_:  Fitted `FraudNet`.
        card1_vocab_, addr1_vocab_, deviceinfo_vocab_:
            value -> index mappings (index 0 = OOV).
        numeric_cols_: list of numeric column names from training.
        scaler_: `StandardScaler` fit on train numerics.
        numeric_median_: per-column train medians used to impute
            NaN at both fit and predict time (no per-frame leak).
        val_auc_history_: per-epoch val AUC, len = epochs run.
        train_loss_history_: per-epoch train loss, same length.
        best_epoch_: 1-based epoch index of the best val AUC.
        best_val_auc_: float, the best val AUC observed.
        early_stopped_: True iff training halted via early stopping.
    """

    def __init__(  # noqa: PLR0913 — every kwarg is a hyperparameter the script's CLI exposes; bundling them into a Config dataclass would hide tuning surface
        self,
        embed_dim: int = _DEFAULT_EMBED_DIM,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        dropout: float = _DEFAULT_DROPOUT,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_epochs: int = _DEFAULT_MAX_EPOCHS,
        lr: float = _DEFAULT_LR,
        weight_decay: float = _DEFAULT_WEIGHT_DECAY,
        early_stopping_patience: int = _DEFAULT_EARLY_STOPPING_PATIENCE,
        focal_alpha: float = _DEFAULT_FOCAL_ALPHA,
        focal_gamma: float = _DEFAULT_FOCAL_GAMMA,
        device: str = "cpu",
        random_state: int | None = None,
    ) -> None:
        """Construct the model with hyperparameter overrides.

        Args:
            embed_dim: Embedding dim per entity. Default 32 (per spec).
            hidden_dim: Width of the post-concat head + numeric branch.
                Default 64.
            dropout: Dropout probability after each ReLU. Default 0.3.
            batch_size: Mini-batch size for SGD. Default 2048.
            max_epochs: Cap on training epochs. Default 50; early
                stopping typically halts well below.
            lr: Adam learning rate. Default 1e-3.
            weight_decay: Adam L2 weight decay. Default 1e-5.
            early_stopping_patience: Halt after this many epochs
                without val-AUC improvement. Default 5.
            focal_alpha: Focal-loss positive-class weight. Default 0.25.
            focal_gamma: Focal-loss focusing parameter. Default 2.0.
            device: Torch device. Default "cpu".
            random_state: Seed for torch + numpy. If None, uses
                `Settings.seed`.

        Raises:
            ValueError: If any numeric arg has an out-of-bounds value.
        """
        if batch_size < 1:
            raise ValueError(f"FraudNetModel: batch_size must be >= 1, got {batch_size}")
        if max_epochs < 1:
            raise ValueError(f"FraudNetModel: max_epochs must be >= 1, got {max_epochs}")
        if lr <= 0:
            raise ValueError(f"FraudNetModel: lr must be > 0, got {lr}")
        if early_stopping_patience < 1:
            raise ValueError(
                f"FraudNetModel: early_stopping_patience must be >= 1, "
                f"got {early_stopping_patience}"
            )
        settings = get_settings()
        self.embed_dim: int = int(embed_dim)
        self.hidden_dim: int = int(hidden_dim)
        self.dropout: float = float(dropout)
        self.batch_size: int = int(batch_size)
        self.max_epochs: int = int(max_epochs)
        self.lr: float = float(lr)
        self.weight_decay: float = float(weight_decay)
        self.early_stopping_patience: int = int(early_stopping_patience)
        self.focal_alpha: float = float(focal_alpha)
        self.focal_gamma: float = float(focal_gamma)
        self.device: str = str(device)
        self.random_state: int = (
            int(random_state) if random_state is not None else int(settings.seed)
        )

        # Fitted state.
        self.module_: FraudNet | None = None
        self.card1_vocab_: dict[Any, int] | None = None
        self.addr1_vocab_: dict[Any, int] | None = None
        self.deviceinfo_vocab_: dict[Any, int] | None = None
        self.numeric_cols_: list[str] | None = None
        self.scaler_: StandardScaler | None = None
        # Train-time median per numeric column, used at predict time
        # so we never impute val/test with their own statistics
        # (that would be a future-data leak).
        self.numeric_median_: pd.Series[Any] | None = None
        self.val_auc_history_: list[float] = []
        self.train_loss_history_: list[float] = []
        self.best_epoch_: int | None = None
        self.best_val_auc_: float | None = None
        self.early_stopped_: bool | None = None

    # -----------------------------------------------------------------
    # Vocab + scaler.
    # -----------------------------------------------------------------

    def _build_vocabs(
        self,
        train_df: pd.DataFrame,
        extra_frames: tuple[pd.DataFrame, ...] = (),
    ) -> None:
        """Build entity vocabularies from train + extra frames.

        Trade-off: building from train + val + test (transductive) is
        chosen by the caller via `extra_frames`. Train-only would
        push 30-50 % of val/test IDs into OOV, destroying signal on
        a fixed dataset where we know the ID surface in advance.
        """
        for entity in ENTITY_COLUMNS:
            if entity not in train_df.columns:
                raise KeyError(
                    f"FraudNetModel.fit: training frame missing entity column " f"{entity!r}"
                )
        union = pd.concat(
            [train_df[list(ENTITY_COLUMNS)]] + [df[list(ENTITY_COLUMNS)] for df in extra_frames],
            axis=0,
            ignore_index=True,
        )
        self.card1_vocab_ = _build_vocab(union["card1"])
        self.addr1_vocab_ = _build_vocab(union["addr1"])
        self.deviceinfo_vocab_ = _build_vocab(union["DeviceInfo"])

    def _select_numeric_columns(self, df: pd.DataFrame) -> list[str]:
        """Pick numeric feature columns: not entity IDs, not object/string.

        Mirrors `train_lightgbm._select_features` philosophy but also
        excludes the three categorical entities so they only enter
        the model through embeddings.
        """
        excluded = set(ENTITY_COLUMNS)
        cols: list[str] = []
        for col in df.columns:
            if col in excluded:
                continue
            dtype = df[col].dtype
            if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                continue
            cols.append(col)
        return cols

    def _to_tensors(
        self, df: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map a DataFrame to (card1_ids, addr1_ids, device_ids, X_num).

        Assumes the model is fitted (vocabs + scaler + numeric_cols
        are populated).
        """
        if (
            self.card1_vocab_ is None
            or self.addr1_vocab_ is None
            or self.deviceinfo_vocab_ is None
            or self.numeric_cols_ is None
            or self.scaler_ is None
            or self.numeric_median_ is None
        ):
            raise AttributeError(
                "FraudNetModel._to_tensors called before fit completed "
                "(vocabs / scaler / numeric_cols / median not populated)"
            )
        card1 = _map_to_indices(df["card1"], self.card1_vocab_)
        addr1 = _map_to_indices(df["addr1"], self.addr1_vocab_)
        device = _map_to_indices(df["DeviceInfo"], self.deviceinfo_vocab_)

        # Median-impute with the TRAIN median (no future-data leak),
        # then standardize, then nan_to_num as a defensive belt for
        # the rare case where a train column was entirely null
        # (median itself NaN). All in float32 to keep RSS tight at
        # the IEEE-CIS scale (414k * 744 columns -> ~1.2 GB rather
        # than ~2.4 GB at float64).
        x_num_arr = (
            df[self.numeric_cols_]
            .astype(np.float32, copy=False)
            .fillna(self.numeric_median_.astype(np.float32))
            .to_numpy(dtype=np.float32, copy=False)
        )
        x_num_scaled = self.scaler_.transform(x_num_arr).astype(np.float32, copy=False)
        np.nan_to_num(x_num_scaled, nan=0.0, copy=False)

        return (
            torch.from_numpy(card1),
            torch.from_numpy(addr1),
            torch.from_numpy(device),
            torch.from_numpy(x_num_scaled),
        )

    # -----------------------------------------------------------------
    # Fit.
    # -----------------------------------------------------------------

    def fit(  # noqa: PLR0915 — single-pass orchestration: vocab build + scaler + tensorise + train loop + early-stop bookkeeping. Splitting would obscure data-flow.
        self,
        X_train: pd.DataFrame,  # noqa: N803 — sklearn convention
        y_train: pd.Series[int] | np.ndarray[Any, Any],
        X_val: pd.DataFrame,  # noqa: N803 — sklearn convention
        y_val: pd.Series[int] | np.ndarray[Any, Any],
        extra_frames: tuple[pd.DataFrame, ...] = (),
    ) -> Self:
        """Fit the network with focal loss + early stopping on val AUC.

        Args:
            X_train: Training features. Must contain `card1`, `addr1`,
                `DeviceInfo` plus numerics.
            y_train: Binary labels (0 / 1).
            X_val: Validation features (same columns as X_train).
            y_val: Validation labels.
            extra_frames: Extra DataFrames whose entity-ID columns
                contribute to the transductive vocab. Passing
                `(X_val, X_test)` is the standard usage. Labels are
                NOT pulled — only `card1`/`addr1`/`DeviceInfo`.

        Returns:
            self, fitted in place.

        Raises:
            ValueError: If column sets disagree between X_train and
                X_val, or if y_train has only one class.
            KeyError: If any entity column is missing.
        """
        if list(X_train.columns) != list(X_val.columns):
            raise ValueError(
                "FraudNetModel.fit: X_train and X_val must have " "identical column names and order"
            )
        y_train_arr = np.asarray(y_train).reshape(-1).astype(np.float32)
        y_val_arr = np.asarray(y_val).reshape(-1).astype(np.float32)
        if len(np.unique(y_train_arr)) < _N_CLASS_PROBS:
            raise ValueError(
                f"FraudNetModel.fit: y_train must contain both classes; "
                f"got unique values {np.unique(y_train_arr).tolist()}"
            )

        self._build_vocabs(X_train, extra_frames)
        self.numeric_cols_ = self._select_numeric_columns(X_train)
        # Fit StandardScaler on train numerics. Imputation strategy:
        # train-median fillna -> standardize -> defensive nan_to_num
        # in `_to_tensors`. The train median is stashed on self so
        # predict-time imputation uses the same statistics (no
        # future-data leak via per-frame medians).
        # Float32 throughout: keeps peak RSS to ~5 GB on full IEEE-CIS
        # rather than the ~12+ GB the equivalent float64 path would
        # need (each .astype copy is the dominant cost).
        x_num_train_df = X_train[self.numeric_cols_].astype(np.float32, copy=False)
        self.numeric_median_ = x_num_train_df.median()
        x_num_train = x_num_train_df.fillna(self.numeric_median_.astype(np.float32)).to_numpy(
            dtype=np.float32, copy=False
        )
        self.scaler_ = StandardScaler().fit(x_num_train)
        del x_num_train_df, x_num_train  # release the staging copies

        # Tensorise once for the full train + val sets; fits in RAM
        # at IEEE-CIS scale (414k × ~743 floats × 4 bytes ≈ 1.2 GB).
        # If memory becomes a bottleneck, switch to a chunked
        # IterableDataset; for Sprint 3 the all-in-RAM path is fine.
        c_tr, a_tr, d_tr, x_tr = self._to_tensors(X_train)
        c_va, a_va, d_va, x_va = self._to_tensors(X_val)
        y_tr = torch.from_numpy(y_train_arr)

        # Determinism: seed torch + numpy. CPU-only sidesteps the
        # cudnn determinism gotchas; we don't enable
        # `use_deterministic_algorithms(True)` because not every
        # operator supports it, and silent fallback is worse than
        # explicit best-effort seeding.
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        device = torch.device(self.device)
        module = FraudNet(
            card1_vocab_size=len(self.card1_vocab_ or {}),
            addr1_vocab_size=len(self.addr1_vocab_ or {}),
            deviceinfo_vocab_size=len(self.deviceinfo_vocab_ or {}),
            n_numeric=len(self.numeric_cols_),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(device)

        criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma).to(device)
        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        train_ds = TensorDataset(c_tr, a_tr, d_tr, x_tr, y_tr)
        # Generator pinned for deterministic shuffling.
        gen = torch.Generator()
        gen.manual_seed(self.random_state)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # in-process; deterministic + portable
            generator=gen,
        )

        best_val_auc = -1.0
        best_state: dict[str, Any] | None = None
        best_epoch = 0
        no_improve_streak = 0
        early_stopped = False

        self.val_auc_history_ = []
        self.train_loss_history_ = []

        for epoch in range(1, self.max_epochs + 1):
            module.train()
            epoch_loss_sum = 0.0
            n_batches = 0
            for c_b, a_b, d_b, x_b, y_b in train_loader:
                c_dev = c_b.to(device)
                a_dev = a_b.to(device)
                d_dev = d_b.to(device)
                x_dev = x_b.to(device)
                y_dev = y_b.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = module(c_dev, a_dev, d_dev, x_dev)
                loss = criterion(logits, y_dev)
                loss.backward()
                optimizer.step()
                epoch_loss_sum += float(loss.item())
                n_batches += 1
            epoch_loss = epoch_loss_sum / max(n_batches, 1)

            # Val pass (no grad).
            module.eval()
            with torch.no_grad():
                val_logits = module(
                    c_va.to(device),
                    a_va.to(device),
                    d_va.to(device),
                    x_va.to(device),
                )
                val_proba = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = _safe_auc(y_val_arr, val_proba)

            self.train_loss_history_.append(float(epoch_loss))
            self.val_auc_history_.append(float(val_auc))

            _logger.info(
                "fraudnet.epoch_done",
                epoch=epoch,
                train_loss=round(epoch_loss, 6),
                val_auc=round(val_auc, 6),
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
                no_improve_streak = 0
            else:
                no_improve_streak += 1
                if no_improve_streak >= self.early_stopping_patience:
                    early_stopped = True
                    _logger.info(
                        "fraudnet.early_stop",
                        epoch=epoch,
                        best_epoch=best_epoch,
                        best_val_auc=round(best_val_auc, 6),
                    )
                    break

        # Restore the best-by-val-AUC weights before exposing the module.
        if best_state is not None:
            module.load_state_dict(best_state)
        module.eval()

        self.module_ = module
        self.best_epoch_ = int(best_epoch)
        self.best_val_auc_ = float(best_val_auc)
        self.early_stopped_ = bool(early_stopped)
        return self

    # -----------------------------------------------------------------
    # Predict.
    # -----------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray[Any, Any]:  # noqa: N803
        """Return per-row class probabilities `(n, 2)`.

        Mirrors `LightGBMFraudModel.predict_proba` exactly: stack
        `[1 - p, p]` so downstream `proba[:, 1]` indexing works
        without modification.

        Args:
            X: Frame with the entity columns + numeric features.

        Returns:
            float64 ndarray of shape `(len(X), 2)`. Each row sums to 1.

        Raises:
            AttributeError: If called pre-fit.
        """
        if self.module_ is None or self.numeric_cols_ is None:
            raise AttributeError("FraudNetModel must be fit before predict_proba")
        if len(X) == 0:
            return np.empty((0, _N_CLASS_PROBS), dtype=np.float64)
        device = torch.device(self.device)
        c, a, d, x = self._to_tensors(X)
        self.module_.eval()
        with torch.no_grad():
            logits = self.module_(c.to(device), a.to(device), d.to(device), x.to(device))
            p_pos = torch.sigmoid(logits).cpu().numpy().astype(np.float64).reshape(-1)
        p_neg = 1.0 - p_pos
        return np.column_stack([p_neg, p_pos])

    # -----------------------------------------------------------------
    # Save / load.
    # -----------------------------------------------------------------

    def save(self, path: Path) -> tuple[Path, Path]:
        """Persist the fitted model + manifest under `path/`.

        - `path/neural_model.pt` — joblib payload of the full
          `FraudNetModel` instance (carries the fitted FraudNet).
        - `path/neural_model_manifest.json` — sidecar with hparams,
          vocab sizes, n_numeric, best_epoch, best_val_auc, schema
          + content hashes.

        Args:
            path: Destination directory. Created if missing.

        Returns:
            `(model_path, manifest_path)`.

        Raises:
            AttributeError: If called pre-fit.
        """
        if self.module_ is None:
            raise AttributeError("FraudNetModel must be fit before save")
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / _MODEL_FILENAME
        manifest_path = path / _MANIFEST_FILENAME

        # Move module to CPU for portable serialization, restore after.
        original_device = self.device
        self.module_.to("cpu")
        try:
            joblib.dump(self, model_path)
        finally:
            self.module_.to(original_device)

        content_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        manifest = self._build_manifest(content_hash=content_hash)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return model_path, manifest_path

    @classmethod
    def load(cls, path: Path) -> Self:
        """Inverse of `save`. Reads `path/neural_model.pt`.

        Args:
            path: Directory containing the saved model.

        Returns:
            The reconstructed `FraudNetModel`.

        Raises:
            FileNotFoundError: If `path/neural_model.pt` does not exist.
            TypeError: If the joblib payload is not a `FraudNetModel`.
        """
        model_path = path / _MODEL_FILENAME
        loaded = joblib.load(model_path)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"Loaded object at {model_path} is "
                f"{type(loaded).__name__}, expected FraudNetModel"
            )
        return loaded

    # -----------------------------------------------------------------
    # Manifest.
    # -----------------------------------------------------------------

    def _build_manifest(self, content_hash: str) -> dict[str, Any]:
        """Render the manifest dict (called from `save`)."""
        if (
            self.numeric_cols_ is None
            or self.card1_vocab_ is None
            or self.addr1_vocab_ is None
            or self.deviceinfo_vocab_ is None
            or self.best_epoch_ is None
            or self.best_val_auc_ is None
        ):
            raise AttributeError("FraudNetModel._build_manifest called before fit")
        return {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "model_class": "FraudNetModel",
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "early_stopping_patience": self.early_stopping_patience,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "random_state": self.random_state,
            "card1_vocab_size": len(self.card1_vocab_),
            "addr1_vocab_size": len(self.addr1_vocab_),
            "deviceinfo_vocab_size": len(self.deviceinfo_vocab_),
            "n_numeric": len(self.numeric_cols_),
            "numeric_cols_count": len(self.numeric_cols_),
            "best_epoch": self.best_epoch_,
            "best_val_auc": self.best_val_auc_,
            "early_stopped": bool(self.early_stopped_),
            "epochs_run": len(self.val_auc_history_),
            "schema_hash": _schema_fingerprint(self.numeric_cols_),
            "content_hash": content_hash,
        }


# ---------------------------------------------------------------------
# Internal helpers (test-importable).
# ---------------------------------------------------------------------


def _safe_auc(y_true: np.ndarray[Any, Any], y_score: np.ndarray[Any, Any]) -> float:
    """ROC-AUC with the single-class fallback returning 0.5.

    Imported here rather than `sklearn.metrics.roc_auc_score` directly
    so the training loop tolerates a degenerate val window where
    only one class is present (vanishingly rare at IEEE-CIS scale,
    but possible on tiny smoke fixtures).
    """
    from sklearn.metrics import roc_auc_score  # local — avoid sklearn at import time

    if len(np.unique(y_true)) < _N_CLASS_PROBS:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


__all__ = ["ENTITY_COLUMNS", "FocalLoss", "FraudNet", "FraudNetModel"]
