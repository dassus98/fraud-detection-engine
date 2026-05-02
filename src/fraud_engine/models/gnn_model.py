"""FraudGNN GraphSAGE network (Sprint 3 Model C).

`FraudGNN` is the project's third diversity model: a 3-layer GraphSAGE
network operating over the bipartite transaction-entity graph that
3.2.b's `TransactionEntityGraph` defined. Per CLAUDE.md §3 it is
**batch-only** — its outputs feed Model A (LightGBM) as features in
Sprint 5; it does not bear a real-time serving contract itself.

Business rationale:
    Tabular tier features (T1-T4) and entity embeddings (3.4.a Model B)
    each capture per-entity signal in isolation. They cannot represent
    the *graph topology* of organised fraud rings — many cards
    sharing one device, one address chain reused across new accounts,
    etc. GraphSAGE's neighbor-aggregation framework lets each
    transaction's representation incorporate the average behaviour of
    transactions sharing its card / address / device. That signal is
    structurally invisible to the per-entity tier features, which is
    exactly what makes Model C an ensemble-diversity asset.

Trade-offs considered:
    - **Homogeneous `Data` (not `HeteroData`).** Both transaction and
      entity nodes live in the same `x` tensor. Transaction rows
      carry the Tier-5 numeric feature vector (~740 cols);
      entity rows are zero-padded. This sidesteps the
      `to_hetero(...)` ceremony, keeps `SAGEConv` the simplest
      possible, and makes the integration test debuggable. Cost:
      entity rows waste memory on the zero-padding (~14k entity
      nodes × 740 × 4 B ≈ 41 MB at IEEE-CIS scale — acceptable).
    - **Bipartite as undirected.** Each `(txn ↔ entity)` edge is
      added in both directions in the `edge_index` so SAGEConv's
      mean-aggregation flows both ways. Standard PyG convention.
    - **Transductive node-classification with masks** (rather than
      inductive scoring). The graph is built ONCE from train +
      val + test entities. Train mask = train txn nodes; val mask =
      val; test mask = test. Loss is masked. Labels for val/test
      are NEVER passed to training — only the topology + features
      are visible (this is the textbook PyG node-classification
      pattern). CLAUDE.md §3 calls Model C "batch-only", which is
      exactly the transductive contract.
    - **3-layer + `num_neighbors=(10, 10, 10)` per spec**
      ("GraphSAGE 3-layer. Neighbor sampling."). 10 neighbors at
      each hop is the GraphSAGE-paper default.
    - **NeighborLoader for batch training** (per spec). Each batch
      samples a subgraph rooted at `batch_size` train txn nodes,
      expanding `num_neighbors` per layer.
    - **Float32 throughout** (lessons from 3.4.a). Numerics
      downcast at tensorisation; train-time median imputation
      reused at predict.
    - **Focal loss reused from `neural_model.py`** — single source
      of truth for the FL formula.
    - **Cached node logits at fit-time.** After training, the model
      runs ONE full-graph forward pass and stores per-node logits
      on `self.cached_node_logits_`. `predict_proba(df)` becomes
      a TransactionID -> index lookup + sigmoid. Sprint 5 (real-time
      serving) would replace this with subgraph-extraction-per-
      request; Sprint 3 (batch-only per CLAUDE.md §3) doesn't need
      that yet.
    - **`predict_proba(df)` looks up by `TransactionID`** rather
      than re-building the graph. Sprint-3 contract is "batch-only,
      score the parquets". Unknown TransactionIDs raise `KeyError`;
      Sprint 5+ adds inductive support.
    - **Joblib full-instance persistence** with PyG `Data` baked in
      (mirrors `LightGBMFraudModel` / `FraudNetModel`). The bundle
      carries the training graph + features so `predict_proba`
      works after `load`. Cost: ~1.2-1.5 GB at full IEEE-CIS scale
      (gitignored under `models/`).
    - **`scale_pos_weight` not used** — focal loss is the imbalance
      handler.

Cross-references:
    - `src/fraud_engine/models/neural_model.py` — `FraudNet` /
      `FraudNetModel` pattern; `FocalLoss` is imported from there.
    - `src/fraud_engine/features/tier5_graph.py` —
      `TransactionEntityGraph` is the conceptual template (we build
      a PyG `edge_index` directly rather than going via networkx for
      memory efficiency).
    - `docs/ADR/0001-tech-stack.md:62-74` — PyTorch / PyG scoped to
      diversity models.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Self

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

from fraud_engine.config.settings import get_settings
from fraud_engine.models.neural_model import FocalLoss
from fraud_engine.utils.logging import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Persistence filenames; mirrors LightGBMFraudModel / FraudNetModel naming.
_MODEL_FILENAME: Final[str] = "gnn_model.pt"
_MANIFEST_FILENAME: Final[str] = "gnn_model_manifest.json"

# Manifest schema version; bump on incompatible JSON shape changes.
_MANIFEST_SCHEMA_VERSION: Final[int] = 1

# Schema-fingerprint truncation length, mirrors `data/lineage.py`.
_SCHEMA_FINGERPRINT_HEX_CHARS: Final[int] = 16

# Number of class probability columns returned by `predict_proba`.
_N_CLASS_PROBS: Final[int] = 2

# Entity columns embedded in the graph as node-types. Mirrors the
# `TransactionEntityGraph` entity surface (3.2.b) but excludes
# `P_emaildomain` to keep the graph bipartite-density manageable
# (P_emaildomain has 59 unique values × 414k txns -> ~7k edges per
# domain, which would make hub-entity nodes dominate the message-pass).
_ENTITY_COLUMNS: Final[tuple[str, ...]] = ("card1", "addr1", "DeviceInfo")

# Default architecture hyperparameters.
_DEFAULT_HIDDEN_DIM: Final[int] = 64
_DEFAULT_DROPOUT: Final[float] = 0.3
_DEFAULT_NUM_LAYERS: Final[int] = 3  # spec: "GraphSAGE 3-layer"

# Default neighbor-sampling fan-out per layer. Per GraphSAGE paper.
_DEFAULT_NUM_NEIGHBORS: Final[tuple[int, int, int]] = (10, 10, 10)

# Default training hyperparameters.
_DEFAULT_BATCH_SIZE: Final[int] = 1024
_DEFAULT_MAX_EPOCHS: Final[int] = 20
_DEFAULT_LR: Final[float] = 1e-3
_DEFAULT_WEIGHT_DECAY: Final[float] = 1e-5
_DEFAULT_EARLY_STOPPING_PATIENCE: Final[int] = 5

# Default focal-loss hyperparameters (mirrors FraudNetModel).
_DEFAULT_FOCAL_ALPHA: Final[float] = 0.25
_DEFAULT_FOCAL_GAMMA: Final[float] = 2.0

# Required-column contract.
_TRANSACTION_ID_COL: Final[str] = "TransactionID"
_TARGET_COL: Final[str] = "isFraud"


# ---------------------------------------------------------------------
# Network module.
# ---------------------------------------------------------------------


class FraudGNN(nn.Module):
    """3-layer GraphSAGE network for binary fraud node classification.

    Architecture:
        SAGEConv(in_channels, hidden_dim)  -> ReLU -> Dropout
        SAGEConv(hidden_dim, hidden_dim)   -> ReLU -> Dropout
        SAGEConv(hidden_dim, hidden_dim)   -> ReLU -> Dropout
        Linear(hidden_dim, 1)              -> per-node logit

    Forward signature:
        forward(x, edge_index) -> logits  # shape (n_nodes,)

    Caller masks out non-txn / non-train nodes before computing loss.

    Trade-offs considered:
        - **Mean aggregator** (SAGEConv default `aggr="mean"`).
          Simpler than max/lstm; standard GraphSAGE.
        - **No `normalize=True`** — let BatchNorm handle scale; the
          `normalize` flag in SAGEConv L2-normalises the output which
          can collapse the gradient signal at small hidden_dim.
        - **3 layers per spec.** Sprint 4+ may experiment with 2 or 4.
        - **Single output unit** (per-node logit). BCEWithLogits-style
          via FocalLoss in the caller.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        dropout: float = _DEFAULT_DROPOUT,
        num_layers: int = _DEFAULT_NUM_LAYERS,
    ) -> None:
        """Construct the FraudGNN module.

        Args:
            in_channels: Number of input feature dimensions per node.
            hidden_dim: Hidden width for every SAGEConv layer. Default 64.
            dropout: Dropout probability after each ReLU. Default 0.3.
            num_layers: Number of SAGEConv layers. Spec: 3.

        Raises:
            ValueError: If `in_channels`, `hidden_dim`, or `num_layers`
                are out of valid range.
        """
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"FraudGNN: in_channels must be >= 1, got {in_channels}")
        if hidden_dim < 1:
            raise ValueError(f"FraudGNN: hidden_dim must be >= 1, got {hidden_dim}")
        if num_layers < 1:
            raise ValueError(f"FraudGNN: num_layers must be >= 1, got {num_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"FraudGNN: dropout must be in [0, 1), got {dropout}")

        self.in_channels: int = int(in_channels)
        self.hidden_dim: int = int(hidden_dim)
        self.num_layers: int = int(num_layers)
        self.dropout_p: float = float(dropout)

        layers: list[nn.Module] = []
        prev = in_channels
        for _ in range(num_layers):
            layers.append(SAGEConv(prev, hidden_dim))
            prev = hidden_dim
        self.convs = nn.ModuleList(layers)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning per-node logits.

        Args:
            x: Float tensor `(n_nodes, in_channels)`.
            edge_index: Long tensor `(2, n_edges)` of (src, dst) pairs.

        Returns:
            Float tensor `(n_nodes,)` of pre-sigmoid logits.
        """
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.relu(h)
            h = self.dropout(h)
        return self.head(h).reshape(-1)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------
# FraudGNNModel — sklearn-style wrapper.
# ---------------------------------------------------------------------


def _schema_fingerprint(columns: list[str]) -> str:
    """SHA-256 of a sorted column list, hex-truncated."""
    schema_str = json.dumps(sorted(columns), separators=(",", ":"))
    return hashlib.sha256(schema_str.encode("utf-8")).hexdigest()[:_SCHEMA_FINGERPRINT_HEX_CHARS]


def _select_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Pick numeric feature columns: not entity IDs, not non-features, not object/string.

    Mirrors `train_neural._select_columns_for_fraudnet` minus the entity
    columns (which the GNN consumes structurally as nodes, not as numeric
    features).
    """
    excluded = {_TRANSACTION_ID_COL, _TARGET_COL, "TransactionDT", "timestamp"}
    excluded.update(_ENTITY_COLUMNS)
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        dtype = df[col].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            continue
        cols.append(col)
    return cols


class FraudGNNModel:
    """Sklearn-style wrapper owning graph + scaler + the FraudGNN module.

    Public API mirrors `LightGBMFraudModel` / `FraudNetModel`:
        - `fit(X_train, y_train, X_val, y_val, *, X_test=None)` -> Self
        - `predict_proba(X)` -> ndarray shape `(n, 2)` (looks up by
          `TransactionID`; raises `KeyError` on unknown IDs)
        - `save(path)` -> `(model_path, manifest_path)`
        - `load(path)` classmethod

    Fitted state (all `None` pre-fit):
        module_:  Fitted `FraudGNN`.
        data_:    PyG `Data` object (homogeneous; carries node features,
                  edge index, train/val/test masks, n_txn_nodes split).
        txn_index_: dict mapping `TransactionID` -> graph node index.
        entity_index_: dict mapping `(entity_col, value)` -> graph node
                       index (index space starts after the txn nodes).
        scaler_, numeric_median_, numeric_cols_:
                  Preprocessing state (mirrors FraudNetModel).
        cached_node_logits_: ndarray `(n_total_nodes,)` of post-fit
                             forward-pass logits. predict_proba is
                             lookup + sigmoid on this cache.
        val_auc_history_, train_loss_history_, best_epoch_,
        best_val_auc_, early_stopped_: training trajectory.
    """

    def __init__(  # noqa: PLR0913 — every kwarg is a hyperparameter the script's CLI exposes
        self,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
        dropout: float = _DEFAULT_DROPOUT,
        num_layers: int = _DEFAULT_NUM_LAYERS,
        num_neighbors: tuple[int, ...] = _DEFAULT_NUM_NEIGHBORS,
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
            hidden_dim: SAGEConv hidden width. Default 64.
            dropout: Dropout probability. Default 0.3.
            num_layers: Number of SAGEConv layers. Default 3 (per spec).
            num_neighbors: Per-layer neighbor-sampling fan-out. Length
                must equal `num_layers`. Default (10, 10, 10).
            batch_size: NeighborLoader root-node batch size. Default 1024.
            max_epochs: Cap on training epochs. Default 20.
            lr: Adam learning rate. Default 1e-3.
            weight_decay: Adam L2 weight decay. Default 1e-5.
            early_stopping_patience: Halt after N epochs without val-AUC
                improvement. Default 5.
            focal_alpha, focal_gamma: Focal-loss hyperparameters. Same
                defaults as FraudNetModel (0.25, 2.0).
            device: Torch device. Default "cpu".
            random_state: Seed for torch + numpy. If None, uses
                `Settings.seed`.

        Raises:
            ValueError: If any numeric arg is out of valid range, or if
                `len(num_neighbors) != num_layers`.
        """
        if batch_size < 1:
            raise ValueError(f"FraudGNNModel: batch_size must be >= 1, got {batch_size}")
        if max_epochs < 1:
            raise ValueError(f"FraudGNNModel: max_epochs must be >= 1, got {max_epochs}")
        if lr <= 0:
            raise ValueError(f"FraudGNNModel: lr must be > 0, got {lr}")
        if early_stopping_patience < 1:
            raise ValueError(
                f"FraudGNNModel: early_stopping_patience must be >= 1, "
                f"got {early_stopping_patience}"
            )
        if len(num_neighbors) != num_layers:
            raise ValueError(
                f"FraudGNNModel: len(num_neighbors)={len(num_neighbors)} "
                f"must equal num_layers={num_layers}"
            )
        settings = get_settings()
        self.hidden_dim: int = int(hidden_dim)
        self.dropout: float = float(dropout)
        self.num_layers: int = int(num_layers)
        self.num_neighbors: tuple[int, ...] = tuple(int(n) for n in num_neighbors)
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
        self.module_: FraudGNN | None = None
        self.data_: Data | None = None
        self.txn_index_: dict[int, int] | None = None
        self.entity_index_: dict[tuple[str, Any], int] | None = None
        self.numeric_cols_: list[str] | None = None
        self.scaler_: StandardScaler | None = None
        self.numeric_median_: pd.Series[Any] | None = None
        self.cached_node_logits_: np.ndarray[Any, Any] | None = None
        self.n_txn_nodes_: int | None = None
        self.n_entity_nodes_: int | None = None
        self.n_edges_: int | None = None
        self.val_auc_history_: list[float] = []
        self.train_loss_history_: list[float] = []
        self.best_epoch_: int | None = None
        self.best_val_auc_: float | None = None
        self.early_stopped_: bool | None = None

    # -----------------------------------------------------------------
    # Graph construction helpers.
    # -----------------------------------------------------------------

    @staticmethod
    def _build_entity_index(
        frames: tuple[pd.DataFrame, ...],
    ) -> dict[tuple[str, Any], int]:
        """Build a deterministic `(entity_col, value) -> int` index.

        Combines all input frames' entity columns; sorts uniques per
        entity column for index stability across re-runs. Indices start
        at 0 and are local to entity nodes (the caller offsets by
        `n_txn_nodes` to get global graph node indices).
        """
        index: dict[tuple[str, Any], int] = {}
        next_idx = 0
        for entity_col in _ENTITY_COLUMNS:
            uniques: set[Any] = set()
            for df in frames:
                if entity_col not in df.columns:
                    raise KeyError(f"FraudGNNModel: frame missing entity column {entity_col!r}")
                vals = df[entity_col].dropna()
                uniques.update(vals.tolist())
            try:
                sorted_uniques = sorted(uniques)
            except TypeError:
                sorted_uniques = list(uniques)
            for value in sorted_uniques:
                index[(entity_col, value)] = next_idx
                next_idx += 1
        return index

    def _build_edge_index(
        self,
        df: pd.DataFrame,
        txn_offset: int,
        entity_offset: int,
    ) -> torch.Tensor:
        """Vectorised construction of `(2, n_edges_undirected)` long tensor.

        Each row in `df` contributes up to len(_ENTITY_COLUMNS) forward
        edges (one per non-null entity column). The caller passes
        `entity_offset = n_txn_nodes` so entity indices land in the
        global node-index space. Returns the **undirected** edge index
        (forward + reverse stacked).
        """
        if self.entity_index_ is None:
            raise AttributeError("entity_index_ must be populated before _build_edge_index")
        all_src: list[np.ndarray[Any, Any]] = []
        all_dst: list[np.ndarray[Any, Any]] = []
        n_rows = len(df)
        row_idx = np.arange(n_rows) + txn_offset
        for entity_col in _ENTITY_COLUMNS:
            # Map each row's entity value to the entity-local index, then
            # offset to the global graph node index. Build a per-entity
            # value-to-index dict slice so `Series.map` accepts a Mapping
            # (mypy-clean) rather than a Callable[[Any, Any], ...].
            entity_local_map: dict[Any, int] = {
                value: idx for (col, value), idx in self.entity_index_.items() if col == entity_col
            }
            mapped = df[entity_col].map(entity_local_map)
            valid_mask = mapped.notna().to_numpy()
            if not valid_mask.any():
                continue
            src = row_idx[valid_mask]
            dst = mapped[valid_mask].to_numpy(dtype=np.int64) + entity_offset
            all_src.append(src.astype(np.int64))
            all_dst.append(dst)
        if not all_src:
            # Degenerate: no edges (shouldn't happen at IEEE-CIS scale).
            return torch.zeros((2, 0), dtype=torch.long)
        src_concat = np.concatenate(all_src)
        dst_concat = np.concatenate(all_dst)
        forward = np.stack([src_concat, dst_concat])
        # Undirected: stack forward + reverse.
        return torch.from_numpy(np.concatenate([forward, forward[::-1]], axis=1))

    def _featurise_numerics(
        self, df: pd.DataFrame, fit_scaler: bool = False
    ) -> np.ndarray[Any, Any]:
        """Return float32 standardised numeric matrix `(n_rows, n_numeric)`.

        Float32 throughout (lessons from 3.4.a). Train-time median
        imputation for NaN; predict-time reuses the fitted median.
        """
        if self.numeric_cols_ is None:
            raise AttributeError("numeric_cols_ must be populated before _featurise_numerics")
        x_num_df = df[self.numeric_cols_].astype(np.float32, copy=False)
        if fit_scaler:
            self.numeric_median_ = x_num_df.median()
        if self.numeric_median_ is None:
            raise AttributeError("numeric_median_ must be populated before _featurise_numerics")
        x_num_arr = x_num_df.fillna(self.numeric_median_.astype(np.float32)).to_numpy(
            dtype=np.float32, copy=False
        )
        if fit_scaler:
            self.scaler_ = StandardScaler().fit(x_num_arr)
        if self.scaler_ is None:
            raise AttributeError("scaler_ must be populated before _featurise_numerics")
        x_num_scaled: np.ndarray[Any, Any] = self.scaler_.transform(x_num_arr).astype(
            np.float32, copy=False
        )
        np.nan_to_num(x_num_scaled, nan=0.0, copy=False)
        return x_num_scaled

    # -----------------------------------------------------------------
    # Fit.
    # -----------------------------------------------------------------

    def fit(  # noqa: PLR0913, PLR0915, PLR0912 — single-pass orchestration: graph build + scaler + tensorise + train loop + early-stop. Splitting would obscure data-flow.
        self,
        X_train: pd.DataFrame,  # noqa: N803 — sklearn convention
        y_train: pd.Series[int] | np.ndarray[Any, Any],
        X_val: pd.DataFrame,  # noqa: N803
        y_val: pd.Series[int] | np.ndarray[Any, Any],
        X_test: pd.DataFrame | None = None,  # noqa: N803
        y_test: pd.Series[int] | np.ndarray[Any, Any] | None = None,
    ) -> Self:
        """Fit the GNN on the bipartite txn-entity graph with focal loss.

        The graph is built ONCE from train + val + (optional) test:
        - n_txn_nodes = len(train) + len(val) + len(test)
        - n_entity_nodes = sum of unique entity values across all
          frames per `_ENTITY_COLUMNS`
        - x = [train_features, val_features, test_features, zeros_for_entities]
        - edge_index = bipartite (txn ↔ entity), undirected
        - y_train mask labels train txns; val/test masks slice the
          remainder
        - Train loop: NeighborLoader rooted at train txn nodes,
          focal-loss step per batch, val AUC for early stopping
        - After training, ONE full-graph forward pass populates
          `self.cached_node_logits_` so `predict_proba` is fast.

        Args:
            X_train, y_train: Train features + labels. Must contain
                `TransactionID`, `isFraud`, all entity columns + numerics.
            X_val, y_val: Val features + labels (labels masked from loss).
            X_test, y_test: Optional test features + labels (also masked
                from training). If provided, test mask + AUC are
                populated.

        Returns:
            self, fitted in place.

        Raises:
            ValueError: If column sets disagree, `y_train` has only one
                class, or required columns are missing.
            KeyError: If any entity column is missing.
        """
        if list(X_train.columns) != list(X_val.columns):
            raise ValueError(
                "FraudGNNModel.fit: X_train and X_val must have identical " "column names and order"
            )
        if X_test is not None and list(X_test.columns) != list(X_train.columns):
            raise ValueError(
                "FraudGNNModel.fit: X_test must have identical column names "
                "and order if provided"
            )
        for col in (_TRANSACTION_ID_COL,) + _ENTITY_COLUMNS:
            if col not in X_train.columns:
                raise KeyError(f"FraudGNNModel.fit: training frame missing column {col!r}")

        y_train_arr = np.asarray(y_train).reshape(-1).astype(np.float32)
        y_val_arr = np.asarray(y_val).reshape(-1).astype(np.float32)
        y_test_arr: np.ndarray[Any, Any] | None = (
            np.asarray(y_test).reshape(-1).astype(np.float32) if y_test is not None else None
        )
        if len(np.unique(y_train_arr)) < _N_CLASS_PROBS:
            raise ValueError(
                f"FraudGNNModel.fit: y_train must contain both classes; "
                f"got unique values {np.unique(y_train_arr).tolist()}"
            )

        # --- Numeric column selection + scaler fit ---
        self.numeric_cols_ = _select_numeric_columns(X_train)
        # The featurise call uses fit_scaler=True so `self.scaler_` and
        # `self.numeric_median_` are populated based on TRAIN ONLY (no
        # val/test leak through statistics).
        train_features = self._featurise_numerics(X_train, fit_scaler=True)
        val_features = self._featurise_numerics(X_val, fit_scaler=False)
        test_features: np.ndarray[Any, Any] | None = (
            self._featurise_numerics(X_test, fit_scaler=False) if X_test is not None else None
        )

        # --- Entity-node index (transductive: includes val/test entities) ---
        frames: tuple[pd.DataFrame, ...] = (
            (X_train, X_val) if X_test is None else (X_train, X_val, X_test)
        )
        self.entity_index_ = self._build_entity_index(frames)

        # --- Build the homogeneous graph ---
        n_train = len(X_train)
        n_val = len(X_val)
        n_test = len(X_test) if X_test is not None else 0
        n_txn = n_train + n_val + n_test
        n_entity = len(self.entity_index_)
        n_total = n_txn + n_entity
        n_features = len(self.numeric_cols_)

        # Node-feature tensor: txn rows carry numeric features; entity
        # rows are zero-padded.
        x_full = np.zeros((n_total, n_features), dtype=np.float32)
        x_full[:n_train] = train_features
        x_full[n_train : n_train + n_val] = val_features
        if test_features is not None:
            x_full[n_train + n_val : n_txn] = test_features
        x_tensor = torch.from_numpy(x_full)

        # Edge-index assembly (forward + reverse).
        edge_train = self._build_edge_index(X_train, txn_offset=0, entity_offset=n_txn)
        edge_val = self._build_edge_index(X_val, txn_offset=n_train, entity_offset=n_txn)
        edge_components = [edge_train, edge_val]
        if X_test is not None:
            edge_test = self._build_edge_index(
                X_test, txn_offset=n_train + n_val, entity_offset=n_txn
            )
            edge_components.append(edge_test)
        edge_index = torch.cat(edge_components, dim=1)

        # Train/val/test masks over txn nodes only (entity nodes are
        # never targets of the loss).
        train_mask = torch.zeros(n_total, dtype=torch.bool)
        val_mask = torch.zeros(n_total, dtype=torch.bool)
        test_mask = torch.zeros(n_total, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        if X_test is not None:
            test_mask[n_train + n_val : n_txn] = True

        # Per-node label tensor (entity nodes get 0; loss masks them out).
        y_full = np.zeros(n_total, dtype=np.float32)
        y_full[:n_train] = y_train_arr
        y_full[n_train : n_train + n_val] = y_val_arr
        if y_test_arr is not None:
            y_full[n_train + n_val : n_txn] = y_test_arr
        y_tensor = torch.from_numpy(y_full)

        data = Data(
            x=x_tensor,
            edge_index=edge_index,
            y=y_tensor,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        self.data_ = data
        self.n_txn_nodes_ = int(n_txn)
        self.n_entity_nodes_ = int(n_entity)
        self.n_edges_ = int(edge_index.shape[1])

        # Per-spec TransactionID -> graph index lookup.
        train_ids = X_train[_TRANSACTION_ID_COL].astype(np.int64).to_numpy()
        val_ids = X_val[_TRANSACTION_ID_COL].astype(np.int64).to_numpy()
        test_ids = (
            X_test[_TRANSACTION_ID_COL].astype(np.int64).to_numpy()
            if X_test is not None
            else np.array([], dtype=np.int64)
        )
        all_ids = np.concatenate([train_ids, val_ids, test_ids])
        self.txn_index_ = {int(tid): int(i) for i, tid in enumerate(all_ids)}

        _logger.info(
            "fraudgnn.graph_built",
            n_txn=n_txn,
            n_entity=n_entity,
            n_edges=int(edge_index.shape[1]),
            n_features=n_features,
        )

        # --- Determinism ---
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # --- Model + optim + criterion ---
        device = torch.device(self.device)
        module = FraudGNN(
            in_channels=n_features,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            num_layers=self.num_layers,
        ).to(device)
        criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma).to(device)
        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # --- NeighborLoader for batch training (per spec) ---
        gen = torch.Generator()
        gen.manual_seed(self.random_state)
        train_input_nodes = torch.nonzero(train_mask, as_tuple=False).reshape(-1)
        train_loader = NeighborLoader(
            data,
            num_neighbors=list(self.num_neighbors),
            input_nodes=train_input_nodes,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            generator=gen,
        )

        # --- Train loop ---
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
            for batch in train_loader:
                batch_dev = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = module(batch_dev.x, batch_dev.edge_index)
                # NeighborLoader puts root nodes at the head of the batch;
                # `batch.batch_size` is the number of root nodes.
                root_logits = logits[: batch_dev.batch_size]
                root_y = batch_dev.y[: batch_dev.batch_size]
                loss = criterion(root_logits, root_y)
                loss.backward()
                optimizer.step()
                epoch_loss_sum += float(loss.item())
                n_batches += 1
            epoch_loss = epoch_loss_sum / max(n_batches, 1)

            # Val pass (full-graph forward; no neighbor sampling at eval).
            module.eval()
            with torch.no_grad():
                val_logits_full = module(data.x.to(device), data.edge_index.to(device))
                val_proba = torch.sigmoid(val_logits_full[data.val_mask]).cpu().numpy()
            val_auc = _safe_auc(y_val_arr, val_proba)

            self.train_loss_history_.append(float(epoch_loss))
            self.val_auc_history_.append(float(val_auc))

            _logger.info(
                "fraudgnn.epoch_done",
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
                        "fraudgnn.early_stop",
                        epoch=epoch,
                        best_epoch=best_epoch,
                        best_val_auc=round(best_val_auc, 6),
                    )
                    break

        if best_state is not None:
            module.load_state_dict(best_state)
        module.eval()

        # --- Cache full-graph logits for fast predict_proba ---
        with torch.no_grad():
            full_logits = module(data.x.to(device), data.edge_index.to(device))
            self.cached_node_logits_ = full_logits.cpu().numpy().astype(np.float32)

        self.module_ = module
        self.best_epoch_ = int(best_epoch)
        self.best_val_auc_ = float(best_val_auc)
        self.early_stopped_ = bool(early_stopped)
        return self

    # -----------------------------------------------------------------
    # Predict.
    # -----------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray[Any, Any]:  # noqa: N803
        """Return per-row class probabilities `(n, 2)` via TransactionID lookup.

        Sprint-3 transductive contract: `X` rows must reference
        `TransactionID`s that were present at fit time. Unknown IDs
        raise `KeyError` (Sprint 5+ adds inductive scoring).

        Args:
            X: Frame with a `TransactionID` column.

        Returns:
            float64 ndarray `(len(X), 2)` of `[P(class=0), P(class=1)]`.

        Raises:
            AttributeError: If called pre-fit.
            KeyError: If any `TransactionID` is not in the persisted graph.
        """
        if self.module_ is None or self.txn_index_ is None or self.cached_node_logits_ is None:
            raise AttributeError("FraudGNNModel must be fit before predict_proba")
        if _TRANSACTION_ID_COL not in X.columns:
            raise KeyError(
                f"FraudGNNModel.predict_proba: X must contain "
                f"{_TRANSACTION_ID_COL!r}; got columns {list(X.columns)[:5]}..."
            )
        if len(X) == 0:
            return np.empty((0, _N_CLASS_PROBS), dtype=np.float64)

        ids = X[_TRANSACTION_ID_COL].astype(np.int64).to_numpy()
        indices = np.empty(len(ids), dtype=np.int64)
        for i, tid in enumerate(ids):
            int_tid = int(tid)
            if int_tid not in self.txn_index_:
                raise KeyError(
                    f"FraudGNNModel.predict_proba: TransactionID {int_tid} "
                    f"not in persisted graph (transductive contract; Sprint 5+ "
                    f"adds inductive scoring)"
                )
            indices[i] = self.txn_index_[int_tid]

        logits = self.cached_node_logits_[indices]
        # Numerically-stable sigmoid via 1 / (1 + exp(-x)).
        p_pos = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        p_neg = 1.0 - p_pos
        return np.column_stack([p_neg, p_pos])

    # -----------------------------------------------------------------
    # Save / load.
    # -----------------------------------------------------------------

    def save(self, path: Path) -> tuple[Path, Path]:
        """Persist the fitted model bundle + manifest under `path/`.

        - `path/gnn_model.pt` — joblib payload of the full
          `FraudGNNModel` (module + PyG Data + scaler + median +
          vocabularies + txn_index + cached_node_logits).
        - `path/gnn_model_manifest.json` — sidecar with hparams,
          n_txn / n_entity / n_edges, best_epoch, best_val_auc,
          schema + content hashes.

        Args:
            path: Destination directory. Created if missing.

        Returns:
            `(model_path, manifest_path)`.

        Raises:
            AttributeError: If called pre-fit.
        """
        if self.module_ is None:
            raise AttributeError("FraudGNNModel must be fit before save")
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / _MODEL_FILENAME
        manifest_path = path / _MANIFEST_FILENAME

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
        """Inverse of `save`. Reads `path/gnn_model.pt`.

        Args:
            path: Directory containing the saved model.

        Returns:
            The reconstructed `FraudGNNModel`.

        Raises:
            FileNotFoundError: If `path/gnn_model.pt` does not exist.
            TypeError: If the joblib payload is not a `FraudGNNModel`.
        """
        model_path = path / _MODEL_FILENAME
        loaded = joblib.load(model_path)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"Loaded object at {model_path} is "
                f"{type(loaded).__name__}, expected FraudGNNModel"
            )
        return loaded

    # -----------------------------------------------------------------
    # Manifest.
    # -----------------------------------------------------------------

    def _build_manifest(self, content_hash: str) -> dict[str, Any]:
        """Render the manifest dict (called from `save`)."""
        if (
            self.numeric_cols_ is None
            or self.entity_index_ is None
            or self.best_epoch_ is None
            or self.best_val_auc_ is None
            or self.n_txn_nodes_ is None
            or self.n_entity_nodes_ is None
            or self.n_edges_ is None
        ):
            raise AttributeError("FraudGNNModel._build_manifest called before fit")
        return {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "model_class": "FraudGNNModel",
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "num_layers": self.num_layers,
            "num_neighbors": list(self.num_neighbors),
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "early_stopping_patience": self.early_stopping_patience,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "random_state": self.random_state,
            "n_txn_nodes": int(self.n_txn_nodes_),
            "n_entity_nodes": int(self.n_entity_nodes_),
            "n_edges_undirected": int(self.n_edges_),
            "n_numeric": len(self.numeric_cols_),
            "entity_columns": list(_ENTITY_COLUMNS),
            "best_epoch": self.best_epoch_,
            "best_val_auc": self.best_val_auc_,
            "early_stopped": bool(self.early_stopped_),
            "epochs_run": len(self.val_auc_history_),
            "schema_hash": _schema_fingerprint(self.numeric_cols_),
            "content_hash": content_hash,
        }


# ---------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------


def _safe_auc(y_true: np.ndarray[Any, Any], y_score: np.ndarray[Any, Any]) -> float:
    """ROC-AUC with single-class fallback returning 0.5."""
    from sklearn.metrics import roc_auc_score  # local import — sklearn at fit-time only

    if len(np.unique(y_true)) < _N_CLASS_PROBS:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


__all__ = ["FraudGNN", "FraudGNNModel"]
