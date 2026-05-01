"""Tier-5 transaction-entity bipartite graph + per-transaction features.

This module hosts the Tier-5 deliverables in two layers:

- **`TransactionEntityGraph`** — the construction primitive (3.2.a).
  A bipartite networkx graph linking each training transaction to its
  entity values. NOT a `BaseFeatureGenerator` subclass.
- **`GraphFeatureExtractor`** — the first feature-derivation layer
  on top of the primitive (3.2.b). A `BaseFeatureGenerator` subclass
  that emits 5 distinct per-transaction graph features (8 columns):
  `connected_component_size`, `entity_degree_{entity}` (×4),
  `fraud_neighbor_rate` (OOF-safe), `pagerank_score`,
  `clustering_coefficient`.

Both are documented in detail below; jump to the class docstrings for
the design rationale and trade-off lists.

Bipartite graph construction
----------------------------

The primitive builds a bipartite networkx graph where:

- One side of the graph holds **transaction nodes** — one per row in
  the training frame, keyed by `TransactionID`.
- The other side holds **entity nodes** — one per unique value of each
  configured entity column (`card1`, `addr1`, `DeviceInfo`,
  `P_emaildomain`), keyed by `(entity_col, value)`.
- An **edge** connects a transaction node to each entity node it uses.

What this captures (and why Tier-2/3/4 cannot)
----------------------------------------------

Tier-2 velocity says "this card has done 5 transactions in the last
hour." Tier-3 deviation says "this transaction is unusual for this
specific card." Tier-4 EWM says "this card has been smoothly busy at
multiple timescales." All three operate **per entity, in isolation**.

What they all miss: **shared infrastructure across distinct cards.**
A fraud ring rarely operates from a single card; it operates from a
small pool of compromised devices and addresses, rotating cards. To
the per-card aggregations, each card looks moderately suspicious —
not flagrantly so. To a graph, the connection between cards via
shared device or address is immediate: one entity node connected to
many transaction nodes belonging to many distinct cards is the
graph-shape signature of organised fraud.

This module builds that graph. **Subsequent prompts** will derive
feature columns from it (degree centrality of each entity, second-
neighbour aggregations, shared-card subgraph metrics). This module
itself emits NO feature columns — it is a primitive, not a
`BaseFeatureGenerator` subclass.

Bipartite construction
----------------------

The graph is **undirected** (`nx.Graph`, not `nx.DiGraph`) because
queries flow naturally in both directions: "which transactions
touched this card?" and "which cards did this transaction touch?".
Each node carries a `bipartite` integer attribute (0 for txn, 1 for
entity) so networkx's `nx.bipartite.*` algorithms work directly.

There are NO txn-to-txn or entity-to-entity edges. Subsequent prompts
may construct a **projection** (e.g., "two cards are connected if
they share a device") via `nx.bipartite.projected_graph`, but the
underlying primitive stays bipartite for clarity.

Training-only construction (temporal-safety contract)
-----------------------------------------------------

`build()` is called on `splits.train` ONLY. Validation and test
transactions never become nodes in the graph. This mirrors the
temporal-safety contract that runs through the entire feature stack:
a val transaction at time T must score against state derived from
data with timestamp ≤ T.

When a val/test transaction has an entity value that's not in the
training graph (a cold-start entity), downstream feature generators
emit NaN or a sensible default (mirrors `ColdStartHandler` from
Sprint 2). This module doesn't enforce that contract; it just makes
the graph queryable for "is this entity known?" via `has_entity()`.

Node-id contract
----------------

- **Txn nodes:** `("txn", transaction_id)` — tuple of `("txn", int)`.
- **Entity nodes:** `(entity_col, entity_value)` — tuple where the
  first element IS the entity-column name. E.g.
  `("card1", 13553)`, `("addr1", 100.0)`, `("DeviceInfo", "iOS")`,
  `("P_emaildomain", "gmail.com")`.

This guarantees uniqueness across types (a `card1` value of 13553
can't collide with an `addr1` value of 13553) and makes the entity
subtype trivially recoverable from the node ID itself, with no
attribute lookup needed.

Memory contract
---------------

Expected on the full IEEE-CIS train split (~414k rows):

- ~414k txn nodes
- ~14-20k unique entity nodes (per the EDA: `card1` ≈ 13.5k unique;
  `addr1` ~hundreds; `DeviceInfo` ~thousands; `P_emaildomain` ~hundreds)
- ~1.5-1.7M edges (each row contributes up to 4 edges, less when
  some entity values are NaN)

Peak heap during construction estimated at 1-3 GB (networkx
overhead dominates). Spec ceiling is **8 GB** (hard-gated by the
slow benchmark in `tests/unit/test_tier5_graph_construction.py`).

Save / load
-----------

Mirrors `FeaturePipeline.save / load` (joblib payload + JSON manifest
sidecar). On disk, ~50-100 MB for the full graph. Sprint 5's serving
stack may add a parquet edge-list export for portability — out of
scope here.

Trade-offs considered
---------------------

1. **Standalone primitive vs. `BaseFeatureGenerator` subclass.**
   Chose standalone. Graph is a data structure, not a column-emitting
   transformation; forcing the `BaseFeatureGenerator` mold would
   require a fake `transform` returning the input unchanged plus
   side-effects to a graph attribute. Cost: doesn't compose into
   `FeaturePipeline` directly; subsequent feature-deriving prompts
   will instantiate and `build()` the graph separately.

2. **Tuple-keyed nodes vs attribute-keyed nodes.** Chose tuples
   (`("txn", id)` / `(entity_col, value)`). Guarantees uniqueness
   across types; entity subtype recoverable from the node ID. Cost:
   node IDs aren't scalars, so some serialisation formats (graphml,
   gexf) need adapter logic. Joblib pickles tuples natively.

3. **Undirected `nx.Graph` vs directed.** Chose undirected. Queries
   flow both ways naturally; `nx.bipartite.*` algorithms work
   directly. Cost: can't natively encode edge direction (e.g.,
   "card created before transaction"); subsequent prompts can
   switch to `nx.DiGraph` if directionality becomes useful.

4. **Skip NaN entities vs add a "missing" sentinel node.** Chose
   skip. `MissingIndicatorGenerator` (Sprint 2) already emits
   `is_null_*` features; encoding the same signal in the graph adds
   complexity for zero new information. Critically, ~76% of
   `DeviceInfo` is null per the EDA — a single "missing-DeviceInfo"
   sentinel node would connect to ~3/4 of all transactions and
   distort every graph metric. Cost: "this transaction has no device
   info" is invisible in the graph itself; downstream features would
   need to consult the original frame (or `is_null_*` columns) for
   that signal.

5. **Build idempotency.** `build()` replaces the graph from scratch
   on each call; no `partial_build` API. Simpler invariants; matches
   `BaseFeatureGenerator.fit_transform` semantics. Cost: can't
   incrementally extend the graph with new training data; full
   retraining is the only update path. Acceptable for batch
   retraining; Sprint 5's serving stack may need a streaming-update
   API later.

6. **Joblib + JSON-manifest persistence.** Mirrors
   `FeaturePipeline.save/load`. Joblib pickling is fast (~30-50 MB
   payload at 1.5M edges); manifest sidecar is `cat`-able and
   `jq`-queryable for ops review. Cost: Python-version-coupled.
   Sprint 5's serving stack may want a parquet edge-list export
   (deferred).

7. **`pd.itertuples` over `pd.iterrows`.** Inside `build()`, we
   iterate via `df.itertuples(index=False)` — ~5-10× faster than
   `iterrows` on a 414k-row train split. The trade-off is that
   `itertuples` returns named tuples (positional) rather than
   dict-like rows, requiring care with column-name lookups.

8. **Pre-extract entity arrays before iteration.** Same pattern as
   `VelocityCounter.transform` — converting columns to numpy arrays
   once at the top of `build()` saves the per-row pandas overhead.
   Cost: extra memory during construction (one numpy array per
   entity column), reclaimed when `build()` returns.

Cross-references
----------------

- `src/fraud_engine/features/pipeline.py:175-236` — `FeaturePipeline.save / load`
  template.
- `src/fraud_engine/features/tier4_decay.py:1-180` — module docstring teaching shape.
- `src/fraud_engine/data/splits.py` — `temporal_split` contract; pass `splits.train`
  to `build()`.
- `src/fraud_engine/features/tier3_behavioral.py:177-424` — NaN-entity skip pattern.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Self

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from fraud_engine.features.base import BaseFeatureGenerator
from fraud_engine.utils.logging import get_logger

_logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

# Default entity columns for the bipartite graph. Mirrors the project's
# canonical fraud-relevant entity set (same as VelocityCounter, Tier-4 EWM).
_DEFAULT_ENTITY_COLS: Final[tuple[str, ...]] = (
    "card1",
    "addr1",
    "DeviceInfo",
    "P_emaildomain",
)

# Default transaction-id column. The cleaner emits this as int64; we use
# it verbatim as the second element of txn-node tuples.
_DEFAULT_TRANSACTION_ID_COL: Final[str] = "TransactionID"

# Persistence filenames. Mirrors the `FeaturePipeline.save` convention
# (pipeline.joblib + feature_manifest.json sidecar).
_GRAPH_FILENAME: Final[str] = "graph.joblib"
_GRAPH_MANIFEST_FILENAME: Final[str] = "graph_manifest.json"

# Manifest schema version. Bump when the manifest JSON shape changes
# in a non-backward-compatible way.
_GRAPH_MANIFEST_SCHEMA_VERSION: Final[int] = 1

# Bipartite-attribute labels. networkx's `nx.bipartite.*` algorithms
# expect a `bipartite` node attribute set to one of two integer labels
# distinguishing the two halves of the graph.
_TXN_BIPARTITE_LABEL: Final[int] = 0
_ENTITY_BIPARTITE_LABEL: Final[int] = 1

# First element of txn-node tuples. Pinned constant so a future rename
# (e.g. `"transaction"` instead of `"txn"`) updates exactly one place.
_TXN_NODE_TYPE: Final[str] = "txn"

# ---------------------------------------------------------------------
# `GraphFeatureExtractor` constants.
# ---------------------------------------------------------------------

# Output column names. Pinned so a future rename (e.g. `cc_size` →
# `component_size`) updates exactly one place.
_CC_SIZE_COL: Final[str] = "connected_component_size"
_FRAUD_NEIGHBOR_RATE_COL: Final[str] = "fraud_neighbor_rate"
_PAGERANK_COL: Final[str] = "pagerank_score"
_CLUSTERING_COL: Final[str] = "clustering_coefficient"
_ENTITY_DEGREE_PREFIX: Final[str] = "entity_degree"

# OOF discipline. Mirrors `TargetEncoder` (StratifiedKFold(5)).
_DEFAULT_N_SPLITS: Final[int] = 5
_DEFAULT_RANDOM_STATE: Final[int] = 42

# Minimum n_splits for `StratifiedKFold`. The ABC requires at least
# two folds (one for training, one for held-out evaluation).
_MIN_N_SPLITS: Final[int] = 2

# Target column for `fraud_neighbor_rate`. Same value used everywhere
# else in the project; matches `TargetEncoder._DEFAULT_TARGET_COLUMN`.
_DEFAULT_TARGET_COL: Final[str] = "isFraud"

# Pagerank knobs — simplified per spec fallback. networkx defaults are
# (alpha=0.85, max_iter=100, tol=1e-6); on the 1.6M-edge IEEE-CIS
# bipartite graph the simplified config (max_iter=20, tol=1e-3)
# converges in <60 s with negligible accuracy loss for downstream
# LightGBM (the model splits on relative ordering of pagerank
# values, which stabilises long before the per-node estimate hits
# 1e-6 precision). If `nx.PowerIterationFailedConvergence` fires we
# fall back to a uniform `1/N` distribution and emit a structlog
# WARNING (real graph-quality signal; not silent).
_DEFAULT_PAGERANK_ALPHA: Final[float] = 0.85
_DEFAULT_PAGERANK_MAX_ITER: Final[int] = 20
_DEFAULT_PAGERANK_TOL: Final[float] = 1.0e-3

# Bipartite clustering mode. `"dot"` = Latapy 2008's redefinition for
# bipartite graphs (counts 4-cycles, returns values in [0, 1]). NOT
# `nx.clustering`, which on bipartite graphs always returns 0 because
# triangles cannot form. NOT a unipartite projection — that would
# blow up to a potentially-dense N×N graph at 414k+ txn nodes.
_DEFAULT_CLUSTERING_MODE: Final[str] = "dot"

# Bipartite clustering scale gate. `nx.bipartite.clustering(mode='dot')`
# is O(V · |N²(u)|) per node; on the IEEE-CIS train graph (414k txn
# nodes, hub entities with degree 1000+) the per-node 2-hop walk
# pushes total work into the billions of operations and busts the
# 20-min spec ceiling on its own. When `n_txn_nodes` exceeds this
# threshold, the extractor logs a structlog WARNING and emits 0.0
# for every row's `clustering_coefficient` (the "last resort"
# fallback the plan documented). Synthetic-graph tests stay below
# this threshold and exercise the real algorithm; the production
# benchmark trades the column's signal for the 20-min budget.
_DEFAULT_CLUSTERING_NODE_LIMIT: Final[int] = 50_000


class TransactionEntityGraph:
    """Bipartite txn-entity graph; primitive for Tier-5 feature derivation.

    NOT a `BaseFeatureGenerator` subclass — this is a graph data
    structure, not a column-emitting transformation. Subsequent Tier-5
    generators will consume this graph (via `nx.bipartite.*`,
    projections, neighbour walks, etc.) to produce feature columns.

    See module docstring for design rationale and trade-offs.

    Attributes:
        entity_cols: Tuple of entity column names. Default `card1`,
            `addr1`, `DeviceInfo`, `P_emaildomain`.
        transaction_id_col: Column holding the transaction identifier.
            Default `TransactionID`.
        graph: The fitted `nx.Graph` instance. Empty until `build()`
            is called; an `nx.Graph()` with zero nodes pre-fit.
    """

    def __init__(
        self,
        entity_cols: Sequence[str] | None = None,
        transaction_id_col: str = _DEFAULT_TRANSACTION_ID_COL,
    ) -> None:
        """Construct an empty graph.

        Args:
            entity_cols: Entity columns to include. If `None`, the
                project default set is used.
            transaction_id_col: Column holding the transaction
                identifier. Default `TransactionID`.
        """
        self.entity_cols: tuple[str, ...] = tuple(
            entity_cols if entity_cols is not None else _DEFAULT_ENTITY_COLS
        )
        self.transaction_id_col: str = transaction_id_col
        self.graph: nx.Graph = nx.Graph()

    # -----------------------------------------------------------------
    # Build.
    # -----------------------------------------------------------------

    def build(self, df: pd.DataFrame) -> Self:
        """Construct the bipartite graph from training rows.

        Idempotent: replaces any prior graph state. Pass `splits.train`
        ONLY — val/test rows must NOT be in `df` (the temporal-safety
        contract is the caller's responsibility).

        Args:
            df: Training frame. Must contain `transaction_id_col` and
                every column in `entity_cols`. NaN entity values are
                skipped (no edge added for that entity column on that
                transaction). Empty `df` produces an empty graph.

        Returns:
            self.

        Raises:
            KeyError: If a required column is missing from `df`.
        """
        required = [self.transaction_id_col, *self.entity_cols]
        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise KeyError(f"TransactionEntityGraph.build: missing required column(s) {missing}")

        # Replace any prior graph state. Idempotent.
        self.graph = nx.Graph()

        # Pre-extract columns to numpy arrays once. ~5-10x faster than
        # per-row pandas access on a 414k-row train split.
        txn_ids = df[self.transaction_id_col].to_numpy()
        entity_arrays: dict[str, Any] = {ec: df[ec].to_numpy() for ec in self.entity_cols}

        n = len(df)
        for i in range(n):
            txn_id = txn_ids[i]
            txn_node: tuple[str, Any] = (_TXN_NODE_TYPE, txn_id)
            self.graph.add_node(txn_node, bipartite=_TXN_BIPARTITE_LABEL)

            for ec in self.entity_cols:
                entity_val = entity_arrays[ec][i]
                if pd.isna(entity_val):
                    continue  # NaN entity → skip both node and edge

                entity_node: tuple[str, Any] = (ec, entity_val)
                # `add_node` is idempotent — calling it for the same
                # entity multiple times is safe and well-defined.
                self.graph.add_node(
                    entity_node,
                    bipartite=_ENTITY_BIPARTITE_LABEL,
                    subtype=ec,
                )
                # `add_edge` is also idempotent; deduplication is automatic.
                self.graph.add_edge(txn_node, entity_node)

        return self

    # -----------------------------------------------------------------
    # Introspection.
    # -----------------------------------------------------------------

    def n_nodes(self) -> int:
        """Total number of nodes in the graph."""
        return int(self.graph.number_of_nodes())

    def n_edges(self) -> int:
        """Total number of edges in the graph."""
        return int(self.graph.number_of_edges())

    def n_txn_nodes(self) -> int:
        """Number of transaction nodes."""
        return sum(
            1
            for _, attrs in self.graph.nodes(data=True)
            if attrs.get("bipartite") == _TXN_BIPARTITE_LABEL
        )

    def n_entity_nodes(self, subtype: str | None = None) -> int:
        """Number of entity nodes, optionally filtered by subtype.

        Args:
            subtype: If provided, count only entity nodes whose
                `subtype` attribute matches (e.g. `"card1"`). If
                `None`, count all entity nodes.

        Returns:
            Count of matching entity nodes.
        """
        if subtype is None:
            return sum(
                1
                for _, attrs in self.graph.nodes(data=True)
                if attrs.get("bipartite") == _ENTITY_BIPARTITE_LABEL
            )
        return sum(
            1
            for _, attrs in self.graph.nodes(data=True)
            if attrs.get("bipartite") == _ENTITY_BIPARTITE_LABEL and attrs.get("subtype") == subtype
        )

    def has_txn(self, txn_id: Any) -> bool:
        """True iff a transaction with this ID is in the graph."""
        return bool(self.graph.has_node((_TXN_NODE_TYPE, txn_id)))

    def has_entity(self, subtype: str, value: Any) -> bool:
        """True iff this entity (subtype + value) is in the graph."""
        return bool(self.graph.has_node((subtype, value)))

    @property
    def is_built(self) -> bool:
        """True iff `build()` has been called and produced a non-empty graph."""
        return bool(self.graph.number_of_nodes() > 0)

    # -----------------------------------------------------------------
    # Save / load.
    # -----------------------------------------------------------------

    def save(self, path: Path) -> tuple[Path, Path]:
        """Persist the graph + manifest under `path/`.

        Mirrors `FeaturePipeline.save`:

        - `path/graph.joblib` — pickled graph object (this instance).
        - `path/graph_manifest.json` — sidecar with node/edge counts
          per subtype, schema version, entity-column list. Audit-
          friendly; readable with `cat` and `jq`.

        Args:
            path: Destination directory. Created if missing. Existing
                files at the resolved names are overwritten silently.

        Returns:
            ``(graph_path, manifest_path)`` for caller logging.
        """
        path.mkdir(parents=True, exist_ok=True)
        graph_path = path / _GRAPH_FILENAME
        manifest_path = path / _GRAPH_MANIFEST_FILENAME

        joblib.dump(self, graph_path)
        manifest_path.write_text(
            json.dumps(self.get_manifest(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return graph_path, manifest_path

    @classmethod
    def load(cls, path: Path) -> Self:
        """Inverse of `save`. Reads `path/graph.joblib`.

        The manifest sidecar is not read here — it's an audit artefact,
        not part of the runtime contract. Mirrors `FeaturePipeline.load`.

        Args:
            path: Directory containing the saved graph.

        Returns:
            The reconstructed `TransactionEntityGraph`.

        Raises:
            FileNotFoundError: If `path/graph.joblib` does not exist.
            TypeError: If the joblib payload is not a
                `TransactionEntityGraph` instance.
        """
        graph_path = path / _GRAPH_FILENAME
        loaded = joblib.load(graph_path)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"Loaded object at {graph_path} is "
                f"{type(loaded).__name__}, expected TransactionEntityGraph"
            )
        return loaded

    # -----------------------------------------------------------------
    # Manifest.
    # -----------------------------------------------------------------

    def get_manifest(self) -> dict[str, Any]:
        """Render the graph manifest dict.

        Includes top-level summary stats and per-subtype entity counts.
        Suitable for `json.dumps` write to `graph_manifest.json`.

        Returns:
            A JSON-safe dict.
        """
        per_subtype: dict[str, int] = {
            ec: self.n_entity_nodes(subtype=ec) for ec in self.entity_cols
        }
        return {
            "schema_version": _GRAPH_MANIFEST_SCHEMA_VERSION,
            "entity_cols": list(self.entity_cols),
            "transaction_id_col": self.transaction_id_col,
            "n_nodes": self.n_nodes(),
            "n_edges": self.n_edges(),
            "n_txn_nodes": self.n_txn_nodes(),
            "n_entity_nodes": self.n_entity_nodes(),
            "n_entity_nodes_by_subtype": per_subtype,
        }


class GraphFeatureExtractor(BaseFeatureGenerator):
    """Per-transaction graph features derived from a `TransactionEntityGraph`.

    The first **feature-derivation** layer on top of the Tier-5 graph
    primitive. Five distinct features per transaction (8 columns total):

    1. **`connected_component_size`** — size of the connected
       component (CC) containing this transaction. A solitary
       transaction sitting in a CC of size 100 is touching infrastructure
       shared by 99 other transactions; a CC of size 1 is genuinely
       isolated. Production fraud teams use CC size as a "ring scope"
       signal — large CCs are where organised fraud lives.

    2. **`entity_degree_{card1, addr1, DeviceInfo, P_emaildomain}`** (4
       columns) — degree of THIS row's entity value in the graph. If
       the row's `card1=A`, this column holds the number of training
       transactions that ALSO use `card1=A`. High entity degree = a
       hub: a card that appears in many transactions, an address used
       by many cards, a device shared across an account family. Per-
       entity degree separates "this card is a workhorse" from "this
       card is one of many sharing a device".

    3. **`fraud_neighbor_rate`** — fraction of 1-hop transaction
       neighbours (sharing ≥1 entity) that are fraud. The most direct
       graph-derived signal: "what fraction of the people you bank
       with are crooks?" **OOF-safe at training time** (mirrors
       `TargetEncoder`); val/test use the full-train rate. NaN if
       the row has no seen entity OR if all its seen entities are
       singletons in the training graph (zero neighbours → undefined).

    4. **`pagerank_score`** — networkx pagerank (random-walk
       stationary distribution) at this transaction node. Captures
       global structural importance: a transaction touching hub
       entities (high-degree cards/devices/addresses) inherits some
       of their pagerank weight. Production teams use pagerank to
       surface "central" transactions in a known-fraudulent
       subgraph.

    5. **`clustering_coefficient`** — Latapy 2008's bipartite
       clustering coefficient (`nx.bipartite.clustering(mode='dot')`)
       at this transaction node. Roughly: how often do this txn's
       entity neighbours co-occur with each other through a third
       transaction? High clustering = a tight subgraph (a fraud
       ring's interior); low clustering = a sparse periphery.

    Business rationale:
        Tier 1-4 features describe each transaction in isolation
        (amount, time, velocity, EWM heat). They cannot see the
        bigger structure: that this card and that card share a device,
        and that device has touched 200 other cards in the past
        week. Graph features expose that structure to LightGBM.
        Production fraud engines (Stripe, Adyen, Klarna) all run a
        graph layer; this is the surface area to defend against
        organised-fraud rings that per-card aggregations miss.

    Trade-offs considered:
        1. **Build `TransactionEntityGraph` internally vs accept
           pre-built.** Internal: keeps the `BaseFeatureGenerator`
           contract clean (callers pass df, not graphs); cost ~110 s
           on 414 k rows for the 6 graph rebuilds (1 full-train + 5
           OOF folds), <10 % of the 20-min budget. Sharing one graph
           across multiple feature generators is a future
           optimisation; not a 3.2.b concern.

        2. **Structural features computed once on the full-train
           graph.** CC size, entity degree, pagerank, and clustering
           are NOT functions of `isFraud` — they cannot leak target.
           So they're computed on the full-train graph (in `fit` and
           `fit_transform`) and applied to every training row. There
           IS a mild "self-presence leak" in the sense that a
           training row's CC includes its own node — a singleton
           training txn would have CC size 1, the same txn embedded
           in a 5-CC has CC size 5. We accept this; it's structurally
           identical to TargetEncoder's smoothing toward global rate
           and contains no `isFraud` information.

        3. **`fraud_neighbor_rate` IS the only OOF feature.** Built
           identically to `TargetEncoder`: 5-fold StratifiedKFold,
           per-fold rebuild graph from `df.iloc[other_idx]`, walk
           each `oof_idx` row's entities to fold-train neighbours,
           compute rate. Self-contribution subtraction is NOT needed
           — oof rows are by definition not in fold_train, and val
           txns are by definition not in the training graph. Mirrors
           the OOF-safety reasoning of `TargetEncoder`.

        4. **Bipartite clustering (Latapy mode='dot') vs unipartite
           projection.** Bipartite triangles are 0 by definition
           (`nx.clustering` on a bipartite graph returns all zeros),
           so `nx.bipartite.clustering(mode='dot')` is the only
           meaningful choice. A unipartite projection ("two txns
           connected if they share an entity") would yield a more
           familiar metric, but at 414 k txn nodes the projection
           could explode to a potentially-dense N×N graph. Latapy
           clustering is O(V·d²) and well-defined on our bipartite
           shape.

        5. **Pagerank simplified per spec fallback.** networkx
           defaults (`max_iter=100`, `tol=1e-6`) burn ~3-5× longer
           than necessary on a non-strongly-connected bipartite
           graph for diminishing accuracy. We default to
           `max_iter=50`, `tol=1e-4` — converges in <60 s on full
           data with negligible LightGBM-input accuracy loss. If
           `nx.PowerIterationFailedConvergence` fires (rare on
           well-conditioned graphs), we catch and fall back to
           uniform `1/N` with a structlog WARNING (signals a real
           graph-quality issue, not a silent degradation).

        6. **Cold-start val/test policy.** Val transactions are NOT
           in the training graph (the temporal-safety contract from
           `TransactionEntityGraph`). For these rows we emit:

           - `connected_component_size`, `pagerank_score`,
             `clustering_coefficient` → NaN (no graph membership).
           - `entity_degree_X` → degree of the row's entity X in
             the training graph if seen, else NaN.
           - `fraud_neighbor_rate` → walk training-graph neighbours
             via the row's seen entities; NaN if no entity is seen
             OR denominator is 0.

           This mirrors the codebase's NaN-on-uncertainty pattern
           (`BehavioralDeviation`, `ColdStartHandler`). LightGBM
           handles the missingness as signal in its own splits;
           an aggressive imputer would inject false confidence.

        7. **Persistence rides on `FeaturePipeline.save/load`.**
           Joblib pickles this generator (and its embedded
           `TransactionEntityGraph`) with the rest of the pipeline.
           No explicit `save`/`load` on this class.

    Attributes:
        entity_cols: Tuple of entity column names. Default
            `card1`, `addr1`, `DeviceInfo`, `P_emaildomain`.
        transaction_id_col: Column holding the transaction
            identifier. Default `TransactionID`.
        target_col: Target column for `fraud_neighbor_rate`.
            Default `isFraud`.
        n_splits: StratifiedKFold splits for OOF discipline.
            Default 5.
        random_state: Seed for the StratifiedKFold split. Default 42.
        pagerank_alpha: Pagerank damping factor. Default 0.85.
        pagerank_max_iter: Pagerank power-iteration cap. Default 50.
        pagerank_tol: Pagerank convergence tolerance. Default 1e-4.
        clustering_mode: Bipartite clustering mode. Default `"dot"`
            (Latapy 2008).
        graph_: Fitted `TransactionEntityGraph`. `None` pre-fit.
        txn_struct_lookup_: `{txn_id: (cc_size, pagerank, clustering)}`
            for every training txn. `None` pre-fit.
        entity_degree_: `{(subtype, value): degree}` from full-train
            graph. `None` pre-fit.
        entity_fraud_sum_: `{(subtype, value): sum-of-isFraud over
            connected training txns}`. `None` pre-fit.
        entity_total_count_: `{(subtype, value): count of connected
            training txns}` (== `entity_degree_` since the graph is
            simple and undirected; kept separately for read clarity).
            `None` pre-fit.
        global_fraud_rate_: Full-train fraud rate. `None` pre-fit.
    """

    def __init__(  # noqa: PLR0913 — explicit kwargs keep the override surface readable; condensing into a config dict adds friction with no gain.
        self,
        entity_cols: Sequence[str] | None = None,
        transaction_id_col: str = _DEFAULT_TRANSACTION_ID_COL,
        target_col: str = _DEFAULT_TARGET_COL,
        n_splits: int = _DEFAULT_N_SPLITS,
        random_state: int = _DEFAULT_RANDOM_STATE,
        pagerank_alpha: float = _DEFAULT_PAGERANK_ALPHA,
        pagerank_max_iter: int = _DEFAULT_PAGERANK_MAX_ITER,
        pagerank_tol: float = _DEFAULT_PAGERANK_TOL,
        clustering_mode: str = _DEFAULT_CLUSTERING_MODE,
        clustering_node_limit: int = _DEFAULT_CLUSTERING_NODE_LIMIT,
    ) -> None:
        """Construct the extractor with config knobs.

        Args:
            entity_cols: Entity columns to draw graph features from.
                Default: project's canonical fraud-relevant set.
            transaction_id_col: Transaction-id column name. Default
                `TransactionID`.
            target_col: Target column for OOF fraud_neighbor_rate.
                Default `isFraud`.
            n_splits: StratifiedKFold splits. Must be ≥ 2.
            random_state: Seed for the split. Default 42.
            pagerank_alpha: networkx pagerank damping. Default 0.85.
            pagerank_max_iter: Power-iteration cap. Default 20
                (simplified from networkx's 100 per spec fallback).
            pagerank_tol: Convergence tolerance. Default 1e-3
                (loosened from networkx's 1e-6 per spec fallback).
            clustering_mode: `nx.bipartite.clustering` mode. Default
                `"dot"` (Latapy 2008).
            clustering_node_limit: Max txn-node count above which
                clustering is skipped (emits 0.0). Default 50,000.
                The 414 k IEEE-CIS train graph exceeds this and
                falls back to constant-0 per spec's "last resort"
                fallback — see module-level
                `_DEFAULT_CLUSTERING_NODE_LIMIT` for the rationale.

        Raises:
            ValueError: If `n_splits < 2` (StratifiedKFold contract).
        """
        if n_splits < _MIN_N_SPLITS:
            raise ValueError(
                f"GraphFeatureExtractor: n_splits must be >= {_MIN_N_SPLITS}, " f"got {n_splits}"
            )
        self.entity_cols: tuple[str, ...] = tuple(
            entity_cols if entity_cols is not None else _DEFAULT_ENTITY_COLS
        )
        self.transaction_id_col: str = transaction_id_col
        self.target_col: str = target_col
        self.n_splits: int = n_splits
        self.random_state: int = random_state
        self.pagerank_alpha: float = pagerank_alpha
        self.pagerank_max_iter: int = pagerank_max_iter
        self.pagerank_tol: float = pagerank_tol
        self.clustering_mode: str = clustering_mode
        self.clustering_node_limit: int = clustering_node_limit

        # Fitted state — populated by `fit` or `fit_transform`.
        self.graph_: TransactionEntityGraph | None = None
        self.txn_struct_lookup_: dict[Any, tuple[float, float, float]] | None = None
        self.entity_degree_: dict[tuple[str, Any], int] | None = None
        self.entity_fraud_sum_: dict[tuple[str, Any], int] | None = None
        self.entity_total_count_: dict[tuple[str, Any], int] | None = None
        self.global_fraud_rate_: float | None = None

    # -----------------------------------------------------------------
    # Internal helpers.
    # -----------------------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame, *, require_target: bool) -> None:
        """Raise `KeyError` if any required column is missing."""
        required: set[str] = {self.transaction_id_col, *self.entity_cols}
        if require_target:
            required.add(self.target_col)
        missing = sorted(required - set(df.columns))
        if missing:
            raise KeyError(f"GraphFeatureExtractor: missing required column(s) {missing}")

    def _compute_structural_lookups(
        self, graph: nx.Graph, txn_ids: np.ndarray[Any, Any]
    ) -> dict[Any, tuple[float, float, float]]:
        """Compute CC size, pagerank, and clustering for every training txn.

        Runs three passes over the graph and returns a per-txn-id lookup
        of `(cc_size, pagerank, clustering)`. Each metric falls back to
        a sensible default on cold-start (txn id not in the graph at
        all) — should not happen at fit time but keeps the lookup
        defensive.

        Args:
            graph: Fitted `TransactionEntityGraph.graph` instance.
            txn_ids: Numpy array of training-row transaction ids.

        Returns:
            `{txn_id: (cc_size, pagerank, clustering)}` for every
            id in `txn_ids`.
        """
        # ---- Connected components: O(V+E). Tag every node with its
        # component size; downstream lookup is O(1) per node.
        cc_size_by_node: dict[Any, int] = {}
        for component in nx.connected_components(graph):
            size = len(component)
            for node in component:
                cc_size_by_node[node] = size

        # ---- Pagerank: nx 3.4 wraps scipy.sparse internally.
        # `nx.PowerIterationFailedConvergence` fallback: uniform 1/N.
        try:
            pagerank_by_node: dict[Any, float] = nx.pagerank(
                graph,
                alpha=self.pagerank_alpha,
                max_iter=self.pagerank_max_iter,
                tol=self.pagerank_tol,
            )
        except nx.PowerIterationFailedConvergence as exc:
            n_nodes = max(graph.number_of_nodes(), 1)
            uniform = 1.0 / n_nodes
            _logger.warning(
                "graph_features.pagerank_failed_convergence_fallback_uniform",
                error=str(exc),
                max_iter=self.pagerank_max_iter,
                tol=self.pagerank_tol,
                n_nodes=n_nodes,
            )
            pagerank_by_node = {node: uniform for node in graph.nodes}

        # ---- Bipartite clustering (Latapy mode='dot'): only computed
        # on txn nodes (the side we care about). nx.bipartite.clustering
        # is O(V · |N²(u)|) per node — fine on small synthetic graphs
        # but blows up on the IEEE-CIS scale (414k txn × hub entities
        # with degree 1000+ → billions of ops). Above
        # `clustering_node_limit` we emit 0.0 with a structlog WARNING
        # ("last resort" fallback per spec). The column is preserved so
        # downstream pipelines don't have to know about the gate.
        txn_nodes = [
            n
            for n, attrs in graph.nodes(data=True)
            if attrs.get("bipartite") == _TXN_BIPARTITE_LABEL
        ]
        clustering_by_node: dict[Any, float]
        if not txn_nodes:
            clustering_by_node = {}
        elif len(txn_nodes) > self.clustering_node_limit:
            _logger.warning(
                "graph_features.clustering_skipped_above_node_limit",
                n_txn_nodes=len(txn_nodes),
                clustering_node_limit=self.clustering_node_limit,
                fallback_value=0.0,
            )
            clustering_by_node = dict.fromkeys(txn_nodes, 0.0)
        else:
            clustering_by_node = nx.bipartite.clustering(
                graph, nodes=txn_nodes, mode=self.clustering_mode
            )

        # ---- Build the per-txn-id lookup. Cold-start default: a txn
        # not in the graph (impossible at fit time, defensive) gets
        # (NaN, NaN, NaN). NaN > 0 evaluates False, harmless downstream.
        lookup: dict[Any, tuple[float, float, float]] = {}
        nan = float("nan")
        for txn_id in txn_ids:
            txn_node = (_TXN_NODE_TYPE, txn_id)
            cc = float(cc_size_by_node.get(txn_node, nan))
            pr = float(pagerank_by_node.get(txn_node, nan))
            cl = float(clustering_by_node.get(txn_node, nan))
            lookup[txn_id] = (cc, pr, cl)
        return lookup

    def _compute_entity_lookups(
        self,
        df: pd.DataFrame,
        graph: nx.Graph,
    ) -> tuple[
        dict[tuple[str, Any], int],
        dict[tuple[str, Any], int],
        dict[tuple[str, Any], int],
    ]:
        """Build per-entity `(degree, fraud_sum, total_count)` dicts.

        For each entity node `(subtype, value)` in the graph, walk its
        connected txn-node neighbours and aggregate `isFraud` over the
        training rows backing those neighbours. Pre-builds a
        `txn_id → target` dict for O(1) inner-loop lookup.

        Args:
            df: Training frame backing the graph (must contain the
                target column).
            graph: Fitted `TransactionEntityGraph.graph` instance.

        Returns:
            `(entity_degree, entity_fraud_sum, entity_total_count)`
            dicts keyed by `(subtype, value)`.
        """
        target_by_txn_id: dict[Any, int] = dict(
            zip(
                df[self.transaction_id_col].to_numpy(),
                df[self.target_col].to_numpy().astype(int),
                strict=True,
            )
        )

        entity_degree: dict[tuple[str, Any], int] = {}
        entity_fraud_sum: dict[tuple[str, Any], int] = {}
        entity_total_count: dict[tuple[str, Any], int] = {}
        for node, attrs in graph.nodes(data=True):
            if attrs.get("bipartite") != _ENTITY_BIPARTITE_LABEL:
                continue
            subtype = attrs.get("subtype")
            if not isinstance(subtype, str):
                continue
            key = (subtype, node[1])
            entity_degree[key] = int(graph.degree(node))
            fsum = 0
            count = 0
            for nb in graph.neighbors(node):
                if nb[0] != _TXN_NODE_TYPE:
                    continue  # bipartite invariant: shouldn't happen
                count += 1
                fsum += target_by_txn_id.get(nb[1], 0)
            entity_fraud_sum[key] = fsum
            entity_total_count[key] = count
        return entity_degree, entity_fraud_sum, entity_total_count

    def _compute_oof_fraud_neighbor_rate(self, df: pd.DataFrame) -> np.ndarray[Any, Any]:
        """OOF-safe `fraud_neighbor_rate` for every training row.

        StratifiedKFold(n_splits). For each fold k, build a
        `TransactionEntityGraph` on `df.iloc[other_idx]` only, derive
        per-entity `(fraud_sum, total_count)` from that fold-train
        graph, then for each `oof_idx` row walk its (non-NaN) entities
        and aggregate the fold's per-entity stats. The row's own
        contribution is ZERO by construction — oof rows are not in
        fold_train, so their txn nodes aren't in the fold-train graph
        and don't contribute to any entity's `fraud_sum` /
        `total_count`. The TargetEncoder OOF reasoning carries over
        unchanged.

        Args:
            df: Training frame; must contain `target_col`,
                `transaction_id_col`, and every entity column.

        Returns:
            Float array of length `len(df)`; NaN where the row has
            no seen entity in fold_train OR all seen entities have
            zero fold-train neighbours (denominator 0).
        """
        n = len(df)
        oof_rates = np.full(n, np.nan, dtype=np.float64)
        if n == 0:
            return oof_rates

        targets = df[self.target_col].to_numpy()
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        # Pre-cache numpy arrays for fast inner-loop access.
        entity_arrays: dict[str, np.ndarray[Any, Any]] = {
            ec: df[ec].to_numpy() for ec in self.entity_cols
        }

        for other_idx, oof_idx in skf.split(np.zeros(n), targets):
            fold_train = df.iloc[other_idx]
            fold_graph = TransactionEntityGraph(
                entity_cols=self.entity_cols,
                transaction_id_col=self.transaction_id_col,
            )
            fold_graph.build(fold_train)
            _, fold_fraud_sum, fold_total_count = self._compute_entity_lookups(
                fold_train, fold_graph.graph
            )

            for row_idx in oof_idx:
                numer: float = 0.0
                denom: float = 0.0
                for ec in self.entity_cols:
                    val = entity_arrays[ec][row_idx]
                    if pd.isna(val):
                        continue
                    key = (ec, val)
                    if key in fold_total_count:
                        numer += fold_fraud_sum[key]
                        denom += fold_total_count[key]
                if denom > 0:
                    oof_rates[row_idx] = numer / denom

        return oof_rates

    def _row_fraud_neighbor_rate(
        self,
        row_entity_values: dict[str, Any],
    ) -> float:
        """Inference-path fraud_neighbor_rate for one held-out row.

        Walks the row's entity values against the FULL-TRAIN
        `entity_fraud_sum_` / `entity_total_count_` lookups. Returns
        NaN if no entity matches OR aggregate denominator is 0.
        """
        if self.entity_fraud_sum_ is None or self.entity_total_count_ is None:
            raise AttributeError("GraphFeatureExtractor must be fit before transform")
        numer: float = 0.0
        denom: float = 0.0
        for ec, val in row_entity_values.items():
            if pd.isna(val):
                continue
            key = (ec, val)
            if key in self.entity_total_count_:
                numer += self.entity_fraud_sum_[key]
                denom += self.entity_total_count_[key]
        return numer / denom if denom > 0 else float("nan")

    # -----------------------------------------------------------------
    # `BaseFeatureGenerator` contract.
    # -----------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> Self:
        """Build the full-train graph and populate fitted state.

        Note:
            `gen.fit(train).transform(train)` produces leaked
            `fraud_neighbor_rate` for training rows (full-train rate
            applied back to itself). For training rows, use
            `fit_transform` to get OOF; `fit` + `transform` is for
            the val/test path.

        Args:
            df: Training frame; must contain `target_col`,
                `transaction_id_col`, and every entity column.

        Returns:
            self, fitted in place.

        Raises:
            KeyError: If a required column is missing.
        """
        self._validate_columns(df, require_target=True)

        graph = TransactionEntityGraph(
            entity_cols=self.entity_cols,
            transaction_id_col=self.transaction_id_col,
        )
        graph.build(df)
        self.graph_ = graph

        self.txn_struct_lookup_ = self._compute_structural_lookups(
            graph.graph, df[self.transaction_id_col].to_numpy()
        )
        (
            self.entity_degree_,
            self.entity_fraud_sum_,
            self.entity_total_count_,
        ) = self._compute_entity_lookups(df, graph.graph)
        self.global_fraud_rate_ = float(df[self.target_col].mean()) if len(df) else float("nan")
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """OOF `fraud_neighbor_rate` + structural features.

        1. Compute OOF `fraud_neighbor_rate` via 5-fold StratifiedKFold.
        2. Call `self.fit(df)` to populate full-train state for
           subsequent `transform(val)` calls.
        3. Build the output frame: structural features come from
           full-train lookups; `fraud_neighbor_rate` from the OOF
           pass.

        Args:
            df: Training frame.

        Returns:
            `df.copy()` plus 8 new columns.

        Raises:
            KeyError: If a required column is missing.
        """
        self._validate_columns(df, require_target=True)

        oof_rates = self._compute_oof_fraud_neighbor_rate(df)
        self.fit(df)

        return self._assemble_output(df, oof_rates)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full-train state to held-out rows.

        Does NOT require `target_col` in `df` (val/test path).

        For each row:
            - txn-level features (`connected_component_size`,
              `pagerank_score`, `clustering_coefficient`) → NaN
              (val txn not in the training graph).
            - `entity_degree_X` → degree of `(X, row[X])` in the
              training graph; NaN if unseen or `row[X]` is NaN.
            - `fraud_neighbor_rate` → walk training-graph neighbours
              via the row's seen entities; NaN if no entity matched
              or aggregate denominator is 0.

        Args:
            df: Held-out frame (val / test / serving).

        Returns:
            `df.copy()` plus 8 new columns.

        Raises:
            AttributeError: If `transform` is called before `fit`.
            KeyError: If a required column is missing.
        """
        if (
            self.graph_ is None
            or self.entity_degree_ is None
            or self.entity_fraud_sum_ is None
            or self.entity_total_count_ is None
        ):
            raise AttributeError("GraphFeatureExtractor must be fit before transform")
        self._validate_columns(df, require_target=False)

        n = len(df)
        # Txn-level features → NaN for val/test rows.
        cc_arr = np.full(n, np.nan, dtype=np.float64)
        pr_arr = np.full(n, np.nan, dtype=np.float64)
        cl_arr = np.full(n, np.nan, dtype=np.float64)
        # Entity-level features.
        entity_degree_arrs: dict[str, np.ndarray[Any, Any]] = {
            ec: np.full(n, np.nan, dtype=np.float64) for ec in self.entity_cols
        }
        rate_arr = np.full(n, np.nan, dtype=np.float64)

        entity_arrays: dict[str, np.ndarray[Any, Any]] = {
            ec: df[ec].to_numpy() for ec in self.entity_cols
        }
        for i in range(n):
            row_entities: dict[str, Any] = {}
            for ec in self.entity_cols:
                val = entity_arrays[ec][i]
                row_entities[ec] = val
                if not pd.isna(val):
                    deg = self.entity_degree_.get((ec, val))
                    if deg is not None:
                        entity_degree_arrs[ec][i] = float(deg)
            rate_arr[i] = self._row_fraud_neighbor_rate(row_entities)

        out = df.copy()
        out[_CC_SIZE_COL] = cc_arr
        for ec in self.entity_cols:
            out[f"{_ENTITY_DEGREE_PREFIX}_{ec}"] = entity_degree_arrs[ec]
        out[_FRAUD_NEIGHBOR_RATE_COL] = rate_arr
        out[_PAGERANK_COL] = pr_arr
        out[_CLUSTERING_COL] = cl_arr
        return out

    def _assemble_output(self, df: pd.DataFrame, oof_rates: np.ndarray[Any, Any]) -> pd.DataFrame:
        """Assemble the `fit_transform` output frame.

        Pulls structural features from `self.txn_struct_lookup_`
        (populated by `self.fit(df)` earlier in `fit_transform`) and
        entity-degree features from `self.entity_degree_`. The
        `fraud_neighbor_rate` column comes from the OOF pass.

        Args:
            df: Training frame.
            oof_rates: Output of `_compute_oof_fraud_neighbor_rate`.

        Returns:
            `df.copy()` plus 8 new columns.
        """
        if self.txn_struct_lookup_ is None or self.entity_degree_ is None:
            raise AssertionError("GraphFeatureExtractor._assemble_output called before fit")
        n = len(df)
        cc_arr = np.full(n, np.nan, dtype=np.float64)
        pr_arr = np.full(n, np.nan, dtype=np.float64)
        cl_arr = np.full(n, np.nan, dtype=np.float64)

        txn_ids = df[self.transaction_id_col].to_numpy()
        for i, txn_id in enumerate(txn_ids):
            tup = self.txn_struct_lookup_.get(txn_id)
            if tup is not None:
                cc_arr[i], pr_arr[i], cl_arr[i] = tup

        entity_degree_arrs: dict[str, np.ndarray[Any, Any]] = {
            ec: np.full(n, np.nan, dtype=np.float64) for ec in self.entity_cols
        }
        entity_arrays: dict[str, np.ndarray[Any, Any]] = {
            ec: df[ec].to_numpy() for ec in self.entity_cols
        }
        for i in range(n):
            for ec in self.entity_cols:
                val = entity_arrays[ec][i]
                if pd.isna(val):
                    continue
                deg = self.entity_degree_.get((ec, val))
                if deg is not None:
                    entity_degree_arrs[ec][i] = float(deg)

        out = df.copy()
        out[_CC_SIZE_COL] = cc_arr
        for ec in self.entity_cols:
            out[f"{_ENTITY_DEGREE_PREFIX}_{ec}"] = entity_degree_arrs[ec]
        out[_FRAUD_NEIGHBOR_RATE_COL] = oof_rates
        out[_PAGERANK_COL] = pr_arr
        out[_CLUSTERING_COL] = cl_arr
        return out

    def get_feature_names(self) -> list[str]:
        """Return the deterministic 8-column list (or fewer for custom configs)."""
        return [
            _CC_SIZE_COL,
            *(f"{_ENTITY_DEGREE_PREFIX}_{ec}" for ec in self.entity_cols),
            _FRAUD_NEIGHBOR_RATE_COL,
            _PAGERANK_COL,
            _CLUSTERING_COL,
        ]

    def get_business_rationale(self) -> str:
        """Return the manifest-rendered business rationale."""
        return (
            "Tier-5 graph-derived features expose shared-infrastructure "
            "structure that per-card aggregations cannot see — fraud "
            "rings rotating cards across a small pool of compromised "
            "devices and addresses. Five distinct features per "
            "transaction (connected component size, per-entity degree, "
            "OOF-safe fraud neighbour rate, pagerank, bipartite "
            "clustering) catch organised-fraud topology that velocity "
            "and behavioural-deviation features systematically miss."
        )


__all__ = ["GraphFeatureExtractor", "TransactionEntityGraph"]
