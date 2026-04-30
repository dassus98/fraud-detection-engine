"""Tier-5 transaction-entity bipartite graph: construction primitive.

This module is the **first Tier-5 deliverable** and the foundation for
all subsequent graph-feature generators. It builds a bipartite networkx
graph where:

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
import pandas as pd

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


__all__ = ["TransactionEntityGraph"]
