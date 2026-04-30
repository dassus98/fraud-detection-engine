"""Unit tests for `fraud_engine.features.tier5_graph.TransactionEntityGraph`.

Five contract surfaces:

- `TestSyntheticConstruction`: hand-computed graph counts; NaN-entity
  skip; idempotent entity-node creation; node-attribute correctness;
  build idempotency.
- `TestTrainingOnlyContract`: val transactions never appear as nodes
  in the graph (the temporal-safety contract for downstream features);
  introspection methods return correct values.
- `TestSaveLoad`: joblib + manifest sidecar round-trip; reload
  produces a bit-identical graph; load rejects wrong object types.
- `TestPerformance` (`@pytest.mark.slow`, skip-gated on
  `MANIFEST.json`): full 590k IEEE-CIS train split; build memory
  peak < 8 GB; build wall-time recorded for the report.
- `TestErrorHandling`: missing required columns raise `KeyError`;
  loading from a non-existent path raises `FileNotFoundError`.
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_engine.config.settings import get_settings
from fraud_engine.data.cleaner import TransactionCleaner
from fraud_engine.data.loader import RawDataLoader
from fraud_engine.data.splits import temporal_split
from fraud_engine.features.tier5_graph import TransactionEntityGraph

# Hard memory ceiling per spec: full 590k dataset graph fits in <8 GB.
_MEMORY_CEILING_GB: float = 8.0
_BYTES_PER_GB: float = 1.0e9


# ---------------------------------------------------------------------
# Synthetic-frame helpers.
# ---------------------------------------------------------------------


def _build_simple_frame() -> pd.DataFrame:
    """Minimal 3-row frame with hand-computable graph structure.

    Card layout:
      txn 100 → card1=A, addr1=10
      txn 101 → card1=A, addr1=20
      txn 102 → card1=B, addr1=20

    Expected graph: 3 txn nodes + 2 card nodes (A, B) + 2 addr nodes
    (10, 20) = 7 nodes. Each txn has 2 edges (one to its card, one to
    its addr) = 6 edges total.
    """
    return pd.DataFrame(
        {
            "TransactionID": [100, 101, 102],
            "card1": ["A", "A", "B"],
            "addr1": [10, 20, 20],
            "DeviceInfo": ["d1", "d1", "d2"],
            "P_emaildomain": ["e1", "e2", "e1"],
        }
    )


# ---------------------------------------------------------------------
# `TestSyntheticConstruction`: hand-computed correctness.
# ---------------------------------------------------------------------


class TestSyntheticConstruction:
    """Hand-computed counts on small synthetic frames."""

    def test_minimal_3row_synthetic_graph(self) -> None:
        """3 txns × 4 entity columns; verify exact node and edge counts."""
        df = _build_simple_frame()
        # entity_cols only include card1 and addr1 to keep counts small.
        gen = TransactionEntityGraph(entity_cols=["card1", "addr1"])
        gen.build(df)

        # 3 txn + 2 card1 (A, B) + 2 addr1 (10, 20) = 7 nodes.
        assert gen.n_nodes() == 7
        assert gen.n_txn_nodes() == 3
        assert gen.n_entity_nodes() == 4
        assert gen.n_entity_nodes(subtype="card1") == 2
        assert gen.n_entity_nodes(subtype="addr1") == 2

        # Each of 3 txns has 2 edges = 6 total. No edge dedup applies
        # because (txn, card_a) and (txn, addr_b) are distinct pairs.
        assert gen.n_edges() == 6

    def test_nan_entity_skipped(self) -> None:
        """NaN entity → no entity node created; no edge added for that pair."""
        df = pd.DataFrame(
            {
                "TransactionID": [200, 201],
                "card1": ["A", "A"],
                "addr1": [10, np.nan],
            }
        )
        gen = TransactionEntityGraph(entity_cols=["card1", "addr1"])
        gen.build(df)

        # 2 txn + 1 card1 (A) + 1 addr1 (10) = 4 nodes (the NaN addr1
        # for row 201 is skipped — no `(addr1, NaN)` node exists).
        assert gen.n_nodes() == 4
        assert gen.n_entity_nodes(subtype="addr1") == 1
        # txn 201 has 1 edge (to card1=A only); txn 200 has 2.
        assert gen.n_edges() == 3
        # Verify the would-be NaN node literally isn't there.
        assert not gen.has_entity("addr1", float("nan"))

    def test_idempotent_entity_node_creation(self) -> None:
        """3 txns sharing the same card1 → ONE entity node, 3 edges to it."""
        df = pd.DataFrame(
            {
                "TransactionID": [300, 301, 302],
                "card1": ["A", "A", "A"],
            }
        )
        gen = TransactionEntityGraph(entity_cols=["card1"])
        gen.build(df)

        assert gen.n_nodes() == 4  # 3 txn + 1 card1
        assert gen.n_entity_nodes(subtype="card1") == 1
        assert gen.n_edges() == 3
        # All three txns connect to the single card1=A node.
        card_node = ("card1", "A")
        assert gen.graph.has_node(card_node)
        assert gen.graph.degree(card_node) == 3

    def test_node_attributes_correct(self) -> None:
        """`bipartite` attribute is 0 for txn nodes, 1 for entity; entity nodes have `subtype`."""
        df = _build_simple_frame()
        gen = TransactionEntityGraph(entity_cols=["card1", "addr1"])
        gen.build(df)

        # Txn nodes: bipartite=0, no subtype attribute.
        for txn_id in (100, 101, 102):
            attrs = gen.graph.nodes[("txn", txn_id)]
            assert attrs.get("bipartite") == 0
            assert "subtype" not in attrs

        # Entity nodes: bipartite=1, subtype matches column.
        for value in ("A", "B"):
            attrs = gen.graph.nodes[("card1", value)]
            assert attrs.get("bipartite") == 1
            assert attrs.get("subtype") == "card1"
        for value in (10, 20):
            attrs = gen.graph.nodes[("addr1", value)]
            assert attrs.get("bipartite") == 1
            assert attrs.get("subtype") == "addr1"

    def test_build_idempotent_replaces_graph(self) -> None:
        """`build()` called twice replaces the previous graph (no leftovers)."""
        df1 = pd.DataFrame({"TransactionID": [400, 401], "card1": ["X", "Y"]})
        df2 = pd.DataFrame({"TransactionID": [402, 403], "card1": ["Z", "W"]})
        gen = TransactionEntityGraph(entity_cols=["card1"])
        gen.build(df1)
        assert gen.has_txn(400)
        assert gen.has_entity("card1", "X")
        gen.build(df2)
        # df1's nodes are gone after the second build.
        assert not gen.has_txn(400)
        assert not gen.has_entity("card1", "X")
        # df2's nodes are present.
        assert gen.has_txn(402)
        assert gen.has_entity("card1", "Z")


# ---------------------------------------------------------------------
# `TestTrainingOnlyContract`: temporal-safety + introspection.
# ---------------------------------------------------------------------


class TestTrainingOnlyContract:
    """Training-only contract: val txns must never appear as nodes."""

    def test_val_transactions_not_in_graph(self) -> None:
        """Build on `splits.train` only; assert val TransactionIDs are absent."""
        # Synthetic frame with monotone TransactionDT spanning train + val.
        # We split manually to avoid coupling to `temporal_split`'s
        # specific date thresholds.
        n = 100
        df = pd.DataFrame(
            {
                "TransactionDT": np.arange(n, dtype=np.int64) * 60,
                "TransactionID": np.arange(500, 500 + n, dtype=np.int64),
                "card1": ["A"] * n,
                "addr1": [10] * n,
                "DeviceInfo": ["d"] * n,
                "P_emaildomain": ["e"] * n,
            }
        )
        # First 80 → train; last 20 → val.
        train = df.iloc[:80]
        val = df.iloc[80:]

        gen = TransactionEntityGraph()
        gen.build(train)

        # All training transactions are present.
        for txn_id in train["TransactionID"]:
            assert gen.has_txn(int(txn_id))
        # NO val transaction is present (the temporal-safety gate).
        for txn_id in val["TransactionID"]:
            assert not gen.has_txn(int(txn_id))

    def test_introspection_methods(self) -> None:
        """All introspection methods return correct values on a known graph."""
        df = _build_simple_frame()
        gen = TransactionEntityGraph(entity_cols=["card1", "addr1"])
        gen.build(df)

        # `is_built` flips True after a non-empty build.
        assert gen.is_built is True

        # `has_txn` finds present + absent txns.
        assert gen.has_txn(100)
        assert not gen.has_txn(999)

        # `has_entity` finds present + absent entities.
        assert gen.has_entity("card1", "A")
        assert not gen.has_entity("card1", "Z")
        assert gen.has_entity("addr1", 10)
        assert not gen.has_entity("addr1", 999)

        # `n_entity_nodes` with and without subtype filter.
        assert gen.n_entity_nodes() == 4
        assert gen.n_entity_nodes(subtype="card1") == 2
        assert gen.n_entity_nodes(subtype="DeviceInfo") == 0


# ---------------------------------------------------------------------
# `TestSaveLoad`: joblib + manifest round-trip.
# ---------------------------------------------------------------------


class TestSaveLoad:
    """Persistence: joblib payload + JSON manifest sidecar."""

    def test_save_writes_graph_and_manifest(self, tmp_path: Path) -> None:
        """Both `graph.joblib` and `graph_manifest.json` are written; manifest has expected shape."""
        import json

        df = _build_simple_frame()
        gen = TransactionEntityGraph(entity_cols=["card1", "addr1"])
        gen.build(df)
        graph_path, manifest_path = gen.save(tmp_path)

        assert graph_path.is_file()
        assert manifest_path.is_file()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 1
        assert manifest["entity_cols"] == ["card1", "addr1"]
        assert manifest["transaction_id_col"] == "TransactionID"
        assert manifest["n_nodes"] == 7
        assert manifest["n_edges"] == 6
        assert manifest["n_txn_nodes"] == 3
        assert manifest["n_entity_nodes"] == 4
        assert manifest["n_entity_nodes_by_subtype"] == {"card1": 2, "addr1": 2}

    def test_load_round_trip_produces_identical_graph(self, tmp_path: Path) -> None:
        """save → load reproduces the same node and edge sets bit-for-bit."""
        df = _build_simple_frame()
        gen = TransactionEntityGraph()
        gen.build(df)
        gen.save(tmp_path)

        reloaded = TransactionEntityGraph.load(tmp_path)
        assert reloaded.entity_cols == gen.entity_cols
        assert reloaded.transaction_id_col == gen.transaction_id_col
        assert set(reloaded.graph.nodes) == set(gen.graph.nodes)
        assert set(reloaded.graph.edges) == set(gen.graph.edges)
        # Node attributes preserved.
        for node, attrs in gen.graph.nodes(data=True):
            assert reloaded.graph.nodes[node] == attrs

    def test_load_rejects_wrong_object_type(self, tmp_path: Path) -> None:
        """`load` raises `TypeError` if the joblib payload isn't a `TransactionEntityGraph`."""
        joblib.dump({"not": "a graph"}, tmp_path / "graph.joblib")
        with pytest.raises(TypeError, match="expected TransactionEntityGraph"):
            TransactionEntityGraph.load(tmp_path)


# ---------------------------------------------------------------------
# `TestPerformance`: full-data memory benchmark.
# ---------------------------------------------------------------------


def _manifest_path() -> Path:
    return get_settings().raw_dir / "MANIFEST.json"


class TestPerformance:
    """Spec contract: full 590k-row train split fits in <8 GB."""

    @pytest.mark.slow
    def test_full_data_memory_under_8gb(self) -> None:
        """Build the graph on `splits.train`; tracemalloc peak < 8 GB.

        Pulls the full IEEE-CIS dataset, runs the cleaner + temporal
        split, and builds the graph on the training portion only.
        Records peak heap allocation and build wall-time for the
        completion report.
        """
        if not _manifest_path().is_file():
            pytest.skip("data/raw/MANIFEST.json not present — run `make data-download`.")

        settings = get_settings()
        loader = RawDataLoader()
        full = loader.load_merged(optimize=False)
        cleaned = TransactionCleaner().clean(full)
        splits = temporal_split(cleaned, settings=settings)

        gen = TransactionEntityGraph()

        tracemalloc.start()
        start = time.perf_counter()
        gen.build(splits.train)
        elapsed = time.perf_counter() - start
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_gb = peak / _BYTES_PER_GB

        # Echo to stdout for the completion report.
        print(
            f"\n[tier5-perf] train rows = {len(splits.train):,}; "
            f"nodes = {gen.n_nodes():,}; "
            f"edges = {gen.n_edges():,}; "
            f"build wall = {elapsed:.2f}s; "
            f"tracemalloc peak = {peak_gb:.3f} GB"
        )
        # Per-subtype entity counts for the report.
        for ec in gen.entity_cols:
            print(f"[tier5-perf]   entity_nodes[{ec}] = {gen.n_entity_nodes(subtype=ec):,}")

        # Hard gate at 8 GB.
        assert (
            peak_gb < _MEMORY_CEILING_GB
        ), f"Peak memory {peak_gb:.2f} GB exceeds {_MEMORY_CEILING_GB} GB ceiling"


# ---------------------------------------------------------------------
# `TestErrorHandling`: required columns + bad load paths.
# ---------------------------------------------------------------------


class TestErrorHandling:
    """Boundary conditions: missing columns, bad load paths."""

    def test_build_missing_columns_raises(self) -> None:
        """`build()` raises `KeyError` if a required column is missing."""
        df = pd.DataFrame({"card1": ["A"], "addr1": [10]})  # no TransactionID
        gen = TransactionEntityGraph(entity_cols=["card1", "addr1"])
        with pytest.raises(KeyError, match="TransactionID"):
            gen.build(df)

    def test_load_nonexistent_path_raises(self, tmp_path: Path) -> None:
        """`load` on a path with no `graph.joblib` raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            TransactionEntityGraph.load(tmp_path / "nope")

    def test_is_built_false_pre_build(self) -> None:
        """`is_built` is False on a fresh instance with no `build()` call."""
        gen = TransactionEntityGraph()
        assert gen.is_built is False
        assert gen.n_nodes() == 0

    def test_has_methods_on_empty_graph(self) -> None:
        """`has_txn` / `has_entity` return False on an empty graph."""
        gen = TransactionEntityGraph()
        assert gen.has_txn(123) is False
        assert gen.has_entity("card1", "A") is False
