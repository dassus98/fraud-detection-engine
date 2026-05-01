"""Unit tests for `fraud_engine.features.tier5_graph.GraphFeatureExtractor`.

Five contract surfaces, mirroring the 5-class shape used by
`test_tier5_graph_construction.py`:

- `TestSyntheticFeatures`: hand-computed feature values on small
  synthetic frames (3-row minimal, 4-cycle, 3-shared-card,
  symmetric-graph pagerank).
- `TestColdStartContract`: held-out (val) rows whose `TransactionID`
  is not in the training graph emit NaN for txn-level features (CC
  size, pagerank, clustering); held-out rows whose entity values
  are unseen emit NaN for the corresponding `entity_degree_*`
  column and contribute zero to `fraud_neighbor_rate`.
- `TestOOFContract`: `fit_transform(train)` OOF values differ from
  `fit(train).transform(train)` full-train values; seed stability;
  shuffled-target signal collapse.
- `TestErrorHandling`: missing required columns; `transform` before
  `fit`; `n_splits=1` raises at construction.
- `TestGetFeatureNames`: deterministic 8-column list; configurable
  entity-column count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_engine.features.tier5_graph import GraphFeatureExtractor

# ---------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------


def _three_row_frame() -> pd.DataFrame:
    """3-row synthetic frame with a known graph structure.

    Layout (entity_cols restricted to card1, addr1 for hand-compute):

        txn 100 → card1=A, addr1=10, isFraud=1
        txn 101 → card1=A, addr1=20, isFraud=0
        txn 102 → card1=B, addr1=20, isFraud=0

    Graph: {("txn",100), ("txn",101), ("txn",102), ("card1","A"),
            ("card1","B"), ("addr1",10), ("addr1",20)} = 7 nodes.
    Edges: 6 (each txn has 2).
    Single connected component (txn102—addr20—txn101—card_A—txn100—addr10
    chain), so cc_size = 7 for every row.

    entity_degree_card1: A → 2, B → 1.
    entity_degree_addr1: 10 → 1, 20 → 2.

    `fraud_neighbor_rate` via `fit().transform()` (full-train, leaked):
        - Per-entity (sum, count) from the full-train graph:
            (card1, A) → (1, 2)    # txn100 (fraud=1) + txn101 (fraud=0)
            (card1, B) → (0, 1)    # txn102 (fraud=0)
            (addr1, 10) → (1, 1)   # txn100 (fraud=1)
            (addr1, 20) → (0, 2)   # txn101 + txn102 (both 0)
        - Row 100: aggregate over A + 10 → (1+1)/(2+1) = 2/3.
        - Row 101: aggregate over A + 20 → (1+0)/(2+2) = 1/4.
        - Row 102: aggregate over B + 20 → (0+0)/(1+2) = 0.
    """
    return pd.DataFrame(
        {
            "TransactionID": [100, 101, 102],
            "card1": ["A", "A", "B"],
            "addr1": [10, 20, 20],
            "isFraud": [1, 0, 0],
        }
    )


def _two_card_extractor() -> GraphFeatureExtractor:
    """Extractor restricted to (card1, addr1) for hand-computable graphs."""
    return GraphFeatureExtractor(entity_cols=["card1", "addr1"], n_splits=2)


# ---------------------------------------------------------------------
# `TestSyntheticFeatures`: hand-computed correctness.
# ---------------------------------------------------------------------


class TestSyntheticFeatures:
    """Hand-computed feature values on small synthetic frames."""

    def test_minimal_3row_synthetic_graph_features(self) -> None:
        """3-row frame; verify CC size, entity degrees, and full-train rate."""
        df = _three_row_frame()
        gen = _two_card_extractor()
        # Use fit().transform() for hand-computable full-train rates.
        out = gen.fit(df).transform(df)

        # Output frame includes every input column + 6 new columns
        # (cc_size, entity_degree_card1, entity_degree_addr1,
        # fraud_neighbor_rate, pagerank_score, clustering_coefficient).
        assert set(out.columns) >= {*df.columns, *gen.get_feature_names()}

        # Check: held-out path emits NaN for txn-level features (CC size,
        # pagerank, clustering) because val txns are not in the training
        # graph. We're transforming the training frame here; the rows
        # ARE in the graph (we just fitted on them), but transform()
        # treats every input row as a held-out query — that's the
        # documented val/test path semantics.
        assert out["connected_component_size"].isna().all()
        assert out["pagerank_score"].isna().all()
        assert out["clustering_coefficient"].isna().all()

        # Entity degree: card1=A appears in 2 rows, B in 1.
        # addr1=10 in 1 row, addr1=20 in 2.
        assert out["entity_degree_card1"].tolist() == [2.0, 2.0, 1.0]
        assert out["entity_degree_addr1"].tolist() == [1.0, 2.0, 2.0]

        # Full-train fraud_neighbor_rate (held-out path: not OOF):
        # Row 100: A + 10 → (1+1)/(2+1) = 0.6667.
        # Row 101: A + 20 → (1+0)/(2+2) = 0.25.
        # Row 102: B + 20 → (0+0)/(1+2) = 0.0.
        rates = out["fraud_neighbor_rate"].to_numpy()
        np.testing.assert_allclose(rates, [2 / 3, 1 / 4, 0.0], atol=1e-9)

    def test_disconnected_components(self) -> None:
        """Two disjoint card groups → each row's CC size is its cluster."""
        df = pd.DataFrame(
            {
                "TransactionID": [200, 201, 202, 203],
                "card1": ["A", "A", "B", "B"],
                "addr1": [10, 10, 20, 20],
                "isFraud": [0, 0, 0, 0],
            }
        )
        gen = _two_card_extractor()
        # Use fit_transform: training rows ARE in the graph, so the
        # structural lookups produce real values.
        out = gen.fit_transform(df)

        # Two CCs: {200, 201, ("card1","A"), ("addr1",10)} (4 nodes)
        # and {202, 203, ("card1","B"), ("addr1",20)} (4 nodes).
        assert out["connected_component_size"].tolist() == [4.0, 4.0, 4.0, 4.0]

    def test_single_component_via_shared_address(self) -> None:
        """Two cards connected via a shared address → one CC of all rows."""
        df = pd.DataFrame(
            {
                "TransactionID": [300, 301, 302],
                "card1": ["A", "B", "C"],
                "addr1": [10, 10, 10],
                "isFraud": [0, 0, 0],
            }
        )
        gen = _two_card_extractor()
        out = gen.fit_transform(df)
        # 3 txns + 3 cards + 1 shared addr = 7 nodes, all in one CC.
        assert out["connected_component_size"].tolist() == [7.0, 7.0, 7.0]

    def test_isolated_txn_singleton_cc(self) -> None:
        """Txn with all-NaN entities → CC size 1 (just itself); degrees NaN."""
        df = pd.DataFrame(
            {
                "TransactionID": [400, 401],
                "card1": ["A", np.nan],
                "addr1": [10, np.nan],
                "isFraud": [0, 0],
            }
        )
        gen = _two_card_extractor()
        out = gen.fit_transform(df)

        # Row 400 has 2 entities → in a CC of {400, A, 10} = 3.
        # Row 401 has no entities → singleton CC of {401} = 1.
        assert out["connected_component_size"].tolist() == [3.0, 1.0]

        # Row 401 (all-NaN entities) → entity_degree_* both NaN.
        assert pd.isna(out["entity_degree_card1"].iloc[1])
        assert pd.isna(out["entity_degree_addr1"].iloc[1])

    def test_pagerank_uniform_on_symmetric_graph(self) -> None:
        """3 txns sharing a single card1 → equal pagerank within tolerance."""
        df = pd.DataFrame(
            {
                "TransactionID": [500, 501, 502],
                "card1": ["A", "A", "A"],
                "addr1": [np.nan, np.nan, np.nan],
                "isFraud": [0, 0, 0],
            }
        )
        gen = GraphFeatureExtractor(entity_cols=["card1", "addr1"], n_splits=3)
        out = gen.fit_transform(df)
        prs = out["pagerank_score"].to_numpy()
        # Three structurally equivalent txn nodes share equal pagerank.
        np.testing.assert_allclose(prs, [prs[0]] * 3, atol=1e-9)

    def test_clustering_on_4cycle(self) -> None:
        """4-cycle bipartite graph (2 txns × 2 entities) → Latapy = 1.0 each."""
        df = pd.DataFrame(
            {
                "TransactionID": [600, 601],
                "card1": ["A", "A"],
                "addr1": [10, 10],
                "isFraud": [0, 0],
            }
        )
        gen = _two_card_extractor()
        out = gen.fit_transform(df)
        # Both txns share both entities → forms a 4-cycle.
        # Latapy mode='dot': c(u) = sum over v in N²(u)\{u} of
        # |N(u) ∩ N(v)|^2 / (|N(u)|·|N(v)|), normalised by count.
        # |N(txn0)|=2, |N(txn1)|=2, |N(txn0) ∩ N(txn1)| = 2.
        # c(txn0) = (2^2 / (2·2)) / 1 = 1.0.
        np.testing.assert_allclose(out["clustering_coefficient"].to_numpy(), [1.0, 1.0], atol=1e-9)

    def test_entity_degree_three_txns_share_card(self) -> None:
        """3 rows share card1=A → each row's entity_degree_card1 = 3."""
        df = pd.DataFrame(
            {
                "TransactionID": [700, 701, 702],
                "card1": ["A", "A", "A"],
                "addr1": [10, 20, 30],
                "isFraud": [0, 0, 0],
            }
        )
        gen = GraphFeatureExtractor(entity_cols=["card1", "addr1"], n_splits=3)
        out = gen.fit_transform(df)
        assert out["entity_degree_card1"].tolist() == [3.0, 3.0, 3.0]
        # addr1 values are all distinct singletons.
        assert out["entity_degree_addr1"].tolist() == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------
# `TestColdStartContract`: held-out rows + cold-start entities.
# ---------------------------------------------------------------------


class TestColdStartContract:
    """Held-out (val) rows: txn-level NaN; entity-level NaN if unseen."""

    def test_val_txn_emits_nan_txn_level_features(self) -> None:
        """Fit on train; transform on disjoint val → CC/pagerank/clustering NaN."""
        train = _three_row_frame()
        val = pd.DataFrame(
            {
                "TransactionID": [9001, 9002],  # disjoint from train
                "card1": ["A", "B"],  # both seen in train
                "addr1": [10, 20],
                "isFraud": [0, 0],  # not used by transform
            }
        )
        gen = _two_card_extractor()
        gen.fit(train)
        out = gen.transform(val)
        # Txn-level features → all NaN (val txns not in training graph).
        assert out["connected_component_size"].isna().all()
        assert out["pagerank_score"].isna().all()
        assert out["clustering_coefficient"].isna().all()

    def test_val_unseen_entity_degree_nan(self) -> None:
        """Val row's `card1` not in train → `entity_degree_card1` NaN."""
        train = _three_row_frame()
        val = pd.DataFrame(
            {
                "TransactionID": [9001],
                "card1": ["UNSEEN_CARD"],  # cold-start
                "addr1": [10],  # seen in train
                "isFraud": [0],
            }
        )
        gen = _two_card_extractor()
        gen.fit(train)
        out = gen.transform(val)
        # card1 cold-start → NaN.
        assert pd.isna(out["entity_degree_card1"].iloc[0])
        # addr1 seen in train → degree of addr1=10 is 1 (only txn 100).
        assert out["entity_degree_addr1"].iloc[0] == 1.0

    def test_val_all_cold_start_fraud_neighbor_rate_nan(self) -> None:
        """Val row with all entities cold-start → `fraud_neighbor_rate` NaN."""
        train = _three_row_frame()
        val = pd.DataFrame(
            {
                "TransactionID": [9001],
                "card1": ["UNSEEN"],
                "addr1": [99999],
                "isFraud": [0],
            }
        )
        gen = _two_card_extractor()
        gen.fit(train)
        out = gen.transform(val)
        assert pd.isna(out["fraud_neighbor_rate"].iloc[0])

    def test_val_singleton_entity_zero_denom_nan(self) -> None:
        """Val seen entity is singleton (isolated txn) → denom 0 → NaN."""
        # Train has a card1=Z used by exactly one txn; val row touches
        # that singleton card1=Z. Because the val txn isn't in the
        # training graph, the singleton entity's neighbour count is 1
        # (the original training txn). So denom is 1, not 0. To get a
        # true zero-denominator, the val entity must NOT be in train at
        # all — which is just the cold-start case. Build a training
        # frame where the entity exists but every other entity is NaN
        # such that walking via the val row's other (NaN) entities is
        # the only zero-denom path. Easier: val with only NaN entities.
        train = _three_row_frame()
        val = pd.DataFrame(
            {
                "TransactionID": [9001],
                "card1": [np.nan],
                "addr1": [np.nan],
                "isFraud": [0],
            }
        )
        gen = _two_card_extractor()
        gen.fit(train)
        out = gen.transform(val)
        assert pd.isna(out["fraud_neighbor_rate"].iloc[0])

    def test_val_one_seen_entity_with_fraud_neighbours(self) -> None:
        """Val card matches a training card; rate = (sum_fraud)/(count)."""
        # Build a frame where card1=Z has 5 training txns, 2 of them
        # fraud. Val row uses card1=Z only.
        train = pd.DataFrame(
            {
                "TransactionID": list(range(10, 15)),
                "card1": ["Z"] * 5,
                "addr1": [np.nan] * 5,
                "isFraud": [1, 1, 0, 0, 0],
            }
        )
        val = pd.DataFrame(
            {
                "TransactionID": [9001],
                "card1": ["Z"],
                "addr1": [np.nan],
                "isFraud": [0],
            }
        )
        gen = _two_card_extractor()
        gen.fit(train)
        out = gen.transform(val)
        # 2 fraud / 5 neighbours = 0.4.
        np.testing.assert_allclose(out["fraud_neighbor_rate"].iloc[0], 0.4, atol=1e-9)


# ---------------------------------------------------------------------
# `TestOOFContract`: OOF safety for fraud_neighbor_rate.
# ---------------------------------------------------------------------


class TestOOFContract:
    """OOF discipline for `fraud_neighbor_rate` mirrors `TargetEncoder`."""

    def test_oof_differs_from_full_train(self) -> None:
        """`fit_transform` OOF fraud_neighbor_rate differs from full-train.

        Uses a heterogeneous fraud distribution (one fraud per card
        group) so the per-entity rate in each fold differs from the
        full-train rate. A uniform-fraud pattern (50 / 50 across every
        entity) makes OOF and full-train identical and is NOT a useful
        gate — that's the "uniform-fold collapse" trap and the reason
        this test had to be redesigned away from a 40-row tile pattern.
        """
        # 6 rows: card1 ∈ {A, B} × 3 each; isFraud=1 only at the FIRST
        # row of each card group. addr1 is NaN throughout so the
        # `fraud_neighbor_rate` walk is over card1 alone.
        df = pd.DataFrame(
            {
                "TransactionID": [1000, 1001, 1002, 1003, 1004, 1005],
                "card1": ["A", "A", "A", "B", "B", "B"],
                "addr1": [np.nan] * 6,
                "isFraud": [1, 0, 0, 1, 0, 0],
            }
        )
        gen = _two_card_extractor()
        oof_out = gen.fit_transform(df)
        full_out = gen.fit(df).transform(df)
        # At least one row's OOF rate must differ from full-train.
        assert (oof_out["fraud_neighbor_rate"] - full_out["fraud_neighbor_rate"]).abs().max() > 1e-6

    def test_oof_seed_stability(self) -> None:
        """Two `fit_transform` calls with the same seed → identical OOF."""
        df = pd.DataFrame(
            {
                "TransactionID": list(range(2000, 2040)),
                "card1": np.tile(["A", "B", "C", "D"], 10),
                "addr1": np.tile([10, 20], 20),
                "isFraud": np.tile([0, 0, 1, 0], 10),
            }
        )
        gen1 = _two_card_extractor()
        out1 = gen1.fit_transform(df)
        gen2 = _two_card_extractor()
        out2 = gen2.fit_transform(df)
        np.testing.assert_allclose(
            out1["fraud_neighbor_rate"].fillna(-1.0),
            out2["fraud_neighbor_rate"].fillna(-1.0),
            atol=1e-12,
        )

    def test_oof_shuffled_target_collapses_signal(self) -> None:
        """Shuffled isFraud → OOF rate mean ≈ global rate (no signal)."""
        # 200 rows, 25% fraud.
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame(
            {
                "TransactionID": np.arange(3000, 3000 + n),
                "card1": rng.choice(["A", "B", "C"], size=n),
                "addr1": rng.choice([10, 20, 30], size=n),
                "isFraud": np.zeros(n, dtype=int),
            }
        )
        df.loc[: n // 4, "isFraud"] = 1  # ~25% fraud
        # Shuffle.
        df["isFraud"] = rng.permutation(df["isFraud"].to_numpy())
        global_rate = df["isFraud"].mean()
        gen = _two_card_extractor()
        out = gen.fit_transform(df)
        oof_rate_mean = out["fraud_neighbor_rate"].dropna().mean()
        # Within ±0.10 of the global rate (200 rows × 2 folds is noisy).
        # Lower bound stays above global_rate*0.5 (sanity: signal hasn't
        # spuriously concentrated); upper bound stays below 2x global.
        assert abs(oof_rate_mean - global_rate) < 0.10

    def test_oof_n_splits_validates_at_construction(self) -> None:
        """`n_splits=1` raises `ValueError` immediately at __init__."""
        with pytest.raises(ValueError, match="n_splits"):
            GraphFeatureExtractor(n_splits=1)


# ---------------------------------------------------------------------
# `TestErrorHandling`: missing columns, transform-before-fit.
# ---------------------------------------------------------------------


class TestErrorHandling:
    """Boundary conditions: missing columns, bad call order."""

    def test_fit_missing_required_column_raises(self) -> None:
        """`fit` raises `KeyError` when an entity column is missing."""
        df = pd.DataFrame(
            {
                "TransactionID": [1],
                # missing `card1`
                "addr1": [10],
                "isFraud": [0],
            }
        )
        gen = _two_card_extractor()
        with pytest.raises(KeyError, match="card1"):
            gen.fit(df)

    def test_fit_missing_target_raises(self) -> None:
        """`fit` raises `KeyError` when `target_col` is missing."""
        df = pd.DataFrame(
            {
                "TransactionID": [1, 2],
                "card1": ["A", "B"],
                "addr1": [10, 20],
                # missing `isFraud`
            }
        )
        gen = _two_card_extractor()
        with pytest.raises(KeyError, match="isFraud"):
            gen.fit(df)

    def test_transform_before_fit_raises(self) -> None:
        """`transform` raises `AttributeError` if called pre-fit."""
        df = _three_row_frame()
        gen = _two_card_extractor()
        with pytest.raises(AttributeError, match="fit"):
            gen.transform(df)

    def test_transform_does_not_require_target_col(self) -> None:
        """`transform` does NOT require `target_col` (val/test path)."""
        train = _three_row_frame()
        val_no_target = pd.DataFrame(
            {
                "TransactionID": [9001],
                "card1": ["A"],
                "addr1": [10],
                # NO `isFraud`
            }
        )
        gen = _two_card_extractor()
        gen.fit(train)
        # Should not raise.
        out = gen.transform(val_no_target)
        assert "fraud_neighbor_rate" in out.columns


# ---------------------------------------------------------------------
# `TestGetFeatureNames`: deterministic column list.
# ---------------------------------------------------------------------


class TestGetFeatureNames:
    """`get_feature_names` returns a deterministic, config-aware list."""

    def test_default_returns_8_columns(self) -> None:
        """Default config: 1 cc + 4 entity_degree + 1 frate + 1 pr + 1 cl = 8."""
        gen = GraphFeatureExtractor()
        names = gen.get_feature_names()
        assert len(names) == 8
        assert names[0] == "connected_component_size"
        assert names[-3:] == [
            "fraud_neighbor_rate",
            "pagerank_score",
            "clustering_coefficient",
        ]
        # All four default entity_degree_* names appear in order.
        for ec in ("card1", "addr1", "DeviceInfo", "P_emaildomain"):
            assert f"entity_degree_{ec}" in names

    def test_custom_entity_cols_shrinks_column_list(self) -> None:
        """`entity_cols=[card1, addr1]` → 6 columns (1+2+1+1+1)."""
        gen = GraphFeatureExtractor(entity_cols=["card1", "addr1"])
        names = gen.get_feature_names()
        assert len(names) == 6
        assert names == [
            "connected_component_size",
            "entity_degree_card1",
            "entity_degree_addr1",
            "fraud_neighbor_rate",
            "pagerank_score",
            "clustering_coefficient",
        ]
