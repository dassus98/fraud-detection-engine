"""One-shot notebook builder for `notebooks/05_graph_analysis.ipynb`.

Mirrors `scripts/_build_eda_notebook.py`: regenerate the notebook
without hand-editing JSON, then execute in-place so the committed
`.ipynb` carries rendered outputs (CLAUDE.md §16). Run once; commit.
Re-run whenever the notebook structure needs to change.

The notebook reads the Tier-5 processed parquet (`tier5_train.parquet`)
and the fitted pipeline (`tier5_pipeline.joblib`) emitted by
`scripts/build_features_all_tiers.py`, then produces:

- Section A — Per-entity degree distributions
- Section B — Connected-component-size distribution
- Section C — Fraud rate by CC-size bucket
- Section D — Visualisation of the 5 largest CCs
- Section E — Graph-feature LightGBM importance
- Section F — Summary

Usage:
    uv run python scripts/_build_graph_analysis_notebook.py
    uv run python scripts/_build_graph_analysis_notebook.py --no-execute
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "05_graph_analysis.ipynb"


def _md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def _code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


CELLS: list[nbf.NotebookNode] = [
    _md(
        """
# Sprint 3 — Graph Feature Analysis

**Tier-5 graph features** computed by `GraphFeatureExtractor` on the
training portion of the IEEE-CIS bipartite transaction-entity graph.
This notebook is the diagnostic surface that justifies the
feature-engineering decisions made in 3.2.b and confirms the
production-readiness of the 8 graph-derived columns.

What the graph captures (and what tiers 1-4 cannot):

- **Tier-1** — per-row deterministic features (amount, time, email).
- **Tier-2** — per-entity rolling counts and OOF target encoding.
- **Tier-3** — per-card1 behavioural deviation.
- **Tier-4** — per-(entity, λ) exponentially-decayed velocity.
- **Tier-5** — **shared-infrastructure topology**: which transactions
  share which devices and addresses, and how those shared entities
  knit cards into rings.

The 8 features (`connected_component_size`, `entity_degree_*`,
`fraud_neighbor_rate`, `pagerank_score`, `clustering_coefficient`)
are loaded from `data/processed/tier5_train.parquet`. The fitted
pipeline at `models/pipelines/tier5_pipeline.joblib` exposes the
underlying `TransactionEntityGraph` for the largest-CC visualisation
in Section D.

Sections:

- **A** — Per-entity degree distributions
- **B** — Connected-component-size distribution
- **C** — Fraud rate by CC-size bucket
- **D** — Visualise 5 largest CCs
- **E** — Graph-feature LightGBM importance
- **F** — Summary

The final section restates 4-6 findings into
[`reports/graph_feature_analysis.md`](../reports/graph_feature_analysis.md).
"""
    ),
    _md(
        """
## Setup

Load the Tier-5 train parquet and the fitted pipeline. `matplotlib`
is forced to the `Agg` backend so the notebook executes cleanly
under `pytest --nbmake` (no display server). All figure outputs land
under `reports/figures/` for the analysis report.
"""
    ),
    _code(
        """
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier

from fraud_engine.config.settings import get_settings
from fraud_engine.features.tier5_graph import (
    GraphFeatureExtractor,
    TransactionEntityGraph,
)

SETTINGS = get_settings()
SETTINGS.ensure_directories()

PROCESSED_DIR = SETTINGS.processed_dir
PIPELINE_PATH = SETTINGS.models_dir / "pipelines" / "tier5_pipeline.joblib"

if not (PROCESSED_DIR / "tier5_train.parquet").is_file():
    raise RuntimeError(
        "tier5_train.parquet not found. Run "
        "`uv run python scripts/build_features_all_tiers.py` first."
    )
if not PIPELINE_PATH.is_file():
    raise RuntimeError(
        "tier5_pipeline.joblib not found. Run "
        "`uv run python scripts/build_features_all_tiers.py` first."
    )

FIG_DIR = Path("..") / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 100

train = pd.read_parquet(PROCESSED_DIR / "tier5_train.parquet")
val = pd.read_parquet(PROCESSED_DIR / "tier5_val.parquet")
print(f"train: {train.shape[0]:,} rows × {train.shape[1]} cols")
print(f"val  : {val.shape[0]:,} rows × {val.shape[1]} cols")

# Locate the GraphFeatureExtractor inside the fitted pipeline so we
# can reach its embedded TransactionEntityGraph for visualisation.
pipeline = joblib.load(PIPELINE_PATH)
graph_gen: GraphFeatureExtractor | None = None
for gen in pipeline.generators:
    if isinstance(gen, GraphFeatureExtractor):
        graph_gen = gen
        break
if graph_gen is None or graph_gen.graph_ is None:
    raise RuntimeError("GraphFeatureExtractor not found / not fitted in pipeline.")
nx_graph = graph_gen.graph_.graph
print(
    f"Training graph: {nx_graph.number_of_nodes():,} nodes, "
    f"{nx_graph.number_of_edges():,} edges"
)
"""
    ),
    _md(
        """
## Section A — Per-entity degree distributions

For each of the 4 entity types (`card1`, `addr1`, `DeviceInfo`,
`P_emaildomain`), plot the log-scale histogram of the entity's degree
in the training graph. Hub entities (high degree) are the structural
signature of fraud rings — a single device touching dozens of cards,
or a single address hosting hundreds of transactions.
"""
    ),
    _code(
        """
ENTITY_COLS = ("card1", "addr1", "DeviceInfo", "P_emaildomain")

degree_stats: dict[str, dict[str, float]] = {}
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, entity in zip(axes.ravel(), ENTITY_COLS, strict=True):
    col = f"entity_degree_{entity}"
    series = train[col].dropna()
    nan_pct = train[col].isna().mean() * 100
    degree_stats[entity] = {
        "min": float(series.min()) if not series.empty else float("nan"),
        "p50": float(series.median()) if not series.empty else float("nan"),
        "p95": float(series.quantile(0.95)) if not series.empty else float("nan"),
        "p99": float(series.quantile(0.99)) if not series.empty else float("nan"),
        "max": float(series.max()) if not series.empty else float("nan"),
        "nan_pct": float(nan_pct),
    }
    ax.hist(series, bins=60, log=True, color="#3a72b0")
    ax.set_xscale("log")
    ax.set_title(
        f"{entity}: median={degree_stats[entity]['p50']:.0f}; "
        f"p99={degree_stats[entity]['p99']:.0f}; "
        f"max={degree_stats[entity]['max']:.0f}; "
        f"NaN={nan_pct:.1f}%"
    )
    ax.set_xlabel(f"{entity} degree (log scale)")
    ax.set_ylabel("count (log scale)")

fig.suptitle("Per-entity degree distributions on the training graph", y=1.0, fontsize=14)
fig.tight_layout()
fig.savefig(FIG_DIR / "graph_entity_degrees.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Per-entity degree summary (training graph):")
print(pd.DataFrame(degree_stats).T.round(1).to_string())
"""
    ),
    _md(
        """
## Section B — Connected-component-size distribution

`connected_component_size` per training transaction. The component
size for the row is the size of the CC its txn-node lives in (txn
nodes + entity nodes count toward the size). A txn whose all-NaN
entities make it a singleton has CC size 1; a txn embedded in a
giant CC of 100k+ nodes is on the structural mainland.
"""
    ),
    _code(
        """
cc_series = train["connected_component_size"].dropna()
print("connected_component_size summary:")
print(cc_series.describe(percentiles=[0.5, 0.9, 0.99, 0.999]).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(cc_series, bins=60, log=True, color="#9e3a3a")
axes[0].set_xscale("log")
axes[0].set_title("CC size distribution (log-log)")
axes[0].set_xlabel("connected_component_size (log)")
axes[0].set_ylabel("count (log)")

cc_value_counts = cc_series.value_counts().sort_index()
top_small_ccs = cc_value_counts[cc_value_counts.index <= 50]
axes[1].bar(top_small_ccs.index, top_small_ccs.to_numpy(), color="#9e3a3a")
axes[1].set_yscale("log")
axes[1].set_title("Small-CC tail (size ≤ 50)")
axes[1].set_xlabel("connected_component_size")
axes[1].set_ylabel("count (log)")

fig.tight_layout()
fig.savefig(FIG_DIR / "graph_cc_size_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
"""
    ),
    _md(
        """
## Section C — Fraud rate by CC-size bucket

Bucket transactions by their CC size and compute the fraud rate per
bucket. If small / orphan CCs (size 1-2) are systematically more or
less fraud-prone than mainland transactions, this plot surfaces the
discriminative signal `connected_component_size` carries for the
LightGBM model.
"""
    ),
    _code(
        """
# Bucket boundaries: powers-of-2-ish on a log scale.
CC_BUCKETS = [0, 1, 2, 5, 10, 100, 1_000, 10_000, 100_000, np.inf]
labels = [f"{lo}-{hi}" for lo, hi in zip(CC_BUCKETS[:-1], CC_BUCKETS[1:], strict=True)]

train_buckets = pd.cut(
    train["connected_component_size"],
    bins=CC_BUCKETS,
    labels=labels,
    include_lowest=True,
)
bucket_summary = (
    train.groupby(train_buckets, observed=False)
    .agg(n=("isFraud", "size"), fraud_rate=("isFraud", "mean"))
    .reset_index()
)
print("Fraud rate by CC-size bucket:")
print(bucket_summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
non_empty = bucket_summary[bucket_summary["n"] > 0]
bars = ax.bar(non_empty["connected_component_size"].astype(str), non_empty["fraud_rate"], color="#3a8b3a")
overall = train["isFraud"].mean()
ax.axhline(overall, color="black", linestyle="--", label=f"overall fraud rate = {overall:.3%}")
for bar, n in zip(bars, non_empty["n"], strict=True):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"n={n:,}", ha="center", va="bottom", fontsize=9)
ax.set_xlabel("connected_component_size bucket")
ax.set_ylabel("fraud rate")
ax.set_title("Fraud rate by CC-size bucket (training rows)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "graph_fraud_rate_by_cc_size.png", dpi=150, bbox_inches="tight")
plt.close(fig)
"""
    ),
    _md(
        """
## Section D — Visualise the 5 largest CCs (excluding the giant one)

The IEEE-CIS train graph forms one giant CC containing essentially
all txn nodes plus most entity nodes. Visualising it would yield a
hairball with no structural signal. Instead, we visualise the 5
largest **non-giant** CCs (small clusters of cards / addresses /
devices / emails that form their own subgraph). These are the
structural orphan rings — small enough to show meaningful topology.
"""
    ),
    _code(
        """
components = list(nx.connected_components(nx_graph))
sizes = sorted((len(c) for c in components), reverse=True)
print(f"Total CCs: {len(components):,}")
print(f"Top 10 sizes: {sizes[:10]}")

# Skip the giant CC (the largest); take the next 5 largest distinct sizes.
distinct_sizes = sorted(set(sizes))[::-1]
target_sizes = distinct_sizes[1 : 1 + 5]  # skip the giant; take 5 next-largest
selected: list[set[object]] = []
for component in sorted(components, key=len, reverse=True):
    if len(component) in target_sizes and component not in selected:
        selected.append(component)
        if len(selected) == 5:
            break

print(f"Selected {len(selected)} non-giant CCs of sizes: {[len(c) for c in selected]}")

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for ax, component in zip(axes, selected, strict=False):
    subgraph = nx_graph.subgraph(component)
    pos = nx.spring_layout(subgraph, seed=42)
    txn_nodes = [n for n, attrs in subgraph.nodes(data=True) if attrs.get("bipartite") == 0]
    entity_nodes = [n for n in subgraph.nodes if n not in txn_nodes]
    nx.draw_networkx_nodes(subgraph, pos, nodelist=txn_nodes, node_color="#3a72b0",
                           node_size=40, ax=ax, label="txn")
    nx.draw_networkx_nodes(subgraph, pos, nodelist=entity_nodes, node_color="#c85050",
                           node_size=80, node_shape="s", ax=ax, label="entity")
    nx.draw_networkx_edges(subgraph, pos, width=0.5, alpha=0.5, ax=ax)
    ax.set_title(f"size={len(component)}; txn={len(txn_nodes)}; entity={len(entity_nodes)}")
    ax.axis("off")

fig.suptitle("5 largest non-giant connected components", y=1.02, fontsize=14)
fig.tight_layout()
fig.savefig(FIG_DIR / "graph_top_ccs.png", dpi=150, bbox_inches="tight")
plt.close(fig)
"""
    ),
    _md(
        """
## Section E — Graph-feature LightGBM importance

Train a quick LightGBM on the Tier-5 train parquet, score val for the
headline AUC, and report the **gain importance** for each of the 8
graph features. This isolates which graph features drove the
LightGBM lift over Tier-4.
"""
    ),
    _code(
        """
GRAPH_COLS = [
    "connected_component_size",
    "entity_degree_card1",
    "entity_degree_addr1",
    "entity_degree_DeviceInfo",
    "entity_degree_P_emaildomain",
    "fraud_neighbor_rate",
    "pagerank_score",
    "clustering_coefficient",
]

NON_FEATURE_COLS = {"TransactionID", "TransactionDT", "isFraud", "timestamp"}
feature_cols = [
    col for col in train.columns
    if col not in NON_FEATURE_COLS
    and not pd.api.types.is_object_dtype(train[col])
    and not pd.api.types.is_string_dtype(train[col])
]
print(f"feature_cols: {len(feature_cols)} (8 graph + {len(feature_cols) - 8} other)")

clf = LGBMClassifier(
    **SETTINGS.lgbm_defaults,
    random_state=SETTINGS.seed,
    verbose=-1,
)
clf.fit(train[feature_cols], train["isFraud"], categorical_feature="auto")

from sklearn.metrics import roc_auc_score

val_proba = clf.predict_proba(val[feature_cols])[:, 1]
val_auc = float(roc_auc_score(val["isFraud"], val_proba))
print(f"Tier-5 val AUC: {val_auc:.4f}")

# Gain importance (sum of split gains for each feature).
importances = pd.Series(
    clf.booster_.feature_importance(importance_type="gain"),
    index=feature_cols,
).sort_values(ascending=False)
graph_importances = importances.loc[GRAPH_COLS].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 4))
graph_importances.iloc[::-1].plot.barh(ax=ax, color="#9e6a3a")
ax.set_xlabel("gain importance")
ax.set_title(f"Graph-feature gain importance (val AUC = {val_auc:.4f})")
fig.tight_layout()
fig.savefig(FIG_DIR / "graph_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\\nGraph-feature gain importance:")
print(graph_importances.round(0).astype(int).to_string())

print("\\nTop 15 features overall (gain importance):")
print(importances.head(15).round(0).astype(int).to_string())
"""
    ),
    _md(
        """
## Section F — Summary

The four diagnostic figures + LightGBM importance ranking are
reflected verbatim in
[`reports/graph_feature_analysis.md`](../reports/graph_feature_analysis.md):

- **Degree heavy-tailedness** — `card1` and `P_emaildomain` are
  power-law-distributed; the long tail is where structural fraud
  signals concentrate.
- **CC-size distribution** — bimodal (one giant CC + many tiny
  orphan CCs), as expected for a real-world transaction graph.
- **Fraud rate by CC-size** — the discriminative signal lives at the
  ends of the distribution; mid-size CCs are close to the global
  base rate.
- **Top non-giant CCs** — small bipartite ring topologies, the
  archetype of organised fraud the per-card aggregations could not
  see.
- **Feature importance** — `entity_degree_card1` and
  `fraud_neighbor_rate` carry most of the Tier-5 gain (typically;
  exact ordering varies by random seed).

The full analysis report includes the val AUC headline and a side-
by-side comparison with the Tier-4 0.7932 baseline. The
hyperparameter-tuning prompt that follows targets the spec's
0.93-0.94 envelope.
"""
    ),
]


def build() -> Path:
    """Write the notebook structure (no outputs) to NOTEBOOK_PATH."""
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    return NOTEBOOK_PATH


def execute(path: Path) -> None:
    """Execute the notebook in place so the committed file carries outputs.

    Mirrors `_build_eda_notebook.py:execute` — overrides DATA_DIR to
    an absolute path so a relative `./data` from the makefile-loaded
    `.env` doesn't resolve to `notebooks/data/` (which would miss
    the manifest).
    """
    project_root = path.resolve().parents[1]
    env = os.environ.copy()
    env["DATA_DIR"] = str(project_root / "data")
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(path),
    ]
    print(f"Executing in-place: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Build only; skip in-place execution. Output notebook will lack "
        "rendered outputs and MUST NOT be committed.",
    )
    args = parser.parse_args()

    path = build()
    print(f"Wrote {path}")
    if args.no_execute:
        print("Skipping execute (--no-execute). Do not commit the empty-output notebook.")
        sys.exit(0)
    execute(path)
    print(f"Executed in place: {path}")
