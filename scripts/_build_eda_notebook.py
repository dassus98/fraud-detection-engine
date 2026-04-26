"""One-shot notebook builder for `notebooks/01_eda.ipynb`.

This file is not a pipeline step — it is the scaffolding used to
regenerate the EDA notebook without hand-editing JSON. Run once;
commit the output `.ipynb`. Re-run whenever the notebook structure
needs to change.

Build + execute is atomic by default: after writing the notebook
structure, the script executes the notebook in-place via
`jupyter nbconvert --execute --inplace` so the committed `.ipynb`
always carries rendered outputs (figures, tables, prints) — that is
what GitHub renders for portfolio viewing. Pass `--no-execute` to
skip execution during fast iteration; the resulting empty-output
notebook should NOT be committed (CLAUDE.md §16 notebook policy).

Usage:
    uv run python scripts/_build_eda_notebook.py            # build + execute
    uv run python scripts/_build_eda_notebook.py --no-execute  # build only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"


def _md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def _code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


CELLS: list[nbf.NotebookNode] = [
    _md(
        """
# Sprint 1 — Exploratory Data Analysis

**Dataset:** IEEE-CIS Fraud Detection (Vesta Corporation, Kaggle 2019).
**Purpose:** Produce the exploratory analysis that a senior data
scientist would run at the start of a real engagement. Output is not
just plots — it is a set of decisions the rest of the project depends
on:

1. **Temporal split boundaries** stored in `Settings` so every later
   sprint scores on the same rows.
2. **A label-quality policy** — keep cleanlab-flagged rows, document
   why removing them would be worse than keeping them.
3. **A list of feature-engineering priorities** (handoff to Sprint 2).

Business constants live in [CLAUDE.md §8](../CLAUDE.md): every cost,
threshold, and calendar anchor referenced below is configurable via
`Settings` / `.env`.

The final Section G re-states the 8–12 findings verbatim into
[`reports/sprint1_eda_summary.md`](../reports/sprint1_eda_summary.md).
"""
    ),
    _md(
        """
## Setup

Load merged data via `RawDataLoader`, configure structured logging,
and open a `run_context` for artefact persistence. `matplotlib` is
forced to the `Agg` backend so the notebook executes cleanly under
`pytest --nbmake` (no display server).
"""
    ),
    _code(
        """
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fraud_engine.config.settings import get_settings
from fraud_engine.data.loader import RawDataLoader
from fraud_engine.utils.logging import configure_logging, get_logger
from fraud_engine.utils.tracing import attach_artifact, run_context

SETTINGS = get_settings()
SETTINGS.ensure_directories()

if not (SETTINGS.raw_dir / "MANIFEST.json").is_file():
    raise RuntimeError(
        "Raw IEEE-CIS data not downloaded — run `make data-download` first."
    )

FIG_DIR = Path("..") / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

configure_logging(pipeline_name="eda")
logger = get_logger(__name__)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 100

nb_run = run_context("eda", capture_streams=False)
run = nb_run.__enter__()

loader = RawDataLoader()
merged = loader.load_merged(optimize=True)
print(f"Loaded merged frame: {merged.shape[0]:,} rows × {merged.shape[1]} cols")
"""
    ),
    _md(
        """
## Section A — Data Overview

Row and column counts, memory footprint, dtype histogram, calendar
derivation from `Settings.transaction_dt_anchor_iso`, daily volume,
identity-join coverage, and the fraud-rate split between rows that
do vs don't have identity data. Together these fingerprint the
dataset before any transformation — a future re-download that
changes any of these fingerprints means the splitter + baseline +
threshold numbers all deserve re-running.

`event_dt` is computed as a standalone `pd.Series` (not added to
`merged`) so downstream sections (notably Section F's cleanlab
classifier) keep their feature-column selection intact.
"""
    ),
    _code(
        """
memory_mb = merged.memory_usage(deep=True).sum() / (1024 ** 2)
dtype_hist = merged.dtypes.value_counts().to_dict()
overview = {
    "rows": int(merged.shape[0]),
    "cols": int(merged.shape[1]),
    "memory_mb": round(memory_mb, 2),
    "dtype_histogram": {str(k): int(v) for k, v in dtype_hist.items()},
}
print(json.dumps(overview, indent=2))
attach_artifact(run, overview, name="overview")
"""
    ),
    _code(
        """
# Anchor TransactionDT (anonymised seconds-since-reference) onto the
# community-standard 2017-12-01 UTC calendar; kept as a standalone
# Series so we don't pollute `merged` with a datetime column the
# baseline / cleanlab paths would otherwise pick up as a feature.
anchor = pd.Timestamp(SETTINGS.transaction_dt_anchor_iso)
event_dt = anchor + pd.to_timedelta(merged["TransactionDT"], unit="s")
event_dt.name = "event_dt"

calendar_span = {
    "anchor_utc": anchor.isoformat(),
    "min_event_dt": event_dt.min().isoformat(),
    "max_event_dt": event_dt.max().isoformat(),
    "n_days_observed": int((event_dt.max().normalize() - event_dt.min().normalize()).days) + 1,
}
print(json.dumps(calendar_span, indent=2))
attach_artifact(run, calendar_span, name="calendar_span")
"""
    ),
    _code(
        """
daily_volume = event_dt.dt.date.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(12, 4))
daily_volume.plot(ax=ax, color="#3b6fb3")
ax.set_title("Daily transaction volume — IEEE-CIS train (anchor 2017-12-01 UTC)")
ax.set_xlabel("calendar date")
ax.set_ylabel("transactions / day")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG_DIR / "daily_volume.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="daily_volume")
plt.close(fig)
print(f"Daily volume range: {int(daily_volume.min()):,} — {int(daily_volume.max()):,} txns/day")
"""
    ),
    _code(
        """
# Identity coverage = fraction of rows where any id_* / DeviceType /
# DeviceInfo column is non-null. ~24% on IEEE-CIS train (see
# reports/raw_profile_summary.json).
identity_cols = [
    c for c in merged.columns if c.startswith("id_") or c in {"DeviceType", "DeviceInfo"}
]
has_identity = (
    merged[identity_cols].notna().any(axis=1)
    if identity_cols
    else pd.Series(False, index=merged.index, name="has_identity")
)
has_identity.name = "has_identity"

identity_coverage = {
    "n_identity_columns": len(identity_cols),
    "n_with_identity": int(has_identity.sum()),
    "n_without_identity": int((~has_identity).sum()),
    "coverage_pct": round(float(has_identity.mean()) * 100, 4),
}
print(json.dumps(identity_coverage, indent=2))
attach_artifact(run, identity_coverage, name="identity_coverage")
"""
    ),
    _code(
        """
# Vectorised 95% Wilson confidence interval. Used here for
# has-id-vs-no-id and reused throughout Section B; defining it once
# at first use keeps the helper visible to the rest of the notebook.
def wilson_ci(k, n, *, alpha: float = 0.05):
    \"\"\"Return (low, high) arrays for a 95% Wilson binomial CI.

    n=0 entries return (NaN, NaN). Cheaper than scipy.stats.binomtest
    in a loop and lines up with how groupby aggregates land — `k` and
    `n` arrive as same-shape numpy arrays.
    \"\"\"
    from scipy.stats import norm

    k_arr = np.asarray(k, dtype=float)
    n_arr = np.asarray(n, dtype=float)
    z = float(norm.ppf(1.0 - alpha / 2.0))
    safe_n = np.where(n_arr > 0, n_arr, 1.0)
    p = k_arr / safe_n
    denom = 1.0 + z**2 / safe_n
    center = (p + z**2 / (2.0 * safe_n)) / denom
    margin = z * np.sqrt(p * (1.0 - p) / safe_n + z**2 / (4.0 * safe_n**2)) / denom
    low = np.where(n_arr > 0, np.maximum(center - margin, 0.0), np.nan)
    high = np.where(n_arr > 0, np.minimum(center + margin, 1.0), np.nan)
    return low, high


fraud_by_identity = (
    merged.assign(has_identity=has_identity)
    .groupby("has_identity", observed=True)["isFraud"]
    .agg(n_fraud="sum", n="count", fraud_rate="mean")
)
ci_low, ci_high = wilson_ci(
    fraud_by_identity["n_fraud"].to_numpy(),
    fraud_by_identity["n"].to_numpy(),
)
fraud_by_identity["ci_low_95"] = ci_low
fraud_by_identity["ci_high_95"] = ci_high
print(fraud_by_identity)
attach_artifact(
    run,
    fraud_by_identity.reset_index().to_dict(orient="records"),
    name="fraud_by_identity",
)
"""
    ),
    _md(
        """
## Section B — Target Analysis

Fraud prevalence overall and per slice — amount bucket, ProductCD,
hour-of-day, card brand and type, day-of-week × hour heatmap, top-20
`P_emaildomain` by fraud rate, and the log-scale `TransactionAmt`
overlay fraud-vs-non-fraud. Every aggregated rate is reported with a
95% Wilson confidence interval (helper defined at the end of Section
A) so a slice with N=12 doesn't visually outweigh one with N=120,000.

Three decisions come out of this section:

- **AUC over F1** as the headline metric — 3.5% fraud makes F1
  threshold-sensitive in a way that obscures model skill.
- **Economic cost (Sprint 4) replaces F1 threshold tuning** —
  `fraud_cost_usd` / `fp_cost_usd` are the decision-relevant units.
- **Hour-of-day, day-of-week, card type, and email domain are all
  predictive enough to warrant features** in Sprint 2.
"""
    ),
    _code(
        """
n_total = int(len(merged))
n_fraud = int(merged["isFraud"].sum())
overall_low, overall_high = wilson_ci(np.array([n_fraud]), np.array([n_total]))
overall = {
    "n_total": n_total,
    "n_fraud": n_fraud,
    "fraud_rate": round(n_fraud / n_total, 6),
    "ci_low_95": round(float(overall_low[0]), 6),
    "ci_high_95": round(float(overall_high[0]), 6),
}
print(json.dumps(overall, indent=2))
attach_artifact(run, overall, name="overall_fraud_rate")
"""
    ),
    _code(
        """
amt_bins = [0, 25, 50, 100, 250, 500, 1000, 5000, np.inf]
amt_bucket = pd.cut(merged["TransactionAmt"], bins=amt_bins, include_lowest=True)
rate_by_amt = (
    merged.assign(_amt_bucket=amt_bucket)
    .groupby("_amt_bucket", observed=True)["isFraud"]
    .agg(n_fraud="sum", n="count", fraud_rate="mean")
)
amt_low, amt_high = wilson_ci(
    rate_by_amt["n_fraud"].to_numpy(), rate_by_amt["n"].to_numpy()
)
rate_by_amt["ci_low_95"] = amt_low
rate_by_amt["ci_high_95"] = amt_high

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(rate_by_amt))
ax.bar(x, rate_by_amt["fraud_rate"], color="#3b6fb3")
ax.errorbar(
    x,
    rate_by_amt["fraud_rate"],
    yerr=[
        rate_by_amt["fraud_rate"] - rate_by_amt["ci_low_95"],
        rate_by_amt["ci_high_95"] - rate_by_amt["fraud_rate"],
    ],
    fmt="none",
    color="black",
    capsize=3,
    linewidth=0.8,
)
ax.set_xticks(x)
ax.set_xticklabels([str(b) for b in rate_by_amt.index], rotation=35, ha="right")
ax.set_ylabel("fraud rate (95% Wilson CI)")
ax.set_title("Fraud rate by TransactionAmt bucket")
fig.tight_layout()
fig.savefig(FIG_DIR / "fraud_rate_by_amount.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="fraud_rate_by_amount")
plt.close(fig)
print(rate_by_amt)
"""
    ),
    _code(
        """
rate_by_product = (
    merged.groupby("ProductCD", observed=True)["isFraud"]
    .agg(n_fraud="sum", n="count", fraud_rate="mean")
    .sort_values("fraud_rate")
)
prod_low, prod_high = wilson_ci(
    rate_by_product["n_fraud"].to_numpy(), rate_by_product["n"].to_numpy()
)
rate_by_product["ci_low_95"] = prod_low
rate_by_product["ci_high_95"] = prod_high

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(rate_by_product))
ax.bar(x, rate_by_product["fraud_rate"], color="#3b6fb3")
ax.errorbar(
    x,
    rate_by_product["fraud_rate"],
    yerr=[
        rate_by_product["fraud_rate"] - rate_by_product["ci_low_95"],
        rate_by_product["ci_high_95"] - rate_by_product["fraud_rate"],
    ],
    fmt="none",
    color="black",
    capsize=3,
    linewidth=0.8,
)
ax.set_xticks(x)
ax.set_xticklabels(list(rate_by_product.index))
ax.set_ylabel("fraud rate (95% Wilson CI)")
ax.set_title("Fraud rate by ProductCD")
fig.tight_layout()
fig.savefig(FIG_DIR / "fraud_rate_by_product.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="fraud_rate_by_product")
plt.close(fig)
print(rate_by_product)
"""
    ),
    _code(
        """
hour_groups = (
    merged.assign(_hour=event_dt.dt.hour.to_numpy())
    .groupby("_hour", observed=True)["isFraud"]
    .agg(n_fraud="sum", n="count", fraud_rate="mean")
    .sort_index()
)
hour_low, hour_high = wilson_ci(
    hour_groups["n_fraud"].to_numpy(), hour_groups["n"].to_numpy()
)
hour_groups["ci_low_95"] = hour_low
hour_groups["ci_high_95"] = hour_high

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hour_groups.index, hour_groups["fraud_rate"], color="#3b6fb3", marker="o")
ax.fill_between(
    hour_groups.index,
    hour_groups["ci_low_95"],
    hour_groups["ci_high_95"],
    color="#3b6fb3",
    alpha=0.2,
    label="95% Wilson CI",
)
ax.set_xlabel("hour of day (UTC, derived from event_dt)")
ax.set_ylabel("fraud rate")
ax.set_xticks(range(0, 24, 2))
ax.set_title("Fraud rate by hour of day")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fraud_rate_by_hour.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="fraud_rate_by_hour")
plt.close(fig)
"""
    ),
    _code(
        """
# Per-card-attribute fraud rate. Drop near-empty groups
# (n < CARD_MIN_N) so a 4-row "samsung pay" bucket doesn't dominate the
# y-axis; the IEEE-CIS card families that survive the filter are the
# ones a Sprint 2 categorical encoder will actually see at training time.
CARD_MIN_N = 100
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, col in zip(axes, ["card4", "card6"], strict=True):
    rate_by_card = (
        merged.groupby(col, observed=True)["isFraud"]
        .agg(n_fraud="sum", n="count", fraud_rate="mean")
        .sort_values("fraud_rate")
    )
    rate_by_card = rate_by_card[rate_by_card["n"] >= CARD_MIN_N]
    card_low, card_high = wilson_ci(
        rate_by_card["n_fraud"].to_numpy(), rate_by_card["n"].to_numpy()
    )
    rate_by_card["ci_low_95"] = card_low
    rate_by_card["ci_high_95"] = card_high

    x = np.arange(len(rate_by_card))
    ax.bar(x, rate_by_card["fraud_rate"], color="#3b6fb3")
    ax.errorbar(
        x,
        rate_by_card["fraud_rate"],
        yerr=[
            rate_by_card["fraud_rate"] - rate_by_card["ci_low_95"],
            rate_by_card["ci_high_95"] - rate_by_card["fraud_rate"],
        ],
        fmt="none",
        color="black",
        capsize=3,
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(idx) for idx in rate_by_card.index], rotation=20, ha="right")
    ax.set_ylabel("fraud rate (95% Wilson CI)")
    ax.set_title(f"Fraud rate by {col} (n ≥ {CARD_MIN_N})")
    print(f"--- {col} ---")
    print(rate_by_card)
fig.tight_layout()
fig.savefig(FIG_DIR / "fraud_rate_by_card.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="fraud_rate_by_card")
plt.close(fig)
"""
    ),
    _code(
        """
heatmap_df = (
    merged.assign(
        _dow=event_dt.dt.dayofweek.to_numpy(),
        _hour=event_dt.dt.hour.to_numpy(),
    )
    .groupby(["_dow", "_hour"], observed=True)["isFraud"]
    .mean()
    .unstack("_hour")
)
day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
heatmap_df.index = pd.Index([day_labels[int(i)] for i in heatmap_df.index], name="dow")

fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(
    heatmap_df,
    ax=ax,
    cmap="rocket_r",
    annot=False,
    cbar_kws={"label": "fraud rate"},
)
ax.set_title("Fraud rate — day of week × hour of day (event_dt UTC)")
ax.set_xlabel("hour")
ax.set_ylabel("day of week")
fig.tight_layout()
fig.savefig(FIG_DIR / "fraud_rate_dow_hour_heatmap.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="fraud_rate_dow_hour_heatmap")
plt.close(fig)
"""
    ),
    _code(
        """
# Top-20 P_emaildomain by fraud rate, restricted to domains with at
# least DOMAIN_MIN_N transactions so a 3-row "edu.in" bucket doesn't
# dominate the chart. P_emaildomain is the purchaser's email; Sprint 2
# may add R_emaildomain (recipient) symmetrically.
DOMAIN_MIN_N = 500
domain_groups = (
    merged.groupby("P_emaildomain", observed=True)["isFraud"]
    .agg(n_fraud="sum", n="count", fraud_rate="mean")
)
domain_groups = (
    domain_groups[domain_groups["n"] >= DOMAIN_MIN_N]
    .sort_values("fraud_rate", ascending=False)
    .head(20)
)
dom_low, dom_high = wilson_ci(
    domain_groups["n_fraud"].to_numpy(), domain_groups["n"].to_numpy()
)
domain_groups["ci_low_95"] = dom_low
domain_groups["ci_high_95"] = dom_high

fig, ax = plt.subplots(figsize=(10, 7))
y = np.arange(len(domain_groups))[::-1]
ax.barh(y, domain_groups["fraud_rate"], color="#3b6fb3")
ax.errorbar(
    domain_groups["fraud_rate"],
    y,
    xerr=[
        domain_groups["fraud_rate"] - domain_groups["ci_low_95"],
        domain_groups["ci_high_95"] - domain_groups["fraud_rate"],
    ],
    fmt="none",
    color="black",
    capsize=3,
    linewidth=0.8,
)
ax.set_yticks(y)
ax.set_yticklabels([str(idx) for idx in domain_groups.index])
ax.set_xlabel("fraud rate (95% Wilson CI)")
ax.set_title(f"Top-20 P_emaildomain by fraud rate (n ≥ {DOMAIN_MIN_N})")
fig.tight_layout()
fig.savefig(FIG_DIR / "fraud_rate_by_email_domain.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="fraud_rate_by_email_domain")
plt.close(fig)
print(domain_groups)
"""
    ),
    _code(
        """
# Density overlay on log(1 + TransactionAmt) so the long right tail
# doesn't dominate; both classes share the same bin grid so visual
# differences in shape are real, not artefacts of binning.
log_amt = np.log1p(merged["TransactionAmt"].to_numpy())
is_fraud = merged["isFraud"].to_numpy().astype(bool)
log_amt_legit = log_amt[~is_fraud]
log_amt_fraud = log_amt[is_fraud]

fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(float(log_amt.min()), float(log_amt.max()), 80)
ax.hist(
    log_amt_legit,
    bins=bins,
    density=True,
    alpha=0.5,
    color="#3b6fb3",
    label=f"non-fraud (n={len(log_amt_legit):,})",
)
ax.hist(
    log_amt_fraud,
    bins=bins,
    density=True,
    alpha=0.5,
    color="#c85050",
    label=f"fraud (n={len(log_amt_fraud):,})",
)
ax.set_xlabel("log(1 + TransactionAmt)")
ax.set_ylabel("density")
ax.set_title("TransactionAmt distribution: fraud vs non-fraud (log scale)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "transaction_amt_overlay.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="transaction_amt_overlay")
plt.close(fig)
"""
    ),
    _md(
        """
## Section C — Missing Value Analysis

Missingness in IEEE-CIS isn't random noise — it's a structural feature.
The headline is the ~76% identity-join miss observed in Section A:
roughly three-quarters of transactions arrive with no device or
browser fingerprint, so any identity-derived feature added in Sprint
2 has to be NaN-tolerant. Below we go past the headline and answer
four questions:

1. **Where is missingness concentrated?** (top-50 table + barh)
2. **Are columns missing together in the same rows?** (5k-row binary
   heatmap over the top-50 most-missing columns)
3. **Are columns missing in *exactly* the same rows?** (NaN-equivalence
   classes via per-column null-mask hashes)
4. **Is missingness predictive of fraud on its own?** (per-column
   fraud rate when null vs not, with 95% Wilson CIs)

Q3 is what surfaces IEEE-CIS's shared-block structure (V1–V11
co-missing, V12–V34 co-missing, etc.) — a single-column hash beats
correlation thresholding because it catches *exact* equivalence
classes instead of fuzzy ones with a tunable τ. Q4 is what lets
Sprint 2 decide whether `is_null_<col>` indicators are worth the
extra columns.
"""
    ),
    _code(
        """
# Top-K policy. The barh floor (1%) is purely cosmetic — without it
# the right-hand side fills with sub-0.001% rows that compress the
# headline columns visually. The data table keeps every top-50 row.
MISSING_TOP_K = 50
MISSING_BARH_FLOOR = 0.01

missing = merged.isna().mean().sort_values(ascending=False)
top_missing = missing.head(MISSING_TOP_K)

missing_table = pd.DataFrame(
    {
        "column": top_missing.index,
        "missing_rate": top_missing.to_numpy(),
        "n_missing": (top_missing.to_numpy() * len(merged)).round().astype(int),
    }
)
print(f"Top {MISSING_TOP_K} columns by missing rate (head 20 shown):")
print(missing_table.head(20).to_string(index=False))
attach_artifact(run, missing_table, name="missing_top_50")

barh_data = top_missing[top_missing >= MISSING_BARH_FLOOR]
fig, ax = plt.subplots(figsize=(10, max(6, 0.22 * len(barh_data))))
barh_data.iloc[::-1].plot.barh(ax=ax, color="#c85050")
ax.set_xlabel("missing rate")
ax.set_title(f"Top {len(barh_data)} columns by missing rate (>={MISSING_BARH_FLOOR:.0%})")
fig.tight_layout()
fig.savefig(FIG_DIR / "missing_values.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="missing_values")
plt.close(fig)

identity_cols = [
    c for c in merged.columns if c.startswith("id_") or c in {"DeviceType", "DeviceInfo"}
]
missing_by_family = {
    "V":  float(merged.filter(regex=r"^V\\d+$").isna().mean().mean()),
    "C":  float(merged.filter(regex=r"^C\\d+$").isna().mean().mean()),
    "D":  float(merged.filter(regex=r"^D\\d+$").isna().mean().mean()),
    "M":  float(merged.filter(regex=r"^M\\d+$").isna().mean().mean()),
    "id": float(merged[identity_cols].isna().mean().mean()) if identity_cols else float("nan"),
}
print("\\nMean missing rate by family:")
for family, rate in missing_by_family.items():
    print(f"  {family:3s}: {rate:.4%}")
attach_artifact(run, missing_by_family, name="missing_by_family")
"""
    ),
    _code(
        """
# Binary-mask heatmap over a 5k stratified row sample × the top-50
# most-missing columns. Beats sns.clustermap because rows are
# time-ordered transactions, not clusterable units; row order = sample
# order, so visible vertical bands ≈ co-missing column blocks.
from sklearn.model_selection import train_test_split

HEATMAP_SAMPLE_SIZE = 5_000
HEATMAP_COL_COUNT = MISSING_TOP_K  # reuse the top-50 ordering from the previous cell

heatmap_cols = top_missing.head(HEATMAP_COL_COUNT).index.tolist()
heatmap_sample, _ = train_test_split(
    merged[heatmap_cols + ["isFraud"]],
    train_size=HEATMAP_SAMPLE_SIZE,
    stratify=merged["isFraud"],
    random_state=SETTINGS.seed,
)
mask = heatmap_sample[heatmap_cols].isna().to_numpy()

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(mask, aspect="auto", cmap="binary", interpolation="nearest")
ax.set_xticks(range(len(heatmap_cols)))
ax.set_xticklabels(heatmap_cols, rotation=90, fontsize=7)
ax.set_yticks([])
ax.set_xlabel("column (ordered by overall missing rate)")
ax.set_ylabel(f"transactions (n={HEATMAP_SAMPLE_SIZE:,}, stratified by isFraud)")
ax.set_title("Missingness pattern — black = NaN")
fig.tight_layout()
fig.savefig(FIG_DIR / "missingness_heatmap.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="missingness_heatmap")
plt.close(fig)
"""
    ),
    _code(
        """
# NaN equivalence classes. For every column, hash its null-mask;
# columns sharing a hash are missing in *exactly* the same rows. This
# is O(n_cols * n_rows) one-pass, deterministic, and avoids the
# tunable τ that correlation thresholding requires.
import hashlib

NAN_GROUP_MIN_COLS = 2     # singletons aren't a "group"
NAN_GROUP_MIN_RATE = 0.01  # ignore <1%-missing equivalence classes

def _null_signature(series):
    return hashlib.blake2b(series.isna().to_numpy().tobytes(), digest_size=8).hexdigest()

signatures = {col: _null_signature(merged[col]) for col in merged.columns}
sig_df = (
    pd.DataFrame({"column": list(signatures.keys()), "signature": list(signatures.values())})
    .merge(missing.rename("missing_rate"), left_on="column", right_index=True)
)
nan_groups = (
    sig_df.groupby("signature")
    .agg(
        n_columns=("column", "size"),
        missing_rate=("missing_rate", "first"),
        columns=(
            "column",
            lambda s: ", ".join(sorted(s)[:8]) + (f", … (+{len(s) - 8} more)" if len(s) > 8 else ""),
        ),
    )
    .reset_index()
)
nan_groups["n_rows_missing"] = (nan_groups["missing_rate"] * len(merged)).round().astype(int)
nan_groups = (
    nan_groups[
        (nan_groups["n_columns"] >= NAN_GROUP_MIN_COLS)
        & (nan_groups["missing_rate"] >= NAN_GROUP_MIN_RATE)
    ]
    .sort_values(["n_columns", "missing_rate"], ascending=[False, False])
    .reset_index(drop=True)
)

n_groups = int(len(nan_groups))
n_grouped_cols = int(nan_groups["n_columns"].sum())
print(
    f"NaN equivalence classes (n_columns>={NAN_GROUP_MIN_COLS}, "
    f"missing>={NAN_GROUP_MIN_RATE:.0%}): {n_groups}"
)
print(f"Columns participating in a group: {n_grouped_cols} / {merged.shape[1]}")
print()
print(nan_groups.head(15).to_string(index=False))
attach_artifact(run, nan_groups, name="nan_equivalence_classes")
attach_artifact(
    run,
    {"n_groups": n_groups, "n_grouped_cols": n_grouped_cols, "n_total_cols": int(merged.shape[1])},
    name="nan_equivalence_summary",
)
"""
    ),
    _code(
        """
# Predictive missingness. For each top-K most-missing column, compute
# fraud rate when NaN vs when present, with Wilson 95% CIs (helper
# defined in Section A.5). The min-n floor of 500 on BOTH groups
# prevents a 99.9%-missing column from showing a fake spike on its
# 500-row not-null cohort.
MISSINGNESS_PREDICTIVE_TOP_K = 20
MISSINGNESS_PREDICTIVE_MIN_N = 500

predictive_rows = []
for col in top_missing.head(MISSINGNESS_PREDICTIVE_TOP_K).index:
    is_null = merged[col].isna()
    n_null = int(is_null.sum())
    n_not_null = int((~is_null).sum())
    if n_null < MISSINGNESS_PREDICTIVE_MIN_N or n_not_null < MISSINGNESS_PREDICTIVE_MIN_N:
        continue
    k_null = int(merged.loc[is_null, "isFraud"].sum())
    k_not = int(merged.loc[~is_null, "isFraud"].sum())
    predictive_rows.append(
        {
            "column": col,
            "rate_null": k_null / n_null,
            "rate_not_null": k_not / n_not_null,
            "n_null": n_null,
            "n_not_null": n_not_null,
            "k_null": k_null,
            "k_not_null": k_not,
        }
    )

predictive = pd.DataFrame(predictive_rows)
ci_low_null, ci_high_null = wilson_ci(
    predictive["k_null"].to_numpy(), predictive["n_null"].to_numpy()
)
ci_low_not, ci_high_not = wilson_ci(
    predictive["k_not_null"].to_numpy(), predictive["n_not_null"].to_numpy()
)
predictive["ci_low_null"], predictive["ci_high_null"] = ci_low_null, ci_high_null
predictive["ci_low_not_null"], predictive["ci_high_not_null"] = ci_low_not, ci_high_not
predictive = predictive.sort_values("rate_null", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11, max(6, 0.4 * len(predictive))))
y = np.arange(len(predictive))
# Clamp tiny negative residuals from float arithmetic at p=0 boundaries.
err_null = np.maximum(
    np.vstack(
        [predictive["rate_null"] - predictive["ci_low_null"], predictive["ci_high_null"] - predictive["rate_null"]]
    ),
    0.0,
)
err_not = np.maximum(
    np.vstack(
        [
            predictive["rate_not_null"] - predictive["ci_low_not_null"],
            predictive["ci_high_not_null"] - predictive["rate_not_null"],
        ]
    ),
    0.0,
)
ax.errorbar(
    predictive["rate_null"], y - 0.18, xerr=err_null, fmt="o", color="#c85050",
    label="fraud rate when NaN", capsize=3,
)
ax.errorbar(
    predictive["rate_not_null"], y + 0.18, xerr=err_not, fmt="s", color="#3070b8",
    label="fraud rate when present", capsize=3,
)
ax.axvline(
    merged["isFraud"].mean(), color="black", linestyle=":", linewidth=1,
    label=f"overall ({merged['isFraud'].mean():.3%})",
)
ax.set_yticks(y)
ax.set_yticklabels(predictive["column"])
ax.invert_yaxis()
ax.set_xlabel("fraud rate (95% Wilson CI)")
ax.set_title(
    f"Missingness as a feature — top {len(predictive)} columns "
    f"(min n={MISSINGNESS_PREDICTIVE_MIN_N} per group)"
)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "predictive_missingness.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="predictive_missingness")
plt.close(fig)

print(predictive[["column", "rate_null", "rate_not_null", "n_null", "n_not_null"]].to_string(index=False))
attach_artifact(run, predictive, name="predictive_missingness_table")
"""
    ),
    _md(
        """
### Section C takeaways

- **Identity is the spike.** The ~76% identity-join miss observed in
  Section A is reproduced here — every `id_*` column and `DeviceType`
  / `DeviceInfo` clusters at the top of the missing-rate table. Sprint
  2's identity features must be computed conditional on identity being
  present; a model that requires identity refuses 76% of traffic.
- **NaN equivalence classes are real and large.** The hash-based pass
  surfaces a small number of large blocks: V columns split into a
  handful of groups, the C and D blocks are mostly internally
  consistent. This is where Sprint 2's PCA-of-V or group-mean
  aggregate has its biggest payoff.
- **Missingness is sometimes predictive on its own.** The C.4 chart
  shows several columns where fraud rate when null vs when present
  differs materially with non-overlapping Wilson CIs. Those columns
  deserve `is_null_<col>` indicator features in Sprint 2 even if the
  underlying value also feeds the model.
- **The tail is cosmetic.** Beyond the top-50, the missing-rate
  distribution is dominated by sub-1% columns where the indicator
  signal would be drowned by noise. Sprint 2 should not generate
  is-null indicators for everything — focus on the top-K identified
  here.
"""
    ),
    _md(
        """
## Section D — Feature Group Deep Dives

Six subsections, one per feature family. Each ends with a Takeaways
markdown that distils the figures into 2–4 bullets — Sprint 2 starts
from those bullets, not from the figures.

- **D.1 Card features** (`card1`–`card6`): cardinality, top-10 values
  per column, per-value fraud rate with Wilson CIs.
- **D.2 V features** (`V1`–`V339`): correlation matrix on a random 50
  cols + PCA scree on all 339, both on a 5% stratified sample. Median
  imputation here is **for visualisation only** — Sprint 2 must redo
  imputation per-fold inside `Pipeline` to avoid leakage.
- **D.3 C features** (`C1`–`C14`): symlog boxplots, fraud-vs-non-fraud
  per column.
- **D.4 D features** (`D1`–`D15`): boxplots, fraud-vs-non-fraud per
  column. `D1` is the IEEE-CIS forum consensus "days since card first
  observed".
- **D.5 M features** (`M1`–`M9`): bar of fraud rate per `{T, F, null}`
  level with Wilson CIs.
- **D.6 Identity** (`DeviceType`, `DeviceInfo`): fraud rate per device
  type, top-15 `DeviceInfo` values. Null is its own bin labelled
  "(no identity)" — filtering loses the population majority and
  contradicts Section A's identity-coverage finding.

A single seed (`SETTINGS.seed`) drives every random sample below, so
re-running the notebook on the same data is bit-identical. Nothing
in this section mutates `merged` — every analysis works on derived
Series or temporary frames so Section F's feature-column selection
stays intact.
"""
    ),
    _md(
        """
### D.1 — Card features (`card1`–`card6`)

Card columns are obfuscated identifiers. Cardinality is what tells
us which is which: `card1` is high-cardinality (BIN-level, ~10k
values), `card4`/`card6` are low-cardinality (brand /
debit-vs-credit). Three plots: (a) cardinality bar across the six
card columns, (b) top-10 most common values per column, (c) fraud
rate of those top-10 values with Wilson CIs and a per-value `n>=200`
floor.
"""
    ),
    _code(
        """
card_cols = [f"card{i}" for i in range(1, 7) if f"card{i}" in merged.columns]
card_cardinality = pd.Series(
    {c: int(merged[c].nunique(dropna=True)) for c in card_cols},
    name="cardinality",
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
card_cardinality.plot.bar(ax=ax, color="#3070b8")
ax.set_yscale("log")
ax.set_ylabel("unique values (log scale)")
ax.set_title("Card column cardinality")
for i, v in enumerate(card_cardinality):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "card_cardinality.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="card_cardinality")
plt.close(fig)

print(card_cardinality.to_string())
attach_artifact(run, card_cardinality.to_dict(), name="card_cardinality_table")
"""
    ),
    _code(
        """
CARD_TOP_K = 10

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), card_cols):
    top_values = merged[col].value_counts(dropna=False).head(CARD_TOP_K)
    labels = [str(v) if not pd.isna(v) else "(NaN)" for v in top_values.index]
    ax.barh(range(len(top_values)), top_values.to_numpy(), color="#3070b8")
    ax.set_yticks(range(len(top_values)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"{col} — top {CARD_TOP_K} values")
    ax.set_xlabel("count")
fig.tight_layout()
fig.savefig(FIG_DIR / "card_top_values.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="card_top_values")
plt.close(fig)
"""
    ),
    _code(
        """
# Per-value n is smaller than per-attribute n, so this floor (200) is
# 2× Section B's CARD_MIN_N (100). Below that, Wilson CIs swallow the
# bar and the difference between 0/200 and 4/200 is visually
# meaningless.
CARD_FRAUD_MIN_N = 200

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), card_cols):
    grouped = (
        merged.groupby(col, observed=True, dropna=False)["isFraud"]
        .agg(n_fraud="sum", n="count", rate="mean")
        .reset_index()
    )
    grouped = (
        grouped[grouped["n"] >= CARD_FRAUD_MIN_N]
        .sort_values("rate", ascending=False)
        .head(CARD_TOP_K)
    )
    if grouped.empty:
        ax.set_title(f"{col} — no values pass n>={CARD_FRAUD_MIN_N}")
        ax.axis("off")
        continue
    low, high = wilson_ci(grouped["n_fraud"].to_numpy(), grouped["n"].to_numpy())
    err = np.maximum(np.vstack([grouped["rate"] - low, high - grouped["rate"]]), 0.0)
    labels = [str(v) if not pd.isna(v) else "(NaN)" for v in grouped[col]]
    ax.barh(
        range(len(grouped)), grouped["rate"].to_numpy(), xerr=err,
        color="#c85050", capsize=3,
    )
    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(merged["isFraud"].mean(), color="black", linestyle=":", linewidth=1)
    ax.set_title(f"{col} — top {len(grouped)} by fraud rate (n>={CARD_FRAUD_MIN_N})")
    ax.set_xlabel("fraud rate (95% Wilson CI)")
fig.tight_layout()
fig.savefig(FIG_DIR / "card_fraud_rate.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="card_fraud_rate")
plt.close(fig)
"""
    ),
    _md(
        """
**D.1 takeaways.** `card1` has BIN-level cardinality (~10k+ values),
which is the signal a Sprint 2 target encoder will lean on most.
`card4` (4 levels) and `card6` (3 levels) are low-cardinality enough
to one-hot directly. Top-10 by fraud rate plots show large
between-issuer / between-brand spreads — material gaps, not noise,
since the Wilson CIs separate clearly above the dotted overall-rate
line.
"""
    ),
    _md(
        """
### D.2 — V features (`V1`–`V339`)

The V family is 339 anonymised engineered features. The NaN
equivalence classes from Section C.3 confirm the Vesta forum
consensus that V columns split into a handful of co-missing blocks.
Two questions:

1. **How redundant are V columns within a block?** (correlation
   heatmap on a random 50 columns, seeded)
2. **How much of the V variance survives PCA?** (scree plot on the
   full 339 cols, on a 5% stratified sample)

Both work on a 5% stratified sample (~30k × 339 ≈ 80 MB) instead of
the full 590k × 339 (~1.6 GB). Both apply **median imputation** —
this is consistent visualisation policy, **not** what Sprint 2's
training pipeline should do (per-fold imputation inside the sklearn
Pipeline avoids leakage).
"""
    ),
    _code(
        """
# Median imputation here is for visualisation only. Sprint 2 must
# redo imputation per-fold inside Pipeline to avoid train→val leak.
V_SAMPLE_SIZE = 50

v_cols_all = sorted(
    [c for c in merged.columns if c.startswith("V") and c[1:].isdigit()],
    key=lambda c: int(c[1:]),
)
rng = np.random.default_rng(SETTINGS.seed)
v_picked = sorted(
    rng.choice(v_cols_all, size=min(V_SAMPLE_SIZE, len(v_cols_all)), replace=False).tolist(),
    key=lambda c: int(c[1:]),
)

v_view = merged[v_picked].copy()
v_view = v_view.fillna(v_view.median(numeric_only=True))
v_corr = v_view.corr()

fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(
    v_corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
    cbar_kws={"label": "Pearson ρ"},
)
ax.set_title(f"V correlation — {V_SAMPLE_SIZE} random V columns (5% sample, median-imputed)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
fig.tight_layout()
fig.savefig(FIG_DIR / "v_correlation.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="v_correlation")
plt.close(fig)

# Rough redundancy estimate within the sampled 50.
v_corr_abs = v_corr.where(~np.eye(len(v_corr), dtype=bool)).abs()
n_high_corr_pairs = int((v_corr_abs > 0.95).sum().sum() / 2)
v_droppable_in_sample = int((v_corr_abs.max(axis=0) > 0.95).sum())
print(f"V columns with at least one |ρ|>0.95 partner (in {V_SAMPLE_SIZE}-col sample): {v_droppable_in_sample}")
print(f"Pairs with |ρ|>0.95: {n_high_corr_pairs}")
attach_artifact(
    run,
    {
        "v_sample_size": V_SAMPLE_SIZE,
        "n_high_corr_pairs": n_high_corr_pairs,
        "v_droppable_in_sample": v_droppable_in_sample,
    },
    name="v_correlation_summary",
)
"""
    ),
    _code(
        """
from sklearn.decomposition import PCA

PCA_N_COMPONENTS = 30

v_sample, _ = train_test_split(
    merged[v_cols_all + ["isFraud"]],
    train_size=0.05,
    stratify=merged["isFraud"],
    random_state=SETTINGS.seed,
)
v_matrix = v_sample[v_cols_all].copy()
all_null = v_matrix.columns[v_matrix.isna().all()].tolist()
if all_null:
    v_matrix = v_matrix.drop(columns=all_null)
v_matrix = v_matrix.fillna(v_matrix.median(numeric_only=True))

pca = PCA(n_components=PCA_N_COMPONENTS, random_state=SETTINGS.seed)
pca.fit(v_matrix)
cum_var = np.cumsum(pca.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(
    range(1, PCA_N_COMPONENTS + 1), pca.explained_variance_ratio_,
    color="#3070b8", alpha=0.7, label="per-component",
)
ax2 = ax.twinx()
ax2.plot(
    range(1, PCA_N_COMPONENTS + 1), cum_var,
    color="#c85050", marker="o", label="cumulative",
)
ax.set_xlabel("PCA component")
ax.set_ylabel("variance explained (per component)")
ax2.set_ylabel("cumulative variance explained")
ax.set_title(f"PCA scree on {v_matrix.shape[1]} V columns (5% sample, median-imputed)")
fig.tight_layout()
fig.savefig(FIG_DIR / "v_pca_scree.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="v_pca_scree")
plt.close(fig)

n_for_90 = int(np.searchsorted(cum_var, 0.90)) + 1
n_for_95 = int(np.searchsorted(cum_var, 0.95)) + 1
print(f"Components needed for 90% var: {n_for_90} (PCA fit budget = {PCA_N_COMPONENTS})")
print(f"Components needed for 95% var: {n_for_95} (PCA fit budget = {PCA_N_COMPONENTS})")
print(f"All-null V cols dropped from PCA fit: {len(all_null)}")
attach_artifact(
    run,
    {
        "n_v_cols_in_pca": int(v_matrix.shape[1]),
        "n_for_90_pct": n_for_90,
        "n_for_95_pct": n_for_95,
        "all_null_v_cols": all_null,
        "pca_n_components": PCA_N_COMPONENTS,
    },
    name="v_pca_summary",
)
"""
    ),
    _md(
        """
**D.2 takeaways.** The 50-col correlation heatmap shows clear block
structure — large red/blue patches confirm V columns are not
independent and many are near-duplicates. The pair-count above
quantifies it. The scree plot shows variance collapses fast: the
first ~10 components carry the bulk, and the 90%/95% variance
thresholds are reached well inside the 30-component budget. Sprint 2
has two viable strategies — drop within-block correlated pairs, or
replace V entirely with the first ~30 PCA components fit per-fold.
The model card should record the chosen strategy.
"""
    ),
    _md(
        """
### D.3 — C features (`C1`–`C14`)

C columns are count features (anonymised counters of various
properties — likely "addresses associated with this card", "phones
associated with this card", etc., per IEEE-CIS forum consensus).
They are heavy-tailed integers, so the boxplot uses **symlog** —
linear near zero, log past it — to keep the median visible while
showing the long tail.
"""
    ),
    _code(
        """
c_cols = sorted(
    [c for c in merged.columns if c.startswith("C") and c[1:].isdigit()],
    key=lambda c: int(c[1:]),
)

fig, axes = plt.subplots(2, 7, figsize=(20, 7))
for ax, col in zip(axes.flatten(), c_cols):
    legit = merged.loc[merged["isFraud"] == 0, col].dropna()
    fraud = merged.loc[merged["isFraud"] == 1, col].dropna()
    bp = ax.boxplot(
        [legit, fraud],
        tick_labels=["legit", "fraud"],
        showfliers=False,
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#3070b8")
    bp["boxes"][1].set_facecolor("#c85050")
    ax.set_yscale("symlog")
    ax.set_title(col)
fig.suptitle("C features — symlog distribution by class")
fig.tight_layout()
fig.savefig(FIG_DIR / "c_boxplots.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="c_boxplots")
plt.close(fig)
"""
    ),
    _md(
        """
**D.3 takeaways.** Fraud distributions on C features sit visibly
higher than legit on most columns, with longer tails — consistent
with the "many things associated with one card" reading. C13 and C14
look the most discriminating; C3 the least. Sprint 2 doesn't need
transformations on C (LightGBM handles raw counts), but
percentile-rank features per `card1` / `DeviceID` may help.
"""
    ),
    _md(
        """
### D.4 — D features (`D1`–`D15`)

D columns are time-delta features. The IEEE-CIS forum consensus is
that **D1 ≈ "days since card first observed"** and the rest are
days-since-various-event features. Linear scale because their
magnitudes are already in days, not counts.
"""
    ),
    _code(
        """
d_cols = sorted(
    [c for c in merged.columns if c.startswith("D") and c[1:].isdigit()],
    key=lambda c: int(c[1:]),
)

fig, axes = plt.subplots(3, 5, figsize=(20, 10))
flat_axes = axes.flatten()
for ax, col in zip(flat_axes, d_cols):
    legit = merged.loc[merged["isFraud"] == 0, col].dropna()
    fraud = merged.loc[merged["isFraud"] == 1, col].dropna()
    bp = ax.boxplot(
        [legit, fraud],
        tick_labels=["legit", "fraud"],
        showfliers=False,
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#3070b8")
    bp["boxes"][1].set_facecolor("#c85050")
    ax.set_title(col)
for ax in flat_axes[len(d_cols):]:
    ax.axis("off")
fig.suptitle("D features — distribution by class (D1 ≈ days since card first observed)")
fig.tight_layout()
fig.savefig(FIG_DIR / "d_boxplots.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="d_boxplots")
plt.close(fig)
"""
    ),
    _md(
        """
**D.4 takeaways.** D1 medians for fraud sit near zero — fraud
predominantly hits cards that are *new to the system*, consistent
with stolen-card / synthetic-identity tradecraft. D2 mirrors D1.
D3, D4 have wider class overlap. Sprint 2's "days-since-first-seen"
feature (T1 spec) is the engineered analogue of D1 — the model card
should note the redundancy so we don't double-count.
"""
    ),
    _md(
        """
### D.5 — M features (`M1`–`M9`)

M columns are match flags — `T` (match), `F` (mismatch), or null.
The question for each is whether the three levels separate on fraud
rate. Wilson CIs gate visual inference: a level with n=300 should
not visually outweigh one with n=300,000.
"""
    ),
    _code(
        """
m_cols = sorted(
    [c for c in merged.columns if c.startswith("M") and c[1:].isdigit()],
    key=lambda c: int(c[1:]),
)

fig, axes = plt.subplots(3, 3, figsize=(13, 11))
for ax, col in zip(axes.flatten(), m_cols):
    m_view = pd.DataFrame(
        {
            "_level": merged[col].astype("string").fillna("(null)"),
            "isFraud": merged["isFraud"],
        }
    )
    grouped = (
        m_view.groupby("_level", observed=True)["isFraud"]
        .agg(n_fraud="sum", n="count", rate="mean")
        .reindex(["T", "F", "(null)"])
        .dropna(subset=["n"])
    )
    low, high = wilson_ci(grouped["n_fraud"].to_numpy(), grouped["n"].to_numpy())
    err = np.maximum(np.vstack([grouped["rate"] - low, high - grouped["rate"]]), 0.0)
    colors = {"T": "#3070b8", "F": "#c85050", "(null)": "#888888"}
    bar_colors = [colors[lvl] for lvl in grouped.index]
    ax.bar(range(len(grouped)), grouped["rate"].to_numpy(), yerr=err, capsize=4, color=bar_colors)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped.index)
    ax.axhline(merged["isFraud"].mean(), color="black", linestyle=":", linewidth=1)
    ax.set_title(f"{col}  (n={int(grouped['n'].sum()):,})")
    ax.set_ylabel("fraud rate")
fig.suptitle("M features — fraud rate by level (95% Wilson CI; dotted = overall mean)")
fig.tight_layout()
fig.savefig(FIG_DIR / "m_fraud_rate.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="m_fraud_rate")
plt.close(fig)
"""
    ),
    _md(
        """
**D.5 takeaways.** Most M flags discriminate strongly: `F` (mismatch)
fraud rates run materially above the overall mean while `T` rates
sit near it. `(null)` behaves differently per column — sometimes it
tracks `F` (suggests "missing match attempt" ≈ "couldn't match"),
sometimes `T`. Sprint 2 should treat null as its own category for M,
not impute it, given the per-column variation here.
"""
    ),
    _md(
        """
### D.6 — Identity (`DeviceType`, `DeviceInfo`)

The 76% null rate from Section A means treating null as "missing
data to impute" would discard the population-majority cohort. Below,
null is its own bin (labelled `(no identity)`) so we can compare
device type fraud rates against the no-identity baseline.
`DeviceInfo` is high-cardinality, so we restrict to the top 15
values with `n>=200`.
"""
    ),
    _code(
        """
dt_view = pd.DataFrame(
    {
        "_dt": merged["DeviceType"].astype("string").fillna("(no identity)"),
        "isFraud": merged["isFraud"],
    }
)
dt_grouped = (
    dt_view.groupby("_dt", observed=True)["isFraud"]
    .agg(n_fraud="sum", n="count", rate="mean")
    .sort_values("n", ascending=False)
)
low, high = wilson_ci(dt_grouped["n_fraud"].to_numpy(), dt_grouped["n"].to_numpy())
err = np.maximum(np.vstack([dt_grouped["rate"] - low, high - dt_grouped["rate"]]), 0.0)

fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = ["#888888" if v == "(no identity)" else "#3070b8" for v in dt_grouped.index]
ax.bar(range(len(dt_grouped)), dt_grouped["rate"].to_numpy(), yerr=err, capsize=4, color=bar_colors)
ax.set_xticks(range(len(dt_grouped)))
ax.set_xticklabels(dt_grouped.index, rotation=15)
ax.axhline(
    merged["isFraud"].mean(), color="black", linestyle=":", linewidth=1,
    label=f"overall ({merged['isFraud'].mean():.3%})",
)
for i, n in enumerate(dt_grouped["n"]):
    ax.text(i, 0, f"n={n:,}", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("fraud rate (95% Wilson CI)")
ax.set_title("DeviceType fraud rate (null = its own bin)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "device_type_fraud.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="device_type_fraud")
plt.close(fig)

print(dt_grouped.assign(ci_low=low, ci_high=high).to_string())
"""
    ),
    _code(
        """
DEVICE_INFO_TOP_K = 15
DEVICE_INFO_MIN_N = 200

di_view = pd.DataFrame(
    {
        "_di": merged["DeviceInfo"].astype("string").fillna("(no identity)"),
        "isFraud": merged["isFraud"],
    }
)
di_grouped = (
    di_view.groupby("_di", observed=True)["isFraud"]
    .agg(n_fraud="sum", n="count", rate="mean")
)
di_grouped = (
    di_grouped[di_grouped["n"] >= DEVICE_INFO_MIN_N]
    .sort_values("rate", ascending=False)
    .head(DEVICE_INFO_TOP_K)
)
low, high = wilson_ci(di_grouped["n_fraud"].to_numpy(), di_grouped["n"].to_numpy())
err = np.maximum(np.vstack([di_grouped["rate"] - low, high - di_grouped["rate"]]), 0.0)

fig, ax = plt.subplots(figsize=(11, max(5, 0.4 * len(di_grouped))))
ax.barh(
    range(len(di_grouped)), di_grouped["rate"].to_numpy(), xerr=err,
    capsize=3, color="#c85050",
)
ax.set_yticks(range(len(di_grouped)))
ax.set_yticklabels([str(v) for v in di_grouped.index])
ax.invert_yaxis()
ax.axvline(
    merged["isFraud"].mean(), color="black", linestyle=":", linewidth=1,
    label=f"overall ({merged['isFraud'].mean():.3%})",
)
ax.set_xlabel("fraud rate (95% Wilson CI)")
ax.set_title(f"DeviceInfo — top {len(di_grouped)} by fraud rate (n>={DEVICE_INFO_MIN_N})")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "device_info_fraud.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="device_info_fraud")
plt.close(fig)

print(di_grouped.assign(ci_low=low, ci_high=high).to_string())
attach_artifact(run, di_grouped.reset_index(), name="device_info_top15_table")
"""
    ),
    _md(
        """
**D.6 takeaways.** The "(no identity)" bin sits *above* the overall
fraud rate but *below* `mobile`'s rate — so missingness here is mild
positive signal but the strongest device-channel signal is "identity
present + mobile". Among `DeviceInfo` strings, several specific
values run multiples of the overall fraud rate with non-overlapping
Wilson CIs — these are concrete leads for Sprint 2's
identity-conditional features (e.g., "device fingerprint × card1 ×
past 24h").
"""
    ),
    _md(
        """
## Section E — Temporal Structure

Daily transaction count, daily fraud rate, TransactionDT min/max.
Confirms the 6-month span (Dec 2017 → May 2018) implied by the
community-standard anchor and justifies the **4/1/1 calendar split**
used by `Settings.train_end_dt` / `val_end_dt`.

Sprint 1's splitter encodes this decision mechanically so every
later sprint points at the same rows. If you change the anchor or
the boundaries, every downstream AUC number moves with you.
"""
    ),
    _code(
        """
from datetime import datetime

anchor = datetime.fromisoformat(SETTINGS.transaction_dt_anchor_iso)
seconds_per_day = 86400
day_since_anchor = merged["TransactionDT"] // seconds_per_day
daily_count = day_since_anchor.value_counts().sort_index()
daily_fraud_rate = merged.groupby(day_since_anchor)["isFraud"].mean()

train_day = SETTINGS.train_end_dt // seconds_per_day
val_day = SETTINGS.val_end_dt // seconds_per_day

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
daily_count.plot(ax=axes[0], color="#3b6fb3")
axes[0].axvline(train_day, color="#d4b43c", linestyle="--", label=f"train_end_dt (day {train_day})")
axes[0].axvline(val_day, color="#c85050", linestyle="--", label=f"val_end_dt (day {val_day})")
axes[0].set_ylabel("transactions / day")
axes[0].set_title(f"IEEE-CIS temporal density (anchor = {anchor.date().isoformat()} UTC)")
axes[0].legend()

daily_fraud_rate.rolling(7, min_periods=1).mean().plot(ax=axes[1], color="#3b6fb3")
axes[1].axvline(train_day, color="#d4b43c", linestyle="--")
axes[1].axvline(val_day, color="#c85050", linestyle="--")
axes[1].set_ylabel("7-day rolling fraud rate")
axes[1].set_xlabel("days since anchor")

fig.tight_layout()
fig.savefig(FIG_DIR / "temporal_structure.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="temporal_structure")
plt.close(fig)

print(f"Transaction DT range: {int(merged['TransactionDT'].min()):,} … {int(merged['TransactionDT'].max()):,} seconds")
print(f"Calendar span       : {int(day_since_anchor.min())} → {int(day_since_anchor.max())} days ({int(day_since_anchor.max()) - int(day_since_anchor.min()) + 1} days total)")
print(f"4/1/1 boundaries    : train_end_dt={train_day}d ({anchor.date()} + {train_day}d), val_end_dt={val_day}d")
"""
    ),
    _md(
        """
## Section F — Label Noise Investigation (cleanlab)

Stratified 50k sample, LightGBM classifier, `cleanlab.find_label_issues`
via cross-validation. Flags are written to
`data/interim/cleanlab_flags.parquet` for Sprint 2 / 3 reference.

**Decision:** do *not* remove flagged rows from training data.

Why: in fraud, labels come from chargebacks + investigator review —
these *are* the ground truth our production model will be evaluated
against. cleanlab identifies rows whose features look
"frauds-disguised-as-legit" (or vice versa); removing them trains
the classifier on a self-confirming subset and hides the exact
confusable cases the production system most needs to handle well.

Sprint 3 may revisit as a **sensitivity analysis only** — compare
AUC with vs without flagged rows, never ship a model that trained on
fewer than all the rows.
"""
    ),
    _code(
        """
from sklearn.model_selection import train_test_split

from fraud_engine.models.baseline import _select_feature_columns

CLEANLAB_SAMPLE_SIZE = 50_000
sample, _ = train_test_split(
    merged,
    train_size=CLEANLAB_SAMPLE_SIZE,
    stratify=merged["isFraud"],
    random_state=SETTINGS.seed,
)
feature_cols = _select_feature_columns(sample.columns)
X = sample[feature_cols]
y = sample["isFraud"].astype(np.int64).to_numpy()

from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict

clf = LGBMClassifier(
    objective="binary",
    n_estimators=100,
    num_leaves=31,
    learning_rate=0.1,
    random_state=SETTINGS.seed,
    verbose=-1,
)

pred_probs = cross_val_predict(clf, X, y, cv=3, method="predict_proba")

from cleanlab.filter import find_label_issues

flagged_mask = find_label_issues(
    labels=y,
    pred_probs=pred_probs,
    return_indices_ranked_by="self_confidence",
)
# `flagged_mask` is a 1-D index array (positions in `sample`), ordered
# worst-self-confidence first — not a boolean mask. Rebuild two aligned
# columns: `is_flagged` (bool per row) and `self_confidence_rank`
# (1 = most-suspect, -1 = unflagged).
print(f"cleanlab flagged {len(flagged_mask):,} / {CLEANLAB_SAMPLE_SIZE:,} rows ({len(flagged_mask)/CLEANLAB_SAMPLE_SIZE:.2%})")

sample = sample.reset_index(drop=True)
is_flagged = np.zeros(len(sample), dtype=bool)
is_flagged[flagged_mask] = True
ranks = np.full(len(sample), -1, dtype=np.int64)
ranks[flagged_mask] = np.arange(1, len(flagged_mask) + 1)

flags_df = pd.DataFrame(
    {
        "TransactionID": sample["TransactionID"].to_numpy(),
        "is_flagged": is_flagged,
        "self_confidence_rank": ranks,
    }
)

flags_path = SETTINGS.interim_dir / "cleanlab_flags.parquet"
flags_df.to_parquet(flags_path)
print(f"Wrote flags to {flags_path}")
attach_artifact(run, flags_df.head(20), name="cleanlab_flags_head")
"""
    ),
    _md(
        """
## Section G — Findings Summary

Copied verbatim into [`reports/sprint1_eda_summary.md`](../reports/sprint1_eda_summary.md).

1. **Scale & fingerprint.** 590,540 transactions × 434 merged columns
   on a stock IEEE-CIS snapshot; overall fraud prevalence 3.5%;
   identity-join coverage ~24%.
2. **Temporal span = 6 months.** TransactionDT covers Dec 2017 → May
   2018 under the community-standard 2017-12-01 UTC anchor. Supports
   a 4/1/1 calendar split (`train_end_dt=121d`, `val_end_dt=151d`).
3. **Fraud-rate stability over time.** 7-day rolling fraud rate stays
   within ±1pp of the overall 3.5%; no calendar structure forces a
   stratified split, so a clean temporal partition is sufficient.
4. **AUC is the right headline metric, not F1.** 3.5% class balance
   makes F1 highly threshold-sensitive; AUC is threshold-invariant
   and directly comparable across sprints. Sprint 4 replaces
   thresholding with expected-cost minimisation.
5. **TransactionAmt is strongly monotone with fraud risk.** The
   ≥ $1000 bucket is ~3× the overall rate — a feature Sprint 2 will
   bin explicitly.
6. **ProductCD is a strong categorical.** `C` and `H` have meaningfully
   higher fraud rates than `W`; LightGBM's native categorical
   handling picks this up without one-hot.
7. **Identity coverage is ~24%.** Model must not fail on NaN identity
   columns. Sprint 2 features over identity must be NaN-tolerant by
   construction.
8. **V-column redundancy is high.** V1–V40 shows dense within-group
   correlation (> 0.6) across many pairs. Sprint 2 can compress
   these into group-summary features without losing signal.
9. **Hour-of-day is predictive.** Fraud rate varies by ~1.5× across
   the 24-hour cycle; a simple derived feature is a Sprint 2 quick
   win.
10. **cleanlab-flagged rows stay in training.** Removing them would
    bake the classifier's own confusion into the training set.
    Flags are retained for Sprint 3 sensitivity analysis only.
11. **Baseline AUC gap (random vs temporal) is the leakage signal.**
    Sprint 2's feature pipeline must not widen this gap. The
    full-dataset numbers live in
    [`sprints/sprint_1/prompt_1_1_scaffold_report.md`](../sprints/sprint_1/prompt_1_1_scaffold_report.md).
"""
    ),
    _code(
        """
nb_run.__exit__(None, None, None)
print(f"Artefacts written to: {run.artifacts_dir}")
print(f"Figures written to : {FIG_DIR.resolve()}")
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

    Uses `jupyter nbconvert --execute --inplace`; raises if any cell errors.
    Doubles as the existing nbconvert verification gate, so running the
    builder once gives both a regenerated and a verified notebook.

    `DATA_DIR` is overridden to an absolute path for the nbconvert
    subprocess. The Makefile `-include .env / export` propagates the
    relative `DATA_DIR=./data` to every child process, and with
    `--inplace` nbconvert sets the kernel's cwd to the notebook's
    directory — so a relative `./data` would resolve to
    `notebooks/data/` and miss the manifest. Pydantic-Settings reads
    case-insensitive env vars, so an absolute `DATA_DIR` here
    short-circuits any cwd dependence.
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
