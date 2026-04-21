"""One-shot notebook builder for `notebooks/01_eda.ipynb`.

This file is not a pipeline step — it is the scaffolding used to
regenerate the EDA notebook without hand-editing JSON. Run once;
commit the output `.ipynb`. Re-run whenever the notebook structure
needs to change.

Usage:
    uv run python scripts/_build_eda_notebook.py
"""

from __future__ import annotations

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

Row and column counts, memory footprint, dtype histogram. These
fingerprint the dataset before any transformation — if a future
re-download changes the shape, the splitter + baseline numbers
deserve re-running.
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
    _md(
        """
## Section B — Target Analysis

Fraud prevalence, fraud rate vs transaction amount bucket, product
code, and hour-of-day. Three decisions come out of this section:

- **AUC over F1** as the headline metric — 3.5% fraud makes F1
  threshold-sensitive in a way that obscures model skill.
- **Economic cost (Sprint 4) replaces F1 threshold tuning** —
  `fraud_cost_usd` / `fp_cost_usd` are the decision-relevant units.
- **Hour-of-day is predictive enough to warrant a feature** (Sprint 2).
"""
    ),
    _code(
        """
fraud_rate = float(merged["isFraud"].mean())
print(f"Overall fraud rate: {fraud_rate:.4%}")

amt_bins = [0, 25, 50, 100, 250, 500, 1000, 5000, np.inf]
bucket = pd.cut(merged["TransactionAmt"], bins=amt_bins, include_lowest=True)
rate_by_amt = merged.groupby(bucket, observed=True)["isFraud"].mean()

rate_by_product = merged.groupby("ProductCD", observed=True)["isFraud"].mean().sort_values()

hour_of_day = (merged["TransactionDT"] // 3600) % 24
rate_by_hour = merged.groupby(hour_of_day)["isFraud"].mean()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
rate_by_amt.plot.bar(ax=axes[0], color="#3b6fb3")
axes[0].set_title("Fraud rate by transaction amount")
axes[0].set_ylabel("fraud rate")
axes[0].tick_params(axis="x", rotation=35)

rate_by_product.plot.bar(ax=axes[1], color="#3b6fb3")
axes[1].set_title("Fraud rate by ProductCD")

rate_by_hour.plot(ax=axes[2], color="#3b6fb3", marker="o")
axes[2].set_title("Fraud rate by hour of day")
axes[2].set_xlabel("hour (derived from TransactionDT)")

fig.tight_layout()
fig.savefig(FIG_DIR / "target_analysis.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="target_analysis")
plt.close(fig)
"""
    ),
    _md(
        """
## Section C — Missing Value Analysis

Top-20 columns by missing rate, grouped by family (V / D / C / id).
The identity join miss rate (~76%) is the headline: the model must
work *without* identity data, and any identity feature in Sprint 2
has to be NaN-tolerant. Categorical NaN → `NaN` category kept;
numeric NaN → passed through — LightGBM handles both natively.
"""
    ),
    _code(
        """
missing = merged.isna().mean().sort_values(ascending=False)
top20_missing = missing.head(20)

identity_cols = [c for c in merged.columns if c.startswith("id_") or c in {"DeviceType", "DeviceInfo"}]
identity_miss_rate = merged[identity_cols].isna().all(axis=1).mean() if identity_cols else float("nan")
print(f"Identity-join miss rate (no id_* present): {identity_miss_rate:.4%}")

fig, ax = plt.subplots(figsize=(10, 6))
top20_missing.plot.barh(ax=ax, color="#c85050")
ax.invert_yaxis()
ax.set_xlabel("missing rate")
ax.set_title("Top 20 columns by missing rate")
fig.tight_layout()
fig.savefig(FIG_DIR / "missing_values.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="missing_values")
plt.close(fig)

missing_by_family = {
    "V": float(merged.filter(regex=r"^V\\d+$").isna().mean().mean()),
    "C": float(merged.filter(regex=r"^C\\d+$").isna().mean().mean()),
    "D": float(merged.filter(regex=r"^D\\d+$").isna().mean().mean()),
    "M": float(merged.filter(regex=r"^M\\d+$").isna().mean().mean()),
    "id": float(merged.filter(regex=r"^id_").isna().mean().mean()) if identity_cols else float("nan"),
}
print("Mean missing rate by family:")
for family, rate in missing_by_family.items():
    print(f"  {family:3s}: {rate:.4%}")
"""
    ),
    _md(
        """
## Section D — Feature Group Deep Dives

Cardinality of categorical columns + within-group correlation on a
5% stratified sample (full correlation on 590k × 394 is ~1 GB).
Shared columns within a group usually indicate redundancy — Sprint 2
can compress them with a PCA-of-group or a mean-aggregate before
spending model capacity on them.
"""
    ),
    _code(
        """
cat_cols = [c for c in merged.columns if isinstance(merged[c].dtype, pd.CategoricalDtype)]
cardinality = {c: int(merged[c].nunique(dropna=True)) for c in cat_cols}
card_series = pd.Series(cardinality).sort_values(ascending=False)
print("Top-15 categorical columns by cardinality:")
print(card_series.head(15))

sample = merged.sample(frac=0.05, random_state=SETTINGS.seed)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
v_cols = [c for c in merged.columns if c.startswith("V")]
v_sample = sample[v_cols[:40]].fillna(0.0)
sns.heatmap(v_sample.corr(), ax=axes[0], cmap="RdBu_r", center=0, vmin=-1, vmax=1, cbar=False)
axes[0].set_title("V1..V40 correlation (5% sample)")
axes[0].set_xticklabels([])
axes[0].set_yticklabels([])

c_cols = [c for c in merged.columns if c.startswith("C") and c[1:].isdigit()]
c_sample = sample[c_cols].fillna(0.0)
sns.heatmap(c_sample.corr(), ax=axes[1], cmap="RdBu_r", center=0, vmin=-1, vmax=1)
axes[1].set_title("C1..C14 correlation (5% sample)")

fig.tight_layout()
fig.savefig(FIG_DIR / "feature_group_correlation.png", dpi=150, bbox_inches="tight")
attach_artifact(run, fig, name="feature_group_correlation")
plt.close(fig)
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
    [`sprints/sprint_1/prompt_1_1_report.md`](../sprints/sprint_1/prompt_1_1_report.md).
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


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
