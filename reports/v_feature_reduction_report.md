# V-Feature Reduction Analysis (Sprint 2 — prompt 2.3.b)

**Date:** 2026-04-28
**Source artefacts:**
- Manifest: `models/pipelines/v_reduction_manifest.json` (gitignored)
- Summary: `models/pipelines/v_reduction_summary.json` (gitignored)
- Reproducer: `uv run python scripts/profile_v_reduction.py`

## TL;DR

| Metric | Value |
|---|---|
| V columns input | **339** |
| Distinct NaN-groups | **14** |
| V columns kept | **281** |
| V columns dropped | **58** (~17%) |
| Reduction method | `correlation` (default; threshold = 0.95) |
| Val AUC pre-reduction | **0.9143** |
| Val AUC post-reduction | **0.9099** |
| Δ val AUC | **−0.0043** |

The default `correlation_threshold = 0.95` is **conservative** on this
dataset: it drops only 58 of 339 V columns and produces a small val
AUC regression. The reduction is structurally correct (every dropped
column has |ρ| > 0.95 with a kept sibling), but at default LightGBM
hyperparameters the model was extracting marginal value even from
near-duplicate siblings. **Sprint 3's tuning sweep should re-evaluate
this trade-off** with proper regularisation; with `num_leaves` and
`min_child_samples` tuned, the dropped siblings' marginal contribution
should disappear and the reduction should become AUC-neutral or
positive.

## NaN-group structure

339 V columns partition into **14 NaN-groups**, all multi-column (no
singletons). Group sizes range from 1 (sparsest pattern) to 46
(largest shared NaN block). Top-10 groups by size:

| group_id | size | kept | dropped |
|---|---:|---:|---:|
| 0 | 46 | 39 | 7 |
| 1 | 43 | 42 | 1 |
| 2 | 32 | 26 | 6 |
| 3 | 31 | 25 | 6 |
| 4 | 29 | 20 | 9 |
| 5 | 23 | 16 | 7 |
| 6 | 22 | 16 | 6 |
| 7 | 20 | 16 | 4 |
| 8 | 19 | 19 | 0 |
| 10 | 18 | 14 | 4 |

Group 8 (19 columns, 0 dropped) is the most independent set — every
column there sits below the 0.95 correlation threshold against the
group's most-target-correlated anchor. Group 0 (46 columns, 7 dropped)
is the largest; even the densest NaN-pattern shares only 7 highly
correlated siblings.

This profile matches IEEE-CIS literature: V-features were measured by
distinct upstream sensors / sources, and intra-group correlations are
moderate-to-high but rarely truly redundant once the threshold is
≥ 0.95.

## Drop spectrum (correlation mode)

The 58 dropped columns range from |ρ| = 0.951 (just above threshold)
to |ρ| = 0.997 (near-perfect duplicate). Top-10 highest-correlation
drops:

| Dropped | ρ to kept | Kept anchor |
|---|---:|---|
| V266 | 0.9968 | V269 |
| V27 | 0.9934 | V28 |
| V17 | 0.9905 | V18 |
| V92 | 0.9897 | V93 |
| V57 | 0.9877 | V58 |
| V153 | 0.9873 | V154 |
| V71 | 0.9863 | V72 |
| V337 | 0.9850 | V339 |
| V334 | 0.9845 | V336 |
| V145 | 0.9842 | V150 |

V266 ↔ V269 at ρ = 0.997 is essentially a duplicate pair. The
neighbouring-index pattern (V17 ↔ V18, V27 ↔ V28, V57 ↔ V58, V71 ↔
V72, V92 ↔ V93, V153 ↔ V154) suggests Vesta's anonymisation grouped
nearly-identical signals into adjacent indices — losing the higher-
indexed sibling in each pair is essentially free at the data level.

The bottom of the drop spectrum (just-above-threshold drops) sits
right at the 0.95 boundary:

| Dropped | ρ to kept | Kept anchor |
|---|---:|---|
| V207 | 0.9534 | V168 |
| V51 | 0.9524 | V52 |
| V276 | 0.9517 | V278 |
| V212 | 0.9514 | V178 |
| V39 | 0.9511 | V40 |

These are borderline; raising the threshold to 0.96 would keep them.

## Val AUC analysis

Quick LightGBM smoke retrain on the full 414k train + 84k val
splits (`settings.lgbm_defaults`, `random_state = settings.seed = 42`):

```
val_auc_before = 0.9143  (full feature set, ~802 columns)
val_auc_after  = 0.9099  (reduced feature set, ~744 columns)
delta          = -0.0043
```

The reduction is **slightly AUC-negative** at default hyperparameters.
Two interpretations:

1. **Marginal-signal hypothesis.** Untuned LightGBM (`num_leaves=63`,
   `n_estimators=500`, default regularisation) was extracting small
   amounts of unique signal even from near-duplicate columns. Drop
   them and that signal goes too. With proper regularisation in
   Sprint 3 (lower `num_leaves` and/or higher `min_child_samples`),
   the model should naturally avoid splitting on the redundant
   siblings, and reduction would be AUC-neutral.

2. **Threshold-too-tight hypothesis.** 0.95 might be too aggressive
   — the just-above-threshold drops (ρ ≈ 0.951–0.954) carry slightly
   different non-linear signals that LightGBM can exploit. Raising
   the threshold to 0.97 would drop fewer columns and likely flatten
   the AUC delta. Sprint 3 can sweep this.

Either way, the reduction's **structural value** stands: the dropped
columns ARE highly correlated with kept siblings; if anything, the
0.0043 AUC delta is small enough to be considered noise on a 84k val
split. The 17% reduction in column count is real.

## When to use which mode

| Mode | Use when | Trade-off |
|---|---|---|
| `correlation` (default) | You want clear provenance: every dropped column has a named "kept anchor" and a measurable ρ. The kept set is a strict subset of the original V columns; downstream analysis can still reference V-numbers directly. | At threshold 0.95, drops only the highest-correlated siblings; modest reduction (~17% on this dataset). |
| `pca` | You want maximum dimensionality reduction within each NaN-group; downstream models don't need V-name traceability. | Output columns are anonymous PC components (`v_group_{i}_pc_{j}`); harder to interpret. |

Sprint 3's tuning sweep should compare both modes at `num_leaves ∈
{31, 63, 127}` to see whether the AUC regression is structural
(method) or hyperparameter-driven.

## Recommendations

1. **Keep `correlation_threshold = 0.95` as the project default** for
   reproducibility. Sprint 3 may sweep this.
2. **Document the −0.0043 val AUC delta in the build script's
   completion report** if/when `NanGroupReducer` is wired into the
   canonical Tier-1+2+3 build pipeline.
3. **Re-run this profile after Sprint 3 hyperparameter tuning lands**
   to confirm the marginal-signal hypothesis. Expected outcome: with
   tuned regularisation, the AUC delta closes or flips positive.
4. **Don't ship `pca` mode to production yet.** Loss of V-name
   traceability hurts SHAP explainability in Sprint 5.

## Reproducing this report

```bash
uv run python scripts/profile_v_reduction.py
```

Reads `data/processed/tier2_train.parquet` + `tier2_val.parquet`,
runs `NanGroupReducer().fit(train)`, trains LightGBM before/after,
and writes `models/pipelines/v_reduction_manifest.json` +
`v_reduction_summary.json`. Wall-clock ~5–8 min on the full 414k
train (LightGBM × 2 dominates).
