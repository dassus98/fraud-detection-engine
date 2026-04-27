# Sprint 1 — EDA Summary

**Audience:** Sprint 2's feature-engineering work and any reviewer who
wants the headline numbers without re-running the notebook.

**Source notebook:** [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb)
(executes end-to-end under `pytest --nbmake` in ~65 s).

**Underlying data:** IEEE-CIS Fraud Detection (Vesta Corporation,
Kaggle 2019), merged transactions + identity, 590,540 × 434.

---

## Headline findings

1. **Scale & fingerprint.** 590,540 transactions × 434 merged columns
   on a stock IEEE-CIS snapshot; overall fraud prevalence 3.50%;
   identity-join coverage ~24%.
2. **Temporal span = 6 months.** TransactionDT covers Dec 2017 → May
   2018 under the community-standard `2017-12-01T00:00:00+00:00`
   anchor. Justifies a **4/1/1 calendar split**
   (`train_end_dt = 86400 × 121`, `val_end_dt = 86400 × 151`).
3. **Fraud rate is stable over time.** Weekly fraud rate stays within
   ±1 pp of the overall 3.5% (Wilson 95% CI ribbons in Section E
   confirm the variation is mostly noise); no calendar structure forces
   a stratified split, so a clean temporal partition is sufficient.
4. **AUC is the right headline metric, not F1.** 3.5% class balance
   makes F1 highly threshold-sensitive; AUC is threshold-invariant
   and directly comparable across sprints. Sprint 4 replaces
   thresholding with expected-cost minimisation (see CLAUDE.md §8).
5. **Baseline is already strong.** Random-split AUC 0.9615; temporal-
   split AUC 0.9247 on raw features alone. The 0.037 gap quantifies
   the leakage pressure Sprint 2's features must not amplify.
6. **Missingness has structure, and so does identity.** The V-column
   family resolves to 23 distinct null-mask signatures (one signature
   covers 168 columns) — Sprint 2 can compress aggressively without
   losing signal. The 76% of rows with no identity data show
   materially elevated fraud rate vs the desktop/mobile cohorts —
   absence of identity is itself a feature.

---

## Top 5 engineering decisions informed by EDA

These are the load-bearing calls Sprint 2 onwards inherits from the
work in `notebooks/01_eda.ipynb`. Each is recorded here so a reader
can audit the lineage from finding to decision without re-running
the analysis.

1. **4/1/1 calendar temporal split.** Mechanically encoded in
   `Settings.train_end_dt = 86400 × 121` and
   `Settings.val_end_dt = 86400 × 151`. The KS test on top-10
   features shows moderate drift (not catastrophic) — a single
   calendar partition is safe; stratified moving-window CV would be
   overkill on a 6-month window.
2. **AUC over F1 as the headline metric.** 3.5% class imbalance
   makes F1 highly threshold-sensitive. AUC is threshold-invariant
   and directly comparable across sprints. Sprint 4 replaces the
   threshold with expected-cost minimisation (CLAUDE.md §8).
3. **Keep cleanlab-low-quality rows in training.** Labels come from
   chargebacks and investigator review and *are* the ground truth
   our model is judged against. Removing the rows cleanlab flags as
   confusable trains the classifier on the easy subset and bakes
   its own confusion into the training distribution.
4. **NaN-tolerant identity features.** 76% of rows have no identity
   columns — every identity-derived feature in Sprint 2 must
   produce a sensible value when every `id_*` is null. The
   "(no identity)" cohort has elevated fraud rate, so null-as-
   signal is itself a feature, not a defect.
5. **V-column compression via NaN-group equivalence + correlation
   pruning.** 23 NaN-group signatures cover 284 columns; ~70–80 V
   cols are droppable by ρ > 0.95 on a seeded 50-col sample.
   Sprint 2 reduces the V family to group-summary statistics or
   PCA-of-group, freeing model capacity for genuinely new signal.

---

## Data shape & quality

| Property | Value |
|----------|-------|
| Rows | 590,540 |
| Merged columns | 434 |
| Memory (deep) | ~1.9 GB after `RawDataLoader._optimize` |
| Overall fraud rate | 3.499% |
| Identity coverage | ~24% (76% of rows have *no* identity columns at all) |

**Decisions:**

- **Categorical NaN stays as `NaN` category.** LightGBM handles it
  natively; one-hot would explode dimensionality without a signal
  return.
- **Numeric NaN passes through.** LightGBM's missing-value handling
  is consistent with the production-feature service (Sprint 5).
- **Identity must be NaN-tolerant.** Any Sprint 2 identity-derived
  feature must produce a sensible value for the 76% of rows where
  every `id_*` is null.

## Temporal structure & split choice

The dataset's TransactionDT is integer seconds since an anonymised
anchor. The community treats `2017-12-01 00:00 UTC` as the convention
(documented in `Settings.transaction_dt_anchor_iso`). Under that
anchor:

| Split | Window | Days | Rows | Fraud rate |
|-------|--------|------|------|-----------|
| Train | day 1 → day 121 (Dec 2017 → Mar 2018) | 121 | 414,542 | 3.522% |
| Val   | day 122 → day 151 (Apr 2018) | 30 | 83,571 | 3.410% |
| Test  | day 152 → day 183 (May 2018, partial) | 31 | 92,427 | 3.476% |

Encoded mechanically in `Settings.train_end_dt = 86400 × 121` and
`Settings.val_end_dt = 86400 × 151`; the splitter (`temporal_split`
in [`src/fraud_engine/data/splits.py`](../src/fraud_engine/data/splits.py))
uses strict less-than on the upper bound so every row lands in
exactly one split.

**Why temporal, not random.** Fraud risk drifts: new attack vectors,
new merchant relationships, seasonal cardholder behaviour. A random
split that trains on April rows and tests on January rows is
predicting the past from the future — it inflates AUC and does not
mirror production. Temporal split forces evaluation to mirror what
the model will actually see (only "future" transactions). The gap
between random- and temporal-AUC is then a **leakage-pressure
signal**: the wider it grows in Sprint 2, the more carefully each
new feature must be inspected for time-bleed.

**Why 4/1/1 calendar months, not stratified moving-window CV.** The
fraud rate is uniform within ±1 pp across the calendar — there is no
seasonality strong enough to need stratification. A moving-window CV
is the right tool for Sprint 3's tuning; the baseline only needs one
defensible held-out month.

## Label quality

- **Sample.** 50,000 rows stratified on `isFraud`, seeded from
  `Settings.seed`.
- **Tool.** `cleanlab.rank.get_label_quality_scores` over a
  gradient-boosted baseline's 3-fold cross-validated probabilities.
  Quality scores are continuous in [0, 1] (0 = least trustworthy
  label, 1 = most).
- **Threshold.** Rows with quality score below
  `LABEL_QUALITY_THRESHOLD = 0.05` are marked confusable — 0.05 is
  the cleanlab docs' conservative "very-likely-issue" floor.
- **Result.** 576 / 50,000 rows below 0.05 (1.15%).
- **Artefact.** [`data/interim/cleanlab_quality_scores.parquet`](../data/interim/cleanlab_quality_scores.parquet)
  — columns `TransactionID`, `quality_score`, `below_threshold`.

**Decision:** keep all flagged rows in training.

**Why.** In fraud, the labels come from chargebacks and investigator
review — that *is* the ground truth our production model is judged
against. cleanlab identifies rows whose features look "fraud-
disguised-as-legit" (or vice versa); these are exactly the
confusable cases the production system most needs to handle well.
Removing them trains the classifier on the easy subset and bakes
the model's own confusion into the training distribution — a
self-confirming filter. Sprint 3 may revisit as a **sensitivity
analysis only** (compare AUC with vs without flagged rows; never
ship a model that trained on a filtered set).

## Baseline AUC

Full-dataset run, both variants, via
[`scripts/run_sprint1_baseline.py`](../scripts/run_sprint1_baseline.py)
(also reachable as `make sprint1-baseline`):

| Variant | AUC | Model |
|---------|-----|-------|
| Random (stratified 80/20, seed=42) | **0.9615** | `models/baseline_random_c4dc58d6150d.joblib` |
| Temporal (4/1/1 calendar) | **0.9247** | `models/baseline_temporal_fbd8f9501675.joblib` |
| Gap (random − temporal) | **0.0368** | leakage-pressure signal for Sprint 2 |

Top-5 features by gain in the temporal run (full top-20 in the
attached run artefact `baseline_temporal_result.json`):

| Rank | Feature | Gain |
|------|---------|-----:|
| 1 | V258 | 122,294 |
| 2 | C1 | 59,123 |
| 3 | DeviceInfo | 48,329 |
| 4 | C14 | 44,201 |
| 5 | V294 | 31,871 |

Sprint 2's features should *complement* — not duplicate — these.
Particularly: the V column family is an obvious target for
group-aggregate compression (see Section D heatmap), and any
DeviceInfo-derived feature must remain valid when the column is
NaN (76% of rows).

## Handoffs to Sprint 2

The feature pipeline must:

1. **Not widen the random/temporal AUC gap.** Each new feature is
   suspect until proven temporally clean — write the
   "feature-uses-only-past-data" temporal-integrity test (CLAUDE.md
   §6.3) before merging.
2. **Be NaN-tolerant on identity columns.** 76% of rows have no
   identity data; any feature that fails or imputes silently on
   missing identity is shipping a covert covariate-shift bug.
3. **Compress the V column family.** Within-group correlation > 0.6
   on many V1..V40 pairs (see `reports/figures/feature_group_correlation.png`)
   — group-summary statistics or PCA-of-group will free model
   capacity for genuinely new signal.
4. **Bin TransactionAmt explicitly.** The fraud rate is monotone in
   the amount bucket; an explicit bin column is a Sprint 2 quick
   win that LightGBM cannot infer at split time as cheaply as a
   pre-computed feature.
5. **Add hour-of-day as a derived feature.** Fraud rate varies by
   ~1.5× across the 24-hour cycle (figure: `target_analysis.png`).
6. **Read the splits manifest** at
   `data/interim/splits_manifest.json` instead of recomputing the
   partition. Sprint 4's evaluator will do the same; this is how we
   guarantee every later AUC number is on the same rows the
   baseline reported.

The pipeline must **not**:

- Drop cleanlab-flagged rows (see Label quality above).
- Use post-event columns (anything derived from `isFraud`, the
  outcome of an investigation, etc.).
- Emit features whose value at row R depends on any row whose
  TransactionDT is greater than R's TransactionDT.

## Figures

All saved at 150 DPI to `reports/figures/`. The notebook re-renders
them on every run; the PNGs here are the canonical reference.

| File | Section | Caption |
|------|---------|---------|
| [`target_analysis.png`](figures/target_analysis.png) | B | Fraud rate by amount bucket / ProductCD / hour-of-day |
| [`missing_values.png`](figures/missing_values.png) | C | Top-20 columns by missing rate |
| [`feature_group_correlation.png`](figures/feature_group_correlation.png) | D | V1..V40 and C1..C14 within-group correlation |
| [`temporal_structure.png`](figures/temporal_structure.png) | E | Weekly transaction volume + weekly fraud rate (Wilson 95% CI), with 4/1/1 split boundaries marked |
