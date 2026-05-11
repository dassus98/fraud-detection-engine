# ADR 0002: Temporal train / val / test split (no random splits)

- **Status:** Accepted
- **Date:** 2026-04-22
- **Sprint:** 1 (prompt 1.2.b)

## Context

Fraud-ML's most common silent-failure mode is **time-leakage**: training rows from "the future" (relative to a held-out evaluation row) leak signal that wouldn't be available at inference time, inflating offline metrics while the production model under-performs. The classic vectors:

- **Random train/test split.** Sampling rows uniformly at random places future rows in train and past rows in test. The model "learns" to predict the past from the future. AUC looks great; production AUC is 5-15 points lower.
- **Random K-fold cross-validation.** Same problem K times.
- **Per-entity grouped splits without time discipline.** Better than purely random (no card1 in both train and test), but if a card1's training rows come from after its test rows, leakage still happens.
- **Re-indexing for "convenience"**. Subtle: re-indexing by some hash to "scramble" the data for memory locality, while keeping a time-aware split, still requires assertion that the split itself respects time.

The IEEE-CIS dataset carries `TransactionDT` as integer seconds since a Vesta-anonymised anchor (CLAUDE.md §8). The community-standard interpretation maps the anchor to 2017-12-01 UTC; the relative ordering is unambiguous regardless of anchor choice. This makes a strict temporal split tractable.

Alternatives considered:

1. **Random 80/10/10 split.** Simplest; canonically wrong for fraud.
2. **Stratified random split (preserve fraud rate per fold).** Same time-leakage issue; the stratification only fixes class-balance, not chronology.
3. **Group-K-fold by card1 / addr1.** Avoids entity-overlap but doesn't enforce time ordering.
4. **Temporal split.** Strictly partition by `TransactionDT < train_end_dt < val_end_dt`. Matches the production deployment shape (model serves data from after its training cutoff).
5. **Walk-forward / time-series CV.** Multi-fold variant of (4) for more robust validation; future Sprint 6.x candidate.

## Decision

All train / val / test splits use `temporal_split()` from [`src/fraud_engine/data/splits.py`](../../src/fraud_engine/data/splits.py), parameterised by Settings:

| Boundary | Setting | Value (days) | Threshold (TransactionDT seconds) |
|---|---|---|---|
| train end | `Settings.train_end_dt` | 121 | 10,454,400 |
| val end | `Settings.val_end_dt` | 151 | 13,046,400 |

Rows with `TransactionDT < train_end_dt` go to train; rows with `train_end_dt ≤ TransactionDT < val_end_dt` go to val; the remainder to test.

**No random sampling anywhere.** This applies to:

- The 590,540-row training-data split (Sprint 1).
- All Sprint 2 feature-engineering temporal-integrity tests (`tests/lineage/test_*_temporal.py`).
- Sprint 3's Optuna hyperparameter sweep + isotonic calibration fit (val is the temporal-later slice).
- Sprint 4's economic-threshold optimisation (cost-curve sweep on the test slice).
- Sprint 6.1.b's drift baseline (trained on the train slice only).
- Sprint 6.1.c's performance baseline (operator-curated from the val slice).

**OOF cross-validation IS used within the train slice** (`TargetEncoder`, `ExponentialDecayVelocity`, `GraphFeatureExtractor.fraud_neighbor_rate`) but always with strict-past semantics — folds never see future rows. Each fold's encoder is fit on rows whose `TransactionDT` is strictly less than the current row's; ties don't see each other.

## Rationale

1. **Production-correct evaluation.** A model serves data from after its training cutoff. Temporal splits replicate this; random splits hide failure modes that production exposes.
2. **Catches concept-drift mid-training.** If fraud patterns shifted within the 121-day training window, a temporal split surfaces it in the train→val AUC gap rather than burying it.
3. **Lineage simplicity.** Train / val / test are deterministically the same rows across runs; bug reports referencing "row 42 of val" are stable.
4. **Cheap to enforce.** One `temporal_split()` function + Pandera schema assertion that train/val/test windows don't overlap. No K-fold orchestration overhead.
5. **Matches the Sprint 6.1.b drift baseline.** The drift monitor compares production scores against the train slice's distribution; the temporal-split discipline guarantees that comparison is meaningful (the baseline is genuinely "the past").

## Consequences

- **Cannot use scikit-learn's `cross_val_score` directly** without a custom CV iterator that respects time. Sprint 2 provides `TemporalKFold` for the OOF-within-train cases.
- **Smaller effective val/test slices than random K-fold.** Val ~83K rows, test ~92K rows from the 590K total. Acceptable — both are large enough for stable AUC estimates (>10K positive examples each).
- **Hyperparameter selection variance** is higher than with K-fold because there's only one val slice. Mitigation: Optuna's 100-trial sweep + early stopping at best_iteration provides stochastic averaging across trials.
- **Future Sprint 6.x walk-forward CV** is a deepening, not a replacement. The single-split is the production-correctness floor; walk-forward adds robustness on top.
- **Sprint 6.1.b drift baseline is timestamped by `train_end_dt`.** Operators changing `train_end_dt` must regenerate the baseline (`scripts/build_drift_baseline.py`).
- **Cross-sprint test invariant:** every PR adding feature engineering or training code must include a leakage test that asserts val AUC collapses to ~0.5 when labels are shuffled. The shuffle test is the strongest defence against subtle leakage regressions (e.g., a future PR re-introduces a random split in a sub-pipeline).

## Revisit triggers

- **Concept-drift evidence in production.** If Sprint 6.1.b drift alerts fire repeatedly on the same set of features, the 121-day training window is too short for the true fraud-pattern timescale. Consider lengthening or rolling.
- **Walk-forward CV becomes feasible.** If wall-clock training time drops below ~30 minutes (e.g., post-GPU upgrade), 5-fold walk-forward CV becomes practical and increases hyperparameter robustness.
- **Source data shape changes.** A new dataset (post-IEEE-CIS) without `TransactionDT` requires a different chronology primitive. Re-evaluate the split design in the new dataset's terms.
- **A regulator requires per-entity-grouped validation.** Some jurisdictions want models validated on entity disjoint slices (no card1 in both train and test). Combine temporal + group-K-fold rather than abandon temporal.

## References

- [`src/fraud_engine/data/splits.py`](../../src/fraud_engine/data/splits.py) — implementation: `temporal_split()` + `SplitFrames` dataclass.
- [`src/fraud_engine/config/settings.py:269-289`](../../src/fraud_engine/config/settings.py) — `train_end_dt` + `val_end_dt` Field declarations.
- [`tests/lineage/`](../../tests/lineage/) — temporal-integrity contract tests.
- [`sprints/sprint_1/`](../../sprints/sprint_1/) — Sprint 1 baseline reports (first temporal-split run).
- [`sprints/sprint_3/prompt_3_3_d_report.md`](../../sprints/sprint_3/prompt_3_3_d_report.md) — Sprint 3 train/val/test row counts (~414K / ~83K / ~92K).
- [`docs/MODEL_CARD.md#training-data`](../MODEL_CARD.md#training-data) — train/val/test slice surface for the model card.
- [`docs/FEATURE_DOCUMENTATION.md`](../FEATURE_DOCUMENTATION.md) — Tier 2/3/4 OOF discipline cross-references.
- [`CLAUDE.md` §1 + §8](../../CLAUDE.md) — dataset facts + anchor convention.
- Related: [ADR-0006 — Graph features batch](0006-graph-features-batch.md) (uses temporal split to define the OOF-safe fold structure for `fraud_neighbor_rate`).
