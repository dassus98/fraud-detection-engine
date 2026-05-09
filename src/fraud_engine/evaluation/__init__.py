"""Model evaluation: calibration, economic cost, stratified metrics.

Sprint 3 prompt 3.3.c: calibration tooling — Platt scaling, isotonic
regression, reliability-diagram diagnostic, log_loss / Brier / ECE
helpers, and an automatic chooser (`select_calibration_method`) that
picks whichever method minimises log loss on a held-out split (or
returns "none" if neither beats the uncalibrated baseline).

Sprint 4 prompt 4.1: economic-cost evaluation — `EconomicCostModel`
wraps the `economic_cost` primitive in `utils/metrics.py` with
threshold sweep + sensitivity analysis.

Sprint 4 prompt 4.2: stratified evaluation — `StratifiedEvaluator`
computes per-segment AUC / PR-AUC / economic cost across five
business-meaningful axes (amount bucket, ProductCD, device type,
identity coverage, month) plus a heatmap visualisation. The per-call
decision metrics (`economic_cost`, `precision_recall_at_k`,
`recall_at_fpr`, `compute_psi`) continue to live in
`src/fraud_engine/utils/metrics.py` and are imported from there at
the consumer site.
"""

from __future__ import annotations

from fraud_engine.evaluation.calibration import (
    Calibrator,
    IsotonicCalibrator,
    PlattScaler,
    brier_score,
    expected_calibration_error,
    fit_isotonic_regression,
    fit_platt_scaling,
    log_loss,
    reliability_diagram,
    select_calibration_method,
)
from fraud_engine.evaluation.economic import EconomicCostModel
from fraud_engine.evaluation.stratified import StratifiedEvaluator

__all__ = [
    "Calibrator",
    "EconomicCostModel",
    "IsotonicCalibrator",
    "PlattScaler",
    "StratifiedEvaluator",
    "brier_score",
    "expected_calibration_error",
    "fit_isotonic_regression",
    "fit_platt_scaling",
    "log_loss",
    "reliability_diagram",
    "select_calibration_method",
]
