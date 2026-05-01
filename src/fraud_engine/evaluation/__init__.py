"""Model evaluation: calibration, economic cost, stratified metrics.

Sprint 3 prompt 3.3.c: calibration tooling — Platt scaling, isotonic
regression, reliability-diagram diagnostic, log_loss / Brier / ECE
helpers, and an automatic chooser (`select_calibration_method`) that
picks whichever method minimises log loss on a held-out split (or
returns "none" if neither beats the uncalibrated baseline).

Sprint 4 will add economic-cost evaluation and stratified metrics
under this same package; the operational decision metrics
(`economic_cost`, `precision_recall_at_k`, `recall_at_fpr`,
`compute_psi`) live in `src/fraud_engine/utils/metrics.py` and are
imported from there at the consumer site.
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

__all__ = [
    "Calibrator",
    "IsotonicCalibrator",
    "PlattScaler",
    "brier_score",
    "expected_calibration_error",
    "fit_isotonic_regression",
    "fit_platt_scaling",
    "log_loss",
    "reliability_diagram",
    "select_calibration_method",
]
