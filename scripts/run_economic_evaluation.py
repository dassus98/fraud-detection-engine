"""Sprint 4 economic-evaluation runner: test-set τ + savings + report.

End-to-end orchestrator that closes the Sprint 4 loop on the held-out
test split: load Model A + the calibrator (3.3.d artefacts) → score
the test set → find the cost-optimal threshold via 4.1's
`EconomicCostModel.optimize_threshold` → run the ±20 % sensitivity
grid → run 4.2's `StratifiedEvaluator` → estimate annual savings →
write the report at `reports/economic_evaluation.md` → update `.env`'s
`DECISION_THRESHOLD` in place (with a backup) so Sprint 5's serving
stack reads the cost-optimal value.

Business rationale:
    Sprint 4.1 / 4.2 / 4.3 built the surface (cost model, stratified
    evaluator, defaults YAML, ADR 0003) but never ran it end-to-end
    against the held-out test split. Until this script runs,
    `Settings.decision_threshold` is the placeholder 0.5 from
    `.env`, NOT the cost-curve optimum. This is the prompt that
    makes Sprint 4 production-relevant: it produces the threshold
    Sprint 5's API path will gate scoring against, the savings
    estimate a senior reviewer expects to see in a deployment review,
    and the per-stratum evidence that the global τ doesn't fail
    catastrophically on any single business segment.

Trade-offs considered:
    - **Loads existing artefacts; does NOT retrain.** The 4.4 spec
      is "evaluate on held-out test", not "re-train". If the model
      joblib is stale, that's a `train_lightgbm.py` re-run concern,
      not 4.4's.
    - **Calibrated test scores via `calibrator.transform(...)`** —
      the production-realistic flow. Uncalibrated probabilities
      give a wrong cost surface (ADR 0003: calibration is the
      load-bearing dependency for the Bayes-decision argument).
    - **Annual-savings baseline = τ = 0.5.** The placeholder this
      script replaces. Alternative baselines (no-model 100 % FN
      cost; F1-optimal τ; "block everything") answer different
      questions; the "threshold-optimisation value" question is
      the one this report answers.
    - **`.env` mutated in place with `.env.bak` backup.** Settings
      reads `.env`; the chosen τ has to land there or Sprint 5
      won't see it. The `.bak` is the rollback path. `--dry-run`
      and `--no-update-env` are the escape hatches.
    - **Month axis skipped on test (`month=None`).** Tier-5 parquet
      drops `timestamp` (`build_features_all_tiers.py:110-111`)
      and the test set is approximately one calendar month per
      the temporal split, so within-test month stratification is
      degenerate. Documented in the report's caveats; cross-month
      drift is Sprint 6 monitoring territory.
    - **Cost-curve PNG and stratified-heatmap PNG both committed**
      to `reports/figures/` (not gitignored). The PNGs are the
      durable artefact a reviewer reads when the markdown image
      links go stale.
    - **Annual savings is a linear extrapolation** (per-txn delta
      × monthly volume × 12). Premature precision is a trap; the
      linear estimate is the right level of fidelity for a
      cost-curve evaluation. Report caveats spell this out.
    - **Three acceptance gates reported, not enforced.** Like
      `train_lightgbm.py`, the script runs to completion regardless
      of [0.3, 0.5] / $500K / sensitivity-stability gates. The
      report carries green/red flags inline; the integration test
      enforces only the catastrophic floor.

Cross-references:
    - `scripts/train_lightgbm.py` — structural precedent (Click
      CLI, `Final` constants block, `_render_*_report` builder).
    - `src/fraud_engine/evaluation/economic.py` —
      `EconomicCostModel.optimize_threshold`,
      `sensitivity_analysis` (Sprint 4.1).
    - `src/fraud_engine/evaluation/stratified.py` —
      `StratifiedEvaluator.evaluate`, `plot_heatmap` (Sprint 4.2).
    - `configs/economic_defaults.yaml` — cost provenance
      (Sprint 4.3).
    - `docs/ADR/0003-economic-threshold.md` — justification for
      cost-based τ (Sprint 4.3).

Usage:
    uv run python scripts/run_economic_evaluation.py
    uv run python scripts/run_economic_evaluation.py --quick
    uv run python scripts/run_economic_evaluation.py --dry-run
    uv run python scripts/run_economic_evaluation.py --portfolio-monthly 500000
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

import click
import joblib
import matplotlib

matplotlib.use("Agg")  # headless; no display server

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from fraud_engine.config.settings import Settings, get_settings  # noqa: E402
from fraud_engine.evaluation.calibration import Calibrator  # noqa: E402
from fraud_engine.evaluation.economic import EconomicCostModel  # noqa: E402
from fraud_engine.evaluation.stratified import StratifiedEvaluator  # noqa: E402
from fraud_engine.models.lightgbm_model import LightGBMFraudModel  # noqa: E402
from fraud_engine.utils.logging import get_logger  # noqa: E402

_logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------

_TEST_PARQUET_NAME: Final[str] = "tier5_test.parquet"
_MODELS_SUBDIR: Final[str] = "sprint3"
_CALIBRATOR_FILENAME: Final[str] = "calibrator.joblib"

# Non-feature columns dropped before scoring; mirrors train_lightgbm.py.
_NON_FEATURE_COLS: Final[frozenset[str]] = frozenset(
    {"TransactionID", "TransactionDT", "isFraud", "timestamp"}
)

# Output paths (anchored to repo root via `parents[1]` from this file).
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_REPORT_PATH: Final[Path] = _PROJECT_ROOT / "reports" / "economic_evaluation.md"
_COST_CURVE_FIG_PATH: Final[Path] = (
    _PROJECT_ROOT / "reports" / "figures" / "economic_cost_curve.png"
)
_HEATMAP_FIG_PATH: Final[Path] = (
    _PROJECT_ROOT / "reports" / "figures" / "economic_stratified_heatmap.png"
)

# Acceptance gates (CLAUDE.md §8 + spec).
_OPTIMUM_BAND_LOW: Final[float] = 0.3
_OPTIMUM_BAND_HIGH: Final[float] = 0.5
_ANNUAL_SAVINGS_FLOOR_USD: Final[float] = 500_000.0
_SENSITIVITY_SPREAD_CEILING: Final[float] = 0.20

# Portfolio-volume defaults.
_DEFAULT_PORTFOLIO_MONTHLY: Final[int] = 1_000_000
_MONTHS_PER_YEAR: Final[int] = 12

# Baseline threshold for the savings comparison: the placeholder τ
# this script replaces. Documented in CLAUDE.md §8 +
# configs/economic_defaults.yaml.
_BASELINE_THRESHOLD: Final[float] = 0.5

# .env mutation safety.
_ENV_FILENAME: Final[str] = ".env"
_ENV_BACKUP_SUFFIX: Final[str] = ".bak"
_ENV_THRESHOLD_KEY: Final[str] = "DECISION_THRESHOLD"

# Smoke (--quick) overrides.
_SMOKE_SAMPLE_SIZE: Final[int] = 5_000
_SMOKE_PORTFOLIO_MONTHLY: Final[int] = 100_000

# Report formatting.
_USD_DIGITS: Final[int] = 2
_TAU_DIGITS: Final[int] = 4

# Top / bottom rows of the sensitivity table to render in the report
# (the full 5×5×5 = 125 cells is too dense for a markdown table).
_SENSITIVITY_TOP_N: Final[int] = 5

# Float precision when writing the threshold to .env.
_ENV_THRESHOLD_DIGITS: Final[int] = 6


# ---------------------------------------------------------------------
# Result dataclass.
# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Bundle returned by `run_economic_evaluation` for downstream / test consumption.

    Captures every metric the report renders + the gate-pass booleans.
    Frozen + slotted so tests can't accidentally mutate.
    """

    optimal_threshold: float
    baseline_threshold: float
    cost_curve: pd.DataFrame
    sensitivity: pd.DataFrame
    stratified: pd.DataFrame
    cost_per_txn_at_optimal: float
    cost_per_txn_at_baseline: float
    annual_savings_usd: float
    annual_cost_at_optimal_usd: float
    annual_cost_at_baseline_usd: float
    sensitivity_spread: float
    n_test_rows: int
    portfolio_monthly: int
    optimum_band_gate_pass: bool
    annual_savings_gate_pass: bool
    sensitivity_stability_gate_pass: bool
    report_path: Path
    cost_curve_figure_path: Path
    heatmap_figure_path: Path
    env_updated: bool
    env_path: Path | None
    env_backup_path: Path | None


# ---------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------


def _load_test_parquet(processed_dir: Path) -> pd.DataFrame:
    """Read `tier5_test.parquet`. Helpful error if missing."""
    path = processed_dir / _TEST_PARQUET_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected tier-5 test parquet at {path} — run "
            f"`uv run python scripts/build_features_all_tiers.py` first."
        )
    return pd.read_parquet(path)


def _select_features(df: pd.DataFrame) -> list[str]:
    """LightGBM-ingestable subset: drop non-features + object/string dtypes.

    Mirrors `scripts/train_lightgbm.py:_select_features` exactly so the
    test-time feature set matches what the model was trained on.
    """
    return [
        col
        for col in df.columns
        if col not in _NON_FEATURE_COLS
        and not pd.api.types.is_object_dtype(df[col])
        and not pd.api.types.is_string_dtype(df[col])
    ]


def _stratified_subsample(df: pd.DataFrame, target_n: int, seed: int) -> pd.DataFrame:
    """Stratified subsample to ~`target_n` rows by `isFraud`. Skip if df smaller."""
    if len(df) <= target_n:
        return df.reset_index(drop=True)
    kept, _ = train_test_split(
        df,
        train_size=target_n,
        stratify=df["isFraud"],
        random_state=seed,
    )
    # `cast` because `train_test_split`'s sklearn type stubs are too
    # loose for mypy to narrow back to `pd.DataFrame`. Same pattern as
    # the other Sprint 3 / 4 training scripts (audit gap-fix).
    return cast(pd.DataFrame, kept.reset_index(drop=True))


# ---------------------------------------------------------------------
# Model + calibrator loading.
# ---------------------------------------------------------------------


def _load_model_artefacts(
    models_dir: Path,
) -> tuple[LightGBMFraudModel, Calibrator]:
    """Load the saved Model A + isotonic calibrator from `models_dir`.

    Helpful error messages if either artefact is missing — points at
    `train_lightgbm.py` as the producer.
    """
    sprint3_dir = models_dir / _MODELS_SUBDIR
    model_path = sprint3_dir / "lightgbm_model.joblib"
    calibrator_path = sprint3_dir / _CALIBRATOR_FILENAME

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Expected Model A joblib at {model_path} — run "
            f"`uv run python scripts/train_lightgbm.py` first."
        )
    if not calibrator_path.is_file():
        raise FileNotFoundError(
            f"Expected calibrator at {calibrator_path} — run "
            f"`uv run python scripts/train_lightgbm.py` first "
            f"(`select_calibration_method` writes the calibrator)."
        )

    model = LightGBMFraudModel.load(sprint3_dir)
    calibrator = cast(Calibrator, joblib.load(calibrator_path))
    return model, calibrator


# ---------------------------------------------------------------------
# Cost-curve plot.
# ---------------------------------------------------------------------


def _save_cost_curve_figure(
    cost_curve: pd.DataFrame,
    optimal_tau: float,
    baseline_tau: float,
    out_path: Path,
) -> None:
    """Line plot: cost_per_txn vs threshold; mark optimum + baseline.

    Mirrors `_save_latency_histogram`'s style from `train_lightgbm.py`:
    figsize 8×4.5, dpi=150, tight bbox, vertical lines for the
    notable thresholds.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        cost_curve["threshold"].to_numpy(),
        cost_curve["cost_per_txn"].to_numpy(),
        color="#3a72b0",
        linewidth=1.5,
        label="cost / txn",
    )
    optimal_cost = float(
        cost_curve.loc[cost_curve["threshold"] == optimal_tau, "cost_per_txn"].iloc[0]
    )
    ax.axvline(
        optimal_tau,
        color="#c14242",
        linestyle="--",
        linewidth=1.5,
        label=f"optimal τ = {optimal_tau:.{_TAU_DIGITS}f} (${optimal_cost:.{_USD_DIGITS}f}/txn)",
    )
    # Find the closest swept threshold to the baseline; the baseline
    # τ=0.5 may not land on the linspace grid exactly.
    grid = cost_curve["threshold"].to_numpy()
    nearest_baseline_idx = int(np.argmin(np.abs(grid - baseline_tau)))
    baseline_cost = float(cost_curve["cost_per_txn"].iloc[nearest_baseline_idx])
    ax.axvline(
        baseline_tau,
        color="#3a8b3a",
        linestyle=":",
        linewidth=1.5,
        label=f"baseline τ = {baseline_tau:.2f} (${baseline_cost:.{_USD_DIGITS}f}/txn)",
    )
    ax.set_xlabel("decision threshold τ")
    ax.set_ylabel("expected cost per transaction (USD)")
    ax.set_title(
        f"Economic cost curve  "
        f"(savings @ optimal vs baseline = "
        f"${(baseline_cost - optimal_cost):.{_USD_DIGITS}f}/txn)"
    )
    ax.legend(loc="best")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# .env update helper.
# ---------------------------------------------------------------------


def _update_env_threshold(
    new_threshold: float,
    env_path: Path,
) -> Path:
    """Replace `DECISION_THRESHOLD=...` in `env_path`. Backup first.

    Returns the backup path. Raises `FileNotFoundError` if `env_path`
    is missing — explicit setup is required, not silent creation.

    Behaviour:
        - Writes `<env_path>.bak` (overwriting any prior backup).
        - Locates the line starting with `DECISION_THRESHOLD=` and
          replaces its value (preserves the rest of the file).
        - If the key is not found, appends it with a Sprint-4.4
          provenance comment.
    """
    if not env_path.is_file():
        raise FileNotFoundError(f"Expected .env at {env_path} — copy from .env.example first.")

    backup_path = env_path.with_name(env_path.name + _ENV_BACKUP_SUFFIX)
    original = env_path.read_text(encoding="utf-8")
    backup_path.write_text(original, encoding="utf-8")

    new_value = f"{new_threshold:.{_ENV_THRESHOLD_DIGITS}f}"
    new_line = f"{_ENV_THRESHOLD_KEY}={new_value}"

    lines = original.splitlines()
    found = False
    new_lines: list[str] = []
    for line in lines:
        if line.startswith(f"{_ENV_THRESHOLD_KEY}="):
            new_lines.append(new_line)
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append("# Updated by scripts/run_economic_evaluation.py (Sprint 4.4).")
        new_lines.append(new_line)

    # Preserve trailing newline if original had one.
    suffix = "\n" if original.endswith("\n") else ""
    env_path.write_text("\n".join(new_lines) + suffix, encoding="utf-8")
    _logger.info(
        "economic.env_threshold_updated",
        env_path=str(env_path),
        backup_path=str(backup_path),
        new_threshold=new_threshold,
    )
    return backup_path


# ---------------------------------------------------------------------
# Report rendering.
# ---------------------------------------------------------------------


def _format_dataframe_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Pandas → markdown pipe-table without a row index.

    Keeps the report self-contained (no `tabulate` dependency).
    """
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_markdown(index=False, floatfmt=".4f")


def _render_economic_evaluation_report(  # noqa: PLR0915 — linear markdown builder; mirrors train_lightgbm.py:_render_training_report
    result: EvaluationResult,
    out_path: Path,
) -> None:
    """Emit `reports/economic_evaluation.md`.

    Sections (per plan):
        1. Headline gates table
        2. Optimal threshold + baseline comparison
        3. Annual savings estimate
        4. Sensitivity table
        5. Stratified performance table
        6. Cost-curve figure
        7. Stratified heatmap figure
        8. .env update log
        9. Caveats
        10. Artefacts
        11. References
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    lines: list[str] = []
    lines.append("# Economic Evaluation Report — test set")
    lines.append("")
    lines.append("- **Generated by:** `scripts/run_economic_evaluation.py`")
    lines.append(f"- **Date (UTC):** {today}")
    lines.append(f"- **Test rows:** {result.n_test_rows:,}")
    lines.append(f"- **Portfolio assumption:** {result.portfolio_monthly:,} txns / month")
    lines.append("")

    # 1. Headline gates.
    lines.append("## Acceptance gates")
    lines.append("")
    lines.append("| Gate | Realised | Status |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Optimal τ in [{_OPTIMUM_BAND_LOW}, {_OPTIMUM_BAND_HIGH}] | "
        f"{result.optimal_threshold:.{_TAU_DIGITS}f} | "
        f"{'✅ PASS' if result.optimum_band_gate_pass else '❌ GAP'} |"
    )
    lines.append(
        f"| Annual savings ≥ ${_ANNUAL_SAVINGS_FLOOR_USD:,.0f} | "
        f"${result.annual_savings_usd:,.{_USD_DIGITS}f} | "
        f"{'✅ PASS' if result.annual_savings_gate_pass else '❌ GAP'} |"
    )
    lines.append(
        f"| Sensitivity spread < {_SENSITIVITY_SPREAD_CEILING} | "
        f"{result.sensitivity_spread:.{_TAU_DIGITS}f} | "
        f"{'✅ PASS' if result.sensitivity_stability_gate_pass else '❌ GAP'} |"
    )
    lines.append("")

    # 2. Optimal threshold + baseline comparison.
    lines.append("## Optimal threshold")
    lines.append("")
    lines.append(f"- **Cost-optimal τ:** `{result.optimal_threshold:.{_TAU_DIGITS}f}`")
    lines.append(
        f"- **Baseline τ (placeholder this script replaces):** "
        f"`{result.baseline_threshold:.2f}`"
    )
    lines.append(
        f"- **Per-txn cost at optimal:** " f"${result.cost_per_txn_at_optimal:.{_USD_DIGITS}f}"
    )
    lines.append(
        f"- **Per-txn cost at baseline:** " f"${result.cost_per_txn_at_baseline:.{_USD_DIGITS}f}"
    )
    delta = result.cost_per_txn_at_baseline - result.cost_per_txn_at_optimal
    pct = (
        (delta / result.cost_per_txn_at_baseline * 100.0)
        if result.cost_per_txn_at_baseline > 0
        else 0.0
    )
    lines.append(
        f"- **Per-txn savings (baseline - optimal):** " f"${delta:.{_USD_DIGITS}f}  ({pct:.1f} %)"
    )
    lines.append("")

    # 3. Annual savings.
    lines.append("## Annual savings estimate")
    lines.append("")
    lines.append("| Quantity | Value |")
    lines.append("|---|---|")
    lines.append(
        f"| Cost / txn @ τ_baseline | ${result.cost_per_txn_at_baseline:.{_USD_DIGITS}f} |"
    )
    lines.append(f"| Cost / txn @ τ_optimal  | ${result.cost_per_txn_at_optimal:.{_USD_DIGITS}f} |")
    lines.append(f"| Per-txn savings | ${delta:.{_USD_DIGITS}f} |")
    lines.append(f"| Monthly portfolio (txns) | {result.portfolio_monthly:,} |")
    lines.append(f"| Months per year | {_MONTHS_PER_YEAR} |")
    lines.append(f"| **Annual savings** | **${result.annual_savings_usd:,.{_USD_DIGITS}f}** |")
    lines.append(
        f"| Annual cost @ τ_baseline | ${result.annual_cost_at_baseline_usd:,.{_USD_DIGITS}f} |"
    )
    lines.append(
        f"| Annual cost @ τ_optimal | ${result.annual_cost_at_optimal_usd:,.{_USD_DIGITS}f} |"
    )
    lines.append("")
    lines.append(
        "Linear extrapolation: per-txn delta × monthly portfolio × 12. "
        "Scale linearly for a different portfolio volume."
    )
    lines.append("")

    # 4. Sensitivity table.
    lines.append("## Sensitivity to cost variation (±20 %)")
    lines.append("")
    lines.append(
        f"Spread of optimal τ across the {len(result.sensitivity)}-cell cost grid: "
        f"`{result.sensitivity_spread:.{_TAU_DIGITS}f}`. "
        f"Stability gate: < {_SENSITIVITY_SPREAD_CEILING} → "
        f"{'✅ PASS' if result.sensitivity_stability_gate_pass else '❌ GAP'}."
    )
    lines.append("")
    sens_sorted = result.sensitivity.sort_values("optimal_threshold")
    lines.append(f"### Top {_SENSITIVITY_TOP_N} (lowest τ*)")
    lines.append("")
    lines.append(_format_dataframe_markdown(sens_sorted.head(_SENSITIVITY_TOP_N)))
    lines.append("")
    lines.append(f"### Bottom {_SENSITIVITY_TOP_N} (highest τ*)")
    lines.append("")
    lines.append(_format_dataframe_markdown(sens_sorted.tail(_SENSITIVITY_TOP_N)))
    lines.append("")

    # 5. Stratified performance.
    lines.append("## Stratified performance")
    lines.append("")
    lines.append(
        "Per-segment AUC / PR-AUC / cost on the test set, evaluated at "
        f"τ = {result.optimal_threshold:.{_TAU_DIGITS}f} (the cost-optimal "
        f"value derived above). Month axis intentionally skipped; see Caveats."
    )
    lines.append("")
    lines.append(_format_dataframe_markdown(result.stratified))
    lines.append("")

    # 6. + 7. Figures.
    lines.append("## Figures")
    lines.append("")
    lines.append("### Cost curve")
    lines.append("")
    lines.append(f"![cost curve](figures/{result.cost_curve_figure_path.name})")
    lines.append("")
    lines.append("### Stratified heatmap")
    lines.append("")
    lines.append(f"![stratified heatmap](figures/{result.heatmap_figure_path.name})")
    lines.append("")

    # 8. .env update log.
    lines.append("## `.env` update")
    lines.append("")
    if result.env_updated:
        lines.append(
            f"- `DECISION_THRESHOLD` updated `{result.baseline_threshold:.2f}` → "
            f"`{result.optimal_threshold:.{_ENV_THRESHOLD_DIGITS}f}` in "
            f"`{result.env_path}` on {today}."
        )
        lines.append(f"- Backup: `{result.env_backup_path}`.")
        lines.append("- To roll back: `mv .env.bak .env`.")
    else:
        lines.append(
            f"- `--dry-run` / `--no-update-env` set; **no `.env` mutation performed**. "
            f"To apply the chosen τ, re-run without those flags or edit "
            f"manually: `{_ENV_THRESHOLD_KEY}={result.optimal_threshold:.{_ENV_THRESHOLD_DIGITS}f}`."
        )
    lines.append("")

    # 9. Caveats (the explicit section the spec asks for).
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- **Cost values are industry medians** per CLAUDE.md §8 + "
        "`configs/economic_defaults.yaml`. The ±20 % sensitivity grid above "
        "bounds the impact, but a deployer with materially different cost "
        "economics MUST override `.env` and re-run this script."
    )
    lines.append(
        "- **Test set is one temporal snapshot.** Production drift is not "
        "measured here; that's Sprint 6 monitoring territory. The chosen "
        "τ assumes the test-set fraud profile holds in production."
    )
    lines.append(
        f"- **Portfolio default is {result.portfolio_monthly:,} txns / month.** "
        "Annual savings scale linearly for a different volume; the report "
        "carries the per-txn delta separately so a reader can re-do the "
        "arithmetic."
    )
    lines.append(
        "- **Calibration is the load-bearing dependency** (ADR 0003). A "
        "calibration regression invalidates this τ; Sprint 3.3.c's isotonic "
        "calibrator must remain accurate for the cost surface to be faithful."
    )
    lines.append(
        "- **Per-segment thresholds may yield additional savings** beyond "
        "the global τ. The stratified heatmap above visualises per-stratum "
        "skew; per-segment optimisation is Sprint 5+ territory."
    )
    lines.append(
        "- **Month axis intentionally skipped** — Tier-5 parquet drops "
        "`timestamp` (`build_features_all_tiers.py:110-111`) and the test "
        "set is approximately one calendar month, so within-test month "
        "stratification is degenerate. Cross-month drift is Sprint 6 "
        "monitoring territory."
    )
    lines.append(
        "- **Savings are computed against τ = 0.5** (the placeholder this "
        'script replaces), NOT against "no model at all." Model value '
        "vs threshold-optimisation value are different questions; this "
        "report answers the latter."
    )
    lines.append("")

    # 10. Artefacts.
    lines.append("## Artefacts")
    lines.append("")
    lines.append(f"- This report: `{result.report_path}`")
    lines.append(f"- Cost curve: `{result.cost_curve_figure_path}`")
    lines.append(f"- Stratified heatmap: `{result.heatmap_figure_path}`")
    if result.env_updated and result.env_path is not None:
        lines.append(f"- `.env` (updated): `{result.env_path}`")
        lines.append(f"- `.env.bak` (rollback): `{result.env_backup_path}`")
    lines.append(
        "- Model A: `models/sprint3/lightgbm_model.joblib` "
        "(produced by `scripts/train_lightgbm.py`)"
    )
    lines.append(
        "- Calibrator: `models/sprint3/calibrator.joblib` "
        "(produced by `scripts/train_lightgbm.py` via `select_calibration_method`)"
    )
    lines.append("")

    # 11. References.
    lines.append("## References")
    lines.append("")
    lines.append("- `CLAUDE.md` §8 — Business-Logic Constants (cost defaults).")
    lines.append("- `configs/economic_defaults.yaml` — cost provenance.")
    lines.append("- `docs/ADR/0003-economic-threshold.md` — cost-based τ over F1 / AUC.")
    lines.append(
        "- `src/fraud_engine/evaluation/economic.py` — `EconomicCostModel` " "(Sprint 4.1)."
    )
    lines.append(
        "- `src/fraud_engine/evaluation/stratified.py` — `StratifiedEvaluator` " "(Sprint 4.2)."
    )
    lines.append(
        "- `sprints/sprint_4/prompt_4_1_report.md`, `prompt_4_2_report.md`, "
        "`prompt_4_3_report.md` — Sprint 4 build-up."
    )
    lines.append(
        "- `reports/model_a_training_report.md` — Model A's training " "report (Sprint 3.3.d)."
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------


def run_economic_evaluation(  # noqa: PLR0913, PLR0915 — single-file orchestration; the knobs match the CLI surface
    *,
    settings: Settings,
    portfolio_monthly: int = _DEFAULT_PORTFOLIO_MONTHLY,
    sample_size: int | None = None,
    dry_run: bool = False,
    update_env: bool = True,
    report_path: Path = _REPORT_PATH,
    cost_curve_figure_path: Path = _COST_CURVE_FIG_PATH,
    heatmap_figure_path: Path = _HEATMAP_FIG_PATH,
    env_path: Path | None = None,
    random_state: int | None = None,
) -> EvaluationResult:
    """End-to-end economic evaluation on the held-out test set.

    Args:
        settings: Project settings (paths + seeds).
        portfolio_monthly: Transaction volume / month for the annual
            savings extrapolation. Default 1,000,000.
        sample_size: If not None, stratified-subsample the test frame
            to ~this many rows. Used by `--quick` and the integration
            test smoke.
        dry_run: If True, run + write report but skip `.env` mutation.
        update_env: If False, also skips `.env` mutation. Independent
            of dry_run for finer control.
        report_path: Where to write the markdown report.
        cost_curve_figure_path: Where to write the cost-curve PNG.
        heatmap_figure_path: Where to write the stratified-heatmap PNG.
        env_path: Path to `.env`. Defaults to `<project_root>/.env`.
        random_state: Seed for the smoke subsample. Defaults to
            `settings.seed`.

    Returns:
        `EvaluationResult` with all metrics + artefact paths + gate flags.
    """
    seed = random_state if random_state is not None else settings.seed
    if env_path is None:
        env_path = _PROJECT_ROOT / _ENV_FILENAME

    # --- Load test parquet + optionally subsample ---
    test = _load_test_parquet(settings.processed_dir)
    if sample_size is not None:
        test = _stratified_subsample(test, sample_size, seed=seed)
    n_test_rows = len(test)

    feature_cols = _select_features(test)
    test_x = test[feature_cols]
    test_y = test["isFraud"].to_numpy()

    _logger.info(
        "economic.test_loaded",
        n_test=n_test_rows,
        n_features=len(feature_cols),
        sample_size=sample_size,
    )

    # --- Load model + calibrator ---
    model, calibrator = _load_model_artefacts(settings.models_dir)
    _logger.info(
        "economic.artefacts_loaded",
        model_path=str(settings.models_dir / _MODELS_SUBDIR / "lightgbm_model.joblib"),
        calibrator_path=str(settings.models_dir / _MODELS_SUBDIR / _CALIBRATOR_FILENAME),
    )

    # --- Score: raw → calibrated ---
    proba_raw = model.predict_proba(test_x)[:, 1]
    proba_cal = calibrator.transform(proba_raw)
    _logger.info(
        "economic.scores_computed",
        n_rows=n_test_rows,
        proba_min=float(np.min(proba_cal)),
        proba_max=float(np.max(proba_cal)),
    )

    # --- Threshold + sensitivity ---
    cost_model = EconomicCostModel()
    optimal_tau, cost_curve = cost_model.optimize_threshold(test_y, proba_cal)
    sensitivity = cost_model.sensitivity_analysis(test_y, proba_cal)

    # --- Stratified evaluator at the chosen τ ---
    stratified_evaluator = StratifiedEvaluator(
        cost_model=cost_model,
        threshold=optimal_tau,
    )
    stratified = stratified_evaluator.evaluate(test_y, proba_cal, test, month=None)

    # --- Plot heatmap → save ---
    heatmap_figure_path.parent.mkdir(parents=True, exist_ok=True)
    ax = stratified_evaluator.plot_heatmap(stratified)
    # `Axes.figure` is typed as `Figure | SubFigure` in the matplotlib
    # stubs, but `plot_heatmap` always produces a top-level Figure via
    # `plt.subplots()`. Cast for mypy.
    heatmap_fig = cast(Figure, ax.figure)
    heatmap_fig.savefig(heatmap_figure_path, dpi=150, bbox_inches="tight")
    plt.close(heatmap_fig)

    # --- Plot cost curve → save ---
    _save_cost_curve_figure(
        cost_curve,
        optimal_tau=optimal_tau,
        baseline_tau=_BASELINE_THRESHOLD,
        out_path=cost_curve_figure_path,
    )

    # --- Annual savings ---
    cost_per_txn_at_optimal = float(
        cost_curve.loc[cost_curve["threshold"] == optimal_tau, "cost_per_txn"].iloc[0]
    )
    # Baseline τ=0.5 may not land exactly on the linspace grid; pick
    # the swept threshold closest to 0.5 for the comparison.
    grid = cost_curve["threshold"].to_numpy()
    nearest_baseline_idx = int(np.argmin(np.abs(grid - _BASELINE_THRESHOLD)))
    cost_per_txn_at_baseline = float(cost_curve["cost_per_txn"].iloc[nearest_baseline_idx])
    annual_volume = portfolio_monthly * _MONTHS_PER_YEAR
    annual_cost_at_optimal_usd = cost_per_txn_at_optimal * annual_volume
    annual_cost_at_baseline_usd = cost_per_txn_at_baseline * annual_volume
    annual_savings_usd = annual_cost_at_baseline_usd - annual_cost_at_optimal_usd

    # --- Sensitivity stability ---
    sensitivity_spread = float(
        sensitivity["optimal_threshold"].max() - sensitivity["optimal_threshold"].min()
    )

    # --- Gate evaluation (reported, not enforced) ---
    optimum_band_gate_pass = bool(_OPTIMUM_BAND_LOW <= optimal_tau <= _OPTIMUM_BAND_HIGH)
    annual_savings_gate_pass = bool(annual_savings_usd >= _ANNUAL_SAVINGS_FLOOR_USD)
    sensitivity_stability_gate_pass = bool(sensitivity_spread < _SENSITIVITY_SPREAD_CEILING)

    _logger.info(
        "economic.evaluation_done",
        optimal_threshold=optimal_tau,
        baseline_threshold=_BASELINE_THRESHOLD,
        annual_savings_usd=annual_savings_usd,
        sensitivity_spread=sensitivity_spread,
        optimum_band_gate_pass=optimum_band_gate_pass,
        annual_savings_gate_pass=annual_savings_gate_pass,
        sensitivity_stability_gate_pass=sensitivity_stability_gate_pass,
    )

    # --- .env update (per dry_run / update_env) ---
    env_updated = False
    env_backup_path: Path | None = None
    if update_env and not dry_run:
        env_backup_path = _update_env_threshold(optimal_tau, env_path)
        env_updated = True
    else:
        _logger.info(
            "economic.env_update_skipped",
            reason="dry_run" if dry_run else "no_update_env",
        )

    result = EvaluationResult(
        optimal_threshold=optimal_tau,
        baseline_threshold=_BASELINE_THRESHOLD,
        cost_curve=cost_curve,
        sensitivity=sensitivity,
        stratified=stratified,
        cost_per_txn_at_optimal=cost_per_txn_at_optimal,
        cost_per_txn_at_baseline=cost_per_txn_at_baseline,
        annual_savings_usd=annual_savings_usd,
        annual_cost_at_optimal_usd=annual_cost_at_optimal_usd,
        annual_cost_at_baseline_usd=annual_cost_at_baseline_usd,
        sensitivity_spread=sensitivity_spread,
        n_test_rows=n_test_rows,
        portfolio_monthly=portfolio_monthly,
        optimum_band_gate_pass=optimum_band_gate_pass,
        annual_savings_gate_pass=annual_savings_gate_pass,
        sensitivity_stability_gate_pass=sensitivity_stability_gate_pass,
        report_path=report_path,
        cost_curve_figure_path=cost_curve_figure_path,
        heatmap_figure_path=heatmap_figure_path,
        env_updated=env_updated,
        env_path=env_path if env_updated else None,
        env_backup_path=env_backup_path,
    )

    _render_economic_evaluation_report(result, report_path)
    return result


# ---------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------


@click.command()
@click.option(
    "--portfolio-monthly",
    type=int,
    default=_DEFAULT_PORTFOLIO_MONTHLY,
    show_default=True,
    help="Transactions per month for the annual savings extrapolation.",
)
@click.option(
    "--quick",
    is_flag=True,
    default=False,
    help="5K-row stratified subsample of the test frame; faster smoke run.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Run + write report; skip the .env mutation.",
)
@click.option(
    "--no-update-env",
    is_flag=True,
    default=False,
    help="Same as --dry-run for the .env mutation only (still writes the report).",
)
def main(
    portfolio_monthly: int,
    quick: bool,
    dry_run: bool,
    no_update_env: bool,
) -> None:
    """Sprint 4 economic-evaluation runner. See module docstring for full flow."""
    settings = get_settings()
    settings.ensure_directories()

    sample_size: int | None = None
    if quick:
        sample_size = _SMOKE_SAMPLE_SIZE
        # In quick mode, also drop the portfolio assumption to a smoke
        # value so the report's annual-savings number doesn't claim to
        # be production-relevant.
        if portfolio_monthly == _DEFAULT_PORTFOLIO_MONTHLY:
            portfolio_monthly = _SMOKE_PORTFOLIO_MONTHLY

    update_env = not no_update_env

    result = run_economic_evaluation(
        settings=settings,
        portfolio_monthly=portfolio_monthly,
        sample_size=sample_size,
        dry_run=dry_run,
        update_env=update_env,
    )

    click.echo(click.style("\nrun_economic_evaluation: COMPLETE", fg="green", bold=True))
    click.echo(
        f"  optimal τ          = {result.optimal_threshold:.{_TAU_DIGITS}f}  "
        f"({'PASS' if result.optimum_band_gate_pass else 'GAP'} vs "
        f"[{_OPTIMUM_BAND_LOW}, {_OPTIMUM_BAND_HIGH}])"
    )
    click.echo(f"  baseline τ         = {result.baseline_threshold:.2f}")
    click.echo(
        f"  cost / txn         = ${result.cost_per_txn_at_optimal:.{_USD_DIGITS}f} "
        f"(baseline ${result.cost_per_txn_at_baseline:.{_USD_DIGITS}f})"
    )
    click.echo(
        f"  annual savings     = ${result.annual_savings_usd:,.{_USD_DIGITS}f}  "
        f"({'PASS' if result.annual_savings_gate_pass else 'GAP'} vs "
        f"${_ANNUAL_SAVINGS_FLOOR_USD:,.0f})"
    )
    click.echo(
        f"  sensitivity spread = {result.sensitivity_spread:.{_TAU_DIGITS}f}  "
        f"({'PASS' if result.sensitivity_stability_gate_pass else 'GAP'} vs "
        f"<{_SENSITIVITY_SPREAD_CEILING})"
    )
    click.echo(f"  n_test_rows        = {result.n_test_rows:,}")
    click.echo(f"  portfolio monthly  = {result.portfolio_monthly:,}")
    click.echo(f"  report             = {result.report_path}")
    click.echo(f"  cost-curve figure  = {result.cost_curve_figure_path}")
    click.echo(f"  heatmap figure     = {result.heatmap_figure_path}")
    if result.env_updated:
        click.echo(f"  .env updated       = {result.env_path}")
        click.echo(f"  .env backup        = {result.env_backup_path}")
    else:
        click.echo("  .env updated       = SKIPPED (--dry-run / --no-update-env)")


if __name__ == "__main__":
    main()
