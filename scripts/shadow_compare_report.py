"""Weekly champion-vs-challenger shadow comparison report.

Sprint 5 prompt 5.2.c. Click CLI that:

1. Loads N predictions with both champion + shadow scores (either
   synthesised in `--sample` mode, or — once 5.x ships chargeback
   ingestion — joined from Postgres `predictions` + the structlog
   `shadow.scored` JSONL stream).
2. Calls `ShadowComparison(predictions, costs).run()` to compute
   agreement, correlation, cost, and bootstrap significance.
3. Renders a markdown report with the per-criterion verdict
   (PROMOTE / DO NOT PROMOTE) to `reports/shadow_compare_<date>.md`.
4. Prints a compact stdout summary.

Modes:
    --sample             : generate 1000 synthetic predictions for
                           workflow demonstration. The spec-validation
                           invocation. Writes a representative report.
    --source PATH        : (Sprint 5.x) path to a JSONL of structlog
                           shadow.scored events.
    --labels PATH        : (Sprint 5.x) path to a parquet with
                           ground-truth labels (TransactionID + isFraud).

Output:
    reports/shadow_compare_<date>.md  (default; overridable via --output)
    stdout: one-line verdict summary

Cross-references:
    - `src/fraud_engine/evaluation/shadow_compare.py` — the analysis
      primitive this script wraps.
    - `scripts/run_economic_evaluation.py` — Sprint 4.4 Click +
      markdown-report template this script mirrors.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd

from fraud_engine.config.settings import get_settings
from fraud_engine.evaluation.shadow_compare import (
    ComparisonReport,
    EconomicCosts,
    ShadowComparison,
)
from fraud_engine.utils.logging import configure_logging, get_logger

# ---------------------------------------------------------------------
# Sample-mode constants. Tuned so the synthetic data produces a
# boundary-case verdict (some criteria pass, others fail) — useful
# demo of the per-criterion reporting.
# ---------------------------------------------------------------------

_SAMPLE_N = 1000
_SAMPLE_LABELED_FRACTION = 0.20  # 200 of 1000 labeled — typical chargeback rate
_SAMPLE_SHADOW_NOISE_WEIGHT = 0.15  # shadow = 0.85 * champion + 0.15 * noise
_SAMPLE_SEED = 7  # different from comparison seed to avoid coupling
_SAMPLE_DECISION_THRESHOLD = 0.080000  # post-Sprint-4.4 cost-optimal value

# Markdown report template constants.
_REPORTS_DIR = Path("reports")
_DATE_FORMAT = "%Y-%m-%d"

_logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Sample-mode synthetic data generator.
# ---------------------------------------------------------------------


def _generate_sample_predictions(
    n: int = _SAMPLE_N,
    labeled_fraction: float = _SAMPLE_LABELED_FRACTION,
    shadow_noise_weight: float = _SAMPLE_SHADOW_NOISE_WEIGHT,
    decision_threshold: float = _SAMPLE_DECISION_THRESHOLD,
    seed: int = _SAMPLE_SEED,
) -> pd.DataFrame:
    """Generate a synthetic predictions DataFrame for --sample mode.

    Champion scores are uniform; shadow scores are correlated with
    champion plus uniform noise. Decisions derived from the
    decision_threshold. ~labeled_fraction of rows have an `is_fraud`
    label whose probability is sigmoid-shaped on the champion score
    (so the scores are noisily predictive of the label).

    Args:
        n: Total predictions.
        labeled_fraction: Fraction of N to assign labels to.
        shadow_noise_weight: Mixing weight for shadow noise. 0.15 →
            shadow ≈ 0.85 * champion + 0.15 * noise → correlation ~0.85.
        decision_threshold: Post-Sprint-4.4 cost-optimal value (0.080).
        seed: RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    champion_scores = rng.uniform(0.0, 1.0, n)
    shadow_scores = (1 - shadow_noise_weight) * champion_scores + shadow_noise_weight * rng.uniform(
        0.0, 1.0, n
    )
    champion_decisions = np.where(champion_scores >= decision_threshold, "block", "allow")
    shadow_decisions = np.where(shadow_scores >= decision_threshold, "block", "allow")

    df = pd.DataFrame(
        {
            "request_id": [f"sample-{i:05d}" for i in range(n)],
            "champion_score": champion_scores,
            "shadow_score": shadow_scores,
            "champion_decision": champion_decisions,
            "shadow_decision": shadow_decisions,
        }
    )

    # Labels — assign to a random subset. P(fraud) is sigmoid-shaped on
    # the champion score so labels are noisily correlated with the
    # scores (mimics the real-world relationship between predicted
    # probability and chargeback rate).
    n_labeled = int(n * labeled_fraction)
    labeled_idx = rng.choice(n, size=n_labeled, replace=False)
    is_fraud_full: np.ndarray[None, np.dtype[np.float64]] = np.full(n, np.nan, dtype=np.float64)
    p_fraud = 1.0 / (1.0 + np.exp(-(champion_scores[labeled_idx] * 6.0 - 3.0)))
    is_fraud_full[labeled_idx] = (rng.uniform(0.0, 1.0, n_labeled) < p_fraud).astype(int)
    df["is_fraud"] = is_fraud_full
    return df


# ---------------------------------------------------------------------
# Production-mode data loader (stubbed; real implementation in Sprint 5.x).
# ---------------------------------------------------------------------


def _load_production_predictions(
    source: Path,
    labels: Path | None,  # noqa: ARG001 — Sprint 5.x will use this
) -> pd.DataFrame:
    """Stub for the production data-load path.

    The real implementation will:
      1. Parse `source` (JSONL of structlog `shadow.scored` events).
      2. Optionally join `labels` (parquet with TransactionID + isFraud)
         on the predictions' `txn_id` for the cost path.
      3. Return a DataFrame matching the ShadowComparison contract.

    Sprint 5.2.c MVP raises NotImplementedError here — the production
    flow lands when chargeback ingestion (Sprint 5.x) is in.
    """
    raise NotImplementedError(
        f"Production-mode data loading is Sprint 5.x scope. The current "
        f"5.2.c MVP only supports --sample. Source path: {source}"
    )


# ---------------------------------------------------------------------
# Markdown report renderer.
# ---------------------------------------------------------------------


def _render_markdown_report(  # noqa: PLR0912, PLR0915 — single-purpose markdown renderer; splitting into per-section helpers would scatter the report layout across multiple functions
    report: ComparisonReport,
    *,
    mode: str,
    n_total: int,
    costs: EconomicCosts,
    generated_at: datetime,
) -> str:
    """Render a ComparisonReport into a markdown document.

    Mirrors `scripts/run_economic_evaluation.py`'s style: header table,
    metrics tables, verdict callout, references.
    """
    lines: list[str] = []
    lines.append("# Shadow Comparison Report — champion vs challenger")
    lines.append("")
    lines.append(f"**Generated:** {generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**Mode:** `{mode}`")
    lines.append(f"**N predictions:** {n_total:,}")
    if report.n_labeled is not None:
        lines.append(f"**N labeled (cost path):** {report.n_labeled:,}")
    else:
        lines.append("**N labeled:** 0 (no `is_fraud` column → cost path skipped)")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if report.verdict.promote:
        lines.append("> ✅ **PROMOTE** — all three promotion criteria met.")
    else:
        lines.append("> ⛔ **DO NOT PROMOTE** — at least one criterion failed.")
    lines.append("")
    lines.append("| Criterion | Threshold | Outcome |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Agreement rate | > 0.85 | {report.agreement_rate:.4f} — "
        f"{'PASS' if report.verdict.agreement_pass else 'FAIL'} |"
    )
    if report.cost_improvement is not None:
        lines.append(
            f"| Cost improvement | > 2% | {report.cost_improvement * 100:.2f}% — "
            f"{'PASS' if report.verdict.cost_improvement_pass else 'FAIL'} |"
        )
    else:
        lines.append("| Cost improvement | > 2% | n/a (labels required) — FAIL |")
    if report.bootstrap_p_value is not None:
        lines.append(
            f"| p-value (bootstrap, two-sided) | < 0.05 | "
            f"{report.bootstrap_p_value:.4f} — "
            f"{'PASS' if report.verdict.p_value_pass else 'FAIL'} |"
        )
    else:
        lines.append("| p-value | < 0.05 | n/a (labels required) — FAIL |")
    lines.append("")

    lines.append("## Distribution metrics (population-level)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Agreement rate | {report.agreement_rate:.4f} |")
    lines.append(f"| Score correlation (Pearson) | {report.score_correlation:.4f} |")
    lines.append("")

    if report.n_labeled and report.n_labeled > 0:
        lines.append(f"## Cost analysis (labeled subset, N={report.n_labeled:,})")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        if report.champion_cost_per_txn is not None:
            lines.append(f"| Champion cost per txn | ${report.champion_cost_per_txn:.4f} |")
        if report.shadow_cost_per_txn is not None:
            lines.append(f"| Shadow cost per txn | ${report.shadow_cost_per_txn:.4f} |")
        if report.cost_improvement is not None:
            lines.append(f"| Relative improvement | {report.cost_improvement * 100:.2f}% |")
        if report.bootstrap_mean_diff is not None:
            lines.append(
                f"| Bootstrap mean diff (champion - shadow) | "
                f"${report.bootstrap_mean_diff:.4f} |"
            )
        if report.bootstrap_ci_95 is not None:
            ci_low, ci_high = report.bootstrap_ci_95
            lines.append(f"| Bootstrap 95% CI | (${ci_low:.4f}, ${ci_high:.4f}) |")
        if report.bootstrap_p_value is not None:
            lines.append(f"| Bootstrap two-sided p-value | {report.bootstrap_p_value:.4f} |")
        lines.append("")

    lines.append("## Cost matrix (per CLAUDE.md §8)")
    lines.append("")
    lines.append("| Outcome | USD cost |")
    lines.append("|---|---|")
    lines.append(f"| Missed fraud (FN) | ${costs.fraud_cost:.2f} |")
    lines.append(f"| False-positive block (FP) | ${costs.fp_cost:.2f} |")
    lines.append(f"| True-positive block (TP, investigation) | ${costs.tp_cost:.2f} |")
    lines.append("| Correct allow (TN) | $0.00 |")
    lines.append("")

    lines.append("## Reasons (per criterion)")
    lines.append("")
    for reason in report.verdict.reasons:
        lines.append(f"- {reason}")
    lines.append("")

    lines.append("## Out-of-scope notes")
    lines.append("")
    lines.append(
        "- Production-mode data join (Postgres `predictions` + JSONL "
        "`shadow.scored`) is Sprint 5.x scope; today's CLI supports "
        "`--sample` for workflow demonstration only."
    )
    lines.append(
        "- Per-segment shadow comparison (per ProductCD / DeviceType / etc.) "
        "could reuse `StratifiedEvaluator` from Sprint 4.2; deferred."
    )
    lines.append("")

    lines.append("## References")
    lines.append("")
    lines.append("- `src/fraud_engine/evaluation/shadow_compare.py` — analysis primitive")
    lines.append("- `src/fraud_engine/api/shadow.py` — Sprint 5.2.b shadow source")
    lines.append("- `scripts/create_predictions_table.sql` — Sprint 5.2.a audit-log schema")
    lines.append("- CLAUDE.md §3 (production-API stack), §8 (cost defaults)")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Click CLI.
# ---------------------------------------------------------------------


@click.command(
    context_settings={"max_content_width": 100, "show_default": True},
    help="Generate a champion-vs-challenger shadow comparison report.",
)
@click.option(
    "--sample",
    "sample",
    is_flag=True,
    default=False,
    help="Generate synthetic data (1000 predictions, 200 labeled). "
    "The spec-validation invocation. Default: production mode (5.x stub).",
)
@click.option(
    "--source",
    type=click.Path(path_type=Path),
    default=None,
    help="(Sprint 5.x) Path to JSONL of structlog `shadow.scored` events.",
)
@click.option(
    "--labels",
    type=click.Path(path_type=Path),
    default=None,
    help="(Sprint 5.x) Path to parquet with TransactionID + isFraud labels.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help=f"Override the report output path. Default: {_REPORTS_DIR}/shadow_compare_<date>.md.",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="stdlib log level.",
)
def main(
    sample: bool,
    source: Path | None,
    labels: Path | None,
    output: Path | None,
    log_level: str,
) -> None:
    """CLI entry. See module docstring for the full design rationale."""
    os.environ["LOG_LEVEL"] = log_level.upper()
    get_settings.cache_clear()
    settings = get_settings()
    configure_logging(pipeline_name="shadow_compare_report")

    costs = EconomicCosts(
        fraud_cost=settings.fraud_cost_usd,
        fp_cost=settings.fp_cost_usd,
        tp_cost=settings.tp_cost_usd,
    )

    if sample:
        _logger.info("shadow_compare_report.mode_sample", n=_SAMPLE_N)
        predictions = _generate_sample_predictions()
        mode_label = "sample (synthetic)"
    elif source is not None:
        _logger.info("shadow_compare_report.mode_production", source=str(source))
        predictions = _load_production_predictions(source, labels)
        mode_label = f"production (source={source})"
    else:
        click.echo("ERROR: must specify either --sample or --source. See --help.", err=True)
        sys.exit(2)

    comparison = ShadowComparison(predictions, costs)
    report = comparison.run()
    generated_at = datetime.now(tz=UTC)

    # Render markdown.
    md = _render_markdown_report(
        report,
        mode=mode_label,
        n_total=len(predictions),
        costs=costs,
        generated_at=generated_at,
    )

    if output is None:
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output = _REPORTS_DIR / f"shadow_compare_{generated_at.strftime(_DATE_FORMAT)}.md"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md, encoding="utf-8")
    _logger.info("shadow_compare_report.written", path=str(output), bytes=len(md))

    # Stdout summary.
    click.echo("")
    click.echo("===== shadow comparison summary =====")
    click.echo(f"  mode               : {mode_label}")
    click.echo(f"  n_total            : {report.n_total:,}")
    click.echo(f"  n_labeled          : {report.n_labeled or 0:,}")
    click.echo(f"  agreement_rate     : {report.agreement_rate:.4f}")
    click.echo(f"  score_correlation  : {report.score_correlation:.4f}")
    if report.cost_improvement is not None:
        click.echo(f"  cost_improvement   : {report.cost_improvement * 100:+.2f}%")
    if report.bootstrap_p_value is not None:
        click.echo(f"  bootstrap p-value  : {report.bootstrap_p_value:.4f}")
    click.echo("-------------------------------------")
    click.echo(f"  VERDICT            : {report.verdict.summary}")
    click.echo("-------------------------------------")
    click.echo(f"  report written to  : {output}")


if __name__ == "__main__":
    main()
