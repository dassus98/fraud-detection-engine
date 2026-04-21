"""Generate a custom HTML + JSON profile for the IEEE-CIS raw data.

Reads the merged DataFrame via `RawDataLoader` (un-optimised so null
counts match the source CSV) and emits two artefacts under `reports/`:

    raw_profile.html          — human-readable report, no JS / no fonts
    raw_profile_summary.json  — machine-readable top-line stats

The report covers:
    - Row / column counts, memory footprint, schema version
    - TransactionDT temporal span (seconds and derived day span)
    - Fraud rate overall + stratified by `ProductCD`
    - Identity coverage (share of transactions with any id_* present)
    - Missingness per column (sorted desc)
    - Unique-value counts for categorical / object columns
    - Numeric summary (count, mean, std, min, p50, p95, max) per
      numeric column

Business rationale:
    A one-page profile with the handful of indicators a reviewer
    actually looks at is more useful than a 12 MB ydata-profiling
    dump. The numbers rendered here drive every decision in Sprints
    1-4: the temporal split boundary, the thresholding baseline, the
    missing-aware LightGBM config, and the low-coverage strategy for
    device features.

Trade-offs considered:
    - We hand-roll the HTML to avoid a runtime JS dependency (the
      report is often opened in a sandboxed review tool) and to keep
      diff reviewability — any change to the page is visible in git.
    - We profile the *merged* frame, not individual tables, because
      Sprint 2 onwards consumes only the merged shape; per-table
      stats are derivable from merged + a single ProductCD filter.
    - `unique` counts on wide V* columns are expensive (~10s); we
      gate those behind a numeric-kind check so only object /
      categorical columns pay that cost.

Usage:
    uv run python scripts/profile_raw.py
    uv run python scripts/profile_raw.py --reports-dir custom/path
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd

from fraud_engine.config.settings import get_settings
from fraud_engine.data.loader import RawDataLoader
from fraud_engine.schemas.raw import SCHEMA_VERSION
from fraud_engine.utils.logging import configure_logging, get_logger

# Null-rate above which a column is flagged on the summary card. The
# 50% cut is conventional for IEEE-CIS reviews; columns above it are
# candidates for either drop or a "missing as signal" indicator.
_HIGH_NULL_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class Summary:
    """Top-line indicators shown at the top of the report."""

    rows: int
    cols: int
    memory_mb: float
    schema_version: int
    tx_dt_min: int
    tx_dt_max: int
    tx_dt_span_days: float
    fraud_rate_overall: float
    fraud_rate_by_product: dict[str, float]
    identity_coverage: float
    cols_above_50pct_null: int


def _compute_summary(merged: pd.DataFrame) -> Summary:
    """Compute the high-level indicators for the summary card.

    Args:
        merged: The merged DataFrame from `RawDataLoader.load_merged`.

    Returns:
        A `Summary` dataclass. `tx_dt_span_days` divides the raw
        `TransactionDT` seconds by 86400.
    """
    tx_min = int(merged["TransactionDT"].min())
    tx_max = int(merged["TransactionDT"].max())
    fraud_by_product = merged.groupby("ProductCD", observed=True)["isFraud"].mean().to_dict()
    # Identity coverage: any id_* present means the transaction had a
    # matching identity row during the left-join.
    id_cols = [c for c in merged.columns if c.startswith("id_")]
    if id_cols:
        has_id = merged[id_cols].notna().any(axis=1)
        coverage = float(has_id.mean())
    else:
        coverage = 0.0
    null_rate = merged.isna().mean()
    return Summary(
        rows=int(merged.shape[0]),
        cols=int(merged.shape[1]),
        memory_mb=round(merged.memory_usage(deep=True).sum() / (1024**2), 2),
        schema_version=SCHEMA_VERSION,
        tx_dt_min=tx_min,
        tx_dt_max=tx_max,
        tx_dt_span_days=round((tx_max - tx_min) / 86400.0, 2),
        fraud_rate_overall=float(merged["isFraud"].mean()),
        fraud_rate_by_product={str(k): float(v) for k, v in fraud_by_product.items()},
        identity_coverage=coverage,
        cols_above_50pct_null=int((null_rate > _HIGH_NULL_THRESHOLD).sum()),
    )


def _missingness_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows for the missingness table, sorted by null rate desc."""
    null_rate = df.isna().mean().sort_values(ascending=False)
    total = int(df.shape[0])
    return [
        {
            "column": col,
            "null_rate": float(null_rate[col]),
            "null_count": int(null_rate[col] * total),
            "dtype": str(df[col].dtype),
        }
        for col in null_rate.index
    ]


def _unique_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows for the unique-counts table (object + category cols only)."""
    out: list[dict[str, Any]] = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            n_unique = int(df[col].nunique(dropna=True))
            out.append(
                {
                    "column": col,
                    "n_unique": n_unique,
                    "dtype": str(dtype),
                }
            )
    out.sort(key=lambda row: row["n_unique"], reverse=True)
    return out


def _numeric_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Rows for the numeric summary (every numeric column, one row each)."""
    out: list[dict[str, Any]] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        quantiles = series.quantile([0.5, 0.95])
        out.append(
            {
                "column": col,
                "count": int(series.shape[0]),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "p50": float(quantiles.loc[0.5]),
                "p95": float(quantiles.loc[0.95]),
                "max": float(series.max()),
            }
        )
    return out


def _fmt_float(value: float, places: int = 4) -> str:
    """Format a float consistently across tables."""
    return f"{value:.{places}f}"


def _render_html(
    summary: Summary,
    missingness: list[dict[str, Any]],
    uniques: list[dict[str, Any]],
    numerics: list[dict[str, Any]],
) -> str:
    """Render all four sections into a single self-contained HTML doc.

    The HTML is intentionally minimal — no JS, no external fonts.
    Styles live in an inline <style> block so the file opens the same
    way in a sandboxed viewer as it does in a browser.

    Args:
        summary: Top-line stats.
        missingness: Rows for the missingness table.
        uniques: Rows for the unique-counts table.
        numerics: Rows for the numeric summary.

    Returns:
        A complete HTML document as a string.
    """
    miss_rows = "\n".join(
        "<tr><td>{col}</td><td class='num'>{rate}</td>"
        "<td class='num'>{count:,}</td><td>{dtype}</td></tr>".format(
            col=escape(str(row["column"])),
            rate=_fmt_float(row["null_rate"]),
            count=row["null_count"],
            dtype=escape(row["dtype"]),
        )
        for row in missingness
    )
    uniq_rows = "\n".join(
        "<tr><td>{col}</td><td class='num'>{n:,}</td><td>{dtype}</td></tr>".format(
            col=escape(str(row["column"])),
            n=row["n_unique"],
            dtype=escape(row["dtype"]),
        )
        for row in uniques
    )
    num_rows = "\n".join(
        (
            "<tr><td>{col}</td>"
            "<td class='num'>{count:,}</td>"
            "<td class='num'>{mean}</td>"
            "<td class='num'>{std}</td>"
            "<td class='num'>{mn}</td>"
            "<td class='num'>{p50}</td>"
            "<td class='num'>{p95}</td>"
            "<td class='num'>{mx}</td></tr>"
        ).format(
            col=escape(str(row["column"])),
            count=row["count"],
            mean=_fmt_float(row["mean"], 4),
            std=_fmt_float(row["std"], 4),
            mn=_fmt_float(row["min"], 4),
            p50=_fmt_float(row["p50"], 4),
            p95=_fmt_float(row["p95"], 4),
            mx=_fmt_float(row["max"], 4),
        )
        for row in numerics
    )
    product_rows = "\n".join(
        f"<tr><td>{escape(code)}</td><td class='num'>{_fmt_float(rate)}</td></tr>"
        for code, rate in sorted(summary.fraud_rate_by_product.items())
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IEEE-CIS raw profile</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
          margin: 2rem; color: #1a1a1a; }}
  h1 {{ margin-bottom: 0.25rem; }}
  h2 {{ margin-top: 2rem; border-bottom: 1px solid #ccc; padding-bottom: 0.25rem; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }}
  .card {{ border: 1px solid #ddd; padding: 0.75rem 1rem; border-radius: 6px;
          background: #fafafa; min-width: 12rem; }}
  .card .label {{ font-size: 0.8rem; text-transform: uppercase; color: #666; }}
  .card .value {{ font-size: 1.25rem; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 0.5rem;
          font-size: 0.9rem; }}
  th, td {{ padding: 0.25rem 0.5rem; border-bottom: 1px solid #eee;
          text-align: left; vertical-align: top; }}
  th {{ background: #f0f0f0; position: sticky; top: 0; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .scroll {{ max-height: 400px; overflow-y: auto; border: 1px solid #eee; }}
  footer {{ margin-top: 2rem; color: #888; font-size: 0.8rem; }}
</style>
</head>
<body>
<h1>IEEE-CIS raw data profile</h1>
<p>Schema version <strong>{summary.schema_version}</strong> ·
   Generated from <code>data/raw/</code> via
   <code>scripts/profile_raw.py</code>.</p>

<div class="cards">
  <div class="card"><div class="label">Rows</div>
    <div class="value">{summary.rows:,}</div></div>
  <div class="card"><div class="label">Columns</div>
    <div class="value">{summary.cols:,}</div></div>
  <div class="card"><div class="label">Memory</div>
    <div class="value">{summary.memory_mb:,.1f} MB</div></div>
  <div class="card"><div class="label">Fraud rate</div>
    <div class="value">{summary.fraud_rate_overall:.3%}</div></div>
  <div class="card"><div class="label">Identity coverage</div>
    <div class="value">{summary.identity_coverage:.2%}</div></div>
  <div class="card"><div class="label">TransactionDT span</div>
    <div class="value">{summary.tx_dt_span_days:.1f} days</div></div>
  <div class="card"><div class="label">Cols &gt; 50% null</div>
    <div class="value">{summary.cols_above_50pct_null:,}</div></div>
</div>

<h2>Fraud rate by ProductCD</h2>
<table>
<thead><tr><th>ProductCD</th><th>Fraud rate</th></tr></thead>
<tbody>
{product_rows}
</tbody>
</table>

<h2>Missingness (all columns, sorted desc)</h2>
<div class="scroll">
<table>
<thead><tr><th>Column</th><th>Null rate</th><th>Null count</th><th>Dtype</th></tr></thead>
<tbody>
{miss_rows}
</tbody>
</table>
</div>

<h2>Unique counts (object / categorical columns)</h2>
<div class="scroll">
<table>
<thead><tr><th>Column</th><th>n_unique</th><th>Dtype</th></tr></thead>
<tbody>
{uniq_rows}
</tbody>
</table>
</div>

<h2>Numeric summary</h2>
<div class="scroll">
<table>
<thead><tr>
  <th>Column</th><th>Count</th><th>Mean</th><th>Std</th>
  <th>Min</th><th>p50</th><th>p95</th><th>Max</th>
</tr></thead>
<tbody>
{num_rows}
</tbody>
</table>
</div>

<footer>
  TransactionDT range: {summary.tx_dt_min:,} &rarr; {summary.tx_dt_max:,} seconds.
</footer>
</body>
</html>
"""


@click.command()
@click.option(
    "--reports-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override the output directory. Defaults to <repo>/reports.",
)
def profile(reports_dir: Path | None) -> None:
    """Generate reports/raw_profile.html and raw_profile_summary.json."""
    settings = get_settings()
    configure_logging(pipeline_name="profile_raw")
    logger = get_logger(__name__)

    target_dir = reports_dir or (settings.data_dir.parent / "reports")
    target_dir.mkdir(parents=True, exist_ok=True)
    html_path = target_dir / "raw_profile.html"
    json_path = target_dir / "raw_profile_summary.json"

    loader = RawDataLoader(settings=settings)
    logger.info("profile.load_merged", raw_dir=str(loader.raw_dir))
    merged = loader.load_merged(optimize=False)

    logger.info("profile.compute", rows=int(merged.shape[0]), cols=int(merged.shape[1]))
    summary = _compute_summary(merged)
    missingness = _missingness_rows(merged)
    uniques = _unique_rows(merged)
    numerics = _numeric_rows(merged)

    html_path.write_text(
        _render_html(summary, missingness, uniques, numerics),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(asdict(summary), indent=2) + "\n",
        encoding="utf-8",
    )

    click.echo(click.style(f"Wrote {html_path}", fg="green"))
    click.echo(click.style(f"Wrote {json_path}", fg="green"))
    logger.info(
        "profile.done",
        html=str(html_path),
        json=str(json_path),
        fraud_rate=summary.fraud_rate_overall,
        identity_coverage=summary.identity_coverage,
    )


if __name__ == "__main__":
    profile()
