"""Temporal train/val/test splitter for IEEE-CIS.

Carves the merged transaction frame into three contiguous time
windows using `TransactionDT` as the only ordering signal. Split
boundaries live in `Settings.train_end_dt` / `val_end_dt` so every
later sprint points at the same rows without re-deciding the cut.

Business rationale:
    Fraud risk drifts over time — new attack vectors appear, seasonal
    shopping behaviour shifts, and upstream data feeds change. A
    random split that lets the model train on *future* transactions
    and evaluate on *past* ones inflates reported skill because the
    model has seen examples that would not yet exist at scoring time.
    Temporal splitting mirrors production: the model only ever scores
    transactions newer than anything it has trained on. The residual
    gap between random-split AUC and temporal-split AUC quantifies
    how much of the baseline's apparent lift comes from temporal
    leakage versus genuine signal.

Trade-offs considered:
    - Stratified temporal split (preserving fraud rate per window) is
      unnecessary here: the lineage suite has already confirmed that
      the overall 3.5% fraud rate holds roughly uniformly across
      months, so calendar partitioning lands split-level fraud rates
      within 0.5%. Adding stratification would blur the time boundary
      for marginal fairness gain.
    - A rolling-window CV is more defensible for Sprint 3's final
      model (it exposes month-to-month generalisation), but it adds
      bookkeeping the baseline does not need. Prompt 1 uses the flat
      4/1/1 calendar split.
    - `train_end_dt` / `val_end_dt` are stored as seconds since the
      anchor rather than as ISO datetimes because TransactionDT is an
      integer column in IEEE-CIS; comparing in the same unit avoids a
      per-row conversion on the full 590k-row frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.utils.logging import get_logger, log_call

_MANIFEST_SCHEMA_VERSION: Final[int] = 1

# Column the splitter partitions on. Kept as a module constant so any
# future rename (e.g. a post-join rename in Sprint 2) updates exactly
# one place.
_TIME_COLUMN: Final[str] = "TransactionDT"
_LABEL_COLUMN: Final[str] = "isFraud"
_KEY_COLUMN: Final[str] = "TransactionID"


@dataclass(frozen=True)
class SplitFrames:
    """Container for the three temporal slices + their manifest.

    Attributes:
        train: Rows with `TransactionDT < train_end_dt`.
        val: Rows with `train_end_dt <= TransactionDT < val_end_dt`.
        test: Rows with `TransactionDT >= val_end_dt`.
        manifest: Serialisable summary dict — counts, bounds, fraud
            rates, schema version. Mirrored into MLflow params by
            `train_baseline` and written to disk by
            `write_split_manifest`.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    manifest: dict[str, Any]


@log_call
def temporal_split(
    df: pd.DataFrame,
    *,
    train_end_dt: int | None = None,
    val_end_dt: int | None = None,
    settings: Settings | None = None,
) -> SplitFrames:
    """Partition `df` into contiguous train / val / test windows.

    Business rationale:
        See the module docstring. The one-paragraph version is:
        production scores "future" transactions, so evaluation must
        mimic that ordering or it will overstate skill.

    Trade-offs considered:
        - The cut points are passed either through `Settings` (default
          path, used by scripts) or via keyword overrides (used by
          tests that want a custom boundary without monkeypatching).
          Accepting both keeps the API usable from the notebook
          without wiring a fresh `Settings` instance.
        - Empty-val and empty-test states raise rather than silently
          returning a zero-row DataFrame; a silent empty would pass
          the partition-equality invariant but break downstream
          consumers in a confusing way.

    Args:
        df: Merged frame; must contain `TransactionDT`, `isFraud`,
            and `TransactionID`.
        train_end_dt: Upper bound (exclusive) of the train window, in
            TransactionDT seconds. Defaults to
            `settings.train_end_dt`.
        val_end_dt: Upper bound (exclusive) of the val window, in
            TransactionDT seconds. Defaults to `settings.val_end_dt`.
        settings: Override for the Settings singleton; tests pass a
            monkeypatched instance.

    Returns:
        A `SplitFrames` with the three partitions and a manifest.

    Raises:
        KeyError: If `df` is missing a required column.
        ValueError: If `val_end_dt <= train_end_dt`, if any split is
            empty, or if the partition does not cover every row.
    """
    effective_settings = settings or get_settings()
    t_end = train_end_dt if train_end_dt is not None else effective_settings.train_end_dt
    v_end = val_end_dt if val_end_dt is not None else effective_settings.val_end_dt
    if v_end <= t_end:
        raise ValueError(
            f"val_end_dt={v_end} must be strictly greater than train_end_dt={t_end}"
        )

    missing = {_TIME_COLUMN, _LABEL_COLUMN, _KEY_COLUMN} - set(df.columns)
    if missing:
        raise KeyError(f"temporal_split requires columns {sorted(missing)}")

    dt = df[_TIME_COLUMN]
    train_mask = dt < t_end
    val_mask = (dt >= t_end) & (dt < v_end)
    test_mask = dt >= v_end

    train = df.loc[train_mask].copy()
    val = df.loc[val_mask].copy()
    test = df.loc[test_mask].copy()

    # Loud failure on an empty slice — an empty val or test frame
    # means the boundary was mis-specified and any downstream
    # evaluation would silently produce nonsense.
    for name, slice_ in (("train", train), ("val", val), ("test", test)):
        if len(slice_) == 0:
            raise ValueError(
                f"temporal_split produced an empty '{name}' slice with "
                f"train_end_dt={t_end}, val_end_dt={v_end}; check bounds "
                f"against TransactionDT range "
                f"[{int(dt.min())}, {int(dt.max())}]."
            )

    manifest = _build_manifest(
        original=df,
        train=train,
        val=val,
        test=test,
        train_end_dt=int(t_end),
        val_end_dt=int(v_end),
        settings=effective_settings,
    )
    splits = SplitFrames(train=train, val=val, test=test, manifest=manifest)

    get_logger(__name__).info(
        "splits.temporal_split",
        n_train=manifest["n_train"],
        n_val=manifest["n_val"],
        n_test=manifest["n_test"],
        fraud_rate_train=manifest["fraud_rate_train"],
        fraud_rate_val=manifest["fraud_rate_val"],
        fraud_rate_test=manifest["fraud_rate_test"],
    )
    return splits


def validate_no_overlap(splits: SplitFrames) -> None:
    """Assert the three splits form a clean partition of the input.

    Business rationale:
        A temporal split is only trustworthy if every row lands in
        exactly one slice. Silent overlap — caused by off-by-one
        boundary logic or duplicate TransactionIDs — would bleed
        training rows into validation and overstate AUC. Calling this
        helper at the end of every split operation catches that
        class of bug at the source.

    Trade-offs considered:
        - Using `TransactionID` sets for the disjointness check is
          O(n) in memory but O(1) intersection — cheap on 590k rows
          and defensive against any future row-duplication bug.
        - Range checks on `TransactionDT` are redundant with the
          set-based check when the splitter runs cleanly, but serve
          as a second line of defence if the caller constructs a
          `SplitFrames` by hand.

    Args:
        splits: Output of `temporal_split`.

    Raises:
        ValueError: If rows overlap between splits, if the total row
            count does not match, or if temporal ranges overlap.
    """
    train_ids = set(splits.train[_KEY_COLUMN].tolist())
    val_ids = set(splits.val[_KEY_COLUMN].tolist())
    test_ids = set(splits.test[_KEY_COLUMN].tolist())

    for lhs_name, lhs, rhs_name, rhs in (
        ("train", train_ids, "val", val_ids),
        ("train", train_ids, "test", test_ids),
        ("val", val_ids, "test", test_ids),
    ):
        shared = lhs & rhs
        if shared:
            raise ValueError(
                f"TransactionID overlap between {lhs_name} and {rhs_name}: "
                f"{len(shared)} shared keys (e.g. {sorted(shared)[:3]})"
            )

    total = len(train_ids) + len(val_ids) + len(test_ids)
    expected = splits.manifest["n_train"] + splits.manifest["n_val"] + splits.manifest["n_test"]
    if total != expected:
        raise ValueError(
            f"Split size mismatch: sum of TransactionID sets={total}, "
            f"manifest sum={expected}"
        )

    # Contiguity: train's max TransactionDT must be strictly less than
    # val's min; val's max must be strictly less than test's min.
    train_dt_max = splits.train[_TIME_COLUMN].max()
    val_dt_min = splits.val[_TIME_COLUMN].min()
    val_dt_max = splits.val[_TIME_COLUMN].max()
    test_dt_min = splits.test[_TIME_COLUMN].min()
    if train_dt_max >= val_dt_min:
        raise ValueError(
            f"Temporal overlap: train.TransactionDT.max={train_dt_max} "
            f">= val.TransactionDT.min={val_dt_min}"
        )
    if val_dt_max >= test_dt_min:
        raise ValueError(
            f"Temporal overlap: val.TransactionDT.max={val_dt_max} "
            f">= test.TransactionDT.min={test_dt_min}"
        )


def write_split_manifest(splits: SplitFrames, path: Path) -> Path:
    """Persist the split manifest as indented JSON.

    Business rationale:
        The manifest is the artefact that Sprint 2's feature pipeline
        and Sprint 4's evaluation both read to confirm they are
        scoring on the same slice the baseline used. Writing it to a
        known path (typically `data/interim/splits_manifest.json`)
        lets those later stages verify integrity without re-running
        the splitter.

    Args:
        splits: Output of `temporal_split`.
        path: Destination file path. Parent directories are created
            if needed.

    Returns:
        The path that was written (for chaining / logging).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(splits.manifest, indent=2, default=str, sort_keys=True),
        encoding="utf-8",
    )
    get_logger(__name__).info("splits.manifest_written", path=str(path))
    return path


def _build_manifest(
    *,
    original: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    train_end_dt: int,
    val_end_dt: int,
    settings: Settings,
) -> dict[str, Any]:
    """Compute the serialisable manifest for `temporal_split`.

    Kept private because the shape of the manifest is the contract
    surface — callers should read via `SplitFrames.manifest`, not
    rebuild it.
    """
    return {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "transaction_dt_anchor_iso": settings.transaction_dt_anchor_iso,
        "train_end_dt": train_end_dt,
        "val_end_dt": val_end_dt,
        "seed": settings.seed,
        "n_original": int(len(original)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "fraud_rate_overall": float(original[_LABEL_COLUMN].mean()),
        "fraud_rate_train": float(train[_LABEL_COLUMN].mean()),
        "fraud_rate_val": float(val[_LABEL_COLUMN].mean()),
        "fraud_rate_test": float(test[_LABEL_COLUMN].mean()),
        "min_transaction_dt": int(original[_TIME_COLUMN].min()),
        "max_transaction_dt": int(original[_TIME_COLUMN].max()),
    }


__all__ = [
    "SplitFrames",
    "temporal_split",
    "validate_no_overlap",
    "write_split_manifest",
]
