"""Raw IEEE-CIS loader with schema validation and memory optimisation.

Reads `train_transaction.csv` and `train_identity.csv` from the
configured `raw_dir`, validates each against its pandera schema, and
applies a memory-reduction pass (float64→float32, int64→narrowest
signed int, low-cardinality object→category). The optimised frames
are what the rest of the pipeline consumes.

Business rationale:
    IEEE-CIS transaction.csv is ~650k rows × 394 cols. Loaded naively
    it is ~2 GB in RAM — untenable on a laptop and wasteful on a
    cloud instance. Float32 is more than enough precision for every
    downstream model (LightGBM, SHAP, the NN). Categorical dtypes on
    emails / device info halve memory again and speed up groupbys in
    Sprint 2. Schema validation at load time means a silent drift
    (Kaggle re-release, missing column) fails at ingest rather than
    corrupting features four sprints later.

Trade-offs considered:
    - Default pandas engine is used over `pyarrow` because the pandera
      schemas target numpy dtypes; the pyarrow backend would return
      Arrow types and force a round-trip conversion before validation.
      Load time is ~25-35s on a warm disk — acceptable for a one-shot
      ingest.
    - Memory optimisation runs *after* schema validation so the
      schema is the inbound contract, not an implementation detail.
    - A `categorical_threshold` of 0.5 (unique/total) errs on the
      side of category conversion; text-free columns like
      `P_emaildomain` (hundreds of domains over hundreds of thousands
      of rows) become categorical, while ID-like columns with near-1.0
      uniqueness stay as object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.schemas.raw import (
    IdentitySchema,
    MergedSchema,
    SCHEMA_VERSION,
    TransactionSchema,
)
from fraud_engine.utils.logging import get_logger, log_call

_TRANSACTION_FILENAME: Final[str] = "train_transaction.csv"
_IDENTITY_FILENAME: Final[str] = "train_identity.csv"

# Heuristic for object→category promotion. 0.5 keeps high-cardinality
# ID-like columns as object (unique ratio approaches 1) while promoting
# true categoricals like `card4` / `P_emaildomain`.
_CATEGORICAL_UNIQUE_RATIO: Final[float] = 0.5


@dataclass(frozen=True)
class LoadReport:
    """Lightweight result object for observability.

    Attributes:
        rows: Row count of the returned DataFrame.
        cols: Column count of the returned DataFrame.
        memory_mb: Memory footprint in megabytes, post-optimisation.
        schema_version: The `SCHEMA_VERSION` the frame was validated
            against.
    """

    rows: int
    cols: int
    memory_mb: float
    schema_version: int


class RawDataLoader:
    """Load and validate the raw IEEE-CIS tables.

    Business rationale:
        A single class concentrates (a) filesystem layout knowledge,
        (b) schema enforcement, and (c) memory optimisation so
        downstream code never reimplements any of them. Every later
        sprint asks "where does the raw data come from?" and this is
        the answer.

    Trade-offs considered:
        - A class, not a free function, because the loader carries
          state (configured `raw_dir`, injected `Settings`) and
          downstream tests want to override both cleanly.
        - The loader does not cache results. Re-reading is O(30s) and
          each stage of the pipeline persists its own interim
          artefact, so caching here would only serve interactive
          notebooks — not worth the invalidation risk.

    Attributes:
        raw_dir: Directory containing the two raw CSVs.
        settings: Settings instance used for logging / path defaults.
    """

    def __init__(
        self,
        raw_dir: Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Construct the loader.

        Args:
            raw_dir: Override for the raw data directory. Defaults to
                `settings.raw_dir`.
            settings: Override for the Settings singleton. Tests pass a
                monkeypatched instance.
        """
        self._settings: Settings = settings or get_settings()
        self.raw_dir: Path = raw_dir or self._settings.raw_dir
        self._log = get_logger(__name__)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @log_call
    def load_transactions(self, *, optimize: bool = True) -> pd.DataFrame:
        """Read and validate `train_transaction.csv`.

        Args:
            optimize: If True (default), apply dtype optimisation after
                schema validation. Turn off if a downstream consumer
                needs the raw float64 shape (e.g. bit-for-bit fidelity
                tests).

        Returns:
            A DataFrame of 590,540 rows × 394 cols on the stock Kaggle
            snapshot.

        Raises:
            FileNotFoundError: If `train_transaction.csv` is missing.
            pandera.errors.SchemaErrors: If the file violates
                `TransactionSchema`.
        """
        path = self.raw_dir / _TRANSACTION_FILENAME
        df = self._read_csv(path)
        TransactionSchema.validate(df, lazy=True)
        if optimize:
            df = self._optimize(df)
        self._emit_report(name="transactions", df=df)
        return df

    @log_call
    def load_identity(self, *, optimize: bool = True) -> pd.DataFrame:
        """Read and validate `train_identity.csv`.

        Args:
            optimize: If True (default), apply dtype optimisation after
                schema validation.

        Returns:
            A DataFrame of ~144k rows × 41 cols on the stock Kaggle
            snapshot (identity coverage is ~24% of transactions).

        Raises:
            FileNotFoundError: If `train_identity.csv` is missing.
            pandera.errors.SchemaErrors: If the file violates
                `IdentitySchema`.
        """
        path = self.raw_dir / _IDENTITY_FILENAME
        df = self._read_csv(path)
        IdentitySchema.validate(df, lazy=True)
        if optimize:
            df = self._optimize(df)
        self._emit_report(name="identity", df=df)
        return df

    @log_call
    def load_merged(self, *, optimize: bool = True) -> pd.DataFrame:
        """Left-join identity onto transactions and validate.

        Left-join semantics match production: every transaction appears
        in the output; identity columns are NaN where coverage is
        absent. The merged frame is what every feature generator in
        Sprint 2+ consumes.

        Args:
            optimize: If True (default), apply dtype optimisation on
                the merged frame. Sub-frames are loaded *un-optimised*
                internally so the join on integer keys is lossless.

        Returns:
            The merged DataFrame, row count equal to transactions.

        Raises:
            ValueError: If TransactionID is not unique in either side.
            pandera.errors.SchemaErrors: If the merged shape violates
                `MergedSchema`.
        """
        tx = self.load_transactions(optimize=False)
        idt = self.load_identity(optimize=False)
        merged = tx.merge(
            idt,
            on="TransactionID",
            how="left",
            validate="one_to_one",
        )
        if len(merged) != len(tx):
            raise ValueError(
                f"Left-join row-count drift: tx={len(tx)}, merged={len(merged)}"
            )
        MergedSchema.validate(merged, lazy=True)
        if optimize:
            merged = self._optimize(merged)
        self._emit_report(name="merged", df=merged)
        return merged

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _read_csv(self, path: Path) -> pd.DataFrame:
        """Read a CSV with numpy dtypes and a mandatory existence check.

        Args:
            path: Absolute path to the CSV file.

        Returns:
            A DataFrame loaded with the pandas default C engine.

        Raises:
            FileNotFoundError: If `path` does not exist.
        """
        if not path.is_file():
            raise FileNotFoundError(
                f"Expected raw file at {path} — run `make data-download` first."
            )
        # low_memory=False to prevent mixed-dtype warnings on the wide
        # V1..V339 block; we still receive numpy dtypes.
        return pd.read_csv(path, low_memory=False)

    def _emit_report(self, *, name: str, df: pd.DataFrame) -> None:
        """Log a LoadReport for the frame `df`."""
        report = LoadReport(
            rows=int(df.shape[0]),
            cols=int(df.shape[1]),
            memory_mb=round(df.memory_usage(deep=True).sum() / (1024**2), 2),
            schema_version=SCHEMA_VERSION,
        )
        self._log.info(
            "raw_loader.report",
            name=name,
            rows=report.rows,
            cols=report.cols,
            memory_mb=report.memory_mb,
            schema_version=report.schema_version,
        )

    @staticmethod
    def _optimize(df: pd.DataFrame) -> pd.DataFrame:
        """Downcast numeric dtypes and promote low-card objects to category.

        Runs column-by-column because IEEE-CIS has mixed widths per
        column group; a blanket downcast to float32 would round
        `TransactionAmt` precision we want to keep (it stays float32
        which is still ~7 significant digits — acceptable for USD).

        Business rationale:
            LightGBM, SHAP, and the NN consumers are all float32-native;
            keeping float64 triples memory and disk for no accuracy win.
            Categorical dtypes on string columns accelerate Sprint 2
            aggregations by 2-5x.

        Args:
            df: Input DataFrame; not modified in place.

        Returns:
            A new DataFrame with reduced memory footprint.
        """
        out = df.copy()
        for col in out.columns:
            series = out[col]
            dtype = series.dtype
            if pd.api.types.is_integer_dtype(dtype):
                out[col] = pd.to_numeric(series, downcast="integer")
            elif pd.api.types.is_float_dtype(dtype):
                out[col] = series.astype(np.float32)
            elif dtype == object:
                n = len(series)
                if n == 0:
                    continue
                unique_ratio = series.nunique(dropna=True) / n
                if unique_ratio < _CATEGORICAL_UNIQUE_RATIO:
                    out[col] = series.astype("category")
        return out


__all__ = ["LoadReport", "RawDataLoader"]
