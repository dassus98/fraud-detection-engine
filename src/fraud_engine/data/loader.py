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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
from pandera import DataFrameSchema

from fraud_engine.config.settings import Settings, get_settings
from fraud_engine.schemas.raw import (
    IdentitySchema,
    MergedSchema,
    SCHEMA_VERSION,
    TransactionSchema,
)
from fraud_engine.utils.logging import get_logger, log_call

Split = Literal["train", "test"]

# IEEE-CIS ships train / test under the same directory with different
# filename prefixes. Keying by split lets the public API accept the
# mnemonic string without leaking filesystem details.
_TRANSACTION_FILENAME_BY_SPLIT: Final[dict[Split, str]] = {
    "train": "train_transaction.csv",
    "test": "test_transaction.csv",
}
_IDENTITY_FILENAME_BY_SPLIT: Final[dict[Split, str]] = {
    "train": "train_identity.csv",
    "test": "test_identity.csv",
}

# Kaggle's test_identity.csv release used hyphens (`id-01`) instead of
# underscores (`id_01`); the train release uses underscores. We
# normalise test columns so the same pandera schema validates both.
_IDENTITY_HYPHEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"^id-(\d{2})$")

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
    def load_transactions(
        self,
        split: Split = "train",
        *,
        optimize: bool = True,
    ) -> pd.DataFrame:
        """Read and validate the transaction CSV for `split`.

        Args:
            split: Which partition to load. "train" includes the
                `isFraud` label; "test" does not (Kaggle holds it out
                for scoring).
            optimize: If True (default), apply dtype optimisation after
                schema validation. Turn off if a downstream consumer
                needs the raw float64 shape (e.g. bit-for-bit fidelity
                tests).

        Returns:
            A DataFrame of 590,540 rows × 394 cols on the stock Kaggle
            train snapshot; 506,691 rows × 393 cols on test.

        Raises:
            FileNotFoundError: If the split's transaction CSV is missing.
            pandera.errors.SchemaErrors: If the file violates the
                split's transaction schema.
        """
        path = self.raw_dir / _TRANSACTION_FILENAME_BY_SPLIT[split]
        df = self._read_csv(path)
        self._transaction_schema(split).validate(df, lazy=True)
        if optimize:
            df = self._optimize(df)
        self._emit_report(name=f"transactions.{split}", df=df)
        return df

    @log_call
    def load_identity(
        self,
        split: Split = "train",
        *,
        optimize: bool = True,
    ) -> pd.DataFrame:
        """Read and validate the identity CSV for `split`.

        Args:
            split: Which partition to load.
            optimize: If True (default), apply dtype optimisation after
                schema validation.

        Returns:
            A DataFrame of ~144k rows × 41 cols on the train snapshot;
            ~141k rows × 41 cols on test.

        Raises:
            FileNotFoundError: If the split's identity CSV is missing.
            pandera.errors.SchemaErrors: If the file violates
                `IdentitySchema`.
        """
        path = self.raw_dir / _IDENTITY_FILENAME_BY_SPLIT[split]
        df = self._read_csv(path)
        if split == "test":
            df = self._normalise_test_identity_columns(df)
        IdentitySchema.validate(df, lazy=True)
        if optimize:
            df = self._optimize(df)
        self._emit_report(name=f"identity.{split}", df=df)
        return df

    @log_call
    def load_merged(
        self,
        split: Split = "train",
        *,
        optimize: bool = True,
    ) -> pd.DataFrame:
        """Left-join identity onto transactions for `split` and validate.

        Left-join semantics match production: every transaction appears
        in the output; identity columns are NaN where coverage is
        absent. The merged frame is what every feature generator in
        Sprint 2+ consumes.

        Args:
            split: Which partition to merge.
            optimize: If True (default), apply dtype optimisation on
                the merged frame. Sub-frames are loaded *un-optimised*
                internally so the join on integer keys is lossless.

        Returns:
            The merged DataFrame, row count equal to transactions.

        Raises:
            ValueError: If TransactionID is not unique in either side.
            pandera.errors.SchemaErrors: If the merged shape violates
                the split's merged schema.
        """
        tx = self.load_transactions(split, optimize=False)
        idt = self.load_identity(split, optimize=False)
        merged = tx.merge(
            idt,
            on="TransactionID",
            how="left",
            validate="one_to_one",
        )
        if len(merged) != len(tx):
            raise ValueError(f"Left-join row-count drift: tx={len(tx)}, merged={len(merged)}")
        self._merged_schema(split).validate(merged, lazy=True)
        if optimize:
            merged = self._optimize(merged)
        self._emit_report(name=f"merged.{split}", df=merged)
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
    def _transaction_schema(split: Split) -> DataFrameSchema:
        """Return the transaction schema appropriate for `split`.

        Train carries `isFraud`; test does not. Instead of defining two
        full schemas, we derive the test variant by stripping the one
        label column from the train schema so the contract stays
        DRY across splits.

        Args:
            split: Which partition is being validated.

        Returns:
            The train schema (unchanged) or a shallow copy with
            `isFraud` removed for test.
        """
        if split == "train":
            return TransactionSchema
        return TransactionSchema.remove_columns(["isFraud"])

    @staticmethod
    def _merged_schema(split: Split) -> DataFrameSchema:
        """Return the merged schema appropriate for `split`.

        Args:
            split: Which partition is being validated.

        Returns:
            `MergedSchema` (train) or a label-free derivative (test).
        """
        if split == "train":
            return MergedSchema
        return MergedSchema.remove_columns(["isFraud"])

    @staticmethod
    def _normalise_test_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename Kaggle's test-identity `id-XX` columns to `id_XX`.

        The train / test identity CSVs disagree on a single character
        in the `id_*` column names — a known Kaggle release quirk.
        Normalising here means `IdentitySchema` (and every downstream
        consumer) sees the same shape regardless of split.

        Args:
            df: The raw test-identity DataFrame, freshly read from CSV.

        Returns:
            A DataFrame with columns renamed in-place.
        """
        rename_map: dict[str, str] = {}
        for col in df.columns:
            match = _IDENTITY_HYPHEN_PATTERN.match(str(col))
            if match:
                rename_map[col] = f"id_{match.group(1)}"
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

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
