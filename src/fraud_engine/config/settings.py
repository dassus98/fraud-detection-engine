"""Project-wide configuration surface.

This module is the *single* source of truth for paths, seeds, economic
costs, model defaults, infrastructure URLs, and API binding. Every
hardcoded-able value lives here so downstream modules never reach for
env vars directly and tests can override via monkeypatch on Settings
fields.

Business rationale:
    Centralising configuration lets us (a) audit every tunable in one
    place — critical when economic costs drive decision thresholds, and
    (b) guarantee that Sprint 4's cost-curve optimizer and Sprint 5's
    serving layer agree on the same `decision_threshold`,
    `fraud_cost_usd`, and `fp_cost_usd` without plumbing them through
    call stacks.

Trade-offs considered:
    - A single monolithic Settings class is easier to reason about than
      per-subsystem config objects, and Pydantic's nested models would
      add ceremony that pays off only once the surface is much larger.
    - `get_settings()` is `lru_cache`-wrapped so the .env file is parsed
      once per process. Tests must call `get_settings.cache_clear()`
      after monkeypatching env vars.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root — discovered relative to this file so Settings works no
# matter where the process is launched from. The path
# src/fraud_engine/config/settings.py is 3 parents deep from the repo root.
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

_VALID_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class Settings(BaseSettings):
    """Typed, validated, env-driven configuration.

    Every field carries a `description=` so the .env.example and the
    Settings class never drift: future developers see the business
    meaning next to the default value.

    Attributes:
        See individual `Field(...)` descriptions below.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- paths ------------------------------------------------------
    data_dir: Path = Field(
        default=_PROJECT_ROOT / "data",
        description=(
            "Root for raw/interim/processed data. Gitignored. Subdirs "
            "created on demand by `ensure_directories()`."
        ),
    )
    models_dir: Path = Field(
        default=_PROJECT_ROOT / "models",
        description="Persisted model artefacts (gitignored).",
    )
    logs_dir: Path = Field(
        default=_PROJECT_ROOT / "logs",
        description="Pipeline logs — `{pipeline_name}/{run_id}.log`.",
    )

    # --- reproducibility --------------------------------------------
    seed: int = Field(
        default=42,
        description=(
            "Master seed. Seeds numpy, random, torch, and model-level RNGs "
            "via `utils.seeding.set_all_seeds`. 42 is conventional; any "
            "override must be recorded in the experiment's MLflow run."
        ),
    )

    # --- economic costs (drive thresholding in Sprint 4) ------------
    # `ge=0.0` on each: negative costs are nonsensical and would flip the
    # sign of the expected-cost objective in Sprint 4's threshold sweep.
    fraud_cost_usd: float = Field(
        default=450.0,
        ge=0.0,
        description=(
            "Mean marginal cost of a missed fraud (false negative), in "
            "USD. Calibrated from 2024 industry medians for "
            "card-not-present losses."
        ),
    )
    fp_cost_usd: float = Field(
        default=35.0,
        ge=0.0,
        description=(
            "Customer-friction cost of a false positive alert (support "
            "contact, re-auth friction, downstream churn)."
        ),
    )
    tp_cost_usd: float = Field(
        default=5.0,
        ge=0.0,
        description=(
            "Operational cost of a true positive — analyst review time "
            "to confirm a flagged transaction."
        ),
    )

    # --- model hyperparameter defaults ------------------------------
    lgbm_defaults: dict[str, Any] = Field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "max_depth": -1,
            "n_estimators": 500,
            "min_child_samples": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
        },
        description=(
            "LightGBM default hyperparameters. Sprint 3 tunes via Optuna "
            "and logs the winning set to MLflow."
        ),
    )

    # --- API --------------------------------------------------------
    api_host: str = Field(
        default="0.0.0.0",
        description="Uvicorn bind host. 0.0.0.0 for Docker; 127.0.0.1 for local.",
    )
    api_port: int = Field(
        default=8000,
        description="Uvicorn bind port.",
    )

    # --- infrastructure URLs ---------------------------------------
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for online feature store (sub-ms lookups).",
    )
    postgres_url: str = Field(
        default="postgresql://fraud:fraud@localhost:5432/fraud",
        description="Postgres URL for offline feature store and audit log.",
    )
    mlflow_tracking_uri: str = Field(
        default="./mlruns",
        description=(
            "MLflow tracking URI. Local file store for Sprint 0-3; "
            "switches to a server URI in Sprint 5."
        ),
    )
    mlflow_experiment_name: str = Field(
        default="fraud-detection",
        description=(
            "Default MLflow experiment name. Sprint 3 opens all "
            "training runs under this experiment; Sprint 4 threshold "
            "sweeps reuse it so the full model-selection trail lives "
            "in one tree."
        ),
    )
    mlflow_port: int = Field(
        default=5000,
        description=(
            "Host port the MLflow tracking server binds to in "
            "docker-compose.dev.yml. Same port the UI is served on."
        ),
    )
    prometheus_port: int = Field(
        default=9090,
        description=(
            "Host port for the Prometheus scrape/UI. Drives "
            "docker-compose.dev.yml port publishing."
        ),
    )
    grafana_port: int = Field(
        default=3000,
        description=(
            "Host port for the Grafana UI. Drives " "docker-compose.dev.yml port publishing."
        ),
    )
    grafana_admin_user: str = Field(
        default="admin",
        description=(
            "Initial Grafana admin username seeded into the container "
            "via GF_SECURITY_ADMIN_USER. Override in .env for any "
            "shared environment."
        ),
    )
    grafana_admin_password: SecretStr = Field(
        default=SecretStr("admin"),
        description=(
            "Initial Grafana admin password. `admin/admin` is the "
            "dev-only default; production deployments must override."
        ),
    )

    # --- provider credentials --------------------------------------
    kaggle_username: str | None = Field(
        default=None,
        description="Kaggle username for IEEE-CIS dataset download (Sprint 1).",
    )
    kaggle_key: SecretStr | None = Field(
        default=None,
        description="Kaggle API key (never logged, never echoed).",
    )

    # --- logging ---------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description=(
            "stdlib log level. DEBUG only for local development — the "
            "JSON processor includes feature values at DEBUG, which may "
            "contain PII."
        ),
    )

    # --- decisioning -----------------------------------------------
    decision_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Probability above which a transaction is flagged. 0.5 is a "
            "naive starting point; Sprint 4 replaces this via expected "
            "cost minimisation (see fraud_cost_usd / fp_cost_usd)."
        ),
    )

    # --- temporal split (Sprint 1 onwards) --------------------------
    # IEEE-CIS does not ship a calendar-anchored TransactionDT — Kaggle
    # publishes seconds since an anonymised reference. 2017-12-01 UTC
    # is the community-standard anchor; this is a convention, not a
    # Kaggle-supplied fact. Recorded here so every later stage
    # (feature engineering, evaluation, monitoring) speaks calendar
    # time consistently.
    transaction_dt_anchor_iso: str = Field(
        default="2017-12-01T00:00:00+00:00",
        description=(
            "ISO-8601 anchor for TransactionDT=0. Community convention "
            "for IEEE-CIS; change only if Kaggle re-releases with a "
            "different anchor. Downstream code parses this with "
            "datetime.fromisoformat."
        ),
    )
    train_end_dt: int = Field(
        default=86400 * 121,
        ge=1,
        description=(
            "Upper bound of the train split, in TransactionDT seconds "
            "since the anchor. Default 10,454,400 = 121 days "
            "(Dec 2017 + Jan/Feb/Mar 2018). Rows with TransactionDT "
            "strictly less than this go to train."
        ),
    )
    val_end_dt: int = Field(
        default=86400 * 151,
        ge=2,
        description=(
            "Upper bound of the val split, in TransactionDT seconds "
            "since the anchor. Default 13,046,400 = 151 days "
            "(+ Apr 2018). Rows with TransactionDT in "
            "[train_end_dt, val_end_dt) go to val; the remainder goes "
            "to test."
        ),
    )

    # ---------------------------------------------------------------
    # validators
    # ---------------------------------------------------------------

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        """Accept any case, normalise to upper, reject unknown levels.

        Raises:
            ValueError: If the value is not one of DEBUG/INFO/WARNING/
                ERROR/CRITICAL.
        """
        normalised = value.upper()
        if normalised not in _VALID_LOG_LEVELS:
            raise ValueError(f"log_level={value!r} is not in {sorted(_VALID_LOG_LEVELS)}")
        return normalised

    @field_validator("val_end_dt")
    @classmethod
    def _val_end_after_train_end(cls, value: int, info: ValidationInfo) -> int:
        """Enforce a non-empty val window.

        Raises:
            ValueError: If `val_end_dt <= train_end_dt`.
        """
        train_end = info.data.get("train_end_dt")
        if train_end is not None and value <= train_end:
            raise ValueError(
                f"val_end_dt={value} must be strictly greater than " f"train_end_dt={train_end}"
            )
        return value

    @field_validator("data_dir", "models_dir", "logs_dir")
    @classmethod
    def _resolve_relative_to_project_root(cls, value: Path) -> Path:
        """Resolve relative path overrides against the project root, not CWD.

        Without this, a relative path from `.env` (e.g. `DATA_DIR=./data`)
        resolves against the *current working directory at runtime*. That
        breaks any caller whose CWD is not the project root — most
        visibly the Jupyter / VS Code notebook kernel, whose CWD is
        usually `notebooks/` and which would resolve `./data` to
        `notebooks/data` (no MANIFEST.json there).

        Absolute paths pass through unchanged so a user can still point
        `DATA_DIR` at an external mount or NFS share.

        Args:
            value: The user-supplied path (from .env, env var, or
                explicit kwarg). May be absolute or relative.

        Returns:
            An absolute, resolved Path. Relative inputs are joined
            against `_PROJECT_ROOT` (the path of this file's package
            tree), then resolved.
        """
        if value.is_absolute():
            return value
        return (_PROJECT_ROOT / value).resolve()

    # ---------------------------------------------------------------
    # derived paths
    # ---------------------------------------------------------------

    @property
    def raw_dir(self) -> Path:
        """Raw, unmodified source data (Sprint 1 populates)."""
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        """Transient outputs between pipeline stages."""
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        """Final, model-ready feature tables."""
        return self.data_dir / "processed"

    # ---------------------------------------------------------------
    # side-effecting helpers
    # ---------------------------------------------------------------

    def ensure_directories(self) -> None:
        """Create all writable directories if they don't already exist.

        Idempotent. Called from pipeline entrypoints (Sprint 1+) and
        from the bootstrap smoke test. Swallows `FileExistsError` via
        `exist_ok=True`.
        """
        for path in (
            self.data_dir,
            self.models_dir,
            self.logs_dir,
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-singleton `Settings` instance.

    Wrapped in `lru_cache` so the .env file is parsed exactly once per
    process. Tests that mutate env vars must call
    `get_settings.cache_clear()` before re-reading.

    Returns:
        The cached Settings instance.
    """
    return Settings()
