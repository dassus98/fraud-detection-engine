"""Integration test for `scripts/run_economic_evaluation.py::run_economic_evaluation`.

Spec gates:
- 5K-row smoke completes on a stratified subsample of the test parquet.
- Output files (`reports/economic_evaluation.md`, both PNGs) are written.
- `.env` mutation respects `--dry-run` and writes a `.bak` when applied.
- Catastrophic floor: optimal_τ ∈ (0, 1); annual_savings_usd > 0;
  sensitivity DataFrame shape (125, 6).
- Spec gates ([0.3, 0.5] band, $500K floor, ±20% stability) are
  reported but NOT asserted on the smoke (5K is too noisy to pin).

Skip-gated on `data/processed/tier5_test.parquet`, the saved Model A
joblib, and the calibrator joblib all being present locally. Mirrors
`test_train_lightgbm.py`'s skip pattern.

Module-scoped fixture (lessons from the 3.3.d / 3.4.a fixture-scope
audit gap-fix): the smoke pipeline runs once per test file, not
once per test.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from scripts.run_economic_evaluation import (
    _BASELINE_THRESHOLD,
    _ENV_BACKUP_SUFFIX,
    _ENV_THRESHOLD_KEY,
    _SMOKE_SAMPLE_SIZE,
    EvaluationResult,
    _update_env_threshold,
    run_economic_evaluation,
)

from fraud_engine.config.settings import Settings, get_settings

pytestmark = pytest.mark.integration

_SMOKE_SEED: int = 42
_SMOKE_PORTFOLIO_MONTHLY: int = 100_000


def _processed_dir_has_test() -> bool:
    """True iff the test parquet exists locally."""
    return (get_settings().processed_dir / "tier5_test.parquet").is_file()


def _model_artefacts_present() -> bool:
    """True iff Model A joblib + calibrator joblib both exist locally."""
    sprint3_dir = get_settings().models_dir / "sprint3"
    return (sprint3_dir / "lightgbm_model.joblib").is_file() and (
        sprint3_dir / "calibrator.joblib"
    ).is_file()


@pytest.fixture(scope="module")
def isolated_settings(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Settings]:
    """Settings with isolated logs / mlruns; real models + data dirs preserved.

    Module-scoped (lessons from the 3.3.d / 3.4.a audits) so the smoke
    fixture below trains once for the whole module instead of once per
    test. Uses the manual `pytest.MonkeyPatch.context()` idiom because
    the function-scoped `monkeypatch` fixture can't be used at module
    scope.

    The script reads from `settings.processed_dir` (real) and
    `settings.models_dir` (real); we redirect logs / mlruns to a tmp
    dir so the test doesn't pollute the repo.
    """
    if not _processed_dir_has_test():
        pytest.skip(
            "data/processed/tier5_test.parquet not present — run "
            "`uv run python scripts/build_features_all_tiers.py` first."
        )
    if not _model_artefacts_present():
        pytest.skip(
            "models/sprint3/lightgbm_model.joblib or calibrator.joblib "
            "not present — run `uv run python scripts/train_lightgbm.py` first."
        )

    real_settings = get_settings()
    tmp_path = tmp_path_factory.mktemp("integ_run_economic_evaluation")
    logs_dir = tmp_path / "logs"
    mlruns = tmp_path / "mlruns"

    with pytest.MonkeyPatch.context() as mp:
        # Keep DATA_DIR + MODELS_DIR pointing at the real dirs (we need
        # the actual model + test parquet); redirect only the things
        # this script writes.
        mp.setenv("DATA_DIR", str(real_settings.data_dir))
        mp.setenv("MODELS_DIR", str(real_settings.models_dir))
        mp.setenv("LOGS_DIR", str(logs_dir))
        mp.setenv("MLFLOW_TRACKING_URI", str(mlruns))
        mp.setenv("SEED", str(_SMOKE_SEED))
        mp.setenv("LOG_LEVEL", "WARNING")

        get_settings.cache_clear()
        settings = Settings()
        settings.ensure_directories()
        yield settings
        get_settings.cache_clear()


@pytest.fixture(scope="module")
def smoke_result(
    isolated_settings: Settings,
    tmp_path_factory: pytest.TempPathFactory,
) -> EvaluationResult:
    """Run the smoke pipeline once; share the result across tests.

    Uses a fake `.env` in the tmp dir + `--dry-run` semantics so we
    don't mutate the repo's actual `.env`. Per-test fixtures verify
    the .env behaviour in isolation.
    """
    out_dir = tmp_path_factory.mktemp("smoke_outputs")
    fake_env = out_dir / ".env"
    fake_env.write_text(f"{_ENV_THRESHOLD_KEY}={_BASELINE_THRESHOLD}\n", encoding="utf-8")

    return run_economic_evaluation(
        settings=isolated_settings,
        portfolio_monthly=_SMOKE_PORTFOLIO_MONTHLY,
        sample_size=_SMOKE_SAMPLE_SIZE,
        dry_run=False,
        update_env=True,
        report_path=out_dir / "economic_evaluation.md",
        cost_curve_figure_path=out_dir / "figures" / "cost_curve.png",
        heatmap_figure_path=out_dir / "figures" / "heatmap.png",
        env_path=fake_env,
        random_state=_SMOKE_SEED,
    )


# ---------------------------------------------------------------------
# `TestRunSmoke`.
# ---------------------------------------------------------------------


class TestRunSmoke:
    """5K smoke completes; result carries every required field."""

    def test_returns_evaluation_result(self, smoke_result: EvaluationResult) -> None:
        """Smoke returns a populated `EvaluationResult`."""
        assert isinstance(smoke_result, EvaluationResult)

    def test_optimal_threshold_in_unit_interval(self, smoke_result: EvaluationResult) -> None:
        """Catastrophic floor: optimal_τ ∈ (0, 1)."""
        assert 0.0 < smoke_result.optimal_threshold < 1.0

    def test_baseline_threshold_is_one_half(self, smoke_result: EvaluationResult) -> None:
        """Baseline τ is the documented 0.5 placeholder."""
        assert smoke_result.baseline_threshold == _BASELINE_THRESHOLD

    def test_annual_savings_positive(self, smoke_result: EvaluationResult) -> None:
        """Catastrophic floor: annual savings strictly positive on the smoke.

        With cost-optimal τ vs the 0.5 baseline on a sane Model A,
        savings > 0 should always hold. Spec's $500K floor is not
        asserted on the smoke (5K rows is too noisy).
        """
        assert smoke_result.annual_savings_usd > 0

    def test_n_test_rows_at_or_below_smoke_sample(self, smoke_result: EvaluationResult) -> None:
        """Smoke subsample respects the cap."""
        assert smoke_result.n_test_rows <= _SMOKE_SAMPLE_SIZE

    def test_portfolio_monthly_recorded(self, smoke_result: EvaluationResult) -> None:
        """Result carries the portfolio assumption used."""
        assert smoke_result.portfolio_monthly == _SMOKE_PORTFOLIO_MONTHLY


# ---------------------------------------------------------------------
# `TestOutputFiles`.
# ---------------------------------------------------------------------


class TestOutputFiles:
    """Report markdown + cost-curve PNG + heatmap PNG all written."""

    def test_report_markdown_written(self, smoke_result: EvaluationResult) -> None:
        """`reports/economic_evaluation.md` exists + non-empty."""
        assert smoke_result.report_path.is_file()
        assert smoke_result.report_path.stat().st_size > 0

    def test_cost_curve_png_written(self, smoke_result: EvaluationResult) -> None:
        """Cost-curve PNG exists + non-empty."""
        assert smoke_result.cost_curve_figure_path.is_file()
        assert smoke_result.cost_curve_figure_path.stat().st_size > 0

    def test_heatmap_png_written(self, smoke_result: EvaluationResult) -> None:
        """Stratified-heatmap PNG exists + non-empty."""
        assert smoke_result.heatmap_figure_path.is_file()
        assert smoke_result.heatmap_figure_path.stat().st_size > 0

    def test_report_carries_expected_sections(self, smoke_result: EvaluationResult) -> None:
        """Markdown report has the spec-required sections."""
        body = smoke_result.report_path.read_text(encoding="utf-8")
        for required in (
            "# Economic Evaluation Report",
            "## Acceptance gates",
            "## Optimal threshold",
            "## Annual savings estimate",
            "## Sensitivity to cost variation",
            "## Stratified performance",
            "## Caveats",
            "## Artefacts",
            "## References",
        ):
            assert required in body, f"missing section: {required!r}"

    def test_report_includes_optimal_threshold_value(self, smoke_result: EvaluationResult) -> None:
        """The realised optimal τ appears in the report body."""
        body = smoke_result.report_path.read_text(encoding="utf-8")
        assert f"{smoke_result.optimal_threshold:.4f}" in body


# ---------------------------------------------------------------------
# `TestEnvUpdate`.
# ---------------------------------------------------------------------


class TestEnvUpdate:
    """`.env` mutation respects --dry-run + writes .bak when applied."""

    def test_env_updated_field_set_when_applied(self, smoke_result: EvaluationResult) -> None:
        """The smoke fixture ran with `update_env=True` and a fake `.env`."""
        assert smoke_result.env_updated is True
        assert smoke_result.env_path is not None
        assert smoke_result.env_backup_path is not None

    def test_env_file_contains_new_threshold(self, smoke_result: EvaluationResult) -> None:
        """The fake `.env` carries the new DECISION_THRESHOLD value."""
        assert smoke_result.env_path is not None
        body = smoke_result.env_path.read_text(encoding="utf-8")
        # The line should start with the key; the value is the optimum
        # written at six-digit precision per the script's convention.
        assert f"{_ENV_THRESHOLD_KEY}=" in body
        # Pull the realised value from the report's optimal_τ; it should
        # appear in the file body.
        # (We don't pin the exact float; just confirm the placeholder
        # 0.5 was replaced.)
        lines = [ln for ln in body.splitlines() if ln.startswith(f"{_ENV_THRESHOLD_KEY}=")]
        assert len(lines) == 1
        value = float(lines[0].split("=", 1)[1])
        assert abs(value - smoke_result.optimal_threshold) < 1e-5

    def test_env_backup_carries_prior_value(self, smoke_result: EvaluationResult) -> None:
        """`.env.bak` carries the placeholder value the script replaced."""
        assert smoke_result.env_backup_path is not None
        backup_body = smoke_result.env_backup_path.read_text(encoding="utf-8")
        assert f"{_ENV_THRESHOLD_KEY}={_BASELINE_THRESHOLD}" in backup_body

    def test_dry_run_skips_env_mutation(
        self,
        isolated_settings: Settings,
        tmp_path: Path,
    ) -> None:
        """`dry_run=True` writes the report but leaves `.env` untouched."""
        fake_env = tmp_path / ".env"
        original_text = f"{_ENV_THRESHOLD_KEY}={_BASELINE_THRESHOLD}\n"
        fake_env.write_text(original_text, encoding="utf-8")

        result = run_economic_evaluation(
            settings=isolated_settings,
            portfolio_monthly=_SMOKE_PORTFOLIO_MONTHLY,
            sample_size=_SMOKE_SAMPLE_SIZE,
            dry_run=True,
            update_env=True,
            report_path=tmp_path / "report.md",
            cost_curve_figure_path=tmp_path / "figures" / "cc.png",
            heatmap_figure_path=tmp_path / "figures" / "heat.png",
            env_path=fake_env,
            random_state=_SMOKE_SEED,
        )

        assert result.env_updated is False
        assert fake_env.read_text(encoding="utf-8") == original_text
        backup_path = fake_env.with_name(fake_env.name + _ENV_BACKUP_SUFFIX)
        assert not backup_path.exists()

    def test_no_update_env_skips_env_mutation(
        self,
        isolated_settings: Settings,
        tmp_path: Path,
    ) -> None:
        """`update_env=False` (CLI's --no-update-env) leaves `.env` untouched."""
        fake_env = tmp_path / ".env"
        original_text = f"{_ENV_THRESHOLD_KEY}={_BASELINE_THRESHOLD}\n"
        fake_env.write_text(original_text, encoding="utf-8")

        result = run_economic_evaluation(
            settings=isolated_settings,
            portfolio_monthly=_SMOKE_PORTFOLIO_MONTHLY,
            sample_size=_SMOKE_SAMPLE_SIZE,
            dry_run=False,
            update_env=False,
            report_path=tmp_path / "report.md",
            cost_curve_figure_path=tmp_path / "figures" / "cc.png",
            heatmap_figure_path=tmp_path / "figures" / "heat.png",
            env_path=fake_env,
            random_state=_SMOKE_SEED,
        )

        assert result.env_updated is False
        assert fake_env.read_text(encoding="utf-8") == original_text


class TestUpdateEnvThresholdHelper:
    """`_update_env_threshold` as a unit (small enough to test in isolation)."""

    def test_replaces_existing_key(self, tmp_path: Path) -> None:
        """Existing DECISION_THRESHOLD line is replaced; surrounding lines preserved."""
        env = tmp_path / ".env"
        env.write_text(
            "FRAUD_COST_USD=450\n" f"{_ENV_THRESHOLD_KEY}=0.5\n" "FP_COST_USD=35\n",
            encoding="utf-8",
        )
        backup = _update_env_threshold(0.371234, env)
        body = env.read_text(encoding="utf-8")
        # Key replaced; surrounding lines preserved.
        assert "FRAUD_COST_USD=450" in body
        assert "FP_COST_USD=35" in body
        assert f"{_ENV_THRESHOLD_KEY}=0.371234" in body
        # Backup carries the original.
        assert "DECISION_THRESHOLD=0.5" in backup.read_text(encoding="utf-8")

    def test_appends_when_key_missing(self, tmp_path: Path) -> None:
        """Missing DECISION_THRESHOLD key is appended with provenance comment."""
        env = tmp_path / ".env"
        env.write_text("FRAUD_COST_USD=450\n", encoding="utf-8")
        _update_env_threshold(0.4, env)
        body = env.read_text(encoding="utf-8")
        assert "FRAUD_COST_USD=450" in body
        assert f"{_ENV_THRESHOLD_KEY}=0.400000" in body
        assert "Sprint 4.4" in body  # provenance comment

    def test_missing_env_raises(self, tmp_path: Path) -> None:
        """Missing `.env` raises `FileNotFoundError` (no silent creation)."""
        env = tmp_path / "nope.env"
        with pytest.raises(FileNotFoundError, match=".env"):
            _update_env_threshold(0.4, env)

    def test_preserves_trailing_newline(self, tmp_path: Path) -> None:
        """File ending in newline keeps that ending."""
        env = tmp_path / ".env"
        env.write_text(f"{_ENV_THRESHOLD_KEY}=0.5\n", encoding="utf-8")
        _update_env_threshold(0.4, env)
        assert env.read_text(encoding="utf-8").endswith("\n")


# ---------------------------------------------------------------------
# `TestAcceptanceGates`.
# ---------------------------------------------------------------------


class TestAcceptanceGates:
    """Sensitivity / cost-curve / stratified table shapes are well-formed."""

    def test_sensitivity_dataframe_shape(self, smoke_result: EvaluationResult) -> None:
        """Sensitivity grid is the documented 5×5×5 = 125 cells × 6 columns."""
        assert smoke_result.sensitivity.shape == (125, 6)

    def test_sensitivity_columns(self, smoke_result: EvaluationResult) -> None:
        """Sensitivity DataFrame columns match the spec."""
        assert list(smoke_result.sensitivity.columns) == [
            "fraud_cost",
            "fp_cost",
            "tp_cost",
            "optimal_threshold",
            "optimal_total_cost",
            "optimal_cost_per_txn",
        ]

    def test_cost_curve_shape(self, smoke_result: EvaluationResult) -> None:
        """Cost curve is the documented 99-threshold sweep × 7 columns."""
        assert smoke_result.cost_curve.shape == (99, 7)

    def test_stratified_emits_at_least_amount_axis(self, smoke_result: EvaluationResult) -> None:
        """Stratified output has rows for the amount-bucket axis at minimum."""
        axes = set(smoke_result.stratified["stratum_axis"].unique())
        assert "amount_bucket" in axes

    def test_gate_pass_booleans_are_bool(self, smoke_result: EvaluationResult) -> None:
        """All three gate-pass fields are boolean (not numpy-bool)."""
        assert isinstance(smoke_result.optimum_band_gate_pass, bool)
        assert isinstance(smoke_result.annual_savings_gate_pass, bool)
        assert isinstance(smoke_result.sensitivity_stability_gate_pass, bool)

    def test_annual_cost_arithmetic(self, smoke_result: EvaluationResult) -> None:
        """Annual_cost = cost_per_txn × annual_volume (sanity check)."""
        annual_volume = smoke_result.portfolio_monthly * 12
        np.testing.assert_allclose(
            smoke_result.annual_cost_at_optimal_usd,
            smoke_result.cost_per_txn_at_optimal * annual_volume,
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            smoke_result.annual_cost_at_baseline_usd,
            smoke_result.cost_per_txn_at_baseline * annual_volume,
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            smoke_result.annual_savings_usd,
            smoke_result.annual_cost_at_baseline_usd - smoke_result.annual_cost_at_optimal_usd,
            rtol=1e-9,
        )


# ---------------------------------------------------------------------
# `TestErrorHandling`.
# ---------------------------------------------------------------------


class TestErrorHandling:
    """Helpful errors when artefacts are missing."""

    def test_missing_env_raises(self, isolated_settings: Settings, tmp_path: Path) -> None:
        """Pointing at a non-existent `.env` raises with a clear message."""
        with pytest.raises(FileNotFoundError, match=".env"):
            run_economic_evaluation(
                settings=isolated_settings,
                portfolio_monthly=_SMOKE_PORTFOLIO_MONTHLY,
                sample_size=_SMOKE_SAMPLE_SIZE,
                dry_run=False,
                update_env=True,
                report_path=tmp_path / "report.md",
                cost_curve_figure_path=tmp_path / "figures" / "cc.png",
                heatmap_figure_path=tmp_path / "figures" / "heat.png",
                env_path=tmp_path / "nonexistent.env",
                random_state=_SMOKE_SEED,
            )
