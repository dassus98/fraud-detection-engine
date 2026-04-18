"""Bootstrap acceptance gate.

Runs the Sprint 0 quality checks in sequence and prints a green/red
status table. Non-zero exit code on any failure, with the captured
error block for the failing step printed verbatim.

This is the gate referenced by the Sprint 0 acceptance criteria:
`python scripts/verify_bootstrap.py` must print all-green.

Note:
    This module uses `click.echo` for the final status table. That is
    an intentional carve-out from the "never `print()`" rule in
    docs/CONVENTIONS.md: this script's *product* is a CLI UI, not
    pipeline telemetry. The check steps themselves still go through
    structlog.
"""

from __future__ import annotations

import shutil
import subprocess  # noqa: S404 — running project tools is the whole point
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import click

from fraud_engine.utils.logging import configure_logging, get_logger


@dataclass(frozen=True)
class Check:
    """One row in the verification table.

    Attributes:
        name: Short label (fits in a 10-col display).
        command: argv list, executed via subprocess.run.
    """

    name: str
    command: list[str]


@dataclass
class CheckResult:
    """Outcome of running a Check."""

    name: str
    ok: bool
    duration_s: float
    stdout: str
    stderr: str


# Order matters: fail fast on lint before booting the heavier tools.
_CHECKS: tuple[Check, ...] = (
    Check(name="ruff", command=["uv", "run", "ruff", "check", "."]),
    Check(name="mypy", command=["uv", "run", "mypy", "src"]),
    Check(
        name="pytest",
        command=["uv", "run", "python", "-m", "pytest", "-q", "tests/unit", "--no-cov"],
    ),
    Check(
        name="settings",
        command=[
            "uv",
            "run",
            "python",
            "-c",
            "from fraud_engine.config.settings import get_settings; "
            "s = get_settings(); s.ensure_directories(); "
            "print(f'data_dir={s.data_dir}')",
        ],
    ),
)

# Width of the name column in the status table — tuned to fit the
# longest check name plus some padding.
_NAME_COL_WIDTH: int = 10


def _run_check(check: Check, cwd: Path) -> CheckResult:
    """Run one check and capture timing + streams.

    Args:
        check: The Check to execute.
        cwd: Project root; every check runs from here.

    Returns:
        A CheckResult; `.ok` reflects whether the subprocess exited 0.
    """
    start = time.perf_counter()
    completed = subprocess.run(  # noqa: S603 — commands are hardcoded above
        check.command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    duration = time.perf_counter() - start
    return CheckResult(
        name=check.name,
        ok=completed.returncode == 0,
        duration_s=duration,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _format_row(result: CheckResult) -> str:
    """Render one status row.

    Example output: ``[ OK ] ruff        (0.12s)``

    Args:
        result: The check result.

    Returns:
        A styled string ready for click.echo.
    """
    tag = click.style("[ OK ]", fg="green") if result.ok else click.style("[FAIL]", fg="red")
    name = result.name.ljust(_NAME_COL_WIDTH)
    return f"{tag} {name} ({result.duration_s:5.2f}s)"


@click.command()
@click.option(
    "--fail-fast/--no-fail-fast",
    default=False,
    help="Stop on the first failing check instead of running them all.",
)
def verify(fail_fast: bool) -> None:
    """Run lint/type/test/settings checks and print a status table."""
    project_root = Path(__file__).resolve().parents[1]

    # Surface the script's own progress through structlog so ops can
    # grep the log file later — the click.echo output is only for the
    # human operator watching the shell.
    configure_logging(pipeline_name="verify_bootstrap")
    logger = get_logger(__name__)

    if shutil.which("uv") is None:
        click.echo(click.style("uv is not on PATH — install uv first.", fg="red"), err=True)
        sys.exit(2)

    results: list[CheckResult] = []
    logger.info("verify.start", n_checks=len(_CHECKS), fail_fast=fail_fast)

    for check in _CHECKS:
        logger.info("verify.check.start", check=check.name)
        result = _run_check(check, cwd=project_root)
        results.append(result)
        logger.info(
            "verify.check.done",
            check=check.name,
            ok=result.ok,
            duration_s=round(result.duration_s, 3),
        )
        if fail_fast and not result.ok:
            break

    click.echo("")
    for result in results:
        click.echo(_format_row(result))

    all_ok = all(r.ok for r in results) and len(results) == len(_CHECKS)
    click.echo("")
    if all_ok:
        click.echo(click.style("Bootstrap: GREEN", fg="green", bold=True))
        sys.exit(0)

    click.echo(click.style("Bootstrap: RED", fg="red", bold=True))
    click.echo("")
    # Print captured output for failing checks only — keep the terminal
    # focused on actionable errors rather than every tool's chatter.
    for result in results:
        if not result.ok:
            click.echo(click.style(f"--- {result.name} stdout ---", fg="yellow"))
            click.echo(result.stdout or "(empty)")
            click.echo(click.style(f"--- {result.name} stderr ---", fg="yellow"))
            click.echo(result.stderr or "(empty)")
    sys.exit(1)


if __name__ == "__main__":
    verify()
