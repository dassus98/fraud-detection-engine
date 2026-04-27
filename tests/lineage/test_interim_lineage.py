"""Lineage contract: `@lineage_step` applied to `TransactionCleaner`.

This file is the first integration of the lineage primitive against
a real Sprint 1 transformation. It does NOT modify
`TransactionCleaner.clean()` — the decorator is applied **inline in
the test** so the prompt's blast radius stays at the three new files
and one re-export. Permanent attachment of `@lineage_step` to
`cleaner.clean` and `loader.load_merged` is a later prompt's job.

Business rationale:
    CLAUDE.md §7.2 mandates that every prediction be traceable to its
    raw source via lineage records. Validating the primitive against a
    real cleaner — not just synthetic identity functions — proves the
    decorator handles the bound-method case, the `pandera`-validated
    output frame, and the row-drop accounting that production
    transformations actually exhibit.

Trade-offs considered:
    - The 10-row fixture duplicates `_merged_fixture_df` from
      `tests/unit/test_cleaner.py` rather than promoting it to
      `tests/conftest.py`. Promoting would touch a 5th file and
      enlarge the prompt; intentional duplication keeps blast radius
      tight. A future prompt that wires lineage into more
      transformations should consolidate.
    - All assertions key off `LineageLog.read(run_id, settings)`
      so test failures point at the persisted artefact, not at any
      in-memory `LineageStep` object the decorator might have built.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest
from structlog.contextvars import bind_contextvars, clear_contextvars

from fraud_engine.config.settings import Settings
from fraud_engine.data.cleaner import TransactionCleaner
from fraud_engine.data.lineage import LineageLog, lineage_step
from fraud_engine.utils.logging import new_run_id

pytestmark = pytest.mark.lineage

# Mirror the constants used by `tests/unit/test_cleaner.py::_merged_fixture_df`
# to keep the row-design invariants legible at the call site.
_FRI_MIDNIGHT: int = 0
_SAT_MIDNIGHT: int = 86_400
_SUN_MIDNIGHT: int = 172_800
_MON_MIDNIGHT: int = 259_200
_TUE_MIDNIGHT: int = 345_600
_FRI_2PM: int = 50_400
_SAT_9PM: int = 86_400 + 75_600
_SUN_7AM: int = 172_800 + 25_200


def _merged_fixture_df() -> pd.DataFrame:
    """Return a 10-row frame satisfying `MergedSchema`'s required set.

    Duplicated from `tests/unit/test_cleaner.py`; see the trade-offs
    section in the module docstring above for the rationale.
    """
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "isFraud": [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
            "TransactionDT": [
                _FRI_MIDNIGHT,
                _SAT_MIDNIGHT,
                _SUN_MIDNIGHT,
                _FRI_2PM,
                _FRI_MIDNIGHT,
                _FRI_MIDNIGHT,
                _MON_MIDNIGHT,
                _SAT_9PM,
                _SUN_7AM,
                _TUE_MIDNIGHT,
            ],
            "TransactionAmt": [
                10.0,
                25.0,
                30.0,
                50.0,
                0.0,
                -5.0,
                12.5,
                100.0,
                200.0,
                7.0,
            ],
            "ProductCD": ["W", "W", "H", "W", "W", "W", "W", "C", "R", "S"],
            "P_emaildomain": [
                "gmail.com",
                "Gmail.com ",
                "  YAHOO.com",
                None,
                "gmail.com",
                "yahoo.com",
                "outlook.com",
                "gmail.com",
                "yahoo.com",
                "gmail.com",
            ],
            "R_emaildomain": [
                "yahoo.com",
                None,
                "OUTLOOK.com",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        }
    )


@pytest.fixture
def run_id() -> Iterator[str]:
    """Bind a fresh run_id and unbind on teardown.

    Each test runs under a unique run_id so the lineage artefacts
    never collide between tests, and so the autouse `mock_settings`
    teardown leaves no shared state.
    """
    rid = new_run_id()
    bind_contextvars(run_id=rid)
    yield rid
    clear_contextvars()


@pytest.fixture
def cleaner() -> TransactionCleaner:
    """A cleaner using default Settings (anchor 2017-12-01 UTC)."""
    return TransactionCleaner(settings=Settings())


def test_decorated_cleaner_writes_lineage_record(
    mock_settings: Settings,
    run_id: str,
    cleaner: TransactionCleaner,
) -> None:
    """One JSONL record carries `interim_clean` + correct row counts."""
    wrapped = lineage_step("interim_clean")(cleaner.clean)
    df = _merged_fixture_df()
    wrapped(df)

    steps = LineageLog.read(run_id, settings=mock_settings)
    assert len(steps) == 1
    step = steps[0]
    assert step.step_name == "interim_clean"
    assert step.input_rows == 10
    assert step.output_rows == 8
    assert cleaner.last_report is not None
    assert step.input_rows == cleaner.last_report.rows_in
    assert step.output_rows == cleaner.last_report.rows_out


def test_decorated_cleaner_drop_invariant_holds(
    mock_settings: Settings,
    run_id: str,
    cleaner: TransactionCleaner,
) -> None:
    """`output_rows == input_rows - rows_dropped` (CLAUDE.md §7.3)."""
    wrapped = lineage_step("interim_clean")(cleaner.clean)
    wrapped(_merged_fixture_df())

    steps = LineageLog.read(run_id, settings=mock_settings)
    assert len(steps) == 1
    step = steps[0]
    assert cleaner.last_report is not None
    assert step.output_rows == step.input_rows - cleaner.last_report.rows_dropped


def test_decorated_cleaner_fingerprints_differ_input_to_output(
    mock_settings: Settings,
    run_id: str,
    cleaner: TransactionCleaner,
) -> None:
    """Cleaner adds `timestamp`/`hour`/`day_of_week`/`is_weekend` → schema shifts."""
    wrapped = lineage_step("interim_clean")(cleaner.clean)
    wrapped(_merged_fixture_df())

    steps = LineageLog.read(run_id, settings=mock_settings)
    assert len(steps) == 1
    step = steps[0]
    assert step.input_schema_hash != step.output_schema_hash
