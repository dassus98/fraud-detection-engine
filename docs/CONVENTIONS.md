# Conventions

This document is the **human-readable source of truth** for the
standards this codebase is built to. It mirrors sections 5, 6, 7, and
9 of [CLAUDE.md](../CLAUDE.md) verbatim — CLAUDE.md is the
agent-facing copy, this one is the reader-facing copy, and they are
kept in lockstep.

When the two ever disagree, CLAUDE.md wins and CONVENTIONS.md is
updated to match in the same PR.

---

## 5. Universal Coding Standards

### 5.1 Python

- **Python 3.11+.** `from __future__ import annotations` at the top of every module.
- **Type hints on every function signature, every class attribute, every public variable.** No untyped function bodies.
- **Pydantic for all configuration and API contracts.** Plain dicts are not acceptable for structured data crossing module boundaries.
- **Pandera schemas for all DataFrame contracts.** A DataFrame entering or leaving a module function must have its schema validated at the boundary.
- **`pathlib.Path`, never string paths.** No `os.path.join`. No string concatenation for paths.
- **No `*` imports.** Explicit imports only.

### 5.2 Docstrings

Google-style on every public function, class, and module. The docstring must include:

1. One-sentence summary
2. **Business rationale:** why this code exists and what business problem it solves
3. **Trade-offs considered:** what alternatives were rejected and why
4. Args / Returns / Raises (standard Google sections)

Example:

```python
def compute_velocity_ewm(events: pd.DataFrame, lambda_: float = 0.1) -> pd.Series:
    """Compute exponentially-weighted velocity over past events.

    Business rationale: Standard velocity counts (e.g., transactions in last
    24h) treat all transactions equally, but recent activity is far more
    predictive of fraud than older activity. Exponential decay weights recent
    events more heavily, which matches how production fraud systems track
    "heat" on entities.

    Trade-offs considered:
        - Higher lambda (faster decay) is more sensitive to recent bursts but
          noisier on slow-burn fraud. Lower lambda is smoother but lags.
        - This is a batch computation; in production this would be Redis
          state updated atomically per event (see Sprint 5 architecture).
        - We could maintain a sorted event log and binary-search for the
          window, but the incremental decay formulation is O(1) per event
          and matches what production code would do.

    Args:
        events: DataFrame with a 'timestamp' column (float seconds).
        lambda_: Decay rate per hour. Default 0.1 → half-life ~7 hours.

    Returns:
        A Series of EWM velocity values, indexed identically to events.

    Raises:
        ValueError: If events is missing the 'timestamp' column.
    """
```

If you cannot articulate the business rationale, the code probably should not exist. Stop and ask.

### 5.3 Comments

Comment **why**, not **what**. Code shows what; comments show intent and constraint. A comment like `# increment counter` is noise. A comment like `# OOF encoding: each fold's encoder is fit on other folds to prevent target leakage` is essential context.

### 5.4 No Hardcoded Values

Every threshold, path, hyperparameter, port, URL, magic number, or cost figure lives in:

- `src/fraud_engine/config/settings.py` (Pydantic Settings, reads from `.env`)
- `configs/*.yaml` (tunable hyperparameters, schema definitions, reason codes)
- `.env` (secrets, environment-specific values; gitignored)

If a value genuinely has no sensible alternative (e.g., `0` as a list index, `1.0` as a probability ceiling), justify it inline with a comment. The bar is "would another engineer want to change this without editing source code?" If yes, it goes in config.

### 5.5 Logging

- **`structlog` for all logging.** No `print()`. No raw `logging` module without structured fields.
- **Every function that touches data logs:** input shape, output shape, duration in ms, any warnings. Use the `@log_call` decorator from `src/fraud_engine/utils/logging.py`.
- **Every pipeline run has a `run_id` (UUID4)** generated once at entry and propagated through all logs. This is non-negotiable — it is the only way to trace a prediction back to the data lineage that produced it.
- **Logs go to both:**
    - stdout as JSON (for log aggregation pipelines)
    - `logs/{pipeline_name}/{run_id}.log` as structured text (for local debugging)
- **Log levels:**
    - `DEBUG`: detailed state, feature values, intermediate computations (off in production)
    - `INFO`: pipeline stage entry/exit, row counts, durations
    - `WARNING`: schema drift, missing data above threshold, degraded performance
    - `ERROR`: recoverable failures (retries, fallbacks taken)
    - `CRITICAL`: unrecoverable; process should terminate

### 5.6 Linting and Type Checking

- `ruff check` and `ruff format` must pass. Line length 100.
- `mypy --strict` must pass on all `src/` code.
- `pyright` as a secondary check.
- These run via `make lint` and `make typecheck`. They run in CI. They run pre-commit. There is no path where lint failures are "fixed later."

### 5.7 No Dead Code

If a function is written but unused, either use it or delete it. No commented-out code blocks. No "we might need this someday." If it's worth keeping, it goes in a docstring example or a notebook.

---

## 6. Testing Standards

### 6.1 Framework and Layout

- **pytest, not unittest.** Fixtures in `tests/conftest.py`.
- **No test writes to the real `data/` directory.** Use `tmp_path`.
- **Hypothesis for property-based tests** on data transformations.
- **httpx.AsyncClient for FastAPI integration tests.**

Three test categories:

| Type | Location | What it covers |
|------|----------|----------------|
| Unit | `tests/unit/test_<module>.py` | One module, no external dependencies, mocked I/O |
| Integration | `tests/integration/test_<flow>.py` | Multi-component flows (feature pipeline → model, API end-to-end) |
| Lineage | `tests/lineage/test_<contract>.py` | Schema validation, temporal integrity, data contracts |

### 6.2 Coverage Requirements

- ≥80% line coverage on `src/`. Lower coverage on a module requires a comment in the module-level docstring explaining why.
- Every public function in `src/` has at least one corresponding unit test.
- Every data transformation has both a unit test (correctness) and a lineage test (schema preservation, temporal integrity).

### 6.3 Critical Test Patterns

These tests are **mandatory** for fraud-ML correctness. They have caught real bugs in this exact project before.

**Temporal integrity test** (per feature module):

```python
def test_feature_uses_only_past_data(small_temporal_df):
    """Assert the feature value at time T is unchanged if we recompute it
    using only rows with timestamp < T. Look-ahead leakage is the most
    common bug in fraud ML; this test catches it."""
    # ... see tests/lineage/test_*_temporal.py for the pattern
```

**Shuffled-labels leakage test** (per training pipeline):

```python
def test_no_target_leakage_via_shuffle(features, labels):
    """Train the model on shuffled labels. Validation AUC should collapse
    to ~0.5. If it doesn't, target encoding or some other feature is
    leaking the label into the training data."""
    # ... see tests/integration/test_*_no_target_leak.py
```

**Schema contract test** (per pipeline stage):

```python
def test_output_matches_schema(input_df):
    """Output of every transformation must validate against its declared
    pandera schema. Schema mismatches must fail at the boundary, not
    silently corrupt downstream stages."""
```

### 6.4 What "Passing Tests" Means

`make test` returns 0. Coverage report shows ≥80%. No skipped tests without an issue-tracker reference. No `@pytest.mark.xfail` without a justification comment.

**Never modify a failing test to make it pass.** If a test fails, the test is almost always right and the code is wrong. Find the bug, not a way around the test.

---

## 7. Data Contracts and Lineage

### 7.1 Schema Registry

Every DataFrame schema is defined as a pandera `DataFrameSchema` in `src/fraud_engine/schemas/`. Schemas are versioned. Schema changes require:

1. Increment the version in `configs/schemas.yaml`
2. Document what changed and why in the schema's module docstring
3. Update all consumers
4. Add a migration test if the change is non-backward-compatible

### 7.2 Lineage Logging

Every transformation is wrapped with `@lineage_step` (from `src/fraud_engine/data/lineage.py`). This decorator logs:

- step name
- input schema hash, output schema hash
- input row count, output row count
- duration in ms
- run_id

Lineage logs land in `logs/lineage/{run_id}/lineage.jsonl`. They must be queryable with `jq`. They must allow tracing any prediction back to its raw source.

### 7.3 Row Count Invariants

Any transformation that drops rows must log each drop with a reason. Total row delta must equal logged drop count. Lineage test enforces this.

---

## 9. Anti-Patterns — Do Not Do These

These behaviors look productive but produce worse outcomes. They have happened before in this codebase.

1. **Modifying tests to pass.** If a test fails, the test is right. Find the bug.
2. **Deleting flaky tests.** Flaky tests reveal race conditions and non-determinism. Fix the root cause (usually unset seeds, ordering dependencies, or shared state).
3. **Silencing linters with `# noqa` without justification.** Each ignore needs a specific inline comment explaining why.
4. **`TODO: fix later` without an issue reference.** TODOs rot. Either do the work or file the ticket.
5. **Skipping docstrings on "obvious" functions.** Business rationale is almost never obvious. Write it.
6. **Inlining config "just for this one value".** Each exception multiplies. Every value goes in config.
7. **`print()` for "quick debugging".** Structured logs replace print in every case.
8. **Mock data when real data is available.** IEEE-CIS is the dataset. Mocks are for unit tests only, never for demonstrations or notebooks.
9. **Premature optimization.** Get correctness first, measure, then optimize. Latency claims without benchmarks are unverified.
10. **Claiming a sprint complete without running verification.** The verification protocol (CLAUDE.md section 11) is the gate. No exceptions.
11. **Reading from `archive/v1-original`.** That branch is portfolio history. The new build is independent. If you find yourself tempted to copy from it, stop and ask John.
12. **Running git commands.** See CLAUDE.md section 2. This is the most important rule in the project.
