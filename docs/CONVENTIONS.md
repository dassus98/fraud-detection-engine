# Conventions

This document is the **source of truth** for every piece of code
produced in this repository. Every agent (human or otherwise) working
on this project must read it before touching code. Deviations are
allowed only with an explicit, in-line justification.

Content below is the Universal Standards section, reproduced verbatim.

---

## Universal Standards (Apply To Every Prompt)

These rules apply to every piece of code Claude Code produces. They should be in every agent's working memory.

### Code Quality

- **Python 3.11+.** Type hints on every function signature and class attribute. Use `from __future__ import annotations` at the top of every module.

- **Docstrings:** Google-style, on every public function, class, and module. Explain what the code does, why the business logic is what it is, and what the trade-offs were. Example:

    ```python
    def compute_velocity_ewm(events: pd.DataFrame, lambda_: float = 0.1) -> pd.Series:
        """Compute exponentially-weighted velocity over past events.

        Business rationale: Standard velocity counts treat a transaction 1 hour ago
        identically to one 7 days ago, but recent activity is far more predictive
        of fraud. Exponential decay weights recent events more heavily.

        Trade-offs considered:
            - Higher lambda (faster decay) is more sensitive to recent patterns but
              noisier. Lower lambda is smoother but slower to react to sudden
              fraud-ring activity.
            - This is a point-in-time computation; in production, this would be
              maintained as a running state in Redis (see Sprint 5).

        Args:
            events: DataFrame with a 'timestamp' column (float seconds).
            lambda_: Decay rate per hour. Default 0.1 → half-life ~7 hours.

        Returns:
            A Series of EWM velocity values, indexed identically to `events`.

        Raises:
            ValueError: If `events` is missing the 'timestamp' column.
        """
    ```

- **Comments:** Every non-trivial block gets a comment explaining _why_, not _what_. The code shows what; comments show intent.

- **No hardcoded values.** Every threshold, path, hyperparameter, URL, port, or magic number goes in `config/` (see Sprint 0) or `.env`. If a value genuinely has no sensible alternative (e.g., `0` as an index), justify it in a comment.

- **Type checking:** `mypy --strict` passes on all `src/` code. `pyright` as a secondary check.

- **Linting:** `ruff check` and `ruff format` pass. Line length 100.

- **No dead code.** If a function is written but not used, either use it or delete it. No commented-out code blocks.


### Logging

- **Structured logs via `structlog`.** Never `print()` in production code. Never `logging` without structured fields.
- **Every function that touches data logs:** inputs (shape, schema hash), outputs (shape, schema hash), duration (ms), and any warnings.
- **Log levels:**
    - `DEBUG`: detailed state, feature values, intermediate computations (off in production)
    - `INFO`: pipeline stage entry/exit, row counts, durations
    - `WARNING`: schema drift, missing data above threshold, degraded performance
    - `ERROR`: recoverable failures (retries, fallbacks)
    - `CRITICAL`: unrecoverable failures (process should terminate)
- **Every log includes a `run_id`** (UUID4) that ties together all logs from a single pipeline execution. The run_id is generated once at pipeline entry and propagated.
- **Logs go to both stdout (JSON) and `logs/{pipeline_name}/{run_id}.log`.**

### Testing

- **pytest, not unittest.** Fixtures in `conftest.py`. No test writes to the real `data/` directory — use `tmp_path`.
- **Every module in `src/` has a corresponding `tests/unit/test_{module}.py` with ≥80% line coverage.**
- **Every data transformation has a property-based test** (hypothesis). Example: "for any DataFrame with the input schema, the output satisfies the output schema and preserves row count if and only if the transformation is row-preserving."
- **Every feature function has a temporal correctness test:** assert that computing feature F at time T uses only data from rows where `timestamp < T`. This catches look-ahead bias, the most common bug in fraud ML.
- **Integration tests in `tests/integration/`** verify multi-component flows (e.g., feature pipeline → model → API).
- **Data lineage tests in `tests/lineage/`** verify the schema contract at every stage: raw → cleaned → features → model input → predictions.

### Git & CI

- **Conventional Commits.** `feat(features): add tier-3 behavioral deviation features`. Body explains _why_.
- **Branch per prompt.** Merge to `main` only after verification passes. Tag sprint completions (`sprint-1-complete`).
- **Pre-commit hooks:** ruff, mypy, pytest (fast suite only), secret scanning (`detect-secrets`).
- **CI (GitHub Actions):** lint → type-check → test → build Docker image → smoke-test API. Fail fast, fail loud.

### Data Contracts

- **Every DataFrame has a `pandera` schema** defined in `src/schemas/`. Schemas are validated at every pipeline stage boundary.
- **Schema registry:** A YAML file (`configs/schemas.yaml`) documents every schema, its version, and its evolution history.

### Reproducibility

- **Seed everything.** `SEED = int(os.environ["SEED"])` in every module that uses randomness. Default 42. Set numpy, random, torch, and lightgbm seeds.
- **Pin dependencies.** `uv` with a locked `uv.lock` file. No unpinned versions.
- **Model artifacts are content-hashed.** File name includes the SHA256 of the training data + config + code version.
