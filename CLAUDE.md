> **This file is the source of truth for every Claude Code session in this repository.** Read it in full at the start of every session. Do not skim. Every rule below has been put here because skipping it has historically produced bugs, leakage, or rework.

---

## 1. Project Identity

**Name:** Real-Time Fraud Detection Engine **Owner:** John Das **Audience for the finished work:** Senior Data Scientist / ML Engineer hiring committees at Wealthsimple, Mercury, RBC, Nubank. **Dataset:** IEEE-CIS Fraud Detection (Kaggle, Vesta Corporation, 2019). 590,540 e-commerce transactions, 3.5% fraud rate, 431 features. **Primary language:** Python 3.11+ **Working directory:** `~/projects/fraud-detection-engine`

This is a portfolio project being built **as if it were a real engagement at a Canadian fintech.** Every architectural decision must be defensible in those terms — not as an academic exercise, not as a Kaggle submission. The reviewer is a senior data scientist who has built fraud systems in production. They will see through theatrical complexity and they will reward justified simplicity.

A previous iteration of this project exists on the `archive/v1-original` branch and is preserved as portfolio history. **Do not reference, copy from, or look at that branch.** This is a clean rebuild with stricter standards.

---

## 2. Critical Rules — Version Control

**The agent's git/gh authority is scoped to the PR-plumbing carve-out in §2.1. Everything else is forbidden.** When in doubt, don't run a git/gh command — write the completion report and hand the task back to John.

### 2.1 Allowed (the carve-out)

- `git checkout -b sprint-<N>/prompt-<X>-<Y>-<slug>` — create the feature branch off the current `main`. Naming per [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) §1.1.
- `git push -u origin <feature-branch>` — push the current feature branch to origin so a PR can be opened. **Never `main`. Never `--force` or `--force-with-lease`.**
- `gh pr create` — open the PR with the title format and body template from CONTRIBUTING.md §2.1–§2.2.
- Read-only diagnostics: `git status`, `git log`, `git diff`, `git branch --show-current`, `gh pr view`, `gh pr list`, `gh pr checks`. These don't change state and can be run freely.

### 2.2 Forbidden (the rule)

Everything outside §2.1, including:

- `git add`, `git commit`, `git commit --amend` — staging and commit authorship belong to John. Pre-commit hooks have historically misbehaved with mixed staged/unstaged trees, and commit messages are part of the portfolio's `git log --oneline main`.
- `git push --force`, `git push --force-with-lease`, any push to `main`.
- `git merge`, `git rebase`, `git pull` (which can rebase/merge), `git cherry-pick`.
- `git reset --hard`, `git revert`, `git restore`, `git clean`.
- `git tag`, `git stash`, `git rm`, `git filter-repo`, `git submodule`, `git checkout <existing-branch>` (use only `-b` to create new), `git switch`.
- `gh pr merge`, `gh pr close --delete-branch`, `gh release` of any kind.
- Any shell pipeline whose tail end is one of the above.

### 2.3 Why this shape

John owns the *consequential* operations: he commits (so each commit reflects his review of the staged diff and his message-authorship for the portfolio history), and he merges (so `main` only advances through his explicit action). The agent owns the *mechanical* plumbing in between — branch create, push, PR open. This mirrors how a senior engineer pairs with a reviewer who never touches `main` directly.

Unsolicited commits by Claude have historically produced merge conflicts, uninformative history, and forced rewrites that lose work. Holding the line at "agent never commits, never pushes to main, never merges, never tags" eliminates those failure modes while letting the agent open ~80–100 PRs across the project without paging John for the mechanical parts.

### 2.4 At the end of a prompt

After the work is implemented, tested, and the completion report is written:

1. State: "Verification passed. Ready for John to commit on `<feature-branch>`."
2. **Wait for John to commit** (and confirm the commit by saying so, or by the working tree going clean — `git status` reads are allowed).
3. Then `git push -u origin <feature-branch>` and `gh pr create` with the CONTRIBUTING.md §2.2 body template.
4. State: "PR open at <url>. Ready for John to squash-merge."

If no feature branch exists (Sprint 0's grandfathered single branch, or John ran the prompt on `main` accidentally), do not retroactively create branches over committed work — write the completion report and ask John how to proceed.

The full branching/PR/merge convention lives in [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md). Read it every session.

### 2.5 Other commands

**Bash and other commands are unrestricted** subject to the rest of this document. Run tests, run linters, install packages, build Docker images, modify files, create directories — all fine. The carve-out above is *only* for git/gh.

---

## 3. Architecture Overview

```
Raw IEEE-CIS data
    ↓ [data loader + schema validation]
Cleaned interim data (Parquet)
    ↓ [feature pipeline: T1 basic → T2 aggregations → T3 behavioral
        → T4 exponential decay → T5 graph]
Feature-engineered data (Parquet)
    ↓ [LightGBM training + Optuna tuning + calibration]
Production model (joblib)
    ↓ [economic threshold optimization with cost function]
Calibrated, threshold-tuned model
    ↓ [FastAPI + Redis (real-time features) + Postgres (batch features)]
Production API (Docker)
    ↓ [SHAP explainability + shadow mode + prediction logging]
    ↓ [Prometheus + Grafana monitoring + PSI drift detection]
```

**Production model:** LightGBM (Model A). Chosen for inference latency (<15ms) and SHAP interpretability. **Diversity models:** FraudNet (entity-embedding NN, Model B) and FraudGNN (PyTorch Geometric, Model C). Model B is shadow-deployable. Model C is batch-only (its outputs feed Model A as features). **Decision threshold:** Determined by economic cost function (see section 8), not 0.5 or F1. **Latency budget:** <100ms P95 end-to-end. **Target metrics:** AUC ≥ 0.93 on temporal split, >90% fraud capture at <2% false positive rate.

---

## 4. Repository Layout

```
fraud-detection-engine/
├── .github/workflows/         # CI configuration
├── configs/                   # YAML configs (schemas, hyperparameters, costs)
├── data/                      # gitignored; raw, interim, processed
├── docs/                      # ADRs, conventions, architecture, model card
├── logs/                      # gitignored; structured JSON + per-run dirs
├── notebooks/                 # numbered, exploratory + demo
├── scripts/                   # CLI entry points (download, train, evaluate)
├── src/fraud_engine/
│   ├── api/                   # FastAPI app, feature service, shadow, SHAP
│   ├── config/                # Pydantic Settings
│   ├── data/                  # Loader, cleaner, splits, lineage
│   ├── features/              # T1-T5 generators, pipeline, temporal guards
│   ├── models/                # LightGBM, neural, GNN, baseline, tuning
│   ├── evaluation/            # Economic cost, calibration, stratified
│   ├── monitoring/            # Drift, performance, Prometheus metrics
│   ├── schemas/               # pandera schemas (raw, interim, features)
│   └── utils/                 # Logging, tracing, seeding, MLflow
├── sprints/                   # Per-prompt completion reports
└── tests/
    ├── unit/                  # Module-level
    ├── integration/           # Multi-component flows
    └── lineage/               # Schema + temporal integrity contracts
```

If a file's home is unclear, default to `src/fraud_engine/utils/`. If a script could be a CLI entry point or a one-off, default to `scripts/`.

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

|Type|Location|What it covers|
|---|---|---|
|Unit|`tests/unit/test_<module>.py`|One module, no external dependencies, mocked I/O|
|Integration|`tests/integration/test_<flow>.py`|Multi-component flows (feature pipeline → model, API end-to-end)|
|Lineage|`tests/lineage/test_<contract>.py`|Schema validation, temporal integrity, data contracts|

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

## 8. Business-Logic Constants You Must Know

These values inform countless decisions across the codebase. Memorize the defaults.

|Constant|Default|Where set|Business meaning|
|---|---|---|---|
|`FRAUD_COST_USD`|450|`.env`|Avg cost of a missed fraud: $150 txn + $25 chargeback fee + $75 investigation + $50 scheme penalty + $150 reputation/regulatory|
|`FP_COST_USD`|35|`.env`|Cost of blocking a legit txn: $15 support call + 5% churn × $400 CLV|
|`TP_COST_USD`|5|`.env`|Investigation cost on a confirmed-fraud block|
|`DECISION_THRESHOLD`|0.5 (initial), then optimized in Sprint 4|`.env`|Probability above which we block. Optimized to minimize expected cost.|
|Latency budget (P95)|100ms|configs|End-to-end API latency, including Redis lookup, inference, SHAP, logging|
|Fraud rate (dataset)|3.5%|known|IEEE-CIS class imbalance; informs sampling and metric choices|
|Identity coverage|~24%|known|Only 24% of transactions have device/identity data; model must work without it|

These costs are configurable. Sensitivity analysis (Sprint 4) confirms decisions are stable under ±20% variation. If you change a default, run the sensitivity analysis again.

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
10. **Claiming a sprint complete without running verification.** The verification protocol (section 11) is the gate. No exceptions.
11. **Reading from `archive/v1-original`.** That branch is portfolio history. The new build is independent. If you find yourself tempted to copy from it, stop and ask John.
12. **Running git/gh commands outside the §2.1 carve-out.** Branch create + feature-branch push + `gh pr create` is the entire allowance. No commits, no merges, no force-push, no tags, no switching to existing branches. See section 2.
13. **Committing notebooks without executed outputs.** Empty-output `.ipynb` files do not render on GitHub and erase the portfolio signal. See section 16 (notebook commit policy) — every committed notebook must carry rendered outputs, and `make notebooks` is the canonical regenerate-and-execute command.

---

## 10. Operating Procedure for Each Prompt

When John pastes a prompt, follow this procedure:

### Before starting work

1. **Read this CLAUDE.md in full.** Even if you read it earlier in the session.
2. **Read `docs/CONTRIBUTING.md` in full.** Branching, PR conventions, and the squash-merge-per-prompt model. Non-negotiable from Sprint 1 onwards.
3. **Read `docs/CONVENTIONS.md`** if it exists (it duplicates parts of this file but may have updates).
4. **Read the most recent `sprints/sprint_X/prompt_Y_report.md`** to understand current state.
5. **Read any files the prompt explicitly references.**
6. **Read `configs/schemas.yaml`** to know current schema versions.
7. **Read `src/fraud_engine/config/settings.py`** to know current configuration surface.
8. State explicitly: "I have read CLAUDE.md, CONTRIBUTING.md, and the prior completion report. Current state is: [one-sentence summary]. Proceeding with [prompt task]."

### During work

1. **Plan first.** For any non-trivial prompt, output a numbered plan before writing code. Wait for John to confirm before proceeding if the prompt is ambiguous.
2. **Small, verifiable steps.** Do not write 1000 lines and then run tests. Write a module, test it, move on.
3. **Run tests after each meaningful change.** `make test-fast` is fast enough to run constantly.
4. **Run linters before declaring a file done.** `make lint && make typecheck`.
5. **Do not invent file contents.** If you need to know what's in a file, read it.
6. **Do not invent library APIs.** If unsure, check the actual installed version's docs or source.

### Before declaring complete

1. Run the full verification protocol (section 11).
2. Write `sprints/sprint_X/prompt_Y_report.md` with: what was built, files changed, test results, deviations from prompt, anything John should know.
3. State explicitly: "Verification passed. Ready for John to commit on `<feature-branch>`."
4. **Wait for John to commit.** Do not stage. Do not commit. Do not suggest a commit message unless asked.
5. Once John has committed, run the §2.4 push + `gh pr create` sequence and post the PR URL back. Then stop — the merge is John's.

---

## 11. Verification Protocol

After every prompt, before declaring complete, run all of these. All must pass.

```bash
# 1. Lint
make lint

# 2. Type check
make typecheck

# 3. Fast unit tests
make test-fast

# 4. Lineage tests (data contracts)
make test-lineage

# 5. Verification script (sprint-specific; created in Sprint 0)
python scripts/verify_bootstrap.py     # for Sprint 0
# or
python scripts/verify_lineage.py       # for Sprint 1+

# 6. If any notebook was touched: rebuild + execute every committable
# notebook in place. Builder defaults to executing after writing, so
# this is also the way to update the .ipynb files for commit. See §16.
make notebooks
make nb-test
```

If any return non-zero, do not proceed. Diagnose. Fix. Re-run. Only after a clean green run is the prompt complete.

For sprints that touch the full pipeline, also run:

```bash
make test-integration
```

This is slower (~2-5 min) but catches multi-component bugs that unit tests miss.

---

## 12. Common Commands

```bash
# Dependency management (uv)
uv sync                                 # install from lockfile
uv add <package>                        # add a dependency
uv run python -m <module>               # run a module in the env

# Testing
make test                               # full suite
make test-fast                          # unit only
make test-integration                   # integration only
make test-lineage                       # lineage only
pytest tests/unit/test_X.py -v          # one file
pytest -k "test_name_substring" -v      # filter

# Linting and types
make lint
make format                             # ruff format (auto-fix)
make typecheck

# Data
make data-download                      # downloads IEEE-CIS via Kaggle API
python scripts/profile_raw.py           # profiling report

# Local services (Redis, Postgres, MLflow, Prometheus, Grafana)
docker compose -f docker-compose.dev.yml up -d
docker compose -f docker-compose.dev.yml down

# API
make serve                              # uvicorn with reload
docker compose up                       # production-like

# Cleanup
make clean                              # caches, build artifacts
```

---

## 13. Sprint Status

Update this table as sprints complete. Read it at the start of every session.

|Sprint|Status|Notes|
|---|---|---|
|0 — Foundation & Environment|Not started|Bootstrap, data acquisition, observability|
|1 — Data Profiling, EDA & Baseline|Not started|EDA + temporal split + baseline LightGBM|
|2 — Feature Engineering Tiers 1-3|Not started|Basic, aggregations, behavioral|
|3 — Advanced Features & Models|Not started|Tier 4-5 + LightGBM tuned + NN + GNN|
|4 — Economic Evaluation|Not started|Cost function + threshold + stratified|
|5 — Production API & Shadow Mode|Not started|FastAPI + Redis + SHAP + shadow|
|6 — Monitoring & Documentation|Not started|Grafana + drift + model card + README|

---

## 14. Where to Get Context

If you need information not in this file:

- **Master plan:** `docs/PROJECT_PLAN.md` (the full project plan from the .docx, if converted to markdown)
- **Branching, PR, and merge conventions:** `docs/CONTRIBUTING.md` (one branch + one PR per prompt, squash-merge to `main`; non-negotiable from Sprint 1 onwards)
- **Conventions:** `docs/CONVENTIONS.md` (this file's standards section, kept in sync)
- **Architecture decisions:** `docs/ADR/*.md` (one per major decision)
- **Data dictionary:** `docs/DATA_DICTIONARY.md` (every feature group, source, meaning)
- **Sprint history:** `sprints/sprint_X/prompt_Y_report.md`
- **Schemas:** `configs/schemas.yaml`
- **Configuration surface:** `src/fraud_engine/config/settings.py`
- **Logging conventions:** `docs/OBSERVABILITY.md`
- **Operations runbook:** `docs/RUNBOOK.md` (Sprint 6 onwards)

If the answer is in none of these and not derivable from code, ask John. Do not guess.

---

## 15. Final Reminders

- **Git/gh: only the §2.1 carve-out.** Branch create + feature-branch push + `gh pr create`. Commits, merges, tags, force-push are John's, every time.
- **Every Python file starts with `from __future__ import annotations`.**
- **Every function has a Google-style docstring with business rationale and trade-offs.**
- **Zero hardcoded values outside config.**
- **Zero `print()` statements. Use the logger.**
- **Temporal integrity tests are mandatory for any feature using time-windowed data.**
- **Notebooks ship with rendered outputs.** Every committed `.ipynb` is regenerated and executed via `make notebooks` before the PR is opened. See §16.
- **Long-running WSL commands.** Any `wsl -d Ubuntu bash -lc '<cmd>'` that may run >60s — most notably `make test-lineage` and `make test` — must use the detached-daemon polling pattern in §17. A blocking call that produces no stdout for ~60s is killed by the WSL service with `Wsl/Service/E_UNEXPECTED`. `vmIdleTimeout=-1`, the `cd /c/Users/dchit && ` UNC-cwd workaround, and `nohup`/`setsid`/`disown` *inside* a foreground call are all individually insufficient.
- **Read this file at the start of every session.**

---

## 16. Notebook Commit Policy

**Every committed `.ipynb` carries rendered outputs.** Notebooks are part of the portfolio surface — GitHub renders them with their stored outputs as the public face of the EDA, profiling, and demo work. An empty-output notebook on `main` erases the very signal it was meant to convey.

### 16.1 The rule

A notebook is committable only after it has been executed end-to-end against the current code, with all cells producing fresh outputs. No "I'll regenerate before merging" exceptions — treat a PR-ready notebook the way a PR-ready test is treated: green, in place, before the PR is opened.

### 16.2 The canonical command

```bash
make notebooks
```

Rebuilds and executes every committable notebook in place. After this returns 0:

- `notebooks/01_eda.ipynb` is regenerated from `scripts/_build_eda_notebook.py` and executed by the builder's atomic build-and-execute step.
- `notebooks/00_observability_demo.ipynb` is executed in place via `jupyter nbconvert --execute --inplace`.

Every additional committable notebook added in a future sprint must be appended to this Makefile target — single point of enforcement.

### 16.3 Why the builder is atomic

`scripts/_build_eda_notebook.py` writes the notebook structure and then immediately runs `jupyter nbconvert --execute --inplace` on it before exiting. Running the builder always produces a notebook with rendered outputs; there is no intermediate "wrote but not executed" state to forget about.

For fast iteration, the builder accepts `--no-execute`, which skips the execute step. The resulting notebook lacks outputs and **must not be committed.** The builder prints a reminder when invoked with that flag.

### 16.4 Verification gate

Step 6 of the [verification protocol](#11-verification-protocol) runs `make notebooks` followed by `make nb-test`. The first regenerates and executes; the second runs the `nbmake` smoke harness as an independent gate. Both must return 0 before the PR is opened.

### 16.5 What this rule prevents

Past failures: empty-output notebooks merged into `main`, breaking GitHub's notebook renderer; reviewers seeing only code without the plots that justify the conclusions; "I'll re-run later" turning into "I forgot, and now the outputs disagree with the report." The policy is process-level enforcement against all three.

---

## 17. WSL Long-Running-Command Pattern

A `wsl -d Ubuntu bash -lc '<cmd>'` invocation whose foreground process produces no stdout for ~60 seconds is killed by the WSL service with the error `Catastrophic failure: Wsl/Service/E_UNEXPECTED`. `vmIdleTimeout=-1` in `~/.wslconfig` keeps the VM alive but does not prevent the per-invocation kill. This affects any verification gate that exceeds the window — most notably `make test-lineage` (≈ 3 min total, with a ~2 min silent parquet fixture load before the first test prints).

### 17.1 The pattern

For runs that may exceed ~60 seconds, launch the work as a fully-detached daemon and poll its log file from short subsequent calls.

**One call to start the daemon (returns in <5 s):**

```bash
wsl -d Ubuntu bash -lc "rm -f ~/run.log ~/run.done; \
  nohup bash -c '<your-command>; echo EXIT=\$? > ~/run.done' \
  > ~/run.log 2>&1 < /dev/null & disown; echo STARTED"
```

**Then poll with short calls (each completes in <5 s, well under the kill window):**

```bash
wsl -d Ubuntu bash -lc \
  "if [ -f ~/run.done ]; then cat ~/run.done; tail -40 ~/run.log; \
   else echo RUNNING; tail -3 ~/run.log; fi"
```

The detached process reparents to `init` (PID 1) and runs to completion in the long-lived WSL VM regardless of whether any `wsl.exe` host process is currently attached. The poll calls are short enough that they never hit the kill window.

### 17.2 What does NOT work

Each of these alone is insufficient and was empirically falsified during prompt 1.2.b:

- `vmIdleTimeout=-1` (already set; the VM stays up but the foreground call still gets killed).
- `cd /c/Users/dchit && wsl …` to move Git Bash off the UNC share (does not affect the kill — the kill is a WSL-service action, not an SMB-session action).
- `nohup`, `setsid`, `disown` *inside* a still-foreground `wsl` invocation (the host call blocks waiting for the bash session to exit, and the kill fires before that).
- `Monitor` tailing a log via the Bash tool while the foreground call is running (the foreground call still gets killed).

Only `nohup … & disown; echo STARTED` followed by polling works reliably.

### 17.3 When to use it

- `make test-lineage`, `make test-integration`, `make test` — anything > 60 s.
- Any one-off long pipeline run (full-data loader, baseline training, Optuna sweeps).

Short commands (lint, typecheck, single-file unit tests, `git status`) run fine inline.

---

_End of CLAUDE.md. Last updated: Sprint 1 prompt 1.2.b — added §17 WSL long-running-command pattern (and the §15 reminder pointing at it)._
