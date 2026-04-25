# Sprint 0 — Gate 0.1 Report

**Task:** Verification-only checkpoint before Sprint 1 (data
acquisition).
**Date:** 2026-04-21
**Branch state:** uncommitted; ready for John
**Prompts covered:** 0.1.a → 0.1.h
**Gate verdict:** ✅ **PASS**

---

## 1. Verification commands

Four commands from the spec. All exited `0`; logs below are the
**complete, unedited** output captured from a fresh WSL/Ubuntu session.

### 1.1 `uv run python scripts/verify_bootstrap.py`

Exit code: `0`

```
{"n_checks": 5, "fail_fast": false, "event": "verify.start", "pipeline": "verify_bootstrap", "run_id": "dc370a44bb9b4a378f081c649ba44975", "logger": "__main__", "level": "info", "timestamp": "2026-04-21T19:09:50.518782Z"}
{"check": "ruff", "event": "verify.check.start", "pipeline": "verify_bootstrap", "run_id": "dc370a44bb9b4a378f081c649ba44975", "logger": "__main__", "level": "info", "timestamp": "2026-04-21T19:09:50.519426Z"}
{"check": "ruff", "ok": true, "duration_s": 0.118, "event": "verify.check.done", "pipeline": "verify_bootstrap", "run_id": "dc370a44bb9b4a378f081c649ba44975", "logger": "__main__", "level": "info", "timestamp": "2026-04-21T19:09:50.637899Z"}
{"check": "format", "event": "verify.check.start", "pipeline": "verify_bootstrap", "run_id": "dc370a44bb9b4a378f081c649ba44975", "logger": "__main__", "level": "info", "timestamp": "2026-04-21T19:09:50.638166Z"}
{"check": "format", "ok": true, "duration_s": 0.058, "event": "verify.check.done", "pipeline": "verify_bootstrap", "run_id": "dc370a44bb9b4a378f081c649ba44975", "logger": "__main__", "level": "info", "timestamp": "2026-04-21T19:09:50.695919Z"}
{"check": "mypy", "event": "verify.check.start", "pipeline": "verify_bootstrap", "run_id": "dc370a44bb9b4a378f081c649ba44975", "logger": "__main__", "level": "info", "timestamp": "2026-04-21T19:09:50.696262Z"}
{"check": "mypy", "ok": true, "duration_s": 3.34, "event": "verify.check.done", "pipeline": "verify_bootstrap", "run_id": "dc370a44bb9b4a378f081c649ba44975", "logger": "__main__", "level": "info", "timestamp": "2026-04-21T19:09:54.036404Z"}
{"check": "pytest", "event": "verify.check.start", "pipeline": "verify_bootstrap", "run_id": "dc370a44bb9b4a378f081c649ba44975", "logger": "__main__", "level": "info", "timestamp": "2026-04-21T19:09:54.036751Z"}
{"check": "pytest", "ok": true, "duration_s": 12.668, "event": "verify.check.done", "pipeline": "verify_bootstrap", "run_id": "dc370a44bb9b4a378f081c649ba44975", "logger": "__main__", "level": "info", "timestamp": "2026-04-21T19:10:06.705555Z"}
{"check": "settings", "event": "verify.check.start", "pipeline": "verify_bootstrap", "run_id": "dc370a44bb9b4a378f081c649ba44975", "logger": "__main__", "level": "info", "timestamp": "2026-04-21T19:10:06.706390Z"}
{"check": "settings", "ok": true, "duration_s": 0.222, "event": "verify.check.done", "pipeline": "verify_bootstrap", "run_id": "dc370a44bb9b4a378f081c649ba44975", "logger": "__main__", "level": "info", "timestamp": "2026-04-21T19:10:06.928736Z"}

[ OK ] ruff       ( 0.12s)
[ OK ] format     ( 0.06s)
[ OK ] mypy       ( 3.34s)
[ OK ] pytest     (12.67s)
[ OK ] settings   ( 0.22s)

Bootstrap: GREEN
```

### 1.2 `uv run make lint`

Exit code: `0`

```
uv run ruff check src tests scripts
All checks passed!
```

### 1.3 `uv run make typecheck`

Exit code: `0`

```
uv run mypy src
Success: no issues found in 20 source files
```

### 1.4 `uv run make test-fast`

Exit code: `0`

```
uv run python -m pytest tests/unit -q --no-cov
........................................................................ [ 50%]
......................................................................   [100%]
=============================== warnings summary ===============================
.venv/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:64
  /home/dchit/projects/fraud-detection-engine/.venv/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:64: PyparsingDeprecationWarning: 'oneOf' deprecated - use 'one_of'
    prop = Group((name + Suppress("=") + comma_separated(value)) | oneOf(_CONSTANTS))

.venv/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:85
.venv/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:85
.venv/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:85
.venv/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:85
.venv/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:85
.venv/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:85
  /home/dchit/projects/fraud-detection-engine/.venv/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:85: PyparsingDeprecationWarning: 'parseString' deprecated - use 'parse_string'
    parse = parser.parseString(pattern)

.venv/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:89
.venv/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:89
.venv/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:89
.venv/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:89
.venv/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:89
.venv/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:89
  /home/dchit/projects/fraud-detection-engine/.venv/lib/python3.11/site-packages/matplotlib/_fontconfig_pattern.py:89: PyparsingDeprecationWarning: 'resetCache' deprecated - use 'reset_cache'
    parser.resetCache()

.venv/lib/python3.11/site-packages/matplotlib/_mathtext.py:45
  /home/dchit/projects/fraud-detection-engine/.venv/lib/python3.11/site-packages/matplotlib/_mathtext.py:45: PyparsingDeprecationWarning: 'enablePackrat' deprecated - use 'enable_packrat'
    ParserElement.enablePackrat()

tests/unit/test_baseline.py: 6 warnings
tests/unit/test_mlflow_setup.py: 4 warnings
  /home/dchit/projects/fraud-detection-engine/.venv/lib/python3.11/site-packages/mlflow/tracking/_tracking_service/utils.py:184: FutureWarning: The filesystem tracking backend (e.g., './mlruns') is deprecated as of February 2026. Consider transitioning to a database backend (e.g., 'sqlite:///mlflow.db') to take advantage of the latest MLflow features. See https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store for migration guidance.
    return FileStore(store_uri, store_uri)

tests/unit/test_splits.py::TestTemporalSplit::test_split_empty_val_raises
tests/unit/test_splits.py::TestTemporalSplit::test_split_empty_test_raises
tests/unit/test_splits.py::TestTemporalSplit::test_split_empty_train_raises
tests/unit/test_splits.py::TestTemporalSplit::test_split_missing_column_raises
  /home/dchit/projects/fraud-detection-engine/.venv/lib/python3.11/site-packages/structlog/stdlib.py:1148: UserWarning: Remove `format_exc_info` from your processor chain if you want pretty exceptions.
    ed = p(logger, meth_name, cast(EventDict, ed))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
142 passed, 28 warnings in 6.14s
```

**Totals:** 142 passed, 0 failed, 0 skipped, 0 xfail, 28 warnings
(all third-party or cosmetic — see §4).

---

## 2. File counts per directory

Top-level directories only (excluding `__pycache__`, `.pytest_cache`,
`.mypy_cache`, `.ruff_cache`, and `mlruns`):

| Directory    | Files |
|--------------|-------|
| `configs/`   | 2     |
| `data/`      | 8     |
| `docker/`    | 2     |
| `docs/`      | 4     |
| `notebooks/` | 3     |
| `scripts/`   | 6     |
| `src/`       | 20    |
| `tests/`     | 20    |
| `sprints/`   | 18    |
| `.github/`   | 1     |
| *(top-level files)* | 15 |

Note: `data/` contains only the gitignored raw-data `MANIFEST.json`
and its profiling artefacts — the 590k-row IEEE-CIS parquets live
outside the repo per `.gitignore`. `mlruns/` is likewise gitignored
and omitted from the count.

---

## 3. Lines of code in `src/`

`find src -name '*.py' | xargs wc -l` output (verbatim):

```
     3 src/fraud_engine/api/__init__.py
     7 src/fraud_engine/__init__.py
    16 src/fraud_engine/models/__init__.py
   372 src/fraud_engine/models/baseline.py
     3 src/fraud_engine/evaluation/__init__.py
   362 src/fraud_engine/config/settings.py
     7 src/fraud_engine/config/__init__.py
   257 src/fraud_engine/schemas/raw.py
    17 src/fraud_engine/schemas/__init__.py
    20 src/fraud_engine/data/__init__.py
   296 src/fraud_engine/data/loader.py
   317 src/fraud_engine/data/splits.py
   306 src/fraud_engine/utils/metrics.py
    51 src/fraud_engine/utils/__init__.py
   506 src/fraud_engine/utils/logging.py
   210 src/fraud_engine/utils/mlflow_setup.py
    92 src/fraud_engine/utils/seeding.py
   378 src/fraud_engine/utils/tracing.py
     3 src/fraud_engine/features/__init__.py
     3 src/fraud_engine/monitoring/__init__.py
  3226 total
```

**Headline:** **3 226 lines** of Python across **20 files** in 9
subpackages, with `mypy --strict` clean on all of them.

The top five modules by LOC — `logging.py` (506), `tracing.py` (378),
`baseline.py` (372), `settings.py` (362), `splits.py` (317) — are
the Sprint 0 and Sprint 1 load-bearing modules; every other file is a
small `__init__.py` or a narrow utility.

---

## 4. Warnings triage (from `test-fast`)

All 28 warnings are **accepted** — none is actionable at the Sprint 0
gate.

| Count | Source | Disposition |
|-------|--------|-------------|
| 17 | matplotlib `PyparsingDeprecationWarning` in `_fontconfig_pattern.py` / `_mathtext.py` | Upstream library; surfaces whenever matplotlib is imported (EDA notebook path). No code of ours is involved. Will clear when matplotlib catches up with pyparsing. |
| 10 | mlflow `FutureWarning` about the `./mlruns` file-store backend being deprecated (February 2026) | Known — surfaced in `test_baseline.py` (6) and `test_mlflow_setup.py` (4). Migration to SQLite/remote is a Sprint 5/6 concern once the tracking server is stood up. Tracked as a TODO (§6). |
| 4 | structlog `UserWarning: Remove format_exc_info from your processor chain` on 4 `TemporalSplit` `.raises` tests in `test_splits.py` | Cosmetic — structlog's tip fires because `format_exc_info` is in the stdlib bridge chain. Removing it would change exception rendering in prod logs; the test-only noise is not worth the trade. Tracked as a TODO (§6). |

No warning represents a test failure, silent pass, or masked bug.

---

## 5. Deviations from prompt 0.1.a → 0.1.h

Each deviation is already documented in the per-prompt report; listed
here as a consolidated audit trail.

| Prompt | Deviation | Rationale | Report |
|--------|-----------|-----------|--------|
| 0.1.a | `tests/**/__init__.py` carry `from __future__ import annotations` instead of being empty | Consistency with CLAUDE.md §5.1 and the `src/` convention | [0_1_a](prompt_0_1_a_report.md) |
| 0.1.a | Pre-existing `.gitkeep` markers left in populated dirs (`notebooks/`, `tests/integration/`, `tests/lineage/`, `sprints/sprint_0/`) | Cleanup was out of scope; removing them requires `git rm` which is prohibited by CLAUDE.md §2 | [0_1_a](prompt_0_1_a_report.md) |
| 0.1.b | Ruff ruleset extended to `E,F,I,N,UP,B,SIM,ARG,RET,PTH,PL` (spec listed a narrower set) | Caught real issues (PLR2004, PLR0402, SIM105) in pre-existing code | [0_1_b](prompt_0_1_b_report.md) |
| 0.1.f | `conftest.py` added IEEE-CIS-shaped `tiny_transactions_df` fixture while leaving the pre-existing fictional `small_transactions_df` / `small_identity_df` fixtures in place | Fixtures are dead-weight with zero consumers; removal was out of scope for 0.1.f | [0_1_f](prompt_0_1_f_report.md) |
| 0.1.f | `verify_bootstrap.py` runs the five checks directly rather than shelling out to `make lint/typecheck/test-fast` | One fewer indirection layer in CI logs; `make` targets still work for humans | [0_1_f](prompt_0_1_f_report.md) |
| 0.1.g | CI workflow calls `ruff check`, `ruff format --check`, `mypy src`, `pytest … --no-cov`, `verify_bootstrap.py` directly — not via `make` | Matches the Makefile targets verbatim; direct commands make CI logs more readable | [0_1_g](prompt_0_1_g_report.md) |
| 0.1.g | Codecov upload step present but gated `if: ${{ false }}` | Spec said "keep the step but skip until token is wired"; flipping it live is a documented two-line diff | [0_1_g](prompt_0_1_g_report.md) |
| 0.1.g | `detect-secrets` pre-commit hook prints 15× "unstaged baseline" on re-runs | Cosmetic workflow artefact; overall pre-commit exit 0; resolves when John `git add`s the refreshed baseline | [0_1_g](prompt_0_1_g_report.md) |
| 0.1.h | `docs/CONVENTIONS.md` is a **verbatim** mirror of CLAUDE.md §5/6/7/9 rather than a paraphrase | Spec said "copy"; verbatim makes drift detectable via plain `diff` | [0_1_h](prompt_0_1_h_report.md) |
| 0.1.h | ADR keeps per-component "Why …" rationale subsections in addition to the four top-level sections | Existing structure is richer than the template; preserved rather than flattened | [0_1_h](prompt_0_1_h_report.md) |

No deviation changes the factual content or the acceptance surface of
any prompt.

---

## 6. TODOs carried into Sprint 1+

These are explicitly deferred, not dropped. Each has an anchor in
the codebase or a report entry.

1. **Codecov upload** — [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
   step is inert pending `CODECOV_TOKEN` in GitHub Actions secrets.
   Enable path documented inline and in [0_1_g §2.1](prompt_0_1_g_report.md#21-gap-fill--codecov-step).
2. **Docker stack bring-up** — `docker-compose.dev.yml` stands (Redis,
   Postgres, MLflow, Prometheus, Grafana). Local bring-up is deferred
   until end of project per John's direction (local machine issue);
   no Sprint 0 test depends on the stack being up.
3. **MLflow filesystem backend deprecation** — 10 `FutureWarning`s in
   tests. Migration to SQLite/remote tracking is a Sprint 5/6 concern
   once the serving/monitoring layer lands.
4. **structlog `format_exc_info` tip** — 4 `UserWarning`s on split
   validation tests. Processor chain is intentional for production
   log shape; test-only noise will be filtered once we own the
   pytest warning config (Sprint 1+).
5. **Dead-weight `.gitkeep` markers** — 4 files in populated dirs
   (§5 of this report, 0.1.a deviation). Cleanup is a one-line
   `git rm` John can take whenever convenient; low priority.
6. **Unused fictional fixtures** — `small_transactions_df` /
   `small_identity_df` in [conftest.py](../../tests/conftest.py).
   Zero consumers; delete when Sprint 1 fixture surface stabilises.

None of these blocks Sprint 1 entry.

---

## 7. Gate acceptance

- [x] `verify_bootstrap.py` — Bootstrap: GREEN (§1.1).
- [x] `make lint` — "All checks passed!" (§1.2).
- [x] `make typecheck` — "no issues found in 20 source files" (§1.3).
- [x] `make test-fast` — 142 passed, 0 failed, 0 skipped (§1.4).
- [x] File counts per directory captured (§2).
- [x] `wc -l` on `src/*.py` captured — 3 226 total (§3).
- [x] Warnings triaged (§4).
- [x] Deviations and TODOs consolidated (§5 – §6).
- [x] Gate report written.

**Verdict: PASS.** Sprint 0 is complete. Ready for Sprint 1 (data
profiling, EDA, temporal split, baseline).

**No git action from me** (CLAUDE.md §2). Ready for John to commit.
