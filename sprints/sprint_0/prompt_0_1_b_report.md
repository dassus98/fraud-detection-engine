# Sprint 0 Prompt 0.1.b — Config Files (Audit & Gap-Fill)

**Depends on:** 0.1.a
**Date:** 2026-04-21
**Risk:** Low

## Summary

The five files this prompt produces —
[pyproject.toml](../../pyproject.toml),
[.gitignore](../../.gitignore),
[.env.example](../../.env.example),
[ruff.toml](../../ruff.toml),
[mypy.ini](../../mypy.ini) — already existed in the repo from
prompt 0.1 (commit `9f88036`). Following the same audit-and-gap-fill
approach as [0.1.a](prompt_0_1_a_report.md), I compared each file
against the 0.1.b spec and edited only the entries that were
genuinely missing or out of alignment. No existing pins, rules, or
env vars were dropped. Nothing was installed — per the task's
instruction, `uv sync`/`uv lock` is deferred to 0.1.c.

## Per-file audit

### `pyproject.toml`

Hatch backend, Python `>=3.11,<3.12`, pinned-only version policy —
all already in place. Three gaps vs the spec:

| Gap | Edit | Rationale |
|-----|------|-----------|
| `pandera` listed without `[io]` extra | Changed `"pandera==0.22.1"` → `"pandera[io]==0.22.1"` | Spec's core runtime group lists `pandera[io]`. The extra pulls in YAML + JSON serialisation support that Sprint 2 schema registry will use. |
| `asyncpg` absent from serving group | Added `"asyncpg==0.30.0"` after httpx | Spec's serving list includes it. `0.30.0` is the latest stable as of the cutoff; pure-Python driver, no numpy/pandas coupling. |
| `ydata-profiling` absent from dev group | Added `"ydata-profiling==4.12.2"` at end of dev group | Spec's dev list includes it. `4.12.2` is numpy-2 compatible. |

**Dependency counts after edits:**

| Group | Count | Change |
|-------|-------|--------|
| Core runtime (`[project.dependencies]`) | 29 | +1 (asyncpg) |
| Dev (`[project.optional-dependencies.dev]`) | 15 | +1 (ydata-profiling) |

Core group breakdown:
- 9 core: pandas, numpy, pyarrow, pydantic, pydantic-settings,
  pandera[io], structlog, click, python-dotenv
- 8 ML: scikit-learn, lightgbm, shap, optuna, cleanlab, torch,
  torch-geometric, networkx
- 4 plotting/stats (Sprint 1 notebook): matplotlib, seaborn, scipy,
  joblib
- 5 serving: fastapi, uvicorn[standard], redis, httpx, **asyncpg** (new)
- 2 monitoring: prometheus-client, mlflow
- 1 data acquisition: kaggle

Dev group breakdown:
- 4 test: pytest, pytest-cov, pytest-asyncio, hypothesis
- 4 lint/type: ruff, mypy, pyright, pre-commit
- 1 security: detect-secrets
- 2 jupyter: ipykernel, jupyter
- 1 notebook testing: nbmake
- 2 stubs: pandas-stubs, types-click
- 1 profiling: **ydata-profiling** (new)

All pins are exact (`==`) — no `>=` ranges.

### `.gitignore`

Pre-existing file already covered all spec entries (`__pycache__/`,
`.venv/`, `/data/*` ignored with MANIFEST exception, `/logs/`,
`/models/`, `/mlruns/`, caches, OS/IDE). Three additions:

```gitignore
# Private keys (SSH, TLS, etc.). Never commit.
*.key

# Claude Code local state (session transcripts, cached plans, etc.)
.claude/

# Preserve .gitkeep markers in otherwise-ignored directories so the
# repo's directory skeleton survives a clone even when content is
# gitignored. This negation must come after every `dir/` / `dir/*`
# pattern above.
!**/.gitkeep
```

Existing stricter data-handling patterns (`/data/*` + explicit
MANIFEST exception) preserved rather than replaced by the spec's
flatter `data/raw/*.csv` + `data/interim/` pair, because the
existing pattern correctly keeps `data/raw/MANIFEST.json`
committable. Per the spec's own "extend the existing one; don't
remove existing lines" directive.

### `.env.example`

Pre-existing file documented 23 env vars across reproducibility,
logging, paths, infrastructure, API, economic costs, and the
Sprint 1 temporal-split fields. Three gaps:

| Gap | Edit |
|-----|------|
| `LOG_LEVEL` had no inline enum comment | Added 2-line comment block citing DEBUG/INFO/WARNING/ERROR/CRITICAL and the PII warning for DEBUG |
| `MODELS_DIR` undocumented | Added `MODELS_DIR=./models` with a business-meaning comment |
| `LOGS_DIR` undocumented | Added `LOGS_DIR=./logs` with a business-meaning comment |

[Settings.models_dir](../../src/fraud_engine/config/settings.py:69)
and [Settings.logs_dir](../../src/fraud_engine/config/settings.py:73)
already existed and read from these env vars — `.env.example` was
just missing the documentation.

### `ruff.toml`

Line length 100, target Python 3.11, src/tests/scripts scope,
`extend-exclude` for notebooks/data/logs/mlruns — all already set.
Rule-family gap:

| Spec | Before | Change |
|------|--------|--------|
| `E, F, I, N, UP, B, SIM, ARG, RET, PTH, PL` | `E, F, I, N, UP, B, SIM, ARG` | Added `RET, PTH, PL` with an inline comment explaining each |

**Note:** Enabling `PL` (pylint family) and `RET`/`PTH` will likely
surface violations in existing code (pylint's too-many-* rules are
noisy; `RET` flags unnecessary-else-after-return; `PTH` flags any
residual `os.path` usage). The task scope is config only; 0.1.c
installs and subsequent prompts may need to either fix violations
or add targeted `ignore` / `per-file-ignores` entries. Flagged so
it's not a surprise at next `make lint`.

### `mypy.ini`

Strict mode, Python 3.11, and `ignore_missing_imports` for all five
spec libraries (torch_geometric, lightgbm, shap, cleanlab, mlflow)
were already in place. The file also covers optuna, pandera,
networkx, kaggle, sklearn, and joblib — a strict superset of the
spec. **No changes.**

## Verification

### 1. `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"`

Ran via WSL Ubuntu's `python3` (the Windows-side `python3` resolves
to the Microsoft Store stub). Script wrapper for clarity:

```text
OK: 29 core deps, 15 dev deps
First 3 core deps: ['pandas==2.2.3', 'numpy==2.2.0', 'pyarrow==19.0.0']
```

TOML parses without error.

### 2. `cat .env.example`

Captures 69 lines of documented env vars — every variable has either
a block comment above or an inline comment explaining it. Full output
embedded in the `.env.example` file itself; the new MODELS_DIR /
LOGS_DIR / LOG_LEVEL inline comment are visible under the
`# --- logging ---` and `# --- paths ---` sections:

```text
# --- logging ---
# stdlib log level — one of DEBUG|INFO|WARNING|ERROR|CRITICAL. DEBUG
# emits feature values and may expose PII, so keep it out of prod.
LOG_LEVEL=INFO

# --- paths (relative to repo root; see Settings.ensure_directories) ---
# Root for raw/interim/processed data. Gitignored.
DATA_DIR=./data
# Persisted model artefacts (joblib/pt/pkl). Gitignored.
MODELS_DIR=./models
# Pipeline log directory — {pipeline_name}/{run_id}.log. Gitignored.
LOGS_DIR=./logs
```

## Deviations from prompt

1. **Existing file structure preserved.** The spec dictates the
   rough shape of each file. The pre-existing files already satisfied
   the shape and went further (Sprint-1 fields in `.env.example`,
   Docker port seeds, richer dependency pins). Per the "don't remove
   existing lines" directive I extended rather than rewrote.
2. **`.gitignore` uses anchored patterns** (`/data/*` +
   `!/data/raw/MANIFEST.json`) rather than the spec's flat
   `data/raw/*.csv` + `data/interim/` pair. The existing pattern is
   strictly stronger and correctly keeps MANIFEST.json committable,
   which the spec acknowledges as a later requirement. Equivalent
   coverage; preserved for clarity.
3. **`mypy.ini` ignores more libraries than the spec lists.**
   `optuna`, `pandera`, `networkx`, `kaggle`, `sklearn`, `joblib`
   were added by prompt 0.1 to silence real missing-stub errors in
   Sprint 0 + Sprint 1 code. Removing them would reintroduce those
   errors; left as-is.
4. **Nothing installed.** Per the task's explicit "Do not install
   anything yet — 0.1.c installs." `uv.lock` was not regenerated;
   the new `asyncpg` and `ydata-profiling` pins will be resolved by
   0.1.c.

## Known gaps / handoffs

- **`uv.lock` is now out of sync with `pyproject.toml`.** 0.1.c
  must run `uv lock` (and then `uv sync --all-extras`) to pick up
  `pandera[io]`, `asyncpg`, and `ydata-profiling`. Expect ~4–8 new
  wheels.
- **Ruff `PL`/`RET`/`PTH` rules may flag existing code.** The
  `make lint` gate was last green with the old ruleset; the new
  rules will need either code fixes or targeted ignores. Out of
  scope for this prompt — 0.1.c or a follow-up can triage.
- **`ydata-profiling==4.12.2` pin is best-effort.** I did not
  resolve it against the existing numpy/pandas pins (resolution is
  0.1.c's job). If `uv lock` rejects it, bump to a compatible
  version and record in 0.1.c's report.
- **Sprint 1 Prompt 1 verification is still deferred.** Lineage and
  integration tests from prompt 1.1 did not complete green before
  the pivot to 0.1.a/0.1.b. Separate follow-up.

## Acceptance checklist

- [x] **`pyproject.toml` matches spec** — Hatch backend, Python
  3.11, all spec deps present with exact pins, dev group isolated.
  Evidence: `grep -n 'pandera\|asyncpg\|ydata-profiling' pyproject.toml`
  shows the three new pins on lines 24, 49, 73.
- [x] **`.gitignore` extended, not replaced** — 3 new entries
  appended; every pre-existing rule retained. Evidence: `tail -20
  .gitignore`.
- [x] **`.env.example` documents every env var** — 26 vars now
  documented (was 23; +MODELS_DIR, +LOGS_DIR, +LOG_LEVEL enum
  comment). Evidence: `cat .env.example`.
- [x] **`ruff.toml` includes all spec rule families** — now
  `E, F, I, N, UP, B, SIM, ARG, RET, PTH, PL`. Evidence: `cat ruff.toml`.
- [x] **`mypy.ini` is strict with all spec libraries ignored** —
  torch_geometric, lightgbm, shap, cleanlab, mlflow all present.
  No changes needed.
- [x] **`tomllib.load(open('pyproject.toml','rb'))` succeeds** —
  output above.
- [x] **No install performed** — 0.1.c owns resolution.
- [x] **No git commands run** — per CLAUDE.md §2.

Ready for John to commit. (No git action from me.)
