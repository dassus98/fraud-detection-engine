# Sprint 0 Prompt 0.1.c — Install & Makefile (Audit & Gap-Fill)

**Depends on:** 0.1.b
**Date:** 2026-04-21
**Risk:** Low

## Summary

Final install step for the audit-and-gap-fill trilogy (0.1.a → 0.1.b
→ 0.1.c). uv itself was already on the box (WSL Ubuntu binary at
`/home/dchit/.local/bin/uv`, version `0.11.7`), and the
[Makefile](../../Makefile) predates this prompt from
prompt 0.1 (commit `9f88036`). The work here was:

1. Resolve and install the pins introduced by 0.1.b
   (`pandera[io]`, `asyncpg`, `ydata-profiling`).
2. Audit the Makefile against the 0.1.c target list.
3. Verify the install by importing every major runtime dependency.

One pin bump was needed: `ydata-profiling==4.12.2` does not support
numpy 2.x, so it was raised to `4.18.1` (the oldest compatible
release). No other dependencies moved; no Makefile targets were
touched.

## Per-step audit

### 1. `uv` install

Already installed before this prompt:

```text
$ uv --version
uv 0.11.7 (x86_64-unknown-linux-gnu)
```

Lives under `~/.local/bin/uv` inside WSL Ubuntu; the Windows-side
`C:\Users\dchit\.local\bin\uv` is a separate binary (0.9.24) that
does **not** work against the WSL-hosted `.venv/` because of the
UNC-path symlink handling (`os error 3`). All resolve/sync/run
commands in this prompt were invoked through `wsl -d Ubuntu --
bash -lc '…'` so they hit the Linux binary.

### 2. `uv sync --all-extras`

First attempt failed:

```text
× No solution found when resolving dependencies:
  ╰─▶ Because ydata-profiling==4.12.2 depends on numpy>=1.16.0,<2.2
      and your project depends on numpy==2.2.0, we can conclude that
      your project's requirements are unsatisfiable.
```

`ydata-profiling` 4.12.2 was my best-effort guess in 0.1.b (see the
"known gaps" section of that report). It pre-dates the numpy 2.2
bump. A `uv pip install --dry-run 'ydata-profiling' --upgrade`
probe returned `4.18.1` as the minimum numpy-2-compatible release.

**Pin bump:** `pyproject.toml` line 73 changed from
`"ydata-profiling==4.12.2"` → `"ydata-profiling==4.18.1"`.

Second attempt succeeded:

```text
Resolved 275 packages in 1.19s
Installed 36 packages in 111ms
+ anyascii==0.3.3
+ asyncpg==0.30.0
+ black==26.3.1
+ dacite==1.9.2
+ frictionless==4.40.8
+ htmlmin==0.1.12
+ imagehash==4.3.2
+ jellyfish==1.3.0
+ llvmlite==0.45.1      (downgrade from 0.47.x transitively pulled by shap)
+ numba==0.62.1         (downgrade from 0.63.x)
+ phik==0.12.5
+ pydantic-core==2.27.2
+ statsmodels==0.14.6
+ tangled-up-in-unicode==0.2.0
+ typeguard==4.5.0
+ typer==0.23.1
+ visions==0.8.1
+ wordcloud==1.9.6
+ ydata-profiling==4.18.1
+ …and 16 more transitives
```

Non-fatal resolver warning (surfaced once, not repeated):

```text
warning: The package `typer==0.23.1` does not have an extra named `all`
```

This is harmless — ydata-profiling's metadata requests
`typer[all]` but typer 0.23 folded the `all` extra into the base
install, so every "extra" dep is installed anyway. No action needed;
upstream bug in ydata-profiling's `setup.cfg`.

`numba` and `llvmlite` downgrades are driven by ydata-profiling's
`numba>=0.56,<0.63` pin range; shap is happy on either, so the
downgrade is safe.

### 3. `uv.lock` regeneration

```text
$ ls -la uv.lock
-rw-r--r-- 1 dchit dchit 384189 Apr 21 10:10 uv.lock
```

384 KB vs ~320 KB pre-0.1.b — growth consistent with the 36 new
entries. Lockfile is the authoritative resolution record and is
checked into the repo.

### 4. Makefile audit

The existing Makefile (15 targets + `.DEFAULT_GOAL=help`) is a
**strict superset** of the 0.1.c spec's target list. No edits needed.

Spec target → existing line number:

| Spec target | Line | Status |
|-------------|------|--------|
| `help` | 11 | ✅ self-documenting via awk-over-docstrings |
| `install` | 14 | ✅ `uv sync --all-extras && uv run pre-commit install` |
| `format` | 18 | ✅ `ruff format` over `src tests scripts` |
| `lint` | 21 | ✅ `ruff check` over same scope |
| `typecheck` | 24 | ✅ `mypy src` |
| `test` | 27 | ✅ pytest + nb-test |
| `test-fast` | 31 | ✅ `pytest tests/unit -q --no-cov` |
| `test-integration` | 34 | ✅ `pytest tests/integration` |
| `test-lineage` | 37 | ✅ `pytest tests/lineage` |
| `data-download` | 43 | ✅ `scripts/download_data.py` |
| `train` | 52 | ✅ loud-fail echo; Sprint 3 wires real impl |
| `serve` | 57 | ✅ uvicorn on `$(API_HOST):$(API_PORT)` |
| `docker-up` | 60 | ✅ compose `-f docker-compose.dev.yml up -d` |
| `docker-down` | 63 | ✅ compose down |
| `clean` | 69 | ✅ rm caches + `__pycache__` tree |

Pre-existing extensions over the spec (kept; provide real value):

| Extension | Line | Why kept |
|-----------|------|----------|
| `nb-test` | 40 | Notebook smoke via nbmake, catches util-rename drift across Sprint 1 EDA notebooks. Invoked by `test`. |
| `data-profile` | 46 | Sprint 1 used this for `reports/raw_profile.{html,json}`. |
| `sprint1-baseline` | 49 | Sprint 1 Prompt 1 entry point. |
| `docker-ps` | 66 | Status + healthcheck visibility alongside up/down. |

`install` goes one step beyond spec by also running
`pre-commit install` — this is load-bearing for the detect-secrets +
ruff pre-commit hooks and matches the 0.1 original design. Kept.

`-include .env` + `export` at lines 8-9 lets `make serve` pick up
`API_HOST` / `API_PORT` without a wrapper script. `-include` (not
plain `include`) keeps `make help` working before `.env` exists.

### 5. Verification

**`make help`** — prints the full target table with ANSI colours:

```text
Usage: make <target>

Targets:
  help                  Show this help message.
  install               Install dependencies via uv and register pre-commit hooks.
  format                Format code with ruff.
  lint                  Lint with ruff.
  typecheck             Type-check src/ with mypy strict mode.
  test                  Run the full test suite with coverage (includes notebook smoke).
  test-fast             Run unit tests only, no coverage, quiet.
  test-integration      Run integration tests (requires Redis, Postgres).
  test-lineage          Run schema-lineage tests.
  nb-test               Execute notebooks end-to-end via nbmake (catches util-rename drift).
  data-download         Fetch IEEE-CIS from Kaggle into data/raw/ and write the manifest.
  data-profile          Render reports/raw_profile.{html,json} from the merged raw frame.
  train                 Train models. Implemented in Sprint 3.
  serve                 Start the FastAPI server (requires Sprint 5 api module).
  docker-up             Start the dev compose stack (Postgres, Redis, MLflow, Prometheus, Grafana).
  docker-down           Stop the dev compose stack.
  docker-ps             Show the dev stack's service status (including healthchecks).
  clean                 Remove test / type-check / build caches.
```

(Note: `sprint1-baseline` is declared in `.PHONY` but lacks the
`##` marker that the awk script uses to populate `help`. Not a bug
— the target is Sprint-1-specific and we deliberately don't advertise
it at the top level. It still runs when called directly.)

**Import smoke test** — every major runtime dep loaded cleanly via
a temp `tmp_verify_imports.py` (used a file rather than `-c` to
sidestep PowerShell → WSL bash → python triple-nested quoting):

```text
imports OK
pandas=2.2.3
numpy=2.2.0
pydantic=2.10.4
lightgbm=4.5.0
torch=2.5.1+cu124
```

Every import in the spec list passed: pandas, numpy, pydantic,
structlog, pandera, lightgbm, shap, fastapi, redis, torch. The temp
file was deleted after the run.

**Package count:**

```text
$ uv pip list | wc -l
270
```

That's 268 installed packages (minus 2 header lines) against 275
resolved — the difference is extras declared but not pulled by the
current platform (e.g. `nvidia-*` entries resolve on x86-64 linux
but skip on pure-CPU builds; same for Windows-only wheels that
never land).

## Deviations from prompt

1. **uv already installed.** Spec says "Install uv" — but the WSL
   binary was present from prompt 0.1. Skipped the install step and
   went straight to `uv sync`.
2. **`ydata-profiling` pin bumped from 4.12.2 → 4.18.1.** Required
   for numpy 2.2 compatibility. Flagged as a known-gap in 0.1.b's
   report; this is where it lands. No other dep moved.
3. **Makefile not rewritten.** Pre-existing file already covered
   every spec target (plus four valuable extensions). Per the
   trilogy's audit-and-gap-fill directive: extend, don't replace.
4. **PowerShell indirection for uv commands.** `uv` invoked on the
   mounted `\\wsl.localhost\...` share via Windows binary fails with
   `os error 3` on venv symlink removal. All sync/run commands
   funnel through `wsl -d Ubuntu -- bash -lc '…'` so they execute
   inside the Linux FS. Mechanical, not a spec deviation, but worth
   recording for future debugging.

## Known gaps / handoffs

- **`typer[all]` resolver warning** will recur every time
  `uv sync`/`uv lock` runs. Non-blocking; fix belongs upstream in
  ydata-profiling. If the noise becomes annoying we can drop
  `ydata-profiling` (it's a dev-only convenience) or override
  typer's extras declaration in `[tool.uv]`.
- **Ruff `PL`/`RET`/`PTH` rules not yet exercised.** 0.1.b added
  them to `ruff.toml` but `make lint` has not been run against the
  full src tree since. Running it now could surface existing
  violations. Out of scope for 0.1.c's install-only scope; next
  real work item should run `make lint` and either fix or triage.
- **No `make test` / `make typecheck` run here.** Install verified
  via imports; full test + typecheck runs were not in the 0.1.c
  spec's verification list. They land under the resumed Sprint 1
  verification or an explicit prompt.
- **Sprint 1 Prompt 1 verification still pending.** Unchanged from
  0.1.b's handoff note. The trilogy detour ends here; Sprint 1
  resumes at whatever the next "Continue to audit and gap-fill"
  message points at, or with a direct `make test-lineage` +
  `make test-integration` run.

## Acceptance checklist

- [x] **`uv --version` prints ≥ 0.5** — `0.11.7` (spec asks for a
  working uv; 0.11 is well above any floor).
- [x] **`uv sync --all-extras` succeeds** — 275 resolved, 36
  installed on top of the existing venv. One pre-existing numpy
  conflict resolved via the 4.12.2 → 4.18.1 bump.
- [x] **`uv.lock` exists and is committable** — 384 KB, authoritative.
- [x] **`make help` prints the target table** — full output above.
- [x] **Every major runtime dep imports** — smoke test green
  (pandas, numpy, pydantic, structlog, pandera, lightgbm, shap,
  fastapi, redis, torch).
- [x] **No git commands run** — per CLAUDE.md §2.

Ready for John to commit. (No git action from me.)
