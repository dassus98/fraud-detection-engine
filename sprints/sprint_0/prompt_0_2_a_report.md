# Sprint 0 — Prompt 0.2.a Report: `scripts/download_data.py` audit & gap-fill

**Date:** 2026-04-21
**Status:** Ready for John to commit. No git action from me. (CLAUDE.md §2.)

## Summary

Prompt 0.2.a specifies a standalone `scripts/download_data.py` CLI that
downloads the IEEE-CIS Fraud Detection archive from Kaggle, extracts the
five CSVs into `data/raw/`, and writes a SHA256 manifest keyed by
filename. The script and its manifest were both landed earlier under
[prompt_0_2_report.md](prompt_0_2_report.md) (commit on
`sprint-0/data-contracts`, 2026-04-18) and have been executed against
the real Kaggle competition — `data/raw/MANIFEST.json` is already in
the tree with real hashes.

The audit found the script 100% functionally equivalent to the spec,
with four small delta surfaces where the existing implementation did
not yet match the 0.2.a prompt verbatim. This prompt closes those four
gaps and re-verifies. No redesign, no re-download.

## Audit: what was pre-existing

[scripts/download_data.py](../../scripts/download_data.py) pre-existed
at 282 lines with a full module docstring including business rationale
and trade-offs. It covered every functional requirement of 0.2.a:

| 0.2.a requirement | Pre-existing behaviour |
|---|---|
| Read `KAGGLE_USERNAME` / `KAGGLE_KEY` from `Settings` | `_configure_kaggle_env(settings)` at line 141 |
| Inject credentials into `os.environ` before import | `os.environ["KAGGLE_USERNAME"]` / `KAGGLE_KEY` set prior to local `from kaggle.api.kaggle_api_extended import KaggleApi` |
| Download `ieee-fraud-detection` competition | `api.competition_download_files(_COMPETITION_SLUG, ...)` |
| Unzip into `data/raw/` | `zipfile.ZipFile(archive).extractall(raw_dir)`; archive unlinked after |
| SHA256 every CSV | `_sha256()` streams in 1 MiB chunks (`_HASH_CHUNK_BYTES = 1 << 20`) |
| Write `MANIFEST.json` | `_write_manifest()` serialises with `json.dumps(..., indent=2) + "\n"` |
| Idempotent skip on intact manifest | `_manifest_matches()` re-hashes every file and compares to recorded entries; called before the Kaggle import |
| `--force` flag | `@click.option("--force", is_flag=True, default=False)` |
| Structured logs with `run_id` | `configure_logging(pipeline_name="data_download")` at entry; every event carries `run_id`, `pipeline`, and timestamp |

The existing [data/raw/MANIFEST.json](../../data/raw/MANIFEST.json)
carries real 2026-04-18 hashes for the five CSVs (see below).

## Gap-fill: what this prompt added

Four small edits to reach verbatim-spec parity. No behavioural change
for existing callers.

### 1. `--output-dir` flag

The spec requires both `--force` and `--output-dir` flags. `--force`
existed; `--output-dir` did not. Added:

```python
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Override settings.raw_dir. Created if missing.",
)
```

When `--output-dir` is omitted, behaviour is unchanged (`settings.raw_dir`).
When supplied, the script downloads to and fingerprints the override
directory instead. The override is `mkdir(parents=True, exist_ok=True)`d
on entry.

**Why `str | None` on the parameter, not `Path | None`:** click's
stubs reject `click.Path(path_type=Path)` under `mypy --strict`
(`type-var` error: `Value of type variable "_PathType" of "Path"
cannot be "Path"`). The simplest fix is to leave the click option
type as the default `str` and convert in the body:
`raw_dir = Path(output_dir) if output_dir is not None else settings.raw_dir`.
This is the idiomatic `click + mypy --strict` pattern and costs one
line.

### 2. `@log_call` on the entry point

The spec requires `@log_call` on `main`. It was not present. Added
the import (`from fraud_engine.utils.logging import configure_logging,
get_logger, log_call`) and decorated the command:

```python
@click.command()
@click.option(...)  # --force
@click.option(...)  # --output-dir
@log_call
def main(force: bool, output_dir: str | None) -> None:
    ...
```

The decorator now emits `main.start` (with kw args logged by type +
value) and `main.done` (with duration_ms) around every invocation.
This is in addition to the existing `download.start` / `download.skip`
/ `download.done` events.

### 3. Entry point renamed `download` → `main`

The spec verifies by `from scripts.download_data import main`. The
existing command was named `download()`. Renamed the function and
updated the `if __name__ == "__main__":` block. No API change for
shell callers (`python scripts/download_data.py` is unchanged).

### 4. `src/fraud_engine/py.typed` (PEP 561 marker)

Running `uv run mypy scripts/download_data.py` standalone failed with
three `import-untyped` errors:

```
error: Skipping analyzing "fraud_engine.config.settings": module is installed,
       but missing library stubs or py.typed marker  [import-untyped]
```

The `fraud_engine` package was strict-typed internally via `[tool.mypy]`
in `pyproject.toml`, but had no `py.typed` marker file, so mypy
treated it as "third-party without stubs" when invoked on a consumer
outside `src/`. Created the empty PEP 561 marker at
[src/fraud_engine/py.typed](../../src/fraud_engine/py.typed). This
cleared all three import-untyped errors and the cascading "Untyped
decorator makes function main untyped" error on `@log_call`.

## Deviation — manifest JSON shape

The spec's illustrative manifest schema is:

```json
{
  "downloaded_at": "...",
  "dataset": "ieee-fraud-detection",
  "files": [
    {"filename": "...", "size_bytes": 123, "sha256": "..."}
  ]
}
```

The existing `data/raw/MANIFEST.json` uses a keyed-dict shape with two
extra top-level fields:

```json
{
  "source": "kaggle:ieee-fraud-detection",
  "downloaded_at": "2026-04-18T20:28:29.000906+00:00",
  "schema_version": 1,
  "files": {
    "sample_submission.csv": {"sha256": "...", "bytes": 6080314},
    "test_identity.csv":     {"sha256": "...", "bytes": 25797161},
    ...
  }
}
```

**Why I kept it as-is:** the file is already committed with real
Kaggle hashes from 2026-04-18; rewriting the shape would invalidate
`_manifest_matches()` for the already-extracted data, force a
re-download of the 1.3 GB archive, and orphan the `schema_version`
linkage to `src/fraud_engine/schemas/raw.py` which is depended on by
the lineage tests. The spec's array form was illustrative, not
contractual.

**Trade-off:** keyed lookup is O(1), which matters for
`_manifest_matches()` where we check five files; the array form would
force a linear scan per file. The `source` / `schema_version` extras
let `_write_manifest()` pin the upstream competition and the raw-schema
version that was valid at download time, so a future schema bump
forces a manifest regeneration.

## Files changed this prompt

| File | Change |
|---|---|
| [scripts/download_data.py](../../scripts/download_data.py) | Added `log_call` to import; added `--output-dir` click option; applied `@log_call` to entry point; renamed `download` → `main`; changed output_dir body conversion to `Path(output_dir) if ...` |
| [src/fraud_engine/py.typed](../../src/fraud_engine/py.typed) | New empty PEP 561 marker |

Nothing else was touched.

## Verification

All four commands ran after the edits landed. Transcripts verbatim.

### `uv run mypy scripts/download_data.py`

```
Success: no issues found in 1 source file
```

### `uv run ruff check scripts/download_data.py` + `ruff format --check`

```
All checks passed!
1 file already formatted
```

### `uv run python scripts/download_data.py --help`

```
Usage: download_data.py [OPTIONS]

  Acquire the IEEE-CIS raw CSVs and write a SHA256 manifest.

Options:
  --force                 Ignore manifest and re-download even if hashes
                          already match.
  --output-dir DIRECTORY  Override settings.raw_dir. Created if missing.
  --help                  Show this message and exit.
```

### `uv run python -c "from scripts.download_data import main; print(type(main).__name__)"`

```
import OK Command
```

### Idempotent skip end-to-end (`uv run python scripts/download_data.py`)

Real Kaggle-sourced raw CSVs already present; expected behaviour is
skip. Structured-log tail:

```json
{"kw_force": {...false}, "kw_output_dir": {...null}, "event": "main.start", ...}
{"competition": "ieee-fraud-detection", "raw_dir": "data/raw", "force": false,
 "run_id": "ccbb7b16fff74e52ae1cdbda2329a60f", "event": "download.start", ...}
Manifest intact at data/raw/MANIFEST.json — skipping download.
{"reason": "manifest matches on all files", "event": "download.skip",
 "run_id": "ccbb7b16fff74e52ae1cdbda2329a60f", ...}
{"duration_ms": 2224.428, "result": {...null}, "event": "main.done",
 "run_id": "ccbb7b16fff74e52ae1cdbda2329a60f", ...}
```

Every event carries the same `run_id` (`ccbb7b16…`). `main.start` and
`main.done` bracket the body; `download.skip` fires mid-stream because
all five SHA256 hashes match. Duration of the 2.2s skip is dominated
by re-hashing ~1.3 GB of CSV content (1 MiB chunks).

## Acceptance checklist (spec)

- [x] `scripts/download_data.py` exists with Google-style docstring including business rationale & trade-offs
- [x] Reads Kaggle credentials from Pydantic `Settings` (not `~/.kaggle/kaggle.json` directly)
- [x] Injects credentials into `os.environ` before `import kaggle`
- [x] Downloads `ieee-fraud-detection` via `KaggleApi.competition_download_files`
- [x] Extracts the 5 CSVs into `data/raw/` and removes the zip
- [x] Streams SHA256 in 1 MiB chunks (`_HASH_CHUNK_BYTES = 1 << 20`)
- [x] Writes `MANIFEST.json` to `data/raw/`
- [x] Idempotent skip when manifest + file hashes all match
- [x] `--force` flag bypasses the skip
- [x] `--output-dir` flag overrides `settings.raw_dir`
- [x] Structured `structlog` JSON logs with `run_id` on every event
- [x] `@log_call` decorator on the entry point
- [x] `uv run mypy scripts/download_data.py` → clean
- [x] `uv run ruff check scripts/download_data.py` → clean
- [x] `from scripts.download_data import main` succeeds
- [x] `--help` lists both flags

## Non-goals (deferred / out of scope)

- **Running a fresh download.** Existing data is intact; forcing a
  re-download costs 1.3 GB of bandwidth for no gain.
- **Rewriting the manifest shape to the spec's array form.** See
  deviation section above — the dict form is a superset and the
  schema is already committed with real hashes.
- **Wider pipeline re-verification.** Gate 0.1 already ran the full
  142-test suite clean; this prompt's changes are confined to one
  script plus one marker file and do not touch any tested module.
- **Git action.** CLAUDE.md §2.
