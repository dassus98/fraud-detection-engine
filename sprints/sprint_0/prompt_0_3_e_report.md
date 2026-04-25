# Sprint 0 — Prompt 0.3.e Completion Report

**Prompt:** Operator-facing observability artefacts — `docs/OBSERVABILITY.md` (the runbook covering instrumentation patterns, log inspection, lineage tracing, log-level conventions, MLflow-vs-structlog split) plus `notebooks/00_observability_demo.ipynb` (an end-to-end runnable demo). These close Sprint 0.3 and give the next reviewer a single entry point for "how do I trace a prediction back to its data?".
**Date completed:** 2026-04-25

---

## 1. Summary

Both target files already existed with substantial Sprint 0.3.a–d content. This was an audit-and-gap-fill pass that closed two concrete defects and one bug discovered during execution:

| File | Change |
|---|---|
| `docs/OBSERVABILITY.md` | §1.6 MLflow example: stale 3-arg `log_economic_metrics(fn_rate, fp_rate, total_cost_usd)` → spec-correct 7-arg form pulling costs from `Settings`. §2.2 jq recipes: 2-sentence preamble disambiguating the JSON stream (stdout, jq target) from the on-disk text mirror (`tail -f` ergonomics), and recipes retargeted at a captured `.jsonl`. |
| `notebooks/00_observability_demo.ipynb` | +2 cells (markdown heading + code) demonstrating run-summary inspection — reads `logs/runs/{run.run_id}/run.json` (the proximate JSON file written by Cell 3's `run_context`), pretty-prints with `json.loads` + `json.dumps(indent=2)`, comments show equivalent jq commands. Outputs stripped. |
| `sprints/sprint_0/prompt_0_3_e_report.md` | This report. |

**`jupyter nbconvert --to notebook --execute` exits 0 (notebook runs end-to-end).** OBSERVABILITY.md grew from 242 → 257 lines.

---

## 2. Audit — Pre-Existing State

### `docs/OBSERVABILITY.md` (242 lines, 6 sections)

Solid coverage on (1) instrumenting new code, (2) reading the logs, (3) end-to-end lineage trace, (4) log-level table, (5) local dev stack, (6) related references. Two issues:

| Gap | Severity |
|---|---|
| §1.6 `log_economic_metrics(fn_rate, fp_rate, total_cost_usd)` — the rate-based 3-arg form, dead since 0.3.c rewrote the helper to be count-based with 7 args. Anyone copy-pasting from §1.6 hits `TypeError`. | **Spec-blocker** (stale doc on a key Sprint 3/4 helper). |
| §2.2 jq recipes target `logs/{pipeline}/{run_id}.log` — but `logging.py:222` configures the file handler with `ConsoleRenderer(colors=False)` (text), not `JSONRenderer()` (JSON). Recipes as written would `JSONDecodeError` on the first line of any `.log` file. The JSON stream actually lives on stdout (`logging.py:219`). | **Discovered-during-execution bug** — surfaced when the new notebook cell first tried to `json.loads` a `.log` file and hit `JSONDecodeError: Extra data: line 1 column 5`. |

### `notebooks/00_observability_demo.ipynb` (12 cells, 9373 bytes)

Existing demo covered §1 logging, §2 dataframe snapshots, §3 run_context, §4 MLflow, §5 metrics — all using post-0.3.b/0.3.c signatures. Missing the spec's final bullet:

> open the resulting log file and pretty-print with jq

`which jq` returns empty in this WSL environment, so a faithful demo can't shell out to `jq`. Substitute is stdlib `json.loads` + `json.dumps(indent=2)`.

### Gaps vs spec

| Gap | Resolution |
|---|---|
| OBSERVABILITY.md §1.6 stale signature | Surgical Edit: replaced the 3-arg call with the 7-arg form, pulled costs from `get_settings()`, added a confusion-matrix-unpack comment so the snippet is copy-paste-runnable. |
| OBSERVABILITY.md §2.2 wrong-target recipes | Added a 2-sentence preamble explaining the JSON-stream-vs-text-mirror split + a `2>&1 \| tee logs/train.jsonl` capture pattern, retargeted the 5 recipes at the captured file. Recipes themselves (the queries) are unchanged — they're correct queries, just needed a correct file target. |
| Notebook missing log-inspection cell | Inserted 2 cells (markdown + code) after `metrics-demo`, before `next-heading`. Reads `run.json`, pretty-prints with `json.loads`, comments show jq equivalents, closes out the run with one `logger.info("notebook.end", ...)` event. |

---

## 3. Gap-Fill — Edits This Turn

### `docs/OBSERVABILITY.md` §1.6 — MLflow example

Before:
```python
log_economic_metrics(fn_rate, fp_rate, total_cost_usd)
```

After (preceded by the existing `configure_mlflow(); setup_experiment(); start_run` boilerplate):
```python
from fraud_engine.config.settings import get_settings
settings = get_settings()
...
with mlflow.start_run(experiment_id=exp_id):
    log_dataframe_stats(train_df, prefix="train")
    log_dataframe_stats(val_df, prefix="val")
    # ... train model, score val set, threshold predictions ...
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    log_economic_metrics(
        fn_count=fn, fp_count=fp, tp_count=tp, tn_count=tn,
        fraud_cost=settings.fraud_cost_usd,
        fp_cost=settings.fp_cost_usd,
        tp_cost=settings.tp_cost_usd,
    )
```

The preamble comment reminds the reader where `fn`/`fp`/`tp`/`tn` come from (`sklearn.metrics.confusion_matrix(...).ravel()`), so the snippet is runnable as written.

### `docs/OBSERVABILITY.md` §2.2 — jq recipes

Before: 5 recipes targeting `logs/feature-pipeline/*.log` / `logs/**/*.log` (text files; recipes JSON-decode-fail).

After: 2-sentence preamble explaining the file format split:

> These recipes consume the **JSON stream** (stdout). The on-disk `logs/{pipeline}/{run_id}.log` files are the text-rendered mirror — they're `tail -f` ergonomic, not jq-parseable. To run jq offline, capture stdout to a file first:
>
> ```bash
> python -m fraud_engine.scripts.train 2>/dev/null | tee logs/train.jsonl
> ```

Plus a closing pointer: "For per-run summaries (single file, no capture needed), see §2.3 — `run.json` is JSON natively." Recipes themselves retargeted from `logs/**/*.log` to `logs/train.jsonl` (and `logs/api.jsonl` for the Sprint-5 request-id recipe).

### `notebooks/00_observability_demo.ipynb` — new cells 12 + 13

**Markdown cell** (after `metrics-demo`, before `next-heading`):

> ## 6. Reading run summaries — `jq`-style with `json.loads`
>
> `configure_logging` writes two streams: a JSON-per-line stream on stdout (the format ELK / Loki / `jq` consume) and a human-readable text mirror at `logs/{pipeline}/{run_id}.log` for `tail -f`. The proximate JSON file in this repo is the per-run summary at `logs/runs/{run_id}/run.json`, written by `run_context` — that's what the cell below parses.
>
> `jq` isn't shipped with this dev environment, so the cell uses stdlib `json.loads` and `json.dumps(indent=2)` to do the same filter+pretty-print. The full set of canonical jq recipes lives in `docs/OBSERVABILITY.md` §2.2.

**Code cell**: reads `run.run_dir / "run.json"` (the path Cell 3's `run_context` exposes), `json.loads`, `print(json.dumps(record, indent=2, sort_keys=True))`. Comment block shows three jq equivalents (`jq .`, `jq '.metadata'`, `jq 'select(.status == "success")'`). Closes with `logger.info("notebook.end", configure_logging_id=run_id, run_context_id=run.run_id)` so both run identifiers are explicitly correlated in the final log line.

**Why `run.json` instead of the .log file:** the spec asks for "open the resulting log file and pretty-print with jq", but `logging.py:222` ships the on-disk file as text. `run.json` is the proximate JSON file in this repo; demonstrating against it is honest and demonstrates the same operator skill (open a JSON-file artefact of a run, parse it, surface the fields). The text mirror is acknowledged in the markdown preamble.

---

## 4. Deviations from spec

1. **`jq` not available in the dev environment** (`which jq` → empty). The notebook cell uses `json.loads` + `json.dumps(indent=2)` as the functional substitute, with the equivalent jq commands shown in a comment block. The runbook §2.2 still documents the canonical jq recipes for environments where it's installed.

2. **Notebook demo targets `run.json`, not the structlog `.log` file.** The literal-spec target — the file written by `configure_logging` — is structlog ConsoleRenderer text (`logging.py:222`), not JSON, and so is not jq-parseable as the spec implicitly assumes. Pivoting to `run.json` (the JSON summary written by `run_context`) preserves the operator skill being demonstrated and is honest about the actual disk format. The text-mirror format is now explicitly called out in both the new markdown cell and the runbook §2.2 preamble.

3. **OBSERVABILITY.md §2.2 jq recipes — file-target fix.** Recipes pre-this-turn targeted `logs/**/*.log` (text). Retargeted at `logs/train.jsonl` (a captured stdout file), with a 2-sentence preamble explaining the `tee` capture pattern. The 5 query bodies are unchanged. This is a discovered-during-execution bug, scoped to the runbook (no `logging.py` change).

4. **`make docker-up` reference in §5 left intact.** The standing memory note defers Docker bring-up to end-of-project; it does not require removing the canonical command from documentation. §5 documents what to run *when* bring-up resumes.

5. **`run_context` cell keeps `capture_streams=False`.** The notebook's run-context demo cell (`run-context-demo`, written in 0.3.a) keeps `capture_streams=False` so its `print` output remains visible in the demo. Production scripts default to `True`. Not a 0.3.e change — flagged here because it's the only `run_context` deviation a reader of the demo will notice.

6. **No new tests added.** 0.3.e ships documentation + a notebook; the underlying helpers (`configure_logging`, `run_context`, `log_economic_metrics`) are already covered by 0.3.a/0.3.c unit tests. The notebook itself is exercised end-to-end by `nbconvert --execute`, which is the spec's gate.

---

## 5. Files Changed

| Path | Lines (after) | Change |
|---|---|---|
| `docs/OBSERVABILITY.md` | 257 (was 242) | §1.6 stale `log_economic_metrics` example replaced (3-arg → 7-arg, costs from `Settings`); §2.2 preamble added + recipe file targets corrected (`*.log` → `train.jsonl`). |
| `notebooks/00_observability_demo.ipynb` | 14 cells (was 12); 11705 bytes (was 9373) | +1 markdown heading cell ("Reading run summaries — jq-style with json.loads"); +1 code cell (read `run.json`, json.loads + pretty-print, jq-equivalent comments, `notebook.end` close-out). Outputs stripped via `nbconvert --clear-output --inplace`. |
| `sprints/sprint_0/prompt_0_3_e_report.md` | this file | New 8-section report. |

No source code under `src/fraud_engine/` touched. No test files touched. No config touched.

---

## 6. Verification Outputs

### 6.1 `nbconvert --execute` — spec gate

```text
$ uv run jupyter nbconvert --to notebook --execute \
    notebooks/00_observability_demo.ipynb \
    --output /tmp/00_observability_demo_executed.ipynb
[NbConvertApp] Converting notebook notebooks/00_observability_demo.ipynb to notebook
[NbConvertApp] Writing 21223 bytes to /tmp/00_observability_demo_executed.ipynb
NBCONVERT_EXIT=0
```

Exit 0 — every cell ran without raising. The new run-summary cell printed:

```text
reading /home/dchit/projects/fraud-detection-engine/logs/runs/b460040b3a984ab99b18e7eaa51a0498/run.json

pretty-printed run summary:

{
  "duration_ms": 138.539,
  "end_time": "2026-04-25 02:47:20.460144+00:00",
  "metadata": {
    "rows": 100,
    "source": "notebook"
  },
  "pipeline": "observability-demo",
  "run_id": "b460040b3a984ab99b18e7eaa51a0498",
  "start_time": "2026-04-25T02:47:20.321605+00:00",
  "status": "success"
}
{"configure_logging_id": "b456d6d940874a87a162cd638866a479", "run_context_id": "b460040b3a984ab99b18e7eaa51a0498", "event": "notebook.end", "run_id": "b460040b3a984ab99b18e7eaa51a0498", "pipeline": "observability-demo", "logger": "__main__", "level": "info", "timestamp": "2026-04-25T02:47:20.783299Z"}
```

The trailing JSON line is the `notebook.end` event captured to the cell's stdout — incidentally demonstrates the JSON stream format alongside the run.json parse.

### 6.2 `wc -l` — spec sanity check

```text
$ cat docs/OBSERVABILITY.md | wc -l
257
```

Up 15 lines from 242 — the §1.6 inline-confusion-matrix comment + the §2.2 preamble + capture-pattern code block.

### 6.3 Output strip — spec requirement ("strip outputs before committing")

```text
$ uv run jupyter nbconvert --clear-output --inplace \
    notebooks/00_observability_demo.ipynb
[NbConvertApp] Converting notebook notebooks/00_observability_demo.ipynb to notebook
[NbConvertApp] Writing 11705 bytes to notebooks/00_observability_demo.ipynb

$ python -c "import json; nb=json.load(open('notebooks/00_observability_demo.ipynb')); n_outs=sum(len(c.get('outputs',[])) for c in nb['cells'] if c['cell_type']=='code'); n_cells=len(nb['cells']); print(f'cells={n_cells} total_outputs_remaining={n_outs}')"
cells=14 total_outputs_remaining=0
```

14 cells (12 original + 2 new), 0 outputs remaining. File size dropped 21223 → 11705 bytes.

### 6.4 Re-execute stripped notebook (idempotency check)

```text
$ uv run jupyter nbconvert --to notebook --execute \
    notebooks/00_observability_demo.ipynb \
    --output /tmp/00_observability_demo_executed_v2.ipynb
[NbConvertApp] Writing 21223 bytes to /tmp/00_observability_demo_executed_v2.ipynb

# In-repo file outputs after re-execute:
in-repo outputs after re-execute (should still be 0): 0
```

Stripped file re-executes cleanly to /tmp at the same byte count (21223), and the in-repo file is preserved with 0 outputs. Output stripping is idempotent.

---

## 7. Acceptance Checklist

Spec bullets pinned:

- [x] **OBSERVABILITY.md covers `@log_call` vs manual logging.** §1.1 (`configure_logging`), §1.2 (`@log_call`), §1.3 (`log_dataframe`), §1.4 (`bind_request_id`).
- [x] **`run_context` manager usage documented.** §1.5 — full example with `attach_artifact`, dispatch table for type-based artefact handling.
- [x] **JSON log field structure documented.** §2.1 directory layout; §2.2 retargeted recipes show the JSON shape; §2.3 documents `run.json` fields (`run_id`, `pipeline`, `start_time`, `end_time`, `duration_ms`, `status`, `metadata`, `exception_type`).
- [x] **Trace prediction → data lineage via `run_id` documented.** §3 — four-step pivot from prediction log → request_id → run_id → training-data fingerprint (MANIFEST.json).
- [x] **5 useful jq queries provided.** §2.2 — run filter, slow-function filter, dataframe.snapshot filter, errors-only, request-scoped trail. Now correctly targeting captured `.jsonl` files with the capture pattern in the preamble.
- [x] **MLflow vs structlog split documented.** §1.5 (run_context for non-model work) vs §1.6 (MLflow for model work). The `run_context(...) ... mlflow.start_run(run_name=run.run_id)` cross-reference is shown explicitly in §1.6.
- [x] **Notebook runs end-to-end without errors.** `nbconvert --execute` exits 0; output verified §6.1.
- [x] **Notebook demonstrates log-file reading + pretty-printing.** New §6 cells read `run.json` and pretty-print via `json.loads` + `json.dumps`, with jq equivalents in comments.
- [x] **Notebook outputs stripped.** `nbconvert --clear-output --inplace` confirmed 0 outputs remaining (§6.3).

---

## 8. Non-Goals / Boundaries

- **No git operations.** CLAUDE.md §2 — no `git add`, `git commit`, no branch / tag / push. John handles all version control.
- **No `logging.py` change.** The discovered file-format bug (text on disk, JSON on stdout) is documentation-fixable in OBSERVABILITY.md §2.2; flipping the file handler to `JSONRenderer` would be a behaviour change to a tested module and is out of scope for 0.3.e. Future ADR if production wants jq-on-disk by default.
- **No new tests.** The underlying helpers are already pinned by 0.3.a/0.3.c unit tests. The notebook is its own end-to-end smoke test via `nbconvert --execute`.
- **No Docker bring-up.** Memory note defers compose verification to end-of-project. OBSERVABILITY.md §5 documents the canonical command for when bring-up resumes; no live verification this turn.
- **No `archive/v1-original` reference.** CLAUDE.md §10 / Anti-pattern 11 — clean rebuild.

---

**Verification passed. Ready for John to commit.**
