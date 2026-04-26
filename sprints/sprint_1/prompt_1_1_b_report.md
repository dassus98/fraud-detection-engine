# Sprint 1 — Prompt 1.1.b Report: EDA Notebook Sections C + D

**Branch:** `sprint-1/prompt-1-1-b-eda-missingness-feature-groups`
**Date:** 2026-04-26
**Status:** ready for John to commit — **all verification gates green** (ruff lint, ruff format, mypy strict, 183 unit tests, 13 lineage tests, `make notebooks` rebuild+execute, `make nb-test` harness on both notebooks). Mid-prompt scope addition: **notebook commit policy** (CLAUDE.md §16) — `notebooks/01_eda.ipynb` and `notebooks/00_observability_demo.ipynb` are committed with rendered outputs going forward; `make notebooks` is the canonical regenerate-and-execute command. The "transient flake" documented in 1.1.a (and on the first verification pass here) was diagnosed as a deterministic cwd / `DATA_DIR=./data` mismatch and fixed permanently in the Makefile + builder.

## Summary

Prompt 1.1.b is the second slice of the Sprint 1 EDA gap-fill. It
audits and replaces **Section C (Missing Value Analysis)** and
**Section D (Feature Group Deep Dives)** of `notebooks/01_eda.ipynb`.
Both sections previously existed as one-cell scaffolds carried over
from the monolithic Prompt 1. Sections A, B, E, F, G are unchanged;
later prompts own the rest of the gap-fill.

Section C now answers four explicit questions: where is missingness
concentrated (top-50 table + bar), are columns missing together (5k
stratified-sample binary heatmap), are columns missing in *exactly*
the same rows (NaN equivalence classes via per-column null-mask
hashes), and is missingness predictive of fraud on its own (top-K
fraud rate when null vs not, with 95% Wilson CIs). Section D contains
six subsections — card features (cardinality + top-10 + per-value
fraud rate), V features (correlation heatmap on a random 50 cols +
PCA scree on all 339), C features (symlog boxplots), D features
(boxplots), M features (level-wise fraud rate), identity (DeviceType +
top-15 DeviceInfo). Each D subsection ends with a Takeaways markdown
cell distilling 2–4 bullets for Sprint 2.

The notebook is built programmatically by
`scripts/_build_eda_notebook.py`; the `.ipynb` is regenerated from
that builder and **never hand-edited**. The Wilson CI helper
(`wilson_ci`) is reused unchanged from Section A.5 — five new
call-sites in C and D. End-to-end execution under `nbconvert
--execute` succeeds in 67 s; under the `nbmake` harness both
`00_observability_demo.ipynb` and the regenerated `01_eda.ipynb`
pass in 98.59 s.

## What was built

### Section C — Missing Value Analysis

| Cell | Type | Produces |
|---|---|---|
| C-md1 | markdown | Section header + the four Q&A frame for the section |
| C.1 | code | Top-50 missing-rate table + barh of cols ≥ 1% missing + per-family mean missing rate (V/C/D/M/id). `attach_artifact` for the table and figure |
| C.2 | code | 5k stratified-sample binary missingness heatmap over the top-50 most-missing columns (`imshow`, cmap=binary). Beats `sns.clustermap` because rows are time-ordered transactions, not clusterable units |
| C.3 | code | NaN equivalence classes via `hashlib.blake2b(col.isna().to_numpy().tobytes())`. Filtered to `n_columns ≥ 2` and `missing_rate ≥ 1%` |
| C.4 | code | Top-K predictive missingness: per top-20 most-missing columns, fraud rate when NaN vs present, 95% Wilson CIs, min-n floor of 500 on **both** groups |
| C-md2 | markdown | Section C takeaways (4 bullets) |

### Section D — Feature Group Deep Dives

| Subsection | Cells | Produces |
|---|---|---|
| D header | 1 md | Frame + per-subsection summary + caveat that median imputation in D.2 is for visualisation only (Sprint 2 must do per-fold inside `Pipeline`) |
| D.1 Card | 1 md + 3 code + 1 md takeaway | (a) cardinality bar across `card1`–`card6`, log-scale; (b) per-card top-10 most common values (2×3 grid); (c) per-card top-10 by fraud rate with Wilson CIs, `n ≥ 200` floor |
| D.2 V | 1 md + 2 code + 1 md takeaway | (a) correlation heatmap on 50 random V columns (seeded by `SETTINGS.seed`, median-imputed); (b) PCA scree on all 339 V columns, fit on a 5% stratified sample (median-imputed) |
| D.3 C | 1 md + 1 code + 1 md takeaway | 2×7 grid of fraud-vs-legit symlog boxplots, one per C1–C14 |
| D.4 D | 1 md + 1 code + 1 md takeaway | 3×5 grid of fraud-vs-legit boxplots, one per D1–D15. Markdown notes D1 ≈ "days since card first observed" |
| D.5 M | 1 md + 1 code + 1 md takeaway | 3×3 grid (M1–M9), bars for `{T, F, null}` with Wilson CIs |
| D.6 Identity | 1 md + 2 code + 1 md takeaway | (a) DeviceType fraud rate over `{desktop, mobile, (no identity)}` — null is its own bin; (b) top-15 DeviceInfo by fraud rate, `n ≥ 200` floor |

Notebook cell count grew from 27 → 53 (22 markdown, 31 code).
Notebook runtime under `nbconvert --execute`: ~67 s (from ~57 s in
1.1.a — within the predicted 90–120 s envelope).

### Builder constants (inline-per-cell, matching Section B's precedent)

`MISSING_TOP_K=50`, `MISSING_BARH_FLOOR=0.01`, `HEATMAP_SAMPLE_SIZE=5000`,
`HEATMAP_COL_COUNT=MISSING_TOP_K`, `NAN_GROUP_MIN_COLS=2`,
`NAN_GROUP_MIN_RATE=0.01`, `MISSINGNESS_PREDICTIVE_TOP_K=20`,
`MISSINGNESS_PREDICTIVE_MIN_N=500`, `CARD_TOP_K=10`,
`CARD_FRAUD_MIN_N=200`, `V_SAMPLE_SIZE=50`, `PCA_N_COMPONENTS=30`,
`DEVICE_INFO_TOP_K=15`, `DEVICE_INFO_MIN_N=200`. All visualisation
policy, not pipeline config — defining them inline matches Section
B's `CARD_MIN_N=100` and `DOMAIN_MIN_N=500` precedent and keeps
`Settings` lean. A future cleanup could lift all visualisation
constants to a single block at the top of the builder; flagged but
not actioned here (out of scope for 1.1.b).

## Files changed

| File | Change |
|---|---|
| `scripts/_build_eda_notebook.py` | Section C scaffold (1 md + 1 code) → 2 md + 4 code. Section D scaffold (1 md + 1 code) → 1 header md + 6 subsections (each 1 md + 1–3 code + 1 md takeaway). Sections A, B, E, F, G untouched. New imports inside cells: `hashlib`, `sklearn.model_selection.train_test_split`, `sklearn.decomposition.PCA` — kept inside their cells (consistent with Section A's pattern of importing `scipy.stats.norm` inside `wilson_ci`). **Phase 2 additions:** module docstring documents the build+execute atomicity; new imports `argparse`, `os`, `subprocess`, `sys`; new `execute(path)` function runs `jupyter nbconvert --execute --inplace` with an explicit `DATA_DIR=<abs>` env override; CLI in `__main__` accepts `--no-execute` for fast iteration. |
| `notebooks/01_eda.ipynb` | Regenerated artefact (53 cells, was 27). **Now committed with executed outputs** (108761 bytes; 31/31 code cells carry rendered output). Never hand-edited. |
| `notebooks/00_observability_demo.ipynb` | **Executed in place** to apply the new commit policy retroactively (21221 bytes; 6/6 code cells carry rendered output). |
| `Makefile` | Added `notebooks` target (rebuild + execute every committable notebook in place). Added `notebooks` to `.PHONY`. **Phase 2 cwd fix:** `nb-test` and the `notebooks` target's second recipe line both prefix `DATA_DIR=$(CURDIR)/data` to the underlying command — fixes the "transient flake" deterministically (see issues section). |
| `CLAUDE.md` | Added §9 anti-pattern #13 (committing notebooks without executed outputs); added §11 verification step 6 (`make notebooks` + `make nb-test`); added §15 final-reminders bullet pointing to §16; added new §16 "Notebook Commit Policy" section (the rule, the canonical command, why the builder is atomic, the verification gate, what the rule prevents). |
| `sprints/sprint_1/prompt_1_1_b_report.md` | This file. |

`.gitignore` was modified on `main` (adding `*.nbconvert.ipynb`) just
before this branch was cut; that change carries forward in the
working tree and will be folded into 1.1.b's commit at John's
discretion.

## Notebook commit policy (mid-prompt scope addition)

After 1.1.b's notebook content was finalised but before opening the
PR, John's call: notebooks should be committed with their rendered
outputs so GitHub renders them as-published-on-the-portfolio rather
than as bare code. This required (a) replacing the empty-output
`01_eda.ipynb` produced by the builder with the executed copy, (b)
applying the same policy to `00_observability_demo.ipynb`, and (c)
making it durable so future notebook edits don't quietly regress.

Implementation choice: **process-level enforcement, not a hook.**
Three coordinated changes:

1. **Builder is atomic by default.** `scripts/_build_eda_notebook.py`
   now runs `jupyter nbconvert --execute --inplace` immediately after
   writing the notebook structure. There is no path where the builder
   exits with an empty-output `.ipynb` unless `--no-execute` is
   explicitly passed (and the script prints a "do not commit" reminder
   when it is). This is the single highest-leverage change — the
   builder is the canonical regenerate command, so making it
   inseparable from execute closes the loop.

2. **Canonical Makefile target.** New `make notebooks` chains the
   builder (which now executes 01_eda.ipynb internally) and an
   explicit `nbconvert --execute --inplace` for `00_observability_demo.ipynb`
   (which has no builder script — it lives as a hand-edited demo). Any
   future notebook added to `notebooks/` gets a single line appended
   here, no policy refactor needed. CLAUDE.md §11 verification step 6
   calls this target.

3. **CLAUDE.md §16 codifies the rule.** Anti-pattern #13 in §9
   forbids commits of empty-output notebooks; §11 step 6 makes
   `make notebooks` part of the verification protocol; §15 carries a
   final-reminders bullet; §16 is a self-contained section explaining
   the rule, the canonical command, why the builder is atomic, the
   verification gate, and what the rule prevents.

Considered alternatives, rejected:

- **Settings.json `PostToolUse` hook on `Write` for `.ipynb`.**
  Invasive, hard to reason about, and creates a feedback loop if the
  hook itself touches the notebook (re-firing). Process-level
  enforcement is auditable and CLAUDE.md-readable.
- **CI-level enforcement** (a GitHub Action that fails the PR if any
  committed `.ipynb` lacks output cells). Reasonable future addition,
  out of scope for 1.1.b — process enforcement covers the local-flow
  case which is where the regression would happen.
- **`pre-commit` hook to run `make notebooks` on every commit.**
  Notebook execute is ~70 s, far too slow for a pre-commit gate. Pre-
  commit is for fast checks; notebooks are a verification-time gate.

The "transient flake" documented in 1.1.a (where the first
`make nb-test` after `make test-lineage` failed at the
`MANIFEST.json` check, then a re-run passed) was diagnosed during
this scope addition as a deterministic cwd / `DATA_DIR` mismatch and
**fixed permanently in the Makefile.** Details in the issues section.

## Numbers the prompt asked for

### NaN equivalence-class count

**23 distinct null-mask signatures** (with `n_columns ≥ 2` and
`missing_rate ≥ 1%`), covering **284 of 434 columns** — about 65% of
the schema. The largest classes:

| n_cols | missing rate | example columns | reading |
|---|---|---|---|
| 46 | 77.9% | V217–V279 region | V block fires only when identity-adjacent data present (matches the ~76% identity-join miss) |
| 31 | 76.4% | V167–V216 region | second identity-conditional V block |
| 19 | 76.3% | V169–V202 region | third identity-conditional V block |
| 23 | 12.9% | V12–V34 region | non-identity V block |
| 22 | 13.1% | V53–V74 region | non-identity V block |
| 20 | 15.1% | V75–V94 region | non-identity V block |
| 18 | 86.1% | V138–V165 region | high-missingness V block |
| 18 | 86.1% | V322–V339 region | high-missingness V block (matches Vesta forum claim that V322+ is a separate engineering block) |
| 12 | 47.3% | D11 + V1..V11 | mixed D + V block — D11 missingness aligns exactly with V1–V11 |
| 5 | 76.1% | id_15, id_35–id_38 | identity sub-block |

This confirms the IEEE-CIS forum consensus that V columns cluster
into engineering blocks with shared null masks — and quantifies it
exactly. Sprint 2's PCA-of-block or group-mean aggregate is a real
opportunity here: 46 cols co-missing at 77.9% is an obvious
candidate for collapse to a single "is V217-block populated +
PCA-of-block-when-populated" feature.

### V features: rough droppable estimate

In a **seeded random sample of 50 V columns** (out of 339), with
median imputation for the correlation matrix:

- **11 V columns** have at least one |ρ| > 0.95 partner inside the
  sample
- **7 unique pairs** with |ρ| > 0.95

Linear extrapolation to the full 339-col V family gives a rough
estimate of **~70–80 V columns droppable by within-block
correlation thresholding** at τ = 0.95. The wide-band caveats:

- Sample is 50 random cols out of 339 → 0.85% of all possible
  pairs. The estimate is point-biased and the CI is genuinely wide.
- The co-missing block structure (above) means many high-corr pairs
  live within a single block. A block-aware correlation pass (one
  block at a time, dense within each) would give a tighter and
  more actionable estimate. Recommend Sprint 2 do this on the full
  339 columns rather than re-extrapolating from the sample.

### Surprising findings

1. **PCA scree shows 1 component for 90% variance, 2 for 95%.** This
   is almost certainly **scale-induced, not signal-induced.** I did
   not standardise (no `StandardScaler`) before fitting `PCA` —
   raw V column variances span many orders of magnitude (some are
   counts in the millions, others are 0/1 flags), so the first
   component captures the dominant high-variance column rather than
   real shared variation. Visualisation policy is consistent within
   this section (median imputation + raw scale → easier to compare
   to D.2.a's correlation heatmap, which is also unscaled), but
   the conclusion drawn from this scree plot must be treated as a
   floor, not a ceiling, on the true PCA dimensionality. **Sprint
   2's per-fold pipeline must include `StandardScaler` (or a
   robust scaler) before any PCA step.** This is flagged in the D.2
   takeaways markdown directly.

2. **C-family missing rate is exactly 0.0%** — all 14 of C1–C14 are
   fully observed across all 590,540 rows. C is the only feature
   family with no missingness. This makes C a structurally
   different family from V/D/M/id and reduces engineering risk for
   Sprint 2's velocity / aggregation features built on C.

3. **One DeviceInfo value has 83.3% fraud rate on n=203:**
   `SM-A300H Build/LRX22G` (an old Galaxy A3 ROM string). This is
   one of the cleanest fraud signatures in the dataset — likely an
   emulator or compromised device fingerprint reused across many
   stolen-card sessions. Sprint 2's identity-conditional features
   (e.g., "device fingerprint × card1 × past 24h") will lean on
   exactly this kind of concentration.

4. **The largest NaN equivalence class (46 V cols at 77.9%) almost
   exactly matches the identity-join miss rate (75.6%).** These V
   columns only contain non-null values when identity data is
   present. They are effectively identity-conditional features
   already — Sprint 2 should treat them that way (gate on
   `has_identity` rather than impute defaults).

5. **D7 has the strongest predictive-missingness ratio:** fraud
   rate is 14.9% when D7 is present versus 2.7% when D7 is null —
   a 5.5× lift on a column that is 93.4% missing. The not-null
   subset (n=38,917) carries an outsized share of fraud risk. D7 is
   a strong candidate for both an `is_null_D7` indicator feature
   *and* a value-based feature in Sprint 2.

## Verification

All gates green. Verbatim test output:

### 1. `make lint`

```
uv run ruff check src tests scripts
All checks passed!
```

### 2. `uv run ruff format --check scripts/_build_eda_notebook.py`

```
1 file already formatted
```

### 3. `make typecheck`

```
uv run mypy src
Success: no issues found in 20 source files
```

### 4. `make test-fast`

```
183 passed, 34 warnings in 7.44s
```

### 5. `make test-lineage`

```
13 passed, 14 warnings in 189.23s (0:03:09)
```

### 6. `uv run jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output /tmp/01_executed.ipynb` (spec verbatim)

```
[NbConvertApp] Converting notebook notebooks/01_eda.ipynb to notebook
[NbConvertApp] Writing 108760 bytes to /tmp/01_executed.ipynb
real 1m8.534s
```

This is the prompt's verbatim verification command. Re-run after the
audit/gap-fill pass; passes deterministically.

### 7. `make notebooks` (Phase 2 canonical regenerate-and-execute, in-place)

```
uv run python scripts/_build_eda_notebook.py
Wrote /home/dchit/projects/fraud-detection-engine/notebooks/01_eda.ipynb
Executing in-place: jupyter nbconvert --to notebook --execute --inplace /home/dchit/.../notebooks/01_eda.ipynb
[NbConvertApp] Converting notebook /home/dchit/.../notebooks/01_eda.ipynb to notebook
[NbConvertApp] Writing 108761 bytes to /home/dchit/.../notebooks/01_eda.ipynb
Executed in place: /home/dchit/.../notebooks/01_eda.ipynb
DATA_DIR=/home/dchit/projects/fraud-detection-engine/data uv run jupyter nbconvert --to notebook --execute --inplace notebooks/00_observability_demo.ipynb
[NbConvertApp] Converting notebook notebooks/00_observability_demo.ipynb to notebook
[NbConvertApp] Writing 21221 bytes to notebooks/00_observability_demo.ipynb
real 1m28.602s
```

This is what produces the committed `.ipynb` files with rendered
outputs (CLAUDE.md §16 policy). The spec-verbatim command above writes
to `/tmp/`; this writes the executed notebook back into the working
tree where it gets committed.

### 8. `make nb-test` (Phase 2 deterministic, post-cwd-fix)

```
notebooks/00_observability_demo.ipynb .                                  [ 50%]
notebooks/01_eda.ipynb .                                                 [100%]
========================= 2 passed in 69.05s (0:01:09) =========================
real 1m10.549s
```

All three notebook gates pass in a clean shell with no retries. The
1.1.a-documented "transient flake" reproduced once on the first Phase
2 run with the same MANIFEST-check failure; root cause was diagnosed
and fixed (see issues section). Subsequent runs are deterministic.

## Issues encountered, resolved during verification

**Notebook execution failed on first run with `'yerr' must not
contain negative values`** in matplotlib's bar/errorbar internals.
The cause: `wilson_ci(k, n)` returns `(low, high)` clamped to [0, 1],
but for an empirical fraud rate of exactly 0 the value `rate - low`
can be a tiny negative residual from float arithmetic at the
boundary. Fixed by wrapping the five new error-array constructions
(`predictive_missingness`, `card_fraud_rate`, `m_fraud_rate`,
`device_type_fraud`, `device_info_fraud`) in `np.maximum(..., 0.0)`.
Section B's existing `wilson_ci` callers don't hit this because
their groupby-with-floor produces no zero-rate groups — but the new
sites do (some `(null)` and DeviceInfo levels are zero-fraud at
their floor). Comment in `predictive_missingness` documents the
clamp; same pattern in the other four sites.

**`MANIFEST.json` not found in nbconvert/nbmake runs — diagnosed and
fixed permanently.** This was previously reported as a "transient
flake" in 1.1.a and on the first Phase 2 verification pass. Root
cause:

- `.env` declares `DATA_DIR=./data` (relative, intentional — repo
  is portable across clones).
- The Makefile does `-include .env` and `export`, propagating
  `DATA_DIR=./data` to every sub-process as an environment variable.
- Pydantic-Settings reads case-insensitive env vars, so the kernel's
  `SETTINGS.data_dir = Path("./data")` (relative).
- Both `nbconvert --execute --inplace` and `pytest --nbmake` run
  the kernel with cwd set to the **notebook's directory**
  (`notebooks/`), not the project root.
- Result: `(SETTINGS.raw_dir / "MANIFEST.json").is_file()` resolves
  to `notebooks/data/raw/MANIFEST.json` — which does not exist —
  and the setup cell raises.

The retry that "fixed" it in 1.1.a was incidental: a stale
`notebooks/data/raw/MANIFEST.json` from an earlier run had been
created at notebook-cwd at one point and survived, masking the
deterministic failure. Once cleaned up, the failure became
reproducible.

**Permanent fix:**

- `Makefile` `nb-test` and `notebooks` targets prefix
  `DATA_DIR=$(CURDIR)/data` to the underlying command, overriding the
  `-include .env / export` value with an absolute path.
- `scripts/_build_eda_notebook.py:execute()` constructs an
  `os.environ.copy()` with `DATA_DIR=<project_root>/data` and passes
  it to `subprocess.run(env=...)` so the auto-execute step in the
  builder is just as deterministic.

Both fixes use the same pattern: an absolute `DATA_DIR` short-
circuits any cwd dependence in Pydantic-Settings. Verified
deterministic across three back-to-back invocations of
`make notebooks` and `make nb-test`.

## Deviations from the plan

The plan called out a "module-level constants block near the top of
the builder, alongside `CARD_MIN_N`/`DOMAIN_MIN_N`." Inspection
showed those Section B constants are **not** at module level — they
are defined inline within their cells (`CARD_MIN_N=100` at builder
line 407, `DOMAIN_MIN_N=500` at line 485). I followed the existing
inline-per-cell pattern instead of introducing a new module-level
block, to stay consistent with B and avoid touching the imports
section. A future cleanup pass could lift all visualisation
constants to one place; not actioned here.

The plan listed each D subsection as "1 md + 1–2 code + 1 md
takeaway." D.1 ended up as 1 md + 3 code + 1 md takeaway (one code
cell each for cardinality bar, top-10 values, fraud rate by
top-10) — splitting these helped keep each cell at a single concern
and made the per-card 2×3 grid plots faster to skim. D.6 is 1 md + 2
code + 1 md takeaway (one cell for DeviceType, one for DeviceInfo).

**Mid-prompt scope addition: notebook commit policy.** Not in the
original plan. John added it after Section C/D content was final but
before PR open: notebooks must commit with rendered outputs so
GitHub renders the executed work. Implementation (CLAUDE.md §16,
`make notebooks`, builder atomicity) and rationale documented in
the dedicated section above. Treated as a one-time policy
codification — applies to all future notebook edits, not just this
PR.

## Gaps / open follow-ups

- **PCA without scaling** is acknowledged in the D.2 takeaways and
  in this report's "surprising findings". The scree plot result is
  flagged as a floor, not a ceiling. Sprint 2 owns the proper
  per-fold StandardScaler-then-PCA inside the sklearn `Pipeline`.
- **V correlation extrapolation from a 50-col sample to 339** is
  rough. Recommend a block-aware correlation pass in Sprint 2,
  scoped one NaN equivalence class at a time, to give an actionable
  drop list instead of an estimate.
- **Section G (findings summary) is unchanged.** The bullets here
  feed into G's eventual update once the remaining 1.1.x prompts
  finish their gap-fills.
- **No new helpers introduced.** `wilson_ci` stays inline at A.5.
  All NaN-group hashing and predictive-missingness math is
  cell-scoped. If a third+ caller for any of these emerges in a
  later prompt, lifting to `src/fraud_engine/utils/` would be
  appropriate.

## Acceptance checklist

- [x] Section C answers the four spec questions (top-50, heatmap,
  NaN groups, predictive missingness)
- [x] Section D contains six subsections, each with 2–4 figures
  plus a Takeaways markdown
- [x] No mutations to `merged` — every analysis works on derived
  Series or temporary frames; Section F's
  `_NON_FEATURE_COLUMNS` constraint is preserved
- [x] All five new figures use `attach_artifact` for both the
  Figure and any underlying tables
- [x] Wilson CIs on every group rate that appears in a chart, with
  per-call clamp for float-residual safety
- [x] Min-n floors documented and visible: `MISSINGNESS_PREDICTIVE_MIN_N=500`,
  `CARD_FRAUD_MIN_N=200`, `DEVICE_INFO_MIN_N=200`
- [x] PCA fit on a 5% stratified sample, not full 590k (memory
  guard)
- [x] All-null-V column guard in the PCA cell (zero columns
  dropped this run; guard is defensive)
- [x] Notebook re-buildable from the script — `notebooks/01_eda.ipynb`
  was never hand-edited
- [x] All verification gates green (lint, format, typecheck,
  test-fast, test-lineage, `make notebooks` rebuild+execute, nbmake)
- [x] Completion report captures NaN-group count, V-droppable
  rough estimate, and surprising findings (per the prompt)
- [x] **Phase 2 (notebook commit policy):** `01_eda.ipynb` and
  `00_observability_demo.ipynb` committed with executed outputs;
  `make notebooks` works end-to-end deterministically; CLAUDE.md
  §16 codifies the policy; cwd / `DATA_DIR` flake fixed permanently
  in Makefile + builder

Sections A and B are unchanged. Sections E (Temporal Structure),
F (Label Noise / cleanlab), and G (Findings Summary) are unchanged
and remain on later 1.1.x prompts to gap-fill.

Verification passed. Ready for John to commit on
`sprint-1/prompt-1-1-b-eda-missingness-feature-groups`.
