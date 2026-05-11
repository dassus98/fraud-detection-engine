# Sprint 6 — Prompt 6.2.a: Model Card + Feature Documentation

## Summary

Sprint 6.1.a-e built the **monitoring + operations** layer. Sprint 6.2.a adds the **portfolio-facing model documentation** so a senior reviewer (Wealthsimple / Mercury / RBC / Nubank hiring committee per CLAUDE.md §1) can audit the model's intended use, fairness considerations, training data, performance across slices, and operating envelope without reading the codebase.

Two artefacts:

1. **[`docs/MODEL_CARD.md`](../../docs/MODEL_CARD.md)** — Google's [Model Cards for Model Reporting (Mitchell et al. 2018)](https://arxiv.org/abs/1810.03677) format. All 9 sections (Model Details, Intended Use, Factors, Metrics, Training Data, Evaluation Data, Quantitative Analyses, Ethical Considerations, Caveats and Recommendations) with concrete numbers + footnoted source citations for every claim. ~485 LOC.

2. **[`docs/FEATURE_DOCUMENTATION.md`](../../docs/FEATURE_DOCUMENTATION.md)** — grouped by tier → generator (per the user's confirmed pick). Top: pipeline overview + ASCII diagram + count summary. 6 tier sections (0-5) × 12 generator subsections × {count, sample features, business rationale verbatim from manifest, cost prose, failure-mode prose}. 2 appendices (full-listing pointer with `jq` recipes + failure-mode quick-reference table). ~325 LOC.

**Risk: Low → realised Low.** Pure documentation. Both sanity scripts pass (12/12 generators covered; 9/9 Mitchell et al. sections present). No source code changes, no test changes, no regression possibility.

## Files changed

| Path | Change | LOC |
|---|---|---|
| `docs/MODEL_CARD.md` | NEW — Mitchell et al. 9-section format with footnoted citations | +485 |
| `docs/FEATURE_DOCUMENTATION.md` | NEW — Tier 0-5 sections + per-generator subsections + 2 appendices | +325 |
| `sprints/sprint_6/prompt_6_2_a_report.md` | NEW — this report | +(this file) |

**No changes** to source code, configs, schemas, settings, compose files, tests, Makefile, Dockerfile, `CLAUDE.md`, or any prior monitoring artefact.

## What `docs/MODEL_CARD.md` covers

### All 9 Mitchell et al. sections + ~30 footnoted numbers

| Section | Content highlights |
|---|---|
| **Model Details** | Model name + owner + content_hash version pointer; type (binary classifier); training algorithm (LightGBM + isotonic calibration); Optuna 100-trial sweep; best iteration 72; val AUC 0.8281; decision rule + threshold (τ = 0.080); diversity models (Model B FraudNet shadow, Model C FraudGNN batch-only); license + citation + contact. |
| **Intended Use** | Primary use cases (real-time per-txn screening, cost-optimal blocking); primary users (fraud-ops, ML platform, compliance); 5 explicit out-of-scope cases (credit risk, ATO, AML, identity fraud, raw-uncalibrated thresholding). |
| **Factors** | 5 relevant factors (amount_bucket, ProductCD, DeviceType, identity_coverage, month) + 3 NOT-in-training-data factors (race/gender/age, geography, income) flagged as fairness limitations. |
| **Metrics** | Overall metric table (AUC 0.8281 val / 0.8070 test; Brier / ECE / log-loss with pre+post-calibration values; latency P95 70.98ms no-shadow / 81.4ms shadow-failing); cost-optimal threshold table (τ=0.080, analytical limit τ*≈0.0729, sensitivity spread 0.06, annual savings $28.96M); stratified-metrics summary with all 5 Sprint 4.2 test gates. |
| **Training Data** | IEEE-CIS facts (590,540 txns, 3.5% fraud, 24% identity coverage); citation (Kaggle / Vesta 2019); temporal split table (train ~414K / val ~83K / test ~92K with day boundaries 121 / 151). |
| **Evaluation Data** | Val + test slice details; the Sprint 6.1.c PerformanceMonitor production-evaluation surface. |
| **Quantitative Analyses** | Overall headline; per-slice summary; cross-model comparison table (Model A / B / C with AUC / Brier / ECE / latency); calibration improvement (Brier 0.0769 → 0.0254 = 67% reduction); intersectional analyses noted as scoped but not run. |
| **Ethical Considerations** | False-positive customer impact (~$1.1M/month friction cost at 3.5% block rate); demographic fairness gap (IEEE-CIS lacks race/gender/age) + 3 deployer responsibilities; opaque V-features; geographic + payment-instrument fairness limitations. |
| **Caveats and Recommendations** | 8 numbered caveats (24% identity, 2019 vintage, cost-sensitivity, calibration is load-bearing, temporal-split discipline, transaction-vs-customer layer, fraud_neighbor_rate OOF discipline, 685-feature maintenance commitment). |

### Citation discipline

Every number in the metrics + dataset facts sections links to its source artefact:
- Sprint report numbers → relative paths like `[prompt_3_3_d_report.md:43](../sprints/sprint_3/prompt_3_3_d_report.md)`.
- Config values → `[configs/economic_defaults.yaml](../configs/economic_defaults.yaml)`.
- ADRs → `[ADR-0003](ADR/0003-economic-threshold.md)`.

This is the portfolio-reader's "can I trust this?" gate — every claim is traceable.

## What `docs/FEATURE_DOCUMENTATION.md` covers

### Pipeline overview + count summary

ASCII diagram of the 5-tier pipeline + a summary table showing each generator's count and percentage of the 685-feature total.

**Key facts surfaced up-front:**
- 685 engineered features across 12 generators in 6 tiers (0-5).
- 89% of features (Tier 0 NaN reduction + Tier 1 MissingIndicators) are predictive-missingness boilerplate; only 11% carry substantive behavioural / velocity / graph signal. This sets reader expectations correctly.

### Per-tier structure (6 sections)

Each tier section has the same shape:
- **Purpose** (one paragraph).
- **Latency contribution** (from Sprint 5.1.f's P95 70.98ms breakdown).
- **Failure modes** (common ways the tier degrades).

Per-generator subsections under each tier:
- Count + sample feature names (first 5-10).
- Business rationale (verbatim from manifest's `rationale` field — preserves the existing Sprint 2/3 prose).
- Cost (compute / memory / latency).
- Failure modes (per-generator specifics drawn from the class docstring).

### Appendices

- **Appendix A** — pointer to `models/pipelines/feature_manifest.json` (the machine-readable source of truth) + 3 ready-to-run `jq` recipes for enumerating, counting, and listing by generator.
- **Appendix B** — failure-modes-by-tier quick-reference table for operators.

### Cross-references

Wikilinks to `DATA_DICTIONARY.md` (raw IEEE-CIS columns), `MODEL_CARD.md` (downstream metrics), `RUNBOOK.md` (operator procedures), the source generator files in `src/fraud_engine/features/`, and the Sprint 5.1.f latency report.

## Design decisions (7)

### Decision 1 — FEATURE_DOCUMENTATION.md grouped by tier → generator

User confirmed via `AskUserQuestion`. The manifest's rationale field is per-generator (every feature from `AmountTransformer` shares one paragraph); grouping by generator preserves that natural shape without duplicating prose 330 times for `MissingIndicatorGenerator`. The full 685-row listing lives in the manifest JSON + the appendix-A `jq` recipes.

### Decision 2 — MODEL_CARD.md follows Mitchell et al. 2018's exact 9-section names

Reviewers searching for "Ethical Considerations" or "Caveats and Recommendations" find them via the canonical name. The sanity script (test 2) enforces section presence so future PRs can't accidentally rename sections.

### Decision 3 — Pull metrics from existing reports; no fresh evaluation

User confirmed via `AskUserQuestion`. All ~30 numbers in MODEL_CARD are sourced from existing artefacts with `Last refreshed: 2026-05-11` documented at the bottom. The retraining procedure in `RUNBOOK.md` Step 8 (validation) names updating MODEL_CARD.md as part of the deploy procedure.

### Decision 4 — Cross-reference DATA_DICTIONARY.md, not duplicate

`docs/DATA_DICTIONARY.md` already covers raw IEEE-CIS columns. FEATURE_DOCUMENTATION adds the engineered-feature angle (which generators produce what, why, what they cost). Overlap is avoided via explicit wikilinks back to the data dictionary.

### Decision 5 — Verification: 2 sanity scripts + manual review

Spec says "Manual review" — to make manual review faster + catch the most common future regression (a PR adds a generator class but forgets to update FEATURE_DOCUMENTATION), the report includes 2 Python sanity checks:

1. Every generator in `feature_manifest.json` appears in FEATURE_DOCUMENTATION.md.
2. All 9 Mitchell et al. section names appear in MODEL_CARD.md.

Both run in the verification block below.

### Decision 6 — Mirror `docs/RUNBOOK.md` + `docs/OBSERVABILITY.md` style

Operator-focused tone, numbered sections, code blocks for every command/recipe, real file paths, structured tables. Same heading depth (h2 for major sections, h3 for subsections) so the docs feel uniform.

### Decision 7 — Citation discipline in MODEL_CARD

Every quantitative claim → footnoted with a relative-path link to its source. This is the "can I trust this?" gate for portfolio reviewers. Source paths use `../sprints/sprint_X/prompt_X_X_X_report.md` style so they resolve on both GitHub and local Markdown renderers.

## Verification

### Sanity check 1 — every generator covered in FEATURE_DOCUMENTATION.md

```text
$ uv run python /tmp/sanity_check.py
OK 1 — all 12 generators covered in FEATURE_DOCUMENTATION.md.
OK 2 — all 9 Mitchell et al. sections present in MODEL_CARD.md.
```

The check script (one-shot, not committed):

```python
import json, re
ROOT = "/home/dchit/projects/fraud-detection-engine"

# Check 1: every generator in feature_manifest appears in FEATURE_DOCUMENTATION.md
manifest_generators = sorted({
    f["generator"]
    for f in json.load(open(f"{ROOT}/models/pipelines/feature_manifest.json"))["features"]
})
doc_text = open(f"{ROOT}/docs/FEATURE_DOCUMENTATION.md").read()
missing = [g for g in manifest_generators if g not in doc_text]
assert not missing, f"FEATURE_DOCUMENTATION missing generators: {missing}"
print(f"OK 1 — all {len(manifest_generators)} generators covered.")

# Check 2: MODEL_CARD has all 9 Mitchell et al. sections
mc = open(f"{ROOT}/docs/MODEL_CARD.md").read()
required = ["Model Details", "Intended Use", "Factors", "Metrics",
            "Training Data", "Evaluation Data", "Quantitative Analyses",
            "Ethical Considerations", "Caveats"]
missing = [s for s in required if not re.search(rf"##\s+{re.escape(s)}", mc)]
assert not missing, f"MODEL_CARD missing sections: {missing}"
print("OK 2 — all 9 Mitchell et al. sections present.")
```

### Pre-commit on the 2 new docs

```text
$ uv run pre-commit run --files docs/MODEL_CARD.md docs/FEATURE_DOCUMENTATION.md
trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
check yaml...........................................(no files to check)Skipped
check toml...........................................(no files to check)Skipped
check for added large files..............................................Passed
check for merge conflicts................................................Passed
mixed line ending........................................................Passed
ruff.................................................(no files to check)Skipped
ruff-format..........................................(no files to check)Skipped
Detect secrets...........................................................Passed
mypy (strict, src only)..............................(no files to check)Skipped
pytest (unit, fast)..................................(no files to check)Skipped
```

### No regression — docs-only PR

This PR touches no source / config / test files. No unit / integration suite needs to run; the prior 819-unit / 16-integration baselines remain unchanged.

## Sample MODEL_CARD section excerpt

```markdown
## Caveats and Recommendations

### 1. 24% identity coverage — the model must degrade gracefully

76% of training transactions have no identity features (`id_*` columns all
null). The model is designed to work without them — Tier 3 `ColdStartHandler`
tags this case explicitly, and the Tier 1/2/4/5 features carry enough signal
that the no-identity slice still achieves AUC > 0.75 per Sprint 4.2.
**Production deployers should monitor the no-identity-slice performance
separately** since identity-availability in their environment may differ
from IEEE-CIS's 24%.

### 2. 2019 vintage — recalibration is mandatory

[... 6 more numbered caveats ...]
```

## Sample FEATURE_DOCUMENTATION section excerpt

```markdown
## Tier 4 — Temporal decay

- **Purpose:** exponentially-decayed velocity that smooths the fixed-window
  cliff of Tier 2. Where `card1_velocity_1h` drops to 0 the moment activity
  ages past 60 minutes, `card1_v_ewm_lambda_0.05` decays gracefully...
- **Latency contribution:** ≈ 5–10 ms at inference (Redis MGET across 24 keys)
- **Failure modes:**
  - Fraud-weighted variant requires OOF discipline. Training uses a
    two-pass `read → push` pattern: each row reads the prior EWM state
    (no leakage), then pushes its own label into the state for the next
    row. A bug that swaps the order leaks the current label.
  - Redis state lag: the EWM state at inference time is the most-recent
    state Sprint 5.1.b wrote. ...

### Generator: `ExponentialDecayVelocity` (src/fraud_engine/features/tier4_decay.py:365)
- **Count:** 24 features
- **Sample features:** `card1_v_ewm_lambda_0.05`, `card1_fraud_v_ewm_lambda_0.05`, ...
- **Business rationale:**
  > Per-(entity, λ) exponentially-decayed velocity (EWM). Smooths the
  > window-boundary cliff that fixed-window velocity has by construction —
  > recent activity decays gracefully rather than dropping off at an
  > arbitrary boundary. ...
- **Cost:** ≈ 5–10 ms at inference (Redis MGET + per-row EWM update is O(1)).
- **Failure modes:**
  - OOF-discipline regression — a future PR that refactors the two-pass ...
```

## Deviations from plan

None.

## Cross-references

- [`docs/MODEL_CARD.md`](../../docs/MODEL_CARD.md) — the produced model card.
- [`docs/FEATURE_DOCUMENTATION.md`](../../docs/FEATURE_DOCUMENTATION.md) — the produced feature documentation.
- [`models/pipelines/feature_manifest.json`](../../models/pipelines/feature_manifest.json) — 685-feature machine-readable source of truth.
- [`docs/DATA_DICTIONARY.md`](../../docs/DATA_DICTIONARY.md) — raw IEEE-CIS column reference (cross-referenced from FEATURE_DOCUMENTATION).
- [`docs/RUNBOOK.md`](../../docs/RUNBOOK.md) — Sprint 6.1.e operator runbook (cross-referenced from MODEL_CARD §Caveats).
- [`docs/OBSERVABILITY.md`](../../docs/OBSERVABILITY.md) — style template for both new docs.
- [`docs/ADR/0001-tech-stack.md`](../../docs/ADR/0001-tech-stack.md) + [`docs/ADR/0003-economic-threshold.md`](../../docs/ADR/0003-economic-threshold.md) — quoted in MODEL_CARD.
- `sprints/sprint_1/` ... `sprints/sprint_5/` — source of every quantitative claim in MODEL_CARD.

## Out of scope (Sprint 6.x+)

- **Diagrams** — both docs could carry mermaid pipeline diagrams; defer until the static-rendering toolchain is in place.
- **Auto-regenerate FEATURE_DOCUMENTATION from manifest** — a future `scripts/gen_feature_docs.py` could template the markdown. Defer; the manifest doesn't churn often + manual review is the spec.
- **CLAUDE.md §14 update** — adding `docs/FEATURE_DOCUMENTATION.md` to the doc index. Defer to a Sprint 6.2.x audit-and-gap-fill PR.
- **Pre-commit hook for the sanity checks** — the one-shot scripts above would graduate to a pre-commit hook in a future cleanup PR.
- **Fresh StratifiedEvaluator run on the latest model** — user confirmed pull-from-existing-reports is preferred. Future Sprint 6.x audit can refresh.
- **Per-feature failure-mode listings** — at 685 features that's untenable; failure modes are documented per generator (since features from the same generator share failure modes).
- **Hugging Face Hub upload of the model card** — automation that publishes the model card alongside an artefact mirror. Out of scope for portfolio project.
- **Per-deployment fairness supplement template** — MODEL_CARD's Ethical Considerations §Demographic fairness names the deployer's responsibility but doesn't provide a template. A future Sprint 6.x can ship one.
