# Sprint 0 — Prompt 0.1.h Completion Report

**Task:** Docs first-draft — `docs/CONVENTIONS.md`,
`docs/ADR/0001-tech-stack.md`, and a rewritten `README.md`.

**Date:** 2026-04-21
**Branch state:** uncommitted; ready for John
**Pattern:** audit-and-gap-fill (same as 0.1.a through 0.1.g)

---

## 1. Scope reconciliation

All three docs pre-existed from the Sprint 0 bootstrap commit. Each
audited against the 0.1.h spec:

| Artefact | Audit finding |
|----------|----------------|
| [`docs/CONVENTIONS.md`](../../docs/CONVENTIONS.md) | Existed but was **not** the verbatim mirror of CLAUDE.md §5/6/7/9 the spec requires — it carried a condensed, earlier version of the same material. Missing the literal "temporal integrity" wording the spec greps for. **Rewrite.** |
| [`docs/ADR/0001-tech-stack.md`](../../docs/ADR/0001-tech-stack.md) | Spec-complete on Python/uv/LightGBM/SHAP/Redis/FastAPI/Pydantic/pandera/structlog/MLflow. Missing explicit rows for **PyTorch (experimental)**, **PostgreSQL (batch store)**, and **Grafana (dashboards)** called out in the spec's decision list. **Extend.** |
| [`README.md`](../../README.md) | Existed as a "Sprint 0 placeholder" stub — correct info, wrong shape. Missing the spec's title (`# Fraud Detection Engine`), Canadian-fintech tagline, Architecture Overview block from CLAUDE.md §3, and the `archive/v1-original` Previous Iteration section. **Rewrite.** |

No pre-existing content was destroyed — every factual claim in the
prior README and ADR was preserved or expanded, not dropped.

---

## 2. Files changed by this prompt

### 2.1 Rewrite — `docs/CONVENTIONS.md`

Replaced the condensed earlier version with a verbatim mirror of
CLAUDE.md §5 (Universal Coding Standards 5.1–5.7), §6 (Testing
Standards 6.1–6.4), §7 (Data Contracts and Lineage 7.1–7.3), and §9
(Anti-Patterns, 12 items). Added a preamble naming the dual-doc
relationship:

> CLAUDE.md is the agent-facing copy, this one is the reader-facing
> copy, and they are kept in lockstep. When the two ever disagree,
> CLAUDE.md wins and CONVENTIONS.md is updated to match in the same
> PR.

The spec's grep-q check for "temporal integrity" now hits three
lines (§6.1 table, §6.2 coverage rule, §6.3 test pattern heading) —
see §5.2 of this report.

### 2.2 Extend — `docs/ADR/0001-tech-stack.md`

Three additions, all surgical:

1. **Decision table** gained three rows — `Experimental ML |
   PyTorch`, `Batch store | PostgreSQL`, `Dashboards | Grafana` — and
   column widths were normalised.
2. **New rationale sections** — *Why PyTorch (experimental track
   only)* (scoped rationale: diversity models, torch-geometric
   requirement, not on serving path) and *Why Prometheus + Grafana*
   (pull-based scraping, fraud-native dashboards, local-friendly).
   The existing *Why Redis* section was expanded to *Why Redis +
   PostgreSQL (two-tier feature store)* — Postgres was already named
   in the original trade-off paragraph; it's now explicit.
3. **Consequences** section gained a new **Neutral** paragraph
   (Python-ecosystem skew, MLflow lock-in) and the Negative bullet
   was extended to call out the four-service local stack mitigated
   by `docker-compose.dev.yml`. Revisit triggers extended with a
   Grafana/Prometheus replacement scenario.

LightGBM / SHAP / torch-geometric / cleanlab / Optuna / FastAPI /
Pydantic+pandera / structlog / MLflow / prometheus-client rationale
is unchanged.

### 2.3 Rewrite — `README.md`

Structural rewrite to match the spec's outline:

| Section | What it carries |
|---------|-----------------|
| `# Fraud Detection Engine` + tagline | Spec-mandated title; Canadian-fintech framing. |
| Intro paragraph | End-to-end one-liner: LightGBM + SHAP + economic threshold + FastAPI + Redis/Postgres + Prometheus/Grafana. Points readers to CONVENTIONS.md and `docs/ADR/`. |
| Quick Start | `uv sync --all-extras`, `.env` copy, `make install`, `verify_bootstrap.py`. Follow-up block lists `make lint` / `format` / `typecheck` / `test-fast` / `test` / `test-lineage`. Links to `ci.yml`. |
| Architecture Overview | ASCII pipeline block copied verbatim from CLAUDE.md §3, followed by the five bullets (production model, diversity models, decision threshold, latency budget, target metrics). |
| Repository Layout | Eight-entry directory tree with inline comments. |
| Sprint Status | Seven-row table (Foundation → Monitoring & Documentation), `in progress` on Sprint 0, rest `pending`. |
| Previous Iteration | Links `archive/v1-original` via `../../tree/archive/v1-original`, notes the rebuild is independent and cross-refs CLAUDE.md §9 anti-pattern 11. |

The former "Setup / Quickstart / Layout" headings were absorbed into
the new structure; no content was lost.

---

## 3. Deviations from prompt spec

1. **CONVENTIONS.md is verbatim from CLAUDE.md, not a paraphrase.**
   The spec says "copy" — I read that strictly. Any future drift will
   show up as a diff between the two files, which is easier to audit
   than a paraphrase.
2. **ADR keeps the pre-existing "Why …" per-component structure** on
   top of the new Decision table entries. The spec lists
   Status/Context/Decision/Rationale/Consequences as top-level
   sections; the existing ADR folds Rationale into per-choice
   subsections, which is richer. I kept it and only added the three
   new components rather than flattening the whole ADR.
3. **README places the tagline as a blockquote immediately under the
   H1.** The spec shows an italic bullet; markdown renders blockquote
   slightly differently but carries the same emphasis and renders
   correctly on GitHub. No information loss.

None of the deviations change the spec's factual content.

---

## 4. Other files touched

None. No source code changes, no config changes, no test changes.
Docs-only prompt.

---

## 5. Verification output

### 5.1 `ls docs/CONVENTIONS.md docs/ADR/0001-tech-stack.md README.md`

```
-rw-r--r-- 1 dchit dchit  5167 Apr 21 14:45 README.md
-rw-r--r-- 1 dchit dchit  9105 Apr 21 14:44 docs/ADR/0001-tech-stack.md
-rw-r--r-- 1 dchit dchit 10595 Apr 21 14:41 docs/CONVENTIONS.md
```

All three files present.

### 5.2 `grep -q "temporal integrity" docs/CONVENTIONS.md && echo OK`

```
docs/CONVENTIONS.md:126: | Lineage | `tests/lineage/test_<contract>.py` | Schema validation, temporal integrity, data contracts |
docs/CONVENTIONS.md:132: - Every data transformation has both a unit test (correctness) and a lineage test (schema preservation, temporal integrity).
docs/CONVENTIONS.md:138: **Temporal integrity test** (per feature module):
```

Three hits, including the `Temporal integrity test` pattern heading
in §6.3. Spec check passes.

### 5.3 `uv run python scripts/verify_bootstrap.py`

```
[ OK ] ruff       ( 0.12s)
[ OK ] format     ( 0.06s)
[ OK ] mypy       ( 3.45s)
[ OK ] pytest     (13.09s)
[ OK ] settings   ( 0.24s)

Bootstrap: GREEN
```

No regression from the docs rewrite. Sprint 0 acceptance gate
remains green.

---

## 6. Acceptance checklist

- [x] `docs/CONVENTIONS.md` rewritten as verbatim mirror of CLAUDE.md
      §5/§6/§7/§9.
- [x] `docs/CONVENTIONS.md` contains the string `temporal integrity`
      (three occurrences — see §5.2).
- [x] `docs/ADR/0001-tech-stack.md` Decision table lists PyTorch,
      PostgreSQL, and Grafana alongside the pre-existing choices.
- [x] `docs/ADR/0001-tech-stack.md` carries
      Status/Context/Decision/Rationale/Consequences content, with
      per-component rationale blocks and Positive/Negative/Neutral
      consequences.
- [x] `README.md` title is `# Fraud Detection Engine` with the
      Canadian-fintech tagline.
- [x] `README.md` has Quick Start, Architecture Overview (CLAUDE.md
      §3 pipeline), Sprint Status, and Previous Iteration sections
      referencing `archive/v1-original`.
- [x] `scripts/verify_bootstrap.py` — Bootstrap: GREEN (§5.3).
- [x] Completion report written.

---

## 7. Post-completion — John's actions

1. Review [prompt_0_1_h_report.md](prompt_0_1_h_report.md) and the
   diffs in:
   - [docs/CONVENTIONS.md](../../docs/CONVENTIONS.md) — full rewrite.
   - [docs/ADR/0001-tech-stack.md](../../docs/ADR/0001-tech-stack.md)
     — three new decision rows and two new rationale sections.
   - [README.md](../../README.md) — structural rewrite.
2. Commit on a branch of his choice (likely folded into the ongoing
   Sprint 0 arc). No tag — still Sprint 0 housekeeping.
3. If CLAUDE.md §5/6/7/9 ever changes, update `docs/CONVENTIONS.md`
   in the same PR to keep the lockstep promise in its preamble.

Sprint 0 Prompt 0.1.h acceptance is fully met. **No git action from
me** (CLAUDE.md §2). Ready for John to commit.
