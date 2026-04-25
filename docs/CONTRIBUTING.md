# Contributing

This document is the **branching, PR, and merge conventions** for the
fraud-detection-engine. It is the senior-engineer-portfolio counterpart
to [CLAUDE.md](../CLAUDE.md) §10 (per-prompt operating procedure): the
agent works the prompt, the human owns version control, and the *shape*
of what lands on `main` is captured here.

The reviewer audience for this repo is hiring committees at Wealthsimple,
Mercury, RBC, and Nubank. They will read `git log --graph main`, browse
the PR list, and judge whether the workflow looks like a senior
engineer's habits. The conventions below are tuned to that audience.

---

## 1. Branching model — trunk-based, one branch per prompt

The unit of work is the **prompt**, not the sprint. Each prompt gets:

- its own short-lived branch off the latest `main`,
- a single PR back to `main`,
- a squash-merge so `main` has one commit per logical unit.

`main` is always in a known-green state: lint, typecheck, tests,
`scripts/verify_bootstrap.py` all pass before any PR is merged. There
are no long-lived sprint branches, no release branches, and no merge
commits on `main` outside the squash-merge per PR.

### 1.1 Branch naming

```
sprint-<N>/prompt-<X>-<Y>-<short-slug>
```

- `<N>` — sprint number (0, 1, 2, …)
- `<X>-<Y>` — prompt number within the sprint, dot-separated promoted
  to dash (so `0.3.c` becomes `0-3-c`)
- `<short-slug>` — 2–4 dashed words capturing the prompt's intent

Examples:

```
sprint-0/prompt-0-1-a-bootstrap
sprint-0/prompt-0-3-c-mlflow-setup
sprint-1/prompt-1-2-b-temporal-split
sprint-3/prompt-3-4-c-optuna-tuning
```

The sprint folder structure (`sprints/sprint_<N>/prompt_<X>_<Y>_report.md`)
mirrors branch names so a reviewer can match a PR to its completion
report instantly.

### 1.2 One branch in flight at a time

Open a branch, finish the prompt, merge the PR, delete the branch.
Don't stack multiple in-flight branches — sprints in this repo are
sequential by design (each prompt builds on the previous one's reports
and code), so parallel branches just create needless rebases.

### 1.3 Sprint 0 is grandfathered

Sprint 0 was built on a single `sprint-0/bootstrap` branch before this
convention existed. It lands as one PR. From **Sprint 1 onwards** every
prompt gets its own branch.

---

## 2. PR conventions

### 2.1 Title

```
<sprint>.<prompt>: <imperative summary>
```

Examples:

```
0.3.c: wire MLflow tracking + experiment bootstrap
1.2.b: temporal train/val/test split with leakage guard
3.4.a: Optuna hyperparameter search for LightGBM
```

Under 70 chars. The body holds the detail; the title is what shows in
`git log --oneline`.

### 2.2 Body

Use this template (it mirrors the prompt completion report):

```markdown
## What this PR does
- 1–3 bullets, the business rationale not the file list

## Files changed
- short list of the meaningful files (skip tests + docs unless they're the point)

## Test plan
- [x] make lint
- [x] make typecheck
- [x] pytest tests/unit
- [x] pytest tests/lineage   (if touching pipelines)
- [x] pytest tests/integration   (if touching multi-component flows)
- [x] python scripts/verify_bootstrap.py   (or verify_lineage.py for sprint 1+)

## Completion report
Full report: [sprints/sprint_X/prompt_Y_report.md](sprints/sprint_X/prompt_Y_report.md)
```

The completion report is the *durable* artifact — it lives in-repo
forever and survives the PR being deleted. The PR body is the
*transient* index that points to it.

### 2.3 Squash-merge, always

`gh pr merge --squash --delete-branch` (or the equivalent button in the
GitHub UI). Reasons:

- One PR = one commit on `main`. `git log --oneline main` reads as a
  feature changelog, not a stream of WIP commits.
- The branch's WIP history (typo fixes, "actually do X correctly",
  reverted attempts) is uninteresting forever after merge — squash
  drops it.
- The PR URL itself is the permanent anchor for the full diff if
  anyone (you, a reviewer, future-you) wants the granular trail.

`--no-ff` merges and rebase-merges are **not** used in this repo.

### 2.4 PR size

A PR should land in **one sitting** for a reviewer — under ~400 LOC
diff is the target, under ~800 LOC is the cap. If a prompt is larger
than that, it was scoped wrong; split it before opening the PR. Senior
reviewers reject big PRs on principle.

---

## 3. Pre-merge checklist

Every PR must satisfy all of these before merge:

1. **All verification commands return 0** (CLAUDE.md §11):
   ```bash
   make lint
   make typecheck
   make test       # or chunked equivalent if make test OOMs locally
   python scripts/verify_bootstrap.py   # sprint 0
   python scripts/verify_lineage.py     # sprint 1+
   ```
2. **Completion report exists** at `sprints/sprint_<N>/prompt_<X>_<Y>_report.md`
   and is referenced in the PR body.
3. **No `print()`, no hardcoded values, no missing docstrings** —
   covered by lint, but worth eyeballing the diff once.
4. **No `archive/v1-original` references** anywhere (CLAUDE.md §9 #11).
5. **No git commands run by the agent** (CLAUDE.md §2). The diff
   should land cleanly on `main` via human-driven push + PR.

Pre-commit hooks enforce most of (1) and (3) automatically. (2), (4),
and (5) are eyeball checks.

---

## 4. Sprint completion

When the last prompt of a sprint merges:

1. Pull `main` locally — it should now contain every prompt's
   squash-merged commit plus the latest gate report.
2. Run the full sprint gate locally one more time on `main` to confirm
   nothing degraded between the last prompt's PR check and the merged
   state:
   ```bash
   make lint && make typecheck && make test && \
     python scripts/verify_bootstrap.py
   ```
3. Tag and push:
   ```bash
   git tag sprint-<N>-complete
   git push origin sprint-<N>-complete
   ```
4. Update `CLAUDE.md` §13 sprint status table in the next sprint's
   first PR (not as its own commit on `main`).

Tags are immutable in this repo. If a sprint needs a hotfix after
tagging, that's a new prompt + PR, not a re-tag.

---

## 5. Roles — who does what

Per CLAUDE.md §2.1 (the PR-plumbing carve-out):

| Action | Agent (Claude Code) | Human (John) |
|---|---|---|
| Read CLAUDE.md, this file, prior reports | ✅ | — |
| Implement the prompt | ✅ | — |
| Run tests / lint / typecheck locally | ✅ | — |
| Write the prompt completion report | ✅ | — |
| Create feature branch (`git checkout -b`) | ✅ | — |
| Stage and commit (`git add` / `git commit`) | ❌ | ✅ |
| Push feature branch (`git push -u origin <branch>`) | ✅ (never `main`, never `--force`) | — |
| Open PR (`gh pr create`) | ✅ | — |
| Read-only diagnostics (`git status`, `gh pr view`, etc.) | ✅ | — |
| Squash-merge PR (`gh pr merge --squash --delete-branch`) | ❌ | ✅ |
| Tag (`git tag sprint-<N>-complete`) | ❌ | ✅ |
| Force-push, push to `main`, rebase, reset --hard, revert | ❌ | (avoid) |

CLAUDE.md §2.1 is the load-bearing rule for what the agent may run; §2.2
is the exhaustive forbidden list. The agent's deliverable per prompt is:
a clean working tree + a completion report → (after John commits) a
pushed branch + an open PR. The squash-merge to `main` is John's click.

---

## 6. Why this shape

- **Trunk-based** matches what the target audience (modern fintech)
  uses day-to-day. Long-lived sprint branches signal legacy practice.
- **Squash-merge per PR** keeps `main` history reviewable in
  `git log --oneline` — recruiters and senior engineers actually read
  this.
- **One PR per prompt** produces ~80–100 small reviewable PRs across
  the project. That PR list *is* part of the portfolio.
- **Completion report in-repo** survives if the PR is ever deleted or
  the GitHub repo is migrated. The git tree is the durable artifact.
- **Human-owned commits and merges** prevent the failure mode where the
  agent makes a "helpful" commit at an awkward moment (hook failures
  mid-amend, force-pushes, accidental tag motion). The agent handles the
  mechanical plumbing in between (branch + push + `gh pr create`) so
  John isn't typing six commands per prompt, but the consequential
  decisions — what's in the commit, what lands on `main`, what gets
  tagged — stay with him.

---

## 7. Reading order at the start of every Claude Code session

Per CLAUDE.md §10:

1. `CLAUDE.md` — full read, every session.
2. **`docs/CONTRIBUTING.md`** (this file) — full read, every session.
3. `docs/CONVENTIONS.md` — coding standards mirror.
4. The most recent `sprints/sprint_<N>/prompt_<X>_<Y>_report.md`.
5. Any files the prompt explicitly references.

Then state the standard preflight: *"I have read CLAUDE.md,
CONTRIBUTING.md, and the prior completion report. Current state is:
[one-sentence summary]. Proceeding with [prompt task]."*

---

_End of CONTRIBUTING.md._
