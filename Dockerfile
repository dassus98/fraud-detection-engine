# syntax=docker/dockerfile:1.7
#
# Multi-stage Dockerfile for the fraud-detection FastAPI service.
#
# Sprint 5 prompt 5.1.g. Builds the production image from the Sprint 5.1.f
# `src/fraud_engine/api/main.py` keystone — non-root, minimal-base, with a
# /health-driven HEALTHCHECK so an orchestrator can route around an
# unhealthy instance.
#
# Layout:
#   Stage 1 (builder): python:3.11-slim-bookworm + build-essential + uv.
#     Installs all 270+ runtime deps into /opt/venv via `uv sync --frozen
#     --no-dev --no-install-project`, then `uv pip install -e .` for the
#     project package itself.
#   Stage 2 (runtime): python:3.11-slim-bookworm + libgomp1 (LightGBM
#     OpenMP runtime) + curl (HEALTHCHECK). Copies /opt/venv from
#     builder, runs as a non-root `app` user (UID 10001).
#
# Build:
#   docker build -t fraud-engine:dev .
#
# Run (standalone — assumes Redis/Postgres reachable on the host network).
# Pass REDIS_URL + POSTGRES_URL via -e flags; see .env.example for the
# canonical credential format. The standalone path is mostly for
# debugging; the compose path below is the supported entry point.
#   docker run --rm -p 8000:8000 \
#     -e REDIS_URL=redis://host.docker.internal:6379/0 \
#     -e POSTGRES_URL=... \
#     fraud-engine:dev
#
# Run (compose — preferred):
#   docker compose up -d
#
# Image size + build time are measured during 5.1.g spec verification and
# recorded in `sprints/sprint_5/prompt_5_1_g_report.md`.

ARG PYTHON_VERSION=3.11-slim-bookworm
ARG UV_VERSION=0.11.7

# Named uv-source stage. Buildkit forbids ARG substitution inside the
# `--from=<image>` field (image reference is parsed before ARG resolution),
# but FROM does substitute ARGs cleanly — so we declare uv as a named
# stage here and reference it by name from the builder's COPY below.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-source

# ---------------------------------------------------------------------
# Stage 1: builder — uv + deps + project install.
# ---------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS builder

# Pull uv in via the named stage above so the version stays pinned via
# the UV_VERSION ARG.
COPY --from=uv-source /uv /usr/local/bin/uv

# Build deps. LightGBM + numpy + pyarrow have native wheels for slim
# x86_64 so build-essential is rarely invoked, but keeping it lets a
# rare from-source compile succeed (e.g. on a future torch-geometric
# bump that ships no wheel for our pinned torch version).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy lock-deciding files first so the deps layer caches cleanly.
# README.md is required because pyproject.toml's [project] table
# references it; uv reads pyproject.toml during sync.
COPY pyproject.toml uv.lock README.md ./

# Create the venv up front + install runtime deps only (no dev extras,
# no project install). `--frozen` enforces the lockfile bit-exactly.
RUN uv venv /opt/venv \
    && uv sync --frozen --no-dev --no-install-project

# Now copy the project source. This layer is the cache-bust frontier:
# changes to src/ rebuild from here, but the deps layer above stays
# cached.
COPY src/ ./src/

# Install the project NON-editable so the package contents live inside
# `/opt/venv/lib/python3.11/site-packages/fraud_engine/`. A `-e` (editable)
# install would write a `.pth` pointing back at `/build/src/...`, but
# the runtime stage doesn't have `/build/` — only `/opt/venv` and
# `/app/src`. Without this, `import fraud_engine` raises
# `ModuleNotFoundError` at uvicorn startup.
RUN uv pip install --python /opt/venv/bin/python .

# ---------------------------------------------------------------------
# Stage 2: runtime — minimal base + non-root user.
# ---------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS runtime

# libgomp1: OpenMP runtime that LightGBM links against. Without it,
# `import lightgbm` raises `ImportError: libgomp.so.1: cannot open
# shared object file` at API startup.
# curl: used by HEALTHCHECK below. Adds ~6 MB; alternatives (urllib via
# python -c) would couple the healthcheck to the venv path which is
# fragile across Dockerfile rewrites.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. UID 10001 is high enough to avoid host UID collisions
# (typical UIDs are 1000-1999) and matches the Kubernetes pod-security
# `runAsUser: 10001` convention. `--system` skips the home-dir-creation
# spam in `useradd`'s logs; we set WORKDIR explicitly below.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --shell /bin/bash app

WORKDIR /app

# Copy the populated venv from the builder. --chown ensures the runtime
# user owns the python tree (it's read-only at runtime, but ownership
# matters if a debug user shells in and needs to inspect or rebuild).
COPY --from=builder --chown=app:app /opt/venv /opt/venv

# Copy the project files the API needs at runtime. NOT data/ (large +
# gitignored), NOT logs/ (per-run scratch), NOT tests/ (verification),
# NOT auxiliary models (fraudnet/fraudgnn — Model B/C are out of scope
# for the Sprint 5.1.f request path). The .dockerignore catches most of
# this; explicit COPYs here are belt-and-suspenders.
COPY --chown=app:app src/                                                       ./src/
COPY --chown=app:app configs/                                                   ./configs/
COPY --chown=app:app pyproject.toml README.md                                   ./
COPY --chown=app:app models/sprint3/lightgbm_model.joblib                       ./models/sprint3/
COPY --chown=app:app models/sprint3/calibrator.joblib                           ./models/sprint3/
COPY --chown=app:app models/sprint3/lightgbm_model_manifest.json                ./models/sprint3/
COPY --chown=app:app models/pipelines/tier1_pipeline.joblib                     ./models/pipelines/

# Switch to non-root for the runtime layers.
USER app

# Activate the venv via PATH so `uvicorn` and `python` resolve to the
# venv copies. PYTHONUNBUFFERED so stdout flushes per-line (essential
# for structlog JSON to land in `docker logs` immediately).
# PYTHONDONTWRITEBYTECODE saves a few KB of .pyc files in the image.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# /health returns 200 immediately once the process is up. The lifespan
# (model joblib load + Redis/Postgres connect) takes ~2-3 s; start-period
# 15 s budgets that with safety. interval=10s + retries=3 means a
# stuck process is marked unhealthy within ~40 s.
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/health || exit 1

# Use the array form so signals (SIGTERM from `docker stop`) reach
# uvicorn directly without going through a shell, which would swallow
# them and force a 10s SIGKILL kill instead of a graceful shutdown.
CMD ["uvicorn", "fraud_engine.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
