FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=0

# Supply-chain safe: copy uv from the pinned upstream image layer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder

WORKDIR /app

# ── Layer A: resolve and install dependencies only ───────────────
COPY pyproject.toml uv.lock README.md ./

# --no-install-project: skip flowportfolio itself so this expensive
# layer is fully reused on source-only commits.
RUN uv sync --frozen --no-install-project --no-dev

# ── Layer B: copy source, then install the package ───────────────
COPY flowportfolio/ ./flowportfolio/

# Full sync: installs flowportfolio into the already-populated venv.
RUN uv sync --frozen --no-dev

# ----------------------------------------------------------------
# Stage: orchestrator-builder
# Extends builder with the locked `orchestrator` optional dependency
# group (Prefect).
# ----------------------------------------------------------------
FROM builder AS orchestrator-builder

RUN uv sync --frozen --extra orchestrator --no-dev

# ----------------------------------------------------------------
# Stage: runtime
# Shared foundation for all runtime targets.
# ----------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Create non-root user before any COPY so --chown can resolve them
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 \
               --gid 1001 \
               --no-create-home \
               --shell /sbin/nologin \
               appuser

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/flowportfolio /app/flowportfolio

USER appuser

# Default entrypoint verifying import
CMD ["python", "-c", "import flowportfolio; print('flowportfolio', flowportfolio.__version__)"]

# ----------------------------------------------------------------
# Stage: orchestrator
# Inherits the shared runtime base and copies the orchestrator-specific
# venv (with locked Prefect).
# ----------------------------------------------------------------
FROM runtime AS orchestrator

COPY --from=orchestrator-builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/flowportfolio /app/flowportfolio

CMD ["prefect", "worker", "start", "--pool", "default-agent-pool"]
