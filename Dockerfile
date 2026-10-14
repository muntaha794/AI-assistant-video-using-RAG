# ── AI Meeting Assistant ─ production image ──────────────────────────────────
# Installs the full dependency set (including the ML extras: Whisper, torch,
# pyannote, ChromaDB) plus ffmpeg for audio decoding.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first for better build caching.
COPY Requirements.txt requirements-base.txt requirements-ml.txt requirements-dev.txt ./
RUN pip install -r Requirements.txt

# Application code.
COPY alembic.ini pyproject.toml ./
COPY migrations ./migrations
COPY app ./app
COPY eval ./eval
COPY scripts ./scripts

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/uploads /app/data/chroma \
    && chown -R appuser:appuser /app/data
USER appuser

ENV APP_ENV=production \
    AUTO_CREATE_TABLES=false \
    UPLOAD_DIR=/app/data/uploads \
    CHROMA_PERSIST_DIR=/app/data/chroma \
    LOG_JSON=true

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

# Run migrations, then serve. $PORT is injected by the platform when present.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
