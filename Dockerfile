# syntax=docker/dockerfile:1
#
# Single-image deployment: the React bundle is built in stage one and served
# by FastAPI in stage two, so the whole product runs as one free container.
# Render's free web-service tier (see render.yaml at the repo root) is the
# recommended $0 host.
#
#   docker build -t rag-knowledge-platform .
#   docker run -p 7860:7860 rag-knowledge-platform
#
# With no environment variables it boots into deterministic demo mode
# (extractive answering) on SQLite + FAISS, seeded with a small demo corpus.
# Supply ANTHROPIC_API_KEY to upgrade to Claude-generated answers in place.

# ---------------------------------------------------------------------------
# Stage 1 - build the frontend
# ---------------------------------------------------------------------------
FROM node:22-alpine AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
ENV VITE_API_BASE=""
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 - runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    APP_ENV=production \
    SEED_ON_STARTUP=true

# faiss-cpu ships wheels for manylinux; libgomp1 is its only runtime shared
# library dependency not already present in the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/
COPY --from=frontend /build/dist ./backend/static

RUN mkdir -p backend/data/runtime backend/data/faiss_index && chown -R app:app /app
USER app

WORKDIR /app/backend
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4).status == 200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
