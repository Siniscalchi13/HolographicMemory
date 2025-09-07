############################
# Builder: wheels + native  #
############################
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config curl git \
    libfftw3-dev libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY holographic-fs /build/holographic-fs
COPY services/api/requirements.txt /build/requirements.txt

RUN python -m pip install --upgrade pip wheel setuptools pybind11 && \
    python -m pip wheel /build/holographic-fs/native/holographic -w /build/wheels && \
    python -m pip wheel /build/holographic-fs -w /build/wheels && \
    python -m pip wheel -r /build/requirements.txt -w /build/wheels

############################
# Development image (reload)
############################
FROM python:3.12-slim AS api-dev

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config \
    libfftw3-dev libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/wheels /wheels
RUN python -m pip install --upgrade pip && \
    python -m pip install /wheels/* && \
    python -m pip install pytest

# Mount code in dev via bind mounts; keep structure present
COPY services/api /app/services/api
COPY holographic-fs /app/holographic-fs

# Install editable for hot reload
RUN python -m pip install -e /app/holographic-fs

ENV HOLO_ROOT=/data \
    GRID_SIZE=64 \
    HOLO_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173 \
    PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --retries=5 CMD curl -fsS http://localhost:${PORT}/healthz || exit 1

CMD ["uvicorn", "services.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

############################
# Production image (minimal)
############################
FROM python:3.12-slim AS api

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfftw3-3 libopenblas0-pthread curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*

# Copy API code only
COPY services/api /app/services/api

# Create non-root user
RUN useradd -m -u 10001 appuser && mkdir -p /data && chown -R appuser:appuser /data
USER appuser

ENV HOLO_ROOT=/data \
    GRID_SIZE=64 \
    PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:${PORT}/healthz || exit 1

CMD ["uvicorn", "services.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
