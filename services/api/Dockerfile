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
# CUDA builder (holographic_gpu)
############################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS cuda-builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common gnupg ca-certificates curl git cmake build-essential pkg-config \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
       python3.12 python3.12-dev python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m pip install --upgrade pip pybind11 numpy

COPY holographic-fs /build/holographic-fs

# Configure and build the GPU pybind11 module with CUDA backend
RUN cmake -S /build/holographic-fs/native/holographic -B /build/gpu \
        -Dpybind11_DIR=$(python3.12 -m pybind11 --cmake-dir) \
        -DBUILD_CUDA=ON -DBUILD_METAL=OFF -DBUILD_ROCM=OFF \
    && cmake --build /build/gpu --config Release -j

# Collect the resulting module for later stages
RUN mkdir -p /gpu_artifacts && \
    cp /build/gpu/*.so /gpu_artifacts/ || true

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
COPY services /app/services
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

# Copy all services
COPY services /app/services

# Create non-root user
RUN useradd -m -u 10001 appuser && mkdir -p /data && chown -R appuser:appuser /data
USER appuser

ENV HOLO_ROOT=/data \
    GRID_SIZE=64 \
    PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:${PORT}/healthz || exit 1

CMD ["uvicorn", "services.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

############################
# Production GPU (CUDA) image
############################
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS api-gpu-cuda

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Python 3.12 + runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common gnupg ca-certificates curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
       python3.12 python3.12-venv python3.12-dev python3-pip \
       libfftw3-3 libopenblas0-pthread \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/wheels /wheels
RUN python3.12 -m pip install --upgrade pip && \
    python3.12 -m pip install --no-cache-dir /wheels/*

# Install holographic_gpu from CUDA builder
COPY --from=cuda-builder /gpu_artifacts /tmp/gpu
RUN python3.12 - <<'PY'
import sysconfig, shutil, glob
plat = sysconfig.get_paths().get('platlib') or sysconfig.get_paths()['purelib']
mods = glob.glob('/tmp/gpu/holographic_gpu*.so')
assert mods, 'holographic_gpu module not built'
for m in mods:
    shutil.copy2(m, plat)
print('Installed holographic_gpu into', plat)
PY

# Copy services
COPY services /app/services

# Create non-root user and data dir
RUN useradd -m -u 10001 appuser && mkdir -p /data && chown -R appuser:appuser /data
USER appuser

ENV HOLO_ROOT=/data \
    GRID_SIZE=64 \
    PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:${PORT}/healthz || exit 1

CMD ["python3.12", "-m", "uvicorn", "services.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
