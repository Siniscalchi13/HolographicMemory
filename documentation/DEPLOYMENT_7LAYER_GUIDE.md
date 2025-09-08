# Deployment Guide: 7‑Layer Holographic Memory

Version: 1.0

## Prerequisites
- Python 3.11+
- Build native holographic backends (see holographic-fs/native/holographic/README.md)
- Optional: Prometheus for telemetry scraping

## Environment Variables
- `HOLO_ROOT`: data root (default `./data`)
- `GRID_SIZE`: holographic grid size (e.g., 64)
- `HLOG_DATA_DIR`: wave persistence path (defaults under HOLO_ROOT)
- `HOLO_TOPK`: Top‑K for v4 (default 32)
- `HOLO_MICRO_THRESHOLD`: small‑file micro cutoff (default 256)
- `HOLO_FALLBACK_BASE64`: `true|false` sidecar fallback (Vault overrides to false)

## Services
- API service (FastAPI) initializes:
  - Router: `MathematicalRouter`
  - Telemetry: `PerformanceTelemetry`
  - Vault guard: `SecurityGuard`

## Start
1. Build native modules: `make cpp`
2. Install Python libs: `pip install -e holographic-fs`
3. Run API: `uvicorn services.api.app:app --reload`

## Vault Policy
- Secrets detected are routed to Vault, persisted with micro header only and random `doc_id`.
- No sidecar; 3D backend is required for reconstruction; ensure encryption at rest.

