HolographicMemory (Standalone)

This repository is the standalone version of TAI's holographic memory: C++ engine with a thin Python orchestration layer (CLI + optional API). It mirrors the architecture and math from TAI without re‑inventing algorithms.

Components

- Library: `holographic-fs/` (Python package with C++ extensions, CLI `holo`)
- Engine: `holographic-fs/native/holographic/` (C++17 via pybind11, FFTW/Accelerate)
- API: `services/api/` (FastAPI microservice for store/search/stats)
- Proofs: `documentation/proofs/` (Coq + markdown formal proof docs)
- Desktop: `desktop/holo-desktop` (Electron app for drag-and-drop store/search/stats)
- Kubernetes: `k8s/` (Deployment, Service, PVC, Secret)

Install & Build

- Build C++ engine: `make cpp`
- Install library: `cd holographic-fs && python -m pip install -e .`
- Run tests: `make test`

CLI

- Initialize: `holo init ./data --grid-size 64`
- Store: `holo store ./data/file.pdf`
- Search: `holo search file --root ./data`
- Stats: `holo stats --root ./data`
- Recall: not implemented by current C++ engine

API

- Local: `python -m pip install -r services/api/requirements.txt && uvicorn services.api.app:app --reload --port 8000`
- Docker (dev): `make build-images && make dev` → http://localhost:8000
- Docker (prod): `make prod`
- Endpoints: `/healthz`, `/stats`, `/search`, `/list`, `/store`, `/metrics` (download is 501 until recall is implemented)
 - Auth: `X-API-Key` header (set `HOLO_API_KEY` on server and desktop)

Desktop GUI

- Prereq: API running locally (`make api` or `make dev`)
- Start app: `cd desktop/holo-desktop && npm install && npm start`
- Features: drag-and-drop store, search by filename, live stats

Enterprise Settings

- API key: set `HOLO_API_KEY` on server; desktop reads `HOLO_API_KEY` env
- Allowed origins: `HOLO_ALLOWED_ORIGINS` (comma-separated)
- Metrics: Prometheus at `/metrics`

Kubernetes

- Apply manifests: `kubectl apply -f k8s/`
- Configure Secret `holo-secrets` `api_key`
- PVC `holo-data` retains stored index and files

Math & Proofs

- See `documentation/proofs/` (copied from TAI): capacity theorem, correctness, Coq models
- The implementation follows the same encoding and query calculus; changes here adjust packaging/integration only

Notes

- This repo does not modify TAI; it references the same math/algorithms and hosts a standalone build + API surface.
- Current engine API is string‑based; binary files are indexed by `filename:..` headers to enable wave persistence.
- If you need byte‑level recall, the C++ engine requires an explicit method; tracked for a later update.
