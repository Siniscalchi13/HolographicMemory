# HolographicMemory (Standalone)

This repository is the standalone version of TAI's holographic memory: C++ engine with a thin Python orchestration layer (CLI + optional API). It mirrors the architecture and math from TAI without re‑inventing algorithms.

Components

- Library: `holographic-fs/` (Python package with C++ extensions, CLI `holo`)
- Engine: `holographic-fs/native/holographic/` (C++17 via pybind11, FFTW/Accelerate)
- API: `services/api/` (FastAPI microservice for store/search/stats)
- Proofs: `documentation/proofs/` (Coq + markdown formal proof docs)
- Web UI: `services/api/static/` (Web interface served by FastAPI)
- Kubernetes: `k8s/` (Deployment, Service, PVC, Secret)

Install & Build

- Build C++ engine: `make cpp`
- Install library: `cd holographic-fs && python -m pip install -e .`
- Run tests: `make test`

Performance & Benchmarks

- Reproducible benchmarking: see `documentation/benchmarks/REPRODUCIBLE_BENCHMARKS.md`
- Report ranges with conditions (hardware, flags, dimension, corpus):
  - On Mac Studio M2 Max (this repo, package API, conservative harness):
    - Store (looped writes via Python): P50 typically ~100–250 ops/s for 4KB payloads
    - doc_id recall: ~0.2–0.5 ms P50 (O(1) in corpus size)
    - Search (semantic): grows with corpus size (O(n))
  - Native batch peak throughput (hitting store_batch directly) and GPU acceleration (Metal) can raise these significantly; see the benchmark harness JSON outputs and experimental GPU notes.
  - Historical peak numbers (e.g., ~140k–155k ops/s) were measured under specific native-only, SIMD-optimized conditions; treat as best-case and verify via the harness.

CLI

- Initialize: `holo init ./data --grid-size 64`
- Store: `holo store ./data/file.pdf`
- Search: `holo search file --root ./data`
- Stats: `holo stats --root ./data`
- Recall: not implemented by current C++ engine

API

- Local: `python -m pip install -r services/api/requirements.txt && uvicorn services.api.app:app --reload --port 8000`
- Docker (dev): `make build-images && make dev` → [http://localhost:8000](http://localhost:8000)
- Docker (prod): `make prod`
- Endpoints: `/healthz`, `/stats`, `/search`, `/list`, `/store`, `/metrics` (download is 501 until recall is implemented)
- Auth: `X-API-Key` header (set `HOLO_API_KEY` on server and desktop)

Web UI

- Prereq: API running locally (`make api` or `make dev`)
- Open browser: [http://localhost:8000](http://localhost:8000)
- Features: drag-and-drop store, search by filename, live stats, file previews

Enterprise Settings

- API key: set `HOLO_API_KEY` on server; web UI has settings panel for API key
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
 - Experimental GPU (Metal) scaffolding is included for Apple Silicon; see `holographic-fs/native/holographic/metal` and the benchmarking guide for usage and caveats.
