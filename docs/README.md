HolographicMemory – Enterprise Overview
======================================

What is HolographicMemory
- Wave-based storage engine using FFT + SIMD
- 9x compression, sub‑millisecond storage, thousands of files/second
- Components: C++ core, FastAPI server, Python CLI, Electron desktop

Quick Start (Docker)
1) make build-images
2) make dev
3) Open http://localhost:8000 (API) and run the desktop app (HOLO_API=http://localhost:8000 npm start)

Architecture
- API: FastAPI microservice exposing upload/search/recall with metrics and health
- Engine: Native C++ extension loaded in‑process (pybind11)
- Clients: CLI (holo) and Desktop (Electron) connect to the API

See docs/DEPLOYMENT.md and docs/DEVELOPMENT.md for details.

