Validation Results (Snapshot)
=============================

Planned
- Unit validation for v4/H4K8 decode against synthetic fixtures (top‑K spectral bins).
- O(N·K) complexity confirmation with timing vs K and N.
- End‑to‑end store→retrieve across size regimes; fallback correctness with/without 3D backend.

Status
- Code paths updated: sidecar fallback fixed; .hwp decode fallback added; HRR backend restored to CPU.
- Runtime validation to be executed on target hardware (Metal/CUDA/ROCm availability varies).

