Architecture Justification (Math)
=================================

Principle
- Match algorithmic complexity and data locality to hardware strengths.

Design (GPU-First Architecture)
- GPU for all operations: batch FFTs, vector ops, 7-layer math, SNR calculations, wave validation — ideal for GPUs.
- CPU for fallback: Only used when GPU unavailable; provides same math API parity.
- 3D exact‑recall backend: spatial superposition provides byte‑perfect recall; v4/H4K8 act as semantic/compact representations.

Fallback Chain (Mathematically Sound)
1) GPU backend (primary) — full math API with Metal/CUDA/ROCm acceleration.
2) CPU backend (fallback) — same math API when GPU unavailable.
3) Exact recall (3D) — direct spatial readback.
4) HRR chunk/base64 — associative store if present.
5) v4/H4K8 decode — inverse sparse spectral reconstruction.

Optimality
- Ensures minimal latency on retrieval and maximal throughput on ingest while preserving mathematical correctness.

