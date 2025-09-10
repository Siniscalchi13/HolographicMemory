Existing Math Analysis
======================

Components Reviewed
- GPU encode (Metal/CUDA/ROCm) — batch FFT path maps input rows to holographic vectors.
- CPU HRR engine — unitary FFT, phase‑only keys, associative memory via circular convolution.
- v4 decoder (C++) — reconstructs bytes from H4K8/HWP4V001 sparse spectral payloads.

Findings
- Unitary scaling: both forward/inverse FFT paths normalize by 1/√N (HRR binding) ensuring numerical stability.
- Phase‑only keys improve unbinding robustness (HRR), maintaining invertibility with conjugate in frequency.
- v4 decoder uses stable varint parsing and 10‑bit phase quantization; inverse synthesis matches spec.

Performance Model
- Encode (GPU): dominated by batched FFT + memory bandwidth; scales with batch size and dimension.
- Decode (CPU): O(N·K) sparse reconstruction with small K (≤32) is efficient; dominated by per‑bin phase rotations and resampling.

Parity
- H4M1 is intentionally header‑only; no mismatch — requires 3D backend to recall.
- H4K8/v4 decode is CPU‑side and portable; proven correct by direct synthesis of spectral bins.

