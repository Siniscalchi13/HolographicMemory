Code ↔ Math Mapping (Current)
=============================

Purpose
- Provide a verifiable map from mathematical operations to concrete code artifacts with file/line references.

Core Wave Mathematics
- Wave superposition (encode): ψ_i = iFFT(FFT(data_i) × codebook_i); ψ_total = Σ_i ψ_i
  - `services/holographic-memory/core/native/holographic/metal/holographic_memory.metal:10` (forward transform kernel)
  - `services/holographic-memory/core/native/holographic/metal/holographic_memory.metal:61` (seeded phase codebook multiply)
  - `services/holographic-memory/core/native/holographic/metal/holographic_memory.metal:38` (inverse transform kernel)
  - `services/holographic-memory/core/native/holographic/metal/holographic_memory.metal:111` (time-domain accumulation)
  - Orchestration (encode): `services/holographic-memory/core/native/holographic/gpu_binding.cpp:168`

- Correlation decoding: reconstruction = iFFT(FFT(ψ_total) × conj(codebook_i))
  - `services/holographic-memory/core/native/holographic/metal/holographic_memory.metal:86` (conjugate codebook multiply)
  - Orchestration (decode): `services/holographic-memory/core/native/holographic/gpu_binding.cpp:199`

SNR and Capacity
- SNR_k ≈ sqrt(D_k / N_k); D_k* = M · (α_k² / N_k) / Σ_j (α_j² / N_j)
  - Host-side 7-layer state and calculations: `services/holographic-memory/core/native/holographic/gpu_binding.cpp:236`

ECC RS(255,223)
- Encode parity (GPU kernel):
  - `services/holographic-memory/core/native/holographic/metal/holographic_memory.metal:144`
  - Host driver: `services/holographic-memory/core/native/holographic/metal/MetalBackend.mm:560`
  - Python binding: `services/holographic-memory/core/native/holographic/gpu_binding.cpp:852`
- Decode/correct (BM/Chien/Forney, host-side):
  - `services/holographic-memory/core/native/holographic/metal/MetalBackend.mm:694`
  - Python binding: `services/holographic-memory/core/native/holographic/gpu_binding.cpp:867`
- Recall integration and ECC application per chunk:
  - `services/holographic-memory/core/holographicfs/memory.py:504`

Containers and Compression
- HGMC2/HGMC3 write/read + ECC metadata and parity blobs:
  - Write: `services/holographic-memory/core/holographicfs/memory.py:240`
  - Read:  `services/holographic-memory/core/holographicfs/memory.py:431`
- Sparse + entropy coding kernels/bindings:
  - `services/holographic-memory/core/native/holographic/metal/holographic_memory.metal:880` (entropy coding; reference kernel)
  - Python bindings: `services/holographic-memory/core/native/holographic/gpu_binding.cpp:1008`

H4M1/H4K8/v4 Retrieval Math
- Inverse synthesis (sparse bins → dense spectrum → iDFT):
  - Reference: `documentation/mathematics/H4M1_H4K8_RETRIEVAL_MATH.md:1`
  - Decoder:   `services/holographic-memory/core/native/holographic/hwp_v4_decode.cpp`

Python Orchestration (GPU‑only heavy path)
- Store bytes to container with superposition and ECC:
  - `services/holographic-memory/core/holographicfs/memory.py:220`
- Recall bytes via FFT/correlation and ECC correction:
  - `services/holographic-memory/core/holographicfs/memory.py:431`

Notes
- FFT normalization: Metal forward kernel applies 1/N scaling; inverse omits scaling; MPSGraph paths use unitary scaling. Encode/decode pipelines cancel scaling internally; standardization is tracked in the build plan.
