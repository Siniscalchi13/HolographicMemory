# ECC Design: Wave ECC for HolographicMemory

This document describes the Wave-based ECC used in HolographicMemory containers: parameters, code paths, and validation. Wave ECC combines seeded, redundant spectral “views” with correlation-based verification to enable robust, variable-length error correction without rigid block sizing.

## Parameters and Limits
- Redundancy level `R` (int ≥ 2 recommended; default 3): number of parity views.
- Seed base (uint32): deterministic seed (derived from content hash) used to derive per-view seeds.
- Data size: arbitrary length (no k/n padding requirements).
- Detection threshold: cosine similarity between recovered views; current default threshold 0.95.
- Overhead: Parity size scales with R and signal dimension; examples from tests:
  - 10 bytes, R=3 → parity ≈ 240 bytes
  - 57 bytes, R=4 → parity ≈ 1824 bytes
  Actual overhead depends on chosen FFT dimension and internal layout.

## Implementation Overview
- Binding: `services/holographic-memory/core/native/holographic/gpu_binding.cpp`
- APIs:
  - `wave_ecc_encode(data: bytes, redundancy_level: int, seed_base: int) -> bytes`
  - `wave_ecc_decode(data: bytes, parity: bytes, redundancy_level: int, seed_base: int) -> tuple[bytes, int]`
- Method (high level):
  1) Encode forms R seeded spectral parity views using FFT → seeded codebook → iFFT.
  2) Decode runs conjugate correlation per view to reconstruct spectral estimates.
  3) Valid views are identified via pairwise cosine similarity; corrupted views are ignored.
  4) The best valid view is iFFT’d to time domain and re-quantized to bytes.
  5) Parity revalidation on corrected bytes ensures integrity (see adapter below).

## Integration Points
- Container store path (HGMC2):
  - Path: `services/holographic-memory/core/holographicfs/memory.py`
  - Function: `Memory.store_bytes(...)`
  - Behavior: Computes per-chunk Wave ECC parity (`ecc_scheme=2`), with header fields:
    - `ecc_k = redundancy_level` (R)
    - `ecc_r = seed_base` (deterministic per container)
- Container recall path (HGMC2):
  - Path: `services/holographic-memory/core/holographicfs/memory.py`
  - Function: `Memory.retrieve_bytes(...)`
  - Behavior: GPU correlation reconstructs chunk bytes; `verify_and_correct_rs(...)` adapter calls `wave_ecc_decode`, then recomputes parity via `wave_ecc_encode` and raises on mismatch.
- Adapter for backward compatible API name:
  - Path: `services/holographic-memory/core/holographicfs/memory.py`
  - Function: `verify_and_correct_rs(payload, parity, k=3, r=0)`
  - Maps `k → redundancy_level`, `r → seed_base`; enforces parity recheck post-correction.

## Tests and Validation
- Wave ECC tests (repo top level): `test_wave_ecc_simple.py`, `test_wave_ecc.py`, `test_wave_ecc_debug.py`, `test_wave_ecc_proof.py`
  - No-error roundtrip succeeds across sizes
  - Byte corruption recovered for typical R (3–5)
  - Parity revalidation enforced
- E2E container parity corruption: `services/holographic-memory/core/tests/test_hgmc_e2e.py::test_hgmc2_parity_mismatch_fails_e2e`
  - Corrupting a parity byte causes recall failure (via parity mismatch).

## Operational Guidance
- Use `WAVE_ECC_REDUNDANCY` env var to set redundancy level (default 3).
- Seed base is derived from content hash; do not override arbitrarily.
- `HLOG_GPU_ONLY=1` forces the GPU-backed HGMC2 path in tests/environments.
- Production runtime boundary: CPU-native modules are disabled; Wave ECC runs via the native GPU binding surface.

## Acceptance Criteria
- Correctness:
  - No-error multi-chunk payload: exact roundtrip
  - Injected byte corruptions: corrected under configured redundancy level R
  - Parity tamper: recall fails with parity mismatch after correction
- API/Contract:
  - `wave_ecc_decode` returns `(corrected_bytes, errors_detected:int)`
  - `verify_and_correct_rs` uses Wave ECC and enforces parity recheck

## Future Work
- Formalize similarity thresholds and adaptive view selection policies
- Extend HGMC3 validation with Wave ECC parity streams
- Micro-benchmarks for parity overhead and throughput vs. R and dimension
