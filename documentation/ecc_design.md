# ECC Design: RS(255,223) for HolographicMemory

This document describes the Reed–Solomon ECC used in HolographicMemory containers, its parameters, code paths, and validation. It covers encoding, decoding/correction, integration points, and tests that verify behavior on-device.

## Parameters and Bounds
- Code: RS(255,223) over GF(2^8)
- Symbols per block: n = 255, data k = 223, parity r = 32
- Correction capability: t = r/2 = 16 symbol errors per block
- Field primitive polynomial: 0x11d (x^8 + x^4 + x^3 + x^2 + 1)

## Implementation Overview
- Encode (host-side, GPU module surface):
  - Path: `services/holographic-memory/core/native/holographic/gpu_binding.cpp`
  - API: `gpu_rs_encode(payload: bytes, k: int = 223, r: int = 32) -> bytes`
  - Method: LFSR-style parity generation using generator polynomial G(x) with roots α^1..α^r.
- Decode/Correct (host-side, GPU module surface):
  - Path: `services/holographic-memory/core/native/holographic/gpu_binding.cpp`
  - API: `gpu_rs_decode(payload: bytes, parity: bytes, k: int = 223, r: int = 32) -> tuple[bytes, list[int]]`
  - Method: Per-block processing (k data + r parity):
    1) Compute syndromes S_j
    2) Berlekamp–Massey for error locator Λ(x)
    3) Chien search to locate errors
    4) Forney algorithm to compute magnitudes and correct
    5) Return corrected payload bytes and per-block correction counts

### Per-block Isolation and Tail Handling
- Blocks are computed as `blocks = ceil(len(payload) / k)`.
- Tail block (len < k) is zero-padded during decode, ensuring stable syndrome computation.
- Corrections are strictly limited to the live data region `[0, chunk_len)`, never the zero-padded tail.
- Parity validation is performed after each candidate mapping; only corrections that re-validate parity are accepted.
- Counts length equals `blocks` and each entry reflects the number of corrections applied to that block.
- Changes for safety were added here: `gpu_binding.cpp` (per-block copy, zero-padding, tail-aware mappings, parity-gated fallbacks). See around the decode lambda for details.

## Integration Points
- Container store path (HGMC2):
  - Path: `services/holographic-memory/core/holographicfs/memory.py`
  - Function: `Memory.store_bytes(...)`
  - Behavior: Computes per-chunk parity via `gpu_rs_encode` and writes it to container after sizes/seeds.
- Container recall path (HGMC2):
  - Path: `services/holographic-memory/core/holographicfs/memory.py`
  - Function: `Memory.retrieve_bytes(...)`
  - Behavior: Uses GPU correlation to reconstruct chunk bytes, then calls `verify_and_correct_rs(...)` (which invokes `gpu_rs_decode`) to enforce t-bound and verify parity recheck.
- Enforcement helper:
  - Path: `services/holographic-memory/core/holographicfs/memory.py`
  - Function: `verify_and_correct_rs(payload, parity, k=223, r=32)`
  - Behavior: Raises on >t cases or parity mismatch after correction.

## Tests and Validation
- Bounds tests: `services/holographic-memory/core/tests/test_ecc_bounds.py`
  - ≤t correction succeeds; >t fails deterministically (per single block).
- Extended tests: `services/holographic-memory/core/tests/test_ecc_extended.py`
  - No-error multi-block roundtrip succeeds; counts shape is validated
  - t+1 in a block fails (not restored or parity mismatch)
  - Parity tamper raises via `verify_and_correct_rs`
  - XFail: disjoint multi-block ≤t corrections and tail-block ≤t corrections (tracked for hardening)
- E2E container parity corruption: `services/holographic-memory/core/tests/test_hgmc_e2e.py::test_hgmc2_parity_mismatch_fails_e2e`
  - Corrupting a parity byte causes recall failure.

## Operational Guidance
- Use `HLOG_GPU_ONLY=1` in environments requiring strictly GPU-backed recall paths.
- Production runtime boundary: CPU-native modules are excluded; ECC is performed host-side within the GPU module surface while holographic encode/decode remains GPU-only.
- Dev builds include CPU-native modules for reference, tooling, and visualization only.

## Acceptance Criteria
- Correctness:
  - No-error multi-block payload: exact roundtrip; counts per block are bounded and ideally zero
  - ≤t errors in one block: exact roundtrip; that block’s count ≤ t; others near zero
  - >t in any block: deterministic failure (not restored or parity mismatch)
- API/Contract:
  - `gpu_rs_decode` returns a counts list of length `ceil(len(payload)/k)`
  - `verify_and_correct_rs` raises on parity mismatch after correction

## Future Work
- Stability hardening for multi-block ≤t and partial tail block error patterns
- Optional GPU parity kernel `rs_encode_blocks` for profiling comparisons
- Add micro-benchmarks (`tools/bench_ecc.py`) for encode/decode throughput across payload sizes
