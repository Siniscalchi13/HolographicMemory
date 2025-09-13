Holographic Containers (HGMC2/HGMC3)
====================================

Overview
--------
The HolographicMemory store/recall path persists data as GPU-encoded holographic containers under the mount’s `.holofs/hlog/containers/` directory. These containers encapsulate multiplexed, seeded holographic representations rather than traditional byte streams.

Key Properties
--------------
- GPU-only encoding/decoding (Metal path implemented). CPU orchestrates I/O only.
- True multiplexing: chunks are encoded via FFT → seeded complex codebook → iFFT and superposed in time-domain; decode requires FFT(ψ) → conjugate codebook → iFFT.
- No traditional byte files; no HRR/base64 fallbacks.
- Honest accounting: `HoloFS.stats().holo_bytes` sums actual holographic bytes on disk.

Formats
-------
1) HGMC2 (raw ψ)
- Magic: `HGMC2\x00`
- Header:
  - `u32 dim`            – FFT dimension used
  - `u32 orig_size`      – original byte size
  - `u32 chunk_count`    – number of encoded chunks
  - `u32 sizes[chunk_count]`
  - `u32 seeds[chunk_count]`
- Payload:
  - `float32 psi[dim]`   – time-domain superposed field (imag set to 0 for container)

2) HGMC3 (compressed ψ)
- Magic: `HGMC3\x00`
- Header (same preface as HGMC2) plus:
  - `u32 encoded_size`
- Payload:
  - Entropy-coded sparse coefficients (GPU `gpu_sparse_encoding` + `gpu_entropy_coding`)
- Decode path:
  - GPU `gpu_entropy_decoding` → `gpu_sparse_decoding` → dense ψ → correlation decode.

Decode Integrity
----------------
Decode requires the original per-chunk seeds. Altering header seeds results in decode failure or corrupted output, demonstrating correlation-based retrieval (not contiguous byte reads).

Location
--------
- Containers: `<mount>/.holofs/hlog/containers/<doc_id>.hgc`
- Index/metadata: `<mount>/.holofs/index.json`, `.holofs/dimension_map.json`, `.holofs/hlog/container_map.json`

Roadmap
-------
- ECC: Add chunk-level ECC (RS/LDPC) to guarantee bit-perfect recall under FP variance and quantization.
- 7-layer routing: Route chunk groups per layer with target SNR and capacity enforcement; expose per-layer SNR/BER/bytes.
- CUDA/ROCm parity: Port kernels and bindings to non-Metal platforms.

