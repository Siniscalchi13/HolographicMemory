H4M1/H4K8 Retrieval Mathematics
================================

Scope
- Defines the binary formats and inverse synthesis math used for retrieval.
- Focuses on H4K8 and HWP4V001 (sparse spectral) reconstruction; clarifies H4M1.

Formats
- H4M1: Micro header only. Contains doc_id8, orig_size, optional dimension/layer_count. No spectral coefficients; not recoverable by design.
- H4K8: Micro+coeffs variant. Stores up to K<=8 sparse bins with per‑layer scale. Recoverable via sparse IDFT.
- HWP4V001: Full v4. Multiple named layers, each with K-sparse indices, quantized amplitudes (u8) and phases (10-bit in u16), plus per-layer amp scale.

Inverse Synthesis (H4K8/v4)
- Accumulator for complex spectrum S[k] with real/imag components.
- For each layer entry i: S[idx[i]] += amp_scale * (amps_q[i]/255) * exp(j * phase_q[i]), where phase_q[i]∈[0,2π) mapped from 10-bit.
- Inverse DFT: s[n] = (1/N) ∑_k S[k]·exp(j·2πkn/N), N=dimension. Use unitary scaling for stability.
- Normalize s to [0,1] by min/max; map to bytes via round(s*255).
- Resample via linear interpolation when N ≠ original_size.

Complexity
- Synthesis: O(N·K) when accumulating sparse bins, with O(N·N) fallback in naive IDFT (reference implementation may loop all k for clarity).
- With FFT-based synthesis of sparse deltas, can be reduced further; current decoder prioritizes clarity and robustness.

Stability & Error Bounds
- Quantization introduces ≤1/255 amplitude error; phase quantization ≤ π/1023 radians.
- Errors accumulate linearly with K; empirical PSNR acceptable for small K (≤32) for semantic sketching.

Notes
- H4M1 requires an auxiliary exact-recall backend (3D) to recover bytes. Without it, retrieval is not possible.

