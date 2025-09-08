# Performance and Complexity Analysis (7‑Layer Holographic Memory)

Version: 1.0

## 1. Computational Complexity

- Store (sketch path):
  - Build signal + FFT/DWT: O(n log n) (n = target dimension for sketch); for small fixed n, effectively O(1).
  - Top‑K selection: O(n) with single pass or O(n log K) with heap (n ≫ K).
  - Serialization: O(K).
  - Overall per file: O(n log n + n) ≈ O(n log n), practically dominated by O(K) with small n.

- Retrieve (query scoring):
  - Resonance against field ψ_k: O(D_k) per used layer; 1–2 layers typical.
  - ANN index over sketches (optional): sublinear average‑time per query.

- Optimization:
  - Dimension allocation D_k*: closed‑form; O(L) time.
  - Threshold τ*: closed‑form via cost equality or O(log U) binary search over size range.

## 2. Space Complexity

- Field memory: O(M) where M = Σ_k D_k.
- On‑disk per file:
  - Micro format: O(1) ≈ 16 B.
  - Sparse v4: O(K) coefficients (3–4 B each) + small header.

## 3. Small‑File Expansion Elimination

- Policy: choose micro when s ≤ τ* with τ* = min{s: c_v4(s) ≤ min(c_micro, c_microK8)}.
- Guarantee: stored size ≤ min(c_micro, c_microK8) for small files, by selection rule.

## 4. Convergence and Optimality

- D_k* allocation is closed‑form from Lagrangian optimality; no iterative convergence required.
- τ* selection is optimal for the piecewise‑constant cost model; binary search finds the smallest s satisfying the inequality.

## 5. Scaling Laws and SNR

- Per layer, SNR_k ≈ √(D_k / N_k); capacity scales linearly with D_k for fixed target SNR.
- Layering reduces N_k per layer, improving SNR and retrieval robustness.

## 6. Practical Constants

- Typical v4: K = 32, coeff ≈ 3–4 B → 100–150 B per layer, two layers ≈ 200–300 B + header.
- Micro: ≈ 16 B header; optional micro+coeffs K=8 fits ≈ 24–40 B total.

