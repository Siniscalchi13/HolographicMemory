# HMC Noise Bounds and Binding Invertibility — Formal Proofs

Authors: AIOS/TAI Research Team  
Institution: SmartHaus Group — TAI  
Version: 0.1 (Research Draft)

---

## 1. Setup and Assumptions

- Field F = ℂ^d with standard inner product ⟨·|·⟩ and norm ∥·∥₂.

- Encodings E: Σ → F such that ∥E(x)∥₂ = 1 and pairwise coherence bounded: for x≠y, |⟨E(x)|E(y)⟩| ≤ μ with μ ∈ [0, 1/√d].

- Hologram H = Σᵢ αᵢ E(xᵢ), with saliences αᵢ ≥ 0 and Σᵢ αᵢ² ≤ S (bounded energy).

- Noise ν with ∥ν∥₂ ≤ ε and random phases i.i.d. uniform on [0, 2π).

- Query q with ∥E(q)∥₂ = 1.

---

## 2. Noise False-Positive Bound (PO-HMC-003)

We analyze R(q, H+ν) = |⟨E(q)|H+ν⟩|² = |⟨E(q)|H⟩ + ⟨E(q)|ν⟩|².

- Signal term: s := ⟨E(q)|H⟩. If q ∉ {xᵢ} then |s| ≤ Σᵢ αᵢ |⟨E(q)|E(xᵢ)⟩| ≤ μ Σᵢ αᵢ ≤ μ √n √(Σᵢ αᵢ²) ≤ μ √n √S by Cauchy–Schwarz.

- Noise term: z := ⟨E(q)|ν⟩. Since ∥E(q)∥₂ = 1 and ∥ν
₂ ≤ ε, we have |z| ≤ ε. With random phase, E[z] = 0 and Var(z) ≤ ε²/d under isotropic ν.

Choose threshold τ = ε √(c log d / d) + μ √n √S, with c ≥ 2.

Then with probability ≥ 1 − d^{−c/2}, |z| ≤ ε √(c log d / d) (via a sub-Gaussian tail on isotropic projections). Thus

|s + z| ≤ |s| + |z| ≤ μ √n √S + ε √(c log d / d) = τ.

Therefore, for non-members q, P(R(q, H+ν) ≥ τ²) ≤ d^{−c/2}.

Constants: take μ ≤ 1/√d, S bounded constant, and n ≤ poly(d) for practical regimes → τ = O(ε √(log d / d)) + O(√(S n / d)).

---

## 3. Binding Invertibility Conditions (PO-HMC-004)

Define binding via circular convolution in Fourier domain:

- bind(a,b) := F⁻¹(F(a) ⊙ F(b))

- unbind(x,a) := F⁻¹(F(x) ⊘ F(a))

Sufficient conditions for unbind(bind(a,b), a) = b:

1) Non-vanishing spectrum: ∀k, F(a)[k] ≠ 0.

2) Conditioning bound: κ(a) := max_k |F(a)[k]| / min_k |F(a)[k]| ≤ K (finite), ensuring numerical stability.

3) Orthogonality of leakage: Numerical errors δ follow ∥δ∥₂ ≤ η with η bounded by machine precision times κ(a).

Proof:

F(unbind(bind(a,b), a)) = (F(bind(a,b)) ⊘ F(a)) = ((F(a) ⊙ F(b)) ⊘ F(a)) = F(b) under (1). Applying F⁻¹ yields b. Stability: with finite κ(a), relative error on division is ≤ O(κ(a)·ulp), hence ∥Δb∥₂ ≤ η' with η' = O(κ(a)) times floating-point unit; thus practical invertibility within tolerance.

---

## 4. Discussion

- The threshold τ scales as ε √(log d / d) plus cross-talk term μ √n √S. With structured E achieving μ = O(1/√d), the cross-talk term is O(√(S n / d)). For fixed S and n ≪ d, τ shrinks as d grows.

- Binding invertibility hinges on choosing spectra for a with bounded away-from-zero magnitudes; random unit-modulus spectra (abs(F(a)[k]) = 1) satisfy κ(a)=1.

---

End of document.
