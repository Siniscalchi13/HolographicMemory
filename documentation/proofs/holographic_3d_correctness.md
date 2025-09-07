# Correctness Proofs for 3D Holographic Memory

## 1. Exact Reconstruction (Direct Disjoint Encoding)

**Claim.** For a document D = {b₀,…,b_{L−1}} stored at linear offsets n_i = (s+i) mod N³ with ψ[n_i] ← ψ[n_i] + (b_i/255.0), retrieval by b̂_i = round(255·Re{ψ[n_i]}) yields b̂_i = b_i.

**Assumptions.** (i) Disjointness: no other writes touch {n_i} during the lifetime of D. (ii) Numerical rounding error |ε| < 0.5/255 in Re{ψ[n_i]}.

**Proof.** For each i, ψ[n_i] = b_i/255.0 + ε. Thus 255·Re{ψ[n_i]} = b_i + 255·ε. Under (ii), |255·ε| < 0.5, so round(·) returns b_i. ∎

## 2. Orthogonal Superposition Projection

**Claim.** Let {φ_k} be an orthogonal set with ⟨φ_j,φ_k⟩ = 0 for j≠k, and let ψ = Σ_k a_k·φ_k. Then a_k = ⟨ψ,φ_k⟩ / ∥φ_k∥².

**Proof.** ⟨ψ,φ_k⟩ = ⟨Σ_j a_j·φ_j, φ_k⟩ = Σ_j a_j·⟨φ_j,φ_k⟩ = a_k·⟨φ_k,φ_k⟩ by orthogonality. Divide both sides by ∥φ_k∥². ∎

## 3. Seqlock Reader Correctness (Per Tile)

**Claim.** If a reader returns a snapshot where the tile’s seqlock sequence was even and unchanged before and after copying, then the copied data corresponds to a consistent state (not torn by concurrent writes).

**Proof.** Writer transitions seq: even → odd (start) → even (finish). If reader observes even s₀, copies tile, then observes even s₁ = s₀, no writer could have changed the tile contents between the two reads; otherwise seq would differ or be odd. ∎

## 4. Multi‑Process Consistency

**Claim.** With per‑tile seqlocks and an index guarded by a writer mutex or RCU epoch, concurrent processes performing reads while a single writer updates (tile‑local) will see either the old or the new state, never a torn state.

**Sketch.** Tile contents use seqlock correctness (3). The document index must publish entries atomically (insert fully visible), and retrieval reads index under a consistent epoch. Thus readers either find the old mapping or the new mapping after insertion. ∎

## 5. Energy and Precision Bounds

**Claim.** In direct disjoint encoding with complex128, round‑trip error ε satisfies |ε| < 10⁻¹⁵ under normal accumulation bounds.

**Sketch.** The write is a single assignment per voxel (no accumulation in disjoint mode), so the only error arises from representation of A = b/255.0 in binary64 and rounding on retrieval. ULP at this scale is ~1e‑16..1e‑15, well below 0.5/255 threshold for rounding. ∎
