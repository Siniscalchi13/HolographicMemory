# HolographicMemory — Proofs Index (current)

Legend: [I] Informal (Markdown), [F] Formal (Coq), [T] Tested (unit/property)

Scope note: This index reflects proofs present in this repository. Some formal items are axiomatic. Unrelated summaries (e.g., AIUCP “42 proofs”) are out of scope for HolographicMemory.

## Holographic Memory (HMC)

- HMC-FFT: FFT norm preservation [I][F]
  - Statement: With unitary normalization, DFT preserves ℓ² norm.
  - Formal: `documentation/proofs/coq/HMC_FFT.v` (fft_preserves_snorm; relies on axiom FFT_is_unitary)

- HMC-Corr: Correlation bounded by one [I][F]
  - Statement: For normalized vectors, |⟨x,y⟩| ≤ 1.
  - Formal: `documentation/proofs/coq/HMC_Correlation.v` (correlation_bounded_by_one; uses axiom abs_ip_cs)

- HMC-ExactRecall: Exact recall for block writes [F]
  - Statement: Under fit precondition, read-after-write returns the written block exactly.
  - Formal: `documentation/proofs/coq/HolographicExactRecall.v` (exact_recall; fully proven)

## 7‑Layer and Control (selected)

- HM_7Layer: Layer properties [F]
  - File: `documentation/proofs/coq/HM_7Layer.v` (status: present; coverage TBD)

- Budget/Policy/Ordering [F]
  - Files: `ACLTtermination.v`, `BudgetMonotonicity.v`, `DeterministicReplay.v`, `EventOrder.v`, `MSC_Selection.v`, `PolicyNonEscalation.v`, `TokenBucket.v`, `WALDurability.v`
  - Note: Outside core holographic wave math; status varies by file.

## Quantum (ancillary examples)

- QMC-Core/POVM/CHSH [F]
  - Files: `QMC_Core.v`, `QMC_POVM.v`, `QMC_CHSH.v`
  - Note: Demonstrative; not required for computational holography pipeline.

---

Status and CI
- Coq modules compile status and axiom usage to be reported via CI.
- Axioms to be documented per file and minimized over time (replace with constructive proofs where feasible).
