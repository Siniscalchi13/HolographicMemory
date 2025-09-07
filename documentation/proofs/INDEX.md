# TAI Calculus Suite — Proofs Index (v1)

This index is the authoritative map of mathematical results to their formalizations and code/test anchors. Each item has a stable ID, a precise statement, status, and references.

Legend: [I] Informal (paper/Markdown), [F] Formal (Coq), [T] Tested (property/unit tests)

## Quantum Measurement Calculus (QMC)

- QMC-1: Unitary norm preservation [I][F][T]
  - Statement: For any unitary U and state ψ in C^n, ||Uψ||₂ = ||ψ||₂.
  - Code: `services/aiucp/quantum_core/mathematical_implementation.py:148`
  - Tests: `tests/test_quantum_math_core.py:17`
  - Formal: `proofs/coq/QMC_Core.v` (unitary_preserves_snorm)

- QMC-2: Projective measurement completeness [I][F][T]
  - Statement: For Hermitian observable with orthonormal eigenvectors v_i, probabilities p_i = |⟨v_i, ψ⟩|² sum to 1.
  - Code: `services/aiucp/quantum_core/mathematical_implementation.py:82`
  - Tests: `tests/test_quantum_math_core.py:31`
  - Formal: `proofs/coq/QMC_Core.v` (projective_measurement_completeness, via Parseval axiom)

- QMC-3: POVM validity [I][F][T]
  - Statement: For a POVM {E_k} with ΣE_k=I and E_k ≥ 0, Σ_k ⟨ψ|E_k|ψ⟩ = 1.
  - Code: `services/aiucp/quantum_core/mathematical_implementation.py:141`
  - Tests: `tests/test_quantum_math_core.py:78`
  - Formal: `proofs/coq/QMC_POVM.v` (povm_probabilities_sum_to_one)

- QMC-4: Tsirelson bound (CHSH) [I][F][T]
  - Statement: The CHSH parameter satisfies S ≤ 2√2 with equality at optimal angles.
  - Code: `services/aiucp/quantum_core/mathematical_implementation.py:222`
  - Tests: `tests/test_quantum_math_core.py:58`
  - Formal: `proofs/coq/QMC_CHSH.v` (chsh_upper_bound, chsh_reaches_bound)

## Holographic Memory Calculus (HMC)

- HMC-1: Exact recall for disjoint placement [I][F][T]
  - Statement: Storing bytes b into a disjoint segment [s, s+|b|) and retrieving that segment yields b exactly.
  - Formal: `proofs/coq/HolographicExactRecall.v`
  - Code: `macos-holographic-demo/PythonBackend/holographic_server.py:244`
  - Tests: `tests/unit/test_holographic_memory.py:23`

- HMC-2: FFT unitarity and norm preservation [I][F][T]
  - Statement: The discrete Fourier transform (with orthonormal normalization) preserves the ℓ² norm.
  - Code: `services/aiucp/holographic_memory/wave_functions.py`
  - Tests: `tests/test_quantum_math_core.py:44`
  - Formal: `proofs/coq/HMC_FFT.v` (fft_preserves_snorm)

- HMC-3: Correlation search normalization [I][F][T]
  - Statement: With normalized vectors, similarity |⟨q, d⟩| ∈ [0,1].
  - Code: `macos-holographic-demo/PythonBackend/holographic_server.py:392`
  - Tests: (to add explicit bound test)
  - Formal: `proofs/coq/HMC_Correlation.v` (correlation_bounded_by_one)

## Model Selection Calculus (MSC)

- MSC-1: Score monotonicity in competence [I][F][T]
  - Statement: Holding cost and penalties fixed, J increases monotonically with competence.
  - Code: `services/aiucp/verbum-field-engine/selection.py:94`
  - Tests: `services/aiucp/verbum-field-engine/tests/test_selection.py:14`
  - Formal: `proofs/coq/MSC_Selection.v` (J_monotone_in_comp)

- MSC-2: Capacity overflow penalty [I][F]
  - Statement: If τ > L_m then ∂J/∂τ ≤ −η < 0 (penalizes over-context).
  - Code: `services/aiucp/verbum-field-engine/selection.py:137`
  - Tests: (to add)
  - Formal: `proofs/coq/MSC_Selection.v` (J_decreases_with_tau_above_Lm)

## AIUCP Orchestration Calculus (AIOC)

- AIOC-1: Replay determinism under fixed catalog [F]
  - Statement: Given deterministic adapters and fixed inputs, transcript is unique.
  - Formal: `proofs/coq/DeterministicReplay.v`
  - Code: Event-sourced orchestrator (design), tests: (to add)

## Policy & Privacy Calculus (PPC)

- PPC-1: Non-escalation of privileges via joins [F]
  - Statement: Effective capability from base and grants never exceeds cap when each grant ≤ cap.
  - Formal: `proofs/coq/PolicyNonEscalation.v`

---

Status rolls up into CI: Coq build + Python tests. New items will be appended with IDs and references as they are formalized.
