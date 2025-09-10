GPU Math API Parity Plan
========================

Goal
----
Achieve GPU-first operation with full math API parity so that all mathematical features are available directly via the GPU backend (Metal/CUDA/ROCm), with CPU as fallback only.

Scope (API Parity Targets)
--------------------------
- initialize_7layer_decomposition(total_budget)
- optimize_layer_dimensions()
- get_layer_stats()  # includes dimensions, loads, target/current SNR, capacity usage, initialized flag
- calculate_layer_snr(layer_idx)
- update_layer_snrs()
- enforce_capacity_theorem()
- validate_wave_properties()  # normalization, orthogonality, capacity compliance
- analyze_interference_patterns()  # visibility, phase coherence, bell_test_passed
- validate_bell_inequality()

Approach
--------
1) Control-Plane Data Model (GPU side)
   - Maintain a small 7-layer state structure in the GPU wrapper:
     - dimension D_k, load estimate N_k, importance α_k, target SNR S_k, current SNR
   - Expose getters/setters, serialize into Python dicts for telemetry.

2) Algorithms
   - Dimension allocation (Theorem 1.1): replicate CPU logic on host; apply to GPU-side state.
   - SNR calculation: compute on host from (D_k, N_k) initially; later, augment with device-side statistics.
   - Capacity enforcement: adjust D_k floors; maintain global budget.
   - Wave validation: compute field_norm from device or host-accessible aggregates; orthogonality via sampled dot-products (device kernels later).
   - Interference metrics: visibility/phase coherence via sampled statistics; bell measurement using correlations over sampled indices.

3) Implementation Phases
   - P0: Host-side parity (compute metrics on host using GPU-produced vectors; preserve API shape).
   - P1: Device-side kernels for sampled correlations (performance and accuracy).
   - P2: Telemetry integration and stability hardening (error bounds, sampling strategies).

Acceptance Criteria
-------------------
- All parity APIs callable from `holographic_gpu` with correct return types/keys.
- Formula test suite passes; 7-layer stats consistent between CPU and GPU for same inputs.
- `/capabilities` reflects GPU parity and indicates 7-layer initialized when configured via GPU.
- Performance: parity methods add negligible overhead relative to core GPU encode.

Risks & Mitigations
-------------------
- Numerical drift vs CPU: validate with tolerance thresholds; use double on host summaries.
- Platform variance (Metal/CUDA/ROCm): keep P0 host-side; incrementally add per-platform kernels.

Dependencies
------------
- Existing GPU wrapper `holographic_gpu` and `GPUBackend.hpp` interfaces.
- Math validation tests: `tests/mathematics/test_math_formulas.py`.

Out of Scope (for this milestone)
---------------------------------
- Full Coq proof completion (tracked separately).
- Full device-only math validation without host fallback.

