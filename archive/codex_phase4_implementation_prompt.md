# Codex Implementation Prompt: Phase 4 - 7-Layer Routing for Bytes

## Project Context
You are implementing **Phase 4** of the HolographicMemory build plan. This is a GPU-accelerated computational holographic memory system using FFT-based wave interference, seeded codebook correlation, and Wave ECC for error correction.

**Current Status**: Phase 3.5 Complete (83% overall progress)
- ✅ Wave ECC fully implemented and production-ready
- ✅ All 5 Wave ECC tests passing
- ✅ Production tools (benchmarking, stress testing, monitoring) implemented
- ✅ GPU acceleration with 14 Metal shader pipelines working

**Your Task**: Implement Phase 4 - 7-Layer Routing for Bytes

## Phase 4 Requirements

### Core Implementation Tasks

#### 1. Layer Routing Implementation
**Objective**: Route chunk groups to appropriate 7-layer subspaces based on content policy

**7-Layer Architecture**:
- **Layer 1 (Identity)**: User metadata, session information
- **Layer 2 (Knowledge)**: Factual data, structured content  
- **Layer 3 (Experience)**: Procedural knowledge, workflows
- **Layer 4 (Preference)**: User choices, personalization data
- **Layer 5 (Context)**: Temporal and spatial relationships
- **Layer 6 (Wisdom)**: High-level insights, patterns
- **Layer 7 (Vault)**: Encrypted, sensitive data

**Implementation Requirements**:
- Route chunk groups to layers by policy (content analysis, metadata, user classification)
- Implement layer assignment logic in `services/holographic-memory/core/holographicfs/memory.py`
- Update GPU binding to support per-layer operations in `services/holographic-memory/core/native/holographic/gpu_binding.cpp`

#### 2. Per-Layer α Scaling for SNR Targets
**Objective**: Apply layer-specific scaling to achieve target SNR values

**Mathematical Requirements**:
- Implement `initialize_7layer_decomposition()` function
- Implement `update_layer_snrs()` function  
- Implement `enforce_capacity_theorem()` function
- Apply SNR formula: `SNR_k ≈ sqrt(D_k / N_k)`
- Enforce capacity bounds: `D_k ≥ S_k² N_k`

**Implementation Files**:
- `services/holographic-memory/core/native/holographic/holographic_memory.cpp`
- `services/holographic-memory/core/native/holographic/gpu_binding.cpp`

#### 3. Per-Layer Telemetry
**Objective**: Expose per-layer metrics via `stats()` API

**Required Metrics**:
- Per-layer SNR values
- Per-layer BER (Bit Error Rate)
- Per-layer byte counts
- Per-layer capacity utilization
- Cross-layer interference measurements

**Implementation**:
- Extend `HoloFS.stats()` to return `layer_metrics` dictionary
- Add per-layer telemetry collection in GPU binding
- Update monitoring dashboard to display layer metrics

#### 4. Comprehensive Testing
**Objective**: Validate routing distribution, capacity enforcement, and e2e recall

**Test Requirements**:
- Routing distribution tests (verify chunks go to correct layers)
- Capacity enforcement tests (verify `D_k ≥ S_k² N_k` bounds)
- End-to-end recall tests across all layers
- Cross-layer interference tests
- SNR degradation tests under load

**Test Files**:
- `services/holographic-memory/core/tests/test_7layer_routing.py`
- `services/holographic-memory/core/tests/test_layer_capacity.py`
- `services/holographic-memory/core/tests/test_layer_telemetry.py`

## Implementation Guidance

### File Structure & Key Files

#### Core Implementation Files
```
services/holographic-memory/core/
├── holographicfs/
│   └── memory.py                    # Main Python API - implement layer routing logic
├── native/holographic/
│   ├── holographic_memory.cpp       # 7-layer core implementation
│   ├── gpu_binding.cpp             # GPU bindings - add layer functions
│   └── metal/
│       └── holographic_memory.metal # Metal shaders - add layer operations
└── tests/
    ├── test_7layer_routing.py       # NEW: Layer routing tests
    ├── test_layer_capacity.py       # NEW: Capacity enforcement tests
    └── test_layer_telemetry.py      # NEW: Telemetry tests
```

#### Mathematical Functions to Implement
```cpp
// In holographic_memory.cpp
void initialize_7layer_decomposition(int total_dimensions);
void update_layer_snrs(const std::vector<int>& layer_loads);
bool enforce_capacity_theorem(const std::vector<double>& snr_targets);

// In gpu_binding.cpp  
pybind11::dict get_layer_metrics();
void route_to_layer(const std::vector<uint8_t>& data, int layer_id);
std::vector<uint8_t> retrieve_from_layer(const std::vector<uint8_t>& query, int layer_id);
```

#### Python API Extensions
```python
# In memory.py
class HoloFS:
    def store_with_layer_routing(self, data: bytes, layer_policy: str = "auto") -> str:
        """Store data with automatic or manual layer routing"""
        
    def get_layer_stats(self) -> Dict[str, Any]:
        """Get per-layer statistics and metrics"""
        
    def rebalance_layers(self) -> Dict[str, Any]:
        """Rebalance layer dimensions based on current loads"""
```

### Implementation Steps

#### Step 1: Layer Routing Logic
1. **Content Analysis**: Implement policy-based layer assignment
   - Auto-detect content type (text, binary, structured data)
   - Apply user-specified layer preferences
   - Handle Vault layer encryption requirements

2. **GPU Integration**: Update Metal shaders for layer operations
   - Add layer-specific codebook generation
   - Implement per-layer superposition
   - Add layer isolation in correlation operations

#### Step 2: Mathematical Functions
1. **Dimension Allocation**: Implement Theorem 1.1 optimal allocation
   ```cpp
   D_k* = M · (α_k² / N_k) / Σ_j (α_j² / N_j)
   ```

2. **SNR Management**: Implement SNR calculation and enforcement
   ```cpp
   SNR_k ≈ sqrt(D_k / N_k)
   D_k ≥ S_k² N_k  // Capacity theorem enforcement
   ```

3. **Capacity Monitoring**: Track and enforce layer capacity bounds

#### Step 3: Telemetry & Monitoring
1. **Metrics Collection**: Add per-layer statistics gathering
2. **API Extension**: Extend `stats()` to include layer metrics
3. **Dashboard Updates**: Update monitoring dashboard for layer visualization

#### Step 4: Testing & Validation
1. **Unit Tests**: Test individual layer operations
2. **Integration Tests**: Test cross-layer interactions
3. **Performance Tests**: Validate SNR and capacity under load
4. **End-to-End Tests**: Verify complete store/retrieve cycle

### Success Criteria

#### Functional Requirements
- ✅ Chunks are correctly routed to appropriate layers based on policy
- ✅ Per-layer SNR targets are achieved and maintained
- ✅ Capacity theorem bounds are enforced (`D_k ≥ S_k² N_k`)
- ✅ Layer telemetry is accurate and accessible via `stats()`
- ✅ Cross-layer interference is minimized
- ✅ End-to-end recall works across all layers

#### Performance Requirements
- ✅ Layer routing adds <5ms latency to store operations
- ✅ SNR calculations complete in <1ms
- ✅ Telemetry collection adds <1ms overhead
- ✅ Memory usage scales linearly with layer count

#### Quality Requirements
- ✅ All tests pass consistently
- ✅ No CPU compute in heavy path (GPU-only enforcement)
- ✅ Mathematical formulas implemented correctly
- ✅ Documentation updated with layer routing procedures

### Validation Commands

After implementation, validate with:
```bash
# Build the system
cd /Users/smarthaus/Projects/GitHub/HolographicMemory
cmake --build build_holo -j 4

# Run layer routing tests
PYTHONPATH=build_holo venv313/bin/pytest -q -o addopts='' services/holographic-memory/core/tests/test_7layer_routing.py

# Run capacity enforcement tests  
PYTHONPATH=build_holo venv313/bin/pytest -q -o addopts='' services/holographic-memory/core/tests/test_layer_capacity.py

# Run telemetry tests
PYTHONPATH=build_holo venv313/bin/pytest -q -o addopts='' services/holographic-memory/core/tests/test_layer_telemetry.py

# Validate Wave ECC still works
PYTHONPATH=build_holo venv313/bin/python test_wave_ecc.py
```

### Critical Constraints

#### Architecture Constraints
- **GPU-Only Heavy Path**: No CPU compute in store/recall operations
- **SOA Boundaries**: Maintain service boundaries and module ownership
- **Python 3.13**: All code must work with Python 3.13 (cp313)
- **Metal Primary**: Metal shaders are primary GPU backend

#### Mathematical Constraints
- **Orthogonal Layers**: `H = ⊕_{k=1}^7 H_k` with orthogonal projectors
- **SNR Bounds**: Maintain `SNR_k ≈ sqrt(D_k / N_k)` relationships
- **Capacity Theorem**: Enforce `D_k ≥ S_k² N_k` for all layers
- **Wave ECC Integration**: Maintain Wave ECC error correction

#### Quality Constraints
- **No Breaking Changes**: Maintain existing API compatibility
- **Test Coverage**: All new functionality must have tests
- **Documentation**: Update build plan and implementation docs
- **Performance**: Maintain or improve current performance levels

## Expected Deliverables

1. **Layer Routing Implementation**: Complete 7-layer routing system
2. **Mathematical Functions**: SNR management and capacity enforcement
3. **Telemetry System**: Per-layer metrics and monitoring
4. **Test Suite**: Comprehensive testing for all layer operations
5. **Documentation Updates**: Updated build plan and implementation guides
6. **Validation Results**: All tests passing, performance metrics within targets

## Success Metrics

- **Phase 4 Completion**: 4/4 components implemented and tested
- **Overall Progress**: 83% → 90% complete
- **Next Phase Ready**: Phase 5 (CUDA/ROCm parity) can begin
- **Production Ready**: Layer routing system ready for production use

---

**Remember**: This is a mathematically rigorous system. All implementations must follow the established mathematical formulas and maintain the GPU-only heavy path. Focus on correctness, performance, and maintainability.
