# + 🌊 holographic

## =============

# PURPOSE & OVERVIEW

## =============

**Purpose**: C++ holographic memory modules providing high‑throughput wave‑encoded storage and retrieval with three-tier wave persistence.
**Domain**: Wave‑based encoding, SIMD vectorization (NEON/Accelerate), FFT‑assisted operations for pattern superposition and recall, and automatic wave persistence.
**Status**: Working with wave persistence (multiple compiled `.so` artifacts present)
**Dependencies**: C++17, pybind11, FFTW (Linux), Apple Accelerate (macOS), Python 3.11/3.12/3.13
**Integration**: Used by orchestrator as `holographic_wave_simd` (and variants) for stats/memory functions with automatic wave persistence.
**Wave Persistence**: Three-tier storage system with individual patterns, current field, and timestamped snapshots.

## =============

# DIRECTORY STRUCTURE

## =============

```
holographic/
├── CMakeLists.txt                    # Size: 0.5 KB - Alt CMake config | Status: Optional
├── setup.py                          # Size: 3.5 KB - Build script (FFTW/Accelerate logic) | Status: Working
├── build_fast.py                     # Size: 1.9 KB - Convenience builder (fast variant) | Status: Working
├── build_optimized.py                # Size: 2.2 KB - Convenience builder (optimized variant) | Status: Working
├── build_wave.py                     # Size: 1.3 KB - Convenience builder (wave SIMD variant) | Status: Working
├── holographic_memory.cpp            # Size: 14.1 KB - Native baseline impl | Status: Working
├── holographic_memory_fast.cpp       # Size: 12.0 KB - Fast variant | Status: Working
├── holographic_memory_optimized.cpp  # Size: 17.3 KB - Optimized variant | Status: Working
├── holographic_wave_simd.cpp         # Size: 15.5 KB - SIMD wave encoder | Status: Working
├── holographic_cpp.cpython-311-darwin.so     # Size: ~270 KB - C++ module (Py 3.11) | Status: Working
├── holographic_cpp.cpython-312-darwin.so     # Size: ~300 KB - C++ module (Py 3.12) | Status: Working
├── holographic_fast.cpython-311-darwin.so    # Size: 272.6 KB - Fast module (Py 3.11) | Status: Working
├── holographic_wave_simd.cpython-311-darwin.so # Size: 269.3 KB - SIMD module (Py 3.11) | Status: Working
├── patterns/                                 # Holographic wave pattern files (.hwp)
│   ├── document.pdf.hwp                      # Individual file patterns
│   ├── image.png.hwp
│   └── ...
├── current.hwp                               # Complete memory field
└── snapshots/                                # Historical snapshots
    ├── snapshot_20250908T002359.hwp
    └── ...
└── build/                            # Purpose: Generated artifacts | Status: Working
    ├── lib.macosx-*/holographic_cpp.cpython-*.so
    └── temp.macosx-*/holographic_memory.o
```

## =============

# WAVE PERSISTENCE IMPLEMENTATION

## =============

### 🌊 Three-Tier Wave Storage System

All holographic memory backends now include automatic wave persistence with three storage tiers:

#### **1. Individual File Patterns**

- **Automatic Saving**: When text begins with `filename:<name>`, saves individual wave pattern
- **Location**: `{HLOG_DATA_DIR}/patterns/{original_filename}.hwp`
- **Format**: Binary `.hwp` files with rich metadata headers
- **Compression**: 90-98% size reduction compared to original files
- **Backend Support**: All backends (native, fast, optimized, wave SIMD)

#### **2. Current Complete Memory Field**

- **Automatic Updates**: Updated after each store operation
- **Location**: `{HLOG_DATA_DIR}/current.hwp`
- **Purpose**: Live holographic memory state with all files superposed
- **Size**: Depends on grid resolution (32³ = ~500KB, 64³ = ~4MB, 128³ = ~32MB)

#### **3. Historical Snapshots**

- **Automatic Creation**: Hourly snapshots + guaranteed shutdown snapshot
- **Location**: `{HLOG_DATA_DIR}/snapshots/snapshot_YYYYmmddTHHMMSS.hwp`
- **Purpose**: Timestamped backups for disaster recovery and analysis
- **Retention**: All snapshots kept (configurable)

### 🔧 Configuration

- **Environment Variable**: `HLOG_DATA_DIR` (if set, uses this directory)
- **Docker Default**: `/app/data/holographic_memory`
- **Local Default**: `data/holographic_memory`
- **Automatic Creation**: Directories created automatically

### 📁 Wave File Format

Each `.hwp` file contains:

- **Header**: Magic number + version + data type + dimension + timestamp + metadata length
- **Metadata**: JSON with file information, creation time, and semantic data
- **Payload**: Compressed wave data (complex128 or float32 amplitude+phase)

### 🚀 Performance

- **No API Changes**: Existing Python calls continue to work
- **Automatic Persistence**: Wave files saved automatically without user intervention
- **Efficient Storage**: Wave patterns are 90-98% smaller than original files
- **Fast Access**: Binary format optimized for speed

## =============

# SUBDIRECTORY DETAILS

## =============

### 📁 build/

**Purpose**: Platform‑specific build outputs from setuptools/CMake.
**Contents**: Shared libraries (.so) and intermediate objects.
**Integration**: Not imported directly; `.so` modules imported by Python.
**Status**: Working.

## =============

# FILE DETAILS

## =============

### 📄 setup.py (3.5 KB)

**Purpose**: Primary build script with FFTW/Accelerate detection.
**Function**: Configures `Pybind11Extension` with `-O3`, OpenMP (Linux), and platform flags.
**Dependencies**: pybind11, FFTW (Linux), Accelerate (macOS).
**Integration**: `python3 setup.py build_ext --inplace`.
**Status**: Working.

### 🧠 holographic_memory*.cpp (12–17 KB)

**Purpose**: Core holographic memory implementations (baseline/fast/optimized).
**Function**: Wave encoding, pattern storage, and retrieval; vectorization; optional FFT usage.
**Dependencies**: pybind11, platform SIMD, optional FFTW/Accelerate.
**Integration**: Imported as `holographic_cpp`/`holographic_fast`.
**Status**: Working.

### 🌊 holographic_wave_simd.cpp (15.5 KB)

**Purpose**: Fully SIMD vectorized wave memory (NEON/Accelerate) encoder.
**Function**: Vectorized sine approximations, phase/amplitude fields, and trace storage.
**Dependencies**: pybind11; NEON (ARM); Accelerate on macOS.
**Integration**: Orchestrator imports as `holographic_wave_simd` for stats.
**Status**: Working.

### 🧩 *.cpython-*.so (270–301 KB)

**Purpose**: Compiled C++ modules for supported CPython versions.
**Function**: Runtime acceleration and wave‑based memory.
**Status**: Working.

## =============

# ARCHITECTURE & INTEGRATION

## =============

**How This Directory Works**: Provides multiple C++ backends (baseline, fast, SIMD) to meet different performance envelopes and platform constraints.
**Data Flow**: Text → wave pattern encoding → superposition storage → similarity/phase‑aware recall.
**External Dependencies**: FFTW on Linux, Apple Accelerate on macOS, OpenMP where available.
**Performance Characteristics**: SIMD‑aligned buffers; vectorized wave generation; optional FFT acceleration; designed for O(1) recall on cached patterns.
**Integration Points**: `aicp-orchestrator/quantum_orchestrator` expects `holographic_wave_simd` for stats and memory count.

## =============

# DEVELOPMENT & DEPLOYMENT

## =============

**Build Process**:

- Local macOS (Accelerate): `python3 setup.py build_ext --inplace`
- Local Linux (FFTW/OpenMP): ensure `libfftw3-dev` installed, then `python3 setup.py build_ext --inplace`
- Variant helpers: `python3 build_fast.py`, `python3 build_wave.py`, `python3 build_optimized.py`
- Docker/Linux: `python3 build_linux.py` (used by multi-stage Dockerfile)

**Testing**:

- Import check: `python -c "import holographic_cpp as m; print(hasattr(m,'HolographicMemory'))"`
- Microbench: time store/query for short strings to verify vectorization path.

**Deployment**:

- Multi-stage Docker builds compile Linux C++ `.so` and install to `/opt/holo/holographic`
- Runtime uses `PYTHONPATH=/app:/opt/holo/holographic` so prebuilt modules load even with bind mounts

**Monitoring**:

- Export memory counts and module type (cpp/fast/wave) to orchestrator health.

**Troubleshooting**:

- Linker errors for FFTW: ensure brew/apt FFTW present and visible to `pkg-config`
- OpenMP errors (Linux): `libgomp1` must be present at runtime
- Apple Silicon: avoid AVX flags; rely on Accelerate/NEON; rebuild when Python minor version changes

## =============

# Cross-Platform Build Detection

## =============

- `build_detect.py` centralizes platform detection and flags
- Linux: `-fopenmp`, `-march=native` (+ `-mavx2 -mfma` on x86_64)
- macOS: Accelerate framework, optional OpenMP on Intel Macs

## =============

# Wave Persistence (Three‑Tier)

## =============

- Patterns: `{HLOG_DATA_DIR}/patterns/{original_filename}.hwp` saved during store (when first line begins with `filename:`)
- Current: `{HLOG_DATA_DIR}/current.hwp` updated after each store/batch
- Snapshots: `{HLOG_DATA_DIR}/snapshots/snapshot_YYYYmmddTHHMMSS.hwp` hourly and at shutdown

Internals:

- File header: magic + version + kind (1=complex128, 2=float32 amp+phase) + dimension + timestamp + metadata length
- Metadata: small JSON blob with type (pattern/current/snapshot), filename/session/seq, backend
- Payload: contiguous raw data (complex128 interleaved for native/optimized, float arrays for wave-simd)

Parent: ../ (service: holographic-memory)
Children: n/a (leaf)
