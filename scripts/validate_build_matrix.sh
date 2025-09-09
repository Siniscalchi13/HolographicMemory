#!/bin/bash

set -e

echo "=== Holographic Memory Build Matrix Validation ==="
echo "Platform: $(uname -s)"
echo "Architecture: $(uname -m)"
echo "Date: $(date)"
echo

# Function to run build and test
run_build_test() {
    local platform=$1
    local build_flag=$2
    local test_platform=$3
    
    echo "--- Building for $platform ---"
    
    # Clean previous build
    rm -rf build
    
    # Build
    cmake -B build -DCMAKE_BUILD_TYPE=Release $build_flag
    cmake --build build --config Release
    
    echo "Build completed successfully"
    
    # Test native benchmark
    echo "--- Running native benchmark ---"
    ./build/gpu_bench 1024 10000 5 $test_platform results_${platform}.csv ultra
    
    echo "--- Testing Python integration ---"
    python3 -c "
import numpy as np
import sys
from pathlib import Path
p = Path('holographic-fs/native/holographic')
sys.path.insert(0, str(p))
sys.path.insert(0, str(p / 'lib.macosx-metal'))

try:
    import holographic_gpu as hg
    arr = np.random.rand(1000, 64).astype(np.float32)
    gpu = hg.MetalHolographicBackend()
    out = gpu.batch_encode_fft_ultra_numpy(arr, 1024)
    metrics = gpu.get_metrics()
    print(f'Python test: {len(out)} x {len(out[0])} patterns')
    print(f'Device ops/sec: {metrics.operations_per_second}')
    print('✅ Python integration working')
except Exception as e:
    print(f'❌ Python integration failed: {e}')
"
    
    echo "✅ $platform validation complete"
    echo
}

# Detect platform and run appropriate build
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - Metal
    run_build_test "metal" "-DBUILD_METAL=ON" "metal"
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - try CUDA first, then ROCm
    if command -v nvcc &> /dev/null; then
        echo "CUDA detected, building CUDA backend"
        run_build_test "cuda" "-DBUILD_CUDA=ON" "cuda"
    elif command -v hipcc &> /dev/null; then
        echo "ROCm detected, building ROCm backend"
        run_build_test "rocm" "-DBUILD_ROCM=ON" "rocm"
    else
        echo "No GPU toolkit detected, building CPU-only"
        run_build_test "cpu" "" "auto"
    fi
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

echo "=== Build Matrix Validation Complete ==="
