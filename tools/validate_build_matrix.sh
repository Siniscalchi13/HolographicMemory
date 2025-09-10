#!/bin/bash

# Cross-platform GPU build validation script
# Tests Metal (macOS), CUDA (Linux), and ROCm (Linux) builds

set -e

echo "🔧 Cross-Platform GPU Build Validation"
echo "======================================"

# Detect OS and available GPU toolkits
OS=$(uname -s)
echo "OS: $OS"

# Function to test Metal build (macOS)
test_metal() {
    echo "🍎 Testing Metal build (macOS)..."
    
    cd holographic-fs/native/holographic
    
    # Build Metal
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_METAL=ON
    cmake --build build --config Release
    
    # Test GPU availability
    python3 -c "
import sys
sys.path.insert(0, 'build')
import holographic_gpu as hg
gpu = hg.MetalHolographicBackend()
print(f'Metal GPU available: {gpu.available()}')
if gpu.available():
    print('✅ Metal build successful')
else:
    print('❌ Metal build failed')
    exit(1)
"
    
    # Run native benchmark
    echo "Running Metal benchmark..."
    ./build/gpu_bench 1024 1000 3 metal results_metal.csv ultra
    
    # Check results
    if [ -f results_metal.csv ]; then
        echo "✅ Metal benchmark completed"
        echo "Results:"
        cat results_metal.csv
    else
        echo "❌ Metal benchmark failed"
        exit 1
    fi
    
    cd ../..
}

# Function to test CUDA build (Linux)
test_cuda() {
    echo "🚀 Testing CUDA build (Linux)..."
    
    # Check if CUDA is available
    if ! command -v nvcc &> /dev/null; then
        echo "⚠️  CUDA not available, skipping CUDA test"
        return 0
    fi
    
    cd holographic-fs/native/holographic
    
    # Build CUDA
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CUDA=ON
    cmake --build build --config Release
    
    # Run native benchmark
    echo "Running CUDA benchmark..."
    ./build/gpu_bench 1024 1000 3 cuda results_cuda.csv ultra
    
    # Check results
    if [ -f results_cuda.csv ]; then
        echo "✅ CUDA benchmark completed"
        echo "Results:"
        cat results_cuda.csv
    else
        echo "❌ CUDA benchmark failed"
        exit 1
    fi
    
    cd ../..
}

# Function to test ROCm build (Linux)
test_rocm() {
    echo "🔥 Testing ROCm build (Linux)..."
    
    # Check if ROCm is available
    if ! command -v hipcc &> /dev/null; then
        echo "⚠️  ROCm not available, skipping ROCm test"
        return 0
    fi
    
    cd holographic-fs/native/holographic
    
    # Build ROCm
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_ROCM=ON
    cmake --build build --config Release
    
    # Run native benchmark
    echo "Running ROCm benchmark..."
    ./build/gpu_bench 1024 1000 3 rocm results_rocm.csv ultra
    
    # Check results
    if [ -f results_rocm.csv ]; then
        echo "✅ ROCm benchmark completed"
        echo "Results:"
        cat results_rocm.csv
    else
        echo "❌ ROCm benchmark failed"
        exit 1
    fi
    
    cd ../..
}

# Main execution
case $OS in
    Darwin)
        test_metal
        ;;
    Linux)
        test_cuda
        test_rocm
        ;;
    *)
        echo "❌ Unsupported OS: $OS"
        exit 1
        ;;
esac

echo ""
echo "🎯 Build validation completed successfully!"
echo "All available GPU platforms tested and working."