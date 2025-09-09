#!/usr/bin/env python3

import numpy as np
import sys
from pathlib import Path

# Add paths
p = Path('holographic-fs/native/holographic')
sys.path.insert(0, str(p / 'build'))

def validate_gpu_accuracy():
    """Validate GPU FFT accuracy against CPU FFT reference."""
    
    try:
        import holographic_gpu as hg
    except ImportError:
        print("❌ GPU module not available")
        return False
    
    # Test parameters
    batch_size = 100
    data_length = 64
    pattern_dimension = 1024
    
    # Generate test data
    np.random.seed(42)  # Reproducible
    test_data = np.random.randn(batch_size, data_length).astype(np.float32)
    
    print(f"Testing accuracy with {batch_size} samples, {data_length} -> {pattern_dimension}")
    
    # CPU reference (numpy FFT)
    print("Computing CPU reference...")
    cpu_results = []
    for i in range(batch_size):
        # Pad to pattern dimension
        padded = np.zeros(pattern_dimension, dtype=np.float32)
        padded[:data_length] = test_data[i]
        
        # FFT with unitary scaling
        fft_result = np.fft.fft(padded)
        magnitude = np.abs(fft_result)
        
        cpu_results.append(magnitude)
    
    cpu_results = np.array(cpu_results)
    
    # GPU computation
    print("Computing GPU results...")
    gpu = hg.MetalHolographicBackend()
    if not gpu.available():
        print("❌ GPU not available")
        return False
    
    gpu_results = gpu.batch_encode_fft_ultra_numpy(test_data, pattern_dimension)
    gpu_results = np.array(gpu_results)
    
    # Compare results
    print("Comparing results...")
    
    # Calculate differences
    abs_diff = np.abs(gpu_results - cpu_results)
    rel_diff = abs_diff / (np.abs(cpu_results) + 1e-8)
    
    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)
    mean_rel_diff = np.mean(rel_diff)
    
    print(f"Max absolute difference: {max_abs_diff:.6f}")
    print(f"Max relative difference: {max_rel_diff:.6f}")
    print(f"Mean absolute difference: {mean_abs_diff:.6f}")
    print(f"Mean relative difference: {mean_rel_diff:.6f}")
    
    # Tolerance checks
    abs_tolerance = 1e-4
    rel_tolerance = 1e-3
    
    abs_pass = max_abs_diff < abs_tolerance
    rel_pass = max_rel_diff < rel_tolerance
    
    print(f"\nTolerance checks:")
    print(f"Absolute tolerance ({abs_tolerance}): {'✅ PASS' if abs_pass else '❌ FAIL'}")
    print(f"Relative tolerance ({rel_tolerance}): {'✅ PASS' if rel_pass else '❌ FAIL'}")
    
    if abs_pass and rel_pass:
        print("✅ Accuracy validation PASSED")
        return True
    else:
        print("❌ Accuracy validation FAILED")
        return False

if __name__ == "__main__":
    success = validate_gpu_accuracy()
    sys.exit(0 if success else 1)
