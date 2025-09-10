#!/usr/bin/env python3
"""
GPU Acceleration Verification Script
===================================

This script verifies that GPU acceleration is working properly across all services
in the SOA system. It tests Metal shaders, GPU kernels, and service integration.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Any, Tuple
import subprocess
import json

# Add services to path
sys.path.insert(0, 'services')
sys.path.insert(0, 'services/holographic-memory/core/native/holographic/build')

def test_gpu_initialization():
    """Test GPU backend initialization"""
    print("🔧 Testing GPU Backend Initialization...")
    
    try:
        import holographic_gpu
        gpu = holographic_gpu.HolographicGPU()
        
        if gpu.initialize():
            print("✅ GPU backend initialized successfully")
            return True, gpu
        else:
            print("❌ GPU backend initialization failed")
            return False, None
    except Exception as e:
        print(f"❌ GPU initialization error: {e}")
        return False, None

def test_metal_shaders():
    """Test Metal shader compilation and loading"""
    print("\n🎨 Testing Metal Shader Compilation...")
    
    shader_file = "services/holographic-memory/core/native/holographic/metal/holographic_memory.metal"
    
    if not os.path.exists(shader_file):
        print(f"❌ Shader file not found: {shader_file}")
        return False
    
    try:
        with open(shader_file, 'r') as f:
            shader_content = f.read()
        
        # Check for key shader functions
        required_shaders = [
            "enhanced_vector_add",
            "batch_holographic_store", 
            "holographic_similarity_search",
            "holographic_fft_transform",
            "batch_holographic_encode_fft",
            "gpu_sparse_encoding",
            "gpu_entropy_coding",
            "gpu_entropy_decoding",
            "gpu_sparse_decoding"
        ]
        
        missing_shaders = []
        for shader in required_shaders:
            if f"kernel void {shader}" not in shader_content:
                missing_shaders.append(shader)
        
        if missing_shaders:
            print(f"❌ Missing shaders: {missing_shaders}")
            return False
        
        print("✅ All required Metal shaders found")
        return True
        
    except Exception as e:
        print(f"❌ Shader file error: {e}")
        return False

def test_gpu_kernels(gpu):
    """Test individual GPU kernels"""
    print("\n⚡ Testing GPU Kernels...")
    
    if not gpu:
        print("❌ No GPU backend available")
        return False
    
    # Test data
    test_size = 1024
    test_real = np.random.rand(test_size).astype(np.float32)
    test_imag = np.random.rand(test_size).astype(np.float32)
    
    kernel_tests = [
        ("Enhanced Vector Add", lambda: gpu.enhanced_vector_add(test_real, test_imag, test_real, test_imag)),
        ("Batch Holographic Store", lambda: gpu.batch_holographic_store(test_real, test_imag, 7)),
        ("Holographic FFT Transform", lambda: gpu.holographic_fft_transform(test_real, test_imag)),
        ("Batch Holographic Encode FFT", lambda: gpu.batch_holographic_encode_fft(test_real, test_imag, 7)),
        ("GPU Holographic Quantize", lambda: gpu.gpu_holographic_quantize(test_real, test_imag, 8)),
        ("GPU Quantization Statistics", lambda: gpu.gpu_quantization_statistics(test_real, test_imag, 8))
    ]
    
    results = {}
    for kernel_name, test_func in kernel_tests:
        try:
            start_time = time.time()
            result = test_func()
            execution_time = time.time() - start_time
            
            if result is not None:
                print(f"✅ {kernel_name}: {execution_time:.4f}s")
                results[kernel_name] = {"success": True, "time": execution_time}
            else:
                print(f"❌ {kernel_name}: No result returned")
                results[kernel_name] = {"success": False, "time": 0}
                
        except Exception as e:
            print(f"❌ {kernel_name}: {e}")
            results[kernel_name] = {"success": False, "time": 0, "error": str(e)}
    
    return results

def test_compression_pipeline(gpu):
    """Test the new compression pipeline kernels"""
    print("\n🗜️ Testing Compression Pipeline...")
    
    if not gpu:
        print("❌ No GPU backend available")
        return False
    
    # Test data
    test_size = 512
    test_real = np.random.rand(test_size).astype(np.float32)
    test_imag = np.random.rand(test_size).astype(np.float32)
    
    compression_tests = [
        ("Sparse Encoding", lambda: gpu.gpu_sparse_encoding(test_real, test_imag, 0.1, 256)),
        ("Entropy Coding", lambda: gpu.gpu_entropy_coding(test_real, test_imag)),
        ("Entropy Decoding", lambda: gpu.gpu_entropy_decoding(test_real, test_imag)),
        ("Sparse Decoding", lambda: gpu.gpu_sparse_decoding(test_real, test_imag, np.arange(256, dtype=np.uint32), 256))
    ]
    
    results = {}
    for test_name, test_func in compression_tests:
        try:
            start_time = time.time()
            result = test_func()
            execution_time = time.time() - start_time
            
            if result is not None:
                print(f"✅ {test_name}: {execution_time:.4f}s")
                results[test_name] = {"success": True, "time": execution_time}
            else:
                print(f"❌ {test_name}: No result returned")
                results[test_name] = {"success": False, "time": 0}
                
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            results[test_name] = {"success": False, "time": 0, "error": str(e)}
    
    return results

def test_service_gpu_integration():
    """Test GPU integration with services"""
    print("\n🔗 Testing Service-GPU Integration...")
    
    try:
        # Test Orchestrator GPU integration
        from services.orchestrator.orchestrator import HolographicMemoryOrchestrator
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = HolographicMemoryOrchestrator(temp_dir)
            
            # Test GPU operations through orchestrator
            test_data = b"Test data for GPU processing"
            result = orchestrator.process_data(test_data, {"use_gpu": True})
            
            if result and "gpu_processed" in result:
                print("✅ Orchestrator GPU integration working")
            else:
                print("❌ Orchestrator GPU integration failed")
                return False
        
        # Test Router GPU integration
        from services.router.mathematical_router import MathematicalRouter
        router = MathematicalRouter()
        
        routing_result = router.route_content(test_data, {"filename": "test.txt"})
        if routing_result and "format" in routing_result:
            print("✅ Router GPU integration working")
        else:
            print("❌ Router GPU integration failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Service-GPU integration error: {e}")
        return False

def test_gpu_performance():
    """Test GPU performance benchmarks"""
    print("\n📊 Testing GPU Performance...")
    
    try:
        import holographic_gpu
        gpu = holographic_gpu.HolographicGPU()
        
        if not gpu.initialize():
            print("❌ GPU not available for performance testing")
            return False
        
        # Performance test data
        sizes = [1024, 4096, 16384]
        results = {}
        
        for size in sizes:
            print(f"\n   Testing size: {size}")
            test_real = np.random.rand(size).astype(np.float32)
            test_imag = np.random.rand(size).astype(np.float32)
            
            # Test FFT performance
            start_time = time.time()
            for _ in range(10):  # Multiple iterations for average
                gpu.holographic_fft_transform(test_real, test_imag)
            fft_time = (time.time() - start_time) / 10
            
            # Test encoding performance
            start_time = time.time()
            for _ in range(10):
                gpu.batch_holographic_encode_fft(test_real, test_imag, 7)
            encode_time = (time.time() - start_time) / 10
            
            results[size] = {
                "fft_time": fft_time,
                "encode_time": encode_time,
                "fft_ops_per_sec": size / fft_time,
                "encode_ops_per_sec": size / encode_time
            }
            
            print(f"     FFT: {fft_time:.4f}s ({size/fft_time:.0f} ops/s)")
            print(f"     Encode: {encode_time:.4f}s ({size/encode_time:.0f} ops/s)")
        
        return results
        
    except Exception as e:
        print(f"❌ GPU performance test error: {e}")
        return False

def test_gpu_memory_management():
    """Test GPU memory management"""
    print("\n💾 Testing GPU Memory Management...")
    
    try:
        import holographic_gpu
        gpu = holographic_gpu.HolographicGPU()
        
        if not gpu.initialize():
            print("❌ GPU not available for memory testing")
            return False
        
        # Test large data processing
        large_size = 65536  # 64K elements
        test_real = np.random.rand(large_size).astype(np.float32)
        test_imag = np.random.rand(large_size).astype(np.float32)
        
        print(f"   Processing {large_size} elements...")
        
        start_time = time.time()
        result = gpu.batch_holographic_encode_fft(test_real, test_imag, 7)
        processing_time = time.time() - start_time
        
        if result is not None:
            print(f"✅ Large data processing: {processing_time:.4f}s")
            print(f"   Throughput: {large_size/processing_time:.0f} elements/s")
            return True
        else:
            print("❌ Large data processing failed")
            return False
            
    except Exception as e:
        print(f"❌ GPU memory test error: {e}")
        return False

def generate_gpu_report(test_results):
    """Generate comprehensive GPU acceleration report"""
    print("\n" + "=" * 60)
    print("📊 GPU ACCELERATION VERIFICATION REPORT")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in test_results.items():
        if isinstance(results, dict):
            category_passed = sum(1 for r in results.values() if r.get("success", False))
            category_total = len(results)
            total_tests += category_total
            passed_tests += category_passed
            
            print(f"\n🔧 {category.upper()}:")
            print(f"   Passed: {category_passed}/{category_total}")
            
            for test_name, result in results.items():
                status = "✅" if result.get("success", False) else "❌"
                time_info = f" ({result.get('time', 0):.4f}s)" if result.get("time") else ""
                print(f"     {status} {test_name}{time_info}")
                
                if "error" in result:
                    print(f"       Error: {result['error']}")
        else:
            # Boolean results
            total_tests += 1
            if results:
                passed_tests += 1
                print(f"\n✅ {category.upper()}: PASSED")
            else:
                print(f"\n❌ {category.upper()}: FAILED")
    
    print(f"\n📈 OVERALL RESULTS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print(f"\n🎉 ALL GPU ACCELERATION TESTS PASSED!")
        print("✅ GPU acceleration is working across all services")
    else:
        print(f"\n⚠️  SOME GPU ACCELERATION TESTS FAILED")
        print("❌ GPU acceleration needs attention")
    
    print("=" * 60)

def main():
    """Main GPU acceleration verification function"""
    print("🚀 Starting GPU Acceleration Verification")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: GPU Initialization
    gpu_available, gpu = test_gpu_initialization()
    test_results["GPU Initialization"] = gpu_available
    
    # Test 2: Metal Shaders
    shaders_ok = test_metal_shaders()
    test_results["Metal Shaders"] = shaders_ok
    
    # Test 3: GPU Kernels
    if gpu_available:
        kernel_results = test_gpu_kernels(gpu)
        test_results["GPU Kernels"] = kernel_results
    else:
        test_results["GPU Kernels"] = {}
    
    # Test 4: Compression Pipeline
    if gpu_available:
        compression_results = test_compression_pipeline(gpu)
        test_results["Compression Pipeline"] = compression_results
    else:
        test_results["Compression Pipeline"] = {}
    
    # Test 5: Service Integration
    service_integration = test_service_gpu_integration()
    test_results["Service Integration"] = service_integration
    
    # Test 6: Performance
    if gpu_available:
        performance_results = test_gpu_performance()
        test_results["Performance"] = performance_results
    else:
        test_results["Performance"] = False
    
    # Test 7: Memory Management
    if gpu_available:
        memory_results = test_gpu_memory_management()
        test_results["Memory Management"] = memory_results
    else:
        test_results["Memory Management"] = False
    
    # Generate report
    generate_gpu_report(test_results)
    
    return test_results

if __name__ == "__main__":
    main()
