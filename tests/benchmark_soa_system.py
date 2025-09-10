#!/usr/bin/env python3
"""
Comprehensive SOA System Benchmark
Tests performance across all services and components.
"""

import time
import asyncio
import sys
import os
import tempfile
from pathlib import Path
import statistics
import psutil
import gc

# Add services to path
sys.path.insert(0, 'services')
sys.path.insert(0, 'services/holographic-memory/core/native/holographic/build')
sys.path.insert(0, 'services/holographic-memory/api')

def benchmark_service_initialization():
    """Benchmark service initialization times."""
    print("🚀 Benchmarking Service Initialization...")
    
    results = {}
    
    # Test Orchestrator
    start_time = time.time()
    try:
        from services.orchestrator.orchestrator import HolographicMemoryOrchestrator
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = HolographicMemoryOrchestrator(temp_dir)
        results['orchestrator'] = time.time() - start_time
        print(f"✅ Orchestrator: {results['orchestrator']:.3f}s")
    except Exception as e:
        print(f"❌ Orchestrator: {e}")
        results['orchestrator'] = None
    
    # Test Router
    start_time = time.time()
    try:
        from services.router.mathematical_router import MathematicalRouter
        router = MathematicalRouter()
        results['router'] = time.time() - start_time
        print(f"✅ Router: {results['router']:.3f}s")
    except Exception as e:
        print(f"❌ Router: {e}")
        results['router'] = None
    
    # Test Vault
    start_time = time.time()
    try:
        from services.vault.security_guard import SecurityGuard
        guard = SecurityGuard()
        results['vault'] = time.time() - start_time
        print(f"✅ Vault: {results['vault']:.3f}s")
    except Exception as e:
        print(f"❌ Vault: {e}")
        results['vault'] = None
    
    # Test Telemetry
    start_time = time.time()
    try:
        from services.telemetry.performance_tracker import PerformanceTelemetry
        telemetry = PerformanceTelemetry()
        results['telemetry'] = time.time() - start_time
        print(f"✅ Telemetry: {results['telemetry']:.3f}s")
    except Exception as e:
        print(f"❌ Telemetry: {e}")
        results['telemetry'] = None
    
    # Test GPU Backend
    start_time = time.time()
    try:
        import holographic_gpu
        gpu = holographic_gpu.HolographicGPU()
        if gpu.initialize():
            results['gpu'] = time.time() - start_time
            print(f"✅ GPU Backend: {results['gpu']:.3f}s")
        else:
            print("❌ GPU Backend: Initialization failed")
            results['gpu'] = None
    except Exception as e:
        print(f"❌ GPU Backend: {e}")
        results['gpu'] = None
    
    # Test File Processor
    start_time = time.time()
    try:
        from file_processor import file_processor
        results['file_processor'] = time.time() - start_time
        print(f"✅ File Processor: {results['file_processor']:.3f}s")
    except Exception as e:
        print(f"❌ File Processor: {e}")
        results['file_processor'] = None
    
    return results

def benchmark_gpu_operations():
    """Benchmark GPU operations."""
    print("\n🔥 Benchmarking GPU Operations...")
    
    results = {}
    
    try:
        import holographic_gpu
        gpu = holographic_gpu.HolographicGPU()
        if not gpu.initialize():
            print("❌ GPU not available for benchmarking")
            return results
        
        # Test batch encoding
        import numpy as np
        test_data = np.random.rand(100, 1000).astype(np.float32)
        
        # Warm up
        gpu.batch_encode_numpy(test_data, 512)
        
        # Benchmark batch encoding
        times = []
        for _ in range(10):
            start_time = time.time()
            result = gpu.batch_encode_numpy(test_data, 512)
            times.append(time.time() - start_time)
        
        results['batch_encoding'] = {
            'mean': statistics.mean(times),
            'std': statistics.stdev(times),
            'min': min(times),
            'max': max(times)
        }
        print(f"✅ Batch Encoding: {results['batch_encoding']['mean']:.3f}s ± {results['batch_encoding']['std']:.3f}s")
        
        # Test similarity search
        times = []
        for _ in range(10):
            start_time = time.time()
            similarity = gpu.similarity_search(test_data[0], test_data[1:10])
            times.append(time.time() - start_time)
        
        results['similarity_search'] = {
            'mean': statistics.mean(times),
            'std': statistics.stdev(times),
            'min': min(times),
            'max': max(times)
        }
        print(f"✅ Similarity Search: {results['similarity_search']['mean']:.3f}s ± {results['similarity_search']['std']:.3f}s")
        
    except Exception as e:
        print(f"❌ GPU Operations: {e}")
    
    return results

def benchmark_file_processing():
    """Benchmark file processing operations."""
    print("\n📁 Benchmarking File Processing...")
    
    results = {}
    
    try:
        from file_processor import file_processor
        
        # Test different file types
        test_files = {
            'text': b"This is a test document for holographic memory processing. " * 100,
            'csv': b"name,age,city\nJohn,25,New York\nJane,30,Los Angeles\nBob,35,Chicago\n" * 50,
            'json': b'{"test": "data", "numbers": [1, 2, 3, 4, 5], "nested": {"key": "value"}}' * 20
        }
        
        for file_type, content in test_files.items():
            times = []
            for _ in range(10):
                start_time = time.time()
                info = file_processor.get_file_info(f"test.{file_type}", content)
                times.append(time.time() - start_time)
            
            results[f'process_{file_type}'] = {
                'mean': statistics.mean(times),
                'std': statistics.stdev(times),
                'min': min(times),
                'max': max(times)
            }
            print(f"✅ Process {file_type.upper()}: {results[f'process_{file_type}']['mean']:.3f}s ± {results[f'process_{file_type}']['std']:.3f}s")
        
    except Exception as e:
        print(f"❌ File Processing: {e}")
    
    return results

def benchmark_service_communication():
    """Benchmark service communication."""
    print("\n🔄 Benchmarking Service Communication...")
    
    results = {}
    
    try:
        from services.router.mathematical_router import MathematicalRouter
        from services.vault.security_guard import SecurityGuard
        from services.telemetry.performance_tracker import PerformanceTelemetry
        
        router = MathematicalRouter()
        guard = SecurityGuard()
        telemetry = PerformanceTelemetry()
        
        # Test routing
        test_data = b"Test data for holographic processing"
        times = []
        for _ in range(100):
            start_time = time.time()
            routing = router.route_content(test_data, {'filename': 'test.txt'})
            times.append(time.time() - start_time)
        
        results['routing'] = {
            'mean': statistics.mean(times),
            'std': statistics.stdev(times),
            'min': min(times),
            'max': max(times)
        }
        print(f"✅ Routing: {results['routing']['mean']:.6f}s ± {results['routing']['std']:.6f}s")
        
        # Test vault operations
        times = []
        for _ in range(100):
            start_time = time.time()
            vault_id = guard.generate_vault_id()
            times.append(time.time() - start_time)
        
        results['vault_operations'] = {
            'mean': statistics.mean(times),
            'std': statistics.stdev(times),
            'min': min(times),
            'max': max(times)
        }
        print(f"✅ Vault Operations: {results['vault_operations']['mean']:.6f}s ± {results['vault_operations']['std']:.6f}s")
        
        # Test telemetry
        times = []
        for _ in range(100):
            start_time = time.time()
            telemetry.track_compression(1000, 100, 'test')
            times.append(time.time() - start_time)
        
        results['telemetry'] = {
            'mean': statistics.mean(times),
            'std': statistics.stdev(times),
            'min': min(times),
            'max': max(times)
        }
        print(f"✅ Telemetry: {results['telemetry']['mean']:.6f}s ± {results['telemetry']['std']:.6f}s")
        
    except Exception as e:
        print(f"❌ Service Communication: {e}")
    
    return results

def benchmark_memory_usage():
    """Benchmark memory usage."""
    print("\n💾 Benchmarking Memory Usage...")
    
    results = {}
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    try:
        # Test service initialization memory usage
        from services.orchestrator.orchestrator import HolographicMemoryOrchestrator
        from services.router.mathematical_router import MathematicalRouter
        from services.vault.security_guard import SecurityGuard
        from services.telemetry.performance_tracker import PerformanceTelemetry
        
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = HolographicMemoryOrchestrator(temp_dir)
            router = MathematicalRouter()
            guard = SecurityGuard()
            telemetry = PerformanceTelemetry()
            
            after_services = process.memory_info().rss / 1024 / 1024  # MB
            results['services_memory'] = after_services - initial_memory
            print(f"✅ Services Memory: {results['services_memory']:.1f} MB")
        
        # Test GPU memory usage
        try:
            import holographic_gpu
            gpu = holographic_gpu.HolographicGPU()
            if gpu.initialize():
                after_gpu = process.memory_info().rss / 1024 / 1024  # MB
                results['gpu_memory'] = after_gpu - after_services
                print(f"✅ GPU Memory: {results['gpu_memory']:.1f} MB")
        except Exception as e:
            print(f"❌ GPU Memory: {e}")
        
        # Test file processing memory usage
        try:
            from file_processor import file_processor
            test_content = b"Test content" * 10000
            info = file_processor.get_file_info("test.txt", test_content)
            
            after_file_processing = process.memory_info().rss / 1024 / 1024  # MB
            results['file_processing_memory'] = after_file_processing - after_services
            print(f"✅ File Processing Memory: {results['file_processing_memory']:.1f} MB")
        except Exception as e:
            print(f"❌ File Processing Memory: {e}")
        
    except Exception as e:
        print(f"❌ Memory Benchmark: {e}")
    
    return results

def benchmark_concurrent_operations():
    """Benchmark concurrent operations."""
    print("\n⚡ Benchmarking Concurrent Operations...")
    
    results = {}
    
    async def concurrent_routing():
        """Test concurrent routing operations."""
        from services.router.mathematical_router import MathematicalRouter
        router = MathematicalRouter()
        
        async def route_data(data, metadata):
            return router.route_content(data, metadata)
        
        # Create test data
        test_data = [(b"Test data " + str(i).encode(), {'filename': f'test{i}.txt'}) for i in range(100)]
        
        start_time = time.time()
        tasks = [route_data(data, metadata) for data, metadata in test_data]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        return end_time - start_time, len(results)
    
    try:
        # Test concurrent routing
        routing_time, count = asyncio.run(concurrent_routing())
        results['concurrent_routing'] = {
            'time': routing_time,
            'operations': count,
            'ops_per_second': count / routing_time
        }
        print(f"✅ Concurrent Routing: {count} operations in {routing_time:.3f}s ({results['concurrent_routing']['ops_per_second']:.1f} ops/s)")
        
    except Exception as e:
        print(f"❌ Concurrent Operations: {e}")
    
    return results

def generate_benchmark_report(all_results):
    """Generate a comprehensive benchmark report."""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE SOA SYSTEM BENCHMARK REPORT")
    print("="*60)
    
    # Service Initialization
    print("\n🚀 SERVICE INITIALIZATION:")
    for service, time_taken in all_results.get('initialization', {}).items():
        if time_taken is not None:
            print(f"   {service.capitalize()}: {time_taken:.3f}s")
        else:
            print(f"   {service.capitalize()}: FAILED")
    
    # GPU Operations
    print("\n🔥 GPU OPERATIONS:")
    for operation, stats in all_results.get('gpu_operations', {}).items():
        print(f"   {operation.replace('_', ' ').title()}: {stats['mean']:.3f}s ± {stats['std']:.3f}s")
    
    # File Processing
    print("\n📁 FILE PROCESSING:")
    for operation, stats in all_results.get('file_processing', {}).items():
        print(f"   {operation.replace('_', ' ').title()}: {stats['mean']:.3f}s ± {stats['std']:.3f}s")
    
    # Service Communication
    print("\n🔄 SERVICE COMMUNICATION:")
    for operation, stats in all_results.get('service_communication', {}).items():
        print(f"   {operation.replace('_', ' ').title()}: {stats['mean']:.6f}s ± {stats['std']:.6f}s")
    
    # Memory Usage
    print("\n💾 MEMORY USAGE:")
    for component, memory in all_results.get('memory_usage', {}).items():
        print(f"   {component.replace('_', ' ').title()}: {memory:.1f} MB")
    
    # Concurrent Operations
    print("\n⚡ CONCURRENT OPERATIONS:")
    for operation, stats in all_results.get('concurrent_operations', {}).items():
        print(f"   {operation.replace('_', ' ').title()}: {stats['ops_per_second']:.1f} ops/s")
    
    print("\n" + "="*60)
    print("✅ BENCHMARK COMPLETED SUCCESSFULLY")
    print("="*60)

def main():
    """Run comprehensive SOA system benchmark."""
    print("🚀 Starting Comprehensive SOA System Benchmark")
    print("="*60)
    
    all_results = {}
    
    # Run all benchmarks
    all_results['initialization'] = benchmark_service_initialization()
    all_results['gpu_operations'] = benchmark_gpu_operations()
    all_results['file_processing'] = benchmark_file_processing()
    all_results['service_communication'] = benchmark_service_communication()
    all_results['memory_usage'] = benchmark_memory_usage()
    all_results['concurrent_operations'] = benchmark_concurrent_operations()
    
    # Generate report
    generate_benchmark_report(all_results)
    
    return all_results

if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n🛑 Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
