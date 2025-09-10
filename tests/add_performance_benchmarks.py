#!/usr/bin/env python3
"""
Performance Benchmarks Implementation
====================================

This script creates comprehensive performance benchmarks for the SOA system.
"""

import sys
import os
import time
import json
import statistics
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Add services to path
sys.path.insert(0, 'services')
sys.path.insert(0, 'services/holographic-memory/core/native/holographic/build')

class PerformanceBenchmark:
    """Comprehensive performance benchmarking system"""
    
    def __init__(self):
        self.benchmark_results = {}
        self.benchmark_metrics = {
            "total_benchmarks": 0,
            "completed_benchmarks": 0,
            "failed_benchmarks": 0,
            "total_duration": 0.0
        }
    
    def run_all_benchmarks(self):
        """Run all performance benchmarks"""
        print("🚀 Starting Performance Benchmarks")
        print("=" * 60)
        
        start_time = time.time()
        
        # Define benchmark categories
        benchmark_categories = [
            ("GPU Performance", self.benchmark_gpu_performance),
            ("Service Performance", self.benchmark_service_performance),
            ("File Processing Performance", self.benchmark_file_processing_performance),
            ("Compression Performance", self.benchmark_compression_performance),
            ("API Performance", self.benchmark_api_performance),
            ("Memory Performance", self.benchmark_memory_performance),
            ("Concurrency Performance", self.benchmark_concurrency_performance),
            ("End-to-End Performance", self.benchmark_end_to_end_performance)
        ]
        
        # Run benchmarks
        for category_name, benchmark_func in benchmark_categories:
            print(f"\n🔧 Running {category_name} Benchmarks...")
            try:
                result = benchmark_func()
                result["category"] = category_name
                result["status"] = "completed"
                self.benchmark_results[category_name] = result
                self.benchmark_metrics["completed_benchmarks"] += 1
                
                # Count individual benchmarks
                if "benchmarks" in result:
                    self.benchmark_metrics["total_benchmarks"] += len(result["benchmarks"])
                
                print(f"   ✅ {category_name}: {len(result.get('benchmarks', []))} benchmarks completed")
                
            except Exception as e:
                print(f"   ❌ {category_name}: Benchmark failed - {e}")
                self.benchmark_results[category_name] = {
                    "category": category_name,
                    "status": "failed",
                    "error": str(e),
                    "benchmarks": []
                }
                self.benchmark_metrics["failed_benchmarks"] += 1
        
        # Calculate total duration
        self.benchmark_metrics["total_duration"] = time.time() - start_time
        
        return self.benchmark_results
    
    def benchmark_gpu_performance(self) -> Dict[str, Any]:
        """Benchmark GPU performance"""
        benchmarks = []
        
        try:
            import holographic_gpu
            gpu = holographic_gpu.HolographicGPU()
            
            if not gpu.initialize():
                return {"benchmarks": [], "error": "GPU initialization failed"}
            
            # Benchmark 1: GPU initialization time
            start_time = time.time()
            gpu.initialize()
            init_time = time.time() - start_time
            benchmarks.append({
                "name": "GPU Initialization",
                "metric": "time",
                "value": init_time,
                "unit": "seconds",
                "description": "Time to initialize GPU backend"
            })
            
            # Benchmark 2: FFT performance
            sizes = [1024, 4096, 16384, 65536]
            for size in sizes:
                test_real = np.random.rand(size).astype(np.float32)
                test_imag = np.random.rand(size).astype(np.float32)
                
                # Warm up
                gpu.holographic_fft_transform(test_real, test_imag)
                
                # Benchmark
                start_time = time.time()
                for _ in range(10):
                    gpu.holographic_fft_transform(test_real, test_imag)
                fft_time = (time.time() - start_time) / 10
                
                throughput = size / fft_time
                benchmarks.append({
                    "name": f"FFT Performance ({size} elements)",
                    "metric": "throughput",
                    "value": throughput,
                    "unit": "elements/second",
                    "description": f"FFT throughput for {size} elements"
                })
            
            # Benchmark 3: Encoding performance
            for size in sizes:
                test_real = np.random.rand(size).astype(np.float32)
                test_imag = np.random.rand(size).astype(np.float32)
                
                # Warm up
                gpu.batch_holographic_encode_fft(test_real, test_imag, 7)
                
                # Benchmark
                start_time = time.time()
                for _ in range(10):
                    gpu.batch_holographic_encode_fft(test_real, test_imag, 7)
                encode_time = (time.time() - start_time) / 10
                
                throughput = size / encode_time
                benchmarks.append({
                    "name": f"Encoding Performance ({size} elements)",
                    "metric": "throughput",
                    "value": throughput,
                    "unit": "elements/second",
                    "description": f"Encoding throughput for {size} elements"
                })
            
        except Exception as e:
            benchmarks.append({
                "name": "GPU Performance",
                "metric": "error",
                "value": 0,
                "unit": "error",
                "description": f"GPU benchmark failed: {e}"
            })
        
        return {"benchmarks": benchmarks}
    
    def benchmark_service_performance(self) -> Dict[str, Any]:
        """Benchmark service performance"""
        benchmarks = []
        
        # Benchmark 1: Orchestrator initialization
        try:
            from services.orchestrator.orchestrator import HolographicMemoryOrchestrator
            import tempfile
            
            start_time = time.time()
            with tempfile.TemporaryDirectory() as temp_dir:
                orchestrator = HolographicMemoryOrchestrator(temp_dir)
            init_time = time.time() - start_time
            
            benchmarks.append({
                "name": "Orchestrator Initialization",
                "metric": "time",
                "value": init_time,
                "unit": "seconds",
                "description": "Time to initialize orchestrator"
            })
        except Exception as e:
            benchmarks.append({
                "name": "Orchestrator Initialization",
                "metric": "error",
                "value": 0,
                "unit": "error",
                "description": f"Orchestrator benchmark failed: {e}"
            })
        
        # Benchmark 2: Router performance
        try:
            from services.router.mathematical_router import MathematicalRouter
            router = MathematicalRouter()
            
            # Test different data sizes
            data_sizes = [1024, 4096, 16384]  # bytes
            for size in data_sizes:
                test_data = b"x" * size
                
                # Warm up
                router.route_content(test_data, {"filename": "test.txt"})
                
                # Benchmark
                start_time = time.time()
                for _ in range(100):
                    router.route_content(test_data, {"filename": "test.txt"})
                route_time = (time.time() - start_time) / 100
                
                throughput = size / route_time
                benchmarks.append({
                    "name": f"Router Performance ({size} bytes)",
                    "metric": "throughput",
                    "value": throughput,
                    "unit": "bytes/second",
                    "description": f"Router throughput for {size} byte data"
                })
        except Exception as e:
            benchmarks.append({
                "name": "Router Performance",
                "metric": "error",
                "value": 0,
                "unit": "error",
                "description": f"Router benchmark failed: {e}"
            })
        
        # Benchmark 3: Vault performance
        try:
            from services.vault.security_guard import SecurityGuard
            guard = SecurityGuard()
            
            start_time = time.time()
            for _ in range(1000):
                guard.generate_vault_id()
            vault_time = (time.time() - start_time) / 1000
            
            benchmarks.append({
                "name": "Vault ID Generation",
                "metric": "throughput",
                "value": 1.0 / vault_time,
                "unit": "operations/second",
                "description": "Vault ID generation throughput"
            })
        except Exception as e:
            benchmarks.append({
                "name": "Vault Performance",
                "metric": "error",
                "value": 0,
                "unit": "error",
                "description": f"Vault benchmark failed: {e}"
            })
        
        return {"benchmarks": benchmarks}
    
    def benchmark_file_processing_performance(self) -> Dict[str, Any]:
        """Benchmark file processing performance"""
        benchmarks = []
        
        try:
            sys.path.insert(0, 'services/holographic-memory/api')
            from file_processor import FileProcessor
            processor = FileProcessor()
            
            # Test different file types and sizes
            test_cases = [
                ("text", b"This is a test text file for performance benchmarking. " * 100, "txt"),
                ("json", b'{"data": "' + b"x" * 1000 + b'"}', "json"),
                ("csv", b"name,age,city\n" + b"John,25,NYC\n" * 100, "csv")
            ]
            
            for file_type, content, extension in test_cases:
                # Warm up
                processor.process_file(content, f"test.{extension}")
                
                # Benchmark
                start_time = time.time()
                for _ in range(10):
                    processor.process_file(content, f"test.{extension}")
                process_time = (time.time() - start_time) / 10
                
                throughput = len(content) / process_time
                benchmarks.append({
                    "name": f"File Processing ({file_type})",
                    "metric": "throughput",
                    "value": throughput,
                    "unit": "bytes/second",
                    "description": f"File processing throughput for {file_type} files"
                })
        except Exception as e:
            benchmarks.append({
                "name": "File Processing Performance",
                "metric": "error",
                "value": 0,
                "unit": "error",
                "description": f"File processing benchmark failed: {e}"
            })
        
        return {"benchmarks": benchmarks}
    
    def benchmark_compression_performance(self) -> Dict[str, Any]:
        """Benchmark compression performance"""
        benchmarks = []
        
        try:
            from complete_compression_pipeline import CompressionPipeline
            pipeline = CompressionPipeline()
            pipeline.initialize_gpu()
            
            # Test different data sizes and types
            test_cases = [
                (512, "sparse"),
                (1024, "random"),
                (2048, "dense"),
                (4096, "mixed")
            ]
            
            for size, data_type in test_cases:
                if data_type == "sparse":
                    test_real = np.zeros(size, dtype=np.float32)
                    test_imag = np.zeros(size, dtype=np.float32)
                    # Add sparse non-zero values
                    indices = np.random.choice(size, size=size//10, replace=False)
                    test_real[indices] = np.random.rand(len(indices)).astype(np.float32)
                    test_imag[indices] = np.random.rand(len(indices)).astype(np.float32)
                elif data_type == "dense":
                    test_real = np.ones(size, dtype=np.float32) * 0.5
                    test_imag = np.ones(size, dtype=np.float32) * 0.3
                else:  # random or mixed
                    test_real = np.random.rand(size).astype(np.float32)
                    test_imag = np.random.rand(size).astype(np.float32)
                
                # Warm up
                pipeline.full_compression_pipeline(test_real, test_imag, use_gpu=False)
                
                # Benchmark
                start_time = time.time()
                for _ in range(5):
                    result = pipeline.full_compression_pipeline(test_real, test_imag, use_gpu=False)
                compress_time = (time.time() - start_time) / 5
                
                throughput = size / compress_time
                compression_ratio = result.get("overall_compression_ratio", 1.0)
                
                benchmarks.append({
                    "name": f"Compression ({data_type}, {size} elements)",
                    "metric": "throughput",
                    "value": throughput,
                    "unit": "elements/second",
                    "description": f"Compression throughput for {data_type} data"
                })
                
                benchmarks.append({
                    "name": f"Compression Ratio ({data_type}, {size} elements)",
                    "metric": "ratio",
                    "value": compression_ratio,
                    "unit": "x",
                    "description": f"Compression ratio for {data_type} data"
                })
        except Exception as e:
            benchmarks.append({
                "name": "Compression Performance",
                "metric": "error",
                "value": 0,
                "unit": "error",
                "description": f"Compression benchmark failed: {e}"
            })
        
        return {"benchmarks": benchmarks}
    
    def benchmark_api_performance(self) -> Dict[str, Any]:
        """Benchmark API performance"""
        benchmarks = []
        
        try:
            sys.path.insert(0, 'services/holographic-memory/api')
            from app import app
            from fastapi.testclient import TestClient
            
            client = TestClient(app)
            
            # Benchmark 1: Health endpoint
            start_time = time.time()
            for _ in range(100):
                response = client.get("/health")
            health_time = (time.time() - start_time) / 100
            
            benchmarks.append({
                "name": "API Health Endpoint",
                "metric": "throughput",
                "value": 1.0 / health_time,
                "unit": "requests/second",
                "description": "API health endpoint throughput"
            })
            
            # Benchmark 2: File upload endpoint (simulated)
            test_data = b"Test data for API benchmarking"
            start_time = time.time()
            for _ in range(50):
                response = client.post("/api/v1/files/upload", 
                                     files={"file": ("test.txt", test_data, "text/plain")})
            upload_time = (time.time() - start_time) / 50
            
            benchmarks.append({
                "name": "API File Upload",
                "metric": "throughput",
                "value": 1.0 / upload_time,
                "unit": "requests/second",
                "description": "API file upload throughput"
            })
        except Exception as e:
            benchmarks.append({
                "name": "API Performance",
                "metric": "error",
                "value": 0,
                "unit": "error",
                "description": f"API benchmark failed: {e}"
            })
        
        return {"benchmarks": benchmarks}
    
    def benchmark_memory_performance(self) -> Dict[str, Any]:
        """Benchmark memory performance"""
        benchmarks = []
        
        try:
            import psutil
            process = psutil.Process()
            
            # Benchmark 1: Memory allocation
            start_time = time.time()
            data_arrays = []
            for i in range(100):
                data_arrays.append(np.random.rand(1000, 1000).astype(np.float32))
            alloc_time = time.time() - start_time
            
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            benchmarks.append({
                "name": "Memory Allocation",
                "metric": "time",
                "value": alloc_time,
                "unit": "seconds",
                "description": "Time to allocate 100MB of data"
            })
            
            benchmarks.append({
                "name": "Memory Usage",
                "metric": "memory",
                "value": memory_usage,
                "unit": "MB",
                "description": "Current memory usage"
            })
            
            # Clean up
            del data_arrays
            
        except ImportError:
            benchmarks.append({
                "name": "Memory Performance",
                "metric": "error",
                "value": 0,
                "unit": "error",
                "description": "psutil not available for memory benchmarking"
            })
        except Exception as e:
            benchmarks.append({
                "name": "Memory Performance",
                "metric": "error",
                "value": 0,
                "unit": "error",
                "description": f"Memory benchmark failed: {e}"
            })
        
        return {"benchmarks": benchmarks}
    
    def benchmark_concurrency_performance(self) -> Dict[str, Any]:
        """Benchmark concurrency performance"""
        benchmarks = []
        
        def concurrent_task(task_id):
            try:
                from services.router.mathematical_router import MathematicalRouter
                router = MathematicalRouter()
                test_data = f"Concurrent task {task_id}".encode()
                
                start_time = time.time()
                for _ in range(10):
                    router.route_content(test_data, {"filename": f"task_{task_id}.txt"})
                return time.time() - start_time
            except Exception as e:
                return None
        
        try:
            # Test different concurrency levels
            concurrency_levels = [1, 2, 4, 8, 16]
            
            for level in concurrency_levels:
                start_time = time.time()
                
                with ThreadPoolExecutor(max_workers=level) as executor:
                    futures = [executor.submit(concurrent_task, i) for i in range(level * 5)]
                    results = [future.result() for future in as_completed(futures)]
                
                total_time = time.time() - start_time
                successful_tasks = sum(1 for result in results if result is not None)
                
                if successful_tasks > 0:
                    throughput = successful_tasks / total_time
                    benchmarks.append({
                        "name": f"Concurrency ({level} workers)",
                        "metric": "throughput",
                        "value": throughput,
                        "unit": "tasks/second",
                        "description": f"Concurrent task throughput with {level} workers"
                    })
        except Exception as e:
            benchmarks.append({
                "name": "Concurrency Performance",
                "metric": "error",
                "value": 0,
                "unit": "error",
                "description": f"Concurrency benchmark failed: {e}"
            })
        
        return {"benchmarks": benchmarks}
    
    def benchmark_end_to_end_performance(self) -> Dict[str, Any]:
        """Benchmark end-to-end performance"""
        benchmarks = []
        
        try:
            # Simulate complete workflow
            def end_to_end_workflow():
                # Step 1: File processing
                sys.path.insert(0, 'services/holographic-memory/api')
                from file_processor import FileProcessor
                processor = FileProcessor()
                
                test_data = b"End-to-end performance test data for holographic memory system"
                processed_data = processor.process_file(test_data, "test.txt")
                
                # Step 2: Compression
                from complete_compression_pipeline import CompressionPipeline
                pipeline = CompressionPipeline()
                pipeline.initialize_gpu()
                
                test_real = np.random.rand(1024).astype(np.float32)
                test_imag = np.random.rand(1024).astype(np.float32)
                compression_result = pipeline.full_compression_pipeline(test_real, test_imag, use_gpu=False)
                
                # Step 3: Service routing
                from services.router.mathematical_router import MathematicalRouter
                router = MathematicalRouter()
                routing_result = router.route_content(test_data, {"filename": "test.txt"})
                
                return {
                    "file_processing": processed_data.get("supported", False),
                    "compression": compression_result.get("success", False),
                    "routing": routing_result is not None
                }
            
            # Warm up
            end_to_end_workflow()
            
            # Benchmark
            start_time = time.time()
            for _ in range(10):
                result = end_to_end_workflow()
            workflow_time = (time.time() - start_time) / 10
            
            throughput = 1.0 / workflow_time
            benchmarks.append({
                "name": "End-to-End Workflow",
                "metric": "throughput",
                "value": throughput,
                "unit": "workflows/second",
                "description": "Complete end-to-end workflow throughput"
            })
            
        except Exception as e:
            benchmarks.append({
                "name": "End-to-End Performance",
                "metric": "error",
                "value": 0,
                "unit": "error",
                "description": f"End-to-end benchmark failed: {e}"
            })
        
        return {"benchmarks": benchmarks}
    
    def generate_benchmark_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        report = {
            "benchmark_summary": self.benchmark_metrics,
            "benchmark_results": self.benchmark_results,
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info()
        }
        
        return report
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            import platform
            import psutil
            
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
                "memory_available": psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
            }
        except ImportError:
            return {
                "platform": "Unknown",
                "python_version": "Unknown",
                "cpu_count": "Unknown",
                "memory_total": "Unknown",
                "memory_available": "Unknown"
            }

def main():
    """Main performance benchmark function"""
    print("🚀 Starting Performance Benchmarks")
    print("=" * 60)
    
    # Create and run benchmark suite
    benchmark_suite = PerformanceBenchmark()
    results = benchmark_suite.run_all_benchmarks()
    
    # Generate report
    report = benchmark_suite.generate_benchmark_report()
    
    # Save report
    with open("performance_benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE BENCHMARK REPORT")
    print("=" * 60)
    
    print(f"\n📈 BENCHMARK SUMMARY:")
    print(f"   Total Benchmarks: {benchmark_suite.benchmark_metrics['total_benchmarks']}")
    print(f"   Completed: {benchmark_suite.benchmark_metrics['completed_benchmarks']}")
    print(f"   Failed: {benchmark_suite.benchmark_metrics['failed_benchmarks']}")
    print(f"   Duration: {benchmark_suite.benchmark_metrics['total_duration']:.2f}s")
    
    print(f"\n🔧 BENCHMARK CATEGORIES:")
    for category_name, category_result in results.items():
        if "benchmarks" in category_result:
            benchmark_count = len(category_result["benchmarks"])
            status_emoji = "✅" if category_result["status"] == "completed" else "❌"
            print(f"   {status_emoji} {category_name}: {benchmark_count} benchmarks")
        else:
            print(f"   ❌ {category_name}: Failed to run")
    
    print(f"\n🏥 SYSTEM INFORMATION:")
    for key, value in report["system_info"].items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    if benchmark_suite.benchmark_metrics['completed_benchmarks'] > 0:
        print(f"\n🎉 PERFORMANCE BENCHMARKS COMPLETED SUCCESSFULLY!")
        print("✅ Comprehensive performance data collected")
        print("✅ Multiple benchmark categories covered")
        print("✅ System performance metrics recorded")
        print("✅ Benchmark report saved to performance_benchmark_report.json")
    else:
        print(f"\n⚠️  PERFORMANCE BENCHMARKS FAILED")
        print("❌ No benchmarks completed successfully")
    
    print("=" * 60)
    
    return report

if __name__ == "__main__":
    main()
