#!/usr/bin/env python3
"""
TAI System Performance and Load Testing

This module provides comprehensive performance and load testing for the complete TAI system,
including stress testing, performance benchmarks, and scalability validation.

Enterprise-grade validation ensures production readiness under load.
"""

import os
import sys
import time
import requests
import hashlib
import json
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import statistics

import pytest
import numpy as np

# Import the system manager and client
from test_tai_system_startup import tai_system, TAISystemManager
from test_tai_e2e_integration import TAIClient, TAITestDataGenerator, TestData


@dataclass
class PerformanceMetrics:
    """Performance metrics for testing"""
    operation: str
    duration: float
    success: bool
    error: Optional[str] = None
    data_size: Optional[int] = None
    throughput: Optional[float] = None


@dataclass
class LoadTestConfig:
    """Configuration for load testing"""
    concurrent_users: int
    operations_per_user: int
    test_duration: int  # seconds
    ramp_up_time: int  # seconds
    data_size: int  # bytes


class TAIPerformanceTester:
    """Performance testing utilities for TAI system"""
    
    def __init__(self, client: TAIClient):
        self.client = client
        self.metrics: List[PerformanceMetrics] = []
    
    def measure_operation(self, operation_name: str, operation_func, *args, **kwargs) -> PerformanceMetrics:
        """Measure performance of an operation"""
        start_time = time.time()
        success = False
        error = None
        
        try:
            result = operation_func(*args, **kwargs)
            success = True
        except Exception as e:
            error = str(e)
            result = None
        
        duration = time.time() - start_time
        
        metric = PerformanceMetrics(
            operation=operation_name,
            duration=duration,
            success=success,
            error=error
        )
        
        self.metrics.append(metric)
        return metric
    
    def generate_test_data(self, size_mb: int) -> TestData:
        """Generate test data of specified size"""
        content = "A" * (size_mb * 1024 * 1024)
        return TestData(
            content=content,
            filename=f"test_{size_mb}mb.txt",
            size=len(content),
            hash=hashlib.sha256(content.encode()).hexdigest()
        )
    
    def test_upload_performance(self, data_size_mb: int, iterations: int = 10) -> List[PerformanceMetrics]:
        """Test upload performance"""
        test_data = self.generate_test_data(data_size_mb)
        metrics = []
        
        for i in range(iterations):
            metric = self.measure_operation(
                f"upload_{data_size_mb}mb",
                self.client.upload_file,
                test_data
            )
            metric.data_size = data_size_mb * 1024 * 1024
            metric.throughput = metric.data_size / metric.duration if metric.success else 0
            metrics.append(metric)
        
        return metrics
    
    def test_retrieval_performance(self, doc_id: str, iterations: int = 10) -> List[PerformanceMetrics]:
        """Test retrieval performance"""
        metrics = []
        
        for i in range(iterations):
            metric = self.measure_operation(
                "retrieve",
                self.client.retrieve_file,
                doc_id
            )
            metrics.append(metric)
        
        return metrics
    
    def test_search_performance(self, query: str, iterations: int = 10) -> List[PerformanceMetrics]:
        """Test search performance"""
        metrics = []
        
        for i in range(iterations):
            metric = self.measure_operation(
                "search",
                self.client.search_files,
                query
            )
            metrics.append(metric)
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.metrics:
            return {}
        
        successful_metrics = [m for m in self.metrics if m.success]
        failed_metrics = [m for m in self.metrics if not m.success]
        
        summary = {
            "total_operations": len(self.metrics),
            "successful_operations": len(successful_metrics),
            "failed_operations": len(failed_metrics),
            "success_rate": len(successful_metrics) / len(self.metrics) if self.metrics else 0,
            "average_duration": statistics.mean([m.duration for m in successful_metrics]) if successful_metrics else 0,
            "median_duration": statistics.median([m.duration for m in successful_metrics]) if successful_metrics else 0,
            "p95_duration": np.percentile([m.duration for m in successful_metrics], 95) if successful_metrics else 0,
            "p99_duration": np.percentile([m.duration for m in successful_metrics], 99) if successful_metrics else 0,
            "min_duration": min([m.duration for m in successful_metrics]) if successful_metrics else 0,
            "max_duration": max([m.duration for m in successful_metrics]) if successful_metrics else 0,
        }
        
        # Add throughput metrics for upload operations
        upload_metrics = [m for m in successful_metrics if m.operation.startswith("upload") and m.throughput]
        if upload_metrics:
            summary["average_throughput_mbps"] = statistics.mean([m.throughput / (1024 * 1024) for m in upload_metrics])
            summary["max_throughput_mbps"] = max([m.throughput / (1024 * 1024) for m in upload_metrics])
        
        return summary


class TAILoadTester:
    """Load testing utilities for TAI system"""
    
    def __init__(self, client: TAIClient):
        self.client = client
        self.results: List[Dict[str, Any]] = []
    
    def simulate_user_workload(self, user_id: int, config: LoadTestConfig) -> List[Dict[str, Any]]:
        """Simulate a single user workload"""
        user_results = []
        test_data = TAITestDataGenerator.generate_3mb_text()
        
        for operation in range(config.operations_per_user):
            start_time = time.time()
            success = False
            error = None
            
            try:
                # Upload file
                upload_result = self.client.upload_file(test_data)
                doc_id = upload_result["doc_id"]
                
                # Retrieve file
                retrieved_content = self.client.retrieve_file(doc_id)
                
                # Verify integrity
                if retrieved_content.decode() == test_data.content:
                    success = True
                else:
                    error = "Content mismatch"
                    
            except Exception as e:
                error = str(e)
            
            duration = time.time() - start_time
            
            user_results.append({
                "user_id": user_id,
                "operation": operation,
                "duration": duration,
                "success": success,
                "error": error,
                "timestamp": time.time()
            })
        
        return user_results
    
    def run_load_test(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Run load test with specified configuration"""
        print(f"Starting load test: {config.concurrent_users} users, {config.operations_per_user} ops/user")
        
        start_time = time.time()
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
            # Submit user workloads
            futures = []
            for user_id in range(config.concurrent_users):
                future = executor.submit(self.simulate_user_workload, user_id, config)
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                user_results = future.result()
                all_results.extend(user_results)
        
        total_duration = time.time() - start_time
        
        # Calculate statistics
        successful_ops = [r for r in all_results if r["success"]]
        failed_ops = [r for r in all_results if not r["success"]]
        
        if successful_ops:
            durations = [r["duration"] for r in successful_ops]
            summary = {
                "total_operations": len(all_results),
                "successful_operations": len(successful_ops),
                "failed_operations": len(failed_ops),
                "success_rate": len(successful_ops) / len(all_results),
                "total_duration": total_duration,
                "operations_per_second": len(all_results) / total_duration,
                "average_response_time": statistics.mean(durations),
                "median_response_time": statistics.median(durations),
                "p95_response_time": np.percentile(durations, 95),
                "p99_response_time": np.percentile(durations, 99),
                "min_response_time": min(durations),
                "max_response_time": max(durations),
                "concurrent_users": config.concurrent_users,
                "operations_per_user": config.operations_per_user
            }
        else:
            summary = {
                "total_operations": len(all_results),
                "successful_operations": 0,
                "failed_operations": len(failed_ops),
                "success_rate": 0,
                "total_duration": total_duration,
                "operations_per_second": 0,
                "error": "All operations failed"
            }
        
        return summary


class TestTAIPerformance:
    """Test suite for TAI system performance"""
    
    @pytest.fixture
    def client(self):
        """TAI client fixture"""
        return TAIClient()
    
    @pytest.fixture
    def performance_tester(self, client):
        """Performance tester fixture"""
        return TAIPerformanceTester(client)
    
    @pytest.fixture
    def load_tester(self, client):
        """Load tester fixture"""
        return TAILoadTester(client)
    
    def test_upload_performance_benchmarks(self, performance_tester):
        """Test upload performance benchmarks"""
        with tai_system():
            time.sleep(10)
            
            # Test different file sizes
            test_sizes = [1, 3, 10]  # MB
            
            for size_mb in test_sizes:
                print(f"Testing {size_mb}MB upload performance...")
                
                metrics = performance_tester.test_upload_performance(size_mb, iterations=5)
                successful_metrics = [m for m in metrics if m.success]
                
                assert len(successful_metrics) > 0, f"No successful {size_mb}MB uploads"
                
                # Performance assertions
                avg_duration = statistics.mean([m.duration for m in successful_metrics])
                avg_throughput = statistics.mean([m.throughput / (1024 * 1024) for m in successful_metrics if m.throughput])
                
                print(f"{size_mb}MB upload: {avg_duration:.2f}s avg, {avg_throughput:.2f} MB/s avg")
                
                # Assertions based on size
                if size_mb == 3:
                    assert avg_duration < 30, f"3MB upload too slow: {avg_duration:.2f}s"
                    assert avg_throughput > 0.1, f"3MB upload throughput too low: {avg_throughput:.2f} MB/s"
    
    def test_retrieval_performance_benchmarks(self, performance_tester):
        """Test retrieval performance benchmarks"""
        with tai_system():
            time.sleep(10)
            
            # Upload a test file first
            test_data = TAITestDataGenerator.generate_3mb_text()
            upload_result = performance_tester.client.upload_file(test_data)
            doc_id = upload_result["doc_id"]
            
            # Test retrieval performance
            metrics = performance_tester.test_retrieval_performance(doc_id, iterations=10)
            successful_metrics = [m for m in metrics if m.success]
            
            assert len(successful_metrics) > 0, "No successful retrievals"
            
            avg_duration = statistics.mean([m.duration for m in successful_metrics])
            print(f"3MB retrieval: {avg_duration:.2f}s avg")
            
            # Should complete within 5 seconds
            assert avg_duration < 5, f"3MB retrieval too slow: {avg_duration:.2f}s"
    
    def test_search_performance_benchmarks(self, performance_tester):
        """Test search performance benchmarks"""
        with tai_system():
            time.sleep(10)
            
            # Upload some test files first
            for i in range(5):
                test_data = TAITestDataGenerator.generate_3mb_text()
                performance_tester.client.upload_file(test_data)
            
            # Test search performance
            metrics = performance_tester.test_search_performance("test document", iterations=10)
            successful_metrics = [m for m in metrics if m.success]
            
            assert len(successful_metrics) > 0, "No successful searches"
            
            avg_duration = statistics.mean([m.duration for m in successful_metrics])
            print(f"Search: {avg_duration:.2f}s avg")
            
            # Should complete within 1 second
            assert avg_duration < 1, f"Search too slow: {avg_duration:.2f}s"
    
    def test_concurrent_upload_load(self, load_tester):
        """Test concurrent upload load"""
        with tai_system():
            time.sleep(10)
            
            # Light load test
            config = LoadTestConfig(
                concurrent_users=5,
                operations_per_user=3,
                test_duration=60,
                ramp_up_time=10,
                data_size=1024 * 1024  # 1MB
            )
            
            results = load_tester.run_load_test(config)
            
            print(f"Load test results: {results}")
            
            # Assertions
            assert results["success_rate"] > 0.9, f"Success rate too low: {results['success_rate']:.2%}"
            assert results["operations_per_second"] > 1, f"Throughput too low: {results['operations_per_second']:.2f} ops/s"
            assert results["average_response_time"] < 30, f"Response time too high: {results['average_response_time']:.2f}s"
    
    def test_stress_testing(self, load_tester):
        """Test system under stress"""
        with tai_system():
            time.sleep(10)
            
            # Stress test with higher load
            config = LoadTestConfig(
                concurrent_users=10,
                operations_per_user=5,
                test_duration=120,
                ramp_up_time=20,
                data_size=3 * 1024 * 1024  # 3MB
            )
            
            results = load_tester.run_load_test(config)
            
            print(f"Stress test results: {results}")
            
            # Stress test should maintain some level of performance
            assert results["success_rate"] > 0.8, f"Stress test success rate too low: {results['success_rate']:.2%}"
            assert results["operations_per_second"] > 0.5, f"Stress test throughput too low: {results['operations_per_second']:.2f} ops/s"
    
    def test_memory_usage_under_load(self, client):
        """Test memory usage under load"""
        with tai_system():
            time.sleep(10)
            
            # Upload multiple large files
            doc_ids = []
            for i in range(10):
                test_data = TAITestDataGenerator.generate_3mb_text()
                upload_result = client.upload_file(test_data)
                doc_ids.append(upload_result["doc_id"])
            
            # Check memory state
            memory_state = client.get_memory_state()
            
            # Verify system is still responsive
            assert "documents" in memory_state, "Memory state missing after load"
            assert len(memory_state["documents"]) >= 10, "Not all documents in memory state"
            
            # Test retrieval is still fast
            start_time = time.time()
            retrieved_content = client.retrieve_file(doc_ids[0])
            retrieval_time = time.time() - start_time
            
            assert retrieval_time < 5, f"Retrieval too slow under load: {retrieval_time:.2f}s"
            assert retrieved_content.decode() == TAITestDataGenerator.generate_3mb_text().content, "Content corrupted under load"
    
    def test_performance_degradation_analysis(self, performance_tester):
        """Test performance degradation over time"""
        with tai_system():
            time.sleep(10)
            
            # Run multiple rounds of performance tests
            rounds = 5
            round_results = []
            
            for round_num in range(rounds):
                print(f"Performance round {round_num + 1}/{rounds}")
                
                # Test upload performance
                metrics = performance_tester.test_upload_performance(1, iterations=3)
                successful_metrics = [m for m in metrics if m.success]
                
                if successful_metrics:
                    avg_duration = statistics.mean([m.duration for m in successful_metrics])
                    round_results.append(avg_duration)
                else:
                    round_results.append(float('inf'))
                
                # Small delay between rounds
                time.sleep(2)
            
            # Analyze performance degradation
            if len(round_results) > 1:
                first_round = round_results[0]
                last_round = round_results[-1]
                degradation = (last_round - first_round) / first_round
                
                print(f"Performance degradation: {degradation:.2%}")
                
                # Should not degrade more than 50%
                assert degradation < 0.5, f"Performance degraded too much: {degradation:.2%}"
    
    def test_system_recovery_under_load(self, client):
        """Test system recovery under load"""
        with tai_system() as manager:
            time.sleep(10)
            
            # Start load
            def continuous_load():
                for _ in range(20):
                    try:
                        test_data = TAITestDataGenerator.generate_3mb_text()
                        upload_result = client.upload_file(test_data)
                        client.retrieve_file(upload_result["doc_id"])
                    except:
                        pass
                    time.sleep(0.5)
            
            # Start load in background
            load_thread = threading.Thread(target=continuous_load)
            load_thread.start()
            
            # Wait for load to start
            time.sleep(5)
            
            # Restart a service
            assert manager.stop_service("holographic-memory"), "Failed to stop service under load"
            time.sleep(3)
            assert manager.start_service("holographic-memory"), "Failed to restart service under load"
            time.sleep(10)
            
            # Wait for load to complete
            load_thread.join()
            
            # Verify system is still functional
            test_data = TAITestDataGenerator.generate_3mb_text()
            upload_result = client.upload_file(test_data)
            retrieved_content = client.retrieve_file(upload_result["doc_id"])
            assert retrieved_content.decode() == test_data.content, "System not functional after recovery"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
