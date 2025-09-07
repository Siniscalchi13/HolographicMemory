#!/usr/bin/env python3
"""
TAI System End-to-End Integration Tests

This module provides comprehensive end-to-end testing for the complete TAI system,
including all services, workflows, and user journeys.

Enterprise-grade validation ensures production readiness.
"""

import os
import sys
import time
import requests
import hashlib
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager

import pytest
import numpy as np

# Import the system manager from startup tests
from test_tai_system_startup import tai_system, TAISystemManager


@dataclass
class TestData:
    """Test data for E2E testing"""
    content: str
    filename: str
    size: int
    hash: str


class TAITestDataGenerator:
    """Generates test data for TAI system testing"""
    
    @staticmethod
    def generate_3mb_text() -> TestData:
        """Generate 3MB of text data"""
        content = "This is a test document for TAI holographic memory system. " * 50000
        content = content[:3 * 1024 * 1024]  # Exactly 3MB
        
        return TestData(
            content=content,
            filename="test_3mb.txt",
            size=len(content),
            hash=hashlib.sha256(content.encode()).hexdigest()
        )
    
    @staticmethod
    def generate_binary_data() -> TestData:
        """Generate binary test data"""
        content = os.urandom(1024 * 1024)  # 1MB of random data
        
        return TestData(
            content=content,
            filename="test_binary.bin",
            size=len(content),
            hash=hashlib.sha256(content).hexdigest()
        )
    
    @staticmethod
    def generate_json_data() -> TestData:
        """Generate JSON test data"""
        data = {
            "test_id": "e2e_integration_test",
            "timestamp": time.time(),
            "data": [{"id": i, "value": f"test_value_{i}"} for i in range(1000)],
            "metadata": {
                "system": "TAI",
                "version": "1.0.0",
                "test_type": "e2e_integration"
            }
        }
        content = json.dumps(data, indent=2)
        
        return TestData(
            content=content,
            filename="test_data.json",
            size=len(content),
            hash=hashlib.sha256(content.encode()).hexdigest()
        )


class TAIClient:
    """Client for interacting with TAI system services"""
    
    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url
        self.orchestrator_url = f"{base_url}:8000"
        self.quantum_url = f"{base_url}:8001"
        self.memory_url = f"{base_url}:8002"
        self.vfe_url = f"{base_url}:8081"
        self.web_url = f"{base_url}:8501"
    
    def upload_file(self, test_data: TestData) -> Dict[str, Any]:
        """Upload file through orchestrator"""
        files = {
            'file': (test_data.filename, test_data.content, 'application/octet-stream')
        }
        
        response = requests.post(
            f"{self.orchestrator_url}/files/upload",
            files=files,
            timeout=60
        )
        
        assert response.status_code == 200, f"Upload failed: {response.status_code} - {response.text}"
        return response.json()
    
    def retrieve_file(self, doc_id: str) -> bytes:
        """Retrieve file through orchestrator"""
        response = requests.get(
            f"{self.orchestrator_url}/files/{doc_id}",
            timeout=30
        )
        
        assert response.status_code == 200, f"Retrieve failed: {response.status_code} - {response.text}"
        return response.content
    
    def search_files(self, query: str) -> List[Dict[str, Any]]:
        """Search files through orchestrator"""
        response = requests.post(
            f"{self.orchestrator_url}/files/search",
            json={"query": query},
            timeout=30
        )
        
        assert response.status_code == 200, f"Search failed: {response.status_code} - {response.text}"
        return response.json()
    
    def get_memory_state(self) -> Dict[str, Any]:
        """Get holographic memory state"""
        response = requests.get(
            f"{self.memory_url}/memory-state",
            timeout=10
        )
        
        assert response.status_code == 200, f"Memory state failed: {response.status_code} - {response.text}"
        return response.json()
    
    def test_quantum_operation(self) -> Dict[str, Any]:
        """Test quantum core operation"""
        response = requests.post(
            f"{self.quantum_url}/quantum/operation",
            json={"operation": "test", "data": [1, 2, 3, 4]},
            timeout=30
        )
        
        assert response.status_code == 200, f"Quantum operation failed: {response.status_code} - {response.text}"
        return response.json()
    
    def test_model_inference(self, prompt: str) -> Dict[str, Any]:
        """Test model inference through VFE"""
        response = requests.post(
            f"{self.vfe_url}/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100
            },
            timeout=30
        )
        
        assert response.status_code == 200, f"Model inference failed: {response.status_code} - {response.text}"
        return response.json()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health from orchestrator"""
        response = requests.get(
            f"{self.orchestrator_url}/health",
            timeout=10
        )
        
        assert response.status_code == 200, f"Health check failed: {response.status_code} - {response.text}"
        return response.json()


class TestTAIE2EIntegration:
    """Test suite for TAI system end-to-end integration"""
    
    @pytest.fixture
    def client(self):
        """TAI client fixture"""
        return TAIClient()
    
    @pytest.fixture
    def test_data(self):
        """Test data fixtures"""
        return {
            "text": TAITestDataGenerator.generate_3mb_text(),
            "binary": TAITestDataGenerator.generate_binary_data(),
            "json": TAITestDataGenerator.generate_json_data()
        }
    
    def test_complete_file_workflow(self, client, test_data):
        """Test complete file upload, storage, and retrieval workflow"""
        with tai_system():
            # Wait for system to be ready
            time.sleep(10)
            
            # Test with different file types
            for data_type, test_data in test_data.items():
                print(f"Testing {data_type} file workflow...")
                
                # Upload file
                upload_result = client.upload_file(test_data)
                assert "doc_id" in upload_result, f"Upload result missing doc_id: {upload_result}"
                doc_id = upload_result["doc_id"]
                
                # Verify file was stored in holographic memory
                memory_state = client.get_memory_state()
                assert "documents" in memory_state, "Memory state missing documents"
                
                # Retrieve file
                retrieved_content = client.retrieve_file(doc_id)
                
                # Verify perfect reconstruction
                if isinstance(test_data.content, str):
                    assert retrieved_content.decode() == test_data.content, f"{data_type} content mismatch"
                else:
                    assert retrieved_content == test_data.content, f"{data_type} binary content mismatch"
                
                # Verify hash matches
                retrieved_hash = hashlib.sha256(retrieved_content).hexdigest()
                assert retrieved_hash == test_data.hash, f"{data_type} hash mismatch"
                
                print(f"✅ {data_type} file workflow completed successfully")
    
    def test_holographic_memory_persistence(self, client, test_data):
        """Test holographic memory persistence and wave patterns"""
        with tai_system():
            time.sleep(10)
            
            # Upload multiple files
            doc_ids = []
            for data_type, test_data in test_data.items():
                upload_result = client.upload_file(test_data)
                doc_ids.append(upload_result["doc_id"])
            
            # Check memory state
            memory_state = client.get_memory_state()
            assert len(memory_state.get("documents", [])) >= len(test_data), "Not all documents in memory state"
            
            # Verify wave persistence files exist (if accessible)
            # This would require access to the file system where wave files are stored
            # For now, we verify through the API that persistence is working
            
            # Test interference patterns
            for doc_id in doc_ids:
                # This would test the 3D interference pattern generation
                # Implementation depends on available API endpoints
                pass
    
    def test_quantum_processing_integration(self, client):
        """Test quantum processing integration"""
        with tai_system():
            time.sleep(10)
            
            # Test quantum operation
            quantum_result = client.test_quantum_operation()
            assert "result" in quantum_result, "Quantum operation missing result"
            
            # Verify quantum metrics
            assert "quantum_metrics" in quantum_result, "Quantum operation missing metrics"
    
    def test_model_inference_integration(self, client):
        """Test model inference integration"""
        with tai_system():
            time.sleep(10)
            
            # Test model inference
            inference_result = client.test_model_inference("Test prompt for TAI system")
            assert "choices" in inference_result, "Model inference missing choices"
            
            # Verify response structure
            assert len(inference_result["choices"]) > 0, "Model inference returned no choices"
            assert "message" in inference_result["choices"][0], "Model inference missing message"
    
    def test_search_and_retrieval(self, client, test_data):
        """Test search and retrieval functionality"""
        with tai_system():
            time.sleep(10)
            
            # Upload files
            doc_ids = []
            for data_type, test_data in test_data.items():
                upload_result = client.upload_file(test_data)
                doc_ids.append(upload_result["doc_id"])
            
            # Test search functionality
            search_results = client.search_files("test document")
            assert isinstance(search_results, list), "Search results not a list"
            
            # Test semantic search
            semantic_results = client.search_files("holographic memory system")
            assert isinstance(semantic_results, list), "Semantic search results not a list"
    
    def test_system_health_monitoring(self, client):
        """Test system health monitoring"""
        with tai_system():
            time.sleep(10)
            
            # Get system health
            health = client.get_system_health()
            assert "status" in health, "Health response missing status"
            assert "services" in health, "Health response missing services"
            
            # Verify all services are healthy
            for service_name, service_health in health["services"].items():
                assert service_health["healthy"], f"Service {service_name} not healthy: {service_health}"
    
    def test_concurrent_operations(self, client, test_data):
        """Test concurrent operations"""
        with tai_system():
            time.sleep(10)
            
            import concurrent.futures
            import threading
            
            results = []
            errors = []
            
            def upload_and_retrieve(data_type, test_data):
                try:
                    # Upload
                    upload_result = client.upload_file(test_data)
                    doc_id = upload_result["doc_id"]
                    
                    # Retrieve
                    retrieved_content = client.retrieve_file(doc_id)
                    
                    # Verify
                    if isinstance(test_data.content, str):
                        success = retrieved_content.decode() == test_data.content
                    else:
                        success = retrieved_content == test_data.content
                    
                    results.append(success)
                except Exception as e:
                    errors.append(str(e))
            
            # Run concurrent operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for data_type, test_data in test_data.items():
                    for _ in range(3):  # 3 operations per data type
                        future = executor.submit(upload_and_retrieve, data_type, test_data)
                        futures.append(future)
                
                # Wait for completion
                concurrent.futures.wait(futures)
            
            # Verify results
            assert len(errors) == 0, f"Concurrent operations failed: {errors}"
            assert all(results), "Some concurrent operations failed"
            assert len(results) == len(test_data) * 3, "Not all concurrent operations completed"
    
    def test_error_handling(self, client):
        """Test error handling and recovery"""
        with tai_system():
            time.sleep(10)
            
            # Test invalid file upload
            with pytest.raises(AssertionError):
                client.upload_file(TestData("", "empty.txt", 0, ""))
            
            # Test invalid doc_id retrieval
            with pytest.raises(AssertionError):
                client.retrieve_file("invalid_doc_id")
            
            # Test invalid search query
            with pytest.raises(AssertionError):
                client.search_files("")
    
    def test_performance_benchmarks(self, client, test_data):
        """Test performance benchmarks"""
        with tai_system():
            time.sleep(10)
            
            # Test 3MB file upload performance
            start_time = time.time()
            upload_result = client.upload_file(test_data["text"])
            upload_time = time.time() - start_time
            
            # Should complete within 30 seconds
            assert upload_time < 30, f"3MB upload took {upload_time:.2f}s, expected < 30s"
            
            # Test retrieval performance
            start_time = time.time()
            retrieved_content = client.retrieve_file(upload_result["doc_id"])
            retrieval_time = time.time() - start_time
            
            # Should complete within 5 seconds
            assert retrieval_time < 5, f"3MB retrieval took {retrieval_time:.2f}s, expected < 5s"
            
            # Verify perfect reconstruction
            assert retrieved_content.decode() == test_data["text"].content, "Perfect reconstruction failed"
    
    def test_system_recovery(self, client, test_data):
        """Test system recovery after service restart"""
        with tai_system() as manager:
            time.sleep(10)
            
            # Upload file
            upload_result = client.upload_file(test_data["text"])
            doc_id = upload_result["doc_id"]
            
            # Restart holographic memory service
            assert manager.stop_service("holographic-memory"), "Failed to stop holographic memory"
            time.sleep(5)
            assert manager.start_service("holographic-memory"), "Failed to restart holographic memory"
            time.sleep(10)
            
            # Verify file is still retrievable
            retrieved_content = client.retrieve_file(doc_id)
            assert retrieved_content.decode() == test_data["text"].content, "File not recoverable after service restart"
    
    def test_data_integrity_validation(self, client, test_data):
        """Test data integrity across all operations"""
        with tai_system():
            time.sleep(10)
            
            # Upload and verify each data type
            for data_type, test_data in test_data.items():
                # Upload
                upload_result = client.upload_file(test_data)
                doc_id = upload_result["doc_id"]
                
                # Retrieve multiple times
                for _ in range(3):
                    retrieved_content = client.retrieve_file(doc_id)
                    
                    # Verify content integrity
                    if isinstance(test_data.content, str):
                        assert retrieved_content.decode() == test_data.content, f"{data_type} content integrity failed"
                    else:
                        assert retrieved_content == test_data.content, f"{data_type} binary integrity failed"
                    
                    # Verify hash integrity
                    retrieved_hash = hashlib.sha256(retrieved_content).hexdigest()
                    assert retrieved_hash == test_data.hash, f"{data_type} hash integrity failed"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
