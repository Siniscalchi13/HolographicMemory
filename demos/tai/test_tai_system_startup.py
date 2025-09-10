#!/usr/bin/env python3
"""
TAI System Startup Validation Tests

This module provides comprehensive testing for the complete TAI system startup,
including all services, dependencies, and inter-service communication.

Enterprise-grade validation ensures production readiness.
"""

import os
import sys
import time
import requests
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager

import pytest


@dataclass
class TAIService:
    """TAI service configuration"""
    name: str
    port: int
    health_endpoint: str
    startup_timeout: int = 30
    health_timeout: int = 5


# TAI System Services Configuration
TAI_SERVICES = [
    TAIService("aicp-orchestrator", 8000, "/health"),
    TAIService("quantum-core", 8001, "/health"),
    TAIService("holographic-memory", 8002, "/health"),
    TAIService("verbum-field-engine", 8081, "/health"),
    TAIService("web-interface", 8501, "/healthz"),
    TAIService("enterprise-dashboard", 8502, "/health"),
]

# Service startup order (dependencies first)
STARTUP_ORDER = [
    "quantum-core",
    "holographic-memory", 
    "verbum-field-engine",
    "aicp-orchestrator",
    "web-interface",
    "enterprise-dashboard",
]


class TAISystemManager:
    """Manages TAI system lifecycle for testing"""
    
    def __init__(self):
        self.services: Dict[str, TAIService] = {
            service.name: service for service in TAI_SERVICES
        }
        self.running_services: Dict[str, subprocess.Popen] = {}
        self.base_url = "http://localhost"
        
    def start_service(self, service_name: str) -> bool:
        """Start a single TAI service"""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
            
        service = self.services[service_name]
        
        # Check if already running
        if self.is_service_healthy(service_name):
            print(f"✅ {service_name} already running and healthy")
            return True
            
        print(f"🚀 Starting {service_name} on port {service.port}...")
        
        # Start service using docker-compose
        try:
            cmd = ["docker-compose", "up", "-d", service_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"❌ Failed to start {service_name}: {result.stderr}")
                return False
                
            # Wait for service to be healthy
            if self.wait_for_service_health(service_name, service.startup_timeout):
                print(f"✅ {service_name} started successfully")
                return True
            else:
                print(f"❌ {service_name} failed health check")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌ {service_name} startup timed out")
            return False
        except Exception as e:
            print(f"❌ Error starting {service_name}: {e}")
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """Stop a single TAI service"""
        print(f"🛑 Stopping {service_name}...")
        
        try:
            cmd = ["docker-compose", "stop", service_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✅ {service_name} stopped successfully")
                return True
            else:
                print(f"❌ Failed to stop {service_name}: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error stopping {service_name}: {e}")
            return False
    
    def start_all_services(self) -> bool:
        """Start all TAI services in dependency order"""
        print("🚀 Starting complete TAI system...")
        
        for service_name in STARTUP_ORDER:
            if not self.start_service(service_name):
                print(f"❌ Failed to start TAI system - {service_name} failed")
                return False
                
        print("✅ All TAI services started successfully")
        return True
    
    def stop_all_services(self) -> bool:
        """Stop all TAI services"""
        print("🛑 Stopping complete TAI system...")
        
        success = True
        # Stop in reverse order
        for service_name in reversed(STARTUP_ORDER):
            if not self.stop_service(service_name):
                success = False
                
        if success:
            print("✅ All TAI services stopped successfully")
        else:
            print("❌ Some services failed to stop")
            
        return success
    
    def is_service_healthy(self, service_name: str) -> bool:
        """Check if a service is healthy"""
        if service_name not in self.services:
            return False
            
        service = self.services[service_name]
        url = f"{self.base_url}:{service.port}{service.health_endpoint}"
        
        try:
            response = requests.get(url, timeout=service.health_timeout)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def wait_for_service_health(self, service_name: str, timeout: int = 30) -> bool:
        """Wait for a service to become healthy"""
        service = self.services[service_name]
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_service_healthy(service_name):
                return True
            time.sleep(1)
            
        return False
    
    def get_service_status(self) -> Dict[str, Dict]:
        """Get status of all services"""
        status = {}
        
        for service_name, service in self.services.items():
            is_healthy = self.is_service_healthy(service_name)
            status[service_name] = {
                "port": service.port,
                "healthy": is_healthy,
                "url": f"{self.base_url}:{service.port}{service.health_endpoint}"
            }
            
        return status
    
    def test_inter_service_communication(self) -> Dict[str, bool]:
        """Test communication between services"""
        results = {}
        
        # Test Orchestrator -> Quantum Core
        try:
            response = requests.get("http://localhost:8000/quantum/status", timeout=5)
            results["orchestrator_to_quantum"] = response.status_code == 200
        except:
            results["orchestrator_to_quantum"] = False
            
        # Test Orchestrator -> Holographic Memory
        try:
            response = requests.get("http://localhost:8000/memory/status", timeout=5)
            results["orchestrator_to_memory"] = response.status_code == 200
        except:
            results["orchestrator_to_memory"] = False
            
        # Test Orchestrator -> Verbum Field Engine
        try:
            response = requests.get("http://localhost:8000/models/status", timeout=5)
            results["orchestrator_to_vfe"] = response.status_code == 200
        except:
            results["orchestrator_to_vfe"] = False
            
        # Test Web Interface -> Orchestrator
        try:
            response = requests.get("http://localhost:8501/healthz", timeout=5)
            results["web_interface_healthy"] = response.status_code == 200
        except:
            results["web_interface_healthy"] = False
            
        return results


@contextmanager
def tai_system():
    """Context manager for TAI system lifecycle"""
    manager = TAISystemManager()
    
    try:
        if not manager.start_all_services():
            raise RuntimeError("Failed to start TAI system")
        yield manager
    finally:
        manager.stop_all_services()


class TestTAISystemStartup:
    """Test suite for TAI system startup validation"""
    
    def test_individual_service_startup(self):
        """Test each service can start individually"""
        manager = TAISystemManager()
        
        for service_name in TAI_SERVICES:
            with pytest.raises(Exception, match="Failed to start"):
                # This should not raise if service starts successfully
                if not manager.start_service(service_name):
                    raise Exception(f"Failed to start {service_name}")
                    
            # Verify service is healthy
            assert manager.is_service_healthy(service_name), f"{service_name} not healthy after startup"
            
            # Stop service
            assert manager.stop_service(service_name), f"Failed to stop {service_name}"
    
    def test_service_dependency_order(self):
        """Test services start in correct dependency order"""
        manager = TAISystemManager()
        
        # Start services in dependency order
        for service_name in STARTUP_ORDER:
            assert manager.start_service(service_name), f"Failed to start {service_name} in order"
            assert manager.is_service_healthy(service_name), f"{service_name} not healthy after startup"
    
    def test_complete_system_startup(self):
        """Test complete TAI system startup"""
        with tai_system() as manager:
            # Verify all services are running
            status = manager.get_service_status()
            
            for service_name, service_info in status.items():
                assert service_info["healthy"], f"{service_name} not healthy: {service_info}"
    
    def test_inter_service_communication(self):
        """Test communication between services"""
        with tai_system() as manager:
            # Wait for all services to be ready
            time.sleep(10)
            
            # Test inter-service communication
            comm_results = manager.test_inter_service_communication()
            
            # Verify critical communication paths
            assert comm_results["orchestrator_to_quantum"], "Orchestrator cannot communicate with Quantum Core"
            assert comm_results["orchestrator_to_memory"], "Orchestrator cannot communicate with Holographic Memory"
            assert comm_results["orchestrator_to_vfe"], "Orchestrator cannot communicate with Verbum Field Engine"
            assert comm_results["web_interface_healthy"], "Web Interface not healthy"
    
    def test_health_endpoints(self):
        """Test all health check endpoints"""
        with tai_system() as manager:
            status = manager.get_service_status()
            
            for service_name, service_info in status.items():
                # Test health endpoint directly
                try:
                    response = requests.get(service_info["url"], timeout=5)
                    assert response.status_code == 200, f"{service_name} health endpoint failed: {response.status_code}"
                    
                    # Verify response contains expected data
                    if service_name == "aicp-orchestrator":
                        data = response.json()
                        assert "status" in data, "Orchestrator health response missing status"
                    elif service_name == "holographic-memory":
                        data = response.json()
                        assert "modules" in data, "Holographic Memory health response missing modules"
                        
                except requests.RequestException as e:
                    pytest.fail(f"Health endpoint test failed for {service_name}: {e}")
    
    def test_graceful_shutdown(self):
        """Test graceful shutdown of all services"""
        manager = TAISystemManager()
        
        # Start all services
        assert manager.start_all_services(), "Failed to start services for shutdown test"
        
        # Verify all are running
        status = manager.get_service_status()
        running_count = sum(1 for s in status.values() if s["healthy"])
        assert running_count == len(TAI_SERVICES), f"Expected {len(TAI_SERVICES)} services running, got {running_count}"
        
        # Stop all services
        assert manager.stop_all_services(), "Failed to stop services gracefully"
        
        # Verify all are stopped
        time.sleep(5)  # Give services time to shut down
        status = manager.get_service_status()
        running_count = sum(1 for s in status.values() if s["healthy"])
        assert running_count == 0, f"Expected 0 services running after shutdown, got {running_count}"
    
    def test_system_under_load(self):
        """Test system startup under load conditions"""
        with tai_system() as manager:
            # Simulate load by making concurrent requests
            import concurrent.futures
            
            def make_requests():
                results = []
                for _ in range(10):
                    try:
                        response = requests.get("http://localhost:8000/health", timeout=5)
                        results.append(response.status_code == 200)
                    except:
                        results.append(False)
                return results
            
            # Make concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_requests) for _ in range(5)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Verify system remains stable
            all_results = [item for sublist in results for item in sublist]
            success_rate = sum(all_results) / len(all_results)
            assert success_rate >= 0.9, f"System stability under load: {success_rate:.2%} success rate"
    
    def test_configuration_validation(self):
        """Test configuration validation across all services"""
        with tai_system() as manager:
            # Test that all services have valid configurations
            status = manager.get_service_status()
            
            for service_name, service_info in status.items():
                assert service_info["healthy"], f"{service_name} configuration invalid"
                
                # Test service-specific configuration
                if service_name == "holographic-memory":
                    # Test holographic memory specific config
                    try:
                        response = requests.get("http://localhost:8002/health", timeout=5)
                        data = response.json()
                        assert "modules" in data, "Holographic Memory missing module configuration"
                    except:
                        pytest.fail(f"Holographic Memory configuration validation failed")
                
                elif service_name == "quantum-core":
                    # Test quantum core specific config
                    try:
                        response = requests.get("http://localhost:8001/health", timeout=5)
                        data = response.json()
                        assert "status" in data, "Quantum Core missing status configuration"
                    except:
                        pytest.fail(f"Quantum Core configuration validation failed")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
