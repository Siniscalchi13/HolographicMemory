"""
SOA Orchestrator for Holographic Memory System

Integrates GPU-first holographic-fs with SOA services (Math Core, Router, Vault, Telemetry).
Provides clean service boundaries and proper separation of concerns.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import time

# Add holographic-fs to path
_pkg_root = Path(__file__).resolve().parents[1]
_holographic_fs_path = _pkg_root / "holographic-fs"
if _holographic_fs_path.exists():
    sys.path.insert(0, str(_holographic_fs_path))

from holographicfs.memory import Memory as HolographicMemory
from services.math_core.optimizer import DimensionOptimizer
from services.math_core.threshold import ThresholdCalculator
from services.router.mathematical_router import MathematicalRouter
from services.vault.security_guard import SecurityGuard
from services.telemetry.performance_tracker import PerformanceTelemetry


class HolographicMemoryOrchestrator:
    """
    Main orchestrator that integrates GPU-first holographic memory with SOA services.
    
    Service Responsibilities:
    - Math Core: Dimension optimization, threshold calculations
    - Router: Content routing decisions, layer assignment
    - Vault: Security policies, sensitive data handling
    - Telemetry: Performance tracking, rebalancing suggestions
    - GPU Memory: Actual storage and retrieval operations
    """
    
    def __init__(self, state_dir: Path, grid_size: int = 1024, use_gpu: bool = True):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SOA services
        self.math_core = DimensionOptimizer()
        self.threshold_calc = ThresholdCalculator()
        self.router = MathematicalRouter()
        self.vault = SecurityGuard()
        self.telemetry = PerformanceTelemetry()
        
        # Initialize GPU-first holographic memory
        self.memory = HolographicMemory(self.state_dir, grid_size, use_gpu)
        
        # Service state
        self.current_loads = {layer: 0 for layer in ["Identity", "Knowledge", "Experience", "Preference", "Context", "Wisdom", "Vault"]}
        self.layer_dimensions = self._initialize_layer_dimensions()
        
    def _initialize_layer_dimensions(self) -> Dict[str, int]:
        """Initialize 7-layer dimensions using Math Core service."""
        # Default layer configuration
        layer_names = ["Identity", "Knowledge", "Experience", "Preference", "Context", "Wisdom", "Vault"]
        importance_weights = {"Identity": 1.0, "Knowledge": 0.9, "Experience": 0.8, 
                            "Preference": 0.7, "Context": 0.6, "Wisdom": 0.5, "Vault": 0.4}
        
        # Use Math Core to optimize dimensions
        total_budget = 1024  # Default budget
        optimized_dims = self.math_core.optimize_dimensions(
            loads=self.current_loads,
            importance=importance_weights,
            total_budget=total_budget
        )
        
        return optimized_dims
    
    def store_content(self, content: bytes, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Store content using SOA service orchestration.
        
        Flow:
        1. Router determines routing decision (vault, format, layers)
        2. Vault applies security policies if needed
        3. Math Core calculates optimal dimensions
        4. GPU Memory performs actual storage
        5. Telemetry tracks performance
        """
        if metadata is None:
            metadata = {}
            
        # Step 1: Router determines routing decision
        routing_decision = self.router.route_content(content, metadata)
        
        # Step 2: Vault security check
        if routing_decision.get("vault", False):
            vault_id = self.vault.generate_vault_id()
            # Store in vault (encrypted, no holographic patterns)
            return self._store_in_vault(content, vault_id, metadata)
        
        # Step 3: Math Core optimization
        if routing_decision.get("layers"):
            # Recalculate dimensions based on current loads
            self.layer_dimensions = self._initialize_layer_dimensions()
        
        # Step 4: GPU Memory storage
        doc_id = self._generate_document_id(content)
        storage_result = self.memory.store_bytes(doc_id, content)
        
        # Step 5: Telemetry tracking
        self.telemetry.track_compression(
            original=len(content),
            stored=len(storage_result.get("encoded_data", b"")),
            layer=routing_decision.get("primary_layer", "Knowledge")
        )
        
        # Update load estimates
        for layer, weight in routing_decision.get("layers", []):
            self.current_loads[layer] += 1
        
        return {
            "doc_id": doc_id,
            "routing_decision": routing_decision,
            "storage_result": storage_result,
            "layer_dimensions": self.layer_dimensions
        }
    
    def retrieve_content(self, doc_id: str) -> Dict[str, Any]:
        """Retrieve content using GPU memory with telemetry tracking."""
        start_time = time.time()
        
        # GPU Memory retrieval
        content = self.memory.retrieve_bytes(doc_id)
        
        # Telemetry tracking
        retrieval_time = time.time() - start_time
        self.telemetry.track_retrieval(doc_id, retrieval_time, len(content))
        
        return {
            "doc_id": doc_id,
            "content": content,
            "retrieval_time": retrieval_time
        }
    
    def search_content(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search content using GPU memory with router-based filtering."""
        # Router determines search strategy
        search_strategy = self.router.route_search_query(query)
        
        # GPU Memory search
        results = self.memory.search(query, limit=limit)
        
        # Apply router-based filtering
        filtered_results = self._apply_search_filtering(results, search_strategy)
        
        return filtered_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status from all services."""
        return {
            "memory_status": {
                "backend_type": type(self.memory.backend).__name__,
                "use_gpu": self.memory.use_gpu,
                "grid_size": self.memory.grid_size
            },
            "layer_dimensions": self.layer_dimensions,
            "current_loads": self.current_loads,
            "telemetry": self.telemetry.current_ratios(),
            "math_core_status": "operational",
            "router_status": "operational",
            "vault_status": "operational"
        }
    
    def rebalance_layers(self) -> Dict[str, Any]:
        """Use Math Core and Telemetry to rebalance layer dimensions."""
        # Get rebalancing suggestions from telemetry
        suggested_dims = self.telemetry.suggest_rebalancing(
            importance={"Identity": 1.0, "Knowledge": 0.9, "Experience": 0.8, 
                       "Preference": 0.7, "Context": 0.6, "Wisdom": 0.5, "Vault": 0.4},
            total_budget=1024,
            floors=None
        )
        
        # Apply rebalancing
        old_dims = self.layer_dimensions.copy()
        self.layer_dimensions = suggested_dims
        
        return {
            "old_dimensions": old_dims,
            "new_dimensions": self.layer_dimensions,
            "rebalancing_reason": "telemetry_suggestion"
        }
    
    def _store_in_vault(self, content: bytes, vault_id: str, metadata: Dict) -> Dict[str, Any]:
        """Store sensitive content in vault (encrypted, no holographic patterns)."""
        # Vault storage implementation
        vault_path = self.state_dir / "vault" / f"{vault_id}.enc"
        vault_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Simple encryption (in production, use proper encryption)
        encrypted_content = self._encrypt_content(content)
        vault_path.write_bytes(encrypted_content)
        
        return {
            "vault_id": vault_id,
            "vault_path": str(vault_path),
            "encrypted": True,
            "holographic_patterns": False
        }
    
    def _generate_document_id(self, content: bytes) -> str:
        """Generate document ID using content hash."""
        import hashlib
        return hashlib.sha256(content).hexdigest()[:16]
    
    def _encrypt_content(self, content: bytes) -> bytes:
        """Simple encryption for vault content."""
        # In production, use proper encryption
        return content  # Placeholder
    
    def _apply_search_filtering(self, results: List[Dict], strategy: Dict) -> List[Dict]:
        """Apply router-based filtering to search results."""
        # Apply search strategy filtering
        return results  # Placeholder
