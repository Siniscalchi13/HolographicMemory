"""
Shared types and interfaces for SOA services.

Defines common data structures and service contracts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json


class LayerType(Enum):
    """7-Layer holographic memory layer types."""
    IDENTITY = "Identity"
    KNOWLEDGE = "Knowledge"
    EXPERIENCE = "Experience"
    PREFERENCE = "Preference"
    CONTEXT = "Context"
    WISDOM = "Wisdom"
    VAULT = "Vault"


class StorageFormat(Enum):
    """Storage format types."""
    MICRO = "micro"
    MICROK8 = "microK8"
    V4 = "v4"
    VAULT = "vault"


@dataclass
class RoutingDecision:
    """Routing decision from Router service."""
    vault: bool
    format: StorageFormat
    layers: List[tuple[str, float]]  # (layer_name, weight)
    k: int  # Top-K for sparse path
    primary_layer: str
    security_level: int = 0


@dataclass
class LayerDimensions:
    """Layer dimension configuration from Math Core."""
    dimensions: Dict[str, int]
    total_budget: int
    importance_weights: Dict[str, float]
    target_snr: Dict[str, float]


@dataclass
class StorageResult:
    """Storage result from GPU Memory."""
    doc_id: str
    encoded_data: bytes
    layer_assignments: Dict[str, float]
    compression_ratio: float
    storage_time: float


@dataclass
class RetrievalResult:
    """Retrieval result from GPU Memory."""
    doc_id: str
    content: bytes
    retrieval_time: float
    accuracy_score: float


@dataclass
class SearchResult:
    """Search result from GPU Memory."""
    doc_id: str
    relevance_score: float
    snippet: str
    metadata: Dict[str, Any]


@dataclass
class TelemetryMetrics:
    """Telemetry metrics from Telemetry service."""
    compression_ratios: Dict[str, float]
    layer_loads: Dict[str, int]
    performance_metrics: Dict[str, float]
    rebalancing_suggestions: Optional[Dict[str, int]] = None


@dataclass
class VaultEntry:
    """Vault entry for sensitive data."""
    vault_id: str
    encrypted_content: bytes
    access_policies: List[str]
    audit_trail: List[Dict[str, Any]]


# Service interfaces
class IService:
    """Base interface for all SOA services."""
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        raise NotImplementedError
    
    def health_check(self) -> bool:
        """Health check."""
        raise NotImplementedError


class IMathCoreService(IService):
    """Math Core service interface."""
    
    def optimize_dimensions(
        self, 
        loads: Dict[str, int], 
        importance: Dict[str, float], 
        total_budget: int,
        floors: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]:
        """Optimize layer dimensions."""
        raise NotImplementedError
    
    def calculate_threshold(
        self, 
        c_micro: int, 
        c_v4_curve: callable, 
        c_microk8: Optional[int] = None
    ) -> int:
        """Calculate storage format threshold."""
        raise NotImplementedError


class IRouterService(IService):
    """Router service interface."""
    
    def route_content(self, content: bytes, metadata: Dict[str, Any]) -> RoutingDecision:
        """Route content to appropriate layers and format."""
        raise NotImplementedError
    
    def route_search_query(self, query: str) -> Dict[str, Any]:
        """Route search query to appropriate strategy."""
        raise NotImplementedError


class IVaultService(IService):
    """Vault service interface."""
    
    def detect_secrets(self, content: bytes) -> bool:
        """Detect sensitive content."""
        raise NotImplementedError
    
    def generate_vault_id(self) -> str:
        """Generate secure vault ID."""
        raise NotImplementedError
    
    def store_vault_entry(self, content: bytes, policies: List[str]) -> VaultEntry:
        """Store content in vault."""
        raise NotImplementedError


class ITelemetryService(IService):
    """Telemetry service interface."""
    
    def track_compression(self, original: int, stored: int, layer: str) -> None:
        """Track compression metrics."""
        raise NotImplementedError
    
    def track_retrieval(self, doc_id: str, time: float, size: int) -> None:
        """Track retrieval metrics."""
        raise NotImplementedError
    
    def get_current_metrics(self) -> TelemetryMetrics:
        """Get current telemetry metrics."""
        raise NotImplementedError
    
    def suggest_rebalancing(
        self, 
        importance: Dict[str, float], 
        total_budget: int,
        floors: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]:
        """Suggest layer rebalancing."""
        raise NotImplementedError


class IMemoryService(IService):
    """GPU Memory service interface."""
    
    def store_bytes(self, doc_id: str, content: bytes) -> StorageResult:
        """Store content bytes."""
        raise NotImplementedError
    
    def retrieve_bytes(self, doc_id: str) -> bytes:
        """Retrieve content bytes."""
        raise NotImplementedError
    
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search content."""
        raise NotImplementedError
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get memory service capabilities."""
        raise NotImplementedError
