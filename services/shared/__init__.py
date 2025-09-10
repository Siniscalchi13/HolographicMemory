"""
Shared services package for SOA architecture.

Provides common types, interfaces, and utilities for service communication.
"""

from .types import (
    LayerType,
    StorageFormat,
    RoutingDecision,
    LayerDimensions,
    StorageResult,
    RetrievalResult,
    SearchResult,
    TelemetryMetrics,
    VaultEntry,
    IService,
    IMathCoreService,
    IRouterService,
    IVaultService,
    ITelemetryService,
    IMemoryService
)

__all__ = [
    "LayerType",
    "StorageFormat", 
    "RoutingDecision",
    "LayerDimensions",
    "StorageResult",
    "RetrievalResult",
    "SearchResult",
    "TelemetryMetrics",
    "VaultEntry",
    "IService",
    "IMathCoreService",
    "IRouterService",
    "IVaultService",
    "ITelemetryService",
    "IMemoryService"
]
