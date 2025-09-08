"""
Holographic Memory - Enterprise-grade 7-layer holographic memory system.

This package provides a mathematically grounded holographic memory architecture
with proven capacity bounds, security guarantees, and sub-millisecond search.
"""

__version__ = "1.0.0"
__author__ = "SmartHaus Group"
__email__ = "dev@smarthaus.ai"

from .memory import HolographicMemory
from .vault import SecureVault
from .router import SemanticRouter

__all__ = [
    "HolographicMemory",
    "SecureVault", 
    "SemanticRouter",
    "__version__",
]
