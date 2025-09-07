"""Integration adapters for backends (holographic core, FAISS)."""

from .holographic_backend import HoloBackend
from .faiss_index import FaissIndex

__all__ = [
    "HoloBackend",
    "FaissIndex",
]

