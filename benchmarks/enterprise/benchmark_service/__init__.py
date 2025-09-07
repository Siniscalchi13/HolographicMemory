"""
AIUCP Benchmark Service

Provides SOA components for benchmarking the computational holography stack.

Modules:
- contracts: Pydantic-free dataclasses for data contracts
- adapters: Integration with holographic backend and FAISS
- services: Data generation, metrics, reporting, theory
- pipelines: End-to-end workflow glue
- orchestrator: Orchestrates benchmark categories and writes reports
"""

__all__ = [
    "contracts",
    "adapters",
    "services",
    "pipelines",
    "orchestrator",
]

