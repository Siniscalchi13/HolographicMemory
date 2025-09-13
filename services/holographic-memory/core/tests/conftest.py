"""
Shared pytest configuration for HolographicMemory core tests.

Responsibilities:
- Ensure imports work from source tree without installation
- Prefer local native build for holographic_gpu over site-packages
- Provide a helper to probe GPU availability
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def pytest_configure(config):
    # Add core package path
    repo_core = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_core))
    # Add native build path for holographic_gpu
    native_build = repo_core / "native" / "holographic" / "build"
    sys.path.insert(0, str(native_build))


def gpu_available() -> bool:
    try:
        import holographic_gpu as hg  # type: ignore

        plats = getattr(hg, "available_platforms", lambda: [])()
        return bool(plats)
    except Exception:
        return False

