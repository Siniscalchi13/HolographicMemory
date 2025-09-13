import os
import sys
from pathlib import Path
import importlib


def test_no_cpu_imports_in_prod_profile(monkeypatch):
    # Enforce prod-like profile and no-CPU flag
    monkeypatch.setenv("HOLO_BUILD_PROFILE", "prod")
    monkeypatch.setenv("HLOG_NO_CPU", "1")
    # Ensure a clean import of memory module
    sys.modules.pop("holographicfs.memory", None)
    sys.modules.pop("holographic_cpp", None)
    # Ensure package path is available without installation
    pkg_root = Path(__file__).resolve().parents[2] / "core"
    sys.path.insert(0, str(pkg_root))
    from holographicfs import memory as mem

    # CPU module should not be imported under these flags
    assert "holographic_cpp" not in sys.modules

    # Memory should prefer GPU backend if available
    m = mem.Memory(mem.Path("/tmp/.holofs-test"), grid_size=32, use_gpu=True)
    # In prod/no-CPU mode, either GPU is active, or CPU fallback is disabled
    if m.use_gpu:
        # When GPU is usable, ensure the wrapper exists
        assert hasattr(m, "gpu_backend") and m.gpu_backend is not None
    else:
        # If GPU unavailable in this environment, ensure no CPU fallback engaged
        assert not hasattr(m, "backend") or m.backend is None
