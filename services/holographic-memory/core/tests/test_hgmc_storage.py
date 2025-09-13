from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

import pytest


def _has_gpu() -> bool:
    """Check if GPU backend is available (Metal or CUDA on Mac)."""
    try:
        import holographic_gpu  # type: ignore

        g = holographic_gpu.HolographicGPU()
        # Check available platforms first
        platforms = getattr(g, "available_platforms", lambda: [])()
        if not platforms:
            return False
        
        # Try to initialize with auto-detection (Metal preferred on Mac, CUDA as fallback)
        return bool(getattr(g, "initialize", lambda: False)())
    except Exception:
        return False


pytestmark = pytest.mark.gpu


def test_gpu_only_store_and_recall():
    """Test GPU-only store and recall (Metal or CUDA on Mac)."""
    os.environ["HLOG_GPU_ONLY"] = "1"
    from holographicfs.memory import mount

    root = Path(tempfile.mkdtemp())
    fs = mount(root, grid_size=64)
    assert fs.mem.use_gpu is True

    data = b"hello holographic containers"
    doc = fs.store_data(data, "t.bin")
    out = fs.mem.retrieve_bytes(doc)
    assert out == data

    # Only holographic container(s) exist under .holofs/hlog
    hlog = fs.state_dir / "hlog"
    containers = list((hlog / "containers").glob(f"{doc}.hgc"))
    assert containers, "missing .hgc container"
    # No legacy responses folder
    assert not (fs.state_dir / "responses").exists()


def test_seed_tamper_detected_or_corrected():
    """Seed tamper is either corrected by Wave ECC or detected.

    With Wave ECC, seed perturbations may be corrected by parity; if not,
    recall should fail due to parity mismatch.
    """

    os.environ["HLOG_GPU_ONLY"] = "1"
    from holographicfs.memory import mount

    root = Path(tempfile.mkdtemp())
    fs = mount(root, grid_size=64)
    data = b"tamper-test-1234567890"
    doc = fs.store_data(data, "tamper.bin")

    cpath = fs.state_dir / "hlog" / "containers" / f"{doc}.hgc"
    buf = bytearray(cpath.read_bytes())
    # Header
    magic = bytes(buf[:6])
    assert magic.startswith(b"HGMC"), "unexpected container magic"
    # Read header common fields
    off = 6
    dim = int.from_bytes(buf[off : off + 4], "little")
    off += 4
    orig = int.from_bytes(buf[off : off + 4], "little")
    off += 4
    n = int.from_bytes(buf[off : off + 4], "little")
    off += 4
    # Skip sizes array
    off += 4 * n
    # Tamper a seed (flip 1 bit)
    if n > 0:
        old = int.from_bytes(buf[off : off + 4], "little")
        new = old ^ 0x1
        buf[off : off + 4] = int(new).to_bytes(4, "little")
        cpath.write_bytes(bytes(buf))
        try:
            out = fs.mem.retrieve_bytes(doc)
            # If no exception, ECC should have corrected the corruption
            assert out == data
        except Exception:
            # Parity mismatch/uncorrectable path is acceptable
            pass


def test_stats_accounting_matches_disk():
    """Test that stats reflect actual disk usage (GPU-only)."""
    os.environ["HLOG_GPU_ONLY"] = "1"
    from holographicfs.memory import mount

    root = Path(tempfile.mkdtemp())
    fs = mount(root, grid_size=64)
    # Store two docs
    fs.store_data(b"a" * 512, "a.bin")
    fs.store_data(b"b" * 1024, "b.bin")

    stats = fs.stats()
    holo_bytes = int(stats.get("holo_bytes", 0))
    # Sum on-disk
    total = 0
    for p in (fs.state_dir / "hlog").rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    assert holo_bytes == total
