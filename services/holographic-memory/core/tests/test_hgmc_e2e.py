from __future__ import annotations

import os
import tempfile
from pathlib import Path
import pytest


def _gpu_ready() -> bool:
    try:
        import holographic_gpu as hg  # type: ignore
        plats = hg.available_platforms()  # type: ignore[attr-defined]
        if not plats:
            return False
        g = hg.HolographicGPU()  # type: ignore[attr-defined]
        return bool(getattr(g, "initialize", lambda: False)())
    except Exception:
        return False


_force_run = os.environ.get("HLOG_FORCE_E2E", "0").lower() in ("1", "true", "on", "yes")
pytestmark = [] if _force_run else pytest.mark.skipif(not _gpu_ready(), reason="GPU backend not available")


def test_hgmc2_parity_mismatch_fails_e2e():
    """E2E: Corrupt parity blob in HGMC2 and expect ECC failure on recall."""
    os.environ["HLOG_GPU_ONLY"] = "1"
    from holographicfs.memory import mount

    root = Path(tempfile.mkdtemp())
    fs = mount(root, grid_size=64)
    data = b"ECC-e2e-test-contents-1234567890"
    doc = fs.store_data(data, "ecc.bin")

    cpath = fs.state_dir / "hlog" / "containers" / f"{doc}.hgc"
    buf = bytearray(cpath.read_bytes())
    off = 6  # skip magic
    dim = int.from_bytes(buf[off:off+4], "little"); off += 4
    orig = int.from_bytes(buf[off:off+4], "little"); off += 4
    n = int.from_bytes(buf[off:off+4], "little"); off += 4
    off += 4 * n  # sizes
    off += 4 * n  # seeds
    ecc_scheme = int.from_bytes(buf[off:off+4], "little"); off += 4
    ecc_k = int.from_bytes(buf[off:off+4], "little"); off += 4
    ecc_r = int.from_bytes(buf[off:off+4], "little"); off += 4
    # Accept Wave ECC (2) header semantics
    if ecc_scheme == 1:
        # Legacy RS header is no longer produced by active code
        assert False, "legacy RS scheme detected unexpectedly"
    elif ecc_scheme == 2:
        # Wave ECC: ecc_k=redundancy_level (>0), ecc_r=seed_base (>=0)
        assert ecc_k > 0 and ecc_r >= 0
    else:
        raise AssertionError(f"Unsupported ECC scheme: {ecc_scheme}")
    # Corrupt first parity blob if present
    if n > 0:
        plen = int.from_bytes(buf[off:off+4], "little")
        off += 4
        if plen > 0:
            # flip a bit in the first parity byte
            buf[off] ^= 0x01
            cpath.write_bytes(bytes(buf))
            with pytest.raises(Exception):
                _ = fs.mem.retrieve_bytes(doc)
