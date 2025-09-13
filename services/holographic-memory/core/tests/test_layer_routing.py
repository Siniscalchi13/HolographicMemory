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


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="GPU backend not available")


def _read_hgmc2_header(p: Path):
    buf = p.read_bytes()
    off = 6
    dim = int.from_bytes(buf[off:off+4], 'little'); off += 4
    orig = int.from_bytes(buf[off:off+4], 'little'); off += 4
    n = int.from_bytes(buf[off:off+4], 'little'); off += 4
    sizes = [int.from_bytes(buf[off+i*4:off+i*4+4], 'little') for i in range(n)]
    off += 4*n
    seeds = [int.from_bytes(buf[off+i*4:off+i*4+4], 'little') for i in range(n)]
    off += 4*n
    ecc_scheme = int.from_bytes(buf[off:off+4], 'little'); off += 4
    ecc_k = int.from_bytes(buf[off:off+4], 'little'); off += 4
    ecc_r = int.from_bytes(buf[off:off+4], 'little'); off += 4
    return dim, orig, n, sizes, seeds, ecc_scheme, ecc_k, ecc_r


def test_layer_routing_and_stats_exposed():
    os.environ["HLOG_GPU_ONLY"] = "1"
    from holographicfs.memory import mount

    root = Path(tempfile.mkdtemp())
    fs = mount(root, grid_size=64)

    # Store data with filename hinting preference layer
    payload = (b"user=abc\nprefer=true\n" * 3000)  # ensure multiple chunks
    doc = fs.store_data(payload, "config.json")

    # Container header
    cpath = fs.state_dir / "hlog" / "containers" / f"{doc}.hgc"
    assert cpath.exists()
    dim, orig, n, sizes, seeds, ecc_scheme, ecc_k, ecc_r = _read_hgmc2_header(cpath)
    assert n == len(sizes) == len(seeds)

    # Layered metadata present in container map
    cmap = fs.state_dir / "hlog" / "container_map.json"
    assert cmap.exists()
    db = __import__('json').loads(cmap.read_text(encoding='utf-8'))
    ent = db.get(doc)
    assert ent is not None
    chunk_layers = ent.get('chunk_layers') or []
    alpha_scales = ent.get('layer_alpha_scales') or []
    assert len(chunk_layers) == n
    assert len(alpha_scales) == 7

    # Stats expose per-layer telemetry
    stats = fs.mem.stats()
    assert 'layers' in stats
    layers = stats['layers']
    # chunk_counts aggregated and non-negative
    cc = layers.get('chunk_counts', {}) if isinstance(layers, dict) else {}
    assert isinstance(cc, dict)
    total = sum(int(v) for v in cc.values()) if cc else 0
    assert total >= n

