import os
import io
import math
import random
import tempfile
import pathlib

import pytest


def require_holographic_gpu():
    try:
        import holographic_gpu  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not require_holographic_gpu(), reason="holographic_gpu module not available")
def test_h4k8_decode_length_matches():
    # Synthesize a tiny H4K8 file and ensure decoder returns correct length
    from services.api.hwp_v4 import write_hwp_v4_micro_k8
    # Fixture parameters
    orig_size = 128
    dim = 64
    k = 8
    # random sparse bins
    idx = sorted(random.sample(range(dim), k))
    amps_q = [random.randint(1, 255) for _ in range(k)]
    phs_q = [random.randint(0, 1023) for _ in range(k)]
    amp_scale = 1.0

    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "test.hwp"
        write_hwp_v4_micro_k8(
            p,
            doc_id_hex=("ab" * 32),
            original_size=orig_size,
            dimension=dim,
            indices=idx,
            amps_q=amps_q,
            phs_q=phs_q,
            amp_scale=amp_scale,
        )
        import holographic_gpu as _hg
        dec = _hg.HolographicGPU()
        try:
            dec.initialize("metal")
        except Exception:
            pass
        out = dec.retrieve_bytes(str(p))
        if isinstance(out, str):
            out = out.encode("latin-1", errors="ignore")
        assert isinstance(out, (bytes, bytearray))
        assert len(out) == orig_size


@pytest.mark.skipif(not require_holographic_gpu(), reason="holographic_gpu module not available")
def test_h4m1_unrecoverable():
    # H4M1 is header-only and must not decode
    from services.api.hwp_v4 import write_hwp_v4_micro
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "micro.hwp"
        write_hwp_v4_micro(p, doc_id_hex=("cd" * 32), original_size=42, dimension=0, layers_count=0)
        import holographic_gpu as _hg
        dec = _hg.HolographicGPU()
        with pytest.raises(Exception):
            _ = dec.retrieve_bytes(str(p))

