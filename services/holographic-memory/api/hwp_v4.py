from __future__ import annotations

import io
import math
import struct
from typing import Dict, List, Tuple


# Simple base-128 varint (unsigned) encoder
def _put_varu(buf: io.BufferedWriter | io.BytesIO, n: int) -> None:
    n = int(n)
    if n < 0:
        raise ValueError("varint must be unsigned")
    while True:
        to_write = n & 0x7F
        n >>= 7
        if n:
            buf.write(bytes((to_write | 0x80,)))
        else:
            buf.write(bytes((to_write,)))
            break


def _quantize_amplitude(values: List[float]) -> Tuple[List[int], float]:
    """Quantize amplitudes to 8-bit with per-layer scale.

    Returns (quantized_bytes, scale), where real_value ≈ q/255 * scale.
    """
    if not values:
        return [], 1.0
    m = float(max(values))
    if m <= 1e-12:
        return [0 for _ in values], 1.0
    q = [min(255, max(0, int(round((v / m) * 255.0)))) for v in values]
    return q, m


def _quantize_phase(values: List[float]) -> List[int]:
    """Quantize phases in [-pi, pi) to 10-bit resolution (stored in uint16)."""
    out: List[int] = []
    two_pi = 2.0 * math.pi
    for v in values:
        # wrap to [-pi, pi)
        x = ((float(v) + math.pi) % two_pi) - math.pi
        q = int(round((x + math.pi) * (1023.0 / two_pi)))
        if q < 0:
            q = 0
        if q > 1023:
            q = 1023
        out.append(q)
    return out


def build_sparse_layer(
    name: str,
    amplitudes: List[float],
    phases: List[float],
    top_k: int = 32,
) -> Dict:
    """Build a sparse layer from full amp/phase arrays.

    Select top-K by amplitude magnitude and quantize.
    """
    n = min(len(amplitudes), len(phases))
    if n == 0:
        return {
            "name": name,
            "k": 0,
            "indices": [],
            "amps_q": [],
            "phs_q": [],
            "amp_scale": 1.0,
        }
    # Rank indices by amplitude magnitude (descending)
    order = sorted(range(n), key=lambda i: abs(amplitudes[i]), reverse=True)
    k = min(int(top_k), n)
    sel = order[:k]
    sel.sort()
    amps_sel = [float(amplitudes[i]) for i in sel]
    phs_sel = [float(phases[i]) for i in sel]
    amps_q, amp_scale = _quantize_amplitude(amps_sel)
    phs_q = _quantize_phase(phs_sel)
    return {
        "name": name,
        "k": k,
        "indices": sel,
        "amps_q": amps_q,
        "phs_q": phs_q,
        "amp_scale": float(amp_scale),
    }


def write_hwp_v4(
    out_path,
    *,
    doc_id: str,
    filename: str,
    original_size: int,
    content_type: str,
    dimension: int,
    layers: List[Dict],
) -> None:
    """Write a compact binary .hwp v4 file with sparse layers.

    Binary layout (little-endian):
    - magic[8] = b"HWP4V001"
    - version u8 = 4
    - flags u8   = 0 (bit0: varint indices; bit1: phase10-in-u16)
    - doc_id_len varu, doc_id bytes (UTF-8 hex)
    - filename_len varu, filename bytes (UTF-8)
    - orig_size varu
    - content_type_len varu, content_type bytes (UTF-8)
    - dimension u32
    - layer_count varu
      For each layer:
        - name_len varu, name bytes
        - k varu
        - amp_scale f32
        - indices[k] varu each
        - amplitudes[k] u8 each
        - phases[k] u16 each (10-bit used)
    """
    b = io.BytesIO()
    # Header
    b.write(b"HWP4V001")
    b.write(bytes((4,)))  # version
    flags = 0x01 | 0x02  # varint indices + 10-bit phases in u16
    b.write(bytes((flags,)))
    # doc_id
    did_bytes = (doc_id or "").encode("utf-8")
    _put_varu(b, len(did_bytes))
    b.write(did_bytes)
    # filename
    fn_bytes = (filename or "").encode("utf-8")
    _put_varu(b, len(fn_bytes))
    b.write(fn_bytes)
    # original size
    _put_varu(b, int(original_size))
    # content type
    ct_bytes = (content_type or "").encode("utf-8")
    _put_varu(b, len(ct_bytes))
    b.write(ct_bytes)
    # dimension
    b.write(struct.pack('<I', int(dimension)))
    # layers
    _put_varu(b, len(layers))
    for L in layers:
        name = str(L.get("name", "layer"))
        _put_varu(b, len(name.encode("utf-8")))
        b.write(name.encode("utf-8"))
        k = int(L.get("k", 0))
        _put_varu(b, k)
        # amp_scale
        amp_scale = float(L.get("amp_scale", 1.0))
        b.write(struct.pack('<f', amp_scale))
        # indices
        for idx in L.get("indices", []):
            _put_varu(b, int(idx))
        # amplitudes
        amps_q = L.get("amps_q", [])
        b.write(bytes(int(a) & 0xFF for a in amps_q))
        # phases (store as uint16 with 10-bit resolution)
        phs_q = L.get("phs_q", [])
        for p in phs_q:
            b.write(struct.pack('<H', int(p) & 0x03FF))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(b.getvalue())


def write_hwp_v4_micro(
    out_path,
    *,
    doc_id_hex: str,
    original_size: int,
    dimension: int = 0,
    layers_count: int = 0,
) -> None:
    """Write an ultra-compact micro .hwp suitable for tiny files.

    Layout (little-endian):
    - magic[4] = b"H4M1"
    - flags u8   (bit0: varint present; bit1: micro format)
    - doc_id8 [8] = first 8 raw bytes of SHA-256
    - orig_size varu
    - dimension varu (0 if unused)
    - layer_count varu (0 for none)
    No layer payloads are included to keep size minimal.
    """
    b = io.BytesIO()
    b.write(b"H4M1")
    flags = 0x01 | 0x80  # varint + micro marker
    b.write(bytes((flags,)))
    try:
        did = bytes.fromhex(doc_id_hex)[:8]
        if len(did) < 8:
            did = did.ljust(8, b"\x00")
    except Exception:
        # fallback: hash the hex string bytes
        import hashlib
        did = hashlib.sha256((doc_id_hex or "").encode("utf-8")).digest()[:8]
    b.write(did)
    _put_varu(b, int(original_size))
    _put_varu(b, int(max(0, dimension)))
    _put_varu(b, int(max(0, layers_count)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(b.getvalue())


def write_hwp_v4_micro_k8(
    out_path,
    *,
    doc_id_hex: str,
    original_size: int,
    dimension: int,
    indices: List[int],
    amps_q: List[int],
    phs_q: List[int],
    amp_scale: float,
) -> None:
    """Write a compact micro+coeffs (K<=8) .hwp.

    Layout:
    - magic[4] = b"H4K8"
    - flags u8 (bit0: varint present; bit1: micro variant; bit2: k8)
    - doc_id8 [8] = first 8 raw bytes of SHA-256
    - orig_size varu
    - dimension varu
    - k varu (<=8)
    - amp_scale f32
    - For i in 0..k-1: index varu, amp u8, phase u16
    """
    b = io.BytesIO()
    b.write(b"H4K8")
    flags = 0x01 | 0x80 | 0x04
    b.write(bytes((flags,)))
    try:
        did = bytes.fromhex(doc_id_hex)[:8]
        if len(did) < 8:
            did = did.ljust(8, b"\x00")
    except Exception:
        import hashlib
        did = hashlib.sha256((doc_id_hex or "").encode("utf-8")).digest()[:8]
    b.write(did)
    _put_varu(b, int(original_size))
    _put_varu(b, int(max(0, dimension)))
    k = min(8, len(indices), len(amps_q), len(phs_q))
    _put_varu(b, k)
    b.write(struct.pack('<f', float(amp_scale)))
    for i in range(k):
        _put_varu(b, int(indices[i]))
        b.write(bytes([int(amps_q[i]) & 0xFF]))
        b.write(struct.pack('<H', int(phs_q[i]) & 0x03FF))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(b.getvalue())
