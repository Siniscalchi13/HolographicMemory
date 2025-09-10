from __future__ import annotations

from typing import List, Tuple

from services.aiucp.quantum_core.enhanced_privacy import verify_schnorr_batch


def _hash_challenge(g: int, y: int, t: int, q: int) -> int:
    import hashlib

    h = hashlib.sha256()
    for v in (g, y, t):
        h.update(int(v).to_bytes(64, "big"))
    return int.from_bytes(h.digest(), "big") % q


def _make_proof(g: int, p: int, q: int, x: int, r: int) -> Tuple[int, int, int]:
    y = pow(g, x, p)
    t = pow(g, r, p)
    c = _hash_challenge(g, y, t, q)
    s = (r + c * x) % q
    return y, t, s


def test_schnorr_batch_verify_small_group():
    # Tiny toy parameters with q | (p-1) and g of order q (not secure)
    p = 23
    q = 11
    g = 2  # order 11 modulo 23
    x1, r1 = 7, 5
    x2, r2 = 9, 4
    y1, t1, s1 = _make_proof(g, p, q, x1, r1)
    y2, t2, s2 = _make_proof(g, p, q, x2, r2)
    pub = (g, p, q, y1)
    # First pair validates under pub (y1)
    ok1 = verify_schnorr_batch([(t1, s1)], pub)[0]
    # Second pair should fail against first public key
    ok2 = verify_schnorr_batch([(t2, s2)], pub)[0]
    assert ok1 is True and ok2 is False
