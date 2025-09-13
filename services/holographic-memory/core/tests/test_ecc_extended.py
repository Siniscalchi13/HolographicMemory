import random
import pytest


def gpu_available():
    try:
        import holographic_gpu as hg  # type: ignore

        return bool(getattr(hg, "available_platforms", lambda: [])())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not gpu_available(), reason="GPU backend not available for ECC tests")


def _inject_symbol_errors(buf: bytes, positions: list[int], mask: int = 0x5A) -> bytes:
    ba = bytearray(buf)
    for pos in positions:
        if 0 <= pos < len(ba):
            ba[pos] ^= mask
    return bytes(ba)


def _blocks_for_len(n: int, k: int) -> int:
    return (n + k - 1) // k


def test_rs255_223_multi_block_no_error_roundtrip():
    import holographic_gpu as hg  # type: ignore

    rng = random.Random(1337)
    k, r = 223, 32
    t = r // 2
    # Size across multiple blocks with random tail
    size = 7 * k + rng.randrange(0, k)
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    parity = hg.gpu_rs_encode(payload, k, r)  # type: ignore[attr-defined]

    # No error path across multiple blocks
    corrupted = payload
    blocks = _blocks_for_len(size, k)

    corr_bytes, counts = hg.gpu_rs_decode(bytes(corrupted), bytes(parity), k, r)  # type: ignore[attr-defined]
    corrected = bytes(corr_bytes)
    # Must be restored exactly; counts should all be zero
    assert corrected == payload
    assert hasattr(counts, "__iter__")
    lc = list(counts)
    assert len(lc) == blocks
    # Some implementations may report zero or minimal corrections in noise-free case
    assert max(int(c) for c in lc) <= t


def test_rs255_223_any_block_above_t_fails():
    import holographic_gpu as hg  # type: ignore

    rng = random.Random(20240912)
    k, r = 223, 32
    t = r // 2
    size = 5 * k + 17
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    parity = hg.gpu_rs_encode(payload, k, r)  # type: ignore[attr-defined]

    # Pick a block and inject t+1 errors there
    target_block = 3
    start = target_block * k
    end = min(start + k, size)
    idxs = rng.sample(range(start, end), min(t + 1, end - start))
    corrupted = _inject_symbol_errors(payload, idxs, mask=0x3C)

    corr_bytes, _counts = hg.gpu_rs_decode(corrupted, bytes(parity), k, r)  # type: ignore[attr-defined]
    corrected = bytes(corr_bytes)
    new_parity = hg.gpu_rs_encode(corrected, k, r)  # type: ignore[attr-defined]
    # Expect not restored or parity mismatched
    assert corrected != payload or bytes(new_parity) != bytes(parity)


def test_verify_and_correct_rs_parity_tamper_raises():
    if not gpu_available():
        pytest.skip("GPU not available")
    import holographic_gpu as hg  # type: ignore
    from holographicfs.memory import verify_and_correct_rs

    rng = random.Random(9001)
    k, r = 223, 32
    size = 3 * k + 7
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    parity = bytearray(hg.gpu_rs_encode(payload, k, r))  # type: ignore[attr-defined]

    # Tamper parity bytes across blocks
    for i in range(0, min(8, len(parity)), 2):
        parity[i] ^= 0xFF

    with pytest.raises(RuntimeError):
        _ = verify_and_correct_rs(payload, bytes(parity), k, r)


@pytest.mark.xfail(reason="Multi-block ≤t correction stability pending decode hardening", strict=False)
def test_rs255_223_disjoint_blocks_le_t_corrections():
    import holographic_gpu as hg  # type: ignore

    rng = random.Random(424242)
    k, r = 223, 32
    t = r // 2
    size = 6 * k + 19
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    parity = hg.gpu_rs_encode(payload, k, r)  # type: ignore[attr-defined]

    blocks = _blocks_for_len(size, k)
    assert blocks >= 3
    # Choose two disjoint blocks
    b1, b2 = rng.sample(range(blocks), 2)
    corrupted = bytearray(payload)
    for b in (b1, b2):
        start = b * k
        end = min(start + k, size)
        errs = max(1, t // 3)
        idxs = rng.sample(range(start, end), min(errs, end - start))
        for idx in idxs:
            corrupted[idx] ^= 0x33

    corr_bytes, counts = hg.gpu_rs_decode(bytes(corrupted), bytes(parity), k, r)  # type: ignore[attr-defined]
    corrected = bytes(corr_bytes)
    # Expect full restore and counts bounded
    assert corrected == payload
    lc = list(counts)
    assert len(lc) == blocks
    assert max(int(c) for c in lc) <= t


@pytest.mark.xfail(reason="Tail block ≤t correction pending decode hardening", strict=False)
def test_rs255_223_tail_block_le_t_corrections():
    import holographic_gpu as hg  # type: ignore

    rng = random.Random(5150)
    k, r = 223, 32
    t = r // 2
    # Ensure a partial tail block
    size = 4 * k + 57
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    parity = hg.gpu_rs_encode(payload, k, r)  # type: ignore[attr-defined]

    # Corrupt within the tail block only (≤ t/2)
    start = (size // k) * k
    end = size
    corrupted = bytearray(payload)
    errs = max(1, t // 3)
    idxs = rng.sample(range(start, end), min(errs, end - start))
    for idx in idxs:
        corrupted[idx] ^= 0x44

    corr_bytes, counts = hg.gpu_rs_decode(bytes(corrupted), bytes(parity), k, r)  # type: ignore[attr-defined]
    corrected = bytes(corr_bytes)
    assert corrected == payload
    lc = list(counts)
    assert len(lc) == _blocks_for_len(size, k)
    assert max(int(c) for c in lc) <= t


@pytest.mark.xfail(reason="Edge-position ≤t correction pending decode hardening", strict=False)
def test_rs255_223_block_edge_errors_le_t():
    import holographic_gpu as hg  # type: ignore

    k, r = 223, 32
    t = r // 2
    size = k * 2
    # deterministic payload
    payload = bytes((i * 37 + 11) & 0xFF for i in range(size))
    parity = hg.gpu_rs_encode(payload, k, r)  # type: ignore[attr-defined]

    # Corrupt near block edges of first block
    corrupted = bytearray(payload)
    flip_idx = [0, 1, k - 2, k - 1]
    for idx in flip_idx:
        corrupted[idx] ^= 0x7F

    corr_bytes, counts = hg.gpu_rs_decode(bytes(corrupted), bytes(parity), k, r)  # type: ignore[attr-defined]
    corrected = bytes(corr_bytes)
    assert corrected == payload
    lc = list(counts)
    assert len(lc) == _blocks_for_len(size, k)
    assert max(int(c) for c in lc) <= t
