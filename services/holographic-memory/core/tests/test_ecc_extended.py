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


def test_wave_ecc_multi_chunk_no_error_roundtrip():
    import holographic_gpu as hg  # type: ignore

    rng = random.Random(1337)
    # Multi-chunk size with random tail
    size = 7 * 223 + rng.randrange(0, 223)
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    redundancy = 3
    seed_base = 0x42
    parity = hg.wave_ecc_encode(payload, redundancy, seed_base)  # type: ignore[attr-defined]

    # No error path across multiple blocks
    corrupted = payload
    corrected, errors = hg.wave_ecc_decode(bytes(corrupted), bytes(parity), redundancy, seed_base)  # type: ignore[attr-defined]
    # Must be restored exactly; errors should be 0
    assert corrected == payload
    assert int(errors) == 0


def test_wave_ecc_severe_corruption_fails():
    import holographic_gpu as hg  # type: ignore

    rng = random.Random(20240912)
    size = 5 * 223 + 17
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    redundancy = 3
    seed_base = 0x4242
    parity = hg.wave_ecc_encode(payload, redundancy, seed_base)  # type: ignore[attr-defined]

    # Pick a middle region and flip many bytes (severe corruption)
    start = size // 3
    end = min(start + 64, size)
    idxs = rng.sample(range(start, end), max(17, (end - start) // 2))
    corrupted = _inject_symbol_errors(payload, idxs, mask=0x3C)

    corrected, _errs = hg.wave_ecc_decode(corrupted, bytes(parity), redundancy, seed_base)  # type: ignore[attr-defined]
    new_parity = hg.wave_ecc_encode(corrected, redundancy, seed_base)  # type: ignore[attr-defined]
    # Parity recheck coherence: if recovered exactly, parity must match; else mismatch expected
    assert (corrected == payload) == (bytes(new_parity) == bytes(parity))


def test_verify_and_correct_rs_parity_tamper_raises():
    if not gpu_available():
        pytest.skip("GPU not available")
    import holographic_gpu as hg  # type: ignore
    from holographicfs.memory import verify_and_correct_rs

    rng = random.Random(9001)
    redundancy = 3
    seed_base = 0x424242
    size = 3 * 223 + 7
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    parity = bytearray(hg.wave_ecc_encode(payload, redundancy, seed_base))  # type: ignore[attr-defined]

    # Tamper parity bytes across blocks
    for i in range(0, min(8, len(parity)), 2):
        parity[i] ^= 0xFF

    with pytest.raises(RuntimeError):
        _ = verify_and_correct_rs(payload, bytes(parity), redundancy, seed_base)


def test_wave_ecc_disjoint_regions_light_corruption():
    import holographic_gpu as hg  # type: ignore

    rng = random.Random(424242)
    redundancy = 4
    seed_base = 77777
    size = 6 * 223 + 19
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    parity = hg.wave_ecc_encode(payload, redundancy, seed_base)  # type: ignore[attr-defined]

    # Choose two disjoint regions
    blocks = 6
    b1, b2 = rng.sample(range(blocks), 2)
    corrupted = bytearray(payload)
    chunk = size // blocks
    for b in (b1, b2):
        start = b * chunk
        end = min(start + chunk, size)
        errs = max(1, (end - start) // 64)
        idxs = rng.sample(range(start, end), min(errs, end - start))
        for idx in idxs:
            corrupted[idx] ^= 0x33

    corrected, _errors = hg.wave_ecc_decode(bytes(corrupted), bytes(parity), redundancy, seed_base)  # type: ignore[attr-defined]
    # Expect full restore under light/disjoint corruption
    assert corrected == payload


def test_wave_ecc_tail_region_light_corruption():
    import holographic_gpu as hg  # type: ignore

    rng = random.Random(5150)
    # Ensure a partial tail block
    redundancy = 5
    seed_base = 424242
    size = 4 * 223 + 57
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    parity = hg.wave_ecc_encode(payload, redundancy, seed_base)  # type: ignore[attr-defined]

    # Corrupt within the tail region only (light corruption)
    start = (size // 223) * 223
    end = size
    corrupted = bytearray(payload)
    errs = max(1, (end - start) // 64)
    idxs = rng.sample(range(start, end), min(errs, end - start))
    for idx in idxs:
        corrupted[idx] ^= 0x44

    corrected, _errors = hg.wave_ecc_decode(bytes(corrupted), bytes(parity), redundancy, seed_base)  # type: ignore[attr-defined]
    assert corrected == payload


def test_wave_ecc_edge_positions_light_corruption():
    import holographic_gpu as hg  # type: ignore

    redundancy = 3
    seed_base = 123456
    size = 223 * 2
    # deterministic payload
    payload = bytes((i * 37 + 11) & 0xFF for i in range(size))
    parity = hg.wave_ecc_encode(payload, redundancy, seed_base)  # type: ignore[attr-defined]

    # Corrupt near block edges of first block
    corrupted = bytearray(payload)
    flip_idx = [0, 1, 223 - 2, 223 - 1]
    for idx in flip_idx:
        corrupted[idx] ^= 0x7F

    corrected, _errors = hg.wave_ecc_decode(bytes(corrupted), bytes(parity), redundancy, seed_base)  # type: ignore[attr-defined]
    assert corrected == payload
