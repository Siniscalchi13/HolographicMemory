import os
import random
import pytest


def gpu_available():
    try:
        import holographic_gpu as hg  # type: ignore
        plats = set(hg.available_platforms())  # type: ignore[attr-defined]
        return bool(plats)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not gpu_available(), reason="GPU backend not available for ECC tests")


def _inject_symbol_errors(buf: bytes, positions: list[int]) -> bytes:
    ba = bytearray(buf)
    for pos in positions:
        if 0 <= pos < len(ba):
            ba[pos] ^= 0x5A
    return bytes(ba)


def _blocks_for_len(n: int, k: int) -> int:
    return (n + k - 1) // k


@pytest.mark.parametrize("errors", [0, 1, 8, 16])
def test_rs255_223_corrects_up_to_t(errors):
    import holographic_gpu as hg  # type: ignore
    rng = random.Random(12345)
    k, r = 223, 32
    t = r // 2
    size = 2 * k + 54  # span multiple blocks
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    parity = hg.gpu_rs_encode(payload, k, r)  # type: ignore[attr-defined]

    # inject <= t errors within the first block to avoid dispersing
    idxs = rng.sample(range(0, k), errors) if errors > 0 else []
    corrupted = _inject_symbol_errors(payload, idxs)

    corr_bytes, counts = hg.gpu_rs_decode(corrupted, bytes(parity), k, r)  # type: ignore[attr-defined]
    corrected = bytes(corr_bytes)
    # Parity must match and payload must be restored
    new_parity = hg.gpu_rs_encode(corrected, k, r)  # type: ignore[attr-defined]
    assert corrected == payload
    assert bytes(new_parity) == bytes(parity)
    # Optional: counts should not exceed t
    if hasattr(counts, '__iter__'):
        assert max(int(c) for c in counts) <= t


def test_rs255_223_fails_above_t_in_one_block():
    import holographic_gpu as hg  # type: ignore
    rng = random.Random(777)
    k, r = 223, 32
    t = r // 2
    size = 2 * k + 7
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    parity = hg.gpu_rs_encode(payload, k, r)  # type: ignore[attr-defined]

    # inject t+1 errors within the first block to force failure
    idxs = rng.sample(range(0, k), t + 1)
    corrupted = _inject_symbol_errors(payload, idxs)

    corr_bytes, counts = hg.gpu_rs_decode(corrupted, bytes(parity), k, r)  # type: ignore[attr-defined]
    corrected = bytes(corr_bytes)
    new_parity = hg.gpu_rs_encode(corrected, k, r)  # type: ignore[attr-defined]
    # Expect either not restored or parity mismatch (most robust)
    assert bytes(new_parity) != bytes(parity) or corrected != payload


def test_helper_verify_and_correct_rs_enforces_bounds():
    if not gpu_available():
        pytest.skip("GPU not available")
    import holographic_gpu as hg  # type: ignore
    from holographicfs.memory import verify_and_correct_rs
    rng = random.Random(2025)
    k, r = 223, 32
    t = r // 2
    size = 3 * k + 11
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    parity = hg.gpu_rs_encode(payload, k, r)  # type: ignore[attr-defined]

    # Success for <= t errors
    idxs_ok = rng.sample(range(0, k), t)
    corrected = verify_and_correct_rs(_inject_symbol_errors(payload, idxs_ok), bytes(parity), k, r)
    assert corrected == payload

    # Failure for > t errors in a single block
    idxs_bad = rng.sample(range(0, k), t + 1)
    with pytest.raises(RuntimeError):
        verify_and_correct_rs(_inject_symbol_errors(payload, idxs_bad), bytes(parity), k, r)

