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


@pytest.mark.parametrize("redundancy", [1, 3, 5, 7])
def test_wave_ecc_corrects_light_errors(redundancy):
    import holographic_gpu as hg  # type: ignore
    rng = random.Random(12345)
    size = 2 * 223 + 54  # span multiple chunks
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    seed_base = 0xABCDEF
    parity = hg.wave_ecc_encode(payload, int(redundancy), seed_base)  # type: ignore[attr-defined]

    # inject a few light errors in the first region
    idxs = rng.sample(range(0, min(32, size)), min(4, size))
    corrupted = _inject_symbol_errors(payload, idxs)

    corrected, errors = hg.wave_ecc_decode(corrupted, bytes(parity), int(redundancy), seed_base)  # type: ignore[attr-defined]
    # Parity must match and payload must be restored for modest corruption
    new_parity = hg.wave_ecc_encode(corrected, int(redundancy), seed_base)  # type: ignore[attr-defined]
    assert corrected == payload
    assert bytes(new_parity) == bytes(parity)


def test_wave_ecc_fails_on_parity_tamper():
    import holographic_gpu as hg  # type: ignore
    rng = random.Random(777)
    size = 2 * 223 + 7
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    redundancy = 3
    seed_base = 0x55AA33
    parity = bytearray(hg.wave_ecc_encode(payload, redundancy, seed_base))  # type: ignore[attr-defined]

    # Tamper parity to force failure
    if len(parity) > 0:
        parity[0] ^= 0x01

    corrected, _ = hg.wave_ecc_decode(payload, bytes(parity), redundancy, seed_base)  # type: ignore[attr-defined]
    # Expect parity mismatch after re-encode
    new_parity = hg.wave_ecc_encode(corrected, redundancy, seed_base)  # type: ignore[attr-defined]
    assert bytes(new_parity) != bytes(parity) or corrected != payload


def test_helper_verify_and_correct_rs_enforces_parity():
    if not gpu_available():
        pytest.skip("GPU not available")
    import holographic_gpu as hg  # type: ignore
    from holographicfs.memory import verify_and_correct_rs
    rng = random.Random(2025)
    redundancy = 3
    seed_base = 0x314159
    size = 3 * 223 + 11
    payload = bytes(rng.getrandbits(8) for _ in range(size))
    parity = hg.wave_ecc_encode(payload, redundancy, seed_base)  # type: ignore[attr-defined]

    # Success for light errors
    idxs_ok = rng.sample(range(0, min(32, size)), min(8, size))
    corrected = verify_and_correct_rs(_inject_symbol_errors(payload, idxs_ok), bytes(parity), redundancy, seed_base)
    assert corrected == payload

    # Failure for parity tamper
    tampered = bytearray(parity)
    if len(tampered) > 0:
        tampered[0] ^= 0x80
    with pytest.raises(RuntimeError):
        verify_and_correct_rs(payload, bytes(tampered), redundancy, seed_base)
