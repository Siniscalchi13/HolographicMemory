from __future__ import annotations

import os
import random
import string
from typing import Iterable

import pytest


def _rand_bytes(n: int) -> bytes:
    return os.urandom(n)


def _low_entropy_bytes(n: int) -> bytes:
    return (b"A" * n)


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload,expect",
    [
        (b"API_KEY=xyz", True),
        (b"password=secret", True),
        (b"no secrets here", False),
        (b"ghp_" + b"a" * 30, True),
        (b"Bearer abc.def.ghi", True),
        (b"user: name\nnotes: nothing to see", False),
    ],
)
def test_detect_secrets_patterns(payload: bytes, expect: bool) -> None:
    from services.vault.security_guard import SecurityGuard

    g = SecurityGuard()
    assert g.detect_secrets(payload) is expect


@pytest.mark.unit
@pytest.mark.parametrize("n", [16, 32, 64, 128, 256, 512, 1024])
def test_detect_secrets_entropy(n: int) -> None:
    from services.vault.security_guard import SecurityGuard

    g = SecurityGuard(entropy_threshold=4.0)
    # high entropy buffers of sufficient window size should trip
    payload = _rand_bytes(max(32, n))
    assert g.detect_secrets(payload) is True
    # very low entropy buffers should not
    assert g.detect_secrets(_low_entropy_bytes(max(32, n))) is False


@pytest.mark.unit
def test_vault_id_uniqueness_and_format() -> None:
    from services.vault.security_guard import SecurityGuard

    g = SecurityGuard()
    ids = {g.generate_vault_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(isinstance(i, str) and len(i) >= 16 for i in ids)

