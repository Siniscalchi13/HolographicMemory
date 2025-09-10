from __future__ import annotations

import pytest


@pytest.mark.unit
def test_detect_secrets_patterns():
    from services.vault.security_guard import SecurityGuard

    g = SecurityGuard()
    assert g.detect_secrets(b"api_key=ABC") is True
    assert g.detect_secrets(b"normal text with nothing sensitive") is False


@pytest.mark.unit
def test_generate_vault_id_uniqueness():
    from services.vault.security_guard import SecurityGuard

    g = SecurityGuard()
    ids = {g.generate_vault_id() for _ in range(10)}
    assert len(ids) == 10

