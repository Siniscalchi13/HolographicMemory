from services.vault import SecurityGuard


def test_detect_secrets_simple():
    g = SecurityGuard()
    assert g.detect_secrets(b"API_KEY=abc123") is True
    assert g.detect_secrets(b"just a note") is False


def test_generate_vault_id_length():
    g = SecurityGuard()
    vid = g.generate_vault_id()
    assert isinstance(vid, str)
    assert len(vid) >= 16

