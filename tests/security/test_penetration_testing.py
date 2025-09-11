from __future__ import annotations

import os

import pytest


@pytest.mark.security
def test_api_key_enforced_when_set(api_client, monkeypatch: pytest.MonkeyPatch):  # type: ignore
    # Set API key and verify protected endpoints require it
    monkeypatch.setenv("HOLO_API_KEY", "secret-key")
    # stats requires API key
    r = api_client.get("/stats")
    assert r.status_code in (401, 403)
    # with key provided, allowed
    r2 = api_client.get("/stats", headers={"x-api-key": "secret-key"})
    assert r2.status_code == 200

