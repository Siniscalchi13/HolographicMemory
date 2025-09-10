from __future__ import annotations

import pytest


@pytest.mark.contract
def test_api_version_present(fastapi_app):  # type: ignore
    assert isinstance(fastapi_app.version, str) and len(fastapi_app.version) >= 3

