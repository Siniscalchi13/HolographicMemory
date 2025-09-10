from __future__ import annotations

import pytest


@pytest.mark.contract
def test_openapi_contract_has_core_routes(fastapi_app):  # type: ignore
    schema = fastapi_app.openapi()
    paths = set(schema.get("paths", {}).keys())
    for route in ("/healthz", "/capabilities", "/store", "/download/{doc_id}"):
        assert route in paths, f"Missing route in OpenAPI: {route}"

