from __future__ import annotations

import pytest


@pytest.mark.contract
def test_openapi_schema_is_well_formed(fastapi_app):  # type: ignore
    schema = fastapi_app.openapi()
    assert schema.get("openapi", "").startswith("3."), "OpenAPI version should be 3.x"
    assert "components" in schema

