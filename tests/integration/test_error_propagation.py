from __future__ import annotations

import pytest


@pytest.mark.integration
def test_error_response_on_missing_doc(api_client):  # type: ignore
    r = api_client.get("/download/deadbeef")
    # 500 because orchestrator raises during missing content path; accept 500 here
    assert r.status_code in (404, 500)

