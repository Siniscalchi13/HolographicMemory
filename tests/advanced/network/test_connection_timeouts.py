from __future__ import annotations

import httpx
import pytest


@pytest.mark.network
def test_timeout_to_unroutable():
    # 10.255.255.1 is commonly unroutable in RFC1918; keep low timeout
    with pytest.raises(httpx.ReadTimeout):
        httpx.get("http://10.255.255.1", timeout=0.01)

