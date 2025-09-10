from __future__ import annotations

import os
import pytest


@pytest.mark.unit
def test_routing_format_thresholds():
    from services.router.mathematical_router import MathematicalRouter

    r = MathematicalRouter()
    # micro
    out = r.route_content(b"a" * 100, {"filename": "a.txt", "content_type": "text/plain"})
    assert out["format"] == "micro"
    # microK8
    out = r.route_content(b"b" * 600, {"filename": "b.txt", "content_type": "text/plain"})
    assert out["format"] == "microK8"
    # v4
    out = r.route_content(b"c" * 2000, {"filename": "c.txt", "content_type": "text/plain"})
    assert out["format"] == "v4"


@pytest.mark.unit
def test_route_layers_balanced_weights():
    from services.router.mathematical_router import MathematicalRouter

    r = MathematicalRouter()
    out = r.route_content(b"context about image", {"filename": "img.png", "content_type": "image/png"})
    layers = out["layers"]
    assert 1 <= len(layers) <= 2
    assert abs(sum(w for _, w in layers) - 1.0) < 1e-6


@pytest.mark.unit
def test_secret_content_routes_to_vault():
    from services.router.mathematical_router import MathematicalRouter

    r = MathematicalRouter()
    out = r.route_content(b"password=supersecret", {"filename": "secrets.env", "content_type": "text/plain"})
    assert out["vault"] is True
    assert out["layers"][0][0] == "vault"

