from __future__ import annotations

import os
from typing import Dict, Tuple

import pytest


def _mk_meta(filename: str, content_type: str) -> Dict:
    return {"filename": filename, "content_type": content_type}


@pytest.mark.unit
@pytest.mark.parametrize(
    "filename,ctype",
    [
        ("a.txt", "text/plain"),
        ("b.md", "text/markdown"),
        ("c.pdf", "application/pdf"),
        ("d.csv", "text/csv"),
        ("e.json", "application/json"),
        ("f.yaml", "application/x-yaml"),
        ("g.jpg", "image/jpeg"),
        ("h.png", "image/png"),
        ("i.ini", "text/plain"),
        ("j.toml", "text/plain"),
        ("k.log", "text/plain"),
        ("l.rtf", "application/rtf"),
        ("m.tex", "text/plain"),
        ("n.tsv", "text/tab-separated-values"),
        ("o.bmp", "image/bmp"),
        ("p.tiff", "image/tiff"),
        ("q.doc", "application/msword"),
        ("r.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("s.xml", "application/xml"),
        ("t.html", "text/html"),
    ],
)
@pytest.mark.parametrize("size", [1, 64, 128, 256, 600, 1024, 1500, 5000])
def test_format_selection_and_layers(filename: str, ctype: str, size: int) -> None:
    from services.router.mathematical_router import MathematicalRouter

    os.environ["HOLO_MICRO_THRESHOLD"] = "256"
    os.environ["HOLO_MICRO_K8_MAX"] = "1024"
    r = MathematicalRouter()
    content = (b"x" * size)
    out = r.route_content(content, _mk_meta(filename, ctype))
    assert set(out.keys()) == {"vault", "format", "layers", "K"}
    assert isinstance(out["vault"], bool)
    assert out["format"] in {"micro", "microK8", "v4"}
    if not out["vault"]:
        # layers present, normalized weights
        layers = out["layers"]
        assert 1 <= len(layers) <= 2
        s = sum(w for _, w in layers)
        assert abs(s - 1.0) < 1e-6
        if out["format"] == "micro":
            assert out["K"] == 0
        elif out["format"] == "microK8":
            assert out["K"] == 8
        else:
            assert out["K"] == int(os.getenv("HOLO_TOPK", "32") or 32)


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        b"api_key=XYZ",
        b"password=secret",
        b"token: ghp_abcdefghijklmnopqrstuvwxyz1234",
        b"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.signature",
        b"AWS_SECRET_ACCESS_KEY=AKIA...",
        b"Bearer abc.def.ghi",
    ],
)
def test_secret_detection_routes_to_vault(payload: bytes) -> None:
    from services.router.mathematical_router import MathematicalRouter

    r = MathematicalRouter()
    out = r.route_content(payload, _mk_meta("secrets.env", "text/plain"))
    assert out["vault"] is True
    assert out["layers"] and out["layers"][0][0] == "vault"


@pytest.mark.unit
@pytest.mark.parametrize(
    "micro_thr,microk8_max,sz,expected",
    [
        (256, 1024, 1, "micro"),
        (256, 1024, 256, "micro"),
        (256, 1024, 257, "microK8"),
        (256, 1024, 1024, "microK8"),
        (256, 1024, 1025, "v4"),
        (128, 512, 128, "micro"),
        (128, 512, 400, "microK8"),
        (128, 512, 513, "v4"),
        (1, 2, 3, "v4"),
        (1024, 1024, 1025, "v4"),
    ],
)
def test_threshold_env_overrides(micro_thr: int, microk8_max: int, sz: int, expected: str) -> None:
    from services.router.mathematical_router import MathematicalRouter

    os.environ["HOLO_MICRO_THRESHOLD"] = str(micro_thr)
    os.environ["HOLO_MICRO_K8_MAX"] = str(microk8_max)
    r = MathematicalRouter()
    out = r.route_content(b"x" * sz, _mk_meta("a.txt", "text/plain"))
    assert out["format"] == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "filename,ctype",
    [
        ("report.pdf", "application/pdf"),
        ("notes.txt", "text/plain"),
        ("data.csv", "text/csv"),
        ("config.yaml", "application/x-yaml"),
        ("image.png", "image/png"),
        ("photo.jpg", "image/jpeg"),
        ("doc.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("paper.tex", "text/plain"),
        ("table.tsv", "text/tab-separated-values"),
        ("prefs.ini", "text/plain"),
        ("prefs.toml", "text/plain"),
        ("binary.bin", "application/octet-stream"),
        ("archive.log", "text/plain"),
        ("vector.bmp", "image/bmp"),
        ("scan.tiff", "image/tiff"),
    ],
)
@pytest.mark.parametrize("size", [32, 128, 512, 2048])
def test_layer_assignment_weight_normalization(filename: str, ctype: str, size: int) -> None:
    from services.router.mathematical_router import MathematicalRouter

    r = MathematicalRouter()
    out = r.route_content(b"x" * size, _mk_meta(filename, ctype))
    if out["vault"]:
        return
    layers = out["layers"]
    assert 1 <= len(layers) <= 2
    s = sum(w for _, w in layers)
    assert abs(s - 1.0) < 1e-6
