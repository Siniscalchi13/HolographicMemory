from services.router import MathematicalRouter


def test_router_basic():
    r = MathematicalRouter()
    content = b"Hello world, this is a small text file."
    md = {"filename": "note.txt", "content_type": "text/plain"}
    out = r.route_content(content, md)
    assert isinstance(out, dict)
    assert out.get("format") in ("micro", "microK8", "v4")
    assert isinstance(out.get("layers"), list)

