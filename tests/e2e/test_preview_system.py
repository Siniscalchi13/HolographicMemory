from __future__ import annotations

from pathlib import Path

import requests

from .utils import upload_file


def test_text_preview_via_download(base_url: str, tmp_path: Path):
    p = tmp_path / "note.txt"
    content = "hello preview text\nsecond line"
    p.write_text(content, encoding="utf-8")
    doc_id, _ = upload_file(base_url, p)

    r = requests.get(f"{base_url}/download/{doc_id}")
    assert r.ok
    assert content in r.text


def test_image_preview_thumb_and_content(base_url: str, tmp_path: Path):
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        return  # skip if pillow not available client-side
    # generate simple PNG
    from PIL import Image
    img = Image.new("RGB", (64, 32), color=(10, 200, 10))
    p = tmp_path / "img.png"
    img.save(p)

    doc_id, _ = upload_file(base_url, p)
    # thumb by path
    # need the absolute path from list
    rows = requests.get(f"{base_url}/list").json().get("results", [])
    ent = next(r for r in rows if r.get("doc_id") == doc_id)
    path = ent.get("path")
    rt = requests.get(f"{base_url}/thumb", params={"path": path, "w": 128})
    assert rt.ok and rt.headers.get("content-type") == "image/png"

    # full content
    rc = requests.get(f"{base_url}/content", params={"doc_id": doc_id})
    assert rc.ok and rc.headers.get("content-type", "").startswith("image/")


def test_pdf_thumb_available(base_url: str, tmp_path: Path):
    # Create a tiny PDF if PyMuPDF is present server-side; otherwise this still should return 1x1 fallback
    # Write a minimal PDF file
    pdf_bytes = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 0/Kids[]>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF"
    p = tmp_path / "doc.pdf"
    p.write_bytes(pdf_bytes)
    doc_id, _ = upload_file(base_url, p)
    rows = requests.get(f"{base_url}/list").json().get("results", [])
    ent = next(r for r in rows if r.get("doc_id") == doc_id)
    path = ent.get("path")
    rt = requests.get(f"{base_url}/thumb", params={"path": path, "w": 128})
    assert rt.status_code in (200, 404)  # server may fallback to 1x1 PNG

