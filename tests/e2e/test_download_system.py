from __future__ import annotations

from pathlib import Path

from .utils import upload_file, download_doc, sha256_bytes


def test_upload_and_download_roundtrip(base_url: str, tmp_path: Path):
    p = tmp_path / "roundtrip.bin"
    data = b"\x00\x01\x02\x03some-binary\xff\x00"
    p.write_bytes(data)
    doc_id, _ = upload_file(base_url, p)

    got = download_doc(base_url, doc_id)
    assert got == data
    assert sha256_bytes(got) == sha256_bytes(data)

