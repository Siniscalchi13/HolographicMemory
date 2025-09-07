from __future__ import annotations

from pathlib import Path

import pytest

from .utils import upload_file, list_files


def test_file_upload_flow(base_url: str, tmp_path: Path):
    # Create sample files
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("alpha", encoding="utf-8")
    f2.write_text("beta", encoding="utf-8")

    doc1, name1 = upload_file(base_url, f1)
    doc2, name2 = upload_file(base_url, f2)

    assert doc1 != doc2, "doc_ids must be unique for different content"

    rows = list_files(base_url)
    paths = [r.get("path") for r in rows]
    ids = [r.get("doc_id") for r in rows]
    assert any(name1 in p for p in paths), "Uploaded file a.txt not listed"
    assert any(name2 in p for p in paths), "Uploaded file b.txt not listed"
    assert doc1 in ids and doc2 in ids


def test_concurrent_uploads(base_url: str, tmp_path: Path):
    import concurrent.futures
    files = []
    for i in range(5):
        p = tmp_path / f"f_{i}.txt"
        p.write_text(f"payload-{i}", encoding="utf-8")
        files.append(p)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(lambda p: upload_file(base_url, p)[0], files))
    assert len(set(results)) == len(files)
