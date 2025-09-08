"""End-to-end tests for file upload flow and holographic reconstruction."""

from __future__ import annotations

from pathlib import Path

import concurrent.futures
import pytest
import requests

from .utils import upload_file, list_files


def test_file_upload_flow(base_url: str, tmp_path: Path):
    """Test basic file upload flow with holographic reconstruction."""
    # Create sample files
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("alpha", encoding="utf-8")
    f2.write_text("beta", encoding="utf-8")

    doc1, name1 = upload_file(base_url, f1)
    doc2, name2 = upload_file(base_url, f2)

    assert doc1 != doc2, "doc_ids must be unique for different content"

    rows = list_files(base_url)
    ids = [r.get("doc_id") for r in rows]

    # Check that doc_ids are present (files are stored as .hwp holographic patterns)
    assert doc1 in ids, f"doc_id {doc1} not found in list"
    assert doc2 in ids, f"doc_id {doc2} not found in list"

    # Verify holographic reconstruction works for both files
    for doc_id in [doc1, doc2]:
        response = requests.get(f"{base_url}/content", params={"doc_id": doc_id})
        assert response.status_code == 200, f"Failed to reconstruct file {doc_id}"
        # For .hwp files, content should be reconstructed as text/plain
        content_type = response.headers.get("content-type", "")
        assert content_type.startswith("text/plain"), f"Unexpected content-type for {doc_id}: {content_type}"


def test_concurrent_uploads(base_url: str, tmp_path: Path):
    """Test concurrent file uploads to ensure thread safety."""
    files = []
    for i in range(5):
        p = tmp_path / f"f_{i}.txt"
        p.write_text(f"payload-{i}", encoding="utf-8")
        files.append(p)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(lambda p: upload_file(base_url, p)[0], files))
    assert len(set(results)) == len(files)
