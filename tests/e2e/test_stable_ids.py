from __future__ import annotations

import os
from pathlib import Path

import pytest

from .utils import upload_file


def test_id_stability_same_content(base_url: str, tmp_path: Path):
    p = tmp_path / "same.txt"
    p.write_text("constant", encoding="utf-8")
    d1, _ = upload_file(base_url, p)
    # Upload again with same content (same filename or different) — doc_id should stay the same since it is content-hash
    p2 = tmp_path / "same_copy.txt"
    p2.write_text("constant", encoding="utf-8")
    d2, _ = upload_file(base_url, p2)
    assert d1 == d2


@pytest.mark.skipif(os.getenv("HOLO_E2E_ALLOW_DOCKERCTL", "0") != "1", reason="restart requires docker control")
def test_id_stability_across_restart(base_url: str, tmp_path: Path):
    from .conftest import restart_api_if_allowed

    p = tmp_path / "persist.txt"
    p.write_text("persist me", encoding="utf-8")
    before, _ = upload_file(base_url, p)

    restart_api_if_allowed(base_url)

    # Re-upload same content; expect identical doc_id
    after, _ = upload_file(base_url, p)
    assert before == after

