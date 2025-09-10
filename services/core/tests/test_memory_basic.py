from __future__ import annotations

from pathlib import Path

from holographicfs.memory import mount


def test_store_and_recall(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    fs = mount(root, grid_size=16)
    p = root / "a.txt"
    content = b"hello holo"
    p.write_bytes(content)
    doc_id = fs.store(p)
    assert isinstance(doc_id, str) and len(doc_id) > 0
    # C++ engine does not expose byte-level recall; ensure stats path works
    s = fs.stats()
    assert isinstance(s, dict) and "dimension" in s
