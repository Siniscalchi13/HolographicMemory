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
    assert len(doc_id) == 64
    out = root / "a.out"
    fs.recall(doc_id, out=out)
    assert out.read_bytes()[: len(content)] == content

