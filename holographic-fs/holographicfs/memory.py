from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from .backends import PythonHolographicMemory as _PyHolo


from .index import Index, sha256_file


class Memory:
    """Thin wrapper around Python holographic memory for HoloFS."""

    def __init__(self, state_dir: Path, grid_size: int = 32) -> None:
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.backend = _PyHolo(grid_size=grid_size, persist_dir=self.state_dir)

    def store_file(self, path: Path) -> str:
        data = path.read_bytes()
        meta = self.backend.store_document(data, filename=str(path.name))
        return str(meta["doc_id"])  # type: ignore[index]

    def retrieve_to(self, doc_id: str, out_path: Path) -> Path:
        data = self.backend.retrieve_document(doc_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        return out_path

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:
        return list(self.backend.search(query, top_k=k))

    def stats(self) -> Dict:
        return dict(self.backend.get_stats())


class HoloFS:
    """High-level HoloFS API.

    - Manages an index mapping original paths to holographic doc_ids
    - Wraps the holographic memory backend for store/search/recall
    """

    def __init__(self, root: Path, grid_size: int = 32, state_dir: Optional[Path] = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.state_dir = Path(state_dir or (self.root / ".holofs")).resolve()
        self.grid_size = int(grid_size)
        self.mem = Memory(self.state_dir, grid_size=self.grid_size)
        self.index = Index(self.state_dir)

    def store(self, path: Path, force: bool = False) -> str:
        path = Path(path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(str(path))
        # Skip if identical already indexed
        ent = self.index.lookup_by_path(path)
        digest = sha256_file(path)
        if ent and not force and ent.doc_id == digest and ent.size == path.stat().st_size:
            return ent.doc_id
        doc_id = self.mem.store_file(path)
        self.index.add_or_update(path, doc_id, size=path.stat().st_size)
        return doc_id

    def recall(self, query_or_doc: str, out: Optional[Path] = None, original: bool = False) -> Path:
        # Resolve doc id by name if not obvious
        doc_id = None
        if len(query_or_doc) == 64 and all(c in "0123456789abcdef" for c in query_or_doc):
            doc_id = query_or_doc
        else:
            cands = self.index.lookup_by_name(query_or_doc)
            if not cands:
                raise FileNotFoundError(f"No record matching '{query_or_doc}' in index")
            # choose first match
            ent = cands[0]
            doc_id = ent.doc_id
            if original:
                out = Path(ent.path)
        if out is None:
            out = Path.cwd() / f"{doc_id}.bin"
        return self.mem.retrieve_to(doc_id, out)

    def search(self, query: str, k: int = 5):
        return self.mem.search(query, k)

    def stats(self) -> Dict:
        s = self.mem.stats()
        idx = self.index.stats()
        # On-disk holographic state size
        wave = self.state_dir / "wave_spatial.npy"
        holo_bytes = wave.stat().st_size if wave.exists() else 0
        meta = self.state_dir / "metadata.json"
        holo_bytes += meta.stat().st_size if meta.exists() else 0
        holo_bytes += (self.state_dir / "index.json").stat().st_size if (self.state_dir / "index.json").exists() else 0
        s.update(idx)
        s["holo_bytes"] = int(holo_bytes)
        s["compression_x"] = round((idx.get("original_total_bytes", 0) / holo_bytes), 2) if holo_bytes else None
        return s

    def search_index(self, query: str):
        entries = self.index.lookup_by_name(query)
        return [(e.doc_id, e.path, e.size) for e in entries]


def mount(path: str | Path, grid_size: int = 32, state_dir: Optional[str | Path] = None) -> HoloFS:
    return HoloFS(root=Path(path), grid_size=grid_size, state_dir=Path(state_dir) if state_dir else None)
