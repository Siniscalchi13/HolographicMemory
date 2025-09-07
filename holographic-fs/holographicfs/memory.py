from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import sys
from pathlib import Path as _Path

# Prefer C++ backend
_cpp_loaded = False
try:  # Try import directly if installed in site-packages
    import holographic_cpp as _hn  # type: ignore
    _cpp_loaded = True
except Exception:
    # Try to add local build dir to sys.path: <repo>/holographic-fs/native/holographic
    try:
        _pkg_root = _Path(__file__).resolve().parents[1]  # holographic-fs/
        _cpp_dir = _pkg_root / "native" / "holographic"
        if _cpp_dir.exists():
            p = str(_cpp_dir)
            if p not in sys.path:
                sys.path.insert(0, p)
            import holographic_cpp as _hn  # type: ignore
            _cpp_loaded = True
    except Exception:
        _cpp_loaded = False


from .index import sha256_file
from .index_hm import HolographicIndex


class Memory:
    """Thin wrapper around Python holographic memory for HoloFS."""

    def __init__(self, state_dir: Path, grid_size: int = 32) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if _cpp_loaded:
            try:
                self.backend = _hn.HolographicMemory(int(grid_size))  # type: ignore[name-defined, attr-defined]
            except Exception:
                raise RuntimeError("C++ backend not available or failed to initialize")
        else:
            raise RuntimeError("C++ backend not available. Build the extensions (make cpp)")

    def store_file(self, path: Path) -> str:
        data = path.read_bytes()
        if _cpp_loaded and hasattr(self.backend, "store"):
            # TAI convention: pass filename header to enable wave persistence patterns
            from .index import sha256_file as _sha
            meta = f"filename:{Path(path).name}\nsize:{len(data)}\nsha256:{_sha(path)}\n"
            doc_id = str(self.backend.store(meta))  # type: ignore[attr-defined]
            # Chunk and persist file bytes into HM as base64 responses for holographic-only recall
            try:
                if hasattr(self.backend, "store_response_hrr"):
                    import base64 as _b64
                    CHUNK = 256 * 1024  # 256KB raw
                    total = len(data)
                    chunks = (total + CHUNK - 1) // CHUNK
                    for i in range(chunks):
                        part = data[i * CHUNK : (i + 1) * CHUNK]
                        b64 = _b64.b64encode(part).decode("ascii")
                        self.backend.store_response_hrr(f"{doc_id}#chunk:{i}", b64)  # type: ignore[attr-defined]
                    manifest = {
                        "doc_id": doc_id,
                        "size": total,
                        "chunks": chunks,
                        "chunk_bytes": CHUNK,
                        "filename": Path(path).name,
                    }
                    import json as _json
                    self.backend.store_response_hrr(f"{doc_id}#manifest", _json.dumps(manifest))  # type: ignore[attr-defined]
            except Exception:
                pass
            return doc_id
        raise RuntimeError("C++ backend missing expected 'store' method")

    def retrieve_to(self, doc_id: str, out_path: Path) -> Path:
        # Try holographic-only bytes reconstruction from stored chunks
        try:
            data = self.retrieve_bytes(doc_id)
        except Exception:
            # Fallback: if engine exposes direct retrieve (future), try it
            if _cpp_loaded and hasattr(self.backend, "retrieve"):
                data = self.backend.retrieve(doc_id)  # type: ignore[attr-defined]
            else:
                raise RuntimeError("Holographic retrieve not available for this document")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        return out_path

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:
        if _cpp_loaded and hasattr(self.backend, "query"):
            try:
                return [(r[0], float(r[1]), r[2]) for r in self.backend.query(query, int(k))]  # type: ignore[attr-defined]
            except Exception:
                pass
        return []

    def stats(self) -> Dict:
        if _cpp_loaded and hasattr(self.backend, "get_stats"):
            try:
                return dict(self.backend.get_stats())  # type: ignore[attr-defined]
            except Exception:
                pass
        return {}

    def retrieve_bytes(self, doc_id: str) -> bytes:
        """Reconstruct file bytes stored as HM response chunks.

        Requires chunks stored with keys:
          {doc_id}#manifest (JSON) and {doc_id}#chunk:{i}
        """
        if not (_cpp_loaded and hasattr(self.backend, "retrieve_response_hrr")):
            raise RuntimeError("HM chunk retrieval not supported by backend")
        import json as _json
        import base64 as _b64
        man_txt = self.backend.retrieve_response_hrr(f"{doc_id}#manifest")  # type: ignore[attr-defined]
        if not isinstance(man_txt, str) or not man_txt:
            raise KeyError("Manifest not found for doc_id")
        man = _json.loads(man_txt)
        chunks = int(man.get("chunks", 0))
        buf = bytearray()
        for i in range(chunks):
            part_txt = self.backend.retrieve_response_hrr(f"{doc_id}#chunk:{i}")  # type: ignore[attr-defined]
            if not isinstance(part_txt, str) or not part_txt:
                raise KeyError(f"Missing chunk {i}")
            buf.extend(_b64.b64decode(part_txt.encode("ascii"), validate=False))
        size = int(man.get("size", len(buf)))
        return bytes(buf[:size])


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
        # Index backed by holographic memory itself
        self.index = HolographicIndex(self.mem.backend)

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
        self.index.add_or_update(path, doc_id, size=path.stat().st_size, sha256=digest)
        return doc_id

    def recall(self, query_or_doc: str, out: Optional[Path] = None, original: bool = False) -> Path:
        # Resolve doc id by name if not obvious
        doc_id = None
        # Exact doc id match (supports non-SHA formats like 'mem_123')
        if any(e.doc_id == query_or_doc for e in self.index.all()):
            doc_id = query_or_doc
        elif len(query_or_doc) == 64 and all(c in "0123456789abcdef" for c in query_or_doc):
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
        # Prefer engine retrieval, else fallback to original file copy
        try:
            return self.mem.retrieve_to(doc_id, out)
        except Exception:
            # Fallback: copy original file if available
            ent = next((e for e in self.index.all() if e.doc_id == doc_id), None)
            if not ent:
                raise
            src = Path(ent.path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(src.read_bytes())
            return out

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
        return [(e.doc_id, e.path, e.size, getattr(e, 'mtime', 0.0)) for e in entries]


def mount(path: str | Path, grid_size: int = 32, state_dir: Optional[str | Path] = None) -> HoloFS:
    return HoloFS(root=Path(path), grid_size=grid_size, state_dir=Path(state_dir) if state_dir else None)
