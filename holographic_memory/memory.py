"""
Public HolographicMemory API

Thin, stable wrapper around the internal services and HoloFS engine so
consumer projects can depend on a consistent interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import tempfile
import os

try:
    # Local import for distribution (packaged with include "services*")
    from holographicfs.memory import HoloFS  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "holographicfs is required but not available. Ensure package is installed with native extensions."
    ) from exc


class HolographicMemory:
    """High-level memory facade.

    - Stores bytes using the underlying holographic engine (3D exact recall when available)
    - Retrieves original bytes by `doc_id`
    - Performs simple semantic search
    """

    def __init__(self, root: str | Path | None = None, grid_size: int = 64) -> None:
        self.root = Path(root or os.getenv("HOLO_ROOT", "./data")).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.fs = HoloFS(self.root, grid_size=int(grid_size))

    # ---------------------- Core API ----------------------
    def store(self, content: bytes, filename: str = "document.bin", description: str | None = None) -> str:
        """Store raw bytes and return a stable doc_id (sha256).

        The data is written to a temporary file inside the HoloFS root so the
        index has a durable path reference; the underlying engine persists the
        real holographic state for exact recall.
        """
        tmp_dir = self.root / ".tmp_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # ensure unique name for tmp write
        with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=Path(filename).suffix or ".bin", delete=False) as fh:
            fh.write(content)
            tmp_path = Path(fh.name)
        try:
            doc_id = self.fs.store(tmp_path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)  # py>=3.8
            except Exception:
                pass
        return doc_id

    def retrieve(self, doc_id: str) -> bytes:
        """Retrieve raw bytes for a document id by querying the engine.

        Falls back to any available legacy sidecar indexes transparently inside HoloFS.
        """
        out = self.root / ".recall_tmp" / f"{doc_id}.bin"
        out.parent.mkdir(parents=True, exist_ok=True)
        path = self.fs.mem.retrieve_to(str(doc_id), out)  # type: ignore[attr-defined]
        data = Path(path).read_bytes()
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass
        return data

    def search_semantic(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return (doc_id, score) pairs for a text query."""
        results = self.fs.search(query, k=int(top_k))
        out: List[Tuple[str, float]] = []
        for r in results:
            # results may be (doc_id, score, label)
            try:
                did, score = str(r[0]), float(r[1])
            except Exception:
                continue
            out.append((did, score))
        return out

    # ---------------------- Utilities ----------------------
    def stats(self) -> dict:
        return self.fs.stats()

    def root_dir(self) -> Path:
        return self.root

