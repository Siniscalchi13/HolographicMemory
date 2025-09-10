from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class IndexEntry:
    path: str
    doc_id: str
    size: int
    mtime: float
    sha256: str | None = None


class HolographicIndex:
    """Index backed by the C++ holographic memory itself.

    Stores file metadata as messages in a dedicated session (default: 'holo_index').
    Each message content is JSON: {path, doc_id, size, mtime, sha256}.
    """

    def __init__(self, backend, session: str = "holo_index") -> None:  # backend is holographic_cpp.HolographicMemory
        self.backend = backend
        self.session = session

    def add_or_update(self, path: Path, doc_id: str, size: int, mtime: float | None = None, sha256: str | None = None) -> None:
        p = str(Path(path).resolve())
        mtime_v = float(mtime if mtime is not None else Path(path).stat().st_mtime)
        payload = {
            "path": p,
            "doc_id": doc_id,
            "size": int(size),
            "mtime": mtime_v,
        }
        if sha256:
            payload["sha256"] = sha256
        self.backend.store_message(self.session, "file", json.dumps(payload), 1.0, "")

    def remove(self, path: Path) -> None:
        p = str(Path(path).resolve())
        payload = {"path": p, "deleted": True}
        self.backend.store_message(self.session, "file", json.dumps(payload), 1.0, "")

    def _scan(self) -> Dict[str, IndexEntry]:
        msgs = self.backend.get_session_messages(self.session)
        state: Dict[str, IndexEntry] = {}
        for m in msgs:
            try:
                content = json.loads(m.get("content", "{}"))
                p = content.get("path")
                if not p:
                    continue
                if content.get("deleted"):
                    state.pop(p, None)
                    continue
                e = IndexEntry(
                    path=p,
                    doc_id=str(content.get("doc_id", "")),
                    size=int(content.get("size", 0)),
                    mtime=float(content.get("mtime", 0.0)),
                    sha256=content.get("sha256"),
                )
                state[p] = e
            except Exception:
                continue
        return state

    def lookup_by_path(self, path: Path) -> Optional[IndexEntry]:
        p = str(Path(path).resolve())
        return self._scan().get(p)

    def lookup_by_name(self, name: str) -> List[IndexEntry]:
        name_lower = name.lower()
        out: List[IndexEntry] = []
        for e in self._scan().values():
            if Path(e.path).name.lower().find(name_lower) >= 0:
                out.append(e)
        return out

    def all(self) -> List[IndexEntry]:
        return list(self._scan().values())

    def stats(self) -> Dict[str, int]:
        entries = self._scan()
        total = sum(e.size for e in entries.values())
        return {"files_indexed": len(entries), "original_total_bytes": total}
