from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, List, Tuple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class IndexEntry:
    path: str
    doc_id: str
    size: int
    mtime: float


class Index:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.index_path = state_dir / "index.json"
        self.entries: Dict[str, IndexEntry] = {}
        self._load()

    def _load(self) -> None:
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                for p, e in data.items():
                    self.entries[p] = IndexEntry(**e)
            except Exception:
                self.entries = {}

    def _save(self) -> None:
        payload = {p: asdict(e) for p, e in self.entries.items()}
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.index_path)

    def add_or_update(self, path: Path, doc_id: str, size: int) -> None:
        p = str(path.resolve())
        st = path.stat()
        self.entries[p] = IndexEntry(path=p, doc_id=doc_id, size=size, mtime=st.st_mtime)
        self._save()

    def remove(self, path: Path) -> None:
        p = str(path.resolve())
        if p in self.entries:
            del self.entries[p]
            self._save()

    def lookup_by_path(self, path: Path) -> Optional[IndexEntry]:
        return self.entries.get(str(path.resolve()))

    def lookup_by_name(self, name: str) -> List[IndexEntry]:
        name_lower = name.lower()
        out: List[IndexEntry] = []
        for e in self.entries.values():
            if Path(e.path).name.lower().find(name_lower) >= 0:
                out.append(e)
        return out

    def all(self) -> List[IndexEntry]:
        return list(self.entries.values())

    def stats(self) -> Dict[str, int]:
        total = sum(e.size for e in self.entries.values())
        return {"files_indexed": len(self.entries), "original_total_bytes": total}

