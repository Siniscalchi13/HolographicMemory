from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel
from typing import List
from PIL import Image
import io
import base64
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

from holographicfs.memory import mount
from holographicfs.index import sha256_file
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
import threading
import time as _time
from typing import Optional
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception:
    Observer = None  # type: ignore
    FileSystemEventHandler = object  # type: ignore


def get_fs():
    root = Path(os.getenv("HOLO_ROOT", "./data")).expanduser()
    grid = int(os.getenv("GRID_SIZE", "64"))
    root.mkdir(parents=True, exist_ok=True)
    return mount(root, grid_size=grid)


app = FastAPI(title="Holographic Memory API", version="0.1.0")

# Allow desktop app (Electron) to call the API locally
allowed_env = os.getenv(
    "HOLO_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,capacitor://localhost, null",
)
if allowed_env.strip() == "*":
    cors_kwargs = dict(allow_origins=["*"], allow_credentials=True)
else:
    allow_list = [o.strip() for o in allowed_env.split(",") if o.strip()]
    # Include 'null' to permit file:// origins from Electron
    if "null" not in allow_list:
        allow_list.append("null")
    cors_kwargs = dict(allow_origins=allow_list, allow_credentials=True)
app.add_middleware(
    CORSMiddleware,
    allow_methods=["*"],
    allow_headers=["*"],
    **cors_kwargs,
)

# Simple API key guard
def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv("HOLO_API_KEY", "")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Metrics
counter_store = Counter("holo_store_requests_total", "Store requests")
counter_search = Counter("holo_search_requests_total", "Search requests")
counter_stats = Counter("holo_stats_requests_total", "Stats requests")
counter_watch_events = Counter("holo_watch_events_total", "Watch events processed")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/stats")
def stats(_: bool = Depends(require_api_key)):
    counter_stats.inc()
    fs = get_fs()
    return fs.stats()


@app.get("/search")
def search(q: str = Query(..., min_length=1), k: int = 5, _: bool = Depends(require_api_key)):
    counter_search.inc()
    fs = get_fs()
    results = fs.search_index(q)
    return {"results": [{"doc_id": d, "path": p, "size": s, "mtime": m} for d, p, s, m in results[:k]]}


@app.get("/list")
def list_index(_: bool = Depends(require_api_key)):
    fs = get_fs()
    rows = fs.search_index("") if hasattr(fs, "search_index") else []
    # If search_index requires a query, use all entries
    if not rows:
        entries = fs.index.all()  # type: ignore[attr-defined]
        rows = [(e.doc_id, e.path, e.size, getattr(e, 'mtime', 0.0)) for e in entries]
    return {"results": [{"doc_id": d, "path": p, "size": s, "mtime": m} for d, p, s, m in rows]}


def _ensure_under_root(fs_root: Path, p: Path) -> Path:
    rp = p.resolve()
    if not str(rp).startswith(str(fs_root.resolve())):
        raise HTTPException(status_code=400, detail="Path outside root")
    return rp


class PathPayload(BaseModel):
    path: str


class RenamePayload(BaseModel):
    path: str
    new_path: str


# ----------------------- Watcher System -----------------------

class _WatchHandler(FileSystemEventHandler):  # type: ignore[misc]
    def __init__(self, root: Path, ignore: list[str], on_file) -> None:
        self.root = root
        self.ignore = ignore
        self.on_file = on_file

    def _ok(self, p: Path) -> bool:
        sp = str(p)
        for pat in self.ignore:
            if pat and pat in sp:
                return False
        return p.is_file() and str(p).startswith(str(self.root))

    def on_created(self, event):  # type: ignore[no-redef]
        p = Path(getattr(event, 'src_path', ''))
        if self._ok(p):
            self.on_file(p)
    def on_modified(self, event):  # type: ignore[no-redef]
        p = Path(getattr(event, 'src_path', ''))
        if self._ok(p):
            self.on_file(p)


class Watcher:
    def __init__(self, wid: str, path: Path, ignore: list[str]) -> None:
        self.id = wid
        self.path = path
        self.ignore = ignore
        self.events = 0
        self.errors: list[str] = []
        self._observer: Optional[Observer] = None  # type: ignore[name-defined]
        self._running = False

    def start(self):
        if Observer is None:
            raise RuntimeError("watchdog not available")
        fs = get_fs()
        def _on_file(p: Path):
            try:
                fs.store(p)
                self.events += 1
                counter_watch_events.inc()
            except Exception as e:
                self.errors.append(str(e))
        handler = _WatchHandler(Path(fs.root), self.ignore, _on_file)
        obs = Observer()
        obs.schedule(handler, str(self.path), recursive=True)
        obs.start()
        self._observer = obs
        self._running = True

    def stop(self):
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2.0)
        self._running = False

    def status(self) -> dict:
        return {
            'id': self.id,
            'path': str(self.path),
            'ignore': self.ignore,
            'events': self.events,
            'errors': self.errors[-5:],
            'running': self._running,
        }


_WATCHERS: dict[str, Watcher] = {}
_WATCH_LOCK = threading.Lock()


class WatchPayload(BaseModel):
    path: str
    ignore: list[str] = [".git/", ".DS_Store", "~$", ".tmp"]


@app.post("/watch")
def watch_start(body: WatchPayload, _: bool = Depends(require_api_key)):
    p = Path(body.path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail="Path must exist and be a directory")
    wid = f"w{int(_time.time()*1000)}"
    w = Watcher(wid, p, body.ignore)
    with _WATCH_LOCK:
        _WATCHERS[wid] = w
    try:
        w.start()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return w.status()


@app.get("/watch")
def watch_list(_: bool = Depends(require_api_key)):
    with _WATCH_LOCK:
        return {'watchers': [w.status() for w in _WATCHERS.values()]}


@app.delete("/watch/{wid}")
def watch_stop(wid: str, _: bool = Depends(require_api_key)):
    with _WATCH_LOCK:
        w = _WATCHERS.get(wid)
    if not w:
        raise HTTPException(status_code=404, detail="Watcher not found")
    w.stop()
    with _WATCH_LOCK:
        _WATCHERS.pop(wid, None)
    return {'status': 'stopped', 'id': wid}


@app.get("/watch/{wid}/status")
def watch_status(wid: str, _: bool = Depends(require_api_key)):
    with _WATCH_LOCK:
        w = _WATCHERS.get(wid)
    if not w:
        raise HTTPException(status_code=404, detail="Watcher not found")
    return w.status()


@app.post("/delete")
def delete_file(body: PathPayload, _: bool = Depends(require_api_key)):
    fs = get_fs()
    p = _ensure_under_root(fs.root, Path(body.path))
    try:
        if p.exists():
            p.unlink()
        # Update index
        fs.index.remove(p)  # type: ignore[attr-defined]
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rename")
def rename_file(body: RenamePayload, _: bool = Depends(require_api_key)):
    fs = get_fs()
    src = _ensure_under_root(fs.root, Path(body.path))
    dst = _ensure_under_root(fs.root, Path(body.new_path))
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        # Update index
        ent = fs.index.lookup_by_path(dst)  # type: ignore[attr-defined]
        if ent is None:
            # re-add minimal entry; doc_id unknown from API perspective; re-store if needed later
            fs.index.add_or_update(dst, doc_id=f"doc:{dst.name}", size=dst.stat().st_size)  # type: ignore[attr-defined]
        else:
            fs.index.add_or_update(dst, doc_id=ent.doc_id, size=dst.stat().st_size)  # type: ignore[attr-defined]
        if src != dst:
            try:
                fs.index.remove(src)  # type: ignore[attr-defined]
            except Exception:
                pass
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tree")
def tree(_: bool = Depends(require_api_key)):
    fs = get_fs()
    entries = fs.index.all()  # type: ignore[attr-defined]
    root = Path(fs.root)
    tree: dict = {"name": root.name, "path": str(root), "dirs": {}, "files": []}
    for e in entries:
        rel = str(Path(e.path).resolve().relative_to(root.resolve()))
        parts = [p for p in rel.split("/") if p]
        node = tree
        for d in parts[:-1]:
            node = node["dirs"].setdefault(d, {"name": d, "path": str(root / "/".join(parts[: parts.index(d)+1])), "dirs": {}, "files": []})
        node["files"].append({"name": parts[-1], "path": e.path, "size": e.size, "doc_id": e.doc_id})
    # Convert dirs dict to list
    def _conv(n):
        n["dirs"] = [ _conv(v) for v in n["dirs"].values() ]
        return n
    return _conv(tree)


@app.get("/thumb")
def thumb(path: str, w: int = 256, _: bool = Depends(require_api_key)):
    fs = get_fs()
    # Prefer HM-stored preview (base64) keyed by doc_id
    try:
        ent = fs.index.lookup_by_path(Path(path))  # type: ignore[attr-defined]
        if ent and hasattr(fs.mem.backend, "retrieve_response_hrr"):
            txt = fs.mem.backend.retrieve_response_hrr(ent.doc_id)  # type: ignore[attr-defined]
            if isinstance(txt, str) and txt:
                data = base64.b64decode(txt.encode("ascii"), validate=False)
                return Response(content=data, media_type="image/png")
    except Exception:
        pass
    # Fallback to on-disk thumbnail (non-holographic)
    p = _ensure_under_root(fs.root, Path(path))
    if not p.exists():
        raise HTTPException(status_code=404, detail="Not found")
    try:
        img = Image.open(p)
        img.thumbnail((w, w))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception:
        # Not an image; return 1x1 transparent
        return Response(content=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0cIDATx\x9cc````\x00\x00\x00\x04\x00\x01\x0b\xe7\x83\xbc\x00\x00\x00\x00IEND\xaeB`\x82", media_type="image/png")


@app.get("/fileinfo")
def fileinfo(path: str, _: bool = Depends(require_api_key)):
    fs = get_fs()
    p = _ensure_under_root(fs.root, Path(path))
    ent = fs.index.lookup_by_path(p)  # type: ignore[attr-defined]
    if not ent:
        raise HTTPException(status_code=404, detail="Not found in index")
    sha = None
    try:
        if p.exists() and p.is_file():
            sha = sha256_file(p)
    except Exception:
        sha = None
    has_preview = False
    try:
        if hasattr(fs.mem.backend, 'retrieve_response_hrr'):
            txt = fs.mem.backend.retrieve_response_hrr(ent.doc_id)  # type: ignore[attr-defined]
            has_preview = bool(isinstance(txt, str) and txt)
    except Exception:
        has_preview = False
    return {
        'path': ent.path,
        'doc_id': ent.doc_id,
        'size': ent.size,
        'mtime': getattr(ent, 'mtime', 0.0),
        'sha256': sha,
        'has_preview': has_preview,
    }


@app.post("/store")
async def store(file: UploadFile = File(...), _: bool = Depends(require_api_key)):
    counter_store.inc()
    fs = get_fs()
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    dst = Path(fs.root) / file.filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = await file.read()
    dst.write_bytes(data)
    doc_id = fs.store(dst)
    # Store HM-based thumbnail (base64) for images and PDFs
    thumb_png: bytes | None = None
    # Try raster image
    try:
        img = Image.open(io.BytesIO(data))
        img.thumbnail((384, 384))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        thumb_png = buf.getvalue()
    except Exception:
        # Try PDF first page via PyMuPDF if available
        if fitz is not None and (file.filename or '').lower().endswith('.pdf'):
            try:
                doc = fitz.open(stream=data, filetype='pdf')
                page = doc.load_page(0)
                pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
                thumb_png = pix.tobytes('png')
            except Exception:
                thumb_png = None
    if thumb_png and hasattr(fs.mem.backend, "store_response_hrr"):
        try:
            b64 = base64.b64encode(thumb_png).decode("ascii")
            fs.mem.backend.store_response_hrr(doc_id, b64)  # type: ignore[attr-defined]
        except Exception:
            pass
    return {"doc_id": doc_id, "filename": file.filename}


@app.get("/download/{doc_id}")
def download(doc_id: str, _: bool = Depends(require_api_key)):
    fs = get_fs()
    # Attempt pure HM chunk reconstruction via library
    try:
        data = fs.mem.retrieve_bytes(doc_id)
        return Response(content=data, media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Not retrievable: {e}")


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
