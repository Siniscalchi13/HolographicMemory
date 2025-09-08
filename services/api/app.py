"""Holographic Memory FastAPI application.

FastAPI-based REST API for the Holographic Memory system. Exposes endpoints
for health, store, download, search, and telemetry, plus a minimal dashboard
for live telemetry visualization.
"""
from __future__ import annotations

import math
import os
import zipfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
import io
import base64
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

from holographicfs.memory import mount, HoloFS
from services.api.hwp_v4 import build_sparse_layer, write_hwp_v4, write_hwp_v4_micro
from holographicfs.index import sha256_file
from services.api.router_layers import route_layers
from services.router import MathematicalRouter
from services.telemetry import PerformanceTelemetry
from services.vault import SecurityGuard
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
import threading
import logging
import time as _time
from typing import Optional
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception:
    Observer = None  # type: ignore
    FileSystemEventHandler = object  # type: ignore


FS_SINGLETON: HoloFS | None = None


def get_fs() -> HoloFS:
    global FS_SINGLETON  # noqa: PLW0603
    if FS_SINGLETON is not None:
        return FS_SINGLETON
    root = Path(os.getenv("HOLO_ROOT", "./data")).expanduser()
    grid = int(os.getenv("GRID_SIZE", "64"))
    root.mkdir(parents=True, exist_ok=True)
    FS_SINGLETON = mount(root, grid_size=grid)
    return FS_SINGLETON




app = FastAPI(title="Holographic Memory API", version="0.1.0")
logger = logging.getLogger("holo.api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Allow desktop app (Electron) to call the API locally
allowed_env = os.getenv(
    "HOLO_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,capacitor://localhost, null",
)
if allowed_env.strip() == "*":
    # With wildcard origins, credentials must be disabled per CORS spec
    cors_kwargs = dict(allow_origins=["*"], allow_credentials=False)
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

# Simple API key guard (placed early for route dependency resolution)
def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv("HOLO_API_KEY", "")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Initialize singleton on startup
@app.on_event("startup")
async def _startup():
    _ = get_fs()
    # Initialize SOA services
    app.state.router = MathematicalRouter()
    app.state.telemetry = PerformanceTelemetry()
    app.state.guard = SecurityGuard()

@app.on_event("shutdown")
async def _shutdown():
    pass

# ----------------------- Static Web UI -----------------------
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

@app.get("/")
def web_index():
    idx = _STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"message": "Holographic Memory API", "docs": "/docs"})


@app.get("/dashboard")
def dashboard(_: bool = Depends(require_api_key)):
    # Minimal HTML dashboard with inline JS fetching /telemetry
    api_key = os.getenv('HOLO_API_KEY','')
    html = """
<!DOCTYPE html>
<html>
  <head>
    <meta charset='utf-8' />
    <title>Holographic Memory Dashboard</title>
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
      h1 {{ margin-bottom: 8px; }}
      .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
      pre {{ background: #f6f8fa; padding: 12px; border-radius: 6px; overflow: auto; }}
      .card {{ border: 1px solid #e1e4e8; border-radius: 6px; padding: 12px; }}
    </style>
  </head>
  <body>
    <h1>Holographic Memory Dashboard</h1>
    <p>Live telemetry, compression ratios, and suggested dimension rebalancing.</p>
    <div class="grid">
      <div class="card">
        <h3>Overall</h3>
        <div id="overall"></div>
      </div>
      <div class="card">
        <h3>Suggested Dimensions (D_k*)</h3>
        <pre id="dims"></pre>
      </div>
      <div class="card" style="grid-column: 1 / span 2;">
        <h3>Per-Layer Metrics</h3>
        <pre id="layers"></pre>
      </div>
    </div>
    <script>
      async function refresh() {
        const resp = await fetch('/telemetry', { headers: { 'x-api-key': '""" + api_key + """' } });
        if (!resp.ok) { document.getElementById('overall').innerText = 'Error loading telemetry'; return; }
        const data = await resp.json();
        const t = data.telemetry || {};
        const o = t.overall || {};
        document.getElementById('overall').innerHTML = `
          <b>Original bytes:</b> ${o.bytes_original||0}<br/>
          <b>Stored bytes:</b> ${o.bytes_stored||0}<br/>
          <b>Compression×:</b> ${o.compression_x||'n/a'}<br/>
          <b>Retrievals:</b> ${o.retrievals||0}<br/>
        `;
        document.getElementById('dims').innerText = JSON.stringify(data.suggested_dimensions||{}, null, 2);
        document.getElementById('layers').innerText = JSON.stringify(t.per_layer||{}, null, 2);
      }
      refresh();
      setInterval(refresh, 4000);
    </script>
  </body>
 </html>
    """
    return Response(content=html, media_type="text/html")

# Request logging middleware (method, path, origin, CORS preflight)
@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[override]
    try:
        origin = request.headers.get('origin')
        logger.info("%s %s origin=%s ua=%s", request.method, request.url.path, origin, request.headers.get('user-agent'))
        if request.method == 'OPTIONS':
            logger.info("CORS preflight for %s", request.headers.get('access-control-request-method'))
        response = await call_next(request)
        logger.info("%s %s -> %s", request.method, request.url.path, response.status_code)
        return response
    except Exception:
        logger.exception("request handling error: %s %s", request.method, request.url.path)
        raise

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
    """Get holographic memory statistics and performance metrics."""
    counter_stats.inc()
    fs = get_fs()
    return fs.stats()


@app.get("/search")
def search(q: str = Query(..., min_length=1), k: int = 5, _: bool = Depends(require_api_key)):
    """Search holographic memory for files matching the query."""
    counter_search.inc()
    fs = get_fs()
    results = fs.search_index(q)
    return {"results": [{"doc_id": d, "path": p, "size": s, "mtime": m} for d, p, s, m in results[:k]]}


@app.get("/list")
def list_index(_: bool = Depends(require_api_key)):
    """List all stored holographic memory files with their metadata."""
    fs = get_fs()
    rows = fs.search_index("") if hasattr(fs, "search_index") else []
    # If search_index requires a query, use all entries
    if not rows:
        entries = fs.index.all()  # type: ignore[attr-defined]
        rows = [(e.doc_id, e.path, e.size, getattr(e, 'mtime', 0.0)) for e in entries]
    
    # Get holographic memory stats for compression info
    stats = fs.stats()
    total_original = stats.get("original_total_bytes", 0)
    total_holo = stats.get("holo_bytes", 0)
    file_count = len(rows)
    
    # Estimate per-file holographic size (rough approximation)
    avg_holo_per_file = total_holo / file_count if file_count > 0 else 0
    
    results = []
    for d, p, s, m in rows:
        # Get actual .hwp file size and original file size from metadata
        hwp_path = Path(p)
        holo_size = hwp_path.stat().st_size if hwp_path.exists() else 0

        # Extract original file size from .hwp metadata
        original_size = s  # fallback to index size
        if hwp_path.exists() and hwp_path.suffix.lower() == ".hwp":
            try:
                import json as _json
                hwp_data = _json.loads(hwp_path.read_text(encoding="utf-8"))
                if "original" in hwp_data:
                    original_size = hwp_data["original"].get("size", s)
            except (json.JSONDecodeError, FileNotFoundError, OSError):
                pass  # fallback to index size

        results.append({
            "doc_id": d,
            "path": p,
            "size": original_size,
            "holo_size": holo_size,
            "mtime": m
        })
    
    return {"results": results}


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


class ZipPayload(BaseModel):
    doc_ids: List[str]


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
def thumb(path: str | None = None, doc_id: str | None = None, w: int = 256, _: bool = Depends(require_api_key)):
    fs = get_fs()
    # Resolve entry by path or doc_id
    ent = None
    if path:
        ent = fs.index.lookup_by_path(Path(path))  # type: ignore[attr-defined]
    elif doc_id:
        try:
            entries = fs.index.all()  # type: ignore[attr-defined]
            ent = next((e for e in entries if e.doc_id == doc_id), None)
        except Exception:
            ent = None
    if not ent and not path:
        raise HTTPException(status_code=400, detail="Provide 'path' or 'doc_id'")
    # Prefer HM-stored preview (base64) keyed by doc_id
    try:
        if ent and hasattr(fs.mem.backend, "retrieve_response_hrr"):
            txt = fs.mem.backend.retrieve_response_hrr(ent.doc_id)  # type: ignore[attr-defined]
            if isinstance(txt, str) and txt:
                data = base64.b64decode(txt.encode("ascii"), validate=False)
                return Response(content=data, media_type="image/png")
    except Exception:
        pass
    # Try to retrieve file content from holographic memory
    if ent:
        try:
            # Get file content from holographic memory
            file_content = fs.mem.retrieve_bytes(ent.doc_id)
            if file_content:
                # Try to create thumbnail from the retrieved content
                try:
                    img = Image.open(io.BytesIO(file_content))
                    img.thumbnail((w, w))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    return Response(content=buf.getvalue(), media_type="image/png")
                except Exception:
                    # Not an image, try PDF
                    try:
                        import fitz  # PyMuPDF
                        doc = fitz.open(stream=file_content, filetype="pdf")
                        page = doc[0]
                        pix = page.get_pixmap(matrix=fitz.Matrix(w/page.rect.width, w/page.rect.height))
                        png_data = pix.tobytes("png")
                        doc.close()
                        return Response(content=png_data, media_type="image/png")
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Fallback to on-disk thumbnail (non-holographic) - only if file exists on disk
    try:
        p = _ensure_under_root(fs.root, Path(ent.path if ent else path))
        if p.exists():
            img = Image.open(p)
            img.thumbnail((w, w))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")
    except Exception:
        pass
        
    # Not an image and no holographic content; return 1x1 transparent
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
    """Store a file using holographic compression and return its document ID."""
    counter_store.inc()
    fs = get_fs()
    logger.info("/store upload filename=%s content_type=%s", getattr(file, 'filename', ''), getattr(file, 'content_type', ''))
    if not file.filename:
        logger.warning("/store missing filename")
        raise HTTPException(status_code=400, detail="Missing filename")
    try:
        data = await file.read()
        logger.info("/store received bytes=%s", len(data) if data is not None else 0)
        # Router decision (format/layers/vault)
        metadata = {"filename": file.filename, "content_type": getattr(file, 'content_type', '') or ''}
        routing = getattr(app.state, 'router', None).route_content(data, metadata) if hasattr(app.state, 'router') else {"vault": False, "format": "v4", "layers": [("knowledge", 1.0)], "K": int(os.getenv("HOLO_TOPK", "32") or 32)}
        # Compute doc_id: random nonce for vault; content hash otherwise
        if routing.get("vault"):
            doc_id = getattr(app.state, 'guard', None).generate_vault_id() if hasattr(app.state, 'guard') else __import__('secrets').token_hex(16)
            digest = doc_id  # for consistency in payloads
        else:
            digest = sha256_file(Path(file.filename)) if False else __import__('hashlib').sha256(data).hexdigest()
            doc_id = digest
        # Record adaptive wave dimension mapping for this file size
        try:
            from holographicfs.memory import calculate_optimal_dimension as _calc_dim
            fs.mem.set_dimension_mapping(doc_id, _calc_dim(len(data)))  # type: ignore[attr-defined]
        except Exception:
            pass
        # Create holographic-only pattern file (.hwp) under patterns dir
        patterns_dir = Path(os.getenv("HLOG_DATA_DIR", str(Path(fs.root) / "holographic_memory"))) / "patterns"
        patterns_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(file.filename).name or doc_id[:8]
        hwp_path = patterns_dir / f"{stem}.hwp"
        # Build .hwp payload: store holographic wave pattern ONLY (no original data)
        import json as _json
        # 1) Store original bytes into 3D engine for exact recall (no base64 persistence)
        try:
            if hasattr(fs.mem, "backend3d") and fs.mem.backend3d is not None:  # type: ignore[attr-defined]
                fs.mem.backend3d.store_bytes(data, doc_id)  # type: ignore[attr-defined]
            else:
                # Fallback to high-level API if exposed
                try:
                    fs.mem.store_file(Path(file.filename), stable_id=doc_id)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            # Continue; download will fail if exact recall backend is missing
            pass
        # 2) For non-vault, store lightweight meta in wave engine for mapping (optional)
        if not routing.get("vault"):
            try:
                meta = f"filename:{Path(file.filename).name}\nsize:{len(data)}\nsha256:{digest}\n"
                engine_id = fs.mem.backend.store(meta)  # type: ignore[attr-defined]
                # Persist doc_id -> engine_id mapping inside HM for later real-wave queries
                try:
                    mapping = {"doc_id": doc_id, "engine_id": engine_id, "filename": file.filename, "size": len(data)}
                    # In-engine mapping for same-process retrieval
                    fs.mem.backend.store_response_hrr(f"{doc_id}#engine_mapping", _json.dumps(mapping))  # type: ignore[attr-defined]
                    # On-disk mapping for robustness across restarts
                    mpath = Path(fs.state_dir) / "engine_map.json"  # type: ignore[attr-defined]
                    db = {}
                    if mpath.exists():
                        try:
                            db = _json.loads(mpath.read_text(encoding='utf-8'))
                        except Exception:
                            db = {}
                    db[str(doc_id)] = mapping
                    mpath.write_text(_json.dumps(db, indent=2), encoding='utf-8')
                except Exception:
                    pass
            except Exception:
                pass

        # 3) Write .hwp according to router decision
        try:
            fmt = str(routing.get("format", "v4"))
            if fmt == "micro":
                # Ultra-compact micro header, no coefficients (K=0)
                write_hwp_v4_micro(
                    hwp_path,
                    doc_id_hex=doc_id,
                    original_size=len(data),
                    dimension=0,
                    layers_count=0 if routing.get("vault") else len(routing.get("layers", []) or []),
                )
            elif fmt == "microK8":
                # Small semantic sketch with K=8 for tiny files
                wave_data = fs.mem.get_real_wave_data(doc_id)
                amps = wave_data.get("amplitudes", []) or []
                phs = wave_data.get("phases", []) or []
                dim = int(wave_data.get("dimension", 0) or len(amps))
                layer = build_sparse_layer("k8", list(map(float, amps)), list(map(float, phs)), top_k=8)
                from services.api.hwp_v4 import write_hwp_v4_micro_k8
                write_hwp_v4_micro_k8(
                    hwp_path,
                    doc_id_hex=doc_id,
                    original_size=len(data),
                    dimension=dim,
                    indices=layer.get("indices", []),
                    amps_q=layer.get("amps_q", []),
                    phs_q=layer.get("phs_q", []),
                    amp_scale=float(layer.get("amp_scale", 1.0)),
                )
            else:
                wave_data = fs.mem.get_real_wave_data(doc_id)
                amps = wave_data.get("amplitudes", []) or []
                phs = wave_data.get("phases", []) or []
                dim = int(wave_data.get("dimension", 0) or len(amps))
                # Use router-provided layers and K
                weights = routing.get("layers", []) or [("knowledge", 1.0)]
                top_k = int(routing.get("K", int(os.getenv("HOLO_TOPK", "32") or 32)))
                layers = []
                for lname, _w in weights:
                    L = build_sparse_layer(lname, list(map(float, amps)), list(map(float, phs)), top_k=top_k)
                    layers.append(L)
                if not layers:
                    layers.append(build_sparse_layer("knowledge", list(map(float, amps)), list(map(float, phs)), top_k=top_k))
                write_hwp_v4(
                    hwp_path,
                    doc_id=doc_id,
                    filename=file.filename,
                    original_size=len(data),
                    content_type=getattr(file, 'content_type', '') or '',
                    dimension=dim,
                    layers=layers,
                )
        except Exception as e:
            # If wave computation fails, write a minimal header-only v4 for index visibility
            try:
                write_hwp_v4(
                    hwp_path,
                    doc_id=doc_id,
                    filename=file.filename,
                    original_size=len(data),
                    content_type=getattr(file, 'content_type', '') or '',
                    dimension=0,
                    layers=[],
                )
            except Exception:
                raise e
        # 4) Optional fallback: write sidecar JSON with base64 if requested or 3D backend missing (disabled for Vault)
        try:
            fallback = os.getenv("HOLO_FALLBACK_BASE64", "false").lower() in ("1", "true", "yes")
            no3d = not (hasattr(fs.mem, "backend3d") and fs.mem.backend3d is not None)  # type: ignore[attr-defined]
            if (fallback or no3d) and not routing.get("vault"):
                sidecar = hwp_path.with_suffix(hwp_path.suffix + ".json")
                import base64 as _b64
                payload = {
                    "version": 3,
                    "doc_id": doc_id,
                    "original": {
                        "filename": file.filename,
                        "size": len(data),
                        "sha256": digest,
                        "content_type": getattr(file, 'content_type', '') or '',
                    },
                    "encoding": "base64",
                    "data": _b64.b64encode(data).decode("ascii"),
                }
                sidecar.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

        # 5) Index and telemetry
        try:
            fs.index.add_or_update(hwp_path, doc_id=doc_id, size=len(data))  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            stored = hwp_path.stat().st_size if hwp_path.exists() else 0
            telem = getattr(app.state, 'telemetry', None)
            if telem is not None:
                if routing.get("vault"):
                    telem.track_compression(len(data), stored, "vault")
                else:
                    for lname, _w in (routing.get("layers") or [("knowledge", 1.0)]):
                        telem.track_compression(len(data), stored, str(lname))
        except Exception:
            pass
    except Exception as e:
        logger.exception("/store failed")
        raise HTTPException(status_code=500, detail=str(e))
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
    # Note: no original file is persisted; only .hwp pattern exists
    return {"doc_id": doc_id, "filename": file.filename}


@app.get("/download/{doc_id}")
def download(doc_id: str, _: bool = Depends(require_api_key)):
    fs = get_fs()
    try:
        entries = fs.index.all()  # type: ignore[attr-defined]
        ent = next((e for e in entries if e.doc_id == doc_id), None)
        if ent:
            p = _ensure_under_root(fs.root, Path(ent.path))
            if p.exists():
                # If .hwp, reconstruct from holographic backend; support v3 JSON and v4 binary
                if p.suffix.lower() == ".hwp":
                    # Try legacy JSON (v3) for inline data and metadata
                    try:
                        import json as _json, base64 as _b64
                        j = _json.loads(p.read_text(encoding="utf-8"))
                        if j.get("data"):
                            raw = _b64.b64decode((j.get("data") or "").encode("ascii"), validate=False)
                            original = j.get("original", {})
                            fname = original.get("filename") or f"{doc_id}.bin"
                            content_type = original.get("content_type") or "application/octet-stream"
                            try:
                                getattr(app.state, 'telemetry', None) and app.state.telemetry.track_retrieval()
                            except Exception:
                                pass
                            return Response(content=raw, media_type=content_type, headers={"Content-Disposition": f"attachment; filename={Path(fname).name}"})
                        original = j.get("original", {})
                        fname = original.get("filename") or f"{doc_id}.bin"
                        ctype = original.get("content_type") or "application/octet-stream"
                    except Exception:
                        fname = Path(ent.path).stem
                        ctype = "application/octet-stream"
                    # Reconstruct from engine; if it fails, try v4 sidecar JSON fallback
                    try:
                        data = fs.mem.retrieve_bytes(doc_id)
                        try:
                            getattr(app.state, 'telemetry', None) and app.state.telemetry.track_retrieval()
                        except Exception:
                            pass
                        return Response(content=data, media_type=ctype, headers={"Content-Disposition": f"attachment; filename={Path(fname).name}"})
                    except Exception as _e:
                        # Sidecar: .hwp.json with base64 content
                        sidecar = p.with_suffix(p.suffix + ".json")
                        if sidecar.exists():
                            try:
                                import json as _json, base64 as _b64
                                sj = _json.loads(sidecar.read_text(encoding="utf-8"))
                                if sj.get("data"):
                                    raw = _b64.b64decode((sj.get("data") or "").encode("ascii"), validate=False)
                                    original = sj.get("original", {})
                                    fname2 = original.get("filename") or f"{doc_id}.bin"
                                    ctype2 = original.get("content_type") or "application/octet-stream"
                                    try:
                                        getattr(app.state, 'telemetry', None) and app.state.telemetry.track_retrieval()
                                    except Exception:
                                        pass
                                    return Response(content=raw, media_type=ctype2, headers={"Content-Disposition": f"attachment; filename={Path(fname2).name}"})
                            except Exception:
                                pass
                        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {_e}")
                # Legacy managed files
                try:
                    getattr(app.state, 'telemetry', None) and app.state.telemetry.track_retrieval()
                except Exception:
                    pass
                return FileResponse(str(p), media_type="application/octet-stream", filename=p.name)
        # Fallback to HM bytes (if present)
        data = fs.mem.retrieve_bytes(doc_id)
        try:
            getattr(app.state, 'telemetry', None) and app.state.telemetry.track_retrieval()
        except Exception:
            pass
        return Response(content=data, media_type="application/octet-stream")
    except Exception as e:
        logger.warning("/download failed for %s: %s", doc_id, e)
        raise HTTPException(status_code=404, detail=f"Not retrievable: {e}")


@app.get("/content")
def content(path: str | None = None, doc_id: str | None = None, _: bool = Depends(require_api_key)):
    fs = get_fs()
    if path:
        p = _ensure_under_root(fs.root, Path(path))
    elif doc_id:
        entries = fs.index.all()  # type: ignore[attr-defined]
        ent = next((e for e in entries if e.doc_id == doc_id), None)
        if not ent:
            raise HTTPException(status_code=404, detail="Unknown doc_id")
        p = _ensure_under_root(fs.root, Path(ent.path))
    else:
        raise HTTPException(status_code=400, detail="Provide 'path' or 'doc_id'")

    if not p.exists():
        raise HTTPException(status_code=404, detail="Not found")

    # Check if this is a .hwp holographic file that needs reconstruction
    if p.suffix == '.hwp':
        filename = p.stem
        content_type = ''
        # Try legacy JSON (v3) first
        try:
            import json, base64, mimetypes
            j = json.loads(p.read_text(encoding='utf-8'))
            original = j.get('original', {})
            filename = original.get('filename', filename)
            content_type = original.get('content_type', '')
            enc = j.get('data')
            if enc:
                raw = base64.b64decode(enc)
                if not content_type:
                    content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
                try:
                    getattr(app.state, 'telemetry', None) and app.state.telemetry.track_retrieval()
                except Exception:
                    pass
                return Response(content=raw, media_type=content_type, headers={"Content-Disposition": f"attachment; filename={Path(filename).name}"})
            # else fall through to engine reconstruction
        except Exception:
            # try sidecar .hwp.json fallback
            sidecar = p.with_suffix(p.suffix + ".json")
            if sidecar.exists():
                try:
                    import json, base64, mimetypes
                    sj = json.loads(sidecar.read_text(encoding='utf-8'))
                    original = sj.get('original', {})
                    filename = original.get('filename', filename)
                    content_type = original.get('content_type', '')
                    enc = sj.get('data')
                    if enc:
                        raw = base64.b64decode(enc)
                        if not content_type:
                            content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
                        try:
                            getattr(app.state, 'telemetry', None) and app.state.telemetry.track_retrieval()
                        except Exception:
                            pass
                        return Response(content=raw, media_type=content_type, headers={"Content-Disposition": f"attachment; filename={Path(filename).name}"})
                except Exception:
                    pass
        # Reconstruct via engine using index-resolved doc_id
        entries = fs.index.all()  # type: ignore[attr-defined]
        ent = next((e for e in entries if Path(e.path) == p), None)
        if not ent:
            raise HTTPException(status_code=404, detail="Unknown holographic file")
        try:
            raw = fs.mem.retrieve_bytes(ent.doc_id)
        except Exception as e:
            # last fallback: sidecar
            sidecar = p.with_suffix(p.suffix + ".json")
            if sidecar.exists():
                try:
                    import json, base64
                    sj = json.loads(sidecar.read_text(encoding='utf-8'))
                    enc = sj.get('data')
                    if enc:
                        raw = base64.b64decode(enc)
                    else:
                        raise e
                except Exception:
                    raise HTTPException(status_code=500, detail=f"Reconstruction failed: {e}")
            else:
                raise HTTPException(status_code=500, detail=f"Reconstruction failed: {e}")
        if not content_type:
            import mimetypes
            content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        try:
            getattr(app.state, 'telemetry', None) and app.state.telemetry.track_retrieval()
        except Exception:
            pass
        return Response(content=raw, media_type=content_type, headers={"Content-Disposition": f"attachment; filename={Path(filename).name}"})

    # Default behavior for non-.hwp files
    return FileResponse(str(p), filename=p.name)


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/telemetry")
def telemetry(_: bool = Depends(require_api_key)):
    telem = getattr(app.state, 'telemetry', None)
    if telem is None:
        raise HTTPException(status_code=503, detail="Telemetry not initialized")
    snap = telem.snapshot()
    # Suggested rebalancing using default importance and budget
    default_importance = {
        "identity": 0.8,
        "knowledge": 1.0,
        "experience": 0.9,
        "preference": 0.7,
        "context": 0.8,
        "wisdom": 1.0,
        "vault": 0.1,
    }
    # Default budget from env or heuristic (sum of current dimension map if known)
    total_budget = int(os.getenv("HOLO_DIM_BUDGET", "688") or 688)  # sum of initial dims
    suggested = telem.suggest_rebalancing(default_importance, total_budget, floors=None)
    return {
        "telemetry": snap,
        "suggested_dimensions": suggested,
    }


# Administrative shutdown (dev only; guarded by env flag and API key if set)
@app.post("/shutdown")
def shutdown(_: bool = Depends(require_api_key)):
    if os.getenv("HOLO_ALLOW_SHUTDOWN", "0") != "1":
        raise HTTPException(status_code=403, detail="Shutdown disabled")
    # Give response then exit
    def _stop():
        import os as _os, time as _t
        _t.sleep(0.5)
        _os._exit(0)  # noqa: SLF001
    import threading as _th
    _th.Thread(target=_stop, daemon=True).start()
    return {"status": "shutting_down"}


# ----------------------- Management Operations -----------------------

@app.delete("/files/{doc_id}")
def delete_file(doc_id: str, _: bool = Depends(require_api_key)):
    fs = get_fs()
    # Remove from index
    removed = 0
    try:
        # try typed method
        removed = fs.index.remove_by_doc_id(doc_id)  # type: ignore[attr-defined]
    except Exception:
        # fallback: manual sweep
        try:
            entries = fs.index.all()  # type: ignore[attr-defined]
            for e in entries:
                if e.doc_id == doc_id:
                    fs.index.remove(Path(e.path))  # type: ignore[attr-defined]
                    removed += 1
        except Exception:
            pass
    # Remove persistent holographic responses on disk
    try:
        resp_dir = Path(fs.state_dir) / "responses" / doc_id  # type: ignore[attr-defined]
        if resp_dir.exists():
            for p in sorted(resp_dir.glob("**/*"), reverse=True):
                try:
                    if p.is_file():
                        p.unlink()
                except Exception:
                    pass
            try:
                resp_dir.rmdir()
            except Exception:
                pass
    except Exception:
        pass

    # Remove .hwp/.wave files from patterns directory
    try:
        patterns_dir = Path(os.getenv("HLOG_DATA_DIR", str(Path(fs.root) / "holographic_memory"))) / "patterns"
        if patterns_dir.exists():
            for ext in ['.hwp', '.wave']:
                for hwp_file in patterns_dir.glob(f"*{ext}"):
                    try:
                        if hwp_file.exists():
                            import json
                            try:
                                content = json.loads(hwp_file.read_text())
                                # Check if this file contains our doc_id
                                if (content.get("doc_id") == doc_id or
                                    str(hwp_file).find(doc_id[:8]) != -1):  # partial match for filename
                                    hwp_file.unlink()
                                    print(f"Removed holographic file: {hwp_file}")
                                    removed += 1
                            except Exception:
                                # If we can't read the JSON, try filename-based matching
                                if doc_id[:8] in hwp_file.name:
                                    hwp_file.unlink()
                                    print(f"Removed holographic file (filename match): {hwp_file}")
                                    removed += 1
                    except Exception as e:
                        print(f"Error removing file {hwp_file}: {e}")
                        pass
    except Exception as e:
        print(f"Error in patterns cleanup: {e}")
        pass

    return {"status": "deleted", "doc_id": doc_id, "removed": removed}


@app.post("/zip")
def zip_selected(payload: ZipPayload, _: bool = Depends(require_api_key)):
    fs = get_fs()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for did in payload.doc_ids:
            try:
                entries = fs.index.all()  # type: ignore[attr-defined]
                ent = next((e for e in entries if e.doc_id == did), None)
                name = Path(ent.path).name if ent else f"{did}.bin"
                data: bytes
                if ent and Path(ent.path).exists():
                    p = Path(ent.path)
                    if p.suffix.lower() == ".hwp":
                        # Reconstruct original file from .hwp (legacy or wave-only)
                        import json as _json, base64 as _b64
                        j = _json.loads(p.read_text(encoding="utf-8"))
                        original = j.get("original", {})
                        name = original.get("filename") or name
                        if j.get("data"):
                            data = _b64.b64decode((j.get("data") or "").encode("ascii"), validate=False)
                        else:
                            # Wave-only: recall via engine using doc_id
                            data = fs.mem.retrieve_bytes(did)
                    else:
                        data = p.read_bytes()
                else:
                    data = fs.mem.retrieve_bytes(did)
                zf.writestr(name, data)
            except Exception as e:
                # include an error note in the zip for visibility
                zf.writestr(f"{did}.error.txt", str(e))
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip", headers={"Content-Disposition": "attachment; filename=selected.zip"})


@app.get("/wave/{doc_id}")
def wave_pattern(doc_id: str, n: int = 256, _: bool = Depends(require_api_key)):
    """Return a simple frequency magnitude spectrum for the document bytes (naive DFT).

    This is a lightweight stand-in visualization until direct engine coefficients are exposed.
    """
    fs = get_fs()
    try:
        data = fs.mem.retrieve_bytes(doc_id)
    except Exception:
        # Fallback to on-disk
        entries = fs.index.all()  # type: ignore[attr-defined]
        ent = next((e for e in entries if e.doc_id == doc_id), None)
        if not ent:
            raise HTTPException(status_code=404, detail="Unknown doc_id")
        p = _ensure_under_root(fs.root, Path(ent.path))
        data = p.read_bytes()
    # Build real-valued signal from first N bytes
    N = max(32, min(int(n or 256), 1024))
    xs = [float(b) / 255.0 for b in data[:N]]
    mags: List[float] = []
    for k in range(N):
        re = 0.0; im = 0.0
        for t, x in enumerate(xs):
            ang = -2.0 * math.pi * k * t / N
            re += x * math.cos(ang)
            im += x * math.sin(ang)
        mags.append((re * re + im * im) ** 0.5)
    # Normalize to [0,1]
    mmax = max(mags) if mags else 1.0
    mags = [m / mmax for m in mags]
    return {"doc_id": doc_id, "n": N, "magnitudes": mags}


@app.get("/wave/{doc_id}/real")
def real_wave_pattern(doc_id: str, _: bool = Depends(require_api_key)):
    """Return authentic holographic wave data (amplitude + phase) from the engine."""
    fs = get_fs()
    try:
        wd = fs.mem.get_real_wave_data(doc_id)  # type: ignore[attr-defined]
        return {
            "doc_id": doc_id,
            "magnitudes": wd.get("amplitudes", []),
            "phases": wd.get("phases", []),
            "dimension": wd.get("dimension", 0),
            "source": wd.get("source", "holographic_engine"),
            "authentic": True,
            "engine_id": wd.get("engine_id", "")
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Real holographic data not found: {e}")


@app.post("/wave/collective/real")
def real_collective_interference(payload: dict, _: bool = Depends(require_api_key)):
    fs = get_fs()
    doc_ids = payload.get("doc_ids", [])
    if not isinstance(doc_ids, list) or not doc_ids:
        raise HTTPException(status_code=400, detail="No doc_ids provided")
    try:
        wd = fs.mem.get_collective_interference(doc_ids)  # type: ignore[attr-defined]
        return {
            "doc_ids": doc_ids,
            "magnitudes": wd.get("amplitudes", []),
            "phases": wd.get("phases", []),
            "dimension": wd.get("dimension", 0),
            "source": wd.get("source", "collective_interference"),
            "authentic": True,
            "engine_ids": wd.get("engine_ids", [])
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collective interference failed: {e}")
