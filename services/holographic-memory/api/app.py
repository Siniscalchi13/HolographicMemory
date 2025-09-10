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

@app.get("/capabilities")
def get_capabilities():
    """Get system capabilities and feature availability (accurate, runtime-based)."""
    fs = get_fs()
    mem = getattr(fs, "mem", None)
    backend = getattr(mem, "backend", None)

    # GPU availability
    gpu_available = False
    if hasattr(mem, "gpu_backend") and mem.gpu_backend is not None:  # type: ignore[attr-defined]
        try:
            if hasattr(mem.gpu_backend, "available"):
                gpu_available = bool(mem.gpu_backend.available())  # type: ignore[attr-defined]
            else:
                gpu_available = True
        except Exception:
            gpu_available = True

    # 3D exact recall availability
    exact_recall = bool(getattr(mem, "backend3d", None))

    # 7-layer status via backend stats (if exposed)
    layers_initialized = False
    try:
        if backend is not None and hasattr(backend, "get_layer_stats"):
            stats = backend.get_layer_stats()  # type: ignore[attr-defined]
            layers_initialized = bool(stats.get("layers_initialized", False))  # type: ignore[index]
    except Exception:
        layers_initialized = False

    # Math feature support via method presence
    has_bell = bool(backend is not None and hasattr(backend, "validate_bell_inequality"))
    has_interf = bool(backend is not None and hasattr(backend, "analyze_interference_patterns"))
    has_capacity = bool(backend is not None and hasattr(backend, "enforce_capacity_theorem"))
    has_wave = bool(backend is not None and hasattr(backend, "validate_wave_properties"))
    has_snr = bool(backend is not None and (hasattr(backend, "calculate_layer_snr") or hasattr(backend, "get_layer_stats")))

    # Dimension and memory count (best-effort)
    max_dimension = None
    current_memory_count = None
    try:
        s = fs.stats()
        max_dimension = s.get("dimension")
        current_memory_count = s.get("memory_count")
    except Exception:
        pass

    capabilities = {
        "gpu_acceleration": gpu_available,
        "3d_exact_recall": exact_recall,
        "7layer_decomposition": layers_initialized,
        "bell_inequality_validation": has_bell,
        "interference_analysis": has_interf,
        "capacity_theorem_enforcement": has_capacity,
        "wave_validation": has_wave,
        "snr_monitoring": has_snr,
        "max_dimension": max_dimension,
        "current_memory_count": current_memory_count,
    }

    # Guidance messages for conditional features
    if not layers_initialized:
        capabilities["guidance"] = "7-layer not initialized. Use backend.initialize_7layer_decomposition(total_budget)."
    if not gpu_available:
        capabilities["gpu_guidance"] = "GPU backend unavailable. Install/enable Metal/CUDA/ROCm drivers."
    if not exact_recall:
        capabilities["recall_guidance"] = "3D exact recall backend not loaded. Byte-perfect recall may be limited."

    return capabilities


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
def list_index(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    per_page: int = Query(50, ge=1, le=1000, description="Items per page"),
    _: bool = Depends(require_api_key)
):
    """List all stored holographic memory files with pagination.

    Returns original file size and holographic (.hwp) size to compute compression.
    Uses the on-disk index mapping to recover doc_id and original size.
    """
    fs = get_fs()

    # Scan the actual patterns directory for .hwp files
    patterns_dir = Path(fs.root) / "holographic_memory" / "patterns"
    if not patterns_dir.exists():
        return {"results": [], "total": 0, "page": page, "per_page": per_page, "pages": 0}

    # Get all .hwp files
    hwp_files = list(patterns_dir.glob("*.hwp"))
    total_files = len(hwp_files)

    # Calculate pagination
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_files = hwp_files[start_idx:end_idx]

    results = []
    for hwp_path in page_files:
        try:
            stat = hwp_path.stat()
            rel = str(hwp_path.relative_to(fs.root))
            # Prefer index entry for original size + doc_id
            ent = fs.index.lookup_by_path(hwp_path)  # type: ignore[attr-defined]
            original_size = int(getattr(ent, 'size', 0) or 0)
            doc_id = str(getattr(ent, 'doc_id', '') or '')
            # Fallbacks
            if not doc_id:
                doc_id = hwp_path.stem  # last resort
            if original_size <= 0:
                # As a last resort, approximate with .hwp size (will show 1.0x)
                original_size = int(stat.st_size)
            # Detect .hwp format (H4M1/H4K8/HWP4V001/legacy JSON)
            fmt_magic = "unknown"
            recoverable = False
            try:
                with open(hwp_path, 'rb') as _f:
                    head = _f.read(8)
                if head.startswith(b'H4M1'):
                    fmt_magic = 'H4M1'
                    recoverable = (hwp_path.with_suffix(hwp_path.suffix + ".json").exists())
                elif head.startswith(b'H4K8'):
                    fmt_magic = 'H4K8'
                    recoverable = True
                elif head.startswith(b'HWP4V001'):
                    fmt_magic = 'HWP4V001'
                    recoverable = True
                elif head[:1] == b'{':
                    fmt_magic = 'v3json'
                    recoverable = True
            except Exception:
                fmt_magic = 'unknown'
                recoverable = False

            results.append({
                "doc_id": doc_id,
                "path": rel,
                "original_filename": hwp_path.stem,  # best effort
                "size": original_size,
                "holo_size": int(stat.st_size),
                "mtime": stat.st_mtime,
                "format": fmt_magic,
                "recoverable": recoverable,
            })
        except (OSError, FileNotFoundError):
            continue

    total_pages = (total_files + per_page - 1) // per_page

    return {
        "results": results,
        "total": total_files,
        "page": page,
        "per_page": per_page,
        "pages": total_pages,
    }


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
        raise HTTPException(status_code=503, detail=f"File watcher service unavailable: {str(e)}")
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
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {body.path}")
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {body.path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File deletion failed: {str(e)}")


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
    except FileExistsError:
        raise HTTPException(status_code=409, detail=f"Destination already exists: {body.new_path}")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Source file not found: {body.path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File rename failed: {str(e)}")


@app.get("/tree")
def tree(_: bool = Depends(require_api_key)):
    """Build directory tree from actual filesystem structure."""
    fs = get_fs()
    root = Path(fs.root)
    
    # Build tree from actual filesystem
    def build_tree(path: Path, name: str = None) -> dict:
        if name is None:
            name = path.name
        
        node = {
            "name": name,
            "path": str(path),
            "dirs": [],
            "files": []
        }
        
        try:
            if path.is_dir():
                # Add subdirectories
                for item in sorted(path.iterdir()):
                    if item.is_dir() and not item.name.startswith('.'):
                        node["dirs"].append(build_tree(item))
                    elif item.is_file() and item.suffix == '.hwp':
                        # Only show .hwp files in patterns directory
                        if 'patterns' in str(item):
                            node["files"].append({
                                "name": item.name,
                                "path": str(item.relative_to(root)),
                                "size": item.stat().st_size,
                                "doc_id": item.stem
                            })
        except (OSError, PermissionError):
            pass
        
        return node
    
    return build_tree(root)


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
        exact_recall_ok = False
        try:
            if hasattr(fs.mem, "backend3d") and fs.mem.backend3d is not None:  # type: ignore[attr-defined]
                fs.mem.backend3d.store_bytes(data, doc_id)  # type: ignore[attr-defined]
                exact_recall_ok = True
            else:
                # Fallback to high-level API if exposed (not exact recall)
                try:
                    fs.mem.store_file(Path(file.filename), stable_id=doc_id)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            # Continue; download will fail if exact recall backend is missing
            exact_recall_ok = False
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
            # Avoid non-recoverable micro format (K=0) when exact-recall storage failed
            has3d = bool(exact_recall_ok)
            if fmt == "micro" and not has3d:
                fmt = "microK8"
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
                # Compute wave directly from uploaded bytes (no retrieval dependency)
                wave_data = fs.mem.get_wave_data_from_bytes(data, doc_id)
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
                # v4 path with router-defined layers/K; compute wave from bytes
                wave_data = fs.mem.get_wave_data_from_bytes(data, doc_id)
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
            # Use exact_recall_ok determined above to decide on sidecar persistence
            if (fallback or not exact_recall_ok) and not routing.get("vault"):
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
                # Use the same JSON alias used above to avoid NameError
                sidecar.write_text(_json.dumps(payload), encoding="utf-8")
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
        error_msg = str(e)
        if "disk" in error_msg.lower() or "space" in error_msg.lower():
            raise HTTPException(status_code=507, detail="Insufficient storage space")
        elif "format" in error_msg.lower() or "invalid" in error_msg.lower():
            raise HTTPException(status_code=400, detail=f"Invalid file format: {error_msg}")
        elif "size" in error_msg.lower() or len(data) > 100*1024*1024:  # 100MB
            raise HTTPException(status_code=413, detail="File too large")
        else:
            raise HTTPException(status_code=500, detail=f"Storage failed: {error_msg}")
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
                # If .hwp, reconstruct. Prefer decoding v4 binary via GPU module; support legacy JSON (v3) fallback
                if p.suffix.lower() == ".hwp":
                    # Inspect header for micro/header-only case to provide a clear error early
                    try:
                        with open(p, 'rb') as _f:
                            magic8 = _f.read(8)
                        magic4 = magic8[:4]
                    except Exception:
                        magic4 = b""
                    if magic4 == b'H4M1':
                        # Try legacy sidecar JSON (v3) for inline data and metadata
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
                        raise HTTPException(status_code=409, detail="Stored as header-only (H4M1); no reconstructible data available")
                    # Try modern v4 decode first (H4K8/HWP4V001) using holographic_gpu
                    try:
                        import holographic_gpu as _hg  # type: ignore
                        if hasattr(_hg, 'HolographicGPU'):
                            dec = _hg.HolographicGPU()  # type: ignore[attr-defined]
                            try:
                                # initialize is best-effort; decoder is CPU-side
                                dec.initialize('metal')  # type: ignore[attr-defined]
                            except Exception:
                                pass
                            raw = dec.retrieve_bytes(str(p))  # type: ignore[attr-defined]
                            # Try derive filename from index or path
                            fname = Path(ent.path).stem
                            import mimetypes as _mt
                            ctype = _mt.guess_type(fname)[0] or 'application/octet-stream'
                            try:
                                getattr(app.state, 'telemetry', None) and app.state.telemetry.track_retrieval()
                            except Exception:
                                pass
                            return Response(content=raw, media_type=ctype, headers={"Content-Disposition": f"attachment; filename={Path(fname).name}"})
                    except Exception:
                        pass
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
                        raise HTTPException(status_code=422, detail=f"Data reconstruction failed - 3D backend unavailable or data corrupted: {_e}")
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
                    raise HTTPException(status_code=422, detail=f"Base64 reconstruction failed: {e}")
            else:
                raise HTTPException(status_code=422, detail=f"No reconstruction method available for this data format: {e}")
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
