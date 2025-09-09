"""
Run Uvicorn FastAPI server from inside the bundled app.

This script is optional; the app currently launches uvicorn directly.
Kept as a fallback or for future in-process CPython embedding.
"""
import os
import sys
from pathlib import Path

def _add_paths():
    # App bundle layout: Contents/Resources
    res = Path(__file__).resolve().parent
    site = res / "Python" / "site-packages"
    svcs = res / "services"
    holo = res / "holographic-fs"
    for p in [str(site), str(res), str(svcs), str(holo)]:
        if p not in sys.path:
            sys.path.insert(0, p)

def main():
    _add_paths()
    os.environ.setdefault("HOLO_ROOT", str(Path.home() / "Library/Application Support/HolographicMemory"))
    os.environ.setdefault("GRID_SIZE", "64")
    os.environ.setdefault("HOLO_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,capacitor://localhost, null")
    # Warm up GPU module if available (Metal backend)
    try:
        import holographic_gpu as _hg  # type: ignore
        if hasattr(_hg, "HolographicGPU"):
            be = _hg.HolographicGPU()
            be.initialize()
    except Exception:
        pass
    import uvicorn
    uvicorn.run("services.api.app:app", host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()

