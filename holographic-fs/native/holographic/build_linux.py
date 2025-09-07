"""
Linux build entry for Docker multi-stage builds.

Actions:
- Installs/uses pybind11 and numpy already present in container
- Builds holographic_native and holographic_wave_simd in-place
- Verifies imports post-build
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    import subprocess

    print("[build_linux]", " ".join(cmd))
    p = subprocess.run(cmd, check=True)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main() -> None:
    root = Path(__file__).resolve().parent
    os.chdir(root)
    
    # Build core native module
    _run([sys.executable, "setup.py", "build_ext", "--inplace"])
    # Build SIMD wave variant
    _run([sys.executable, "build_wave.py", "build_ext", "--inplace"])
    # Optional: fast and optimized variants (best-effort)
    try:
        _run([sys.executable, "build_fast.py", "build_ext", "--inplace"])
    except Exception as e:
        print("[build_linux] WARN: fast variant build failed:", e)
    try:
        _run([sys.executable, "build_optimized.py", "build_ext", "--inplace"])
    except Exception as e:
        print("[build_linux] WARN: optimized variant build failed:", e)

    # Copy built modules to target directory
    target_dir = Path("/opt/holo/holographic")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    so_files = list(root.glob("*.so"))
    print(f"[build_linux] Found {len(so_files)} .so files: {[f.name for f in so_files]}")
    
    for so_file in so_files:
        target_file = target_dir / so_file.name
        print(f"[build_linux] Copying {so_file} -> {target_file}")
        import shutil
        shutil.copy2(so_file, target_file)

    # Import check
    sys.path.insert(0, str(root))
    import importlib

    for mod in ["holographic_native", "holographic_wave_simd", "holographic_native_3d"]:
        try:
            importlib.import_module(mod)
            print(f"[build_linux] ✅ import ok: {mod}")
        except Exception as e:
            print(f"[build_linux] ❌ import failed: {mod}: {e}")
            raise


if __name__ == "__main__":
    main()
