#!/usr/bin/env python3
"""Build the Metal-backed Python extension (macOS only).

Produces: holographic_metal.so in lib.macosx-metal/

This build uses metal-cpp headers locally (third_party/Metal-cpp) to avoid
system-wide installation. If not present, it will be cloned automatically.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


def _pybind11_include_flags() -> list[str]:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"], text=True)
        return shlex.split(out.strip())
    except Exception:
        return []


def main() -> int:
    if sys.platform != "darwin":
        print("Metal build is macOS-only; skipping.")
        return 0

    root = Path(__file__).resolve().parent
    src_dir = root / "metal"
    out = root / "lib.macosx-metal"
    out.mkdir(exist_ok=True, parents=True)

    # Python + pybind11 includes
    python_include = sys.exec_prefix + f"/include/python{sys.version_info.major}.{sys.version_info.minor}"
    pybind11_flags = _pybind11_include_flags()  # e.g. ["-I/usr/include/python3.12", "-I.../site-packages/pybind11/include"]

    # macOS SDK (Framework search path)
    sdk_path = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"

    # Sources: Objective-C++ backend + pybind11 binding (legacy-compatible)
    sources = [
        str(src_dir / "MetalBackend.mm"),
        str(root / "metal_binding.mm"),
    ]

    ext_name = "holographic_metal"
    so_out = out / f"{ext_name}.so"
    args = [
        "clang++",
        "-std=c++17",
        "-fobjc-arc",
        "-ObjC++",
        "-shared",
        "-undefined",
        "dynamic_lookup",
        "-O3",
        "-Wall",
        "-framework", "Metal",
        "-framework", "Foundation",
        "-framework", "MetalPerformanceShaders",
        "-I", python_include,
        "-F", f"{sdk_path}/System/Library/Frameworks",
        "-o", str(so_out),
    ] + pybind11_flags + sources

    print("Building Metal extension (Objective-C++ backend)...")
    print(" ".join(args))
    subprocess.check_call(args, cwd=str(root))
    print("✅ Built:", so_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
