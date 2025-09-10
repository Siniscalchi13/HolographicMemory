"""
Cross-platform build detection and flags for holographic modules.

Responsibilities:
- Detect OS/arch and Python ABI
- Discover FFTW/OpenBLAS/Accelerate locations
- Provide normalized compile/link flags for setup/build scripts
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class LibConfig:
    include_dirs: List[str]
    library_dirs: List[str]
    libraries: List[str]
    define_macros: List[Tuple[str, str]]


@dataclass
class BuildConfig:
    os: str
    arch: str
    python_abi: str
    extra_compile_args: List[str]
    extra_link_args: List[str]
    fftw: LibConfig
    blas: LibConfig
    use_accelerate: bool


def _run_cmd(args: List[str]) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, check=False)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:  # pragma: no cover - defensive
        return 1, "", str(e)


def detect_python_abi() -> str:
    # Prefer SOABI (e.g., cpython-312-darwin)
    abi = sysconfig.get_config_var("SOABI") or ""
    if abi:
        return abi
    v = sys.version_info
    return f"cpython-{v.major}{v.minor}-{platform.system().lower()}"


def _brew_prefix(pkg: str) -> str | None:
    code, out, _ = _run_cmd(["brew", "--prefix", pkg])
    return out if code == 0 and out else None


def find_fftw() -> LibConfig:
    system = platform.system()
    if system == "Darwin":
        # Prefer Homebrew
        pref = _brew_prefix("fftw")
        if pref:
            return LibConfig(
                include_dirs=[f"{pref}/include"],
                library_dirs=[f"{pref}/lib"],
                libraries=["fftw3"],
                define_macros=[],
            )
        # MacPorts fallback
        if os.path.exists("/opt/local/include/fftw3.h"):
            return LibConfig(
                include_dirs=["/opt/local/include"],
                library_dirs=["/opt/local/lib"],
                libraries=["fftw3"],
                define_macros=[],
            )
        # Default mac paths
        return LibConfig(
            include_dirs=["/usr/local/include"],
            library_dirs=["/usr/local/lib"],
            libraries=["fftw3"],
            define_macros=[],
        )
    # Linux and others
    # Try pkg-config
    code, out, _ = _run_cmd(["pkg-config", "--cflags", "--libs", "fftw3"])
    if code == 0 and out:
        incs: List[str] = []
        lib_dirs: List[str] = []
        libs: List[str] = []
        for tok in out.split():
            if tok.startswith("-I"):
                incs.append(tok[2:])
            elif tok.startswith("-L"):
                lib_dirs.append(tok[2:])
            elif tok.startswith("-l"):
                libs.append(tok[2:])
        if libs:
            return LibConfig(incs, lib_dirs, libs, [])
    # Fallback common
    return LibConfig(["/usr/include", "/usr/local/include"], ["/usr/lib", "/usr/local/lib"], ["fftw3"], [])


def find_blas() -> LibConfig:
    # Prefer MKL via environment
    cfg = LibConfig(include_dirs=[], library_dirs=[], libraries=[], define_macros=[])
    mklroot = os.environ.get("MKLROOT")
    if mklroot and os.path.exists(mklroot):
        inc = os.path.join(mklroot, "include")
        lib = os.path.join(mklroot, "lib")
        if os.path.isdir(inc):
            cfg.include_dirs.append(inc)
        if os.path.isdir(lib):
            cfg.library_dirs.append(lib)
        cfg.libraries.extend(["mkl_rt"])  # generic runtime
        cfg.define_macros.append(("HAVE_OPENBLAS", "1"))
        return cfg

    # Try pkg-config openblas
    code, out, _ = _run_cmd(["pkg-config", "--cflags", "--libs", "openblas"])
    if code == 0 and out:
        for p in out.split():
            if p.startswith("-I"):
                cfg.include_dirs.append(p[2:])
            elif p.startswith("-L"):
                cfg.library_dirs.append(p[2:])
            elif p.startswith("-l"):
                cfg.libraries.append(p[2:])
        cfg.define_macros.append(("HAVE_OPENBLAS", "1"))
        return cfg

    # Fallback common paths
    for inc in ["/usr/include", "/usr/local/include", "/opt/homebrew/include", "/opt/local/include"]:
        h = os.path.join(inc, "cblas.h")
        if os.path.exists(h) or os.path.exists(os.path.join(inc, "openblas", "cblas.h")):
            cfg.include_dirs.append(inc)
            break
    for lib in ["/usr/lib", "/usr/local/lib", "/opt/homebrew/lib", "/opt/local/lib"]:
        if os.path.exists(os.path.join(lib, "libopenblas.so")) or os.path.exists(
            os.path.join(lib, "libopenblas.dylib")
        ):
            cfg.library_dirs.append(lib)
            cfg.libraries.append("openblas")
            cfg.define_macros.append(("HAVE_OPENBLAS", "1"))
            break
    return cfg


def compute_flags() -> BuildConfig:
    system = platform.system()
    arch = platform.machine().lower()
    python_abi = detect_python_abi()

    extra_compile_args: List[str] = [
        "-std=c++17",
        "-O3",
        "-ffast-math",
        "-funroll-loops",
        "-ftree-vectorize",
    ]
    extra_link_args: List[str] = []

    use_accel = system == "Darwin"

    if system == "Darwin":
        # macOS specific
        extra_compile_args.append("-mmacosx-version-min=10.15")
        if arch == "arm64":
            # Apple Silicon: Accelerate only, avoid OpenMP
            pass
        else:
            # Intel Mac: Allow OpenMP
            extra_compile_args.extend(["-Xpreprocessor", "-fopenmp"])
            extra_link_args.append("-lomp")
        # Link Accelerate
        extra_link_args.extend(["-framework", "Accelerate"])
    else:
        # Linux and others
        extra_compile_args.append("-fopenmp")
        if arch not in ("aarch64", "arm64"):
            extra_compile_args.extend(["-march=native", "-mavx2", "-mfma"])
        else:
            extra_compile_args.append("-march=native")
        extra_link_args.append("-fopenmp")

    fftw = find_fftw()
    blas = find_blas()

    # On macOS, add Accelerate headers if available
    if system == "Darwin":
        hdr = "/System/Library/Frameworks/Accelerate.framework/Headers"
        if os.path.exists(hdr):
            fftw.include_dirs.append(hdr)

    return BuildConfig(
        os=system,
        arch=arch,
        python_abi=python_abi,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        fftw=fftw,
        blas=blas,
        use_accelerate=(system == "Darwin"),
    )


# Convenience exported helpers for setup scripts
def ext_kwargs_common() -> Dict:
    cfg = compute_flags()
    return {
        "include_dirs": cfg.fftw.include_dirs + cfg.blas.include_dirs,
        "library_dirs": cfg.fftw.library_dirs + cfg.blas.library_dirs,
        "libraries": cfg.fftw.libraries + cfg.blas.libraries,
        "extra_compile_args": cfg.extra_compile_args,
        "extra_link_args": cfg.extra_link_args,
        "define_macros": cfg.blas.define_macros,
        "cxx_std": 17,
    }


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_macos() -> bool:
    return platform.system() == "Darwin"


__all__ = [
    "BuildConfig",
    "LibConfig",
    "compute_flags",
    "detect_python_abi",
    "ext_kwargs_common",
    "find_fftw",
    "find_blas",
    "is_linux",
    "is_macos",
]

