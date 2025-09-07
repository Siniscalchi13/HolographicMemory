"""
Build the WAVE-SIMD version of holographic memory
"""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

from build_detect import compute_flags

cfg = compute_flags()

extra_compile_args = ["-std=c++17", "-O3", "-ffast-math", "-funroll-loops", "-ftree-vectorize"]
extra_link_args: list[str] = []

if cfg.os == "Darwin":
    extra_compile_args.append("-mmacosx-version-min=10.15")
    extra_link_args.extend(["-framework", "Accelerate"])  # consistent with SIMD math on macOS
else:
    # Linux: allow native tuning
    if cfg.arch not in ("aarch64", "arm64"):
        extra_compile_args.extend(["-march=native", "-mavx2", "-mfma"])
    else:
        extra_compile_args.append("-march=native")

ext_modules = [
    Pybind11Extension(
        "holographic_wave_simd",
        ["holographic_wave_simd.cpp"],
        include_dirs=[pybind11.get_include()] + (cfg.fftw.include_dirs if cfg.use_accelerate else []),
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=17,
    ),
]

setup(name="holographic_wave_simd", ext_modules=ext_modules, cmdclass={"build_ext": build_ext}, zip_safe=False)
