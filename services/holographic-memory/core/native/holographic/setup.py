"""
Build script for Blazing Fast Holographic Memory
================================================
Cross-platform build using FFTW (Linux) or Accelerate (macOS).
"""

import os
import sys
import pybind11

# Ensure local helpers (e.g., build_detect.py) are importable under PEP 517 builds
sys.path.insert(0, os.path.dirname(__file__))
from build_detect import ext_kwargs_common
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

common = ext_kwargs_common()

# Build profile: default to DEV unless HOLO_BUILD_DEV=0 or HOLO_BUILD_PROFILE=prod
_env_profile = os.environ.get("HOLO_BUILD_PROFILE", "dev").strip().lower()
_env_dev_flag = os.environ.get("HOLO_BUILD_DEV", "1").strip()
DEV_MODE = (_env_profile == "dev" and _env_dev_flag not in ("0", "false", "off"))

ext_modules = []
if DEV_MODE:
    # CPU-native reference modules (DEV-only)
    ext_modules.extend([
        Pybind11Extension(
            "holographic_cpp",
            ["holographic_memory.cpp"],
            include_dirs=[pybind11.get_include()] + common["include_dirs"],
            library_dirs=common["library_dirs"],
            libraries=common["libraries"],
            extra_compile_args=common["extra_compile_args"],
            extra_link_args=common["extra_link_args"],
            define_macros=common["define_macros"],
            cxx_std=17,
        ),
        Pybind11Extension(
            "holographic_cpp_3d",
            ["holographic_native_3d.cpp"],
            include_dirs=[pybind11.get_include()] + common["include_dirs"],
            library_dirs=common["library_dirs"],
            libraries=common["libraries"],
            extra_compile_args=common["extra_compile_args"],
            extra_link_args=common["extra_link_args"],
            define_macros=common["define_macros"],
            cxx_std=17,
        ),
        Pybind11Extension(
            "holographic_event_log",
            ["holographic_event_log.cpp"],
            include_dirs=[pybind11.get_include()] + common["include_dirs"],
            library_dirs=common["library_dirs"],
            libraries=common["libraries"],
            extra_compile_args=common["extra_compile_args"],
            extra_link_args=common["extra_link_args"],
            define_macros=common["define_macros"],
            cxx_std=17,
        ),
        Pybind11Extension(
            "hrr_binding",
            ["hrr_binding.cpp"],
            include_dirs=[pybind11.get_include()] + common["include_dirs"],
            library_dirs=common["library_dirs"],
            libraries=common["libraries"],
            extra_compile_args=common["extra_compile_args"],
            extra_link_args=common["extra_link_args"],
            define_macros=common["define_macros"],
            cxx_std=17,
        ),
    ])
else:
    # Prod: do not build any CPU-native modules here
    pass

setup(
    name="holographic_cpp",
    version="1.0.0",
    author="Philip & Claude",
    description="Blazing Fast C++ Holographic Memory - cross-platform",
    long_description=(
        "Pure C++ implementation of holographic memory using FFTW (Linux) or Accelerate (macOS)."
    ),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.11",
)
