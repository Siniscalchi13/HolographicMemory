"""
Build script for Blazing Fast Holographic Memory
================================================
Cross-platform build using FFTW (Linux) or Accelerate (macOS).
"""

import pybind11
from build_detect import ext_kwargs_common
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

common = ext_kwargs_common()

ext_modules = [
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
]

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
