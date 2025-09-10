"""
Build the FAST variant of holographic memory.
Cross-platform with FFTW (Linux) / Accelerate (macOS).
"""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

from build_detect import ext_kwargs_common

common = ext_kwargs_common()

ext_modules = [
    Pybind11Extension(
        "holographic_fast",
        ["holographic_memory_fast.cpp"],
        include_dirs=[pybind11.get_include()] + common["include_dirs"],
        library_dirs=common["library_dirs"],
        libraries=common["libraries"],
        extra_compile_args=common["extra_compile_args"],
        extra_link_args=common["extra_link_args"],
        define_macros=common["define_macros"],
        cxx_std=17,
    ),
]

setup(name="holographic_fast", ext_modules=ext_modules, cmdclass={"build_ext": build_ext}, zip_safe=False)
