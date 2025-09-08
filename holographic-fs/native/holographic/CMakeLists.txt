cmake_minimum_required(VERSION 3.16)
project(holographic_memory LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(FFTW3 REQUIRED)

pybind11_add_module(holographic_native holographic_memory.cpp)
pybind11_add_module(holographic_native_3d holographic_native_3d.cpp)

target_link_libraries(holographic_native PRIVATE
  Eigen3::Eigen
  FFTW3::fftw3
)

target_link_libraries(holographic_native_3d PRIVATE
  FFTW3::fftw3
)

target_compile_options(holographic_native PRIVATE
  -O3
  -march=native
  -ffast-math
  -DNDEBUG
)

target_compile_options(holographic_native_3d PRIVATE
  -O3
  -march=native
  -ffast-math
  -DNDEBUG
)

# Metal GPU backend (Objective-C++)
if(APPLE)
  enable_language(OBJCXX)
  pybind11_add_module(holographic_gpu
    metal/MetalBackend.mm
    metal/MetalHoloCore.mm
    gpu_binding.cpp)
  target_compile_options(holographic_gpu PRIVATE -O3 -DNDEBUG)
  target_link_libraries(holographic_gpu PRIVATE "-framework Metal" "-framework Foundation" "-framework MetalPerformanceShaders")
  # Ensure .metal sources are available at runtime for source-compile
  file(COPY metal/holographic_memory.metal DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/metal)

  add_executable(gpu_bench benchmarks_gpu.cpp metal/MetalBackend.mm metal/MetalHoloCore.mm)
  target_compile_options(gpu_bench PRIVATE -O3 -DNDEBUG)
  target_link_libraries(gpu_bench PRIVATE "-framework Metal" "-framework Foundation" "-framework MetalPerformanceShaders")
endif()
