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
set(MODULE_SOURCES gpu_binding.cpp)
if(APPLE)
  list(APPEND MODULE_SOURCES metal/MetalBackend.mm metal/MetalHoloCore.mm)
endif()
if(BUILD_CUDA)
  enable_language(CUDA)
  add_definitions(-DUSE_CUDA_BACKEND)
  list(APPEND MODULE_SOURCES cuda/CudaBackend.cu)
endif()
if(BUILD_ROCM)
  add_definitions(-DUSE_HIP_BACKEND)
  list(APPEND MODULE_SOURCES rocm/HipBackend.cpp)
endif()
pybind11_add_module(holographic_gpu ${MODULE_SOURCES})
  target_compile_options(holographic_gpu PRIVATE -O3 -DNDEBUG)
if(APPLE)
  target_link_libraries(holographic_gpu PRIVATE "-framework Metal" "-framework Foundation" "-framework MetalPerformanceShaders" "-framework MetalPerformanceShadersGraph")
endif()
  # Ensure .metal sources are available at runtime for source-compile
  file(COPY metal/holographic_memory.metal DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/metal)

  add_executable(gpu_bench benchmarks_gpu.cpp GPUBackend.cpp)
  target_compile_options(gpu_bench PRIVATE -O3 -DNDEBUG)
  target_link_libraries(gpu_bench PRIVATE "-framework Metal" "-framework Foundation" "-framework MetalPerformanceShaders" "-framework MetalPerformanceShadersGraph")
elseif(UNIX)
  add_executable(gpu_bench benchmarks_gpu.cpp GPUBackend.cpp)
  if(BUILD_CUDA)
    enable_language(CUDA)
    target_sources(gpu_bench PRIVATE cuda/CudaBackend.cu)
    find_library(CUFFT_LIBRARY cufft)
    target_link_libraries(gpu_bench PRIVATE ${CUFFT_LIBRARY})
  endif()
  if(BUILD_ROCM)
    target_sources(gpu_bench PRIVATE rocm/HipBackend.cpp)
    find_library(HIPFFT_LIBRARY hipfft)
    target_link_libraries(gpu_bench PRIVATE ${HIPFFT_LIBRARY})
  endif()
endif()
