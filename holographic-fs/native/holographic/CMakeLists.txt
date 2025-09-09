cmake_minimum_required(VERSION 3.16)
project(holographic_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(pybind11 REQUIRED)

# Platform detection and options
option(BUILD_CUDA "Build CUDA backend" OFF)
option(BUILD_ROCM "Build ROCm backend" OFF)
option(BUILD_METAL "Build Metal backend" ON)

# Platform detection
if(APPLE)
    set(PLATFORM "METAL")
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders)
    find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph)
    enable_language(OBJCXX)
elseif(UNIX)
    # Check for CUDA
    if(BUILD_CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(PLATFORM "CUDA")
        find_library(CUFFT_LIBRARY cufft)
    else()
        # Check for ROCm
        if(BUILD_ROCM)
            find_path(ROCM_PATH hip/hip_runtime.h)
            if(ROCM_PATH)
                set(PLATFORM "ROCM")
                find_library(HIPFFT_LIBRARY hipfft)
            endif()
        endif()
    endif()
elseif(WIN32)
    set(PLATFORM "DIRECTX")
    # Add DirectX support
endif()

# Source files
set(SOURCES
    gpu_binding.cpp
    GPUBackend.cpp
)

# Platform-specific sources
if(PLATFORM STREQUAL "METAL")
    list(APPEND SOURCES
        metal/MetalBackend.mm
        metal/MetalHoloCore.mm
    )
    set(FRAMEWORKS ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${MPS_FRAMEWORK} ${MPSGRAPH_FRAMEWORK})
elseif(PLATFORM STREQUAL "CUDA")
    list(APPEND SOURCES
        cuda/CudaBackend.cu
    )
    set(LIBRARIES ${CUFFT_LIBRARY})
    set(CUDA_SEPARABLE_COMPILATION ON)
elseif(PLATFORM STREQUAL "ROCM")
    list(APPEND SOURCES
        rocm/HipBackend.cpp
    )
    set(LIBRARIES ${HIPFFT_LIBRARY})
endif()

# Create pybind11 module
pybind11_add_module(holographic_gpu ${SOURCES})

# Link libraries
if(FRAMEWORKS)
    target_link_libraries(holographic_gpu PRIVATE ${FRAMEWORKS})
endif()

if(LIBRARIES)
    target_link_libraries(holographic_gpu PRIVATE ${LIBRARIES})
endif()

# Compiler flags
target_compile_definitions(holographic_gpu PRIVATE PLATFORM_${PLATFORM})
target_compile_options(holographic_gpu PRIVATE -O3 -DNDEBUG)

# Create native benchmark executable
set(BENCH_SOURCES benchmarks_gpu.cpp GPUBackend.cpp)

# Add platform-specific sources to benchmark
if(PLATFORM STREQUAL "METAL")
    list(APPEND BENCH_SOURCES
        metal/MetalBackend.mm
        metal/MetalHoloCore.mm
    )
elseif(PLATFORM STREQUAL "CUDA")
    list(APPEND BENCH_SOURCES
        cuda/CudaBackend.cu
    )
elseif(PLATFORM STREQUAL "ROCM")
    list(APPEND BENCH_SOURCES
        rocm/HipBackend.cpp
    )
endif()

add_executable(gpu_bench ${BENCH_SOURCES})
target_compile_options(gpu_bench PRIVATE -O3 -DNDEBUG)
target_compile_definitions(gpu_bench PRIVATE PLATFORM_${PLATFORM})

if(FRAMEWORKS)
    target_link_libraries(gpu_bench PRIVATE ${FRAMEWORKS})
endif()

if(LIBRARIES)
    target_link_libraries(gpu_bench PRIVATE ${LIBRARIES})
endif()

# CUDA specific settings
if(PLATFORM STREQUAL "CUDA")
    set_property(TARGET holographic_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET gpu_bench PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# Ensure .metal sources are available at runtime for source-compile
if(PLATFORM STREQUAL "METAL")
    file(COPY metal/holographic_memory.metal DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/metal)
endif()
