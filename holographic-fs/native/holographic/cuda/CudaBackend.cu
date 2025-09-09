#include "CudaBackend.hpp"

#ifdef USE_CUDA_BACKEND
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace holo {

// Kernels
__global__ void pad_rows_kernel(const float* __restrict__ in,
                                float* __restrict__ out,
                                uint32_t batch,
                                uint32_t in_cols,
                                uint32_t out_cols) {
    uint32_t b = blockIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || j >= out_cols) return;
    float v = 0.0f;
    if (j < in_cols) v = in[b * in_cols + j];
    out[b * out_cols + j] = v;
}

__global__ void magnitude_kernel(const cufftComplex* __restrict__ in,
                                 float* __restrict__ out,
                                 uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float re = in[idx].x;
    float im = in[idx].y;
    out[idx] = sqrtf(re * re + im * im);
}

CudaBackend::CudaBackend() {}

CudaBackend::~CudaBackend() {
    destroy_plan();
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_) cudaGraphDestroy(graph_);
    if (stream_) cudaStreamDestroy(stream_);
    if (d_input_) cudaFree(d_input_);
    if (d_fft_in_) cudaFree(d_fft_in_);
    if (d_output_) cudaFree(d_output_);
    if (h_pinned_) cudaFreeHost(h_pinned_);
}

bool CudaBackend::available() const noexcept {
    int n=0; if (cudaGetDeviceCount(&n) != cudaSuccess) return false; return n>0;
}

bool CudaBackend::initialize(const GPUConfig& cfg) {
    config_ = cfg;
    if (!available()) return false;
    int dev = cfg.device_id;
    if (cudaSetDevice(dev) != cudaSuccess) return false;
    if (!stream_) cudaStreamCreate(&stream_);
    initialized_ = true;
    return true;
}

void CudaBackend::destroy_plan() {
    if (fft_plan_) { cufftDestroy(fft_plan_); fft_plan_ = 0; }
}

void CudaBackend::ensure_buffers(size_t in_bytes, size_t out_bytes, size_t fft_bytes) {
    if (!stream_) cudaStreamCreate(&stream_);
    if (in_bytes > in_bytes_) {
        if (d_input_) cudaFree(d_input_);
        if (h_pinned_) cudaFreeHost(h_pinned_);
        cudaMalloc(&d_input_, in_bytes);
        cudaMallocHost(&h_pinned_, in_bytes);
        in_bytes_ = in_bytes;
    }
    if (fft_bytes > fft_bytes_) {
        if (d_fft_in_) cudaFree(d_fft_in_);
        cudaMalloc(&d_fft_in_, fft_bytes);
        fft_bytes_ = fft_bytes;
    }
    if (out_bytes > out_bytes_) {
        if (d_output_) cudaFree(d_output_);
        cudaMalloc(&d_output_, out_bytes);
        out_bytes_ = out_bytes;
    }
}

std::vector<std::vector<float>> CudaBackend::batch_encode_fft_ultra(const float* ptr,
                                                                    uint32_t batch,
                                                                    uint32_t data_len,
                                                                    uint32_t pattern_dim) {
    if (!initialized_) initialize({GPUPlatform::CUDA, 0, 0});
    const size_t in_bytes = (size_t)batch * data_len * sizeof(float);
    const size_t fft_bytes = (size_t)batch * pattern_dim * sizeof(float);
    const size_t out_bytes = (size_t)batch * pattern_dim * sizeof(float);
    ensure_buffers(in_bytes, out_bytes, fft_bytes);

    // H2D copy using pinned host buffer
    std::memcpy(h_pinned_, ptr, in_bytes);
    cudaMemcpyAsync(d_input_, h_pinned_, in_bytes, cudaMemcpyHostToDevice, stream_);

    // Pad/truncate rows to pattern_dim
    dim3 block(256,1,1);
    dim3 grid((pattern_dim + block.x - 1)/block.x, batch, 1);
    pad_rows_kernel<<<grid, block, 0, stream_>>>((const float*)d_input_, (float*)d_fft_in_, batch, data_len, pattern_dim);

    // cuFFT plan (batched)
    destroy_plan();
    int n[1] = {(int)pattern_dim};
    int istride = 1, ostride = 1;
    int inembed[1] = {(int)pattern_dim};
    int onembed[1] = {(int)pattern_dim};
    int idist = (int)pattern_dim, odist = (int)pattern_dim;
    if (cufftPlanMany(&fft_plan_, 1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, (int)batch) != CUFFT_SUCCESS) {
        throw std::runtime_error("cufftPlanMany failed");
    }
    cufftSetStream(fft_plan_, stream_);

    // Execute FFT batched
    if (cufftExecR2C(fft_plan_, (cufftReal*)d_fft_in_, (cufftComplex*)d_output_) != CUFFT_SUCCESS) {
        throw std::runtime_error("cufftExecR2C failed");
    }

    // Magnitude
    const uint32_t total = batch * pattern_dim;
    dim3 mgrid((total + 255)/256, 1, 1);
    magnitude_kernel<<<mgrid, 256, 0, stream_>>>((const cufftComplex*)d_output_, (float*)d_output_, total);

    // D2H
    std::vector<float> host_out((size_t)total, 0.0f);
    cudaMemcpyAsync(host_out.data(), d_output_, out_bytes, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // Metrics: host elapsed only; device timing could be added with cudaEvents
    // For now, estimate ops/s as batch per host time when called via adapter
    std::vector<std::vector<float>> out; out.reserve(batch);
    for (uint32_t i=0;i<batch;i++) {
        out.emplace_back(host_out.begin()+ (size_t)i*pattern_dim, host_out.begin()+ (size_t)(i+1)*pattern_dim);
    }
    // Populate device metrics crudely (to be refined with events)
    metrics_.ops_per_s = 0;
    metrics_.bandwidth_gbs = 0.0;
    return out;
}

} // namespace holo

#endif
