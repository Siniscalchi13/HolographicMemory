#include "CudaBackend.hpp"

#ifdef USE_CUDA_BACKEND
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <cstring>

namespace holo {

CudaBackend::CudaBackend() {}
CudaBackend::~CudaBackend() {
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (h_pinned_) cudaFreeHost(h_pinned_);
}

bool CudaBackend::available() const noexcept {
    int n=0; if (cudaGetDeviceCount(&n) != cudaSuccess) return false; return n>0;
}

void CudaBackend::ensure_buffers(size_t in_bytes, size_t out_bytes) {
    if (in_bytes > in_bytes_) {
        if (d_input_) cudaFree(d_input_);
        if (h_pinned_) cudaFreeHost(h_pinned_);
        cudaMalloc(&d_input_, in_bytes);
        cudaMallocHost(&h_pinned_, in_bytes);
        in_bytes_ = in_bytes;
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
    size_t in_bytes = (size_t)batch * data_len * sizeof(float);
    size_t out_bytes = (size_t)batch * pattern_dim * sizeof(float);
    ensure_buffers(in_bytes, out_bytes);
    // H2D copy using pinned host buffer
    std::memcpy(h_pinned_, ptr, in_bytes);
    cudaMemcpy(d_input_, h_pinned_, in_bytes, cudaMemcpyHostToDevice);
    // TODO: cuFFT plan + exec (batched), magnitude, scaling into d_output_
    std::vector<float> host_out((size_t)batch * pattern_dim, 0.0f);
    cudaMemcpy(host_out.data(), d_output_, out_bytes, cudaMemcpyDeviceToHost);
    std::vector<std::vector<float>> out; out.reserve(batch);
    for (uint32_t i=0;i<batch;i++) {
        out.emplace_back(host_out.begin()+ (size_t)i*pattern_dim, host_out.begin() + (size_t)(i+1)*pattern_dim);
    }
    return out;
}

} // namespace holo

#endif

