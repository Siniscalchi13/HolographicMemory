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
    create_events();
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

void CudaBackend::create_events() {
    if (!start_event_) cudaEventCreate(&start_event_);
    if (!end_event_) cudaEventCreate(&end_event_);
    if (!fft_start_event_) cudaEventCreate(&fft_start_event_);
    if (!fft_end_event_) cudaEventCreate(&fft_end_event_);
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

    // Rebuild graph if needed
    if (!graph_captured_ || current_pattern_dim_ != pattern_dim) {
        rebuild_graph(batch, data_len, pattern_dim);
    }

    // Host timing
    auto t0 = std::chrono::high_resolution_clock::now();

    // Record overall start event
    cudaEventRecord(start_event_, stream_);

    // Copy input to pinned host buffer; graph reads from pinned buffer to device
    std::memcpy(h_pinned_, ptr, in_bytes);

    // Launch captured graph
    cudaGraphLaunch(graph_exec_, stream_);
    // Record overall end event
    cudaEventRecord(end_event_, stream_);
    cudaStreamSynchronize(stream_);

    // Device time (segmented)
    float device_ms = 0.0f, h2d_ms = 0.0f, fft_ms = 0.0f, d2h_ms = 0.0f;
    cudaEventElapsedTime(&device_ms, start_event_, end_event_);
    // Note: these events surround captured regions; if captured segments change,
    // adjust placements accordingly.
    cudaEventElapsedTime(&fft_ms, fft_start_event_, fft_end_event_);
    // For H2D/D2H, we can place events around memcpy calls pre/post capture in rebuild_graph.
    // Here we report zeros unless events are recorded.
    h2d_ms = 0.0f;
    d2h_ms = 0.0f;

    auto host_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t0).count();

    // Copy back from device (captured graph already includes D2H into d_output_? we copy now)
    std::vector<float> host_out((size_t)batch * pattern_dim, 0.0f);
    cudaMemcpyAsync(host_out.data(), d_output_, out_bytes, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // Metrics
    metrics_.device_ms = device_ms;
    metrics_.h2d_ms = h2d_ms;
    metrics_.fft_ms = fft_ms;
    metrics_.d2h_ms = d2h_ms;
    metrics_.ops_per_s = (device_ms>0.0f) ? (uint64_t)(batch * 1000.0 / device_ms) : 0;
    metrics_.bandwidth_gbs = (device_ms>0.0f) ? ((double)(in_bytes + out_bytes) / (device_ms/1000.0)) / (1024.0*1024.0*1024.0) : 0.0;

    std::vector<std::vector<float>> out; out.reserve(batch);
    for (uint32_t i=0;i<batch;i++) out.emplace_back(host_out.begin()+ (size_t)i*pattern_dim, host_out.begin()+ (size_t)(i+1)*pattern_dim);
    // Populate device metrics crudely (to be refined with events)
    return out;
}

void CudaBackend::rebuild_graph(uint32_t batch, uint32_t data_len, uint32_t pattern_dim) {
    // Destroy previous
    if (graph_exec_) { cudaGraphExecDestroy(graph_exec_); graph_exec_ = nullptr; }
    if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }

    // Plan FFT for (batch, pattern_dim)
    destroy_plan();
    int n[1] = {(int)pattern_dim};
    int istride = 1, ostride = 1;
    int inembed[1] = {(int)pattern_dim};
    int onembed[1] = {(int)pattern_dim};
    int idist = (int)pattern_dim, odist = (int)pattern_dim;
    if (cufftPlanMany(&fft_plan_, 1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, (int)batch) != CUFFT_SUCCESS) {
        throw std::runtime_error("cufftPlanMany failed in rebuild_graph");
    }
    cufftSetStream(fft_plan_, stream_);

    // Begin capture
    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

    // H2D copy from pinned host buffer to device input
    cudaEventRecord(h2d_start_event_, stream_);
    cudaMemcpyAsync(d_input_, h_pinned_, (size_t)batch * data_len * sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaEventRecord(h2d_end_event_, stream_);

    // Pad/truncate to FFT size
    dim3 block(256,1,1);
    dim3 grid((pattern_dim + block.x - 1)/block.x, batch, 1);
    pad_rows_kernel<<<grid, block, 0, stream_>>>((const float*)d_input_, (float*)d_fft_in_, batch, data_len, pattern_dim);

    // FFT
    cudaEventRecord(fft_start_event_, stream_);
    cufftExecR2C(fft_plan_, (cufftReal*)d_fft_in_, (cufftComplex*)d_output_);
    cudaEventRecord(fft_end_event_, stream_);

    // Magnitude
    const uint32_t total = batch * pattern_dim;
    dim3 mgrid((total + 255)/256, 1, 1);
    magnitude_kernel<<<mgrid, 256, 0, stream_>>>((const cufftComplex*)d_output_, (float*)d_output_, total);

    // End capture
    cudaStreamEndCapture(stream_, &graph_);
    cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
    graph_captured_ = true;
    current_pattern_dim_ = pattern_dim;
}

} // namespace holo

#endif
