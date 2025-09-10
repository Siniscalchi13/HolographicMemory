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

// Parallel dot/norm reduction with atomics for global accumulation
__global__ void dot_norm_kernel_cuda(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ out_dot,
                                     float* __restrict__ out_n1,
                                     float* __restrict__ out_n2,
                                     uint32_t n) {
    float local_dot = 0.0f;
    float local_n1  = 0.0f;
    float local_n2  = 0.0f;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < n; i += stride) {
        float x = a[i];
        float y = b[i];
        local_dot += x * y;
        local_n1  += x * x;
        local_n2  += y * y;
    }
    // In-block reduction via warp shuffles
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        local_dot += __shfl_down_sync(0xffffffff, local_dot, offset);
        local_n1  += __shfl_down_sync(0xffffffff, local_n1,  offset);
        local_n2  += __shfl_down_sync(0xffffffff, local_n2,  offset);
    }
    // One thread per warp atomically accumulates to global
    if ((threadIdx.x & (warpSize-1)) == 0) {
        atomicAdd(out_dot, local_dot);
        atomicAdd(out_n1,  local_n1);
        atomicAdd(out_n2,  local_n2);
    }
}

__global__ void corr_offset_kernel_cuda(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float* __restrict__ out,
                                        uint32_t n,
                                        uint32_t o1,
                                        uint32_t o2) {
    float local = 0.0f;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < n; i += stride) {
        float x = a[(i + o1) % n];
        float y = b[(i + o2) % n];
        local += x * y;
    }
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(0xffffffff, local, offset);
    }
    if ((threadIdx.x & (warpSize-1)) == 0) atomicAdd(out, local);
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

std::tuple<float,float,double,float> CudaBackend::analyze_metrics(const float* v1, const float* v2, uint32_t dim) {
    if (!initialized_) initialize({GPUPlatform::CUDA, 0, 0});
    if (!v1 || !v2 || dim == 0) return {0.0f, 0.0f, 0.0, 0.0f};
    // Allocate device buffers for inputs and outputs
    float *d_a=nullptr, *d_b=nullptr, *d_dot=nullptr, *d_n1=nullptr, *d_n2=nullptr;
    float *d_c00=nullptr, *d_c01=nullptr, *d_c20=nullptr, *d_c21=nullptr;
    cudaMalloc(&d_a, dim*sizeof(float));
    cudaMalloc(&d_b, dim*sizeof(float));
    cudaMalloc(&d_dot, sizeof(float)); cudaMalloc(&d_n1, sizeof(float)); cudaMalloc(&d_n2, sizeof(float));
    cudaMalloc(&d_c00, sizeof(float)); cudaMalloc(&d_c01, sizeof(float)); cudaMalloc(&d_c20, sizeof(float)); cudaMalloc(&d_c21, sizeof(float));
    cudaMemsetAsync(d_dot, 0, sizeof(float), stream_);
    cudaMemsetAsync(d_n1,  0, sizeof(float), stream_);
    cudaMemsetAsync(d_n2,  0, sizeof(float), stream_);
    cudaMemsetAsync(d_c00, 0, sizeof(float), stream_);
    cudaMemsetAsync(d_c01, 0, sizeof(float), stream_);
    cudaMemsetAsync(d_c20, 0, sizeof(float), stream_);
    cudaMemsetAsync(d_c21, 0, sizeof(float), stream_);
    cudaMemcpyAsync(d_a, v1, dim*sizeof(float), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_b, v2, dim*sizeof(float), cudaMemcpyHostToDevice, stream_);

    int threads = 256;
    int blocks = (int)std::min<uint32_t>( (dim + threads - 1)/threads, 1024 );
    dot_norm_kernel_cuda<<<blocks, threads, 0, stream_>>>(d_a, d_b, d_dot, d_n1, d_n2, dim);
    corr_offset_kernel_cuda<<<blocks, threads, 0, stream_>>>(d_a, d_b, d_c00, dim, 0, 0);
    corr_offset_kernel_cuda<<<blocks, threads, 0, stream_>>>(d_a, d_b, d_c01, dim, 0, 1);
    corr_offset_kernel_cuda<<<blocks, threads, 0, stream_>>>(d_a, d_b, d_c20, dim, 2, 0);
    corr_offset_kernel_cuda<<<blocks, threads, 0, stream_>>>(d_a, d_b, d_c21, dim, 2, 1);
    cudaStreamSynchronize(stream_);

    float fdot=0.0f, fn1=0.0f, fn2=0.0f, c00=0.0f, c01=0.0f, c20=0.0f, c21=0.0f;
    cudaMemcpy(&fdot, d_dot, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&fn1, d_n1, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&fn2, d_n2, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c00, d_c00, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c01, d_c01, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c20, d_c20, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c21, d_c21, sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_dot); cudaFree(d_n1); cudaFree(d_n2);
    cudaFree(d_c00); cudaFree(d_c01); cudaFree(d_c20); cudaFree(d_c21);

    float vis=0.0f, coh=0.0f, ortho = std::fabs(fdot);
    if (fn1 > 0.0f && fn2 > 0.0f) {
        float n1 = std::sqrt(fn1), n2 = std::sqrt(fn2);
        float num = std::fabs(fdot);
        coh = num / (n1 * n2);
        vis = (num*num) / ((n1*n1) * (n2*n2));
    }
    double S = (double)c00/(double)dim + (double)c01/(double)dim + (double)c20/(double)dim - (double)c21/(double)dim;
    double bell_violation = S - 2.0;
    return {vis, coh, bell_violation, ortho};
}

} // namespace holo

#endif
