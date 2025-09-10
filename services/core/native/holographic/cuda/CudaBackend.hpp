#pragma once

#ifdef USE_CUDA_BACKEND
#include <vector>
#include <string>
#include <cstdint>
#include "../GPUBackend.hpp"
#include <cuda_runtime.h>
#include <cufft.h>

namespace holo {

class CudaBackend {
public:
    CudaBackend();
    ~CudaBackend();
    bool available() const noexcept;
    bool initialize(const GPUConfig& cfg);

    // FFT-based batch encode (cuFFT)
    std::vector<std::vector<float>> batch_encode_fft_ultra(const float* ptr,
                                                           uint32_t batch,
                                                           uint32_t data_len,
                                                           uint32_t pattern_dim);

    struct Metrics {
        double host_ms{0.0};
        double device_ms{0.0};
        double h2d_ms{0.0};
        double fft_ms{0.0};
        double d2h_ms{0.0};
        uint64_t ops_per_s{0};
        double bandwidth_gbs{0.0};
    };
    Metrics metrics() const { return metrics_; }

    // Device-side analysis for two real vectors: returns (vis, coh, bell_violation, orthogonality)
    std::tuple<float,float,double,float> analyze_metrics(const float* v1, const float* v2, uint32_t dim);

private:
    void ensure_buffers(size_t in_bytes, size_t out_bytes, size_t fft_bytes);
    void destroy_plan();
    void create_events();
    void rebuild_graph(uint32_t batch_size, uint32_t data_len, uint32_t pattern_dim);
    void cleanup() {};

    void* d_input_{nullptr};
    void* d_output_{nullptr};
    void* d_fft_in_{nullptr};
    void* h_pinned_{nullptr};
    size_t in_bytes_{0}, out_bytes_{0}, fft_bytes_{0};
    cudaStream_t stream_ {nullptr};
    cudaGraph_t graph_ {nullptr};
    cudaGraphExec_t graph_exec_ {nullptr};
    cufftHandle fft_plan_ {0};
    bool initialized_ {false};
    bool graph_captured_ {false};
    uint32_t current_pattern_dim_ {0};
    GPUConfig config_ {};
    // Timing events
    cudaEvent_t start_event_ {nullptr};
    cudaEvent_t end_event_ {nullptr};
    cudaEvent_t fft_start_event_ {nullptr};
    cudaEvent_t fft_end_event_ {nullptr};
    cudaEvent_t h2d_start_event_ {nullptr};
    cudaEvent_t h2d_end_event_ {nullptr};
    cudaEvent_t d2h_start_event_ {nullptr};
    cudaEvent_t d2h_end_event_ {nullptr};

    Metrics metrics_{};
};

} // namespace holo

#else

namespace holo { class CudaBackend { public: bool available() const noexcept { return false; } }; }

#endif
