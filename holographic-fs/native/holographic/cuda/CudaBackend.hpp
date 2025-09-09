#pragma once

#ifdef USE_CUDA_BACKEND
#include <vector>
#include <string>
#include <cstdint>

namespace holo {

class CudaBackend {
public:
    CudaBackend();
    ~CudaBackend();
    bool available() const noexcept;

    // FFT-based batch encode (cuFFT)
    std::vector<std::vector<float>> batch_encode_fft_ultra(const float* ptr,
                                                           uint32_t batch,
                                                           uint32_t data_len,
                                                           uint32_t pattern_dim);

    struct Metrics {
        double host_ms{0.0};
        double device_ms{0.0};
        uint64_t ops_per_s{0};
        double bandwidth_gbs{0.0};
    };
    Metrics metrics() const { return metrics_; }

private:
    void ensure_buffers(size_t in_bytes, size_t out_bytes);
    void capture_graph(uint32_t batch, uint32_t data_len, uint32_t pattern_dim);

    void* d_input_{nullptr};
    void* d_output_{nullptr};
    void* h_pinned_{nullptr};
    size_t in_bytes_{0}, out_bytes_{0};
    void* stream_{nullptr};
    void* graph_{nullptr};
    void* graph_exec_{nullptr};
    Metrics metrics_{};
};

} // namespace holo

#else

namespace holo { class CudaBackend { public: bool available() const noexcept { return false; } }; }

#endif

