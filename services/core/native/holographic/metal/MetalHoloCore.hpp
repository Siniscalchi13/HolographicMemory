// C++ facade for Metal GPU core operations (wraps Objective-C++ MetalBackend)
#pragma once

#ifdef __APPLE__
#include <vector>
#include <string>
#include <memory>

namespace holo {

class MetalBackend; // fwd

struct PerformanceMetrics {
    double batch_time_ms {0.0};
    double search_time_ms {0.0};
    uint64_t operations_per_second {0};
    double memory_bandwidth_gb_s {0.0};
};

class MetalHoloCore {
public:
    MetalHoloCore();
    ~MetalHoloCore();

    bool available() const noexcept;

    // Compute
    void fft_transform(const std::vector<float>& input,
                       std::vector<float>& real_out,
                       std::vector<float>& imag_out);

    std::vector<std::vector<float>> batch_encode(const std::vector<std::vector<float>>& batch_data,
                                                 uint32_t pattern_dim);
    std::vector<std::vector<float>> batch_encode_fft(const std::vector<std::vector<float>>& batch_data,
                                                     uint32_t pattern_dim);
    std::vector<std::vector<float>> batch_encode_fft_ultra(const std::vector<std::vector<float>>& batch_data,
                                                           uint32_t pattern_dim);

    // Zero-copy NumPy path (accepts contiguous CPU array pointer)
    std::vector<std::vector<float>> batch_encode_from_ptr(const float* ptr,
                                                          uint32_t batch,
                                                          uint32_t data_len,
                                                          uint32_t pattern_dim,
                                                          bool use_ultra);

    std::vector<float> similarity_search(const std::vector<float>& query,
                                         const std::vector<std::vector<float>>& stored);

    PerformanceMetrics metrics() const { return metrics_; }

    // Host-backed device analysis: uploads two vectors and runs device-side kernels
    // to compute interference visibility, phase coherence, CHSH violation, and orthogonality.
    struct DeviceAnalysisResult { float visibility; float coherence; double bell_violation; float orthogonality; };
    DeviceAnalysisResult analyze_metrics_hostback(const float* v1, const float* v2, std::uint32_t dim);

private:
    std::unique_ptr<MetalBackend> backend_;
    PerformanceMetrics metrics_{};
};

} // namespace holo

#endif // __APPLE__
