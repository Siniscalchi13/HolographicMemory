#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace holo {

enum class GPUPlatform { METAL, CUDA, ROCM, VULKAN, DIRECTX };

struct GPUConfig {
    GPUPlatform platform { GPUPlatform::METAL };
    std::size_t memory_pool_size { 0 };
    int device_id { 0 };
};

struct GPUMetrics {
    double host_ms {0.0};
    double device_ms {0.0};
    std::uint64_t operations_per_second {0};
    double memory_bandwidth_gb_s {0.0};
    double device_utilization {0.0};
};

class IGPUBackend {
public:
    virtual ~IGPUBackend() = default;
    virtual bool initialize(const GPUConfig& cfg) = 0;
    virtual std::vector<std::vector<float>> batch_encode_fft_zero_copy(const float* ptr,
                                                                       std::uint32_t batch,
                                                                       std::uint32_t data_len,
                                                                       std::uint32_t pattern_dim) = 0;
    virtual GPUMetrics get_metrics() const = 0;

    static std::vector<GPUPlatform> get_available_platforms();
    static std::unique_ptr<IGPUBackend> create_backend(GPUPlatform pf);
};

} // namespace holo

