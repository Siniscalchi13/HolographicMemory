#pragma once
#ifdef __APPLE__

#include <Metal/Metal.hpp>
#include <vector>
#include <string>
#include <cstdint>

namespace holo {

class MetalHolographicBackend {
public:
    MetalHolographicBackend();
    ~MetalHolographicBackend();
    bool initialize();
    bool is_available() const { return device_ != nullptr && library_ != nullptr; }

    // Batch encode: returns flattened patterns (batch_size x pattern_dim)
    std::vector<float> batch_encode(const std::vector<float>& flat_input,
                                    uint32_t batch_size,
                                    uint32_t data_length,
                                    uint32_t pattern_dim);

    // Simple similarity: query (dim) vs stored (pattern_count x dim)
    std::vector<float> similarity_search(const std::vector<float>& query,
                                         const std::vector<float>& stored,
                                         uint32_t pattern_count,
                                         uint32_t dim);

    struct PerformanceMetrics {
        double batch_encode_time_ms {0.0};
        uint64_t operations_per_second {0};
        double memory_bandwidth_gb_s {0.0};
    };

    PerformanceMetrics get_last_metrics() const { return metrics_; }

private:
    void load_shader_library();
    MTL::ComputePipelineState* create_pipeline(const char* fn);
    void dispatch_kernel(MTL::ComputePipelineState* pso,
                         const std::vector<MTL::Buffer*>& buffers,
                         uint32_t grid_size);

    MTL::Device* device_ {nullptr};
    MTL::CommandQueue* queue_ {nullptr};
    MTL::Library* library_ {nullptr};

    MTL::ComputePipelineState* pso_fft_ {nullptr};
    MTL::ComputePipelineState* pso_interf_ {nullptr};
    MTL::ComputePipelineState* pso_batch_ {nullptr};
    MTL::ComputePipelineState* pso_sim_ {nullptr};

    PerformanceMetrics metrics_ {};
};

} // namespace holo

#endif

