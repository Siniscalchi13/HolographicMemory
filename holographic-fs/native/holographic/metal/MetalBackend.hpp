#pragma once

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_OSX

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <vector>
#include <string>
#include <chrono>

namespace holo {

class MetalBackend {
public:
    MetalBackend();
    ~MetalBackend();
    
    bool available() const noexcept { return device_ != nullptr && library_ != nullptr; }
    
    // Basic validation kernel
    float vector_add(const std::vector<float>& a, const std::vector<float>& b);
    
    // GPU-accelerated batch store
    std::vector<std::vector<float>> batch_holographic_store(
        const std::vector<std::vector<float>>& batch_data, 
        uint32_t pattern_dimension);
    
    // Performance metrics
    struct PerformanceMetrics {
        double fft_time_ms = 0.0;
        double interference_time_ms = 0.0;
        double batch_store_time_ms = 0.0;
        double search_time_ms = 0.0;
        uint64_t operations_per_second = 0;
    };
    
    PerformanceMetrics get_last_metrics() const { return metrics_; }
    
private:
    void load_shaders();
    id<MTLComputePipelineState> create_pipeline(const std::string& function_name);
    
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> library_ = nil;
    
    // Pipeline states for different kernels
    id<MTLComputePipelineState> pso_vector_add_ = nil;
    id<MTLComputePipelineState> pso_fft_ = nil;
    id<MTLComputePipelineState> pso_interference_ = nil;
    id<MTLComputePipelineState> pso_batch_store_ = nil;
    id<MTLComputePipelineState> pso_similarity_ = nil;
    
    PerformanceMetrics metrics_;
};

} // namespace holo

#endif
#endif
