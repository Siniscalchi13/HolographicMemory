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

    // FFT-like transform (naive kernel). Outputs real/imag as separate arrays.
    void fft_transform(const std::vector<float>& input,
                       std::vector<float>& real_out,
                       std::vector<float>& imag_out);

    // Similarity search: query (dim) vs stored (pattern_count x dim). Returns similarities per pattern.
    std::vector<float> similarity_search(const std::vector<float>& query,
                                         const std::vector<std::vector<float>>& stored);

    // FFT-based batch encode path (can be backed by MPS FFT or optimized kernel)
    std::vector<std::vector<float>> batch_encode_fft(
        const std::vector<std::vector<float>>& batch_data,
        uint32_t pattern_dimension);
    // Ultra-optimized path with MPSGraph caching and persistent buffers
    std::vector<std::vector<float>> batch_encode_fft_ultra(
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

    // Compute basic device-side metrics for two real vectors of length dim.
    // Returns {visibility, coherence, bell_violation, orthogonality}
    std::tuple<float,float,double,float> analyze_metrics(const float* v1, const float* v2, uint32_t dim);
    
private:
    void load_shaders();
    id<MTLComputePipelineState> create_pipeline(const std::string& function_name);
    void initialize_mps_fft();
    
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> library_ = nil;
    
    // Pipeline states for different kernels
    id<MTLComputePipelineState> pso_vector_add_ = nil;
    id<MTLComputePipelineState> pso_fft_ = nil;
    id<MTLComputePipelineState> pso_interference_ = nil;
    id<MTLComputePipelineState> pso_dot_norm_ = nil;
    id<MTLComputePipelineState> pso_corr_off_ = nil;
    id<MTLComputePipelineState> pso_batch_store_ = nil;
    id<MTLComputePipelineState> pso_similarity_ = nil;
    id<MTLComputePipelineState> pso_batch_store_fft_ = nil;

    // MPSGraph caching (Objective-C objects, ARC-managed)
    id mps_graph_fft_ = nil;   // MPSGraph*
    id mps_graph_input_ = nil; // MPSGraphTensor*
    id mps_graph_mag_ = nil;   // MPSGraphTensor*

    // Persistent buffer pools
    id input_pool_ = nil;   // NSMutableArray<id<MTLBuffer>>*
    id output_pool_ = nil;  // NSMutableArray<id<MTLBuffer>>*
    id pool_lock_ = nil;    // NSLock*
    
    PerformanceMetrics metrics_;
};

} // namespace holo

#endif
#endif
