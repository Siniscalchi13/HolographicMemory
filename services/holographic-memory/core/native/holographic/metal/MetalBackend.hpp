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
    
    // ============================================================================
    // GPU COMPRESSION PIPELINE - KERNEL 1: QUANTIZATION
    // ============================================================================
    
    // Layer-specific quantization parameters for 7-layer holographic decomposition
    struct QuantizationParams {
        // Phase precision (bits) per layer - critical for recall accuracy
        uint32_t phase_bits[7];     // [12, 12, 12, 10, 10, 8, 6] - decreasing precision
        // Amplitude precision (bits) per layer
        uint32_t amplitude_bits[7]; // [12, 12, 10, 8, 8, 6, 4] - decreasing precision
        // Quantization step sizes (computed from bit precision)
        float phase_step[7];    // 2π / (2^phase_bits)
        float amplitude_step[7]; // max_amplitude / (2^amplitude_bits)
        // Maximum phase error bounds per layer (degrees)
        float max_phase_error[7]; // [0.1, 0.1, 0.1, 0.5, 0.5, 2.0, 2.0]
    };
    
    // GPU Quantization - Basic quantization without error tracking
    std::vector<std::vector<float>> gpu_holographic_quantize(
        const std::vector<std::vector<float>>& input_real,
        const std::vector<std::vector<float>>& input_imag,
        uint32_t layer_index,
        const QuantizationParams& params);
    
    // GPU Quantization with Error Bounds Validation
    // Returns {quantized_real, quantized_imag, phase_errors, amplitude_errors}
    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>, 
               std::vector<std::vector<float>>, std::vector<std::vector<float>>>
    gpu_holographic_quantize_with_validation(
        const std::vector<std::vector<float>>& input_real,
        const std::vector<std::vector<float>>& input_imag,
        uint32_t layer_index,
        const QuantizationParams& params);
    
    // GPU Quantization Statistics Collection
    // Returns {max_phase_err, max_amp_err, mean_phase_err, mean_amp_err}
    std::array<float, 4> gpu_quantization_statistics(
        const std::vector<std::vector<float>>& phase_errors,
        const std::vector<std::vector<float>>& amplitude_errors);
    
    // ============================================================================
    // GPU COMPRESSION PIPELINE - KERNEL 2: BITPLANE EXTRACTION
    // ============================================================================
    
    // Bitplane extraction parameters for zero-tree coding
    struct BitplaneParams {
        uint32_t max_bitplanes[7];        // Maximum bitplanes per layer
        uint32_t significance_threshold[7]; // Threshold for significance
        uint32_t zero_tree_depth[7];      // Maximum zero-tree depth
        float amplitude_scale[7];         // Amplitude scaling factors
    };
    
    // GPU Bitplane Extraction - MSB-first encoding with zero-tree coding
    // Returns {bitplanes, significance_map, zero_tree_map}
    std::tuple<std::vector<std::vector<uint32_t>>, std::vector<uint32_t>, std::vector<uint32_t>>
    gpu_bitplane_extraction(
        const std::vector<std::vector<float>>& quantized_real,
        const std::vector<std::vector<float>>& quantized_imag,
        uint32_t layer_index,
        const BitplaneParams& params);
    
    // ============================================================================
    // GPU COMPRESSION PIPELINE - KERNEL 3: SPARSE ENCODING
    // ============================================================================
    
    // GPU Sparse Encoding - Convert dense coefficients to sparse representation
    // Returns {sparse_real, sparse_imag, sparse_indices, sparse_count}
    std::tuple<std::vector<float>, std::vector<float>, std::vector<uint32_t>, uint32_t>
    gpu_sparse_encoding(
        const std::vector<float>& input_real,
        const std::vector<float>& input_imag,
        float threshold,
        uint32_t max_sparse_count);
    
    // ============================================================================
    // GPU COMPRESSION PIPELINE - KERNEL 4: ENTROPY CODING
    // ============================================================================
    
    // GPU Entropy Coding - Encode sparse coefficients with entropy coding
    // Returns {encoded_data, encoded_size}
    std::tuple<std::vector<uint8_t>, uint32_t>
    gpu_entropy_coding(
        const std::vector<float>& sparse_real,
        const std::vector<float>& sparse_imag,
        const std::vector<uint32_t>& sparse_indices,
        uint32_t sparse_count);
    
    // ============================================================================
    // GPU COMPRESSION PIPELINE - KERNEL 5: ENTROPY DECODING
    // ============================================================================
    
    // GPU Entropy Decoding - Decode entropy-coded data back to sparse coefficients
    // Returns {decoded_real, decoded_imag, decoded_indices, decoded_count}
    std::tuple<std::vector<float>, std::vector<float>, std::vector<uint32_t>, uint32_t>
    gpu_entropy_decoding(
        const std::vector<uint8_t>& encoded_data,
        uint32_t encoded_size);
    
    // ============================================================================
    // GPU COMPRESSION PIPELINE - KERNEL 6: SPARSE DECODING
    // ============================================================================
    
    // GPU Sparse Decoding - Convert sparse representation back to dense coefficients
    // Returns {output_real, output_imag}
    std::tuple<std::vector<float>, std::vector<float>>
    gpu_sparse_decoding(
        const std::vector<float>& sparse_real,
        const std::vector<float>& sparse_imag,
        const std::vector<uint32_t>& sparse_indices,
        uint32_t sparse_count,
        uint32_t output_size);
    
    // GPU Bitplane Statistics - Analyze bitplane distribution and compression potential
    // Returns {bitplane_entropy, significance_ratio, zero_tree_ratio, compression_ratio}
    std::array<float, 4> gpu_bitplane_statistics(
        const std::vector<std::vector<uint32_t>>& bitplanes,
        const std::vector<uint32_t>& significance_map,
        const std::vector<uint32_t>& zero_tree_map,
        uint32_t max_bitplanes);
    
    // GPU Bitplane Reconstruction - Reconstruct coefficients from bitplanes with phase preservation
    std::vector<std::vector<float>> gpu_bitplane_reconstruction(
        const std::vector<std::vector<uint32_t>>& bitplanes,
        const std::vector<uint32_t>& significance_map,
        const std::vector<float>& original_phase,
        uint32_t layer_index,
        const BitplaneParams& params);
    
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
    
    // GPU Compression Pipeline - Kernel 1: Quantization
    id<MTLComputePipelineState> pso_quantize_ = nil;
    id<MTLComputePipelineState> pso_quantize_validation_ = nil;
    id<MTLComputePipelineState> pso_quantize_stats_ = nil;
    
    // GPU Compression Pipeline - Kernel 2: Bitplane Extraction
    id<MTLComputePipelineState> pso_bitplane_extraction_ = nil;
    id<MTLComputePipelineState> pso_bitplane_statistics_ = nil;
    id<MTLComputePipelineState> pso_bitplane_reconstruction_ = nil;
    
    // GPU Compression Pipeline - Kernel 3: Sparse Encoding
    id<MTLComputePipelineState> sparse_encoding_pipeline_ = nil;
    
    // GPU Compression Pipeline - Kernel 4: Entropy Coding
    id<MTLComputePipelineState> entropy_coding_pipeline_ = nil;
    
    // GPU Compression Pipeline - Kernel 5: Entropy Decoding
    id<MTLComputePipelineState> entropy_decoding_pipeline_ = nil;
    
    // GPU Compression Pipeline - Kernel 6: Sparse Decoding
    id<MTLComputePipelineState> sparse_decoding_pipeline_ = nil;

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
