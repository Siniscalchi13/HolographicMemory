// C++ facade for Metal GPU core operations (wraps Objective-C++ MetalBackend)
#pragma once

#ifdef __APPLE__
#include <vector>
#include <string>
#include <memory>
#include "../GPUBackend.hpp"

namespace holo {

class MetalBackend; // fwd

struct PerformanceMetrics {
    double batch_time_ms {0.0};
    double search_time_ms {0.0};
    uint64_t operations_per_second {0};
    double memory_bandwidth_gb_s {0.0};
};

class MetalHoloCore : public IGPUBackend {
public:
    MetalHoloCore();
    ~MetalHoloCore();

    bool available() const noexcept;

    // IGPUBackend interface methods
    bool initialize(const GPUConfig& cfg) override;
    std::vector<std::vector<float>> batch_encode_fft_zero_copy(const float* ptr,
                                                              std::uint32_t batch,
                                                              std::uint32_t data_len,
                                                              std::uint32_t pattern_dim) override;
    GPUMetrics get_metrics() const override;

    // Compute
    void fft_transform(const std::vector<float>& input,
                       std::vector<float>& real_out,
                       std::vector<float>& imag_out);

    // Inverse transform and codebook helpers (wrappers over MetalBackend)
    void ifft_transform(const std::vector<float>& real_in,
                        const std::vector<float>& imag_in,
                        std::vector<float>& time_out);

    void apply_codebook(const std::vector<float>& in_real,
                        const std::vector<float>& in_imag,
                        std::vector<float>& out_real,
                        std::vector<float>& out_imag,
                        uint32_t seed);

    void apply_codebook_conj(const std::vector<float>& in_real,
                             const std::vector<float>& in_imag,
                             std::vector<float>& out_real,
                             std::vector<float>& out_imag,
                             uint32_t seed);

    void accumulate_add_time(std::vector<float>& dst,
                             const std::vector<float>& add);

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
    
    // Removed: bitplane extraction methods - no longer needed with holographic wave reconstruction

private:
    std::unique_ptr<MetalBackend> backend_;
    PerformanceMetrics metrics_{};
};

} // namespace holo

#endif // __APPLE__
