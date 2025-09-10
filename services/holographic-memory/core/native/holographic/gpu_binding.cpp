// Cross-platform GPU binding.
// Exposes a single Python module `holographic_gpu` with a portable
// wrapper that selects the best available backend (Metal/CUDA/ROCm).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <string>

#include "GPUBackend.hpp"
#include "hwp_v4_decode.hpp"
#include "metal/MetalHoloCore.hpp"

namespace py = pybind11;

namespace {

// Map enum to string for Python friendliness
static inline const char* to_str(holo::GPUPlatform pf) {
    switch (pf) {
        case holo::GPUPlatform::METAL: return "metal";
        case holo::GPUPlatform::CUDA:  return "cuda";
        case holo::GPUPlatform::ROCM:  return "rocm";
        case holo::GPUPlatform::VULKAN: return "vulkan";
        case holo::GPUPlatform::DIRECTX: return "directx";
        default: return "unknown";
    }
}

static inline bool from_str(const std::string& s, holo::GPUPlatform& out) {
    std::string t;
    t.resize(s.size());
    std::transform(s.begin(), s.end(), t.begin(), [](unsigned char c){ return (char)std::tolower(c); });
    if (t == "metal") { out = holo::GPUPlatform::METAL; return true; }
    if (t == "cuda")  { out = holo::GPUPlatform::CUDA;  return true; }
    if (t == "rocm")  { out = holo::GPUPlatform::ROCM;  return true; }
    if (t == "vulkan") { out = holo::GPUPlatform::VULKAN; return true; }
    if (t == "directx") { out = holo::GPUPlatform::DIRECTX; return true; }
    return false;
}

class HolographicGPUWrapper {
public:
    HolographicGPUWrapper() = default;

    // Initialize with optional platform string; auto-selects otherwise.
    bool initialize(const std::string& platform = std::string()) {
        std::vector<holo::GPUPlatform> avail = holo::IGPUBackend::get_available_platforms();
        if (!platform.empty()) {
            holo::GPUPlatform pf;
            if (!from_str(platform, pf)) return false;
            backend_ = holo::IGPUBackend::create_backend(pf);
        } else {
            if (avail.empty()) return false;
            backend_ = holo::IGPUBackend::create_backend(avail.front());
        }
        if (!backend_) return false;
        holo::GPUConfig cfg; // defaults are fine for now
        return backend_->initialize(cfg);
    }

    // Backward-compat: some callers probe `is_available`/`available`/`initialize`
    bool is_available() const { return (bool)backend_; }
    bool available()    const { return (bool)backend_; }

    // Zero-copy NumPy batch encode (preferred)
    std::vector<std::vector<float>> batch_encode_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                                                       std::uint32_t pattern_dim) {
        if (!backend_) throw std::runtime_error("GPU backend not initialized");
        if (arr.ndim() != 2) throw std::runtime_error("array must be 2D (batch, data_len)");
        std::uint32_t batch = (std::uint32_t)arr.shape(0);
        std::uint32_t data_len = (std::uint32_t)arr.shape(1);
        const float* ptr = arr.data();
        auto out = backend_->batch_encode_fft_zero_copy(ptr, batch, data_len, pattern_dim);
        last_dim_ = pattern_dim;
        last_patterns_.clear();
        for (const auto& v : out) last_patterns_.push_back(v);
        return out;
    }

    // Convenience: accept list[list[float]] for drop-in compatibility
    std::vector<std::vector<float>> batch_encode(const std::vector<std::vector<float>>& batch,
                                                 std::uint32_t pattern_dim) {
        if (!backend_) throw std::runtime_error("GPU backend not initialized");
        if (batch.empty()) return {};
        const std::uint32_t bs = (std::uint32_t)batch.size();
        const std::uint32_t dl = (std::uint32_t)batch.front().size();
        // Flatten to contiguous buffer for zero-copy path
        std::vector<float> flat;
        flat.reserve(static_cast<size_t>(bs) * dl);
        for (const auto& v : batch) {
            if ((std::uint32_t)v.size() != dl) throw std::runtime_error("inconsistent row sizes in batch");
            flat.insert(flat.end(), v.begin(), v.end());
        }
        auto out = backend_->batch_encode_fft_zero_copy(flat.data(), bs, dl, pattern_dim);
        // Cache for device analysis
        last_dim_ = pattern_dim;
        last_patterns_.clear();
        for (const auto& v : out) last_patterns_.push_back(v);
        return out;
    }

    // Metrics accessor
    holo::GPUMetrics metrics() const {
        if (!backend_) return holo::GPUMetrics{};
        return backend_->get_metrics();
    }

    // Backward-compat alias used by some callers
    holo::GPUMetrics get_last_metrics() const { return metrics(); }

    static std::vector<std::string> available_platforms() {
        std::vector<std::string> out;
        for (auto pf : holo::IGPUBackend::get_available_platforms()) out.emplace_back(to_str(pf));
        return out;
    }

    // Decode .hwp payloads (H4K8/HWP4V001) to raw bytes
    std::string decode_hwp_v4(py::bytes payload) {
        std::string b = payload;
        auto out = holo::decode_hwp_v4_to_bytes(reinterpret_cast<const std::uint8_t*>(b.data()), b.size());
        return std::string(reinterpret_cast<const char*>(out.data()), out.size());
    }

    // Convenience: read file and decode
    std::string retrieve_bytes(const std::string& path) {
        auto out = holo::decode_hwp_v4_file_to_bytes(path);
        return std::string(reinterpret_cast<const char*>(out.data()), out.size());
    }

    // ----------------- Math API Parity (host-side P0) -----------------
    struct LayerState {
        std::string name;
        std::size_t dimension {0};
        double importance_weight {1.0};
        double load_estimate {0.0};
        double target_snr {1.0};
        double current_snr {0.0};
        double capacity_used {0.0};
    };

    void initialize_7layer_decomposition(std::size_t total_budget) {
        total_budget_ = total_budget ? total_budget : 1024;
        static const char* NAMES[7] = {"Identity","Knowledge","Experience","Preference","Context","Wisdom","Vault"};
        for (std::size_t k = 0; k < 7; ++k) {
            layers_[k].name = NAMES[k];
            layers_[k].importance_weight = (k==1?1.3:(k==4?1.1:1.0));
            layers_[k].target_snr = 1.0;
            layers_[k].load_estimate = std::max(1.0, layers_[k].importance_weight * 100.0);
        }
        optimize_layer_dimensions();
        layers_initialized_ = true;
    }

    void optimize_layer_dimensions() {
        if (total_budget_ == 0) return;
        std::array<double,7> q{};
        double sum_q = 0.0;
        for (std::size_t k = 0; k < 7; ++k) {
            double a = layers_[k].importance_weight;
            double n_eff = std::max(1.0, layers_[k].load_estimate);
            q[k] = (a*a)/n_eff;
            sum_q += q[k];
        }
        if (sum_q <= 0.0) {
            for (std::size_t k = 0; k < 7; ++k) layers_[k].dimension = total_budget_/7;
            layers_[0].dimension += (total_budget_%7);
            return;
        }
        std::size_t allocated = 0;
        for (std::size_t k = 0; k < 7; ++k) {
            std::size_t dim_k = static_cast<std::size_t>(std::llround(double(total_budget_) * (q[k]/sum_q)));
            layers_[k].dimension = std::max<std::size_t>(1, dim_k);
            allocated += layers_[k].dimension;
        }
        if (allocated != total_budget_) {
            std::size_t diff = (allocated < total_budget_) ? (total_budget_ - allocated) : (allocated - total_budget_);
            std::size_t max_k = 0;
            for (std::size_t k = 1; k < 7; ++k) if (layers_[k].dimension > layers_[max_k].dimension) max_k = k;
            if (allocated < total_budget_) layers_[max_k].dimension += diff; else {
                layers_[max_k].dimension = (layers_[max_k].dimension > diff) ? (layers_[max_k].dimension - diff) : 1;
            }
        }
    }

    double calculate_layer_snr(std::size_t layer_idx) const {
        if (layer_idx >= 7) return 0.0;
        double n_eff = std::max(1.0, layers_[layer_idx].load_estimate);
        double d_eff = double(layers_[layer_idx].dimension);
        return std::sqrt(std::max(0.0, d_eff / n_eff));
    }

    void update_layer_snrs() {
        for (std::size_t k = 0; k < 7; ++k) {
            layers_[k].current_snr = calculate_layer_snr(k);
            layers_[k].capacity_used = layers_[k].load_estimate / std::max(1.0, double(layers_[k].dimension));
        }
    }

    bool enforce_capacity_theorem() {
        bool rebalanced = false;
        for (std::size_t k = 0; k < 7; ++k) {
            double required = layers_[k].target_snr * layers_[k].target_snr * layers_[k].load_estimate;
            if (double(layers_[k].dimension) < required) {
                layers_[k].dimension = static_cast<std::size_t>(std::ceil(required));
                rebalanced = true;
            }
        }
        if (rebalanced) {
            std::size_t used = 0; for (auto& L : layers_) used += L.dimension;
            if (used > total_budget_) {
                double scale = double(total_budget_) / double(used);
                for (std::size_t k = 0; k < 6; ++k) {
                    layers_[k].dimension = std::max<std::size_t>(1, std::size_t(std::floor(layers_[k].dimension * scale)));
                }
                std::size_t partial = 0; for (std::size_t k=0;k<6;++k) partial += layers_[k].dimension;
                layers_[6].dimension = (partial < total_budget_) ? (total_budget_ - partial) : 1;
            }
        }
        return rebalanced;
    }

    py::dict get_layer_stats() const {
        py::dict stats;
        for (std::size_t k = 0; k < 7; ++k) {
            py::dict layer;
            layer["name"] = layers_[k].name;
            layer["dimension"] = layers_[k].dimension;
            layer["importance_weight"] = layers_[k].importance_weight;
            layer["load_estimate"] = layers_[k].load_estimate;
            layer["target_snr"] = layers_[k].target_snr;
            layer["current_snr"] = layers_[k].current_snr;
            layer["capacity_used"] = layers_[k].capacity_used;
            stats[std::to_string(k).c_str()] = layer;
        }
        stats["total_budget"] = total_budget_;
        stats["layers_initialized"] = layers_initialized_;
        return stats;
    }

    py::dict validate_wave_properties() const {
        py::dict out;
        float field_norm = 0.0f;
        float ortho = 0.0f;
        if (backend_ && !last_patterns_.empty()) {
            const float* a = last_patterns_[0].data();
            const float* b = (last_patterns_.size() > 1 ? last_patterns_[1].data() : last_patterns_[0].data());
            holo::IGPUBackend::DeviceAnalysisResult r;
            if (backend_->analyze_device_metrics(a, b, last_dim_, r)) {
                ortho = r.orthogonality;
                // Approximate norm from coherence*|b|*|a| only if both >0; else leave 0
            }
            // Host-side norm for reporting (cheap)
            double n=0.0; for (std::uint32_t i=0;i<last_dim_;++i) n += double(a[i])*double(a[i]);
            field_norm = float(std::sqrt(n));
        }
        out["field_normalization"] = field_norm;
        out["layer_orthogonality_score"] = ortho;
        bool compliant = true;
        for (const auto& L : layers_) {
            double required = L.target_snr * L.target_snr * L.load_estimate;
            if (double(L.dimension) < required) { compliant = false; break; }
        }
        out["capacity_theorem_compliant"] = compliant;
        return out;
    }

    py::dict analyze_interference_patterns() const {
        py::dict out;
        float vis = 0.0f, coh = 0.0f; double bell = 0.0;
        if (backend_ && !last_patterns_.empty()) {
            const float* a = last_patterns_[0].data();
            const float* b = (last_patterns_.size() > 1 ? last_patterns_[1].data() : last_patterns_[0].data());
            holo::IGPUBackend::DeviceAnalysisResult r;
            if (backend_->analyze_device_metrics(a, b, last_dim_, r)) { vis = r.visibility; coh = r.coherence; bell = r.bell_violation; }
        }
        out["wave_visibility"] = vis;
        out["phase_coherence"] = coh;
        out["bell_violation_measure"] = bell;
        out["bell_test_passed"] = (bell > 0.1);
        return out;
    }

    double validate_bell_inequality() const {
        if (backend_ && !last_patterns_.empty()) {
            const float* a = last_patterns_[0].data();
            holo::IGPUBackend::DeviceAnalysisResult r;
            if (backend_->analyze_device_metrics(a, a, last_dim_, r)) return r.bell_violation;
        }
        auto E = [](double x, double y){ return -std::cos(x - y); };
        double aa=0.0, ap=M_PI/2.0, b=M_PI/4.0, bp=-M_PI/4.0;
        double S = E(aa,b) + E(aa,bp) + E(ap,b) - E(ap,bp);
        return S - 2.0;
    }

    bool layers_initialized() const { return layers_initialized_; }

    // ============================================================================
    // GPU COMPRESSION PIPELINE - KERNEL 1: QUANTIZATION
    // ============================================================================
    
    // Layer-specific quantization parameters for 7-layer holographic decomposition
    struct QuantizationParams {
        // Phase precision (bits) per layer - critical for recall accuracy
        std::array<uint32_t, 7> phase_bits;     // [12, 12, 12, 10, 10, 8, 6] - decreasing precision
        // Amplitude precision (bits) per layer
        std::array<uint32_t, 7> amplitude_bits; // [12, 12, 10, 8, 8, 6, 4] - decreasing precision
        // Quantization step sizes (computed from bit precision)
        std::array<float, 7> phase_step;    // 2π / (2^phase_bits)
        std::array<float, 7> amplitude_step; // max_amplitude / (2^amplitude_bits)
        // Maximum phase error bounds per layer (degrees)
        std::array<float, 7> max_phase_error; // [0.1, 0.1, 0.1, 0.5, 0.5, 2.0, 2.0]
        
        QuantizationParams() {
            // Initialize with default values
            phase_bits = {12, 12, 12, 10, 10, 8, 6};
            amplitude_bits = {12, 12, 10, 8, 8, 6, 4};
            for (int i = 0; i < 7; ++i) {
                phase_step[i] = 2.0f * M_PI / (1 << phase_bits[i]);
                amplitude_step[i] = 1.0f / (1 << amplitude_bits[i]);
            }
            max_phase_error = {0.1f, 0.1f, 0.1f, 0.5f, 0.5f, 2.0f, 2.0f};
        }
    };
    
    // GPU Quantization - Basic quantization without error tracking
    std::vector<std::vector<float>> gpu_holographic_quantize(
        const std::vector<std::vector<float>>& input_real,
        const std::vector<std::vector<float>>& input_imag,
        uint32_t layer_index,
        const QuantizationParams& params) {
        
        if (!backend_) throw std::runtime_error("GPU backend not initialized");
        
        // For now, return a placeholder implementation
        // TODO: Implement actual GPU quantization through the backend
        std::vector<std::vector<float>> result;
        result.reserve(input_real.size());
        
        for (size_t i = 0; i < input_real.size(); ++i) {
            std::vector<float> quantized;
            quantized.reserve(input_real[i].size());
            
            for (size_t j = 0; j < input_real[i].size(); ++j) {
                // Simple quantization: round to nearest step
                float real_val = input_real[i][j];
                float imag_val = input_imag[i][j];
                
                // Convert to polar coordinates
                float amplitude = std::sqrt(real_val * real_val + imag_val * imag_val);
                float phase = std::atan2(imag_val, real_val);
                if (phase < 0.0f) phase += 2.0f * M_PI;
                
                // Quantize
                float quantized_amplitude = std::round(amplitude / params.amplitude_step[layer_index]) * params.amplitude_step[layer_index];
                float quantized_phase = std::round(phase / params.phase_step[layer_index]) * params.phase_step[layer_index];
                
                // Convert back to Cartesian
                float quantized_real = quantized_amplitude * std::cos(quantized_phase);
                quantized.push_back(quantized_real);
            }
            result.push_back(std::move(quantized));
        }
        
        return result;
    }
    
    // GPU Quantization with Error Bounds Validation
    py::tuple gpu_holographic_quantize_with_validation(
        const std::vector<std::vector<float>>& input_real,
        const std::vector<std::vector<float>>& input_imag,
        uint32_t layer_index,
        const QuantizationParams& params) {
        
        if (!backend_) throw std::runtime_error("GPU backend not initialized");
        
        // For now, return a placeholder implementation
        // TODO: Implement actual GPU quantization through the backend
        std::vector<std::vector<float>> quantized_real, quantized_imag;
        std::vector<std::vector<float>> phase_errors, amplitude_errors;
        
        quantized_real.reserve(input_real.size());
        quantized_imag.reserve(input_imag.size());
        phase_errors.reserve(input_real.size());
        amplitude_errors.reserve(input_imag.size());
        
        for (size_t i = 0; i < input_real.size(); ++i) {
            std::vector<float> q_real, q_imag, p_errors, a_errors;
            q_real.reserve(input_real[i].size());
            q_imag.reserve(input_imag[i].size());
            p_errors.reserve(input_real[i].size());
            a_errors.reserve(input_imag[i].size());
            
            for (size_t j = 0; j < input_real[i].size(); ++j) {
                float real_val = input_real[i][j];
                float imag_val = input_imag[i][j];
                
                // Convert to polar coordinates
                float amplitude = std::sqrt(real_val * real_val + imag_val * imag_val);
                float phase = std::atan2(imag_val, real_val);
                if (phase < 0.0f) phase += 2.0f * M_PI;
                
                // Quantize
                float quantized_amplitude = std::round(amplitude / params.amplitude_step[layer_index]) * params.amplitude_step[layer_index];
                float quantized_phase = std::round(phase / params.phase_step[layer_index]) * params.phase_step[layer_index];
                
                // Calculate errors
                float phase_error = std::abs(phase - quantized_phase);
                float amplitude_error = std::abs(amplitude - quantized_amplitude);
                
                // Convert back to Cartesian
                float quantized_real = quantized_amplitude * std::cos(quantized_phase);
                float quantized_imag = quantized_amplitude * std::sin(quantized_phase);
                
                q_real.push_back(quantized_real);
                q_imag.push_back(quantized_imag);
                p_errors.push_back(phase_error);
                a_errors.push_back(amplitude_error);
            }
            
            quantized_real.push_back(std::move(q_real));
            quantized_imag.push_back(std::move(q_imag));
            phase_errors.push_back(std::move(p_errors));
            amplitude_errors.push_back(std::move(a_errors));
        }
        
        return py::make_tuple(quantized_real, quantized_imag, phase_errors, amplitude_errors);
    }
    
    // GPU Quantization Statistics Collection
    std::array<float, 4> gpu_quantization_statistics(
        const std::vector<std::vector<float>>& phase_errors,
        const std::vector<std::vector<float>>& amplitude_errors) {
        
        if (phase_errors.empty() || amplitude_errors.empty()) {
            return {0.0f, 0.0f, 0.0f, 0.0f};
        }
        
        float max_phase_error = 0.0f;
        float max_amplitude_error = 0.0f;
        float sum_phase_error = 0.0f;
        float sum_amplitude_error = 0.0f;
        size_t total_coefficients = 0;
        
        for (size_t i = 0; i < phase_errors.size(); ++i) {
            for (size_t j = 0; j < phase_errors[i].size(); ++j) {
                float p_err = phase_errors[i][j];
                float a_err = amplitude_errors[i][j];
                
                max_phase_error = std::max(max_phase_error, p_err);
                max_amplitude_error = std::max(max_amplitude_error, a_err);
                sum_phase_error += p_err;
                sum_amplitude_error += a_err;
                total_coefficients++;
            }
        }
        
        float mean_phase_error = total_coefficients > 0 ? sum_phase_error / total_coefficients : 0.0f;
        float mean_amplitude_error = total_coefficients > 0 ? sum_amplitude_error / total_coefficients : 0.0f;
        
        return {max_phase_error, max_amplitude_error, mean_phase_error, mean_amplitude_error};
    }
    
    // ============================================================================
    // GPU COMPRESSION PIPELINE - KERNEL 2: BITPLANE EXTRACTION
    // ============================================================================
    
    // Bitplane extraction parameters for zero-tree coding
    // Removed: BitplaneParams struct - no longer needed with holographic wave reconstruction
    
    // Holographic Wave Reconstruction - Direct wave reconstruction bypassing bitplane extraction
    std::vector<std::vector<float>> gpu_holographic_wave_reconstruction(
        const std::vector<std::vector<float>>& original_real,
        const std::vector<std::vector<float>>& original_imag,
        const std::vector<float>& original_phase,
        uint32_t layer_index) {
        
        if (!backend_ || original_real.empty() || original_imag.empty() || original_phase.empty()) {
            return {};
        }
        
        // For now, return the original data as holographic reconstruction
        // In a real implementation, this would use the Metal shader
        std::vector<std::vector<float>> result(original_real.size());
        for (size_t b = 0; b < original_real.size(); ++b) {
            result[b].resize(original_real[b].size() * 2); // 2x for interleaved real/imag
            for (size_t i = 0; i < original_real[b].size(); ++i) {
                result[b][i * 2] = original_real[b][i];     // Real part
                result[b][i * 2 + 1] = original_imag[b][i]; // Imaginary part
            }
        }
        
        return result;
    }
    
    // Removed: bitplane extraction methods - no longer needed with holographic wave reconstruction

    // GPU Compression Pipeline - Kernel 3: Sparse Encoding
    std::tuple<std::vector<float>, std::vector<float>, std::vector<uint32_t>, uint32_t>
    gpu_sparse_encoding(const std::vector<float>& input_real,
                       const std::vector<float>& input_imag,
                       float threshold,
                       uint32_t max_sparse_count) {
        if (!backend_) {
            throw std::runtime_error("GPU backend not initialized");
        }
        
        // For now, return a simple implementation
        // In a real implementation, this would use the Metal shader
        std::vector<float> sparse_real;
        std::vector<float> sparse_imag;
        std::vector<uint32_t> sparse_indices;
        
        for (size_t i = 0; i < input_real.size() && sparse_indices.size() < max_sparse_count; ++i) {
            float magnitude = sqrt(input_real[i] * input_real[i] + input_imag[i] * input_imag[i]);
            if (magnitude > threshold) {
                sparse_real.push_back(input_real[i]);
                sparse_imag.push_back(input_imag[i]);
                sparse_indices.push_back(static_cast<uint32_t>(i));
            }
        }
        
        return {sparse_real, sparse_imag, sparse_indices, static_cast<uint32_t>(sparse_indices.size())};
    }
    
    // GPU Compression Pipeline - Kernel 4: Entropy Coding
    std::tuple<std::vector<uint8_t>, uint32_t>
    gpu_entropy_coding(const std::vector<float>& sparse_real,
                      const std::vector<float>& sparse_imag,
                      const std::vector<uint32_t>& sparse_indices,
                      uint32_t sparse_count) {
        if (!backend_) {
            throw std::runtime_error("GPU backend not initialized");
        }
        
        // Simple entropy coding implementation
        std::vector<uint8_t> encoded_data(sparse_count * 6); // 6 bytes per coefficient
        
        for (uint32_t i = 0; i < sparse_count; ++i) {
            uint32_t offset = i * 6;
            uint32_t index = sparse_indices[i];
            float real = sparse_real[i];
            float imag = sparse_imag[i];
            
            // Calculate magnitude and phase
            float magnitude = sqrt(real * real + imag * imag);
            float phase = atan2(imag, real);
            
            // Quantize magnitude and phase to 8-bit
            uint8_t mag_quantized = static_cast<uint8_t>(std::min(255u, static_cast<uint32_t>(magnitude * 255.0f)));
            uint8_t phase_quantized = static_cast<uint8_t>((phase + M_PI) / (2.0 * M_PI) * 255.0f);
            
            // Pack data: [index(4 bytes), magnitude(1 byte), phase(1 byte)]
            encoded_data[offset] = (index >> 24) & 0xFF;
            encoded_data[offset + 1] = (index >> 16) & 0xFF;
            encoded_data[offset + 2] = (index >> 8) & 0xFF;
            encoded_data[offset + 3] = index & 0xFF;
            encoded_data[offset + 4] = mag_quantized;
            encoded_data[offset + 5] = phase_quantized;
        }
        
        return {encoded_data, sparse_count * 6};
    }
    
    // GPU Compression Pipeline - Kernel 5: Entropy Decoding
    std::tuple<std::vector<float>, std::vector<float>, std::vector<uint32_t>, uint32_t>
    gpu_entropy_decoding(const std::vector<uint8_t>& encoded_data,
                        uint32_t encoded_size) {
        if (!backend_) {
            throw std::runtime_error("GPU backend not initialized");
        }
        
        uint32_t coeff_count = encoded_size / 6;
        std::vector<float> decoded_real(coeff_count);
        std::vector<float> decoded_imag(coeff_count);
        std::vector<uint32_t> decoded_indices(coeff_count);
        
        for (uint32_t i = 0; i < coeff_count; ++i) {
            uint32_t offset = i * 6;
            
            // Decode index
            uint32_t index = ((uint32_t)encoded_data[offset] << 24) |
                           ((uint32_t)encoded_data[offset + 1] << 16) |
                           ((uint32_t)encoded_data[offset + 2] << 8) |
                           (uint32_t)encoded_data[offset + 3];
            
            // Decode quantized magnitude and phase
            uint8_t mag_quantized = encoded_data[offset + 4];
            uint8_t phase_quantized = encoded_data[offset + 5];
            
            // Convert back to float
            float magnitude = static_cast<float>(mag_quantized) / 255.0f;
            float phase = (static_cast<float>(phase_quantized) / 255.0f) * 2.0f * M_PI - M_PI;
            
            // Convert back to Cartesian coordinates
            decoded_real[i] = magnitude * cos(phase);
            decoded_imag[i] = magnitude * sin(phase);
            decoded_indices[i] = index;
        }
        
        return {decoded_real, decoded_imag, decoded_indices, coeff_count};
    }
    
    // GPU Compression Pipeline - Kernel 6: Sparse Decoding
    std::tuple<std::vector<float>, std::vector<float>>
    gpu_sparse_decoding(const std::vector<float>& sparse_real,
                       const std::vector<float>& sparse_imag,
                       const std::vector<uint32_t>& sparse_indices,
                       uint32_t sparse_count,
                       uint32_t output_size) {
        if (!backend_) {
            throw std::runtime_error("GPU backend not initialized");
        }
        
        std::vector<float> output_real(output_size, 0.0f);
        std::vector<float> output_imag(output_size, 0.0f);
        
        // Place sparse coefficients back into dense representation
        for (uint32_t i = 0; i < sparse_count; ++i) {
            uint32_t index = sparse_indices[i];
            if (index < output_size) {
                output_real[index] = sparse_real[i];
                output_imag[index] = sparse_imag[i];
            }
        }
        
        return {output_real, output_imag};
    }

private:
    std::unique_ptr<holo::IGPUBackend> backend_;
    mutable std::vector<std::vector<float>> last_patterns_;
    mutable std::uint32_t last_dim_ {0};
    std::array<LayerState, 7> layers_{};
    std::size_t total_budget_ {1024};
    bool layers_initialized_ {false};
};

} // namespace

PYBIND11_MODULE(holographic_gpu, m) {
    m.doc() = "GPU-accelerated holographic backend (Metal/CUDA/ROCm)";

    py::class_<holo::GPUMetrics>(m, "PerformanceMetrics")
        .def_readonly("host_ms", &holo::GPUMetrics::host_ms)
        .def_readonly("device_ms", &holo::GPUMetrics::device_ms)
        .def_readonly("h2d_time_ms", &holo::GPUMetrics::h2d_time_ms)
        .def_readonly("fft_time_ms", &holo::GPUMetrics::fft_time_ms)
        .def_readonly("d2h_time_ms", &holo::GPUMetrics::d2h_time_ms)
        .def_readonly("operations_per_second", &holo::GPUMetrics::operations_per_second)
        .def_readonly("memory_bandwidth_gb_s", &holo::GPUMetrics::memory_bandwidth_gb_s)
        .def_readonly("device_utilization", &holo::GPUMetrics::device_utilization);

    py::class_<HolographicGPUWrapper>(m, "HolographicGPU")
        .def(py::init<>())
        .def("initialize", &HolographicGPUWrapper::initialize, py::arg("platform") = std::string())
        .def("is_available", &HolographicGPUWrapper::is_available)
        .def("available", &HolographicGPUWrapper::available)
        .def("batch_encode", &HolographicGPUWrapper::batch_encode, py::arg("batch"), py::arg("pattern_dim"))
        .def("batch_encode_numpy", &HolographicGPUWrapper::batch_encode_numpy, py::arg("array"), py::arg("pattern_dim"))
        .def("metrics", &HolographicGPUWrapper::metrics)
        .def("get_last_metrics", &HolographicGPUWrapper::get_last_metrics)
        .def("decode_hwp_v4", &HolographicGPUWrapper::decode_hwp_v4, py::arg("payload"))
        .def("retrieve_bytes", &HolographicGPUWrapper::retrieve_bytes, py::arg("path"))
        // Math API Parity (host-side P0)
        .def("initialize_7layer_decomposition", &HolographicGPUWrapper::initialize_7layer_decomposition, py::arg("total_budget"))
        .def("optimize_layer_dimensions", &HolographicGPUWrapper::optimize_layer_dimensions)
        .def("get_layer_stats", &HolographicGPUWrapper::get_layer_stats)
        .def("calculate_layer_snr", &HolographicGPUWrapper::calculate_layer_snr, py::arg("layer_idx"))
        .def("update_layer_snrs", &HolographicGPUWrapper::update_layer_snrs)
        .def("enforce_capacity_theorem", &HolographicGPUWrapper::enforce_capacity_theorem)
        .def("validate_wave_properties", &HolographicGPUWrapper::validate_wave_properties)
        .def("analyze_interference_patterns", &HolographicGPUWrapper::analyze_interference_patterns)
        .def("validate_bell_inequality", &HolographicGPUWrapper::validate_bell_inequality)
        .def_property_readonly("layers_initialized", &HolographicGPUWrapper::layers_initialized)
        // GPU Compression Pipeline - Kernel 1: Quantization
        .def("gpu_holographic_quantize", &HolographicGPUWrapper::gpu_holographic_quantize,
             py::arg("input_real"), py::arg("input_imag"), py::arg("layer_index"), py::arg("params"))
        .def("gpu_holographic_quantize_with_validation", &HolographicGPUWrapper::gpu_holographic_quantize_with_validation,
             py::arg("input_real"), py::arg("input_imag"), py::arg("layer_index"), py::arg("params"))
        .def("gpu_quantization_statistics", &HolographicGPUWrapper::gpu_quantization_statistics,
             py::arg("phase_errors"), py::arg("amplitude_errors"))
        .def("gpu_holographic_wave_reconstruction", &HolographicGPUWrapper::gpu_holographic_wave_reconstruction,
             py::arg("original_real"), py::arg("original_imag"), py::arg("original_phase"), py::arg("layer_index"));
        // GPU Compression Pipeline - Kernel 2: Bitplane Extraction
        // Removed: bitplane extraction method bindings - no longer needed with holographic wave reconstruction

    // QuantizationParams class
    py::class_<HolographicGPUWrapper::QuantizationParams>(m, "QuantizationParams")
        .def(py::init<>())
        .def_readwrite("phase_bits", &HolographicGPUWrapper::QuantizationParams::phase_bits)
        .def_readwrite("amplitude_bits", &HolographicGPUWrapper::QuantizationParams::amplitude_bits)
        .def_readwrite("phase_step", &HolographicGPUWrapper::QuantizationParams::phase_step)
        .def_readwrite("amplitude_step", &HolographicGPUWrapper::QuantizationParams::amplitude_step)
        .def_readwrite("max_phase_error", &HolographicGPUWrapper::QuantizationParams::max_phase_error);

    // Removed: BitplaneParams class binding - no longer needed with holographic wave reconstruction

    // GPU Compression Pipeline - Kernel 3: Sparse Encoding
    m.def("gpu_sparse_encoding", [](const std::vector<float>& input_real,
                                   const std::vector<float>& input_imag,
                                   float threshold,
                                   uint32_t max_sparse_count) {
        auto wrapper = HolographicGPUWrapper();
        auto result = wrapper.gpu_sparse_encoding(input_real, input_imag, threshold, max_sparse_count);
        return py::make_tuple(
            std::get<0>(result),  // sparse_real
            std::get<1>(result),  // sparse_imag
            std::get<2>(result),  // sparse_indices
            std::get<3>(result)   // sparse_count
        );
    }, "GPU Sparse Encoding - Convert dense coefficients to sparse representation");

    // GPU Compression Pipeline - Kernel 4: Entropy Coding
    m.def("gpu_entropy_coding", [](const std::vector<float>& sparse_real,
                                  const std::vector<float>& sparse_imag,
                                  const std::vector<uint32_t>& sparse_indices,
                                  uint32_t sparse_count) {
        auto wrapper = HolographicGPUWrapper();
        auto result = wrapper.gpu_entropy_coding(sparse_real, sparse_imag, sparse_indices, sparse_count);
        return py::make_tuple(
            std::get<0>(result),  // encoded_data
            std::get<1>(result)   // encoded_size
        );
    }, "GPU Entropy Coding - Encode sparse coefficients with entropy coding");

    // GPU Compression Pipeline - Kernel 5: Entropy Decoding
    m.def("gpu_entropy_decoding", [](const std::vector<uint8_t>& encoded_data,
                                    uint32_t encoded_size) {
        auto wrapper = HolographicGPUWrapper();
        auto result = wrapper.gpu_entropy_decoding(encoded_data, encoded_size);
        return py::make_tuple(
            std::get<0>(result),  // decoded_real
            std::get<1>(result),  // decoded_imag
            std::get<2>(result),  // decoded_indices
            std::get<3>(result)   // decoded_count
        );
    }, "GPU Entropy Decoding - Decode entropy-coded data back to sparse coefficients");

    // GPU Compression Pipeline - Kernel 6: Sparse Decoding
    m.def("gpu_sparse_decoding", [](const std::vector<float>& sparse_real,
                                   const std::vector<float>& sparse_imag,
                                   const std::vector<uint32_t>& sparse_indices,
                                   uint32_t sparse_count,
                                   uint32_t output_size) {
        auto wrapper = HolographicGPUWrapper();
        auto result = wrapper.gpu_sparse_decoding(sparse_real, sparse_imag, sparse_indices, sparse_count, output_size);
        return py::make_tuple(
            std::get<0>(result),  // output_real
            std::get<1>(result)   // output_imag
        );
    }, "GPU Sparse Decoding - Convert sparse representation back to dense coefficients");

    m.def("available_platforms", &HolographicGPUWrapper::available_platforms,
          "Return a list of available GPU platforms (e.g., ['cuda','metal']).");
}
