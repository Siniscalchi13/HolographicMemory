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

    // Encode bytes via seeded codebook multiplexing into time-domain psi
    py::tuple encode_superpose_bytes(py::bytes data, std::uint32_t chunk_size, std::uint32_t seed_base) {
#ifdef PLATFORM_METAL
        std::string s = data;
        if (s.empty()) return py::make_tuple(std::vector<float>{}, 0u, std::vector<std::uint32_t>{}, std::vector<std::uint32_t>{});
        auto core = holo::MetalHoloCore();
        if (!core.available()) throw std::runtime_error("GPU backend not available");
        // Normalize to [0,1]
        std::vector<float> psi; psi.assign(s.size(), 0.0f);
        std::vector<std::uint32_t> seeds;
        std::vector<std::uint32_t> sizes;
        const std::uint32_t dim = (std::uint32_t)s.size();
        for (std::size_t off = 0, idx = 0; off < s.size(); off += chunk_size, ++idx) {
            std::size_t clen = std::min<std::size_t>(chunk_size, s.size() - off);
            std::vector<float> chunk(clen);
            for (std::size_t i = 0; i < clen; ++i) chunk[i] = (float)((unsigned char)s[off + i]) / 255.0f;
            // Pad to dim for FFT kernel simplicity
            std::vector<float> sig(dim, 0.0f); std::copy(chunk.begin(), chunk.end(), sig.begin());
            // FFT
            std::vector<float> re, im; core.fft_transform(sig, re, im);
            // Apply codebook with seeded phase per bin
            std::uint32_t seed = seed_base ^ (std::uint32_t)idx;
            std::vector<float> re2, im2; core.apply_codebook(re, im, re2, im2, seed);
            // iFFT to time
            std::vector<float> t; core.ifft_transform(re2, im2, t);
            // Accumulate into psi
            core.accumulate_add_time(psi, t);
            seeds.push_back(seed);
            sizes.push_back((std::uint32_t)clen);
        }
        return py::make_tuple(psi, dim, seeds, sizes);
#else
        (void)data; (void)chunk_size; (void)seed_base; throw std::runtime_error("GPU platform not supported");
#endif
    }

    // Decode bytes from time-domain psi using seeded conjugate codebook correlation
    py::bytes decode_superposed_bytes(const std::vector<float>& psi_time,
                                      std::uint32_t dim,
                                      const std::vector<std::uint32_t>& seeds,
                                      const std::vector<std::uint32_t>& sizes) {
#ifdef PLATFORM_METAL
        auto core = holo::MetalHoloCore();
        if (!core.available()) throw std::runtime_error("GPU backend not available");
        if (psi_time.size() != (size_t)dim) throw std::runtime_error("psi_time size mismatch");
        // FFT of psi
        std::vector<float> psi_re, psi_im; core.fft_transform(psi_time, psi_re, psi_im);
        std::string out; out.reserve(dim);
        for (std::size_t idx = 0, off = 0; idx < seeds.size(); ++idx) {
            std::vector<float> re2, im2; core.apply_codebook_conj(psi_re, psi_im, re2, im2, seeds[idx]);
            std::vector<float> t; core.ifft_transform(re2, im2, t);
            for (std::size_t i = 0; i < sizes[idx]; ++i) {
                float v = t[i]; if (v < 0.0f) v = 0.0f; if (v > 1.0f) v = 1.0f;
                unsigned char b = (unsigned char)std::lround(v * 255.0f);
                out.push_back((char)b);
            }
            off += sizes[idx];
        }
        return py::bytes(out);
#else
        (void)psi_time; (void)dim; (void)seeds; (void)sizes; throw std::runtime_error("GPU platform not supported");
#endif
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
        // Holographic superposition encode/decode
        .def("encode_superpose_bytes", &HolographicGPUWrapper::encode_superpose_bytes,
             py::arg("data"), py::arg("chunk_size") = (std::uint32_t)4096, py::arg("seed_base") = (std::uint32_t)0)
        .def("decode_superposed_bytes", &HolographicGPUWrapper::decode_superposed_bytes,
             py::arg("psi_time"), py::arg("dim"), py::arg("seeds"), py::arg("sizes"))
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

    // ----------------- ECC (RS(255,223)) host-side encode/decode -----------------
    // RS interface removed from Python module (Phase 3.5). Keep implementation
    // under compile-off block for archival purposes only.
#if 0
    // Minimal, deterministic Reed–Solomon implementation over GF(2^8)
    m.def("gpu_rs_encode", [](py::bytes payload, std::uint32_t k, std::uint32_t r) {
        std::string s = payload;
        if (k == 0 || r == 0) throw std::runtime_error("invalid RS parameters");
        const std::uint32_t n = k + r;
        auto build_tables = [](){
            std::array<std::uint8_t,256> logt{}; logt.fill(0);
            std::array<std::uint8_t,256> alogt{}; alogt.fill(0);
            std::uint16_t x = 1;
            for (int i=0;i<255;++i){ alogt[i]=(std::uint8_t)x; logt[(std::uint8_t)x]=(std::uint8_t)i; x<<=1; if (x & 0x100) x^=0x11d; }
            alogt[255]=alogt[0];
            return std::make_pair(logt, alogt);
        };
        auto build_generator_fn = [&](std::uint32_t r_, const auto& logt_, const auto& alogt_){
            std::vector<std::uint8_t> g; g.push_back(1); std::uint8_t alpha=2, powa=1;
            for (std::uint32_t i=0;i<r_;++i){
                std::vector<std::uint8_t> ng(g.size()+1,0);
                for (std::size_t j=0;j<g.size();++j){
                    std::uint8_t mult = 0;
                    if (g[j] && powa){ int s=(int)logt_[g[j]] + (int)logt_[powa]; s%=255; if (s<0) s+=255; mult = alogt_[(std::size_t)s]; }
                    ng[j] = (std::uint8_t)(ng[j] ^ mult);
                    ng[j+1] = (std::uint8_t)(ng[j+1] ^ g[j]);
                }
                // powa *= alpha
                if (powa) { int s=(int)logt_[powa]+(int)logt_[alpha]; s%=255; if(s<0) s+=255; powa = alogt_[(std::size_t)s]; } else { powa = 0; }
                g.swap(ng);
            }
            if (g.size()>r_) g.erase(g.begin()); if (g.size()<r_) g.resize(r_,0);
            return g;
        };
        auto gf_mul = [](std::uint8_t a, std::uint8_t b, const auto& logt, const auto& alogt){ if(!a||!b) return (std::uint8_t)0; int s=(int)logt[a]+(int)logt[b]; s%=255; if(s<0) s+=255; return alogt[(std::size_t)s]; };
        auto build_generator = [&](std::uint32_t r, const auto& logt, const auto& alogt){
            std::vector<std::uint8_t> g; g.push_back(1); std::uint8_t alpha=2, powa=1;
            for (std::uint32_t i=0;i<r;++i){
                std::vector<std::uint8_t> ng(g.size()+1,0);
                for (std::size_t j=0;j<g.size();++j){ ng[j] ^= gf_mul(g[j], powa, logt, alogt); ng[j+1] ^= g[j]; }
                g.swap(ng); powa = gf_mul(powa, alpha, logt, alogt);
            }
            if (g.size()>r) g.erase(g.begin()); if (g.size()<r) g.resize(r,0);
            return g;
        };
        auto [logt, alogt] = build_tables();
        std::vector<std::uint8_t> gen = build_generator_fn(r, logt, alogt);
        std::uint32_t blocks = (std::uint32_t)((s.size() + k - 1) / k);
        std::vector<std::uint8_t> parity((std::size_t)blocks * r, 0);
        for (std::uint32_t b=0;b<blocks;++b){
            const std::size_t off = (std::size_t)b * k;
            std::vector<std::uint8_t> parr(r,0);
            for (std::uint32_t i=0;i<k;++i){ std::uint8_t di = (off+i < s.size()) ? (std::uint8_t)s[off+i] : 0; std::uint8_t fb = (std::uint8_t)(di ^ parr[0]);
                for (std::uint32_t j=0;j<r-1;++j){ std::uint8_t mult = gf_mul(fb, gen[j], logt, alogt); parr[j] = (std::uint8_t)(parr[j+1] ^ mult); }
                parr[r-1] = gf_mul(fb, gen[r-1], logt, alogt);
            }
            std::memcpy(parity.data() + (std::size_t)b * r, parr.data(), r);
        }
        return py::bytes(reinterpret_cast<const char*>(parity.data()), (py::ssize_t)parity.size());
    }, py::arg("payload"), py::arg("k")=(std::uint32_t)223, py::arg("r")=(std::uint32_t)32,
       "GPU RS(255,223) ECC encode parity for a byte stream (host-side)");

    m.def("gpu_rs_decode", [](py::bytes payload, py::bytes parity, std::uint32_t k, std::uint32_t r){
        std::string s = payload; std::string p = parity;
        if (k == 0 || r == 0) throw std::runtime_error("invalid RS parameters");
        const std::uint32_t n = k + r;
        auto build_tables = [](){
            std::array<std::uint8_t,256> logt{}; logt.fill(0);
            std::array<std::uint8_t,256> alogt{}; alogt.fill(0);
            std::uint16_t x = 1;
            for (int i=0;i<255;++i){ alogt[i]=(std::uint8_t)x; logt[(std::uint8_t)x]=(std::uint8_t)i; x<<=1; if (x & 0x100) x^=0x11d; }
            alogt[255]=alogt[0];
            return std::make_pair(logt, alogt);
        };
        auto gf_add = [](std::uint8_t a, std::uint8_t b){ return (std::uint8_t)(a ^ b); };
        auto gf_mul = [](std::uint8_t a, std::uint8_t b, const auto& logt, const auto& alogt){ if(!a||!b) return (std::uint8_t)0; int s=(int)logt[a]+(int)logt[b]; s%=255; if(s<0) s+=255; return alogt[(std::size_t)s]; };
        auto gf_div = [](std::uint8_t a, std::uint8_t b, const auto& logt, const auto& alogt){ if(!a) return (std::uint8_t)0; if(!b) return (std::uint8_t)0; int s=(int)logt[a] - (int)logt[b]; s%=255; if(s<0) s+=255; return alogt[(std::size_t)s]; };
        auto poly_eval = [&](const std::vector<std::uint8_t>& poly, std::uint8_t x, const auto& logt, const auto& alogt){ std::uint8_t y = poly.empty()?0:poly[0]; for (std::size_t i=1;i<poly.size();++i){ y = gf_add(gf_mul(y, x, logt, alogt), poly[i]); } return y; };
        auto build_generator_fn = [&](std::uint32_t r_, const auto& logt_, const auto& alogt_){
            std::vector<std::uint8_t> g; g.push_back(1); std::uint8_t alpha=2, powa=1;
            for (std::uint32_t i=0;i<r_;++i){
                std::vector<std::uint8_t> ng(g.size()+1,0);
                for (std::size_t j=0;j<g.size();++j){
                    std::uint8_t mult = 0;
                    if (g[j] && powa){ int s=(int)logt_[g[j]] + (int)logt_[powa]; s%=255; if (s<0) s+=255; mult = alogt_[(std::size_t)s]; }
                    ng[j] = (std::uint8_t)(ng[j] ^ mult);
                    ng[j+1] = (std::uint8_t)(ng[j+1] ^ g[j]);
                }
                if (powa) { int s=(int)logt_[powa]+(int)logt_[alpha]; s%=255; if(s<0) s+=255; powa = alogt_[(std::size_t)s]; } else { powa = 0; }
                g.swap(ng);
            }
            if (g.size()>r_) g.erase(g.begin()); if (g.size()<r_) g.resize(r_,0);
            return g;
        };
        auto [logt, alogt] = build_tables();
        std::uint32_t blocks = (std::uint32_t)((s.size() + k - 1) / k);
        if (p.size() < (std::size_t)blocks * r) throw std::runtime_error("parity size too small");
        std::vector<std::uint8_t> out((std::size_t)blocks * k, 0);
        std::vector<int> corrected(blocks, 0);
        // Build generator once for parity validation
        std::vector<std::uint8_t> gen = build_generator_fn(r, logt, alogt);
        // Deterministic decode path only. No variant enumeration.
        auto compute_parity = [&](const std::uint8_t* data, std::size_t len){
            std::vector<std::uint8_t> parr(r,0);
            for (std::uint32_t i=0;i<k;++i){
                std::uint8_t di = (i < len) ? data[i] : 0;
                std::uint8_t fb = (std::uint8_t)(di ^ parr[0]);
                for (std::uint32_t j=0;j<r-1;++j){
                    // parr[j] = parr[j+1] ^ fb*gen[j]
                    std::uint8_t mult = (fb && gen[j]) ? alogt[(std::size_t)((logt[fb]+logt[gen[j]])%255)] : 0;
                    parr[j] = (std::uint8_t)(parr[j+1] ^ mult);
                }
                std::uint8_t multl = (fb && gen[r-1]) ? alogt[(std::size_t)((logt[fb]+logt[gen[r-1]])%255)] : 0;
                parr[r-1] = multl;
            }
            return parr;
        };
        for (std::uint32_t b=0;b<blocks;++b){
            const std::size_t off = (std::size_t)b * k;
            const std::size_t chunk_len = (off < s.size()) ? std::min<std::size_t>(k, s.size() - off) : 0;
            const std::uint8_t* d = (const std::uint8_t*)s.data() + off;
            const std::uint8_t* pr = (const std::uint8_t*)p.data() + (std::size_t)b * r;
            std::vector<std::uint8_t> recv; recv.reserve(n);
            // Copy available data then zero-pad to k
            if (chunk_len > 0) recv.insert(recv.end(), d, d + chunk_len);
            if (chunk_len < k) recv.insert(recv.end(), (std::size_t)(k - chunk_len), (std::uint8_t)0);
            // Append parity for this block
            recv.insert(recv.end(), pr, pr + r);
            auto recv_out_ptr = out.data() + (std::size_t)b * k;
            // Tail-block deterministic shortened RS decode: s = k - chunk_len
            if (chunk_len < k){
                const std::uint32_t sshort = (std::uint32_t)(k - chunk_len);
                // Compute syndromes for shortened RS code
                // The recv buffer is [data_L | zeros_(k-L) | parity_r]
                // Standard syndrome uses full block convention
                std::vector<std::uint8_t> S(r, 0); bool all_zero = true;
                for (std::uint32_t j=0;j<r;++j){
                    std::uint8_t Sj=0;
                    // Process the full codeword as-is (zeros are already in place)
                    for (std::uint32_t i=0;i<n;++i){
                        std::uint8_t c = recv[i];
                        if (c){
                            // Standard RS syndrome: S_j = C(α^(j+1)) where C(x) = Σ c[i]*x^(n-1-i)
                            int ex = ((n - 1 - i) * (j + 1)) % 255;
                            if (ex<0) ex+=255;
                            Sj = gf_add(Sj, alogt[(logt[c] + ex) % 255]);
                        }
                    }
                    S[j]=Sj; all_zero = all_zero && (Sj==0);
                }
                if (all_zero){ std::memcpy(recv_out_ptr, recv.data(), k); continue; }
                // BM
                std::vector<std::uint8_t> Lambda(1,1), Bm(1,1); std::uint8_t bcoef=1; int L=0; int m=1;
                for (std::uint32_t nsy=0;nsy<r;++nsy){
                    std::uint8_t dsc=S[nsy];
                    for (int i=1;i<=L;++i){ if (i < (int)Lambda.size() && (int)(nsy - i) >= 0) dsc = gf_add(dsc, gf_mul(Lambda[(std::size_t)i], S[(std::size_t)(nsy - i)], logt, alogt)); }
                    if (dsc == 0){ m++; continue; }
                    std::vector<std::uint8_t> T = Lambda;
                    auto poly_scale = [&](const std::vector<std::uint8_t>& p, std::uint8_t x){ std::vector<std::uint8_t> outp(p.size()); for (std::size_t ii=0;ii<p.size();++ii) outp[ii]=gf_mul(p[ii], x, logt, alogt); return outp; };
                    std::vector<std::uint8_t> xmb(Bm.size()+ (std::size_t)m, 0); for (std::size_t ii=0;ii<Bm.size();++ii) xmb[ii]=Bm[ii]; for (int sh=0; sh<m; ++sh) xmb.insert(xmb.begin(), 0);
                    std::uint8_t scale = gf_div(dsc, bcoef, logt, alogt);
                    auto scaled = poly_scale(xmb, scale);
                    std::size_t nmax = std::max(Lambda.size(), scaled.size()); std::vector<std::uint8_t> newL(nmax,0);
                    for (std::size_t ii=0;ii<nmax;++ii){ std::uint8_t av = (ii<Lambda.size())?Lambda[Lambda.size()-1-ii]:0; std::uint8_t bv=(ii<scaled.size())?scaled[scaled.size()-1-ii]:0; newL[nmax-1-ii] = gf_add(av, bv); }
                    Lambda.swap(newL);
                    if (2*L <= (int)nsy){ L = (int)nsy + 1 - L; Bm = T; bcoef = dsc; m = 1; } else { m++; }
                }
                // Chien: Xi = alpha^{-i}
                std::vector<int> err_pos; err_pos.reserve(r);
                for (std::uint32_t i=0;i<n;++i){ std::uint8_t x = alogt[(std::size_t)((255 - (int)(i % 255)) % 255)]; if (poly_eval(Lambda, x, logt, alogt)==0){ err_pos.push_back((int)i); } }
                if (err_pos.empty() || (int)err_pos.size() > L){ std::memcpy(recv_out_ptr, recv.data(), k); continue; }
                // Omega and Lambda'
                std::vector<std::uint8_t> Sx = S; std::vector<std::uint8_t> prod(Sx.size()+Lambda.size()-1, 0);
                for (std::size_t ii=0;ii<Sx.size();++ii) for (std::size_t jj=0;jj<Lambda.size();++jj) prod[ii+jj] = gf_add(prod[ii+jj], gf_mul(Sx[ii], Lambda[jj], logt, alogt));
                std::vector<std::uint8_t> Omega(prod.begin(), prod.begin()+ std::min<std::size_t>(prod.size(), r));
                std::vector<std::uint8_t> Lp; for (std::size_t ii=1;ii<Lambda.size();++ii) if (ii%2==1) Lp.push_back(Lambda[ii]);
                // Apply corrections mapped to data indices: idx_full = n-1-pos; data_idx = idx_full - s, if in [0,L)
                auto trial = recv; auto base = recv;
                const char* dbg_tail = std::getenv("HG_ECC_DEBUG");
                if (dbg_tail && *dbg_tail){
                    std::fprintf(stderr, "[ECC] Tail block b=%u L=%d roots=%zu chunk_len=%zu\n", b, L, err_pos.size(), chunk_len);
                }
                for (int pos_i : err_pos){
                    // For Xi = α^{−pos_i}, Xi^{-1} = α^{+pos_i}
                    std::uint8_t Xi_inv = alogt[(std::size_t)(pos_i % 255)];
                    std::uint8_t num = poly_eval(Omega, Xi_inv, logt, alogt);
                    std::uint8_t den = poly_eval(Lp, Xi_inv, logt, alogt);
                    if (!den) continue;
                    std::uint8_t mag = gf_div(num, den, logt, alogt);
                    // Map error position to data index
                    // pos_i is the position in the full codeword where Λ(α^{-pos_i}) = 0
                    // For data bytes, this maps directly to index in recv buffer
                    std::size_t idx = (std::size_t)(n - 1 - pos_i);
                    if (dbg_tail && *dbg_tail){
                        std::fprintf(stderr, "[ECC] Tail root pos=%d idx=%zu chunk_len=%zu\n", pos_i, idx, chunk_len);
                    }
                    // Only correct data bytes, not parity
                    if (idx < chunk_len) {
                        trial[idx] = gf_add(trial[idx], mag);
                        if (dbg_tail && *dbg_tail){
                            std::fprintf(stderr, "[ECC]   Correcting byte at idx=%zu: %02x -> %02x\n", idx, (unsigned)recv[idx], (unsigned)trial[idx]);
                        }
                    }
                }
                auto parr_try = compute_parity(trial.data(), chunk_len);
                if (std::memcmp(parr_try.data(), pr, r) == 0){
                    int changes = 0; for (std::size_t ii=0; ii<chunk_len; ++ii){ if (trial[ii] != base[ii]) ++changes; }
                    corrected[(std::size_t)b] = changes;
                    std::memcpy(recv_out_ptr, trial.data(), k);
                    continue;
                }
                // If not accepted, leave unchanged for this block
                std::memcpy(recv_out_ptr, recv.data(), k);
                continue;
            }
            // Syndromes: S_j = C(α^(j+1)) for generator with roots α^1..α^32
            std::vector<std::uint8_t> S(r, 0); bool all_zero = true;
            for (std::uint32_t j=0;j<r;++j){
                std::uint8_t Sj=0;
                for (std::uint32_t i=0;i<n;++i){
                    std::uint8_t c = recv[i];
                    if (c){
                        // S_j = Σ c[i] * α^((n-1-i)*(j+1))
                        int pow = ((n-1 - i) * (j + 1)) % 255;
                        if (pow<0) pow+=255;
                        Sj = gf_add(Sj, gf_mul(c, alogt[(std::size_t)pow], logt, alogt));
                    }
                }
                S[j]=Sj; all_zero = all_zero && (Sj==0);
            }
            if (all_zero){ std::memcpy(out.data() + (std::size_t)b * k, recv.data(), k); continue; }
            // Berlekamp–Massey
            std::vector<std::uint8_t> Lambda(1,1), B(1,1); std::uint8_t bcoef=1; int L=0; int m=1;
            for (std::uint32_t nsy=0;nsy<r;++nsy){ std::uint8_t dsc=S[nsy]; for (int i=1;i<=L;++i){ if (i < (int)Lambda.size() && (int)(nsy - i) >= 0) dsc = gf_add(dsc, gf_mul(Lambda[(std::size_t)i], S[(std::size_t)(nsy - i)], logt, alogt)); }
                if (dsc == 0){ m++; continue; }
                std::vector<std::uint8_t> T = Lambda;
                // scale = d/b; x^m * B(x)
                auto poly_scale = [&](const std::vector<std::uint8_t>& p, std::uint8_t x){ std::vector<std::uint8_t> out(p.size()); for (std::size_t i=0;i<p.size();++i) out[i]=gf_mul(p[i], x, logt, alogt); return out; };
                std::vector<std::uint8_t> xmb(B.size()+ (std::size_t)m, 0); for (std::size_t i=0;i<B.size();++i) xmb[i]=B[i]; for (int sh=0; sh<m; ++sh) xmb.insert(xmb.begin(), 0);
                std::uint8_t scale = gf_div(dsc, bcoef, logt, alogt);
                auto scaled = poly_scale(xmb, scale);
                // Lambda = Lambda - scaled
                std::size_t nmax = std::max(Lambda.size(), scaled.size()); std::vector<std::uint8_t> newL(nmax,0);
                for (std::size_t i=0;i<nmax;++i){ std::uint8_t av = (i<Lambda.size())?Lambda[Lambda.size()-1-i]:0; std::uint8_t bv=(i<scaled.size())?scaled[scaled.size()-1-i]:0; newL[nmax-1-i] = gf_add(av, bv); }
                Lambda.swap(newL);
                if (2*L <= (int)nsy){ L = (int)nsy + 1 - L; B = T; bcoef = dsc; m = 1; } else { m++; }
            }
            // Chien search
            std::vector<int> err_pos; for (std::uint32_t i=0;i<n;++i){ std::uint8_t x = alogt[(std::size_t)((255 - i) % 255)]; if (poly_eval(Lambda, x, logt, alogt)==0){ err_pos.push_back((int)i); } }
            if (err_pos.empty() || (int)err_pos.size() > L){
                std::memcpy(out.data() + (std::size_t)b * k, recv.data(), k);
                continue;
            }
            // Omega(x) = [S(x)*Lambda(x)] mod x^r
            std::vector<std::uint8_t> Sx = S; std::vector<std::uint8_t> prod(Sx.size()+Lambda.size()-1, 0);
            for (std::size_t i=0;i<Sx.size();++i) for (std::size_t j=0;j<Lambda.size();++j) prod[i+j] = gf_add(prod[i+j], gf_mul(Sx[i], Lambda[j], logt, alogt));
            std::vector<std::uint8_t> Omega(prod.begin(), prod.begin()+ std::min<std::size_t>(prod.size(), r));
            // Lambda'(x)
            std::vector<std::uint8_t> Lp; for (std::size_t i=1;i<Lambda.size();++i) if (i%2==1) Lp.push_back(Lambda[i]);
            // Forney corrections
            // Try primary correction mapping first
            auto apply_corrections = [&](std::vector<std::uint8_t>& vec, bool alt_xi, int idx_mode){
                for (int pos : err_pos){
                    std::uint32_t loc_shift = (idx_mode >= 2) ? (std::uint32_t)(k - chunk_len) : 0u;
                    std::uint32_t pos_adj = (std::uint32_t)((pos + (int)loc_shift) % 255);
                    std::uint32_t exp = alt_xi ? pos_adj : (255u - pos_adj) % 255u;
                    std::uint8_t Xi_inv = alogt[(std::size_t)exp];
                    std::uint8_t num = poly_eval(Omega, Xi_inv, logt, alogt);
                    std::uint8_t den = poly_eval(Lp, Xi_inv, logt, alogt);
                    if (!den) continue;
                    std::uint8_t mag = gf_div(num, den, logt, alogt);
                    std::size_t idx;
                    if (idx_mode == 0) {
                        // Primary mapping: reverse over full codeword
                        idx = (std::size_t)(n - 1 - pos);
                    } else if (idx_mode == 1) {
                        // Alternate mapping: direct index over full codeword
                        idx = (std::size_t)pos;
                    } else if (idx_mode == 2) {
                        // Tail-aligned reverse mapping within data region only (relative to chunk_len)
                        // Map Chien position to data index counted from end of present data.
                        std::size_t ridx = (std::size_t)(n - 1 - pos);
                        // Shift by zero-padding so that errors align to [0, chunk_len)
                        if (ridx >= (k - chunk_len)) idx = ridx - (k - chunk_len); else idx = (std::size_t)k+1;
                    } else if (idx_mode == 3) {
                        // Tail-aligned direct mapping with zero-pad offset
                        std::size_t didx = (std::size_t)pos;
                        if (didx >= (k - chunk_len)) idx = didx - (k - chunk_len); else idx = (std::size_t)k+1;
                    } else {
                        idx = (std::size_t)k+1;
                    }
                    // Apply only within actual data region (avoid modifying padded zeros)
                    if (idx < chunk_len) vec[idx] = gf_add(vec[idx], mag);
                }
            };
            auto recv_base = recv; // backup
            // Try primary mapping
            apply_corrections(recv, /*alt_xi=*/false, /*idx_mode=*/0);
            auto parr = compute_parity(recv.data(), chunk_len);
            bool ok = (std::memcmp(parr.data(), pr, r) == 0);
            if (!ok){
                // Try alternative Xi mapping
                recv = recv_base;
                apply_corrections(recv, /*alt_xi=*/true, /*idx_mode=*/0);
                parr = compute_parity(recv.data(), chunk_len);
                ok = (std::memcmp(parr.data(), pr, r) == 0);
            }
            if (!ok){
                // Try alternative index mapping
                recv = recv_base;
                apply_corrections(recv, /*alt_xi=*/false, /*idx_mode=*/1);
                parr = compute_parity(recv.data(), chunk_len);
                ok = (std::memcmp(parr.data(), pr, r) == 0);
            }
            if (!ok){
                // Try both alternative Xi and index mapping
                recv = recv_base;
                apply_corrections(recv, /*alt_xi=*/true, /*idx_mode=*/1);
                parr = compute_parity(recv.data(), chunk_len);
                ok = (std::memcmp(parr.data(), pr, r) == 0);
            }
            if (!ok){
                // Tail-aligned mapping (for partial blocks)
                recv = recv_base;
                apply_corrections(recv, /*alt_xi=*/false, /*idx_mode=*/2);
                parr = compute_parity(recv.data(), chunk_len);
                ok = (std::memcmp(parr.data(), pr, r) == 0);
            }
            if (!ok){
                recv = recv_base;
                apply_corrections(recv, /*alt_xi=*/true, /*idx_mode=*/2);
                parr = compute_parity(recv.data(), chunk_len);
                ok = (std::memcmp(parr.data(), pr, r) == 0);
            }
            if (!ok){
                // Additional tail-aligned mapping using direct index with zero-pad offset
                recv = recv_base;
                apply_corrections(recv, /*alt_xi=*/false, /*idx_mode=*/3);
                parr = compute_parity(recv.data(), chunk_len);
                ok = (std::memcmp(parr.data(), pr, r) == 0);
            }
            if (!ok){
                recv = recv_base;
                apply_corrections(recv, /*alt_xi=*/true, /*idx_mode=*/3);
                parr = compute_parity(recv.data(), chunk_len);
                ok = (std::memcmp(parr.data(), pr, r) == 0);
            }
            if (ok){
                // Count actual byte changes within the data portion
                int changes = 0;
                for (std::size_t i=0; i<chunk_len; ++i){ if (recv[i] != recv_base[i]) ++changes; }
                corrected[(std::size_t)b] = changes;
            } else {
                // Leave unchanged if validation fails
                recv = recv_base;
                const char* dbg = std::getenv("HG_ECC_DEBUG");
                if (dbg && *dbg){
                    std::fprintf(stderr, "[ECC] Block %u failed parity validation (chunk_len=%zu, L=%d, roots=%zu)\n", b, chunk_len, L, err_pos.size());
                }
                // Tail-block fallback: re-compute syndromes with zero-pad exponent shift
                if (chunk_len < k){
                    std::size_t pad = (std::size_t)(k - chunk_len);
                    // Recompute syndromes with shifted exponent to mirror zero-padding during encode
                    std::vector<std::uint8_t> S2(r, 0);
                    for (std::uint32_t j=0;j<r;++j){
                        std::uint8_t Sj=0;
                        for (std::uint32_t i=0;i<n;++i){
                            std::uint8_t c = recv[i];
                            if (c){
                                int pow = (int)((n - 1 - i) + (int)pad + j) % 255;
                                if (pow<0) pow+=255;
                                Sj = gf_add(Sj, gf_mul(c, alogt[(std::size_t)pow], logt, alogt));
                            }
                        }
                        S2[j]=Sj;
                    }
                    bool all_zero2 = true; for (auto v: S2) all_zero2 = all_zero2 && (v==0);
                    if (!all_zero2){
                        // BM on shifted syndromes
                        std::vector<std::uint8_t> Lambda2(1,1), B2(1,1); std::uint8_t bcoef2=1; int L2=0; int m2=1;
                        for (std::uint32_t nsy=0;nsy<r;++nsy){ std::uint8_t dsc=S2[nsy]; for (int i=1;i<=L2;++i){ if (i < (int)Lambda2.size() && (int)(nsy - i) >= 0) dsc = gf_add(dsc, gf_mul(Lambda2[(std::size_t)i], S2[(std::size_t)(nsy - i)], logt, alogt)); }
                            if (dsc == 0){ m2++; continue; }
                            std::vector<std::uint8_t> T2 = Lambda2;
                            auto poly_scale2 = [&](const std::vector<std::uint8_t>& p, std::uint8_t x){ std::vector<std::uint8_t> outp(p.size()); for (std::size_t ii=0;ii<p.size();++ii) outp[ii]=gf_mul(p[ii], x, logt, alogt); return outp; };
                            std::vector<std::uint8_t> xmb2(B2.size()+ (std::size_t)m2, 0); for (std::size_t ii=0;ii<B2.size();++ii) xmb2[ii]=B2[ii]; for (int sh=0; sh<m2; ++sh) xmb2.insert(xmb2.begin(), 0);
                            std::uint8_t scale2 = gf_div(dsc, bcoef2, logt, alogt);
                            auto scaled2 = poly_scale2(xmb2, scale2);
                            std::size_t nmax2 = std::max(Lambda2.size(), scaled2.size()); std::vector<std::uint8_t> newL2(nmax2,0);
                            for (std::size_t ii=0;ii<nmax2;++ii){ std::uint8_t av = (ii<Lambda2.size())?Lambda2[Lambda2.size()-1-ii]:0; std::uint8_t bv=(ii<scaled2.size())?scaled2[scaled2.size()-1-ii]:0; newL2[nmax2-1-ii] = gf_add(av, bv); }
                            Lambda2.swap(newL2);
                            if (2*L2 <= (int)nsy){ L2 = (int)nsy + 1 - L2; B2 = T2; bcoef2 = dsc; m2 = 1; } else { m2++; }
                        }
                        // Chien search (same mapping as primary)
                        std::vector<int> err_pos2; for (std::uint32_t ii=0;ii<n;++ii){ std::uint8_t x = alogt[(std::size_t)((255 - ii) % 255)]; if (poly_eval(Lambda2, x, logt, alogt)==0){ err_pos2.push_back((int)ii); } }
                        if (!err_pos2.empty() && (int)err_pos2.size() <= L2){
                            // Omega2 and Lp2
                            std::vector<std::uint8_t> Sx2 = S2; std::vector<std::uint8_t> prod2(Sx2.size()+Lambda2.size()-1, 0);
                            for (std::size_t ii=0;ii<Sx2.size();++ii) for (std::size_t jj=0;jj<Lambda2.size();++jj) prod2[ii+jj] = gf_add(prod2[ii+jj], gf_mul(Sx2[ii], Lambda2[jj], logt, alogt));
                            std::vector<std::uint8_t> Omega2(prod2.begin(), prod2.begin()+ std::min<std::size_t>(prod2.size(), r));
                            std::vector<std::uint8_t> Lp2; for (std::size_t ii=1;ii<Lambda2.size();++ii) if (ii%2==1) Lp2.push_back(Lambda2[ii]);
                            auto apply_corr2 = [&](std::vector<std::uint8_t>& vec, bool alt_xi, int idx_mode){
                                for (int pos2 : err_pos2){
                                    std::uint32_t loc_shift2 = (idx_mode >= 2) ? (std::uint32_t)(k - chunk_len) : 0u;
                                    std::uint32_t pos_adj2 = (std::uint32_t)((pos2 + (int)loc_shift2) % 255);
                                    std::uint32_t exp2 = alt_xi ? pos_adj2 : (255u - pos_adj2) % 255u;
                                    std::uint8_t Xi_inv2 = alogt[(std::size_t)exp2];
                                    std::uint8_t num2 = poly_eval(Omega2, Xi_inv2, logt, alogt);
                                    std::uint8_t den2 = poly_eval(Lp2, Xi_inv2, logt, alogt);
                                    if (!den2) continue;
                                    std::uint8_t mag2 = gf_div(num2, den2, logt, alogt);
                                    std::size_t idx2;
                                    if (idx_mode == 0) { idx2 = (std::size_t)(n - 1 - pos2); }
                                    else if (idx_mode == 1) { idx2 = (std::size_t)pos2; }
                                    else if (idx_mode == 2) {
                                        std::size_t ridx2 = (std::size_t)(n - 1 - pos2);
                                        if (ridx2 >= (k - chunk_len)) idx2 = ridx2 - (k - chunk_len); else idx2 = (std::size_t)k+1;
                                    } else if (idx_mode == 3) {
                                        std::size_t didx2 = (std::size_t)pos2;
                                        if (didx2 >= (k - chunk_len)) idx2 = didx2 - (k - chunk_len); else idx2 = (std::size_t)k+1;
                                    } else { idx2 = (std::size_t)k+1; }
                                    if (idx2 < chunk_len) vec[idx2] = gf_add(vec[idx2], mag2);
                                }
                            };
                            // Try same mapping combos on the shifted-syndrome solution
                            recv = recv_base;
                            apply_corr2(recv, /*alt_xi=*/false, /*idx_mode=*/0);
                            auto parr2 = compute_parity(recv.data(), chunk_len);
                            bool ok2 = (std::memcmp(parr2.data(), pr, r) == 0);
                            if (!ok2){ recv = recv_base; apply_corr2(recv, true, 0); parr2 = compute_parity(recv.data(), chunk_len); ok2 = (std::memcmp(parr2.data(), pr, r) == 0); }
                            if (!ok2){ recv = recv_base; apply_corr2(recv, false, 1); parr2 = compute_parity(recv.data(), chunk_len); ok2 = (std::memcmp(parr2.data(), pr, r) == 0); }
                            if (!ok2){ recv = recv_base; apply_corr2(recv, true, 1); parr2 = compute_parity(recv.data(), chunk_len); ok2 = (std::memcmp(parr2.data(), pr, r) == 0); }
                            if (!ok2){ recv = recv_base; apply_corr2(recv, false, 2); parr2 = compute_parity(recv.data(), chunk_len); ok2 = (std::memcmp(parr2.data(), pr, r) == 0); }
                            if (!ok2){ recv = recv_base; apply_corr2(recv, true, 2); parr2 = compute_parity(recv.data(), chunk_len); ok2 = (std::memcmp(parr2.data(), pr, r) == 0); }
                            if (!ok2){ recv = recv_base; apply_corr2(recv, false, 3); parr2 = compute_parity(recv.data(), chunk_len); ok2 = (std::memcmp(parr2.data(), pr, r) == 0); }
                            if (!ok2){ recv = recv_base; apply_corr2(recv, true, 3); parr2 = compute_parity(recv.data(), chunk_len); ok2 = (std::memcmp(parr2.data(), pr, r) == 0); }
                            if (ok2){
                                int changes2 = 0; for (std::size_t ii=0; ii<chunk_len; ++ii){ if (recv[ii] != recv_base[ii]) ++changes2; }
                                corrected[(std::size_t)b] = changes2;
                            } else {
                                recv = recv_base;
                                if (dbg && *dbg){
                                    std::fprintf(stderr, "[ECC] Block %u tail-fallback failed (pad=%zu, L2=%d, roots2=%zu)\n", b, pad, L2, err_pos2.size());
                                }
                            }
                        }
                    }
                }
                // Standard RS fallback (systematic, parity at end): robust mapping
                if (true){
                    // Compute standard syndromes S_j = sum_{i=0..n-1} c[i] * (alpha^{i})^{j+1}
                    std::vector<std::uint8_t> S3(r, 0);
                    for (std::uint32_t jj=0; jj<r; ++jj){
                        std::uint8_t Sj=0;
                        std::uint32_t j1 = jj + 1; // j in [1..r]
                        for (std::uint32_t ii=0; ii<n; ++ii){
                            std::uint8_t c = recv[ii];
                            if (!c) continue;
                            int ex = (int)((ii * j1) % 255);
                            // c * alpha^{ex}
                            std::uint8_t term = alogt[(std::size_t)((logt[c] + ex) % 255)];
                            Sj = gf_add(Sj, term);
                        }
                        S3[jj] = Sj;
                    }
                    bool all_zero3 = true; for (auto v: S3) all_zero3 = all_zero3 && (v==0);
                    if (!all_zero3){
                        // BM on S3
                        std::vector<std::uint8_t> Lambda3(1,1), B3(1,1); std::uint8_t bcoef3=1; int L3=0; int m3=1;
                        for (std::uint32_t nsy=0;nsy<r;++nsy){ std::uint8_t dsc=S3[nsy]; for (int i=1;i<=L3;++i){ if (i < (int)Lambda3.size() && (int)(nsy - i) >= 0) dsc = gf_add(dsc, gf_mul(Lambda3[(std::size_t)i], S3[(std::size_t)(nsy - i)], logt, alogt)); }
                            if (dsc == 0){ m3++; continue; }
                            std::vector<std::uint8_t> T3 = Lambda3;
                            auto poly_scale3 = [&](const std::vector<std::uint8_t>& p, std::uint8_t x){ std::vector<std::uint8_t> outp(p.size()); for (std::size_t ii=0;ii<p.size();++ii) outp[ii]=gf_mul(p[ii], x, logt, alogt); return outp; };
                            std::vector<std::uint8_t> xmb3(B3.size()+ (std::size_t)m3, 0); for (std::size_t ii=0;ii<B3.size();++ii) xmb3[ii]=B3[ii]; for (int sh=0; sh<m3; ++sh) xmb3.insert(xmb3.begin(), 0);
                            std::uint8_t scale3 = gf_div(dsc, bcoef3, logt, alogt);
                            auto scaled3 = poly_scale3(xmb3, scale3);
                            std::size_t nmax3 = std::max(Lambda3.size(), scaled3.size()); std::vector<std::uint8_t> newL3(nmax3,0);
                            for (std::size_t ii=0;ii<nmax3;++ii){ std::uint8_t av = (ii<Lambda3.size())?Lambda3[Lambda3.size()-1-ii]:0; std::uint8_t bv=(ii<scaled3.size())?scaled3[scaled3.size()-1-ii]:0; newL3[nmax3-1-ii] = gf_add(av, bv); }
                            Lambda3.swap(newL3);
                            if (2*L3 <= (int)nsy){ L3 = (int)nsy + 1 - L3; B3 = T3; bcoef3 = dsc; m3 = 1; } else { m3++; }
                        }
                        // Chien search per standard mapping: try both Xi = alpha^{-i} and Xi = alpha^{i}
                        std::vector<int> err_pos3a; err_pos3a.reserve(r);
                        for (std::uint32_t ii=0; ii<n; ++ii){
                            std::uint8_t Xi = alogt[(std::size_t)((255 - (int)(ii % 255)) % 255)];
                            if (poly_eval(Lambda3, Xi, logt, alogt) == 0){ err_pos3a.push_back((int)ii); }
                        }
                        std::vector<int> err_pos3b; err_pos3b.reserve(r);
                        for (std::uint32_t ii=0; ii<n; ++ii){
                            std::uint8_t Xi = alogt[(std::size_t)(ii % 255)];
                            if (poly_eval(Lambda3, Xi, logt, alogt) == 0){ err_pos3b.push_back((int)ii); }
                        }
                        bool used_b = false;
                        std::vector<int>* errp = nullptr;
                        if (!err_pos3a.empty() && (int)err_pos3a.size() <= L3){ errp = &err_pos3a; used_b = false; }
                        else if (!err_pos3b.empty() && (int)err_pos3b.size() <= L3){ errp = &err_pos3b; used_b = true; }
                        if (errp){
                            // Omega3 and Lp3
                            std::vector<std::uint8_t> Sx3 = S3; std::vector<std::uint8_t> prod3(Sx3.size()+Lambda3.size()-1, 0);
                            for (std::size_t ii=0;ii<Sx3.size();++ii) for (std::size_t jj=0;jj<Lambda3.size();++jj) prod3[ii+jj] = gf_add(prod3[ii+jj], gf_mul(Sx3[ii], Lambda3[jj], logt, alogt));
                            std::vector<std::uint8_t> Omega3(prod3.begin(), prod3.begin()+ std::min<std::size_t>(prod3.size(), r));
                            std::vector<std::uint8_t> Lp3; for (std::size_t ii=1;ii<Lambda3.size();++ii) if (ii%2==1) Lp3.push_back(Lambda3[ii]);
                            // Apply corrections: Xi_inv = (alpha^{-i})^{-1} = alpha^{i}; idx = i (direct mapping)
                            auto apply_corr3 = [&](std::vector<std::uint8_t>& vec){
                                for (int ii : *errp){
                                    // If we used Xi=alpha^{-i}, Xi_inv = alpha^{i}; if Xi=alpha^{i}, Xi_inv = alpha^{-i}
                                    std::uint8_t Xi_inv3 = used_b ? alogt[(std::size_t)((255 - (int)(ii % 255)) % 255)] : alogt[(std::size_t)(ii % 255)];
                                    std::uint8_t num3 = poly_eval(Omega3, Xi_inv3, logt, alogt);
                                    std::uint8_t den3 = poly_eval(Lp3, Xi_inv3, logt, alogt);
                                    if (!den3) continue;
                                    std::uint8_t mag3 = gf_div(num3, den3, logt, alogt);
                                    std::size_t idx3 = (std::size_t)ii;
                                    if (idx3 < chunk_len) vec[idx3] = gf_add(vec[idx3], mag3);
                                }
                            };
                            recv = recv_base;
                            apply_corr3(recv);
                            auto parr3 = compute_parity(recv.data(), chunk_len);
                            bool ok3 = (std::memcmp(parr3.data(), pr, r) == 0);
                            if (ok3){
                                int changes3 = 0; for (std::size_t ii=0; ii<chunk_len; ++ii){ if (recv[ii] != recv_base[ii]) ++changes3; }
                                corrected[(std::size_t)b] = changes3;
                            } else {
                                recv = recv_base;
                                if (dbg && *dbg){
                                    std::fprintf(stderr, "[ECC] Block %u standard-fallback failed (L3=%d, roots3=%zu)\n", b, L3, errp->size());
                                }
                            }
                        } else {
                            if (dbg && *dbg){
                                std::fprintf(stderr, "[ECC] Block %u standard-fallback Chien no/too many roots (L3=%d, roots3a=%zu, roots3b=%zu)\n", b, L3, err_pos3a.size(), err_pos3b.size());
                            }
                        }
                    } else {
                        if (dbg && *dbg){
                            std::fprintf(stderr, "[ECC] Block %u standard-fallback syndromes all zero (unexpected)\n", b);
                        }
                    }
                }
                // Alternate standard mapping with b=0 (j from 0..r-1)
                {
                    std::vector<std::uint8_t> S4(r, 0);
                    for (std::uint32_t jj=0; jj<r; ++jj){
                        std::uint8_t Sj=0;
                        for (std::uint32_t ii=0; ii<n; ++ii){
                            std::uint8_t c = recv[ii];
                            if (!c) continue;
                            int ex = (int)((ii * jj) % 255);
                            std::uint8_t term = (jj==0) ? c : alogt[(std::size_t)((logt[c] + ex) % 255)];
                            Sj = gf_add(Sj, term);
                        }
                        S4[jj] = Sj;
                    }
                    bool all_zero4 = true; for (auto v: S4) all_zero4 = all_zero4 && (v==0);
                    if (!all_zero4){
                        std::vector<std::uint8_t> Lambda4(1,1), B4(1,1); std::uint8_t bcoef4=1; int L4=0; int m4=1;
                        for (std::uint32_t nsy=0;nsy<r;++nsy){ std::uint8_t dsc=S4[nsy]; for (int i=1;i<=L4;++i){ if (i < (int)Lambda4.size() && (int)(nsy - i) >= 0) dsc = gf_add(dsc, gf_mul(Lambda4[(std::size_t)i], S4[(std::size_t)(nsy - i)], logt, alogt)); }
                            if (dsc == 0){ m4++; continue; }
                            std::vector<std::uint8_t> T4 = Lambda4;
                            auto poly_scale4 = [&](const std::vector<std::uint8_t>& p, std::uint8_t x){ std::vector<std::uint8_t> outp(p.size()); for (std::size_t ii=0;ii<p.size();++ii) outp[ii]=gf_mul(p[ii], x, logt, alogt); return outp; };
                            std::vector<std::uint8_t> xmb4(B4.size()+ (std::size_t)m4, 0); for (std::size_t ii=0;ii<B4.size();++ii) xmb4[ii]=B4[ii]; for (int sh=0; sh<m4; ++sh) xmb4.insert(xmb4.begin(), 0);
                            std::uint8_t scale4 = gf_div(dsc, bcoef4, logt, alogt);
                            auto scaled4 = poly_scale4(xmb4, scale4);
                            std::size_t nmax4 = std::max(Lambda4.size(), scaled4.size()); std::vector<std::uint8_t> newL4(nmax4,0);
                            for (std::size_t ii=0;ii<nmax4;++ii){ std::uint8_t av = (ii<Lambda4.size())?Lambda4[Lambda4.size()-1-ii]:0; std::uint8_t bv=(ii<scaled4.size())?scaled4[scaled4.size()-1-ii]:0; newL4[nmax4-1-ii] = gf_add(av, bv); }
                            Lambda4.swap(newL4);
                            if (2*L4 <= (int)nsy){ L4 = (int)nsy + 1 - L4; B4 = T4; bcoef4 = dsc; m4 = 1; } else { m4++; }
                        }
                        // Chien try both Xi
                        std::vector<int> err_pos4a; err_pos4a.reserve(r);
                        for (std::uint32_t ii=0; ii<n; ++ii){ std::uint8_t Xi = alogt[(std::size_t)((255 - (int)(ii % 255)) % 255)]; if (poly_eval(Lambda4, Xi, logt, alogt) == 0) err_pos4a.push_back((int)ii); }
                        std::vector<int> err_pos4b; err_pos4b.reserve(r);
                        for (std::uint32_t ii=0; ii<n; ++ii){ std::uint8_t Xi = alogt[(std::size_t)(ii % 255)]; if (poly_eval(Lambda4, Xi, logt, alogt) == 0) err_pos4b.push_back((int)ii); }
                        std::vector<int>* errp4 = nullptr; bool used_b4=false;
                        if (!err_pos4a.empty() && (int)err_pos4a.size() <= L4){ errp4=&err_pos4a; used_b4=false; }
                        else if (!err_pos4b.empty() && (int)err_pos4b.size() <= L4){ errp4=&err_pos4b; used_b4=true; }
                        if (errp4){
                            std::vector<std::uint8_t> Sx4 = S4; std::vector<std::uint8_t> prod4(Sx4.size()+Lambda4.size()-1, 0);
                            for (std::size_t ii=0;ii<Sx4.size();++ii) for (std::size_t jj=0;jj<Lambda4.size();++jj) prod4[ii+jj] = gf_add(prod4[ii+jj], gf_mul(Sx4[ii], Lambda4[jj], logt, alogt));
                            std::vector<std::uint8_t> Omega4(prod4.begin(), prod4.begin() + std::min<std::size_t>(prod4.size(), r));
                            std::vector<std::uint8_t> Lp4; for (std::size_t ii=1;ii<Lambda4.size();++ii) if (ii%2==1) Lp4.push_back(Lambda4[ii]);
                            auto apply_corr4 = [&](std::vector<std::uint8_t>& vec){ for (int ii : *errp4){ std::uint8_t Xi_inv4 = used_b4 ? alogt[(std::size_t)((255 - (int)(ii % 255)) % 255)] : alogt[(std::size_t)(ii % 255)]; std::uint8_t num4 = poly_eval(Omega4, Xi_inv4, logt, alogt); std::uint8_t den4 = poly_eval(Lp4, Xi_inv4, logt, alogt); if (!den4) continue; std::uint8_t mag4 = gf_div(num4, den4, logt, alogt); std::size_t idx4 = (std::size_t)ii; if (idx4 < chunk_len) vec[idx4] = gf_add(vec[idx4], mag4); } };
                            recv = recv_base; apply_corr4(recv); auto parr4 = compute_parity(recv.data(), chunk_len); bool ok4 = (std::memcmp(parr4.data(), pr, r) == 0);
                            if (ok4){ int changes4 = 0; for (std::size_t ii=0; ii<chunk_len; ++ii){ if (recv[ii] != recv_base[ii]) ++changes4; } corrected[(std::size_t)b] = changes4; }
                            else { recv = recv_base; if (dbg && *dbg){ std::fprintf(stderr, "[ECC] Block %u std(b=0)-fallback failed (L4=%d, roots=%zu)\n", b, L4, errp4->size()); } }
                        } else {
                            if (dbg && *dbg){ std::fprintf(stderr, "[ECC] Block %u std(b=0)-fallback Chien no/too many roots (L4=%d, roots4a=%zu, roots4b=%zu)\n", b, L4, err_pos4a.size(), err_pos4b.size()); }
                        }
                    }
                }
            }
            std::memcpy(out.data() + (std::size_t)b * k, recv.data(), k);
        }
        py::list counts; for (auto c : corrected) counts.append(c);
        return py::make_tuple(py::bytes(reinterpret_cast<const char*>(out.data()), (py::ssize_t)s.size()), counts);
    }, py::arg("payload"), py::arg("parity"), py::arg("k")=(std::uint32_t)223, py::arg("r")=(std::uint32_t)32,
       "GPU RS(255,223) ECC decode/correct payload using parity (host-side)");
#endif

    // Wave-based ECC functions
    m.def("wave_ecc_encode", [](py::bytes data, std::uint32_t redundancy_level, std::uint32_t seed_base) {
        std::string s = data;
        if (s.empty() || redundancy_level == 0) {
            return py::bytes("");
        }
        
#if defined(PLATFORM_METAL) || defined(__APPLE__)
        auto core = holo::MetalHoloCore();
        if (!core.available()) throw std::runtime_error("GPU backend not available");
        
        // Convert data to normalized float signal
        std::vector<float> signal(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            signal[i] = static_cast<float>(static_cast<unsigned char>(s[i])) / 255.0f;
        }
        
        // FFT transform to frequency domain
        std::vector<float> real, imag;
        core.fft_transform(signal, real, imag);
        
        // Apply symmetric normalization: kernel FFT divides by N; multiply by sqrt(N)
        // to achieve 1/sqrt(N) forward normalization overall
        {
            const float sqrtN = std::sqrt(static_cast<float>(signal.size()));
            for (std::size_t i = 0; i < real.size(); ++i) {
                real[i] *= sqrtN;
                imag[i] *= sqrtN;
            }
        }

        // Generate redundant views using seeded codebooks
        std::vector<float> parity_real(real.size() * redundancy_level, 0.0f);
        std::vector<float> parity_imag(imag.size() * redundancy_level, 0.0f);
        
        for (std::uint32_t r = 0; r < redundancy_level; ++r) {
            std::uint32_t seed = seed_base ^ r;
            std::vector<float> coded_real, coded_imag;
            
            // Apply seeded codebook
            core.apply_codebook(real, imag, coded_real, coded_imag, seed);
            
            // Store in parity with phase rotation for diversity
            float phase = static_cast<float>(r) * 2.0f * 3.14159265359f / static_cast<float>(redundancy_level);
            float cos_p = std::cos(phase);
            float sin_p = std::sin(phase);
            
            for (size_t i = 0; i < coded_real.size(); ++i) {
                size_t idx = r * real.size() + i;
                // Rotate by phase for redundancy diversity
                parity_real[idx] = coded_real[i] * cos_p - coded_imag[i] * sin_p;
                parity_imag[idx] = coded_real[i] * sin_p + coded_imag[i] * cos_p;
            }
        }
        
        // Convert parity to bytes (interleaved real/imag as float32)
        std::vector<uint8_t> parity_bytes;
        parity_bytes.reserve(parity_real.size() * 8);
        
        for (size_t i = 0; i < parity_real.size(); ++i) {
            // Store as float32 bytes
            float r = parity_real[i];
            float im = parity_imag[i];
            uint8_t* r_bytes = reinterpret_cast<uint8_t*>(&r);
            uint8_t* i_bytes = reinterpret_cast<uint8_t*>(&im);
            for (int j = 0; j < 4; ++j) parity_bytes.push_back(r_bytes[j]);
            for (int j = 0; j < 4; ++j) parity_bytes.push_back(i_bytes[j]);
        }
        
        return py::bytes(reinterpret_cast<const char*>(parity_bytes.data()), parity_bytes.size());
#else
        throw std::runtime_error("Wave ECC requires GPU backend");
#endif
    }, py::arg("data"), py::arg("redundancy_level")=3, py::arg("seed_base")=42,
       "Encode data using wave-based ECC with seeded redundancy");
    
    m.def("wave_ecc_decode", [](py::bytes data, py::bytes parity, std::uint32_t redundancy_level, std::uint32_t seed_base) {
        std::string s = data;
        std::string p = parity;
        
        if (s.empty()) {
            return py::make_tuple(py::bytes(""), 0);
        }
        
#if defined(PLATFORM_METAL) || defined(__APPLE__)
        auto core = holo::MetalHoloCore();
        if (!core.available()) throw std::runtime_error("GPU backend not available");
        
        // Note: Do not compute FFT of potentially corrupted input data.
        // Error detection relies solely on parity view agreement.
        
        // Decode parity bytes back to complex waves
        size_t expected_parity_size = s.size() * redundancy_level * 8;
        if (p.size() != expected_parity_size) {
            // Parity size mismatch, return uncorrected
            return py::make_tuple(py::bytes(s), 0);
        }
        
        std::vector<float> parity_real(s.size() * redundancy_level);
        std::vector<float> parity_imag(s.size() * redundancy_level);
        
        const uint8_t* parity_ptr = reinterpret_cast<const uint8_t*>(p.data());
        for (size_t i = 0; i < parity_real.size(); ++i) {
            float r, im;
            std::memcpy(&r, parity_ptr + i * 8, 4);
            std::memcpy(&im, parity_ptr + i * 8 + 4, 4);
            parity_real[i] = r;
            parity_imag[i] = im;
        }
        
        // Recover each redundant view (conjugate decode) for similarity analysis
        std::vector<std::vector<float>> recovered_reals(redundancy_level);
        std::vector<std::vector<float>> recovered_imags(redundancy_level);
        for (std::uint32_t r = 0; r < redundancy_level; ++r) {
            std::uint32_t seed = seed_base ^ r;
            // Extract this redundant view by undoing phase rotation
            std::vector<float> view_real(s.size());
            std::vector<float> view_imag(s.size());
            float phase = static_cast<float>(r) * 2.0f * 3.14159265359f / static_cast<float>(redundancy_level);
            float cos_p = std::cos(-phase);
            float sin_p = std::sin(-phase);
            for (size_t i = 0; i < view_real.size(); ++i) {
                size_t idx = r * s.size() + i;
                float pr = parity_real[idx];
                float pi = parity_imag[idx];
                view_real[i] = pr * cos_p - pi * sin_p;
                view_imag[i] = pr * sin_p + pi * cos_p;
            }
            // Apply conjugate codebook to recover original-domain components per view
            core.apply_codebook_conj(view_real, view_imag, recovered_reals[r], recovered_imags[r], seed);

            // Optional debug: report magnitude norm of recovered spectrum per view
            if (std::getenv("WAVE_ECC_DEBUG")) {
                double mag2 = 0.0;
                const auto& rr = recovered_reals[r];
                const auto& ri = recovered_imags[r];
                for (std::size_t i = 0; i < rr.size(); ++i) {
                    mag2 += double(rr[i]) * double(rr[i]) + double(ri[i]) * double(ri[i]);
                }
                std::fprintf(stderr, "[Wave ECC] View %u magnitude(norm2): %f\n", r, std::sqrt(std::max(0.0, mag2)));
            }
        }

        // Compute cosine similarity across recovered views (pairwise, averaged per view)
        std::vector<float> correlations(redundancy_level, 1.0f); // [-1,1], higher is better
        std::vector<bool> view_valid(redundancy_level, true);
        float avg_correlation = 0.0f;
        if (redundancy_level >= 2) {
            std::vector<double> mags(redundancy_level, 0.0);
            for (std::uint32_t r = 0; r < redundancy_level; ++r) {
                double m = 0.0;
                const auto& rr = recovered_reals[r];
                const auto& ri = recovered_imags[r];
                for (size_t i = 0; i < rr.size(); ++i) {
                    m += double(rr[i]) * double(rr[i]) + double(ri[i]) * double(ri[i]);
                }
                mags[r] = std::sqrt(std::max(0.0, m));
            }
            std::vector<double> corr_sum(redundancy_level, 0.0);
            std::vector<int> corr_cnt(redundancy_level, 0);
            for (std::uint32_t a = 0; a < redundancy_level; ++a) {
                for (std::uint32_t b = a + 1; b < redundancy_level; ++b) {
                    double dot = 0.0;
                    const auto& ar = recovered_reals[a];
                    const auto& ai = recovered_imags[a];
                    const auto& br = recovered_reals[b];
                    const auto& bi = recovered_imags[b];
                    for (size_t i = 0; i < ar.size(); ++i) {
                        dot += double(ar[i]) * double(br[i]) + double(ai[i]) * double(bi[i]);
                    }
                    double sim = 0.0;
                    if (mags[a] > 0.0 && mags[b] > 0.0) sim = dot / (mags[a] * mags[b]);
                    corr_sum[a] += sim; corr_cnt[a] += 1;
                    corr_sum[b] += sim; corr_cnt[b] += 1;
                }
            }
            for (std::uint32_t r = 0; r < redundancy_level; ++r) {
                correlations[r] = (corr_cnt[r] > 0) ? float(corr_sum[r] / double(corr_cnt[r])) : 1.0f;
                avg_correlation += correlations[r];
            }
            avg_correlation /= float(redundancy_level);
        } else {
            avg_correlation = 1.0f; // Single view: assume valid
        }

        // Debug logging of correlations if enabled
        if (std::getenv("WAVE_ECC_DEBUG")) {
            for (std::uint32_t r = 0; r < redundancy_level; ++r) {
                std::fprintf(stderr, "[Wave ECC] View %u avg similarity with others: %f\n", r, correlations[r]);
            }
        }

        // Threshold for cosine similarity: good views near 1.0
        float threshold = 0.95f;
        int errors_detected = 0;
        for (std::uint32_t r = 0; r < redundancy_level; ++r) {
            if (correlations[r] < threshold) {
                view_valid[r] = false;
                errors_detected++;
                if (std::getenv("WAVE_ECC_DEBUG")) {
                    std::fprintf(stderr, "[Wave ECC] View %u marked as corrupted (similarity=%f)\n", r, correlations[r]);
                }
            }
        }
        
        // Reconstruct from parity views. If some views are invalid, average only valid ones.
        // If none flagged invalid, average all views. Ignore input data for reconstruction.
        {
            // Default: use the first valid view (robust across dims)
            int chosen = -1;
            for (std::uint32_t r = 0; r < redundancy_level; ++r) {
                if (view_valid[r]) { chosen = (int)r; break; }
            }
            if (chosen < 0) chosen = 0; // fallback
            const auto& corrected_real = recovered_reals[(std::size_t)chosen];
            const auto& corrected_imag = recovered_imags[(std::size_t)chosen];
            // iFFT to get corrected signal
            // Compensate for kernel normalization using symmetric scheme:
            // kernel FFT outputs DFT/N; we scaled by sqrt(N) in encode to get DFT/sqrt(N).
            // kernel iFFT divides by N; pre-scale by sqrt(N) here so iFFT returns original.
            const float sqrtN = std::sqrt(static_cast<float>(s.size()));
            std::vector<float> pre_ifft_real = corrected_real;
            std::vector<float> pre_ifft_imag = corrected_imag;
            for (std::size_t i = 0; i < pre_ifft_real.size(); ++i) {
                pre_ifft_real[i] *= sqrtN;
                pre_ifft_imag[i] *= sqrtN;
            }
            std::vector<float> corrected_signal;
            core.ifft_transform(pre_ifft_real, pre_ifft_imag, corrected_signal);
            // Convert back to bytes
            std::vector<uint8_t> corrected_bytes(s.size());
            for (size_t i = 0; i < s.size(); ++i) {
                float val = corrected_signal[i] * 255.0f;
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                corrected_bytes[i] = static_cast<uint8_t>(val + 0.5f);
            }
            return py::make_tuple(
                py::bytes(reinterpret_cast<const char*>(corrected_bytes.data()), corrected_bytes.size()),
                errors_detected
            );
        }
#else
        throw std::runtime_error("Wave ECC requires GPU backend");
#endif
    }, py::arg("data"), py::arg("parity"), py::arg("redundancy_level")=3, py::arg("seed_base")=42,
       "Decode data using wave-based ECC with correlation detection and voting correction");

    m.def("available_platforms", &HolographicGPUWrapper::available_platforms,
          "Return a list of available GPU platforms (e.g., ['cuda','metal']).");
}
