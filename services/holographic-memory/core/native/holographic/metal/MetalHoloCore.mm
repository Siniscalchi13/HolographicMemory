#ifdef __APPLE__
#include "MetalHoloCore.hpp"
#include "MetalBackend.hpp"
#include <chrono>

using namespace std::chrono;

namespace holo {

MetalHoloCore::MetalHoloCore() : backend_(new MetalBackend()) {}
MetalHoloCore::~MetalHoloCore() = default;

bool MetalHoloCore::available() const noexcept { return backend_ && backend_->available(); }

void MetalHoloCore::fft_transform(const std::vector<float>& input,
                                  std::vector<float>& real_out,
                                  std::vector<float>& imag_out) {
    if (!available()) { real_out.clear(); imag_out.clear(); return; }
    auto t0 = high_resolution_clock::now();
    backend_->fft_transform(input, real_out, imag_out);
    auto dt = duration_cast<duration<double, std::milli>>(high_resolution_clock::now() - t0).count();
    metrics_.batch_time_ms = dt;
}

void MetalHoloCore::ifft_transform(const std::vector<float>& real_in,
                                   const std::vector<float>& imag_in,
                                   std::vector<float>& time_out) {
    if (!available()) { time_out.clear(); return; }
    backend_->ifft_transform(real_in, imag_in, time_out);
}

void MetalHoloCore::apply_codebook(const std::vector<float>& in_real,
                                   const std::vector<float>& in_imag,
                                   std::vector<float>& out_real,
                                   std::vector<float>& out_imag,
                                   uint32_t seed) {
    if (!available()) { out_real.clear(); out_imag.clear(); return; }
    backend_->apply_codebook(in_real, in_imag, out_real, out_imag, seed);
}

void MetalHoloCore::apply_codebook_conj(const std::vector<float>& in_real,
                                        const std::vector<float>& in_imag,
                                        std::vector<float>& out_real,
                                        std::vector<float>& out_imag,
                                        uint32_t seed) {
    if (!available()) { out_real.clear(); out_imag.clear(); return; }
    backend_->apply_codebook_conj(in_real, in_imag, out_real, out_imag, seed);
}

void MetalHoloCore::accumulate_add_time(std::vector<float>& dst,
                                        const std::vector<float>& add) {
    if (!available()) return;
    backend_->accumulate_add_time(dst, add);
}

std::vector<std::vector<float>> MetalHoloCore::batch_encode(const std::vector<std::vector<float>>& batch_data,
                                                            uint32_t pattern_dim) {
    if (!available() || batch_data.empty()) return {};
    auto t0 = high_resolution_clock::now();
    auto out = backend_->batch_holographic_store(batch_data, pattern_dim);
    auto dt = duration_cast<duration<double, std::milli>>(high_resolution_clock::now() - t0).count();
    metrics_.batch_time_ms = dt;
    if (dt > 0.0) metrics_.operations_per_second = (uint64_t)(batch_data.size() * 1000.0 / dt);
    double bytes = (double)(batch_data.size() * batch_data[0].size() + out.size() * pattern_dim) * sizeof(float);
    if (dt > 0.0) metrics_.memory_bandwidth_gb_s = (bytes / (dt/1000.0)) / (1024.0*1024.0*1024.0);
    return out;
}

std::vector<float> MetalHoloCore::similarity_search(const std::vector<float>& query,
                                                    const std::vector<std::vector<float>>& stored) {
    if (!available() || stored.empty()) return {};
    auto t0 = high_resolution_clock::now();
    auto sims = backend_->similarity_search(query, stored);
    auto dt = duration_cast<duration<double, std::milli>>(high_resolution_clock::now() - t0).count();
    metrics_.search_time_ms = dt;
    return sims;
}

std::vector<std::vector<float>> MetalHoloCore::batch_encode_fft(const std::vector<std::vector<float>>& batch_data,
                                                                uint32_t pattern_dim) {
    if (!available() || batch_data.empty()) return {};
    auto t0 = high_resolution_clock::now();
    auto out = backend_->batch_encode_fft(batch_data, pattern_dim);
    auto dt = duration_cast<duration<double, std::milli>>(high_resolution_clock::now() - t0).count();
    metrics_.batch_time_ms = dt;
    if (dt > 0.0) metrics_.operations_per_second = (uint64_t)(batch_data.size() * 1000.0 / dt);
    double bytes = (double)(batch_data.size() * batch_data[0].size() + out.size() * pattern_dim) * sizeof(float);
    if (dt > 0.0) metrics_.memory_bandwidth_gb_s = (bytes / (dt/1000.0)) / (1024.0*1024.0*1024.0);
    return out;
}

std::vector<std::vector<float>> MetalHoloCore::batch_encode_fft_ultra(const std::vector<std::vector<float>>& batch_data,
                                                                      uint32_t pattern_dim) {
    if (!available() || batch_data.empty()) return {};
    auto t0 = high_resolution_clock::now();
    auto out = backend_->batch_encode_fft_ultra(batch_data, pattern_dim);
    auto dt = duration_cast<duration<double, std::milli>>(high_resolution_clock::now() - t0).count();
    metrics_.batch_time_ms = dt;
    if (dt > 0.0) metrics_.operations_per_second = (uint64_t)(batch_data.size() * 1000.0 / dt);
    double bytes = (double)(batch_data.size() * batch_data[0].size() + out.size() * pattern_dim) * sizeof(float);
    if (dt > 0.0) metrics_.memory_bandwidth_gb_s = (bytes / (dt/1000.0)) / (1024.0*1024.0*1024.0);
    return out;
}

std::vector<std::vector<float>> MetalHoloCore::batch_encode_from_ptr(const float* ptr,
                                                                     uint32_t batch,
                                                                     uint32_t data_len,
                                                                     uint32_t pattern_dim,
                                                                     bool use_ultra) {
    if (!available() || ptr == nullptr || batch == 0 || data_len == 0) return {};
    // Flatten from contiguous (batch,data_len) with padding/truncation to pattern_dim
    std::vector<std::vector<float>> batch_data;
    batch_data.reserve(batch);
    for (uint32_t b = 0; b < batch; ++b) {
        const float* row = ptr + (size_t)b * data_len;
        std::vector<float> v;
        v.resize((size_t)data_len);
        std::memcpy(v.data(), row, (size_t)data_len * sizeof(float));
        batch_data.emplace_back(std::move(v));
    }
    return use_ultra ? batch_encode_fft_ultra(batch_data, pattern_dim)
                     : batch_encode_fft(batch_data, pattern_dim);
}

MetalHoloCore::DeviceAnalysisResult MetalHoloCore::analyze_metrics_hostback(const float* v1, const float* v2, std::uint32_t dim) {
    DeviceAnalysisResult r{0.0f, 0.0f, 0.0, 0.0f};
    if (!available() || v1 == nullptr || v2 == nullptr || dim == 0) return r;
    auto t0 = high_resolution_clock::now();
    auto tup = backend_->analyze_metrics(v1, v2, dim);
    auto dt = duration_cast<duration<double, std::milli>>(high_resolution_clock::now() - t0).count();
    metrics_.search_time_ms = dt;
    r.visibility = std::get<0>(tup);
    r.coherence = std::get<1>(tup);
    r.bell_violation = std::get<2>(tup);
    r.orthogonality = std::get<3>(tup);
    return r;
}

// ============================================================================
// GPU COMPRESSION PIPELINE - KERNEL 1: QUANTIZATION
// ============================================================================

std::vector<std::vector<float>> MetalHoloCore::gpu_holographic_quantize(
    const std::vector<std::vector<float>>& input_real,
    const std::vector<std::vector<float>>& input_imag,
    uint32_t layer_index,
    const QuantizationParams& params) {
    
    if (!available() || input_real.empty() || input_imag.empty()) return {};
    
    // Convert MetalHoloCore::QuantizationParams to MetalBackend::QuantizationParams
    MetalBackend::QuantizationParams backend_params;
    for (int i = 0; i < 7; ++i) {
        backend_params.phase_bits[i] = params.phase_bits[i];
        backend_params.amplitude_bits[i] = params.amplitude_bits[i];
        backend_params.phase_step[i] = params.phase_step[i];
        backend_params.amplitude_step[i] = params.amplitude_step[i];
        backend_params.max_phase_error[i] = params.max_phase_error[i];
    }
    
    auto t0 = high_resolution_clock::now();
    auto result = backend_->gpu_holographic_quantize(input_real, input_imag, layer_index, backend_params);
    auto dt = duration_cast<duration<double, std::milli>>(high_resolution_clock::now() - t0).count();
    metrics_.batch_time_ms = dt;
    
    return result;
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>, 
           std::vector<std::vector<float>>, std::vector<std::vector<float>>>
MetalHoloCore::gpu_holographic_quantize_with_validation(
    const std::vector<std::vector<float>>& input_real,
    const std::vector<std::vector<float>>& input_imag,
    uint32_t layer_index,
    const QuantizationParams& params) {
    
    if (!available() || input_real.empty() || input_imag.empty()) {
        return {{}, {}, {}, {}};
    }
    
    // Convert MetalHoloCore::QuantizationParams to MetalBackend::QuantizationParams
    MetalBackend::QuantizationParams backend_params;
    for (int i = 0; i < 7; ++i) {
        backend_params.phase_bits[i] = params.phase_bits[i];
        backend_params.amplitude_bits[i] = params.amplitude_bits[i];
        backend_params.phase_step[i] = params.phase_step[i];
        backend_params.amplitude_step[i] = params.amplitude_step[i];
        backend_params.max_phase_error[i] = params.max_phase_error[i];
    }
    
    auto t0 = high_resolution_clock::now();
    auto result = backend_->gpu_holographic_quantize_with_validation(input_real, input_imag, layer_index, backend_params);
    auto dt = duration_cast<duration<double, std::milli>>(high_resolution_clock::now() - t0).count();
    metrics_.batch_time_ms = dt;
    
    return result;
}

std::array<float, 4> MetalHoloCore::gpu_quantization_statistics(
    const std::vector<std::vector<float>>& phase_errors,
    const std::vector<std::vector<float>>& amplitude_errors) {
    
    if (!available() || phase_errors.empty() || amplitude_errors.empty()) {
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }
    
    auto t0 = high_resolution_clock::now();
    auto result = backend_->gpu_quantization_statistics(phase_errors, amplitude_errors);
    auto dt = duration_cast<duration<double, std::milli>>(high_resolution_clock::now() - t0).count();
    metrics_.search_time_ms = dt;
    
    return result;
}

// ============================================================================
// GPU COMPRESSION PIPELINE - KERNEL 2: BITPLANE EXTRACTION - REMOVED
// ============================================================================
// Removed: bitplane extraction methods - no longer needed with holographic wave reconstruction

// Removed: All bitplane extraction method implementations - no longer needed with holographic wave reconstruction

// IGPUBackend interface methods
bool MetalHoloCore::initialize(const GPUConfig& cfg) {
    // Initialize the Metal backend
    backend_ = std::make_unique<MetalBackend>();
    return backend_ != nullptr && available();
}

std::vector<std::vector<float>> MetalHoloCore::batch_encode_fft_zero_copy(const float* ptr,
                                                                         std::uint32_t batch,
                                                                         std::uint32_t data_len,
                                                                         std::uint32_t pattern_dim) {
    if (!available() || !ptr) {
        return {};
    }
    
    // Convert pointer to vector format and use batch_encode_fft_ultra
    std::vector<std::vector<float>> batch_data(batch);
    for (uint32_t i = 0; i < batch; ++i) {
        batch_data[i].resize(data_len);
        std::copy(ptr + i * data_len, ptr + (i + 1) * data_len, batch_data[i].begin());
    }
    
    return backend_->batch_encode_fft_ultra(batch_data, pattern_dim);
}

GPUMetrics MetalHoloCore::get_metrics() const {
    if (!available()) {
        return {};
    }
    
    // Convert PerformanceMetrics to GPUMetrics
    GPUMetrics gpu_metrics;
    gpu_metrics.host_ms = metrics_.batch_time_ms;
    gpu_metrics.device_ms = metrics_.batch_time_ms;
    gpu_metrics.h2d_time_ms = 0.0;
    gpu_metrics.fft_time_ms = 0.0;
    gpu_metrics.d2h_time_ms = 0.0;
    gpu_metrics.operations_per_second = metrics_.operations_per_second;
    gpu_metrics.memory_bandwidth_gb_s = metrics_.memory_bandwidth_gb_s;
    gpu_metrics.device_utilization = 0.0;
    
    return gpu_metrics;
}

} // namespace holo

#endif
