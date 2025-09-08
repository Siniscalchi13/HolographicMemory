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

} // namespace holo

#endif
