#include "GPUBackend.hpp"
#include <chrono>

#ifdef PLATFORM_METAL
#include "metal/MetalHoloCore.hpp"
#endif

#ifdef PLATFORM_CUDA
#include "cuda/CudaBackend.hpp"
#endif

#ifdef PLATFORM_ROCM
#include "rocm/HipBackend.hpp"
#endif

namespace holo {

namespace {

class MetalAdapter : public IGPUBackend {
public:
    bool initialize(const GPUConfig& cfg) override {
#ifdef PLATFORM_METAL
        (void)cfg;
        core_ = std::make_unique<MetalHoloCore>();
        return core_->available();
#else
        (void)cfg; return false;
#endif
    }

    std::vector<std::vector<float>> batch_encode_fft_zero_copy(const float* ptr,
                                                               std::uint32_t batch,
                                                               std::uint32_t data_len,
                                                               std::uint32_t pattern_dim) override {
#ifdef PLATFORM_METAL
        auto t0 = std::chrono::high_resolution_clock::now();
        auto out = core_->batch_encode_from_ptr(ptr, batch, data_len, pattern_dim, /*use_ultra=*/true);
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
        auto m = core_->metrics();
        metrics_.host_ms = ms;
        metrics_.device_ms = m.batch_time_ms;
        metrics_.h2d_time_ms = 0.0;
        metrics_.fft_time_ms = 0.0;
        metrics_.d2h_time_ms = 0.0;
        metrics_.operations_per_second = m.operations_per_second;
        metrics_.memory_bandwidth_gb_s = m.memory_bandwidth_gb_s;
        metrics_.device_utilization = 0.0; // Not available on Metal path
        return out;
#else
        (void)ptr; (void)batch; (void)data_len; (void)pattern_dim; return {};
#endif
    }

    GPUMetrics get_metrics() const override { return metrics_; }

    bool analyze_device_metrics(const float* vec1,
                                const float* vec2,
                                std::uint32_t dimension,
                                DeviceAnalysisResult& out) override {
#ifdef PLATFORM_METAL
        if (!core_) return false;
        // Delegate to Metal core helper; fall back to simple host calc if necessary
        try {
            auto r = core_->analyze_metrics_hostback(vec1, vec2, dimension);
            out.visibility = r.visibility;
            out.coherence = r.coherence;
            out.bell_violation = r.bell_violation;
            out.orthogonality = r.orthogonality;
            return true;
        } catch (...) { return false; }
#else
        (void)vec1; (void)vec2; (void)dimension; (void)out; return false;
#endif
    }

private:
#ifdef PLATFORM_METAL
    std::unique_ptr<MetalHoloCore> core_;
#endif
    GPUMetrics metrics_{};
};

#ifdef PLATFORM_CUDA
class CudaAdapter : public IGPUBackend {
public:
    bool initialize(const GPUConfig& cfg) override {
        (void)cfg;
        backend_ = std::make_unique<CudaBackend>();
        return backend_->available();
    }
    std::vector<std::vector<float>> batch_encode_fft_zero_copy(const float* ptr,
                                                               std::uint32_t batch,
                                                               std::uint32_t data_len,
                                                               std::uint32_t pattern_dim) override {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto out = backend_->batch_encode_fft_ultra(ptr, batch, data_len, pattern_dim);
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
        auto m = backend_->metrics();
        GPUMetrics g; g.host_ms = ms; g.device_ms = m.device_ms; g.h2d_time_ms = m.h2d_ms; g.fft_time_ms = m.fft_ms; g.d2h_time_ms = m.d2h_ms; g.operations_per_second = m.ops_per_s; g.memory_bandwidth_gb_s = m.bandwidth_gbs; g.device_utilization = 0.0; metrics_ = g;
        return out;
    }
    GPUMetrics get_metrics() const override { return metrics_; }

    bool analyze_device_metrics(const float* vec1,
                                const float* vec2,
                                std::uint32_t dimension,
                                DeviceAnalysisResult& out) override {
        if (!backend_) return false;
        try {
            auto t = backend_->analyze_metrics(vec1, vec2, dimension);
            out.visibility = std::get<0>(t);
            out.coherence = std::get<1>(t);
            out.bell_violation = std::get<2>(t);
            out.orthogonality = std::get<3>(t);
            return true;
        } catch (...) { return false; }
    }
private:
    std::unique_ptr<CudaBackend> backend_;
    GPUMetrics metrics_{};
};
#endif

#ifdef PLATFORM_ROCM
class HipAdapter : public IGPUBackend {
public:
    bool initialize(const GPUConfig& cfg) override {
        (void)cfg; backend_ = std::make_unique<HipBackend>(); return backend_->available();
    }
    std::vector<std::vector<float>> batch_encode_fft_zero_copy(const float* ptr,
                                                               std::uint32_t batch,
                                                               std::uint32_t data_len,
                                                               std::uint32_t pattern_dim) override {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto out = backend_->batch_encode_fft_ultra(ptr, batch, data_len, pattern_dim);
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now()-t0).count();
        GPUMetrics g; g.host_ms = ms; metrics_ = g; return out;
    }
    GPUMetrics get_metrics() const override { return metrics_; }
    bool analyze_device_metrics(const float* vec1,
                                const float* vec2,
                                std::uint32_t dimension,
                                DeviceAnalysisResult& out) override {
        if (!backend_) return false;
        try {
            auto t = backend_->analyze_metrics(vec1, vec2, dimension);
            out.visibility = std::get<0>(t);
            out.coherence = std::get<1>(t);
            out.bell_violation = std::get<2>(t);
            out.orthogonality = std::get<3>(t);
            return true;
        } catch (...) { return false; }
    }
private:
    std::unique_ptr<HipBackend> backend_;
    GPUMetrics metrics_{};
};
#endif

} // anonymous

std::vector<GPUPlatform> IGPUBackend::get_available_platforms() {
    std::vector<GPUPlatform> v;
#ifdef PLATFORM_METAL
    {
        MetalHoloCore c; if (c.available()) v.push_back(GPUPlatform::METAL);
    }
#endif
#ifdef PLATFORM_CUDA
    {
        CudaBackend b; if (b.available()) v.push_back(GPUPlatform::CUDA);
    }
#endif
#ifdef PLATFORM_ROCM
    {
        HipBackend b; if (b.available()) v.push_back(GPUPlatform::ROCM);
    }
#endif
    return v;
}

std::unique_ptr<IGPUBackend> IGPUBackend::create_backend(GPUPlatform pf) {
    switch (pf) {
        case GPUPlatform::METAL: return std::unique_ptr<IGPUBackend>(new MetalAdapter());
#ifdef PLATFORM_CUDA
        case GPUPlatform::CUDA: return std::unique_ptr<IGPUBackend>(new CudaAdapter());
#endif
#ifdef PLATFORM_ROCM
        case GPUPlatform::ROCM: return std::unique_ptr<IGPUBackend>(new HipAdapter());
#endif
        default: return nullptr;
    }
}

} // namespace holo
