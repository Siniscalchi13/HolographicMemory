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
        return backend_->batch_encode_fft_zero_copy(ptr, batch, data_len, pattern_dim);
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
        .def_property_readonly("layers_initialized", &HolographicGPUWrapper::layers_initialized);

    m.def("available_platforms", &HolographicGPUWrapper::available_platforms,
          "Return a list of available GPU platforms (e.g., ['cuda','metal']).");
}
