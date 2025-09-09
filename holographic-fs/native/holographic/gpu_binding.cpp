// Cross-platform GPU binding.
// Exposes a single Python module `holographic_gpu` with a portable
// wrapper that selects the best available backend (Metal/CUDA/ROCm).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

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
        return backend_->batch_encode_fft_zero_copy(flat.data(), bs, dl, pattern_dim);
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

private:
    std::unique_ptr<holo::IGPUBackend> backend_;
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
        .def("retrieve_bytes", &HolographicGPUWrapper::retrieve_bytes, py::arg("path"));

    m.def("available_platforms", &HolographicGPUWrapper::available_platforms,
          "Return a list of available GPU platforms (e.g., ['cuda','metal']).");
}
