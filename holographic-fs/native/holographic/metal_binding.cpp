#ifdef __APPLE__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "metal/MetalHolographicBackend.hpp"

namespace py = pybind11;

PYBIND11_MODULE(holographic_metal, m) {
    m.doc() = "Metal GPU acceleration for Holographic Memory (experimental)";

    py::class_<holo::MetalHolographicBackend::PerformanceMetrics>(m, "PerformanceMetrics")
        .def_readonly("batch_encode_time_ms", &holo::MetalHolographicBackend::PerformanceMetrics::batch_encode_time_ms)
        .def_readonly("operations_per_second", &holo::MetalHolographicBackend::PerformanceMetrics::operations_per_second)
        .def_readonly("memory_bandwidth_gb_s", &holo::MetalHolographicBackend::PerformanceMetrics::memory_bandwidth_gb_s);

    py::class_<holo::MetalHolographicBackend>(m, "MetalHolographicBackend")
        .def(py::init<>())
        .def("initialize", &holo::MetalHolographicBackend::initialize)
        .def("is_available", &holo::MetalHolographicBackend::is_available)
        .def("batch_encode", [](holo::MetalHolographicBackend &self, const std::vector<std::vector<float>>& batch, uint32_t pattern_dim){
            if (batch.empty()) return std::vector<std::vector<float>>{};
            uint32_t bs = (uint32_t)batch.size();
            uint32_t dl = (uint32_t)batch[0].size();
            std::vector<float> flat;
            flat.reserve((size_t)bs*dl);
            for (auto &v: batch) flat.insert(flat.end(), v.begin(), v.end());
            auto flat_out = self.batch_encode(flat, bs, dl, pattern_dim);
            std::vector<std::vector<float>> out;
            out.reserve(bs);
            for (uint32_t i=0;i<bs;i++) {
                out.emplace_back(flat_out.begin()+i*pattern_dim, flat_out.begin()+(i+1)*pattern_dim);
            }
            return out;
        })
        .def("similarity_search", [](holo::MetalHolographicBackend &self, const std::vector<float>& query, const std::vector<std::vector<float>>& stored){
            if (stored.empty()) return std::vector<float>{};
            uint32_t pc = (uint32_t)stored.size();
            uint32_t dim = (uint32_t)stored[0].size();
            std::vector<float> flat;
            flat.reserve((size_t)pc*dim);
            for (auto &v: stored) flat.insert(flat.end(), v.begin(), v.end());
            return self.similarity_search(query, flat, pc, dim);
        })
        .def("get_last_metrics", &holo::MetalHolographicBackend::get_last_metrics);
}

#endif

