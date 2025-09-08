#ifdef __APPLE__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "metal/MetalBackend.hpp"

namespace py = pybind11;
using holo::MetalBackend;

PYBIND11_MODULE(holographic_metal, m) {
    m.doc() = "Metal GPU acceleration for Holographic Memory (experimental)";
    
    py::class_<MetalBackend>(m, "_Backend")
        .def(py::init<>())
        .def("available", &MetalBackend::available)
        .def("vector_add", &MetalBackend::vector_add)
        .def("batch_holographic_store", &MetalBackend::batch_holographic_store);

    m.def("vector_add_sum", [](const std::vector<float>& a, const std::vector<float>& b) {
        MetalBackend mb;
        if (!mb.available()) throw std::runtime_error("Metal not available");
        return mb.vector_add(a, b);
    });
    
    m.def("gpu_batch_store", [](const std::vector<std::vector<float>>& batch_data, uint32_t pattern_dimension) {
        MetalBackend mb;
        if (!mb.available()) throw std::runtime_error("Metal GPU not available");
        return mb.batch_holographic_store(batch_data, pattern_dimension);
    });
}

#endif
