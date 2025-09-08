#ifdef __APPLE__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "metal/MetalHoloCore.hpp"

namespace py = pybind11;

PYBIND11_MODULE(holographic_gpu, m) {
    m.doc() = "C++ Metal GPU backend for Holographic Memory";

    py::class_<holo::PerformanceMetrics>(m, "PerformanceMetrics")
        .def_readonly("batch_time_ms", &holo::PerformanceMetrics::batch_time_ms)
        .def_readonly("search_time_ms", &holo::PerformanceMetrics::search_time_ms)
        .def_readonly("operations_per_second", &holo::PerformanceMetrics::operations_per_second)
        .def_readonly("memory_bandwidth_gb_s", &holo::PerformanceMetrics::memory_bandwidth_gb_s);

    py::class_<holo::MetalHoloCore>(m, "MetalHolographicBackend")
        .def(py::init<>())
        .def("available", &holo::MetalHoloCore::available)
        .def("fft_transform", [](holo::MetalHoloCore& self, const std::vector<float>& input){
            std::vector<float> r,i;
            self.fft_transform(input, r, i);
            return py::make_tuple(r,i);
        })
        .def("batch_encode", &holo::MetalHoloCore::batch_encode)
        .def("similarity_search", &holo::MetalHoloCore::similarity_search)
        .def("metrics", &holo::MetalHoloCore::metrics);
}

#endif

