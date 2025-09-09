#ifdef __APPLE__
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
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
        .def("batch_encode_fft", &holo::MetalHoloCore::batch_encode_fft)
        .def("batch_encode_fft_ultra", &holo::MetalHoloCore::batch_encode_fft_ultra)
        // Zero-copy from NumPy (CPU) to C++ (avoids Python loops); still copies to GPU
        .def("batch_encode_numpy", [](holo::MetalHoloCore& self, py::array_t<float, py::array::c_style | py::array::forcecast> arr, uint32_t pattern_dim){
            if (arr.ndim() != 2) throw std::runtime_error("array must be 2D (batch, data_len)");
            uint32_t batch = (uint32_t)arr.shape(0);
            uint32_t data_len = (uint32_t)arr.shape(1);
            const float* ptr = arr.data();
            return self.batch_encode_from_ptr(ptr, batch, data_len, pattern_dim, /*use_ultra=*/false);
        }, py::arg("array"), py::arg("pattern_dim"))
        .def("batch_encode_fft_ultra_numpy", [](holo::MetalHoloCore& self, py::array_t<float, py::array::c_style | py::array::forcecast> arr, uint32_t pattern_dim){
            if (arr.ndim() != 2) throw std::runtime_error("array must be 2D (batch, data_len)");
            uint32_t batch = (uint32_t)arr.shape(0);
            uint32_t data_len = (uint32_t)arr.shape(1);
            const float* ptr = arr.data();
            return self.batch_encode_from_ptr(ptr, batch, data_len, pattern_dim, /*use_ultra=*/true);
        }, py::arg("array"), py::arg("pattern_dim"))
        .def("similarity_search", &holo::MetalHoloCore::similarity_search)
        .def("metrics", &holo::MetalHoloCore::metrics);
}

#endif
