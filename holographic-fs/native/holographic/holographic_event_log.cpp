/*
 * HolographicEventLog - PoC 4D wave event log (x,y,z,t)
 * -----------------------------------------------------
 * - Append event: encode as localized wave packet at current time slice
 * - Reconstruct state up to t: sum spatial field across time and compute
 *   interference intensity via 3D FFT: I = |FFTN(sum_t psi(x,y,z,t))|^2
 * - Goal: enable fast replay/time-travel via constructive interference
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <fftw3.h>

#include <vector>
#include <complex>
#include <string>
#include <mutex>
#include <cmath>
#include <chrono>

namespace py = pybind11;

class HolographicEventLog {
private:
    size_t nx, ny, nz, nt;
    size_t total3, total4;
    std::vector<std::complex<double>> field4d; // flattened [t][z][y][x]
    size_t cur_t;
    uint64_t total_events;
    std::vector<std::complex<double>> work3d;  // size total3
    fftw_plan plan3d;
    std::mutex mtx;

    static inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
        return x + y * nx + z * nx * ny;
    }
    static inline size_t idx4(size_t x, size_t y, size_t z, size_t t, size_t nx, size_t ny, size_t nz) {
        return idx3(x, y, z, nx, ny) + t * (nx * ny * nz);
    }

    static inline double now_sec() {
        return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    }

    // Simple hash to seed placement/phase
    static inline uint64_t hash64(const std::string& s) {
        std::hash<std::string> H;
        return (uint64_t)H(s);
    }

public:
    HolographicEventLog(size_t nx_, size_t ny_, size_t nz_, size_t nt_)
        : nx(nx_), ny(ny_), nz(nz_), nt(nt_),
          total3(nx_ * ny_ * nz_), total4(nx_ * ny_ * nz_ * nt_),
          field4d(total4), cur_t(0), total_events(0), work3d(total3) {
        // Create a 3D FFT plan in-place over work3d
        plan3d = fftw_plan_dft_3d((int)nz, (int)ny, (int)nx,
                                  reinterpret_cast<fftw_complex*>(work3d.data()),
                                  reinterpret_cast<fftw_complex*>(work3d.data()),
                                  FFTW_FORWARD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
    }

    ~HolographicEventLog() {
        fftw_destroy_plan(plan3d);
    }

    void append_event(const std::string& payload) {
        std::lock_guard<std::mutex> lock(mtx);
        uint64_t h = hash64(payload + std::to_string(total_events));
        size_t cx = (size_t)(h ^ 0x9e3779b97f4a7c15ULL) % nx;
        size_t cy = (size_t)((h >> 17) ^ 0xc3a5c85c97cb3127ULL) % ny;
        size_t cz = (size_t)((h >> 29) ^ 0xb492b66fbe98f273ULL) % nz;
        double phase = (double)((h >> 7) & 0xFFFF) / 65535.0 * 2.0 * M_PI;
        std::complex<double> u(std::cos(phase), std::sin(phase));
        // Small localized packet (radius r)
        int r = 2; // 5x5x5 max support
        for (int dz = -r; dz <= r; ++dz) {
            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    int x = (int)cx + dx; int y = (int)cy + dy; int z = (int)cz + dz;
                    if (x < 0 || y < 0 || z < 0 || x >= (int)nx || y >= (int)ny || z >= (int)nz) continue;
                    double d2 = (double)(dx * dx + dy * dy + dz * dz);
                    double amp = std::exp(-d2 / 4.0); // gaussian-ish
                    field4d[idx4((size_t)x, (size_t)y, (size_t)z, cur_t, nx, ny, nz)] += amp * u;
                }
            }
        }
        cur_t = (cur_t + 1) % nt;
        total_events += 1;
    }

    // Sum psi(x,y,z,t) across t<=upto_t and return intensity of 3D FFT
    py::array_t<float> reconstruct_intensity_3d(size_t upto_t) {
        std::lock_guard<std::mutex> lock(mtx);
        if (upto_t >= nt) upto_t = nt - 1;
        // Sum slices
        std::fill(work3d.begin(), work3d.end(), std::complex<double>(0.0, 0.0));
        for (size_t t = 0; t <= upto_t; ++t) {
            const auto* base = &field4d[t * total3];
            for (size_t i = 0; i < total3; ++i) work3d[i] += base[i];
        }
        // FFT then intensity
        fftw_execute(plan3d);
        py::array_t<float> out({(py::ssize_t)nz, (py::ssize_t)ny, (py::ssize_t)nx});
        auto* o = (float*)out.request().ptr;
        for (size_t i = 0; i < total3; ++i) {
            double v = std::norm(work3d[i]);
            o[i] = static_cast<float>(v);
        }
        return out;
    }

    py::dict get_stats() const {
        py::dict d;
        d["nx"] = (py::int_)nx;
        d["ny"] = (py::int_)ny;
        d["nz"] = (py::int_)nz;
        d["nt"] = (py::int_)nt;
        d["events"] = (py::int_)total_events;
        d["current_t"] = (py::int_)cur_t;
        return d;
    }
};

PYBIND11_MODULE(holographic_event_log, m) {
    m.doc() = "Holographic 4D Event Log (PoC)";
    py::class_<HolographicEventLog>(m, "HolographicEventLog")
        .def(py::init<size_t,size_t,size_t,size_t>(),
             py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("nt"))
        .def("append_event", &HolographicEventLog::append_event,
             py::arg("payload"), "Append a new event encoded as a wave packet")
        .def("reconstruct_intensity_3d", &HolographicEventLog::reconstruct_intensity_3d,
             py::arg("upto_t"), "Sum over time and return |FFTN(psi)|^2 as [nz,ny,nx] float32 array")
        .def("get_stats", &HolographicEventLog::get_stats, "Get log stats");
}

