/*
 * HRR Binding PoC: bind/unbind via circular convolution using FFTW
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <fftw3.h>
#include <vector>
#include <complex>
#include <cstring>

namespace py = pybind11;

static py::array_t<std::complex<double>> _fft(py::array_t<std::complex<double>> a, int sign) {
    auto b = a.request();
    if (b.ndim != 1) throw std::invalid_argument("array must be 1D complex128");
    size_t n = (size_t)b.shape[0];
    py::array_t<std::complex<double>> out(n);
    auto* inptr = static_cast<std::complex<double>*>(b.ptr);
    auto* outptr = static_cast<std::complex<double>*>(out.request().ptr);
    std::vector<std::complex<double>> tmp(n);
    std::memcpy(tmp.data(), inptr, n * sizeof(std::complex<double>));
    auto plan = fftw_plan_dft_1d((int)n,
        reinterpret_cast<fftw_complex*>(tmp.data()),
        reinterpret_cast<fftw_complex*>(tmp.data()),
        sign, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    // Unitary normalization (norm="ortho"): both forward and inverse multiply by 1/sqrt(n)
    double scale = 1.0 / std::sqrt((double)n);
    for (size_t i = 0; i < n; ++i) outptr[i] = tmp[i] * scale;
    return out;
}

// Compute phase-only spectrum of key: P[k] = K[k]/|K[k]| (unit modulus)
static py::array_t<std::complex<double>> _phase_spectrum(py::array_t<std::complex<double>> key) {
    auto K = _fft(key, FFTW_FORWARD);
    auto bK = K.request();
    size_t n = (size_t)bK.shape[0];
    auto* Kp = static_cast<std::complex<double>*>(bK.ptr);
    py::array_t<std::complex<double>> P(n);
    auto* Pp = static_cast<std::complex<double>*>(P.request().ptr);
    for (size_t i = 0; i < n; ++i) {
        double mag = std::abs(Kp[i]);
        if (mag > 1e-15) {
            Pp[i] = Kp[i] / mag;  // unit magnitude, preserve phase
        } else {
            Pp[i] = std::complex<double>(1.0, 0.0);
        }
    }
    return P;
}

py::array_t<std::complex<double>> bind(py::array_t<std::complex<double>> key,
                                       py::array_t<std::complex<double>> value) {
    auto bk = key.request();
    auto bv = value.request();
    if (bk.ndim != 1 || bv.ndim != 1 || bk.shape[0] != bv.shape[0])
        throw std::invalid_argument("key and value must be same-length 1D complex128 arrays");
    size_t n = (size_t)bk.shape[0];
    // Phase-only spectral key improves invertibility and stability
    auto P = _phase_spectrum(key);
    auto V = _fft(value, FFTW_FORWARD);
    auto bP = P.request();
    auto bV = V.request();
    auto* Pp = static_cast<std::complex<double>*>(bP.ptr);
    auto* Vp = static_cast<std::complex<double>*>(bV.ptr);
    // Multiply in frequency: Bound = IFFT(P * FFT(value))
    py::array_t<std::complex<double>> F(n);
    auto* Fp = static_cast<std::complex<double>*>(F.request().ptr);
    for (size_t i = 0; i < n; ++i) Fp[i] = Pp[i] * Vp[i];
    // IFFT back (unitary)
    return _fft(F, FFTW_BACKWARD);
}

py::array_t<std::complex<double>> unbind(py::array_t<std::complex<double>> bound,
                                         py::array_t<std::complex<double>> key) {
    auto bb = bound.request();
    auto bk = key.request();
    if (bb.ndim != 1 || bk.ndim != 1 || bb.shape[0] != bk.shape[0])
        throw std::invalid_argument("bound and key must be same-length 1D complex128 arrays");
    size_t n = (size_t)bb.shape[0];
    // FFT(bound), use conjugate of phase-only spectral key for unbinding
    auto B = _fft(bound, FFTW_FORWARD);
    auto P = _phase_spectrum(key);
    auto bB = B.request();
    auto bP = P.request();
    auto* Bp = static_cast<std::complex<double>*>(bB.ptr);
    auto* Pp = static_cast<std::complex<double>*>(bP.ptr);
    // Unbind: IFFT(FFT(bound) * conj(P)) ≈ value
    py::array_t<std::complex<double>> F(n);
    auto* Fp = static_cast<std::complex<double>*>(F.request().ptr);
    for (size_t i = 0; i < n; ++i) Fp[i] = Bp[i] * std::conj(Pp[i]);
    return _fft(F, FFTW_BACKWARD);
}

PYBIND11_MODULE(hrr_binding, m) {
    m.doc() = "HRR binding/unbinding via circular convolution (PoC)";
    m.def("bind", &bind, py::arg("key"), py::arg("value"), "Bind key and value via circular convolution");
    m.def("unbind", &unbind, py::arg("bound"), py::arg("key"), "Unbind with key inverse (conjugate in frequency)");
}
