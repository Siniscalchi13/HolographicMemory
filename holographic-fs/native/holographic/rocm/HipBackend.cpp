#include "HipBackend.hpp"

#ifdef USE_HIP_BACKEND
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <cstring>

namespace holo {

HipBackend::HipBackend() {}
HipBackend::~HipBackend() {
    if (d_input_) hipFree(d_input_);
    if (d_output_) hipFree(d_output_);
    if (h_pinned_) hipHostFree(h_pinned_);
}

bool HipBackend::available() const noexcept {
    int n=0; if (hipGetDeviceCount(&n) != hipSuccess) return false; return n>0;
}

std::vector<std::vector<float>> HipBackend::batch_encode_fft_ultra(const float* ptr,
                                                                   uint32_t batch,
                                                                   uint32_t data_len,
                                                                   uint32_t pattern_dim) {
    // TODO: implement hipFFT path
    return {};
}

} // namespace holo

#endif

