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

std::tuple<float,float,double,float> HipBackend::analyze_metrics(const float* v1, const float* v2, uint32_t dim) {
    if (!v1 || !v2 || dim == 0) return {0.0f, 0.0f, 0.0, 0.0f};
    double dot=0.0, n1=0.0, n2=0.0;
    for (uint32_t i=0;i<dim;++i) { double x=v1[i], y=v2[i]; dot+=x*y; n1+=x*x; n2+=y*y; }
    float vis=0.0f, coh=0.0f, ortho = (float)std::fabs(dot);
    if (n1>0.0 && n2>0.0){ double rn1=std::sqrt(n1), rn2=std::sqrt(n2); double num=std::fabs(dot); coh=(float)(num/(rn1*rn2)); vis=(float)((num*num)/((rn1*rn1)*(rn2*rn2))); }
    auto corr = [&](uint32_t o1,uint32_t o2){ double s=0.0; for(uint32_t i=0;i<dim;++i){ double x=v1[(i+o1)%dim], y=v2[(i+o2)%dim]; s+=x*y;} return s/double(dim); };
    double S = corr(0,0)+corr(0,1)+corr(2,0)-corr(2,1);
    return {vis, coh, S-2.0, ortho};
}

} // namespace holo

#endif
