#pragma once

#ifdef USE_HIP_BACKEND
#include <vector>
#include <string>
#include <cstdint>

namespace holo {

class HipBackend {
public:
    HipBackend();
    ~HipBackend();
    bool available() const noexcept;

    std::vector<std::vector<float>> batch_encode_fft_ultra(const float* ptr,
                                                           uint32_t batch,
                                                           uint32_t data_len,
                                                           uint32_t pattern_dim);

private:
    void* d_input_{nullptr};
    void* d_output_{nullptr};
    void* h_pinned_{nullptr};
};

} // namespace holo

#else
namespace holo { class HipBackend { public: bool available() const noexcept { return false; } }; }
#endif

