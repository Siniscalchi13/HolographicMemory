// Cross-platform native C++ GPU benchmark
#include "GPUBackend.hpp"
#include <chrono>
#include <iostream>
#include <vector>

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    uint32_t dimension = 1024;
    uint32_t batch_size = 10000;
    int repeats = 5;
    std::string platform = "auto";
    if (argc > 1) dimension = (uint32_t)std::stoul(argv[1]);
    if (argc > 2) batch_size = (uint32_t)std::stoul(argv[2]);
    if (argc > 3) repeats = std::stoi(argv[3]);
    if (argc > 4) platform = argv[4];

    auto plats = holo::IGPUBackend::get_available_platforms();
    if (plats.empty()) { std::cerr << "No GPU platform available" << std::endl; return 1; }
    holo::GPUPlatform pf = plats[0];
    if (platform != "auto") {
        if (platform == "metal") pf = holo::GPUPlatform::METAL;
        else if (platform == "cuda") pf = holo::GPUPlatform::CUDA;
        else if (platform == "rocm") pf = holo::GPUPlatform::ROCM;
    }
    auto be = holo::IGPUBackend::create_backend(pf);
    if (!be) { std::cerr << "Failed to create backend" << std::endl; return 2; }
    holo::GPUConfig cfg; cfg.platform = pf; cfg.memory_pool_size = (size_t)batch_size * dimension * sizeof(float) * 4;
    if (!be->initialize(cfg)) { std::cerr << "Failed to initialize backend" << std::endl; return 3; }

    std::vector<float> flat(batch_size * 64);
    for (size_t i=0;i<flat.size();++i) flat[i] = float(i % 64);
    (void)be->batch_encode_fft_zero_copy(flat.data(), batch_size, 64, dimension);

    std::vector<double> host_ms;
    std::vector<uint64_t> dev_ops;
    for (int r=0;r<repeats;r++){
        auto t0 = high_resolution_clock::now();
        auto out = be->batch_encode_fft_zero_copy(flat.data(), batch_size, 64, dimension);
        auto ms = duration_cast<duration<double, milli>>(high_resolution_clock::now()-t0).count();
        host_ms.push_back(ms);
        dev_ops.push_back(be->get_metrics().operations_per_second);
        std::cout << "Run " << (r+1) << ": E2E ops/s=" << (uint64_t)(batch_size/(ms/1000.0))
                  << ", device ops/s=" << be->get_metrics().operations_per_second
                  << ", BW=" << be->get_metrics().memory_bandwidth_gb_s << " GB/s" << std::endl;
    }
    double avg_ms = 0.0; for (double v : host_ms) avg_ms += v; avg_ms /= std::max(1,(int)host_ms.size());
    uint64_t avg_dev = 0; for (auto v : dev_ops) avg_dev += v; avg_dev /= std::max(1,(int)dev_ops.size());
    std::cout << "\nAverage E2E ops/s=" << (uint64_t)(batch_size/(avg_ms/1000.0))
              << ", avg device ops/s=" << avg_dev << std::endl;
    return 0;
}
