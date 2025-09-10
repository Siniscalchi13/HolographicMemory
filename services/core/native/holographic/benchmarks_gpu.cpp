// Cross-platform native C++ GPU benchmark
#include "GPUBackend.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>

using namespace std;
using namespace std::chrono;

struct DetailedMetrics {
    double h2d_time_ms = 0.0;
    double fft_time_ms = 0.0;
    double d2h_time_ms = 0.0;
    double total_device_time_ms = 0.0;
    double total_e2e_time_ms = 0.0;
    uint64_t device_ops_per_sec = 0;
    uint64_t e2e_ops_per_sec = 0;
    double memory_bandwidth_gb_s = 0.0;
    double device_utilization = 0.0;
    double copy_overhead_ratio = 0.0;
    double compute_efficiency = 0.0;
};

int main(int argc, char* argv[]) {
    uint32_t dimension = 1024;
    uint32_t batch_size = 10000;
    int repeats = 5;
    std::string platform = "auto";
    std::string out_csv;
    if (argc > 1) dimension = (uint32_t)std::stoul(argv[1]);
    if (argc > 2) batch_size = (uint32_t)std::stoul(argv[2]);
    if (argc > 3) repeats = std::stoi(argv[3]);
    if (argc > 4) platform = argv[4];
    if (argc > 5) out_csv = argv[5];

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
    struct Row { double e2e_ms, dev_ms, h2d, fft, d2h; uint64_t e2e_ops, dev_ops; double bw; };
    std::vector<Row> rows;
    for (int r=0;r<repeats;r++){
        auto t0 = high_resolution_clock::now();
        auto out = be->batch_encode_fft_zero_copy(flat.data(), batch_size, 64, dimension);
        auto ms = duration_cast<duration<double, milli>>(high_resolution_clock::now()-t0).count();
        host_ms.push_back(ms);
        auto gm = be->get_metrics();
        dev_ops.push_back(gm.operations_per_second);
        rows.push_back({ms, gm.device_ms, gm.h2d_time_ms, gm.fft_time_ms, gm.d2h_time_ms,
                        (uint64_t)(batch_size/(ms/1000.0)), gm.operations_per_second, gm.memory_bandwidth_gb_s});
        std::cout << "Run " << (r+1) << ": E2E ops/s=" << rows.back().e2e_ops
                  << ", device ops/s=" << rows.back().dev_ops
                  << ", H2D=" << rows.back().h2d << " ms, FFT=" << rows.back().fft << " ms, D2H=" << rows.back().d2h << " ms"
                  << ", BW=" << rows.back().bw << " GB/s" << std::endl;
    }
    double avg_ms = 0.0; for (double v : host_ms) avg_ms += v; avg_ms /= std::max(1,(int)host_ms.size());
    uint64_t avg_dev = 0; for (auto v : dev_ops) avg_dev += v; avg_dev /= std::max(1,(int)dev_ops.size());
    std::cout << "\nAverage E2E ops/s=" << (uint64_t)(batch_size/(avg_ms/1000.0))
              << ", avg device ops/s=" << avg_dev << std::endl;

    if (!out_csv.empty()) {
        std::ofstream csv(out_csv);
        csv << "run,end_to_end_time_ms,device_time_ms,h2d_time_ms,fft_time_ms,d2h_time_ms,end_to_end_ops_per_sec,device_ops_per_sec,memory_bandwidth_gb_s\n";
        for (size_t i=0;i<rows.size();++i) {
            csv << (i+1) << "," << rows[i].e2e_ms << "," << rows[i].dev_ms << "," << rows[i].h2d << "," << rows[i].fft
                << "," << rows[i].d2h << "," << rows[i].e2e_ops << "," << rows[i].dev_ops << "," << rows[i].bw << "\n";
        }
        csv.close();
        std::cout << "Wrote CSV: " << out_csv << std::endl;
    }
    return 0;
}
