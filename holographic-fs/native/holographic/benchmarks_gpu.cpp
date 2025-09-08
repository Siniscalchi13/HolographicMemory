// Native C++ GPU benchmarks (Metal)
#ifdef __APPLE__
#include "metal/MetalHoloCore.hpp"
#include <chrono>
#include <iostream>
#include <random>

using namespace std;
using namespace std::chrono;

int main() {
    holo::MetalHoloCore core;
    if (!core.available()) {
        cerr << "GPU not available" << endl;
        return 1;
    }
    const int dim = 1024;
    const size_t batch = 50000;
    vector<vector<float>> data(batch, vector<float>(64));
    for (size_t i=0;i<batch;i++){
        for (int j=0;j<64;j++) data[i][j] = float(j);
    }
    // warmup
    core.batch_encode(data, dim);
    auto t0 = high_resolution_clock::now();
    auto out = core.batch_encode(data, dim);
    auto ms = duration_cast<duration<double, milli>>(high_resolution_clock::now() - t0).count();
    double ops = (double)batch / (ms/1000.0);
    auto m = core.metrics();
    cout << "batch=" << batch << ", dim=" << dim << ", ops/s=" << (uint64_t)ops << ", bw_gb/s=" << m.memory_bandwidth_gb_s << "\n";
    return 0;
}
#endif

