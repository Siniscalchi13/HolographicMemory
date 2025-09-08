// Native C++ GPU benchmarks (Metal)
#ifdef __APPLE__
#include "metal/MetalHoloCore.hpp"
#include <chrono>
#include <iostream>
#include <random>

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    holo::MetalHoloCore core;
    if (!core.available()) {
        cerr << "GPU not available" << endl;
        return 1;
    }
    int dim = 1024;
    size_t batch = 50000;
    int repeats = 5;
    for (int i=1;i<argc;i++){
        string a = argv[i];
        auto eat = [&](const string& flag){ return i+1<argc && a==flag ? string(argv[++i]) : string(); };
        string v;
        if ((v=eat("--dimension")).size()) dim = stoi(v);
        else if ((v=eat("--batch-size")).size()) batch = (size_t)stoll(v);
        else if ((v=eat("--repeats")).size()) repeats = stoi(v);
    }
    vector<vector<float>> data(batch, vector<float>(64));
    for (size_t i=0;i<batch;i++){
        for (int j=0;j<64;j++) data[i][j] = float(j);
    }
    // warmup
    core.batch_encode_fft(data, dim);
    double tot_ms = 0.0;
    for (int r=0;r<repeats;r++){
        auto t0 = high_resolution_clock::now();
        auto out = core.batch_encode_fft(data, dim);
        tot_ms += duration_cast<duration<double, milli>>(high_resolution_clock::now() - t0).count();
    }
    double avg_ms = tot_ms / (double)repeats;
    double ops = (double)batch / (avg_ms/1000.0);
    auto m = core.metrics();
    cout << "op=batch_store_fft" << ", batch=" << batch << ", dim=" << dim << ", p50_ops/s=" << (uint64_t)ops << ", bw_gb/s=" << m.memory_bandwidth_gb_s << "\n";
    return 0;
}
#endif
