#ifdef __APPLE__
#include "MetalHolographicBackend.hpp"
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <chrono>
#include <cstring>

using namespace MTL;

namespace holo {

MetalHolographicBackend::MetalHolographicBackend() {
    device_ = MTL::CreateSystemDefaultDevice();
    if (!device_) return;
    queue_ = device_->newCommandQueue();
}

MetalHolographicBackend::~MetalHolographicBackend() {
    if (pso_fft_) pso_fft_->release();
    if (pso_interf_) pso_interf_->release();
    if (pso_batch_) pso_batch_->release();
    if (pso_sim_) pso_sim_->release();
    if (library_) library_->release();
    if (queue_) queue_->release();
    if (device_) device_->release();
}

bool MetalHolographicBackend::initialize() {
    if (!device_ || !queue_) return false;
    if (!library_) load_shader_library();
    return library_ && pso_batch_;
}

void MetalHolographicBackend::load_shader_library() {
    NS::Error* error = nullptr;
    auto path = NS::String::string("holographic-fs/native/holographic/metal/holographic_kernels.metal", NS::UTF8StringEncoding);
    auto src = NS::String::stringWithContentsOfFile(path, NS::UTF8StringEncoding, nullptr);
    if (!src) return;
    library_ = device_->newLibrary(src, nullptr, &error);
    if (!library_) return;
    pso_fft_ = create_pipeline("holographic_fft");
    pso_interf_ = create_pipeline("wave_interference");
    pso_batch_ = create_pipeline("batch_holographic_encode");
    pso_sim_ = create_pipeline("holographic_similarity");
}

ComputePipelineState* MetalHolographicBackend::create_pipeline(const char* fn) {
    NS::Error* error = nullptr;
    auto fname = NS::String::string(fn, NS::UTF8StringEncoding);
    auto f = library_->newFunction(fname);
    if (!f) return nullptr;
    auto pso = device_->newComputePipelineState(f, &error);
    f->release();
    return pso;
}

void MetalHolographicBackend::dispatch_kernel(ComputePipelineState* pso,
                         const std::vector<Buffer*>& buffers,
                         uint32_t grid_size) {
    auto cmd = queue_->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    for (size_t i = 0; i < buffers.size(); ++i) enc->setBuffer(buffers[i], 0, (NS::UInteger)i);
    NS::UInteger tg = 256;
    NS::UInteger grid = ((grid_size + tg - 1)/tg)*tg;
    enc->dispatchThreads(MTL::Size(grid,1,1), MTL::Size(tg,1,1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();
    cmd->release();
}

std::vector<float> MetalHolographicBackend::batch_encode(const std::vector<float>& flat_input,
                                    uint32_t batch_size,
                                    uint32_t data_length,
                                    uint32_t pattern_dim) {
    if (!initialize()) return {};
    std::vector<float> flat_out(batch_size * pattern_dim, 0.0f);
    auto bin = device_->newBuffer(flat_input.data(), flat_input.size()*sizeof(float), ResourceStorageModeManaged);
    auto bout = device_->newBuffer(flat_out.data(), flat_out.size()*sizeof(float), ResourceStorageModeManaged);
    auto bbs = device_->newBuffer(&batch_size, sizeof(uint32_t), ResourceStorageModeManaged);
    auto bdl = device_->newBuffer(&data_length, sizeof(uint32_t), ResourceStorageModeManaged);
    auto bpd = device_->newBuffer(&pattern_dim, sizeof(uint32_t), ResourceStorageModeManaged);

    auto t0 = std::chrono::high_resolution_clock::now();
    dispatch_kernel(pso_batch_, {bin,bout,bbs,bdl,bpd}, batch_size);
    std::memcpy(flat_out.data(), bout->contents(), flat_out.size()*sizeof(float));
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    metrics_.batch_encode_time_ms = ms;
    metrics_.operations_per_second = (ms>0.0)? (uint64_t)(batch_size * 1000.0 / ms) : 0;
    double bytes = (double)(flat_input.size() + flat_out.size()) * sizeof(float);
    metrics_.memory_bandwidth_gb_s = (ms>0.0)? (bytes / (ms/1000.0)) / (1024.0*1024.0*1024.0) : 0.0;

    bin->release(); bout->release(); bbs->release(); bdl->release(); bpd->release();
    return flat_out;
}

std::vector<float> MetalHolographicBackend::similarity_search(const std::vector<float>& query,
                                         const std::vector<float>& stored,
                                         uint32_t pattern_count,
                                         uint32_t dim) {
    if (!initialize()) return {};
    std::vector<float> sims(pattern_count, 0.0f);
    auto bq = device_->newBuffer(query.data(), query.size()*sizeof(float), ResourceStorageModeManaged);
    auto bs = device_->newBuffer(stored.data(), stored.size()*sizeof(float), ResourceStorageModeManaged);
    auto bo = device_->newBuffer(sims.data(), sims.size()*sizeof(float), ResourceStorageModeManaged);
    auto bpc = device_->newBuffer(&pattern_count, sizeof(uint32_t), ResourceStorageModeManaged);
    auto bd = device_->newBuffer(&dim, sizeof(uint32_t), ResourceStorageModeManaged);

    dispatch_kernel(pso_sim_, {bq, bs, bo, bpc, bd}, pattern_count);
    std::memcpy(sims.data(), bo->contents(), sims.size()*sizeof(float));

    bq->release(); bs->release(); bo->release(); bpc->release(); bd->release();
    return sims;
}

} // namespace holo

#endif
