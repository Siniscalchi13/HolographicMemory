#ifdef __APPLE__
#include "MetalBackend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>

namespace {
static inline NSString* str(const char* s) { return [NSString stringWithUTF8String:s]; }
}

namespace holo {

MetalBackend::MetalBackend() {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) return;
    
    queue_ = [device_ newCommandQueue];
    if (!queue_) return;
    
    load_shaders();
}

MetalBackend::~MetalBackend() {
    // ARC handles cleanup
}

void MetalBackend::load_shaders() {
    NSError* err = nullptr;
    
    // Load embedded shader source
    NSString* srcPath = str("holographic-fs/native/holographic/metal/holographic_memory.metal");
    NSString* file = [NSString stringWithContentsOfFile:srcPath encoding:NSUTF8StringEncoding error:&err];
    if (!file) return;
    
    library_ = [device_ newLibraryWithSource:file options:nil error:&err];
    if (!library_) return;
    
    // Create pipeline states
    pso_vector_add_ = create_pipeline("enhanced_vector_add");
    pso_batch_store_ = create_pipeline("batch_holographic_store");
}

id<MTLComputePipelineState> MetalBackend::create_pipeline(const std::string& function_name) {
    NSError* err = nullptr;
    id<MTLFunction> fn = [library_ newFunctionWithName:str(function_name.c_str())];
    if (!fn) return nil;
    
    id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:fn error:&err];
    return pso;
}

float MetalBackend::vector_add(const std::vector<float>& a, const std::vector<float>& b) {
    if (!available() || a.size() != b.size() || a.empty()) return 0.0f;
    uint32_t n = static_cast<uint32_t>(a.size());

    id<MTLBuffer> ba = [device_ newBufferWithBytes:a.data() length:n * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bb = [device_ newBufferWithBytes:b.data() length:n * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bc = [device_ newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bn = [device_ newBufferWithBytes:&n length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];

    id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    
    [enc setComputePipelineState:pso_vector_add_];
    [enc setBuffer:ba offset:0 atIndex:0];
    [enc setBuffer:bb offset:0 atIndex:1];
    [enc setBuffer:bc offset:0 atIndex:2];
    [enc setBuffer:bn offset:0 atIndex:3];

    NSUInteger tgX = 256;
    NSUInteger gridX = ((n + tgX - 1) / tgX) * tgX;
    [enc dispatchThreads:MTLSizeMake(gridX, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgX, 1, 1)];
    
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    float sum = 0.0f;
    float* out = (float*)[bc contents];
    for (uint32_t i = 0; i < n; ++i) {
        sum += out[i];
    }

    // ARC cleans up
    return sum;
}

std::vector<std::vector<float>> MetalBackend::batch_holographic_store(
    const std::vector<std::vector<float>>& batch_data, 
    uint32_t pattern_dimension) {
    
    if (!available() || batch_data.empty()) return {};
    
    uint32_t batch_size = static_cast<uint32_t>(batch_data.size());
    uint32_t data_length = static_cast<uint32_t>(batch_data[0].size());
    
    // Prepare flattened input data
    std::vector<float> flat_input;
    flat_input.reserve(batch_size * data_length);
    for (const auto& item : batch_data) {
        flat_input.insert(flat_input.end(), item.begin(), item.end());
    }
    
    // Prepare output buffer
    std::vector<float> flat_output(batch_size * pattern_dimension, 0.0f);
    
    auto time_start = std::chrono::high_resolution_clock::now();
    
    id<MTLBuffer> b_input = [device_ newBufferWithBytes:flat_input.data() 
                                                length:flat_input.size() * sizeof(float) 
                                               options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_output = [device_ newBufferWithBytes:flat_output.data()
                                                 length:flat_output.size() * sizeof(float)
                                                options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_batch_size = [device_ newBufferWithBytes:&batch_size
                                                     length:sizeof(uint32_t)
                                                    options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_data_length = [device_ newBufferWithBytes:&data_length
                                                      length:sizeof(uint32_t)
                                                     options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_pattern_dim = [device_ newBufferWithBytes:&pattern_dimension
                                                      length:sizeof(uint32_t)
                                                     options:MTLResourceStorageModeManaged];
    
    id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    
    [enc setComputePipelineState:pso_batch_store_];
    [enc setBuffer:b_input offset:0 atIndex:0];
    [enc setBuffer:b_output offset:0 atIndex:1];
    [enc setBuffer:b_batch_size offset:0 atIndex:2];
    [enc setBuffer:b_data_length offset:0 atIndex:3];
    [enc setBuffer:b_pattern_dim offset:0 atIndex:4];
    
    // 2D dispatch: x = ceil(pattern_dim/4), y = batch_size (kernel computes 4 outputs/thread)
    NSUInteger mtt = [pso_batch_store_ maxTotalThreadsPerThreadgroup];
    // Favor 256 threads per TG for wide reuse
    NSUInteger tgX = MIN(mtt, (NSUInteger)256);
    
    NSUInteger gridX = ((NSUInteger)pattern_dimension + 3u) / 4u;
    MTLSize grid = MTLSizeMake(gridX, (NSUInteger)batch_size, 1);
    MTLSize tpg  = MTLSizeMake(tgX, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tpg];
    
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    
    // Copy results back
    memcpy(flat_output.data(), [b_output contents], flat_output.size() * sizeof(float));
    
    auto time_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
    metrics_.batch_store_time_ms = duration.count();
    metrics_.operations_per_second = static_cast<uint64_t>(batch_size * 1000.0 / duration.count());
    
    // Convert back to 2D vector
    std::vector<std::vector<float>> result;
    result.reserve(batch_size);
    for (uint32_t i = 0; i < batch_size; ++i) {
        std::vector<float> pattern(
            flat_output.begin() + i * pattern_dimension,
            flat_output.begin() + (i + 1) * pattern_dimension
        );
        result.push_back(std::move(pattern));
    }
    
    // ARC cleans up
    
    return result;
}

} // namespace holo

#endif
