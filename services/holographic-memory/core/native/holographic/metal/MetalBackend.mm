#ifdef __APPLE__
#include "MetalBackend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
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
    // Initialize buffer pools and lock
    input_pool_ = [NSMutableArray array];
    output_pool_ = [NSMutableArray array];
    pool_lock_ = [[NSLock alloc] init];

    load_shaders();
}

MetalBackend::~MetalBackend() {
    // ARC handles cleanup
}

void MetalBackend::load_shaders() {
    NSError* err = nullptr;
    
    // Get the directory containing the current executable/binary
    NSString* binaryPath = [[NSBundle mainBundle] executablePath];
    if (!binaryPath) {
        // Fallback: get current working directory
        binaryPath = [[NSFileManager defaultManager] currentDirectoryPath];
    }
    NSString* binaryDir = [binaryPath stringByDeletingLastPathComponent];
    
    // Try multiple shader locations in order of preference
    NSArray* shaderPaths = @[
        // 1. Relative to binary (for installed apps)
        [binaryDir stringByAppendingPathComponent:@"holographic_memory.metal"],
        // 2. Build directory (for development)
        [binaryDir stringByAppendingPathComponent:@"services/holographic-memory/core/native/holographic/build/metal/holographic_memory.metal"],
        // 3. Source directory (for development)
        [binaryDir stringByAppendingPathComponent:@"services/holographic-memory/core/native/holographic/metal/holographic_memory.metal"],
        // 4. Current working directory fallbacks
        @"build/metal/holographic_memory.metal",
        @"metal/holographic_memory.metal",
        @"services/holographic-memory/core/native/holographic/build/metal/holographic_memory.metal",
        @"services/holographic-memory/core/native/holographic/metal/holographic_memory.metal"
    ];
    
    NSString* file = nil;
    NSString* usedPath = nil;
    
    for (NSString* path in shaderPaths) {
        file = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
        if (file) {
            usedPath = path;
            break;
        }
    }
    
    if (!file) {
        NSLog(@"Failed to load Metal shader from any location. Tried paths: %@", shaderPaths);
        NSLog(@"Last error: %@", err.localizedDescription);
        return;
    }
    
    NSLog(@"Loading Metal shader from: %@", usedPath);

    library_ = [device_ newLibraryWithSource:file options:nil error:&err];
    if (!library_) {
        NSLog(@"[MetalBackend] Failed to create Metal library: %@", err.localizedDescription);
        return;
    }
    
    // Create pipeline states
    pso_vector_add_ = create_pipeline("enhanced_vector_add");
    pso_batch_store_ = create_pipeline("batch_holographic_store");
    pso_similarity_ = create_pipeline("holographic_similarity_search");
    pso_fft_ = create_pipeline("holographic_fft_transform");
    pso_ifft_ = create_pipeline("holographic_ifft_transform");
    pso_batch_store_fft_ = create_pipeline("batch_holographic_encode_fft");
    pso_dot_norm_ = create_pipeline("dot_norm_kernel");
    pso_corr_off_ = create_pipeline("correlation_offset_kernel");
    pso_apply_code_ = create_pipeline("apply_codebook");
    pso_apply_code_conj_ = create_pipeline("apply_codebook_conj");
    pso_accumulate_add_ = create_pipeline("accumulate_add_time");
    
    // GPU Compression Pipeline - Kernel 1: Quantization
    pso_quantize_ = create_pipeline("gpu_holographic_quantize");
    pso_quantize_validation_ = create_pipeline("gpu_holographic_quantize_with_validation");
    pso_quantize_stats_ = create_pipeline("gpu_quantization_statistics");
    
    // GPU Compression Pipeline - Kernel 2: Bitplane Extraction
    // Removed: bitplane extraction pipelines - no longer needed with holographic wave reconstruction
    
    // GPU Compression Pipeline - Kernel 3: Sparse Encoding
    sparse_encoding_pipeline_ = create_pipeline("gpu_sparse_encoding");
    
    // GPU Compression Pipeline - Kernel 4: Entropy Coding
    entropy_coding_pipeline_ = create_pipeline("gpu_entropy_coding");
    
    // GPU Compression Pipeline - Kernel 5: Entropy Decoding
    entropy_decoding_pipeline_ = create_pipeline("gpu_entropy_decoding");
    
    // GPU Compression Pipeline - Kernel 6: Sparse Decoding
    sparse_decoding_pipeline_ = create_pipeline("gpu_sparse_decoding");
    
    initialize_mps_fft();
}

id<MTLComputePipelineState> MetalBackend::create_pipeline(const std::string& function_name) {
    NSError* err = nullptr;
    id<MTLFunction> fn = [library_ newFunctionWithName:str(function_name.c_str())];
    if (!fn) {
        NSLog(@"❌ Failed to find Metal function: %s", function_name.c_str());
        return nil;
    }
    
    id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:fn error:&err];
    if (!pso && err) {
        NSLog(@"❌ Failed to create pipeline for %s: %@", function_name.c_str(), [err localizedDescription]);
        return nil;
    }
    
    NSLog(@"✅ Created pipeline for %s", function_name.c_str());
    return pso;
}

// -------- Buffer pool helpers --------
static inline id<MTLBuffer> hb_pop_buffer(NSMutableArray* pool, NSLock* lock, id<MTLDevice> device, NSUInteger size) {
    [lock lock];
    id<MTLBuffer> picked = nil;
    for (NSUInteger i = 0; i < [pool count]; ++i) {
        id<MTLBuffer> buf = (id<MTLBuffer>)[pool objectAtIndex:i];
        if ([buf length] >= size) {
            picked = buf;
            [pool removeObjectAtIndex:i];
            break;
        }
    }
    [lock unlock];
    if (!picked) picked = [device newBufferWithLength:size options:MTLResourceStorageModeManaged];
    return picked;
}

static inline void hb_push_buffer(NSMutableArray* pool, NSLock* lock, id<MTLBuffer> buf) {
    if (!buf) return;
    [lock lock];
    [pool addObject:buf];
    [lock unlock];
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

void MetalBackend::ifft_transform(const std::vector<float>& real_in,
                                  const std::vector<float>& imag_in,
                                  std::vector<float>& time_out) {
    if (!available() || real_in.size() != imag_in.size()) { time_out.clear(); return; }
    uint32_t n = static_cast<uint32_t>(real_in.size());
    time_out.assign(n, 0.0f);

    id<MTLBuffer> bre = [device_ newBufferWithBytes:real_in.data() length:n * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bim = [device_ newBufferWithBytes:imag_in.data() length:n * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bout = [device_ newBufferWithBytes:time_out.data() length:n * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> blen = [device_ newBufferWithBytes:&n length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];

    id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso_ifft_];
    [enc setBuffer:bre offset:0 atIndex:0];
    [enc setBuffer:bim offset:0 atIndex:1];
    [enc setBuffer:bout offset:0 atIndex:2];
    [enc setBuffer:blen offset:0 atIndex:3];
    NSUInteger tgX = 256;
    NSUInteger gridX = ((n + tgX - 1) / tgX) * tgX;
    [enc dispatchThreads:MTLSizeMake(gridX, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgX, 1, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    std::memcpy(time_out.data(), [bout contents], n * sizeof(float));
}

void MetalBackend::apply_codebook(const std::vector<float>& in_real,
                                  const std::vector<float>& in_imag,
                                  std::vector<float>& out_real,
                                  std::vector<float>& out_imag,
                                  uint32_t seed) {
    if (!available() || in_real.size() != in_imag.size()) { out_real.clear(); out_imag.clear(); return; }
    uint32_t n = static_cast<uint32_t>(in_real.size());
    out_real.assign(n, 0.0f);
    out_imag.assign(n, 0.0f);

    id<MTLBuffer> binr = [device_ newBufferWithBytes:in_real.data() length:n * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> binm = [device_ newBufferWithBytes:in_imag.data() length:n * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> boutr = [device_ newBufferWithBytes:out_real.data() length:n * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> boutm = [device_ newBufferWithBytes:out_imag.data() length:n * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bdim = [device_ newBufferWithBytes:&n length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bseed = [device_ newBufferWithBytes:&seed length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];

    id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso_apply_code_];
    [enc setBuffer:binr offset:0 atIndex:0];
    [enc setBuffer:binm offset:0 atIndex:1];
    [enc setBuffer:boutr offset:0 atIndex:2];
    [enc setBuffer:boutm offset:0 atIndex:3];
    [enc setBuffer:bdim offset:0 atIndex:4];
    [enc setBuffer:bseed offset:0 atIndex:5];
    NSUInteger tgX = 256;
    NSUInteger gridX = ((n + tgX - 1) / tgX) * tgX;
    [enc dispatchThreads:MTLSizeMake(gridX, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgX, 1, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    std::memcpy(out_real.data(), [boutr contents], n * sizeof(float));
    std::memcpy(out_imag.data(), [boutm contents], n * sizeof(float));
}

void MetalBackend::apply_codebook_conj(const std::vector<float>& in_real,
                                       const std::vector<float>& in_imag,
                                       std::vector<float>& out_real,
                                       std::vector<float>& out_imag,
                                       uint32_t seed) {
    if (!available() || in_real.size() != in_imag.size()) { out_real.clear(); out_imag.clear(); return; }
    uint32_t n = static_cast<uint32_t>(in_real.size());
    out_real.assign(n, 0.0f);
    out_imag.assign(n, 0.0f);

    id<MTLBuffer> binr = [device_ newBufferWithBytes:in_real.data() length:n * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> binm = [device_ newBufferWithBytes:in_imag.data() length:n * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> boutr = [device_ newBufferWithBytes:out_real.data() length:n * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> boutm = [device_ newBufferWithBytes:out_imag.data() length:n * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bdim = [device_ newBufferWithBytes:&n length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bseed = [device_ newBufferWithBytes:&seed length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];

    id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso_apply_code_conj_];
    [enc setBuffer:binr offset:0 atIndex:0];
    [enc setBuffer:binm offset:0 atIndex:1];
    [enc setBuffer:boutr offset:0 atIndex:2];
    [enc setBuffer:boutm offset:0 atIndex:3];
    [enc setBuffer:bdim offset:0 atIndex:4];
    [enc setBuffer:bseed offset:0 atIndex:5];
    NSUInteger tgX = 256;
    NSUInteger gridX = ((n + tgX - 1) / tgX) * tgX;
    [enc dispatchThreads:MTLSizeMake(gridX, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgX, 1, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    std::memcpy(out_real.data(), [boutr contents], n * sizeof(float));
    std::memcpy(out_imag.data(), [boutm contents], n * sizeof(float));
}

void MetalBackend::accumulate_add_time(std::vector<float>& dst,
                                       const std::vector<float>& add) {
    if (!available() || dst.size() != add.size()) return;
    uint32_t n = static_cast<uint32_t>(dst.size());

    id<MTLBuffer> bdst = [device_ newBufferWithBytes:dst.data() length:n * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> badd = [device_ newBufferWithBytes:add.data() length:n * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> blen = [device_ newBufferWithBytes:&n length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];

    id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso_accumulate_add_];
    [enc setBuffer:bdst offset:0 atIndex:0];
    [enc setBuffer:badd offset:0 atIndex:1];
    [enc setBuffer:blen offset:0 atIndex:2];
    NSUInteger tgX = 256;
    NSUInteger gridX = ((n + tgX - 1) / tgX) * tgX;
    [enc dispatchThreads:MTLSizeMake(gridX, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgX, 1, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    std::memcpy(dst.data(), [bdst contents], n * sizeof(float));
}

std::vector<float> MetalBackend::similarity_search(const std::vector<float>& query,
                                         const std::vector<std::vector<float>>& stored) {
    if (!available() || stored.empty()) return {};
    uint32_t pattern_count = (uint32_t)stored.size();
    uint32_t dim = (uint32_t)stored[0].size();
    // Flatten stored
    std::vector<float> flat;
    flat.reserve((size_t)pattern_count * dim);
    for (auto &v : stored) flat.insert(flat.end(), v.begin(), v.end());

    std::vector<float> sims(pattern_count, 0.0f);
    id<MTLBuffer> bq = [device_ newBufferWithBytes:query.data() length:dim * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bs = [device_ newBufferWithBytes:flat.data() length:flat.size() * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bo = [device_ newBufferWithBytes:sims.data() length:sims.size() * sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bpc = [device_ newBufferWithBytes:&pattern_count length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bd = [device_ newBufferWithBytes:&dim length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];

    id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso_similarity_];
    [enc setBuffer:bq offset:0 atIndex:0];
    [enc setBuffer:bs offset:0 atIndex:1];
    [enc setBuffer:bo offset:0 atIndex:2];
    [enc setBuffer:bpc offset:0 atIndex:3];
    [enc setBuffer:bd offset:0 atIndex:4];
    NSUInteger tgX = 256;
    NSUInteger gridX = ((pattern_count + tgX - 1) / tgX) * tgX;
    [enc dispatchThreads:MTLSizeMake(gridX, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgX, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    memcpy(sims.data(), [bo contents], sims.size() * sizeof(float));
    // ARC cleanup
    return sims;
}

std::tuple<float,float,double,float> MetalBackend::analyze_metrics(const float* v1, const float* v2, uint32_t dim) {
    if (!available() || dim == 0) return {0.0f, 0.0f, 0.0, 0.0f};
    id<MTLBuffer> b1 = [device_ newBufferWithBytes:v1 length:dim*sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b2 = [device_ newBufferWithBytes:v2 length:dim*sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bdot = [device_ newBufferWithLength:sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bn1  = [device_ newBufferWithLength:sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bn2  = [device_ newBufferWithLength:sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> blen = [device_ newBufferWithBytes:&dim length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];

    id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_dot_norm_];
        [enc setBuffer:b1 offset:0 atIndex:0];
        [enc setBuffer:b2 offset:0 atIndex:1];
        [enc setBuffer:bdot offset:0 atIndex:2];
        [enc setBuffer:bn1 offset:0 atIndex:3];
        [enc setBuffer:bn2 offset:0 atIndex:4];
        [enc setBuffer:blen offset:0 atIndex:5];
        // Configure threadgroup size dynamically: prefer 512 for larger dims
        uint32_t tg_size = (dim >= 2048u ? 512u : 256u);
        id<MTLBuffer> btg = [device_ newBufferWithBytes:&tg_size length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];
        [enc setBuffer:btg offset:0 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(tg_size,1,1) threadsPerThreadgroup:MTLSizeMake(tg_size,1,1)];
        [enc endEncoding];
    }
    // Correlations with offsets 0,1,2
    id<MTLBuffer> bc00 = [device_ newBufferWithLength:sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bc01 = [device_ newBufferWithLength:sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bc20 = [device_ newBufferWithLength:sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bc21 = [device_ newBufferWithLength:sizeof(float) options:MTLResourceStorageModeManaged];
    uint32_t z=0, o1=1, o2=2;
    id<MTLBuffer> bz = [device_ newBufferWithBytes:&z length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b1off = [device_ newBufferWithBytes:&o1 length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b2off = [device_ newBufferWithBytes:&o2 length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_corr_off_];
        [enc setBuffer:b1 offset:0 atIndex:0]; [enc setBuffer:b2 offset:0 atIndex:1];
        [enc setBuffer:bc00 offset:0 atIndex:2]; [enc setBuffer:blen offset:0 atIndex:3]; [enc setBuffer:bz offset:0 atIndex:4]; [enc setBuffer:bz offset:0 atIndex:5];
        uint32_t tg_size = (dim >= 2048u ? 512u : 256u);
        id<MTLBuffer> btg = [device_ newBufferWithBytes:&tg_size length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];
        [enc setBuffer:btg offset:0 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(tg_size,1,1) threadsPerThreadgroup:MTLSizeMake(tg_size,1,1)];
        [enc endEncoding];
    }
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_corr_off_];
        [enc setBuffer:b1 offset:0 atIndex:0]; [enc setBuffer:b2 offset:0 atIndex:1];
        [enc setBuffer:bc01 offset:0 atIndex:2]; [enc setBuffer:blen offset:0 atIndex:3]; [enc setBuffer:bz offset:0 atIndex:4]; [enc setBuffer:b1off offset:0 atIndex:5];
        uint32_t tg_size = (dim >= 2048u ? 512u : 256u);
        id<MTLBuffer> btg = [device_ newBufferWithBytes:&tg_size length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];
        [enc setBuffer:btg offset:0 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(tg_size,1,1) threadsPerThreadgroup:MTLSizeMake(tg_size,1,1)];
        [enc endEncoding];
    }
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_corr_off_];
        [enc setBuffer:b1 offset:0 atIndex:0]; [enc setBuffer:b2 offset:0 atIndex:1];
        [enc setBuffer:bc20 offset:0 atIndex:2]; [enc setBuffer:blen offset:0 atIndex:3]; [enc setBuffer:b2off offset:0 atIndex:4]; [enc setBuffer:bz offset:0 atIndex:5];
        uint32_t tg_size = (dim >= 2048u ? 512u : 256u);
        id<MTLBuffer> btg = [device_ newBufferWithBytes:&tg_size length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];
        [enc setBuffer:btg offset:0 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(tg_size,1,1) threadsPerThreadgroup:MTLSizeMake(tg_size,1,1)];
        [enc endEncoding];
    }
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_corr_off_];
        [enc setBuffer:b1 offset:0 atIndex:0]; [enc setBuffer:b2 offset:0 atIndex:1];
        [enc setBuffer:bc21 offset:0 atIndex:2]; [enc setBuffer:blen offset:0 atIndex:3]; [enc setBuffer:b2off offset:0 atIndex:4]; [enc setBuffer:b1off offset:0 atIndex:5];
        uint32_t tg_size = (dim >= 2048u ? 512u : 256u);
        id<MTLBuffer> btg = [device_ newBufferWithBytes:&tg_size length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];
        [enc setBuffer:btg offset:0 atIndex:6];
        [enc dispatchThreads:MTLSizeMake(tg_size,1,1) threadsPerThreadgroup:MTLSizeMake(tg_size,1,1)];
        [enc endEncoding];
    }
    [cmd commit];
    [cmd waitUntilCompleted];

    float fdot=0.0f, fn1=0.0f, fn2=0.0f, f00=0.0f, f01=0.0f, f20=0.0f, f21=0.0f;
    memcpy(&fdot, [bdot contents], sizeof(float));
    memcpy(&fn1, [bn1 contents], sizeof(float));
    memcpy(&fn2, [bn2 contents], sizeof(float));
    memcpy(&f00, [bc00 contents], sizeof(float));
    memcpy(&f01, [bc01 contents], sizeof(float));
    memcpy(&f20, [bc20 contents], sizeof(float));
    memcpy(&f21, [bc21 contents], sizeof(float));

    float vis = 0.0f, coh = 0.0f, ortho = std::fabs(fdot);
    if (fn1 > 0.0f && fn2 > 0.0f) {
        float n1 = std::sqrt(fn1), n2 = std::sqrt(fn2);
        float num = std::fabs(fdot);
        coh = num / (n1 * n2);
        vis = (num*num) / ((n1*n1) * (n2*n2));
    }
    double S = double(f00) + double(f01) + double(f20) - double(f21);
    double bell_violation = S - 2.0;
    return {vis, coh, bell_violation, ortho};
}

void MetalBackend::fft_transform(const std::vector<float>& input,
                       std::vector<float>& real_out,
                       std::vector<float>& imag_out) {
    if (!available() || input.empty()) { real_out.clear(); imag_out.clear(); return; }
    uint32_t n = (uint32_t)input.size();
    real_out.assign(n, 0.0f);
    imag_out.assign(n, 0.0f);
    id<MTLBuffer> bin = [device_ newBufferWithBytes:input.data() length:n*sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> br = [device_ newBufferWithBytes:real_out.data() length:n*sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> bi = [device_ newBufferWithBytes:imag_out.data() length:n*sizeof(float) options:MTLResourceStorageModeManaged];
    id<MTLBuffer> blen = [device_ newBufferWithBytes:&n length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];

    id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso_fft_];
    [enc setBuffer:bin offset:0 atIndex:0];
    [enc setBuffer:br offset:0 atIndex:1];
    [enc setBuffer:bi offset:0 atIndex:2];
    [enc setBuffer:blen offset:0 atIndex:3];
    NSUInteger tgX = 256;
    NSUInteger gridX = ((n + tgX - 1) / tgX) * tgX;
    [enc dispatchThreads:MTLSizeMake(gridX, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgX, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    memcpy(real_out.data(), [br contents], n*sizeof(float));
    memcpy(imag_out.data(), [bi contents], n*sizeof(float));
}

std::vector<std::vector<float>> MetalBackend::batch_encode_fft(
    const std::vector<std::vector<float>>& batch_data,
    uint32_t pattern_dimension) {
    if (!available() || batch_data.empty()) return {};
    uint32_t batch_size = (uint32_t)batch_data.size();
    uint32_t data_length = (uint32_t)batch_data[0].size();

    // Prepare padded input [batch_size, pattern_dimension]
    std::vector<float> flat_in((size_t)batch_size * pattern_dimension, 0.0f);
    for (uint32_t b = 0; b < batch_size; ++b) {
        const auto &src = batch_data[b];
        uint32_t copyN = std::min(pattern_dimension, (uint32_t)src.size());
        std::memcpy(&flat_in[(size_t)b * pattern_dimension], src.data(), copyN * sizeof(float));
    }
    std::vector<float> flat_out((size_t)batch_size * pattern_dimension, 0.0f);

    // Try MPSGraph path when available (macOS 14+)
    bool usedGraph = false;
    if (NSClassFromString(@"MPSGraph") != nil) {
        @autoreleasepool {
            MPSGraph *graph = [[MPSGraph alloc] init];
            MPSShape *shape = (MPSShape*)@[@(batch_size), @(pattern_dimension)];
            MPSGraphTensor *input = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
            MPSGraphFFTDescriptor *fftDesc = [MPSGraphFFTDescriptor descriptor];
            fftDesc.inverse = NO;
            fftDesc.scalingMode = MPSGraphFFTScalingModeUnitary;
            NSArray<NSNumber*> *axes = @[ @1 ];
            MPSGraphTensor *fftT = [graph fastFourierTransformWithTensor:input axes:axes descriptor:fftDesc name:nil];
            // magnitude = sqrt(re^2 + im^2)
            MPSGraphTensor *re = [graph realPartOfTensor:fftT name:nil];
            MPSGraphTensor *im = [graph imaginaryPartOfTensor:fftT name:nil];
            MPSGraphTensor *re2 = [graph squareWithTensor:re name:nil];
            MPSGraphTensor *im2 = [graph squareWithTensor:im name:nil];
            MPSGraphTensor *sum = [graph additionWithPrimaryTensor:re2 secondaryTensor:im2 name:nil];
            MPSGraphTensor *mag = [graph squareRootWithTensor:sum name:nil];

            id<MTLBuffer> bin = [device_ newBufferWithBytes:flat_in.data() length:flat_in.size()*sizeof(float) options:MTLResourceStorageModeManaged];
            id<MTLBuffer> bout = [device_ newBufferWithBytes:flat_out.data() length:flat_out.size()*sizeof(float) options:MTLResourceStorageModeManaged];
            MPSGraphTensorData *inData = [[MPSGraphTensorData alloc] initWithMTLBuffer:bin shape:shape dataType:MPSDataTypeFloat32];
            MPSGraphTensorData *outData = [[MPSGraphTensorData alloc] initWithMTLBuffer:bout shape:shape dataType:MPSDataTypeFloat32];
            NSDictionary *feeds = @{ input : inData };
            NSDictionary *results = @{ mag : outData };
            // Execute on command queue (blocking run)
            [graph runWithMTLCommandQueue:queue_ feeds:(MPSGraphTensorDataDictionary*)feeds targetOperations:nil resultsDictionary:(MPSGraphTensorDataDictionary*)results];
            // Copy back
            std::memcpy(flat_out.data(), [bout contents], flat_out.size()*sizeof(float));
            usedGraph = true;
        }
    }

    if (!usedGraph) {
        // Fallback to optimized compute kernel path
        id<MTLBuffer> bin = [device_ newBufferWithBytes:flat_in.data() length:flat_in.size()*sizeof(float) options:MTLResourceStorageModeManaged];
        id<MTLBuffer> bout = [device_ newBufferWithBytes:flat_out.data() length:flat_out.size()*sizeof(float) options:MTLResourceStorageModeManaged];
        id<MTLBuffer> bbs = [device_ newBufferWithBytes:&batch_size length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];
        id<MTLBuffer> bdl = [device_ newBufferWithBytes:&data_length length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];
        id<MTLBuffer> bpd = [device_ newBufferWithBytes:&pattern_dimension length:sizeof(uint32_t) options:MTLResourceStorageModeManaged];

        id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso_batch_store_fft_];
        [enc setBuffer:bin offset:0 atIndex:0];
        [enc setBuffer:bout offset:0 atIndex:1];
        [enc setBuffer:bbs offset:0 atIndex:2];
        [enc setBuffer:bdl offset:0 atIndex:3];
        [enc setBuffer:bpd offset:0 atIndex:4];
        NSUInteger tgX = 256;
        NSUInteger gridX = ((batch_size + tgX - 1) / tgX) * tgX;
        [enc dispatchThreads:MTLSizeMake(gridX, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgX, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        memcpy(flat_out.data(), [bout contents], flat_out.size()*sizeof(float));
    }

    // Pack 2D result
    std::vector<std::vector<float>> out; out.reserve(batch_size);
    for (uint32_t i=0;i<batch_size;i++){
        out.emplace_back(flat_out.begin()+ (size_t)i*pattern_dimension, flat_out.begin()+ (size_t)(i+1)*pattern_dimension);
    }
    return out;
}

// Ultra-optimized FFT path using cached MPSGraph graph + persistent MTLBuffers
std::vector<std::vector<float>> MetalBackend::batch_encode_fft_ultra(
    const std::vector<std::vector<float>>& batch_data,
    uint32_t pattern_dimension) {
    if (!available() || batch_data.empty()) return {};

    uint32_t batch_size = (uint32_t)batch_data.size();

    bool graph_ok = false;
    if (NSClassFromString(@"MPSGraph") != nil) {
        if (mps_graph_fft_ == nil) {
            @autoreleasepool {
                MPSGraph *graph = [[MPSGraph alloc] init];
                MPSGraphTensor *input = [graph placeholderWithShape:nil dataType:MPSDataTypeFloat32 name:nil];
                MPSGraphFFTDescriptor *fftDesc = [MPSGraphFFTDescriptor descriptor];
                fftDesc.inverse = NO; fftDesc.scalingMode = MPSGraphFFTScalingModeUnitary;
                NSArray<NSNumber*> *axes = @[ @1 ];
                MPSGraphTensor *fftT = [graph fastFourierTransformWithTensor:input axes:axes descriptor:fftDesc name:nil];
                MPSGraphTensor *re = [graph realPartOfTensor:fftT name:nil];
                MPSGraphTensor *im = [graph imaginaryPartOfTensor:fftT name:nil];
                MPSGraphTensor *re2 = [graph squareWithTensor:re name:nil];
                MPSGraphTensor *im2 = [graph squareWithTensor:im name:nil];
                MPSGraphTensor *sum = [graph additionWithPrimaryTensor:re2 secondaryTensor:im2 name:nil];
                MPSGraphTensor *mag = [graph squareRootWithTensor:sum name:nil];
                mps_graph_fft_ = graph;
                mps_graph_input_ = input;
                mps_graph_mag_ = mag;
            }
        }
        graph_ok = (mps_graph_fft_ != nil && mps_graph_input_ != nil && mps_graph_mag_ != nil);
    }

    // Prepare padded input buffer once
    std::vector<float> flat_in((size_t)batch_size * pattern_dimension, 0.0f);
    for (uint32_t b = 0; b < batch_size; ++b) {
        const auto &src = batch_data[b];
        uint32_t copyN = std::min(pattern_dimension, (uint32_t)src.size());
        std::memcpy(&flat_in[(size_t)b * pattern_dimension], src.data(), copyN * sizeof(float));
    }
    std::vector<float> flat_out((size_t)batch_size * pattern_dimension, 0.0f);

    if (graph_ok) {
        NSUInteger inBytes = (NSUInteger)(flat_in.size() * sizeof(float));
        NSUInteger outBytes = (NSUInteger)(flat_out.size() * sizeof(float));
        id<MTLBuffer> bin = hb_pop_buffer((NSMutableArray*)input_pool_, (NSLock*)pool_lock_, device_, inBytes);
        id<MTLBuffer> bout = hb_pop_buffer((NSMutableArray*)output_pool_, (NSLock*)pool_lock_, device_, outBytes);
        std::memcpy([bin contents], flat_in.data(), inBytes);

        MPSShape *shape = (MPSShape*)@[@(batch_size), @(pattern_dimension)];
        MPSGraphTensorData *inTD = [[MPSGraphTensorData alloc] initWithMTLBuffer:bin shape:shape dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *outTD = [[MPSGraphTensorData alloc] initWithMTLBuffer:bout shape:shape dataType:MPSDataTypeFloat32];

        NSDictionary *feeds = @{ (MPSGraphTensor*)mps_graph_input_ : inTD };
        NSDictionary *results = @{ (MPSGraphTensor*)mps_graph_mag_ : outTD };

        auto t0 = std::chrono::high_resolution_clock::now();
        [(MPSGraph*)mps_graph_fft_ runWithMTLCommandQueue:queue_ feeds:(MPSGraphTensorDataDictionary*)feeds targetOperations:nil resultsDictionary:(MPSGraphTensorDataDictionary*)results];
        std::memcpy(flat_out.data(), [bout contents], outBytes);
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
        metrics_.batch_store_time_ms = ms;
        metrics_.operations_per_second = (ms>0)? (uint64_t)(batch_size * 1000.0 / ms) : 0;

        hb_push_buffer((NSMutableArray*)input_pool_, (NSLock*)pool_lock_, bin);
        hb_push_buffer((NSMutableArray*)output_pool_, (NSLock*)pool_lock_, bout);
    } else {
        return batch_encode_fft(batch_data, pattern_dimension);
    }

    std::vector<std::vector<float>> out; out.reserve(batch_size);
    for (uint32_t i=0;i<batch_size;i++){
        out.emplace_back(flat_out.begin()+ (size_t)i*pattern_dimension, flat_out.begin()+ (size_t)(i+1)*pattern_dimension);
    }
    return out;
}

// ============================================================================
// GPU COMPRESSION PIPELINE - KERNEL 1: QUANTIZATION
// ============================================================================

std::vector<std::vector<float>> MetalBackend::gpu_holographic_quantize(
    const std::vector<std::vector<float>>& input_real,
    const std::vector<std::vector<float>>& input_imag,
    uint32_t layer_index,
    const QuantizationParams& params) {
    
    if (!available() || input_real.empty() || input_imag.empty()) return {};
    
    uint32_t coefficient_count = static_cast<uint32_t>(input_real[0].size());
    uint32_t batch_size = static_cast<uint32_t>(input_real.size());
    
    // Flatten input data
    std::vector<float> flat_real, flat_imag;
    flat_real.reserve(batch_size * coefficient_count);
    flat_imag.reserve(batch_size * coefficient_count);
    
    for (uint32_t b = 0; b < batch_size; ++b) {
        flat_real.insert(flat_real.end(), input_real[b].begin(), input_real[b].end());
        flat_imag.insert(flat_imag.end(), input_imag[b].begin(), input_imag[b].end());
    }
    
    // Prepare output buffers
    std::vector<float> output_real(batch_size * coefficient_count, 0.0f);
    std::vector<float> output_imag(batch_size * coefficient_count, 0.0f);
    
    // Create Metal buffers
    id<MTLBuffer> b_input_real = [device_ newBufferWithBytes:flat_real.data() 
                                                      length:flat_real.size() * sizeof(float) 
                                                     options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_input_imag = [device_ newBufferWithBytes:flat_imag.data() 
                                                      length:flat_imag.size() * sizeof(float) 
                                                     options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_output_real = [device_ newBufferWithBytes:output_real.data()
                                                       length:output_real.size() * sizeof(float)
                                                      options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_output_imag = [device_ newBufferWithBytes:output_imag.data()
                                                       length:output_imag.size() * sizeof(float)
                                                      options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_params = [device_ newBufferWithBytes:&params 
                                                  length:sizeof(QuantizationParams) 
                                                 options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_layer = [device_ newBufferWithBytes:&layer_index 
                                                 length:sizeof(uint32_t) 
                                                options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_count = [device_ newBufferWithBytes:&coefficient_count 
                                                 length:sizeof(uint32_t) 
                                                options:MTLResourceStorageModeManaged];
    
    // Execute quantization kernel
    id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    
    [enc setComputePipelineState:pso_quantize_];
    [enc setBuffer:b_input_real offset:0 atIndex:0];
    [enc setBuffer:b_input_imag offset:0 atIndex:1];
    [enc setBuffer:b_output_real offset:0 atIndex:2];
    [enc setBuffer:b_output_imag offset:0 atIndex:3];
    [enc setBuffer:b_params offset:0 atIndex:4];
    [enc setBuffer:b_layer offset:0 atIndex:5];
    [enc setBuffer:b_count offset:0 atIndex:6];
    
    // Dispatch threads
    NSUInteger tgX = 256;
    NSUInteger gridX = ((coefficient_count + tgX - 1) / tgX) * tgX;
    [enc dispatchThreads:MTLSizeMake(gridX, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgX, 1, 1)];
    
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    
    // Copy results back
    memcpy(output_real.data(), [b_output_real contents], output_real.size() * sizeof(float));
    memcpy(output_imag.data(), [b_output_imag contents], output_imag.size() * sizeof(float));
    
    // Convert back to 2D vector
    std::vector<std::vector<float>> result_real, result_imag;
    result_real.reserve(batch_size);
    result_imag.reserve(batch_size);
    
    for (uint32_t b = 0; b < batch_size; ++b) {
        std::vector<float> real_slice(
            output_real.begin() + b * coefficient_count,
            output_real.begin() + (b + 1) * coefficient_count
        );
        std::vector<float> imag_slice(
            output_imag.begin() + b * coefficient_count,
            output_imag.begin() + (b + 1) * coefficient_count
        );
        result_real.push_back(std::move(real_slice));
        result_imag.push_back(std::move(imag_slice));
    }
    
    return result_real; // Return real part for now, could return both
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>, 
           std::vector<std::vector<float>>, std::vector<std::vector<float>>>
MetalBackend::gpu_holographic_quantize_with_validation(
    const std::vector<std::vector<float>>& input_real,
    const std::vector<std::vector<float>>& input_imag,
    uint32_t layer_index,
    const QuantizationParams& params) {
    
    if (!available() || input_real.empty() || input_imag.empty()) {
        return {{}, {}, {}, {}};
    }
    
    uint32_t coefficient_count = static_cast<uint32_t>(input_real[0].size());
    uint32_t batch_size = static_cast<uint32_t>(input_real.size());
    
    // Flatten input data
    std::vector<float> flat_real, flat_imag;
    flat_real.reserve(batch_size * coefficient_count);
    flat_imag.reserve(batch_size * coefficient_count);
    
    for (uint32_t b = 0; b < batch_size; ++b) {
        flat_real.insert(flat_real.end(), input_real[b].begin(), input_real[b].end());
        flat_imag.insert(flat_imag.end(), input_imag[b].begin(), input_imag[b].end());
    }
    
    // Prepare output buffers
    std::vector<float> output_real(batch_size * coefficient_count, 0.0f);
    std::vector<float> output_imag(batch_size * coefficient_count, 0.0f);
    std::vector<float> phase_errors(batch_size * coefficient_count, 0.0f);
    std::vector<float> amplitude_errors(batch_size * coefficient_count, 0.0f);
    
    // Create Metal buffers
    id<MTLBuffer> b_input_real = [device_ newBufferWithBytes:flat_real.data() 
                                                      length:flat_real.size() * sizeof(float) 
                                                     options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_input_imag = [device_ newBufferWithBytes:flat_imag.data() 
                                                      length:flat_imag.size() * sizeof(float) 
                                                     options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_output_real = [device_ newBufferWithBytes:output_real.data()
                                                       length:output_real.size() * sizeof(float)
                                                      options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_output_imag = [device_ newBufferWithBytes:output_imag.data()
                                                       length:output_imag.size() * sizeof(float)
                                                      options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_phase_errors = [device_ newBufferWithBytes:phase_errors.data()
                                                        length:phase_errors.size() * sizeof(float)
                                                       options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_amplitude_errors = [device_ newBufferWithBytes:amplitude_errors.data()
                                                            length:amplitude_errors.size() * sizeof(float)
                                                           options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_params = [device_ newBufferWithBytes:&params 
                                                  length:sizeof(QuantizationParams) 
                                                 options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_layer = [device_ newBufferWithBytes:&layer_index 
                                                 length:sizeof(uint32_t) 
                                                options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_count = [device_ newBufferWithBytes:&coefficient_count 
                                                 length:sizeof(uint32_t) 
                                                options:MTLResourceStorageModeManaged];
    
    // Execute validation kernel
    id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    
    [enc setComputePipelineState:pso_quantize_validation_];
    [enc setBuffer:b_input_real offset:0 atIndex:0];
    [enc setBuffer:b_input_imag offset:0 atIndex:1];
    [enc setBuffer:b_output_real offset:0 atIndex:2];
    [enc setBuffer:b_output_imag offset:0 atIndex:3];
    [enc setBuffer:b_phase_errors offset:0 atIndex:4];
    [enc setBuffer:b_amplitude_errors offset:0 atIndex:5];
    [enc setBuffer:b_params offset:0 atIndex:6];
    [enc setBuffer:b_layer offset:0 atIndex:7];
    [enc setBuffer:b_count offset:0 atIndex:8];
    
    // Dispatch threads
    NSUInteger tgX = 256;
    NSUInteger gridX = ((coefficient_count + tgX - 1) / tgX) * tgX;
    [enc dispatchThreads:MTLSizeMake(gridX, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgX, 1, 1)];
    
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    
    // Copy results back
    memcpy(output_real.data(), [b_output_real contents], output_real.size() * sizeof(float));
    memcpy(output_imag.data(), [b_output_imag contents], output_imag.size() * sizeof(float));
    memcpy(phase_errors.data(), [b_phase_errors contents], phase_errors.size() * sizeof(float));
    memcpy(amplitude_errors.data(), [b_amplitude_errors contents], amplitude_errors.size() * sizeof(float));
    
    // Convert back to 2D vectors
    std::vector<std::vector<float>> result_real, result_imag, result_phase_errors, result_amplitude_errors;
    result_real.reserve(batch_size);
    result_imag.reserve(batch_size);
    result_phase_errors.reserve(batch_size);
    result_amplitude_errors.reserve(batch_size);
    
    for (uint32_t b = 0; b < batch_size; ++b) {
        std::vector<float> real_slice(
            output_real.begin() + b * coefficient_count,
            output_real.begin() + (b + 1) * coefficient_count
        );
        std::vector<float> imag_slice(
            output_imag.begin() + b * coefficient_count,
            output_imag.begin() + (b + 1) * coefficient_count
        );
        std::vector<float> phase_error_slice(
            phase_errors.begin() + b * coefficient_count,
            phase_errors.begin() + (b + 1) * coefficient_count
        );
        std::vector<float> amplitude_error_slice(
            amplitude_errors.begin() + b * coefficient_count,
            amplitude_errors.begin() + (b + 1) * coefficient_count
        );
        
        result_real.push_back(std::move(real_slice));
        result_imag.push_back(std::move(imag_slice));
        result_phase_errors.push_back(std::move(phase_error_slice));
        result_amplitude_errors.push_back(std::move(amplitude_error_slice));
    }
    
    return {result_real, result_imag, result_phase_errors, result_amplitude_errors};
}

std::array<float, 4> MetalBackend::gpu_quantization_statistics(
    const std::vector<std::vector<float>>& phase_errors,
    const std::vector<std::vector<float>>& amplitude_errors) {
    
    if (!available() || phase_errors.empty() || amplitude_errors.empty()) {
        return {0.0f, 0.0f, 0.0f, 0.0f};
    }
    
    uint32_t coefficient_count = static_cast<uint32_t>(phase_errors[0].size());
    uint32_t batch_size = static_cast<uint32_t>(phase_errors.size());
    uint32_t total_coefficients = batch_size * coefficient_count;
    
    // Flatten error data
    std::vector<float> flat_phase_errors, flat_amplitude_errors;
    flat_phase_errors.reserve(total_coefficients);
    flat_amplitude_errors.reserve(total_coefficients);
    
    for (uint32_t b = 0; b < batch_size; ++b) {
        flat_phase_errors.insert(flat_phase_errors.end(), phase_errors[b].begin(), phase_errors[b].end());
        flat_amplitude_errors.insert(flat_amplitude_errors.end(), amplitude_errors[b].begin(), amplitude_errors[b].end());
    }
    
    // Prepare output buffer for statistics
    std::vector<float> statistics(4, 0.0f); // [max_phase_err, max_amp_err, mean_phase_err, mean_amp_err]
    
    // Create Metal buffers
    id<MTLBuffer> b_phase_errors = [device_ newBufferWithBytes:flat_phase_errors.data() 
                                                        length:flat_phase_errors.size() * sizeof(float) 
                                                       options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_amplitude_errors = [device_ newBufferWithBytes:flat_amplitude_errors.data() 
                                                            length:flat_amplitude_errors.size() * sizeof(float) 
                                                           options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_statistics = [device_ newBufferWithBytes:statistics.data()
                                                      length:statistics.size() * sizeof(float)
                                                     options:MTLResourceStorageModeManaged];
    id<MTLBuffer> b_count = [device_ newBufferWithBytes:&total_coefficients 
                                                 length:sizeof(uint32_t) 
                                                options:MTLResourceStorageModeManaged];
    
    uint32_t threadgroup_size = 256;
    id<MTLBuffer> b_tg_size = [device_ newBufferWithBytes:&threadgroup_size 
                                                   length:sizeof(uint32_t) 
                                                  options:MTLResourceStorageModeManaged];
    
    // Execute statistics kernel
    id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    
    [enc setComputePipelineState:pso_quantize_stats_];
    [enc setBuffer:b_phase_errors offset:0 atIndex:0];
    [enc setBuffer:b_amplitude_errors offset:0 atIndex:1];
    [enc setBuffer:b_statistics offset:0 atIndex:2];
    [enc setBuffer:b_count offset:0 atIndex:3];
    [enc setBuffer:b_tg_size offset:0 atIndex:4];
    
    // Dispatch threads
    NSUInteger tgX = threadgroup_size;
    NSUInteger gridX = ((total_coefficients + tgX - 1) / tgX) * tgX;
    [enc dispatchThreads:MTLSizeMake(gridX, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgX, 1, 1)];
    
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    
    // Copy results back
    memcpy(statistics.data(), [b_statistics contents], statistics.size() * sizeof(float));
    
    return {statistics[0], statistics[1], statistics[2], statistics[3]};
}

// ============================================================================
// GPU COMPRESSION PIPELINE - KERNEL 2: BITPLANE EXTRACTION
// ============================================================================

// Removed: gpu_bitplane_extraction method - no longer needed with holographic wave reconstruction

// Removed: gpu_bitplane_statistics method - no longer needed with holographic wave reconstruction

// Removed: gpu_bitplane_reconstruction method - no longer needed with holographic wave reconstruction

// ============================================================================
// GPU COMPRESSION PIPELINE - KERNEL 3: SPARSE ENCODING
// ============================================================================

std::tuple<std::vector<float>, std::vector<float>, std::vector<uint32_t>, uint32_t>
MetalBackend::gpu_sparse_encoding(
    const std::vector<float>& input_real,
    const std::vector<float>& input_imag,
    float threshold,
    uint32_t max_sparse_count) {
    
    if (!device_ || !queue_) {
        throw std::runtime_error("Metal device not initialized");
    }
    
    uint32_t input_size = static_cast<uint32_t>(input_real.size());
    if (input_imag.size() != input_size) {
        throw std::invalid_argument("Input arrays must have same size");
    }
    
    // Create buffers
    id<MTLBuffer> b_input_real = [device_ newBufferWithBytes:input_real.data()
                                                      length:input_size * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_input_imag = [device_ newBufferWithBytes:input_imag.data()
                                                      length:input_size * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_sparse_real = [device_ newBufferWithLength:max_sparse_count * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_sparse_imag = [device_ newBufferWithLength:max_sparse_count * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_sparse_indices = [device_ newBufferWithLength:max_sparse_count * sizeof(uint32_t)
                                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_sparse_count = [device_ newBufferWithLength:sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
    
    // Initialize sparse count to 0
    uint32_t* count_ptr = static_cast<uint32_t*>([b_sparse_count contents]);
    *count_ptr = 0;
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> cmd_buffer = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    
    [encoder setComputePipelineState:sparse_encoding_pipeline_];
    [encoder setBuffer:b_input_real offset:0 atIndex:0];
    [encoder setBuffer:b_input_imag offset:0 atIndex:1];
    [encoder setBuffer:b_sparse_real offset:0 atIndex:2];
    [encoder setBuffer:b_sparse_imag offset:0 atIndex:3];
    [encoder setBuffer:b_sparse_indices offset:0 atIndex:4];
    [encoder setBuffer:b_sparse_count offset:0 atIndex:5];
    [encoder setBytes:&input_size length:sizeof(uint32_t) atIndex:6];
    [encoder setBytes:&threshold length:sizeof(float) atIndex:7];
    [encoder setBytes:&max_sparse_count length:sizeof(uint32_t) atIndex:8];
    
    // Dispatch threads
    MTLSize grid_size = MTLSizeMake(input_size, 1, 1);
    MTLSize threadgroup_size = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    
    [cmd_buffer commit];
    [cmd_buffer waitUntilCompleted];
    
    // Get results
    uint32_t actual_count = *count_ptr;
    std::vector<float> sparse_real(actual_count);
    std::vector<float> sparse_imag(actual_count);
    std::vector<uint32_t> sparse_indices(actual_count);
    
    memcpy(sparse_real.data(), [b_sparse_real contents], actual_count * sizeof(float));
    memcpy(sparse_imag.data(), [b_sparse_imag contents], actual_count * sizeof(float));
    memcpy(sparse_indices.data(), [b_sparse_indices contents], actual_count * sizeof(uint32_t));
    
    return {sparse_real, sparse_imag, sparse_indices, actual_count};
}

// ============================================================================
// GPU COMPRESSION PIPELINE - KERNEL 4: ENTROPY CODING
// ============================================================================

std::tuple<std::vector<uint8_t>, uint32_t>
MetalBackend::gpu_entropy_coding(
    const std::vector<float>& sparse_real,
    const std::vector<float>& sparse_imag,
    const std::vector<uint32_t>& sparse_indices,
    uint32_t sparse_count) {
    
    if (!device_ || !queue_) {
        throw std::runtime_error("Metal device not initialized");
    }
    
    uint32_t max_encoded_size = sparse_count * 6; // 6 bytes per coefficient
    
    // Create buffers
    id<MTLBuffer> b_sparse_real = [device_ newBufferWithBytes:sparse_real.data()
                                                       length:sparse_count * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_sparse_imag = [device_ newBufferWithBytes:sparse_imag.data()
                                                       length:sparse_count * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_sparse_indices = [device_ newBufferWithBytes:sparse_indices.data()
                                                          length:sparse_count * sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_sparse_count = [device_ newBufferWithBytes:&sparse_count
                                                        length:sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_encoded_data = [device_ newBufferWithLength:max_encoded_size
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_encoded_size = [device_ newBufferWithLength:sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> cmd_buffer = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    
    [encoder setComputePipelineState:entropy_coding_pipeline_];
    [encoder setBuffer:b_sparse_real offset:0 atIndex:0];
    [encoder setBuffer:b_sparse_imag offset:0 atIndex:1];
    [encoder setBuffer:b_sparse_indices offset:0 atIndex:2];
    [encoder setBuffer:b_sparse_count offset:0 atIndex:3];
    [encoder setBuffer:b_encoded_data offset:0 atIndex:4];
    [encoder setBuffer:b_encoded_size offset:0 atIndex:5];
    [encoder setBytes:&max_encoded_size length:sizeof(uint32_t) atIndex:6];
    
    // Dispatch threads
    MTLSize grid_size = MTLSizeMake(sparse_count, 1, 1);
    MTLSize threadgroup_size = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    
    [cmd_buffer commit];
    [cmd_buffer waitUntilCompleted];
    
    // Get results
    uint32_t encoded_size = *static_cast<uint32_t*>([b_encoded_size contents]);
    std::vector<uint8_t> encoded_data(encoded_size);
    memcpy(encoded_data.data(), [b_encoded_data contents], encoded_size);
    
    return {encoded_data, encoded_size};
}

// ============================================================================
// GPU COMPRESSION PIPELINE - KERNEL 5: ENTROPY DECODING
// ============================================================================

std::tuple<std::vector<float>, std::vector<float>, std::vector<uint32_t>, uint32_t>
MetalBackend::gpu_entropy_decoding(
    const std::vector<uint8_t>& encoded_data,
    uint32_t encoded_size) {
    
    if (!device_ || !queue_) {
        throw std::runtime_error("Metal device not initialized");
    }
    
    uint32_t max_decoded_count = encoded_size / 6; // 6 bytes per coefficient
    
    // Create buffers
    id<MTLBuffer> b_encoded_data = [device_ newBufferWithBytes:encoded_data.data()
                                                        length:encoded_size
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_encoded_size = [device_ newBufferWithBytes:&encoded_size
                                                         length:sizeof(uint32_t) 
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_decoded_real = [device_ newBufferWithLength:max_decoded_count * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_decoded_imag = [device_ newBufferWithLength:max_decoded_count * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_decoded_indices = [device_ newBufferWithLength:max_decoded_count * sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_decoded_count = [device_ newBufferWithLength:sizeof(uint32_t)
                                                          options:MTLResourceStorageModeShared];
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> cmd_buffer = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    
    [encoder setComputePipelineState:entropy_decoding_pipeline_];
    [encoder setBuffer:b_encoded_data offset:0 atIndex:0];
    [encoder setBuffer:b_encoded_size offset:0 atIndex:1];
    [encoder setBuffer:b_decoded_real offset:0 atIndex:2];
    [encoder setBuffer:b_decoded_imag offset:0 atIndex:3];
    [encoder setBuffer:b_decoded_indices offset:0 atIndex:4];
    [encoder setBuffer:b_decoded_count offset:0 atIndex:5];
    [encoder setBytes:&max_decoded_count length:sizeof(uint32_t) atIndex:6];
    
    // Dispatch threads
    MTLSize grid_size = MTLSizeMake(max_decoded_count, 1, 1);
    MTLSize threadgroup_size = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    
    [cmd_buffer commit];
    [cmd_buffer waitUntilCompleted];
    
    // Get results
    uint32_t decoded_count = *static_cast<uint32_t*>([b_decoded_count contents]);
    std::vector<float> decoded_real(decoded_count);
    std::vector<float> decoded_imag(decoded_count);
    std::vector<uint32_t> decoded_indices(decoded_count);
    
    memcpy(decoded_real.data(), [b_decoded_real contents], decoded_count * sizeof(float));
    memcpy(decoded_imag.data(), [b_decoded_imag contents], decoded_count * sizeof(float));
    memcpy(decoded_indices.data(), [b_decoded_indices contents], decoded_count * sizeof(uint32_t));
    
    return {decoded_real, decoded_imag, decoded_indices, decoded_count};
}

// ============================================================================
// GPU COMPRESSION PIPELINE - KERNEL 6: SPARSE DECODING
// ============================================================================

std::tuple<std::vector<float>, std::vector<float>>
MetalBackend::gpu_sparse_decoding(
    const std::vector<float>& sparse_real,
    const std::vector<float>& sparse_imag,
    const std::vector<uint32_t>& sparse_indices,
    uint32_t sparse_count,
    uint32_t output_size) {
    
    if (!device_ || !queue_) {
        throw std::runtime_error("Metal device not initialized");
    }
    
    // Create buffers
    id<MTLBuffer> b_sparse_real = [device_ newBufferWithBytes:sparse_real.data()
                                                       length:sparse_count * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_sparse_imag = [device_ newBufferWithBytes:sparse_imag.data()
                                                       length:sparse_count * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_sparse_indices = [device_ newBufferWithBytes:sparse_indices.data()
                                                          length:sparse_count * sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_sparse_count = [device_ newBufferWithBytes:&sparse_count
                                                 length:sizeof(uint32_t) 
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_output_real = [device_ newBufferWithLength:output_size * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_output_imag = [device_ newBufferWithLength:output_size * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> cmd_buffer = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];
    
    [encoder setComputePipelineState:sparse_decoding_pipeline_];
    [encoder setBuffer:b_sparse_real offset:0 atIndex:0];
    [encoder setBuffer:b_sparse_imag offset:0 atIndex:1];
    [encoder setBuffer:b_sparse_indices offset:0 atIndex:2];
    [encoder setBuffer:b_sparse_count offset:0 atIndex:3];
    [encoder setBuffer:b_output_real offset:0 atIndex:4];
    [encoder setBuffer:b_output_imag offset:0 atIndex:5];
    [encoder setBytes:&output_size length:sizeof(uint32_t) atIndex:6];
    
    // Dispatch threads
    MTLSize grid_size = MTLSizeMake(output_size, 1, 1);
    MTLSize threadgroup_size = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
    [encoder endEncoding];
    
    [cmd_buffer commit];
    [cmd_buffer waitUntilCompleted];
    
    // Get results
    std::vector<float> output_real(output_size);
    std::vector<float> output_imag(output_size);
    
    memcpy(output_real.data(), [b_output_real contents], output_size * sizeof(float));
    memcpy(output_imag.data(), [b_output_imag contents], output_size * sizeof(float));
    
    return {output_real, output_imag};
}

} // namespace holo

#endif
void holo::MetalBackend::initialize_mps_fft() {
    // Ensure MPS is supported on this device
    if (!device_) return;
#if defined(MPS_SUPPORTS_AUTOMATICALLY)
    // noop
#endif
    // Nothing to set up eagerly; we will create descriptors on demand.
}
