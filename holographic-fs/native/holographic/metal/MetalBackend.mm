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
    
    // Get the current working directory and build absolute paths
    NSString* cwd = [[NSFileManager defaultManager] currentDirectoryPath];
    
    // Try to load from build directory first, then fallback to source
    NSString* buildPath = [cwd stringByAppendingPathComponent:@"holographic-fs/native/holographic/build/metal/holographic_memory.metal"];
    NSString* file = [NSString stringWithContentsOfFile:buildPath encoding:NSUTF8StringEncoding error:&err];
    
    if (!file) {
        // Fallback to source directory
        NSString* srcPath = [cwd stringByAppendingPathComponent:@"holographic-fs/native/holographic/metal/holographic_memory.metal"];
        file = [NSString stringWithContentsOfFile:srcPath encoding:NSUTF8StringEncoding error:&err];
    }
    
    if (!file) {
        NSLog(@"Failed to load Metal shader: %@", err.localizedDescription);
        return;
    }
    
    library_ = [device_ newLibraryWithSource:file options:nil error:&err];
    if (!library_) {
        NSLog(@"Failed to create Metal library: %@", err.localizedDescription);
        return;
    }
    
    // Create pipeline states
    pso_vector_add_ = create_pipeline("enhanced_vector_add");
    pso_batch_store_ = create_pipeline("batch_holographic_store");
    pso_similarity_ = create_pipeline("holographic_similarity_search");
    pso_fft_ = create_pipeline("holographic_fft_transform");
    pso_batch_store_fft_ = create_pipeline("batch_holographic_encode_fft");
    initialize_mps_fft();
}

id<MTLComputePipelineState> MetalBackend::create_pipeline(const std::string& function_name) {
    NSError* err = nullptr;
    id<MTLFunction> fn = [library_ newFunctionWithName:str(function_name.c_str())];
    if (!fn) return nil;
    
    id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:fn error:&err];
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
