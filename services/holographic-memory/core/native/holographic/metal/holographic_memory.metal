// Advanced Metal kernels for Holographic Memory GPU acceleration
// Phase 2: MPS FFT and interference kernels

#include <metal_stdlib>
#include <metal_math>
#include <metal_simdgroup>
using namespace metal;

// FFT kernel using Metal Performance Shaders
kernel void holographic_fft_transform(
    device const float* input_signal [[buffer(0)]],
    device float* real_output [[buffer(1)]],
    device float* imag_output [[buffer(2)]],
    constant uint& signal_length [[buffer(3)]],
    uint thread_id [[thread_position_in_grid]]
) {
    if (thread_id >= signal_length) return;
    
    // Simplified FFT-like transformation for holographic encoding
    // Real implementation would use MPS FFT framework
    float real = 0.0f;
    float imag = 0.0f;
    
    for (uint k = 0; k < signal_length; k++) {
        float angle = -2.0f * M_PI_F * thread_id * k / signal_length;
        float cos_angle = cos(angle);
        float sin_angle = sin(angle);
        
        real += input_signal[k] * cos_angle;
        imag += input_signal[k] * sin_angle;
    }
    
    real_output[thread_id] = real / signal_length;
    imag_output[thread_id] = imag / signal_length;
}

// Wave interference kernel for holographic superposition
kernel void wave_interference_pattern(
    device const float* wave1_real [[buffer(0)]],
    device const float* wave1_imag [[buffer(1)]],
    device const float* wave2_real [[buffer(2)]],
    device const float* wave2_imag [[buffer(3)]],
    device float* interference_real [[buffer(4)]],
    device float* interference_imag [[buffer(5)]],
    constant uint& dimension [[buffer(6)]],
    uint thread_id [[thread_position_in_grid]]
) {
    if (thread_id >= dimension) return;
    
    // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    float a = wave1_real[thread_id];
    float b = wave1_imag[thread_id];
    float c = wave2_real[thread_id];
    float d = wave2_imag[thread_id];
    
    interference_real[thread_id] = a * c - b * d;
    interference_imag[thread_id] = a * d + b * c;
}

// Semantic similarity kernel for holographic search
kernel void holographic_similarity_search(
    device const float* query_vector [[buffer(0)]],
    device const float* stored_vectors [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant uint& vector_count [[buffer(3)]],
    constant uint& dimension [[buffer(4)]],
    uint thread_id [[thread_position_in_grid]]
) {
    if (thread_id >= vector_count) return;
    
    float similarity = 0.0f;
    uint offset = thread_id * dimension;
    
    for (uint i = 0; i < dimension; i++) {
        float diff = query_vector[i] - stored_vectors[offset + i];
        similarity += diff * diff;
    }
    
    // Convert distance to similarity (higher values = more similar)
    similarities[thread_id] = 1.0f / (1.0f + sqrt(similarity));
}

// Batch processing kernel for multiple holographic operations
kernel void batch_holographic_store(
    device const float* input_data [[buffer(0)]],
    device float*       output_patterns [[buffer(1)]],
    constant uint&      batch_size [[buffer(2)]],
    constant uint&      data_length [[buffer(3)]],
    constant uint&      pattern_dimension [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    // 2D grid: x = pattern block (4 outputs), y = batch index b
    const uint i_block = gid.x;
    const uint i0 = i_block * 4u;
    const uint b = gid.y;
    if (b >= batch_size || i0 >= pattern_dimension) return;

    const uint input_offset = b * data_length;
    const float denom = max(1.0f, float(data_length));

    // Threadgroup tile of input_data (up to 256 elements)
    constexpr uint TILE = 256u;
    threadgroup float sdata[TILE];

    // Precompute angle steps for 4 outputs
    const float delta0 = 2.0f * M_PI_F * float(i0 + 0u) / float(pattern_dimension);
    const float delta1 = (i0 + 1u < pattern_dimension) ? 2.0f * M_PI_F * float(i0 + 1u) / float(pattern_dimension) : 0.0f;
    const float delta2 = (i0 + 2u < pattern_dimension) ? 2.0f * M_PI_F * float(i0 + 2u) / float(pattern_dimension) : 0.0f;
    const float delta3 = (i0 + 3u < pattern_dimension) ? 2.0f * M_PI_F * float(i0 + 3u) / float(pattern_dimension) : 0.0f;
    const float c0 = cos(delta0), s0 = sin(delta0);
    const float c1 = cos(delta1), s1 = sin(delta1);
    const float c2 = cos(delta2), s2 = sin(delta2);
    const float c3 = cos(delta3), s3 = sin(delta3);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (uint base = 0; base < data_length; base += TILE) {
        const uint chunk = min(TILE, data_length - base);
        if (tid < chunk) {
            sdata[tid] = input_data[input_offset + base + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Starting phase at position 'base' for each output
        const float phi0 = delta0 * float(base);
        const float phi1 = delta1 * float(base);
        const float phi2 = delta2 * float(base);
        const float phi3 = delta3 * float(base);
        float cs0 = cos(phi0), sn0 = sin(phi0);
        float cs1 = cos(phi1), sn1 = sin(phi1);
        float cs2 = cos(phi2), sn2 = sin(phi2);
        float cs3 = cos(phi3), sn3 = sin(phi3);

        // Accumulate chunk using recurrence
        for (uint j = 0; j < chunk; ++j) {
            const float xj = sdata[j];
            acc0 += xj * cs0;
            acc1 += xj * cs1;
            acc2 += xj * cs2;
            acc3 += xj * cs3;
            // advance phases
            float ncs0 = fma(-sn0, s0, cs0 * c0); float nsn0 = fma(cs0, s0, sn0 * c0); cs0 = ncs0; sn0 = nsn0;
            float ncs1 = fma(-sn1, s1, cs1 * c1); float nsn1 = fma(cs1, s1, sn1 * c1); cs1 = ncs1; sn1 = nsn1;
            float ncs2 = fma(-sn2, s2, cs2 * c2); float nsn2 = fma(cs2, s2, sn2 * c2); cs2 = ncs2; sn2 = nsn2;
            float ncs3 = fma(-sn3, s3, cs3 * c3); float nsn3 = fma(cs3, s3, sn3 * c3); cs3 = ncs3; sn3 = nsn3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store up to 4 outputs
    const uint base_out = b * pattern_dimension + i0;
    output_patterns[base_out + 0u] = acc0 / denom;
    if (i0 + 1u < pattern_dimension) output_patterns[base_out + 1u] = acc1 / denom;
    if (i0 + 2u < pattern_dimension) output_patterns[base_out + 2u] = acc2 / denom;
    if (i0 + 3u < pattern_dimension) output_patterns[base_out + 3u] = acc3 / denom;
}

// Enhanced vector addition for validation
kernel void enhanced_vector_add(
    device const float* in_a [[ buffer(0) ]],
    device const float* in_b [[ buffer(1) ]],
    device float* out_c       [[ buffer(2) ]],
    constant uint& n         [[ buffer(3) ]],
    uint gid                 [[ thread_position_in_grid ]])
{
    if (gid >= n) return;
    
    // Enhanced with bounds checking and optimization
    uint idx = min(gid, n - 1);
    out_c[gid] = in_a[idx] + in_b[idx];
}

// FFT-optimized batch encode (naive DFT form; placeholder for MPS-backed path)
kernel void batch_holographic_encode_fft(
    device const float* input_data       [[ buffer(0) ]],
    device float*       encoded_patterns [[ buffer(1) ]],
    constant uint&      batch_size       [[ buffer(2) ]],
    constant uint&      data_length      [[ buffer(3) ]],
    constant uint&      pattern_dimension [[ buffer(4) ]],
    uint gid                              [[ thread_position_in_grid ]])
{
    if (gid >= batch_size) return;
    const uint in_off = gid * data_length;
    const uint out_off = gid * pattern_dimension;
    // Compute projections with phase recurrence (FFT-like efficiency)
    for (uint i = 0; i < pattern_dimension; ++i) {
        float acc = 0.0f;
        const float delta = 2.0f * M_PI_F * float(i) / float(pattern_dimension);
        float cs = 1.0f; // cos(0)
        float sn = 0.0f; // sin(0)
        const float c = cos(delta);
        const float s = sin(delta);
        const uint limit = min(data_length, pattern_dimension);
        for (uint j = 0; j < limit; ++j) {
            acc += input_data[in_off + j] * cs;
            float ncs = fma(-sn, s, cs * c);
            float nsn = fma(cs, s, sn * c);
            cs = ncs; sn = nsn;
        }
        encoded_patterns[out_off + i] = acc / float(pattern_dimension);
    }
}

// Device-side scalar reductions (naive single-thread kernels)
kernel void dot_norm_kernel(
    device const float* a [[ buffer(0) ]],
    device const float* b [[ buffer(1) ]],
    device float* out_dot [[ buffer(2) ]],
    device float* out_n1  [[ buffer(3) ]],
    device float* out_n2  [[ buffer(4) ]],
    constant uint& n      [[ buffer(5) ]],
    constant uint& tg_sz  [[ buffer(6) ]],
    uint tid              [[ thread_index_in_threadgroup ]])
{
    const uint tg = tg_sz; // actual threads per threadgroup
    const uint SIMD = 32; // lanes per simdgroup (typically 32)
    const uint sidx = tid / SIMD;     // simdgroup index within threadgroup
    const uint sg_max = (tg + SIMD - 1u) / SIMD;
    threadgroup float sg_dot[16];
    threadgroup float sg_n1[16];
    threadgroup float sg_n2[16];

    float local_dot = 0.0f;
    float local_n1  = 0.0f;
    float local_n2  = 0.0f;

    // Strided accumulation across full threadgroup
    for (uint i = tid; i < n; i += tg) {
        float x = a[i];
        float y = b[i];
        local_dot += x * y;
        local_n1  += x * x;
        local_n2  += y * y;
    }

    // Reduce within simdgroup
    float sum_dot = simd_sum(local_dot);
    float sum_n1  = simd_sum(local_n1);
    float sum_n2  = simd_sum(local_n2);

    if (simd_is_first()) {
        if (sidx < 16) {
            sg_dot[sidx] = sum_dot;
            sg_n1[sidx]  = sum_n1;
            sg_n2[sidx]  = sum_n2;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup aggregates across simdgroups
    if (sidx == 0) {
        float acc_dot = 0.0f, acc_n1 = 0.0f, acc_n2 = 0.0f;
        for (uint k = 0; k < min(sg_max, 16u); ++k) {
            acc_dot += sg_dot[k];
            acc_n1  += sg_n1[k];
            acc_n2  += sg_n2[k];
        }
        // Broadcast via shared memory slot 0
        if (tid == 0) {
            out_dot[0] = acc_dot;
            out_n1[0]  = acc_n1;
            out_n2[0]  = acc_n2;
        }
    }
}

kernel void correlation_offset_kernel(
    device const float* a [[ buffer(0) ]],
    device const float* b [[ buffer(1) ]],
    device float* out_val [[ buffer(2) ]],
    constant uint& n      [[ buffer(3) ]],
    constant uint& o1     [[ buffer(4) ]],
    constant uint& o2     [[ buffer(5) ]],
    constant uint& tg_sz  [[ buffer(6) ]],
    uint tid              [[ thread_index_in_threadgroup ]])
{
    const uint tg = tg_sz;
    const uint SIMD = 32;
    const uint sidx = tid / SIMD;
    const uint sg_max = (tg + SIMD - 1u) / SIMD;
    threadgroup float sg_sum[16];

    float local = 0.0f;
    for (uint i = tid; i < n; i += tg) {
        float x = a[(i + o1) % n];
        float y = b[(i + o2) % n];
        local += x * y;
    }
    float ssum = simd_sum(local);
    if (simd_is_first()) {
        if (sidx < 16) sg_sum[sidx] = ssum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sidx == 0 && tid == 0) {
        float acc = 0.0f;
        for (uint k = 0; k < min(sg_max, 16u); ++k) acc += sg_sum[k];
        out_val[0] = acc / (float)n;
    }
}
