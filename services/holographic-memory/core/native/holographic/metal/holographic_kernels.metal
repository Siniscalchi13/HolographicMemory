#include <metal_stdlib>
using namespace metal;

// Naive DFT (for correctness baseline). For production use MPS FFT.
kernel void holographic_fft(
    device const float* input       [[ buffer(0) ]],
    device float*       real_output [[ buffer(1) ]],
    device float*       imag_output [[ buffer(2) ]],
    constant uint&      length      [[ buffer(3) ]],
    uint gid                         [[ thread_position_in_grid ]])
{
    if (gid >= length) return;
    float real = 0.0f;
    float imag = 0.0f;
    for (uint k = 0; k < length; k++) {
        float angle = -2.0f * 3.1415926535f * float(gid) * float(k) / float(length);
        real += input[k] * cos(angle);
        imag += input[k] * sin(angle);
    }
    real_output[gid] = real / float(length);
    imag_output[gid] = imag / float(length);
}

// Pointwise complex multiplication (wave interference)
kernel void wave_interference(
    device const float* wave1_real [[ buffer(0) ]],
    device const float* wave1_imag [[ buffer(1) ]],
    device const float* wave2_real [[ buffer(2) ]],
    device const float* wave2_imag [[ buffer(3) ]],
    device float*       result_real [[ buffer(4) ]],
    device float*       result_imag [[ buffer(5) ]],
    constant uint&      dimension   [[ buffer(6) ]],
    uint gid                        [[ thread_position_in_grid ]])
{
    if (gid >= dimension) return;
    float a = wave1_real[gid];
    float b = wave1_imag[gid];
    float c = wave2_real[gid];
    float d = wave2_imag[gid];
    result_real[gid] = a * c - b * d;
    result_imag[gid] = a * d + b * c;
}

// Batch encoding: simple cosine projection (placeholder for holographic encode)
kernel void batch_holographic_encode(
    device const float* input_data       [[ buffer(0) ]],
    device float*       encoded_patterns [[ buffer(1) ]],
    constant uint&      batch_size       [[ buffer(2) ]],
    constant uint&      data_length      [[ buffer(3) ]],
    constant uint&      pattern_dim      [[ buffer(4) ]],
    uint gid                              [[ thread_position_in_grid ]])
{
    if (gid >= batch_size) return;
    uint in_off = gid * data_length;
    uint out_off = gid * pattern_dim;
    for (uint i = 0; i < pattern_dim; i++) {
        float acc = 0.0f;
        for (uint j = 0; j < data_length; j++) {
            float phase = 2.0f * 3.1415926535f * float(i) * float(j) / float(pattern_dim);
            acc += input_data[in_off + j] * cos(phase);
        }
        encoded_patterns[out_off + i] = acc / float(data_length);
    }
}

// L2 similarity (1/(1+norm(query - pattern))) per pattern
kernel void holographic_similarity(
    device const float* query            [[ buffer(0) ]],
    device const float* stored_patterns  [[ buffer(1) ]],
    device float*       similarities     [[ buffer(2) ]],
    constant uint&      pattern_count    [[ buffer(3) ]],
    constant uint&      dimension        [[ buffer(4) ]],
    uint gid                              [[ thread_position_in_grid ]])
{
    if (gid >= pattern_count) return;
    uint off = gid * dimension;
    float sumsq = 0.0f;
    for (uint i = 0; i < dimension; i++) {
        float diff = query[i] - stored_patterns[off + i];
        sumsq += diff * diff;
    }
    similarities[gid] = 1.0f / (1.0f + sqrt(sumsq));
}

