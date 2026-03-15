#include <metal_stdlib>
using namespace metal;

// FP16-weight matrix-vector multiply.
// Reads weights as half (fp16), input/output as float (fp32).
// Each threadgroup computes one output row.
// weights: [outDim, inDim] row-major fp16
// x: [inDim] fp32
// out: [outDim] fp32
kernel void gemv_f16(
    device const half *weights [[buffer(0)]],
    device const float *x      [[buffer(1)]],
    device float *out          [[buffer(2)]],
    constant uint &inDim       [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint row = tid;
    device const half *w_row = weights + (uint64_t)row * inDim;

    float sum = 0.0f;
    for (uint d = 0; d < inDim; d += 4) {
        half4 wv = *((device const half4 *)(w_row + d));
        float4 xv = *((device const float4 *)(x + d));
        sum += dot(float4(wv), xv);
    }
    out[row] = sum;
}
