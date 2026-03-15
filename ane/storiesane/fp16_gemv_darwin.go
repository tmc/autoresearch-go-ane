//go:build darwin && arm64 && cgo

package storiesane

/*
#cgo CFLAGS: -O3
#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

// NEON fp16-weight matvec: out[outDim] = W_f16[outDim, inDim] @ x_f32[inDim]
// Reads weights as fp16, input/output as fp32. Accumulates in fp32.
// This halves memory bandwidth compared to fp32 sgemv.
static void neon_gemv_f16w(
    float *out,
    const uint16_t *weights_f16,
    const float *x,
    int outDim, int inDim
) {
    for (int o = 0; o < outDim; o++) {
        const uint16_t *row = weights_f16 + (size_t)o * inDim;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);

        int d = 0;
        // Process 8 elements at a time: load 8 fp16, convert to 2×4 fp32, fma with x
        for (; d + 7 < inDim; d += 8) {
            float16x8_t h = vld1q_f16((const __fp16 *)(row + d));
            float32x4_t w0 = vcvt_f32_f16(vget_low_f16(h));
            float32x4_t w1 = vcvt_f32_f16(vget_high_f16(h));
            float32x4_t x0 = vld1q_f32(x + d);
            float32x4_t x1 = vld1q_f32(x + d + 4);
            acc0 = vfmaq_f32(acc0, w0, x0);
            acc1 = vfmaq_f32(acc1, w1, x1);
        }
        // Process 4 elements
        for (; d + 3 < inDim; d += 4) {
            float16x4_t h = vld1_f16((const __fp16 *)(row + d));
            float32x4_t w0 = vcvt_f32_f16(h);
            float32x4_t x0 = vld1q_f32(x + d);
            acc0 = vfmaq_f32(acc0, w0, x0);
        }

        // Reduce
        float32x4_t sum = vaddq_f32(acc0, acc1);
        float result = vaddvq_f32(sum);
        // Scalar tail
        for (; d < inDim; d++) {
            result += (float)(*(const _Float16 *)(row + d)) * x[d];
        }
        out[o] = result;
    }
}
*/
import "C"

import (
	"sync"
	"unsafe"
)

// linearSingleFP16Weights computes out = W_fp16 @ x with fp16 weights and fp32 I/O.
// Uses NEON intrinsics to read fp16 weights and accumulate in fp32.
// For large outDim, splits into parallel goroutines.
func linearSingleFP16Weights(out []float32, wfp16 []uint16, x []float32, outDim, inDim int) {
	if outDim <= 0 || inDim <= 0 {
		return
	}
	const splitThreshold = 2048
	const nSplit = 4
	if outDim > splitThreshold {
		var wg sync.WaitGroup
		chunk := outDim / nSplit
		for s := 0; s < nSplit; s++ {
			start := s * chunk
			size := chunk
			if s == nSplit-1 {
				size = outDim - start
			}
			wg.Add(1)
			go func(start, size int) {
				defer wg.Done()
				C.neon_gemv_f16w(
					(*C.float)(unsafe.Pointer(&out[start])),
					(*C.uint16_t)(unsafe.Pointer(&wfp16[start*inDim])),
					(*C.float)(unsafe.Pointer(&x[0])),
					C.int(size),
					C.int(inDim),
				)
			}(start, size)
		}
		wg.Wait()
		return
	}
	C.neon_gemv_f16w(
		(*C.float)(unsafe.Pointer(&out[0])),
		(*C.uint16_t)(unsafe.Pointer(&wfp16[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		C.int(outDim),
		C.int(inDim),
	)
}
