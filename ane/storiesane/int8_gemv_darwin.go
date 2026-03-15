//go:build darwin && arm64 && cgo

package storiesane

/*
#cgo CFLAGS: -O3
#include <arm_neon.h>
#include <stdint.h>

// Symmetric int8 quantized matvec: out[o] = scale[o] * dot(w_int8[o,:], x[:])
// Uses NEON int8→fp32 widening + FMA for throughput.
static void int8_gemv_symmetric(
    float *out,
    const int8_t *w_int8,
    const float *scales,
    const float *x,
    int outDim, int inDim
) {
    for (int o = 0; o < outDim; o++) {
        const int8_t *row = w_int8 + (size_t)o * inDim;
        float scale = scales[o];
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);

        int d = 0;
        for (; d + 15 < inDim; d += 16) {
            // Load 16 int8 weights
            int8x16_t w = vld1q_s8(row + d);
            // Widen to int16
            int16x8_t w_lo = vmovl_s8(vget_low_s8(w));
            int16x8_t w_hi = vmovl_s8(vget_high_s8(w));
            // Convert to fp32 and FMA with x
            float32x4_t w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_lo)));
            float32x4_t w1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_lo)));
            float32x4_t w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_hi)));
            float32x4_t w3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_hi)));

            float32x4_t x0 = vld1q_f32(x + d);
            float32x4_t x1 = vld1q_f32(x + d + 4);
            float32x4_t x2 = vld1q_f32(x + d + 8);
            float32x4_t x3 = vld1q_f32(x + d + 12);

            acc0 = vfmaq_f32(acc0, w0, x0);
            acc0 = vfmaq_f32(acc0, w1, x1);
            acc1 = vfmaq_f32(acc1, w2, x2);
            acc1 = vfmaq_f32(acc1, w3, x3);
        }
        for (; d + 3 < inDim; d += 4) {
            int8x8_t w8 = vld1_s8(row + d); // loads 8 but we only use low 4
            int16x8_t w16 = vmovl_s8(w8);
            float32x4_t wf = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
            float32x4_t xf = vld1q_f32(x + d);
            acc0 = vfmaq_f32(acc0, wf, xf);
        }

        float dot = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; d < inDim; d++) {
            dot += (float)row[d] * x[d];
        }
        out[o] = dot * scale;
    }
}
*/
import "C"

import (
	"sync"
	"unsafe"
)

// linearSingleInt8 computes out = dequant(W_int8) @ x with int8 weights.
// Uses symmetric quantization: w_fp32 = w_int8 * scale (per-row).
// For large outDim, splits into parallel goroutines.
func linearSingleInt8(out []float32, data []int8, scales []float32, x []float32, outDim, inDim int) {
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
				C.int8_gemv_symmetric(
					(*C.float)(unsafe.Pointer(&out[start])),
					(*C.int8_t)(unsafe.Pointer(&data[start*inDim])),
					(*C.float)(unsafe.Pointer(&scales[start])),
					(*C.float)(unsafe.Pointer(&x[0])),
					C.int(size), C.int(inDim),
				)
			}(start, size)
		}
		wg.Wait()
		return
	}
	C.int8_gemv_symmetric(
		(*C.float)(unsafe.Pointer(&out[0])),
		(*C.int8_t)(unsafe.Pointer(&data[0])),
		(*C.float)(unsafe.Pointer(&scales[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		C.int(outDim), C.int(inDim),
	)
}
