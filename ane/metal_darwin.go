//go:build darwin && cgo

package ane

/*
#cgo darwin LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation
extern int metalGEMV_init(int maxOutDim, int maxInDim);
extern int metalGEMV_exec(float *out, const float *weights, const float *x, int outDim, int inDim);
extern void metalGEMV_cleanup(void);

#include <stdint.h>
typedef struct MetalFP16Handle MetalFP16Handle;
extern MetalFP16Handle* metalFP16GemvCreate(const uint16_t *weights_f16, int outDim, int inDim);
extern int metalFP16GemvExec(MetalFP16Handle *h, float *out, const float *x);
extern void metalFP16GemvDestroy(MetalFP16Handle *h);
*/
import "C"

import (
	"sync"
	"unsafe"
)

var metalOnce sync.Once
var metalReady bool

// metalClassifierInit lazily initializes the Metal device.
func metalClassifierInit(outDim, inDim int) bool {
	metalOnce.Do(func() {
		if C.metalGEMV_init(C.int(outDim), C.int(inDim)) == 0 {
			metalReady = true
		}
	})
	return metalReady
}

// MetalFP16Gemv holds a cached Metal compute shader for fp16 GEMV.
type MetalFP16Gemv struct {
	handle *C.MetalFP16Handle
}

// NewMetalFP16Gemv creates a cached fp16 GEMV on GPU.
func NewMetalFP16Gemv(weights []uint16, outDim, inDim int) *MetalFP16Gemv {
	if !metalClassifierInit(outDim, inDim) {
		return nil
	}
	if len(weights) < outDim*inDim {
		return nil
	}
	h := C.metalFP16GemvCreate((*C.uint16_t)(unsafe.Pointer(&weights[0])), C.int(outDim), C.int(inDim))
	if h == nil {
		return nil
	}
	return &MetalFP16Gemv{handle: h}
}

// Exec runs the fp16 GEMV: out = weights_fp16 @ x_fp32.
func (m *MetalFP16Gemv) Exec(out, x []float32) bool {
	if m == nil || m.handle == nil {
		return false
	}
	return C.metalFP16GemvExec(m.handle, (*C.float)(unsafe.Pointer(&out[0])), (*C.float)(unsafe.Pointer(&x[0]))) == 0
}

// Close releases GPU resources.
func (m *MetalFP16Gemv) Close() {
	if m != nil && m.handle != nil {
		C.metalFP16GemvDestroy(m.handle)
		m.handle = nil
	}
}

// MetalLinearSingle runs a matrix-vector multiply on the GPU via MPS.
// Returns false if Metal is not available or fails.
func MetalLinearSingle(out []float32, weights []float32, x []float32, outDim, inDim int) bool {
	if !metalClassifierInit(outDim, inDim) {
		return false
	}
	if len(out) < outDim || len(weights) < outDim*inDim || len(x) < inDim {
		return false
	}
	status := C.metalGEMV_exec(
		(*C.float)(unsafe.Pointer(&out[0])),
		(*C.float)(unsafe.Pointer(&weights[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		C.int(outDim),
		C.int(inDim),
	)
	return status == 0
}
