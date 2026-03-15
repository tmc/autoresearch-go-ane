//go:build darwin && cgo

package storiesane

/*
#cgo darwin LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation
typedef struct MetalMatmulHandle MetalMatmulHandle;
extern int metalCachedInit(void);
extern MetalMatmulHandle* metalCachedCreate(const float *weights, int outDim, int inDim);
extern int metalCachedExec(MetalMatmulHandle *h, float *out, const float *x);
extern void metalCachedDestroy(MetalMatmulHandle *h);
*/
import "C"

import (
	"sync"
	"unsafe"
)

// MetalMatmul is a cached Metal GPU matmul handle.
// Weights are pre-staged on the GPU. Only input/output are transferred per call.
type MetalMatmul struct {
	handle *C.MetalMatmulHandle
}

var metalCachedOnce sync.Once

// NewMetalMatmul creates a cached Metal matmul with pre-staged weights.
// weights: [outDim * inDim] fp32 row-major.
func NewMetalMatmul(weights []float32, outDim, inDim int) *MetalMatmul {
	metalCachedOnce.Do(func() { C.metalCachedInit() })
	if len(weights) < outDim*inDim {
		return nil
	}
	h := C.metalCachedCreate((*C.float)(unsafe.Pointer(&weights[0])), C.int(outDim), C.int(inDim))
	if h == nil {
		return nil
	}
	return &MetalMatmul{handle: h}
}

// Exec runs the cached matmul: out = weights @ x.
func (m *MetalMatmul) Exec(out, x []float32) bool {
	if m == nil || m.handle == nil {
		return false
	}
	return C.metalCachedExec(m.handle, (*C.float)(unsafe.Pointer(&out[0])), (*C.float)(unsafe.Pointer(&x[0]))) == 0
}

// Close releases GPU resources.
func (m *MetalMatmul) Close() {
	if m != nil && m.handle != nil {
		C.metalCachedDestroy(m.handle)
		m.handle = nil
	}
}
