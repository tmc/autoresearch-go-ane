//go:build darwin && cgo

package storiesane

/*
#cgo darwin LDFLAGS: -framework Metal -framework Foundation
#include <stdint.h>
typedef struct MetalGemvHandle MetalGemvHandle;
extern int metalGemvInit(void);
extern MetalGemvHandle* metalGemvCreate(const uint16_t *weights_f16, int outDim, int inDim);
extern int metalGemvExec(MetalGemvHandle *h, float *out, const float *x);
extern void metalGemvDestroy(MetalGemvHandle *h);
*/
import "C"

import (
	"sync"
	"unsafe"
)

// MetalGemvFP16 is a cached Metal compute shader matmul with fp16 weights.
// Uses a custom compute kernel instead of MPS for lower overhead.
type MetalGemvFP16 struct {
	handle *C.MetalGemvHandle
}

var metalGemvOnce sync.Once
var metalGemvReady bool

// NewMetalGemvFP16 creates a cached Metal compute shader matmul.
// weights: [outDim * inDim] uint16 (fp16, row-major).
func NewMetalGemvFP16(weights []uint16, outDim, inDim int) *MetalGemvFP16 {
	metalGemvOnce.Do(func() {
		metalGemvReady = C.metalGemvInit() == 0
	})
	if !metalGemvReady {
		return nil
	}
	if len(weights) < outDim*inDim {
		return nil
	}
	h := C.metalGemvCreate((*C.uint16_t)(unsafe.Pointer(&weights[0])), C.int(outDim), C.int(inDim))
	if h == nil {
		return nil
	}
	return &MetalGemvFP16{handle: h}
}

// Exec runs the fp16 matmul: out = weights_fp16 @ x_fp32.
func (m *MetalGemvFP16) Exec(out, x []float32) bool {
	if m == nil || m.handle == nil {
		return false
	}
	return C.metalGemvExec(m.handle, (*C.float)(unsafe.Pointer(&out[0])), (*C.float)(unsafe.Pointer(&x[0]))) == 0
}

// Close releases GPU resources.
func (m *MetalGemvFP16) Close() {
	if m != nil && m.handle != nil {
		C.metalGemvDestroy(m.handle)
		m.handle = nil
	}
}
