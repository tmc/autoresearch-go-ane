//go:build darwin && cgo

package storiesane

/*
#cgo darwin LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation
extern int metalGEMV_init(int maxOutDim, int maxInDim);
extern int metalGEMV_exec(float *out, const float *weights, const float *x, int outDim, int inDim);
extern void metalGEMV_cleanup(void);
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
