//go:build darwin && cgo

package storiesane

/*
#cgo darwin CFLAGS: -Wno-deprecated-declarations
#cgo darwin LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
*/
import "C"

import "unsafe"

func sumSquaresGrad(v []float32) float64 {
	if len(v) == 0 {
		return 0
	}
	var s C.float
	C.vDSP_dotpr(
		(*C.float)(unsafe.Pointer(&v[0])),
		1,
		(*C.float)(unsafe.Pointer(&v[0])),
		1,
		&s,
		C.vDSP_Length(len(v)),
	)
	return float64(s)
}

func scaleGradSlice(v []float32, scale float32) {
	if len(v) == 0 || scale == 1 {
		return
	}
	C.vDSP_vsmul(
		(*C.float)(unsafe.Pointer(&v[0])),
		1,
		(*C.float)(unsafe.Pointer(&scale)),
		(*C.float)(unsafe.Pointer(&v[0])),
		1,
		C.vDSP_Length(len(v)),
	)
}
