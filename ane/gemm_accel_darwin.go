//go:build darwin && cgo

package ane

/*
#cgo darwin CFLAGS: -Wno-deprecated-declarations
#cgo darwin LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
*/
import "C"

func accumLinearGradCFAccelerate(dW, dy, x []float32, outCh, inCh, seq int) bool {
	if outCh <= 0 || inCh <= 0 || seq <= 0 {
		return false
	}
	if len(dW) < outCh*inCh || len(dy) < outCh*seq || len(x) < inCh*seq {
		return false
	}
	C.cblas_sgemm(
		C.CblasRowMajor,
		C.CblasNoTrans,
		C.CblasTrans,
		C.int(outCh),
		C.int(inCh),
		C.int(seq),
		C.float(1.0),
		(*C.float)(&dy[0]),
		C.int(seq),
		(*C.float)(&x[0]),
		C.int(seq),
		C.float(1.0),
		(*C.float)(&dW[0]),
		C.int(inCh),
	)
	return true
}

func linearBackwardDXCFAccelerate(dx, w, dy []float32, outCh, inCh, seq int) bool {
	if outCh <= 0 || inCh <= 0 || seq <= 0 {
		return false
	}
	if len(dx) < inCh*seq || len(w) < outCh*inCh || len(dy) < outCh*seq {
		return false
	}
	C.cblas_sgemm(
		C.CblasRowMajor,
		C.CblasTrans,
		C.CblasNoTrans,
		C.int(inCh),
		C.int(seq),
		C.int(outCh),
		C.float(1.0),
		(*C.float)(&w[0]),
		C.int(inCh),
		(*C.float)(&dy[0]),
		C.int(seq),
		C.float(0.0),
		(*C.float)(&dx[0]),
		C.int(seq),
	)
	return true
}

// linearBackwardDX3AccumAccelerate computes dx = w1^T @ dy1 + w2^T @ dy2 + w3^T @ dy3
// in a single CGo crossing, reducing overhead for the QKV backward path.
func linearBackwardDX3AccumAccelerate(dx []float32, w1, dy1, w2, dy2, w3, dy3 []float32, outCh, inCh, seq int) bool {
	if outCh <= 0 || inCh <= 0 || seq <= 0 {
		return false
	}
	n := inCh * seq
	if len(dx) < n {
		return false
	}
	if len(w1) < outCh*inCh || len(dy1) < outCh*seq {
		return false
	}
	if len(w2) < outCh*inCh || len(dy2) < outCh*seq {
		return false
	}
	if len(w3) < outCh*inCh || len(dy3) < outCh*seq {
		return false
	}
	// dx = w1^T @ dy1
	C.cblas_sgemm(
		C.CblasRowMajor, C.CblasTrans, C.CblasNoTrans,
		C.int(inCh), C.int(seq), C.int(outCh),
		C.float(1.0),
		(*C.float)(&w1[0]), C.int(inCh),
		(*C.float)(&dy1[0]), C.int(seq),
		C.float(0.0),
		(*C.float)(&dx[0]), C.int(seq),
	)
	// dx += w2^T @ dy2
	C.cblas_sgemm(
		C.CblasRowMajor, C.CblasTrans, C.CblasNoTrans,
		C.int(inCh), C.int(seq), C.int(outCh),
		C.float(1.0),
		(*C.float)(&w2[0]), C.int(inCh),
		(*C.float)(&dy2[0]), C.int(seq),
		C.float(1.0),
		(*C.float)(&dx[0]), C.int(seq),
	)
	// dx += w3^T @ dy3
	C.cblas_sgemm(
		C.CblasRowMajor, C.CblasTrans, C.CblasNoTrans,
		C.int(inCh), C.int(seq), C.int(outCh),
		C.float(1.0),
		(*C.float)(&w3[0]), C.int(inCh),
		(*C.float)(&dy3[0]), C.int(seq),
		C.float(1.0),
		(*C.float)(&dx[0]), C.int(seq),
	)
	return true
}

// accumLinearGrad3CFAccelerate computes:
//   dW1 += dy1 @ x^T
//   dW2 += dy2 @ x^T
//   dW3 += dy3 @ x^T
// in a single CGo crossing, sharing the common input x.
func accumLinearGrad3CFAccelerate(dW1, dy1, dW2, dy2, dW3, dy3, x []float32, outCh, inCh, seq int) bool {
	if outCh <= 0 || inCh <= 0 || seq <= 0 {
		return false
	}
	if len(x) < inCh*seq {
		return false
	}
	if len(dW1) < outCh*inCh || len(dy1) < outCh*seq {
		return false
	}
	if len(dW2) < outCh*inCh || len(dy2) < outCh*seq {
		return false
	}
	if len(dW3) < outCh*inCh || len(dy3) < outCh*seq {
		return false
	}
	C.cblas_sgemm(
		C.CblasRowMajor, C.CblasNoTrans, C.CblasTrans,
		C.int(outCh), C.int(inCh), C.int(seq),
		C.float(1.0),
		(*C.float)(&dy1[0]), C.int(seq),
		(*C.float)(&x[0]), C.int(seq),
		C.float(1.0),
		(*C.float)(&dW1[0]), C.int(inCh),
	)
	C.cblas_sgemm(
		C.CblasRowMajor, C.CblasNoTrans, C.CblasTrans,
		C.int(outCh), C.int(inCh), C.int(seq),
		C.float(1.0),
		(*C.float)(&dy2[0]), C.int(seq),
		(*C.float)(&x[0]), C.int(seq),
		C.float(1.0),
		(*C.float)(&dW2[0]), C.int(inCh),
	)
	C.cblas_sgemm(
		C.CblasRowMajor, C.CblasNoTrans, C.CblasTrans,
		C.int(outCh), C.int(inCh), C.int(seq),
		C.float(1.0),
		(*C.float)(&dy3[0]), C.int(seq),
		(*C.float)(&x[0]), C.int(seq),
		C.float(1.0),
		(*C.float)(&dW3[0]), C.int(inCh),
	)
	return true
}
