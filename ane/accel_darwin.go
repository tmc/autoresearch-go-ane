//go:build darwin

package ane

import (
	"math"

	"github.com/tmc/apple/accelerate"
)

// --- vector ops (pure Go, vDSP-equivalent) ---

func scaleSliceAccel(v []float32, scale float32) {
	if len(v) == 0 || scale == 1 {
		return
	}
	for i := range v {
		v[i] *= scale
	}
}

func addSliceAccel(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

// reluSquaredBackwardSimple computes dh1 for gate = relu²(h1).
func reluSquaredBackwardSimple(dh1, dGate, h1 []float32) {
	for i := range dh1 {
		r := h1[i]
		if r < 0 {
			r = 0
		}
		dh1[i] = dGate[i] * 2 * r
	}
}

func softmaxRowAccel(out, in []float32) {
	maxv := in[0]
	for i := 1; i < len(in); i++ {
		if in[i] > maxv {
			maxv = in[i]
		}
	}
	sum := 0.0
	for i := range in {
		e := math.Exp(float64(in[i] - maxv))
		out[i] = float32(e)
		sum += e
	}
	inv := float32(1.0 / sum)
	for i := range out {
		out[i] *= inv
	}
}

// --- adam ---

func adamUpdateCFAccelerateChunk(w, g, m, v []float32, b1, b2, invBC1, invBC2, lr, eps, wd float32) bool {
	return false
}

// --- gemm (via purego Accelerate) ---

func accumLinearGradCFAccelerate(dW, dy, x []float32, outCh, inCh, seq int) bool {
	if outCh <= 0 || inCh <= 0 || seq <= 0 {
		return false
	}
	if len(dW) < outCh*inCh || len(dy) < outCh*seq || len(x) < inCh*seq {
		return false
	}
	// dW += dy @ x^T  (outCh x seq) @ (seq x inCh) = (outCh x inCh)
	accelerate.Cblas_sgemm(
		accelerate.CblasRowMajor,
		accelerate.CblasNoTrans,
		accelerate.CblasTrans,
		outCh, inCh, seq,
		1.0, dy, seq,
		x, seq,
		1.0, dW, inCh,
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
	// dx = w^T @ dy  (inCh x outCh) @ (outCh x seq) = (inCh x seq)
	accelerate.Cblas_sgemm(
		accelerate.CblasRowMajor,
		accelerate.CblasTrans,
		accelerate.CblasNoTrans,
		inCh, seq, outCh,
		1.0, w, inCh,
		dy, seq,
		0.0, dx, seq,
	)
	return true
}

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
	accelerate.Cblas_sgemm(
		accelerate.CblasRowMajor, accelerate.CblasTrans, accelerate.CblasNoTrans,
		inCh, seq, outCh,
		1.0, w1, inCh, dy1, seq,
		0.0, dx, seq,
	)
	// dx += w2^T @ dy2
	accelerate.Cblas_sgemm(
		accelerate.CblasRowMajor, accelerate.CblasTrans, accelerate.CblasNoTrans,
		inCh, seq, outCh,
		1.0, w2, inCh, dy2, seq,
		1.0, dx, seq,
	)
	// dx += w3^T @ dy3
	accelerate.Cblas_sgemm(
		accelerate.CblasRowMajor, accelerate.CblasTrans, accelerate.CblasNoTrans,
		inCh, seq, outCh,
		1.0, w3, inCh, dy3, seq,
		1.0, dx, seq,
	)
	return true
}

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
	accelerate.Cblas_sgemm(
		accelerate.CblasRowMajor, accelerate.CblasNoTrans, accelerate.CblasTrans,
		outCh, inCh, seq,
		1.0, dy1, seq, x, seq,
		1.0, dW1, inCh,
	)
	accelerate.Cblas_sgemm(
		accelerate.CblasRowMajor, accelerate.CblasNoTrans, accelerate.CblasTrans,
		outCh, inCh, seq,
		1.0, dy2, seq, x, seq,
		1.0, dW2, inCh,
	)
	accelerate.Cblas_sgemm(
		accelerate.CblasRowMajor, accelerate.CblasNoTrans, accelerate.CblasTrans,
		outCh, inCh, seq,
		1.0, dy3, seq, x, seq,
		1.0, dW3, inCh,
	)
	return true
}

// --- residual blending (saxpy) ---

// scaleAddAccel computes out[i] += scale * src[i] for i in [0, n).
func scaleAddAccel(out, src []float32, scale float32, n int) bool {
	if n <= 0 || len(out) < n || len(src) < n {
		return false
	}
	accelerate.Cblas_saxpy(n, scale, src, 1, out, 1)
	return true
}

// --- linearCF BLAS ---

func linearCFAccelerate(out, weights, in []float32, outCh, inCh, seq int) bool {
	if outCh <= 0 || inCh <= 0 || seq <= 0 {
		return false
	}
	if len(out) < outCh*seq || len(weights) < outCh*inCh || len(in) < inCh*seq {
		return false
	}
	// out = weights @ in  (outCh x inCh) @ (inCh x seq) = (outCh x seq)
	// weights is row-major [outCh, inCh], in is channel-first [inCh, seq],
	// out is channel-first [outCh, seq].
	accelerate.Cblas_sgemm(
		accelerate.CblasRowMajor,
		accelerate.CblasNoTrans,
		accelerate.CblasNoTrans,
		outCh, seq, inCh,
		1.0, weights, inCh,
		in, seq,
		0.0, out, seq,
	)
	return true
}

// --- grad ops ---

func sumSquaresGrad(v []float32) float64 {
	var s float64
	for _, x := range v {
		f := float64(x)
		s += f * f
	}
	return s
}

func scaleGradSlice(v []float32, scale float32) {
	for i := range v {
		v[i] *= scale
	}
}
