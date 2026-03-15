//go:build !darwin

package ane

import "math"

// --- vector ops (pure Go) ---

func scaleSliceAccel(v []float32, scale float32) {
	for i := range v {
		v[i] *= scale
	}
}

func addSliceAccel(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

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

// --- gemm ---

func accumLinearGradCFAccelerate(dW, dy, x []float32, outCh, inCh, seq int) bool {
	return false
}

func linearBackwardDXCFAccelerate(dx, w, dy []float32, outCh, inCh, seq int) bool {
	return false
}

func linearBackwardDX3AccumAccelerate(dx []float32, w1, dy1, w2, dy2, w3, dy3 []float32, outCh, inCh, seq int) bool {
	return false
}

func accumLinearGrad3CFAccelerate(dW1, dy1, dW2, dy2, dW3, dy3, x []float32, outCh, inCh, seq int) bool {
	return false
}

// --- linearCF BLAS ---

func linearCFAccelerate(out, weights, in []float32, outCh, inCh, seq int) bool {
	return false
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
