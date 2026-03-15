//go:build !darwin || !cgo

package ane

import "math"

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

func scaleIntoAccel(dst, src []float32, scale float32) {
	for i := range dst {
		dst[i] = src[i] * scale
	}
}

func addScaledResidualAccel(dst, base, branch []float32, scale float32) {
	for i := range dst {
		dst[i] = base[i] + scale*branch[i]
	}
}

func siluBackwardAccel(dh1, dh3, dGate, h1, h3 []float32) {
	for i := range dh1 {
		sig := float32(1.0 / (1.0 + math.Exp(float64(-h1[i]))))
		siluGrad := sig * (1 + h1[i]*(1-sig))
		dh1[i] = dGate[i] * h3[i] * siluGrad
		dh3[i] = dGate[i] * (h1[i] * sig)
	}
}

func blendResidualInPlaceAccel(sum, base []float32, scale float32) {
	for i := range sum {
		sum[i] = base[i] + (sum[i]-base[i])*scale
	}
}

func siluMulAccel(gate, h1, h3 []float32) {
	for i := range gate {
		gate[i] = silu32(h1[i]) * h3[i]
	}
}

func linearCFAccelerate(out, weights, x []float32, outCh, inCh, seq int) bool {
	return false
}

func gqaAttentionScoresBLAS(scores, q, k []float32, headDim, seq int, alpha float32) {
	// Fallback: pure Go Q^T @ K with scaling.
	for t := 0; t < seq; t++ {
		for j := 0; j < seq; j++ {
			sum := float32(0)
			for d := 0; d < headDim; d++ {
				sum += q[d*seq+t] * k[d*seq+j]
			}
			scores[t*seq+j] = sum * alpha
		}
	}
}

func gqaAttentionValueBLAS(out, v, probs []float32, headDim, seq int) {
	// Fallback: pure Go V @ probs^T.
	for d := 0; d < headDim; d++ {
		for t := 0; t < seq; t++ {
			sum := float32(0)
			for j := 0; j < seq; j++ {
				sum += v[d*seq+j] * probs[t*seq+j]
			}
			out[d*seq+t] = sum
		}
	}
}

func linearSingleGEMV(out, weights, x []float32, outDim, inDim int) bool {
	return false
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
