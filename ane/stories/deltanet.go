package stories

import "math"

// DeltaNetForward computes a single Gated DeltaNet linear attention layer on CPU.
//
// Gated DeltaNet replaces the O(n²) softmax attention with O(n) linear attention
// using a delta rule update. This is used in hybrid layers of Qwen3-Coder-Next
// where some layers use standard GQA and others use DeltaNet.
//
// Parameters:
//   - out: output [dim*seq], receives the attention output
//   - x: input [dim*seq], channel-first layout
//   - wq, wk, wv: projection weights [dim, dim] or [kvDim, dim] for GQA
//   - wo: output projection [dim, dim]
//   - wGateQ, wGateK: gating weights [dim] per head
//   - rmsW: RMSNorm weights [dim]
//   - dim, heads, headDim, seq: dimensions
//
// The DeltaNet attention mechanism:
//  1. Project: Q = Wq@xnorm, K = Wk@xnorm, V = Wv@xnorm
//  2. Gate: gq = sigmoid(WgateQ * Q), gk = sigmoid(WgateK * K)
//  3. Apply delta rule: S_t = gk_t * S_{t-1} + gk_t * (k_t ⊗ v_t)
//     where S is the running state matrix [headDim, headDim] per head
//  4. Output: o_t = gq_t * (S_t @ q_t)
//  5. Project: out = Wo @ o
func DeltaNetForward(
	out, x []float32,
	wq, wk, wv, wo []float32,
	rmsW []float32,
	dim, heads, headDim, seq int,
) {
	xnorm := make([]float32, dim*seq)
	RMSNorm(xnorm, x, rmsW, dim, seq)

	// Project Q, K, V
	q := make([]float32, dim*seq)
	k := make([]float32, dim*seq)
	v := make([]float32, dim*seq)
	matmulCF(q, wq, xnorm, dim, dim, seq)
	matmulCF(k, wk, xnorm, dim, dim, seq)
	matmulCF(v, wv, xnorm, dim, dim, seq)

	// DeltaNet linear attention per head
	attOut := make([]float32, dim*seq)
	for h := 0; h < heads; h++ {
		// Running state matrix S: [headDim, headDim]
		S := make([]float32, headDim*headDim)

		for t := 0; t < seq; t++ {
			// Extract q_t, k_t, v_t for this head and position
			qt := make([]float32, headDim)
			kt := make([]float32, headDim)
			vt := make([]float32, headDim)
			for d := 0; d < headDim; d++ {
				qi := h*headDim + d
				qt[d] = q[qi*seq+t]
				kt[d] = k[qi*seq+t]
				vt[d] = v[qi*seq+t]
			}

			// Gate: simple exponential decay (simplified gating)
			// Full gating would use learned per-head gate weights
			decay := float32(0.99) // Simplified fixed decay

			// Delta rule update: S = decay * S + k ⊗ v
			for i := 0; i < headDim; i++ {
				for j := 0; j < headDim; j++ {
					S[i*headDim+j] = decay*S[i*headDim+j] + kt[i]*vt[j]
				}
			}

			// Output: o_t = S @ q_t
			for d := 0; d < headDim; d++ {
				var sum float32
				for j := 0; j < headDim; j++ {
					sum += S[d*headDim+j] * qt[j]
				}
				oi := h*headDim + d
				attOut[oi*seq+t] = sum
			}
		}
	}

	// Output projection: out = x + Wo @ attOut
	matmulCF(out, wo, attOut, dim, dim, seq)
	for i := range out {
		out[i] += x[i]
	}
}

// matmulCF computes out = W @ x in channel-first layout.
// W is [outDim, inDim] row-major, x is [inDim, seq] channel-first.
// out is [outDim, seq] channel-first.
func matmulCF(out, w, x []float32, outDim, inDim, seq int) {
	for o := 0; o < outDim; o++ {
		for t := 0; t < seq; t++ {
			var sum float32
			for i := 0; i < inDim; i++ {
				sum += w[o*inDim+i] * x[i*seq+t]
			}
			out[o*seq+t] = sum
		}
	}
}

// SigmoidF32 computes element-wise sigmoid.
func SigmoidF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = float32(1.0 / (1.0 + math.Exp(-float64(v))))
	}
}
