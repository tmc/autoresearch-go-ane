//go:build !darwin || !arm64 || !cgo

package storiesane

// linearSingleInt8 computes out = dequant(W_int8) @ x with int8 weights.
// Portable fallback: symmetric quantization w_fp32 = w_int8 * scale (per-row).
func linearSingleInt8(out []float32, data []int8, scales []float32, x []float32, outDim, inDim int) {
	for o := 0; o < outDim; o++ {
		sum := float32(0)
		base := o * inDim
		for d := 0; d < inDim; d++ {
			sum += float32(data[base+d]) * x[d]
		}
		out[o] = sum * scales[o]
	}
}
