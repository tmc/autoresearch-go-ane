package storiesane

import "math"

func buildRoPETables(seq, headDim int) ([]float32, []float32) {
	if seq <= 0 || headDim <= 0 {
		return nil, nil
	}
	half := headDim / 2
	if half <= 0 {
		return nil, nil
	}
	cos := make([]float32, seq*half)
	sin := make([]float32, seq*half)
	for pos := 0; pos < seq; pos++ {
		base := pos * half
		for i := 0; i < half; i++ {
			freq := float64(pos) / math.Pow(10000, float64(2*i)/float64(headDim))
			cos[base+i] = float32(math.Cos(freq))
			sin[base+i] = float32(math.Sin(freq))
		}
	}
	return cos, sin
}

func applyRoPECFInPlace(x []float32, heads, headDim, seq int, ropeCos, ropeSin []float32) {
	if heads <= 0 || headDim <= 1 || seq <= 0 {
		return
	}
	half := headDim / 2
	for h := 0; h < heads; h++ {
		headBase := h * headDim
		for i := 0; i < half; i++ {
			even := (headBase + 2*i) * seq
			odd := even + seq
			for t := 0; t < seq; t++ {
				c := ropeCos[t*half+i]
				s := ropeSin[t*half+i]
				e := x[even+t]
				o := x[odd+t]
				x[even+t] = e*c - o*s
				x[odd+t] = o*c + e*s
			}
		}
	}
}
