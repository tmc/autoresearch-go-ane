//go:build !darwin || !cgo

package storiesane

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
