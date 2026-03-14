//go:build !darwin || !cgo

package storiesane

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
