//go:build !darwin || !cgo

package storiesane

// MetalLinearSingle is a no-op stub on non-darwin platforms.
func MetalLinearSingle(out []float32, weights []float32, x []float32, outDim, inDim int) bool {
	return false
}
