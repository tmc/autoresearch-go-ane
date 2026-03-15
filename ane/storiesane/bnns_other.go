//go:build !darwin || !cgo

package storiesane

// BNNSLinearFP16 is not available on non-darwin platforms.
func BNNSLinearFP16(out []float32, weights []uint16, x []float32, outDim, inDim int) bool {
	return false
}
