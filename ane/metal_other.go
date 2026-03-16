//go:build !darwin || !cgo

package ane

type MetalFP16Gemv struct{}

func NewMetalFP16Gemv(weights []uint16, outDim, inDim int) *MetalFP16Gemv { return nil }
func (m *MetalFP16Gemv) Exec(out, x []float32) bool                      { return false }
func (m *MetalFP16Gemv) Close()                                           {}

// MetalLinearSingle is a no-op stub on non-darwin platforms.
func MetalLinearSingle(out []float32, weights []float32, x []float32, outDim, inDim int) bool {
	return false
}
