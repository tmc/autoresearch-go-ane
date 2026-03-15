//go:build !darwin || !cgo

package ane

func adamUpdateCFAccelerateChunk(w, g, m, v []float32, b1, b2, invBC1, invBC2, lr, eps, wd float32) bool {
	return false
}
