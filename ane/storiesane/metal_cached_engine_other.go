//go:build !darwin || !cgo

package storiesane

func (e *Engine) ensureMetalMatmuls() bool                                      { return false }
func (e *Engine) cleanupMetalMatmuls()                                          {}
func (e *Engine) metalExecQKV(li int, q, k, v, xNorm []float32)                {}
func (e *Engine) metalExecWo(li int, out, attOut []float32)                     {}
func (e *Engine) metalExecFFN(li int, h1, h3, ffOut, xNorm, gate []float32)     {}
func (e *Engine) metalExecClassifier(logits, xNorm []float32) bool              { return false }
