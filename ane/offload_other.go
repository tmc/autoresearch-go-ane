//go:build !darwin

package ane

import "github.com/tmc/autoresearch-go-ane/ane/stories"

type offload struct{}

func newOffload(*stories.ModelWeights, int, bool, bool) *offload               { return nil }
func refreshOffload(*offload, *stories.ModelWeights, int, bool, bool) *offload { return nil }
func (o *offload) close()                                                      {}
func (o *offload) hasRMSForward() bool                                         { return false }
func (o *offload) hasClassifierForward() bool                                  { return false }
func (o *offload) hasSoftmax() bool                                            { return false }
func (o *offload) hasClassifierBackward() bool                                 { return false }
func (o *offload) hasRMSBackward() bool                                        { return false }
func (o *offload) disableRMSForward()                                          {}
func (o *offload) disableClassifierForward()                                   {}
func (o *offload) disableSoftmax()                                             {}
func (o *offload) disableClassifierBackward()                                  {}
func (o *offload) disableRMSBackward()                                         {}
func (o *offload) runRMSForward([]float32, []float32) error {
	return nil
}
func (o *offload) runClassifierForward([]float32, []float32) error {
	return nil
}
func (o *offload) runClassifierSoftmax([]float32, []float32) error {
	return nil
}
func (o *offload) runSoftmax([]float32) error {
	return nil
}
func (o *offload) runClassifierBackward([]float32, []float32) error {
	return nil
}
func (o *offload) runRMSBackward([]float32, []float32, []float32) error {
	return nil
}
