//go:build !darwin

package ane

import "fmt"

// DynamicStatus returns zero-value dynamic-mode state on non-darwin hosts.
func (e *Engine) DynamicStatus() DynamicStatus {
	_ = e
	return DynamicStatus{}
}

func (s DynamicStatus) String() string {
	return fmt.Sprintf(
		"use_ane=%v hybrid_requested=%v layer_fwd_dynamic=%v layer_bwd_dynamic=%v cls_fwd_dynamic=%v cls_bwd_dynamic=%v cls_fwd_static=%v cls_bwd_static=%v layer_init_error=%q backward_init_error=%q",
		s.UseANERequested,
		s.HybridBackwardRequested,
		s.LayerForwardDynamic,
		s.LayerBackwardDynamic,
		s.ClassifierForwardDynamic,
		s.ClassifierBackwardDynamic,
		s.ClassifierForwardStatic,
		s.ClassifierBackwardStatic,
		s.LayerInitError,
		s.BackwardInitError,
	)
}
