//go:build darwin

package storiesane

import "fmt"

// DynamicStatus returns best-effort dynamic-mode state after Prepare.
func (e *Engine) DynamicStatus() DynamicStatus {
	st := DynamicStatus{}
	if e == nil {
		return st
	}
	st.UseANERequested = e.useANE
	st.HybridBackwardRequested = e.hybridBackwardRequested
	st.LayerForwardDynamic = len(e.layers) > 0
	for i := range e.layers {
		if e.layers[i] == nil || !e.layers[i].dynamic {
			st.LayerForwardDynamic = false
			break
		}
	}
	if e.layerInitErr != nil {
		st.LayerInitError = e.layerInitErr.Error()
	}
	if e.hybridBackwardRequested {
		st.LayerBackwardDynamic = len(e.backward) > 0
		for i := range e.backward {
			if e.backward[i] == nil || !e.backward[i].dynamic {
				st.LayerBackwardDynamic = false
				break
			}
		}
		if e.backwardInitErr != nil {
			st.BackwardInitError = e.backwardInitErr.Error()
		}
	}
	if e.off != nil {
		st.ClassifierForwardDynamic = len(e.off.clsFwdDyn) > 0
		st.ClassifierBackwardDynamic = len(e.off.clsBwdDyn) > 0
		st.ClassifierForwardStatic = e.off.clsFwd != nil || len(e.off.clsFwdTil) > 0
		st.ClassifierBackwardStatic = e.off.clsBwd != nil || len(e.off.clsBwdTil) > 0
	}
	return st
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
