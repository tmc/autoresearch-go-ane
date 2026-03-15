package ane

// DynamicStatus reports whether the runtime is using dynamic ANE kernels.
//
// Dynamic kernels keep weights in input surfaces and refresh them in place
// after optimizer updates, instead of recompiling per step.
type DynamicStatus struct {
	UseANERequested           bool
	HybridBackwardRequested   bool
	LayerForwardDynamic       bool
	LayerBackwardDynamic      bool
	ClassifierForwardDynamic  bool
	ClassifierBackwardDynamic bool
	ClassifierForwardStatic   bool
	ClassifierBackwardStatic  bool
	LayerInitError            string
	BackwardInitError         string
}

// CoreDynamic reports whether transformer layer forward/backward paths are
// running in dynamic mode.
func (s DynamicStatus) CoreDynamic() bool {
	if !s.LayerForwardDynamic {
		return false
	}
	if s.HybridBackwardRequested && !s.LayerBackwardDynamic {
		return false
	}
	return true
}

// FullyDynamic reports whether both core layers and classifier head paths are
// dynamic with no static classifier fallback kernels.
func (s DynamicStatus) FullyDynamic() bool {
	if !s.CoreDynamic() {
		return false
	}
	if !s.ClassifierForwardDynamic || !s.ClassifierBackwardDynamic {
		return false
	}
	if s.ClassifierForwardStatic || s.ClassifierBackwardStatic {
		return false
	}
	return true
}
