package ane

// Diagnostics reports best-effort runtime state for a storiesane.Engine.
type Diagnostics struct {
	UseANE                  bool
	LayerForwardRequested   bool
	LayerForwardEnabled     bool
	CompiledLayers          int
	LayerInitError          string
	FinalHeadOffloadEnabled bool
	HybridBackwardRequested bool
	HybridBackwardEnabled   bool
	BackwardInitError       string
	HasRMSForward           bool
	HasClassifierForward    bool
	HasSoftmax              bool
	HasClassifierBackward   bool
	HasRMSBackward          bool
	OffloadDiagnostics      string
}

// Diagnostics returns best-effort runtime state for the engine.
func (e *Engine) Diagnostics() Diagnostics {
	if e == nil {
		return Diagnostics{}
	}
	d := Diagnostics{
		UseANE:                  e.useANE,
		LayerForwardRequested:   e.useANE,
		CompiledLayers:          len(e.layers),
		HybridBackwardRequested: e.hybridBackwardRequested,
	}
	if e.layerInitErr != nil {
		d.LayerInitError = e.layerInitErr.Error()
	}
	if e.backwardInitErr != nil {
		d.BackwardInitError = e.backwardInitErr.Error()
	}
	if e.mw != nil && len(e.layers) == len(e.mw.Layers) && len(e.layers) > 0 && e.layerInitErr == nil && !e.layersDirty {
		d.LayerForwardEnabled = true
	}
	if e.mw != nil && len(e.backward) == len(e.mw.Layers) && len(e.backward) > 0 && e.backwardInitErr == nil && !e.backwardDirty {
		d.HybridBackwardEnabled = true
	}
	if e.off != nil {
		d.FinalHeadOffloadEnabled = !e.offDirty && (e.off.hasRMSForward() || e.off.hasClassifierForward() || e.off.hasSoftmax() || e.off.hasClassifierBackward() || e.off.hasRMSBackward())
		d.HasRMSForward = e.off.hasRMSForward()
		d.HasClassifierForward = e.off.hasClassifierForward()
		d.HasSoftmax = e.off.hasSoftmax()
		d.HasClassifierBackward = e.off.hasClassifierBackward()
		d.HasRMSBackward = e.off.hasRMSBackward()
		d.OffloadDiagnostics = e.off.diagnosticSummary()
	}
	return d
}
