package stories

// MoEConfig extends ModelConfig for Mixture-of-Experts architectures.
type MoEConfig struct {
	ModelConfig              // Base transformer config
	NumExperts      int      // Total experts (512 for Qwen3-Coder-Next)
	NumActiveExperts int     // Top-k experts per token (10)
	HasSharedExpert bool     // Whether a shared expert runs for all tokens
	ExpertHidden    int      // Expert FFN hidden dim (smaller than full hidden)
	// Layer types for hybrid attention.
	// "attention" = standard GQA, "deltanet" = gated linear attention
	LayerTypes []string // Per-layer type, len == NLayers
}

// ExpertW1Size returns the gate projection weight count for a single expert: ExpertHidden * Dim.
func (c MoEConfig) ExpertW1Size() int { return c.ExpertHidden * c.Dim }

// ExpertW2Size returns the down projection weight count for a single expert: Dim * ExpertHidden.
func (c MoEConfig) ExpertW2Size() int { return c.Dim * c.ExpertHidden }

// ExpertW3Size returns the up projection weight count for a single expert: ExpertHidden * Dim.
func (c MoEConfig) ExpertW3Size() int { return c.ExpertHidden * c.Dim }

// RouterSize returns the router weight count: NumExperts * Dim.
func (c MoEConfig) RouterSize() int { return c.NumExperts * c.Dim }

// ExpertWeights holds weights for a single MoE expert's FFN.
type ExpertWeights struct {
	W1 []float32 // gate_proj [expertHidden, dim]
	W2 []float32 // down_proj [dim, expertHidden]
	W3 []float32 // up_proj [expertHidden, dim]
}

// MoELayerWeights extends LayerWeights for MoE layers.
type MoELayerWeights struct {
	// Attention weights (same as dense model).
	Wq, Wk, Wv, Wo []float32
	RMSAtt          []float32

	// Router (gating network).
	RouterWeight []float32 // [numExperts, dim]

	// Expert FFNs.
	Experts      []ExpertWeights // len == numExperts
	SharedExpert *ExpertWeights  // nil if no shared expert

	// FFN norm.
	RMSFFN []float32
}

// MoEModelWeights holds all weights for a MoE model.
type MoEModelWeights struct {
	Config   MoEConfig
	Layers   []MoELayerWeights
	RMSFinal []float32
	Embed    []float32 // [vocab, dim] row-major
	SharedCL bool
}

// NewMoEModelWeights allocates MoE weights from config.
func NewMoEModelWeights(cfg MoEConfig) *MoEModelWeights {
	mw := &MoEModelWeights{
		Config:   cfg,
		Layers:   make([]MoELayerWeights, cfg.NLayers),
		RMSFinal: make([]float32, cfg.Dim),
		Embed:    make([]float32, cfg.Vocab*cfg.Dim),
		SharedCL: true,
	}
	for i := range mw.Layers {
		layer := MoELayerWeights{
			Wq:           make([]float32, cfg.WqSize()),
			Wk:           make([]float32, cfg.WkSize()),
			Wv:           make([]float32, cfg.WvSize()),
			Wo:           make([]float32, cfg.WoSize()),
			RMSAtt:       make([]float32, cfg.Dim),
			RouterWeight: make([]float32, cfg.RouterSize()),
			Experts:      make([]ExpertWeights, cfg.NumExperts),
			RMSFFN:       make([]float32, cfg.Dim),
		}
		for e := range layer.Experts {
			layer.Experts[e] = ExpertWeights{
				W1: make([]float32, cfg.ExpertW1Size()),
				W2: make([]float32, cfg.ExpertW2Size()),
				W3: make([]float32, cfg.ExpertW3Size()),
			}
		}
		if cfg.HasSharedExpert {
			layer.SharedExpert = &ExpertWeights{
				W1: make([]float32, cfg.ExpertW1Size()),
				W2: make([]float32, cfg.ExpertW2Size()),
				W3: make([]float32, cfg.ExpertW3Size()),
			}
		}
		mw.Layers[i] = layer
	}
	return mw
}
