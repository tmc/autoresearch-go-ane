package stories

import "math"

// Legacy compile-time constants for the 110M TinyStories model.
// Kept for backward compatibility — existing code that references these
// continues to work unchanged.
const (
	Dim        = 768
	Hidden     = 2048
	Heads      = 12
	SeqDefault = 256
	NLayers    = 12
	Vocab      = 32000
)

// Derived weight sizes for the default 110M config.
const (
	WQSize = Dim * Dim
	WOSize = Dim * Dim
	W1Size = Hidden * Dim
	W2Size = Dim * Hidden
	W3Size = Hidden * Dim
)

// Llama2Config is the on-disk header for .bin pretrained weights.
type Llama2Config struct {
	Dim       int32
	HiddenDim int32
	NLayers   int32
	NHeads    int32
	NKVHeads  int32
	VocabSize int32
	SeqLen    int32
}

// ModelConfig describes an arbitrary Llama-family architecture.
// It replaces the hardcoded constants for models beyond 110M.
type ModelConfig struct {
	Dim     int // hidden_size (embedding dimension)
	Hidden  int // intermediate_size (FFN hidden dimension)
	Heads      int // num_attention_heads
	KVHeads    int // num_key_value_heads (for GQA; 0 means KVHeads == Heads)
	NLayers    int // num_hidden_layers
	Vocab      int // vocab_size
	Seq        int // max sequence length (0 = use SeqDefault)
	HeadDimOvr int // explicit head_dim override (0 = Dim / Heads)
}

// DefaultConfig returns the ModelConfig matching the legacy 110M constants.
func DefaultConfig() ModelConfig {
	return ModelConfig{
		Dim:     Dim,
		Hidden:  Hidden,
		Heads:   Heads,
		KVHeads: Heads, // MHA (no GQA)
		NLayers: NLayers,
		Vocab:   Vocab,
		Seq:     SeqDefault,
	}
}

// EffectiveKVHeads returns KVHeads, defaulting to Heads if zero (MHA).
func (c ModelConfig) EffectiveKVHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// HeadDim returns the per-head dimension.
// Uses HeadDimOvr if set (e.g. Qwen3 where head_dim != dim/heads),
// otherwise falls back to Dim / Heads.
func (c ModelConfig) HeadDim() int {
	if c.HeadDimOvr > 0 {
		return c.HeadDimOvr
	}
	if c.Heads <= 0 {
		return 0
	}
	return c.Dim / c.Heads
}

// QDim returns the total Q projection dimension: Heads * HeadDim.
// This may differ from Dim when HeadDim is explicitly set.
func (c ModelConfig) QDim() int {
	return c.Heads * c.HeadDim()
}

// KVDim returns the total KV projection dimension: KVHeads * HeadDim.
func (c ModelConfig) KVDim() int {
	return c.EffectiveKVHeads() * c.HeadDim()
}

// IsGQA returns true if the model uses grouped-query attention (KVHeads < Heads).
func (c ModelConfig) IsGQA() bool {
	return c.EffectiveKVHeads() < c.Heads
}

// WqSize returns the Q projection weight count: QDim * Dim.
func (c ModelConfig) WqSize() int { return c.QDim() * c.Dim }

// WkSize returns the K projection weight count: KVDim * Dim.
func (c ModelConfig) WkSize() int { return c.KVDim() * c.Dim }

// WvSize returns the V projection weight count: KVDim * Dim.
func (c ModelConfig) WvSize() int { return c.KVDim() * c.Dim }

// WoSize returns the O projection weight count: Dim * QDim.
func (c ModelConfig) WoSize() int { return c.Dim * c.QDim() }

// W1Size returns the FFN gate projection weight count: Hidden * Dim.
func (c ModelConfig) W1Size() int { return c.Hidden * c.Dim }

// W2Size returns the FFN down projection weight count: Dim * Hidden.
func (c ModelConfig) W2Size() int { return c.Dim * c.Hidden }

// W3Size returns the FFN up projection weight count: Hidden * Dim.
func (c ModelConfig) W3Size() int { return c.Hidden * c.Dim }

// ResidualScale returns 1/sqrt(2*NLayers), used for pre-norm residual scaling.
func (c ModelConfig) ResidualScale() float64 {
	if c.NLayers <= 0 {
		return 1.0
	}
	return 1.0 / math.Sqrt(2.0*float64(c.NLayers))
}

// EffectiveSeq returns Seq, defaulting to SeqDefault if zero.
func (c ModelConfig) EffectiveSeq() int {
	if c.Seq <= 0 {
		return SeqDefault
	}
	return c.Seq
}

// ConfigFromLlama2 converts a Llama2Config header to a ModelConfig.
func ConfigFromLlama2(cfg Llama2Config) ModelConfig {
	vocab := int(cfg.VocabSize)
	if vocab < 0 {
		vocab = -vocab
	}
	kvHeads := int(cfg.NKVHeads)
	if kvHeads <= 0 {
		kvHeads = int(cfg.NHeads)
	}
	return ModelConfig{
		Dim:     int(cfg.Dim),
		Hidden:  int(cfg.HiddenDim),
		Heads:   int(cfg.NHeads),
		KVHeads: kvHeads,
		NLayers: int(cfg.NLayers),
		Vocab:   vocab,
		Seq:     int(cfg.SeqLen),
	}
}
