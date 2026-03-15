package stories

import "testing"

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.Dim != Dim {
		t.Errorf("Dim = %d, want %d", cfg.Dim, Dim)
	}
	if cfg.Hidden != Hidden {
		t.Errorf("Hidden = %d, want %d", cfg.Hidden, Hidden)
	}
	if cfg.Heads != Heads {
		t.Errorf("Heads = %d, want %d", cfg.Heads, Heads)
	}
	if cfg.KVHeads != Heads {
		t.Errorf("KVHeads = %d, want %d (MHA)", cfg.KVHeads, Heads)
	}
	if cfg.NLayers != NLayers {
		t.Errorf("NLayers = %d, want %d", cfg.NLayers, NLayers)
	}
	if cfg.Vocab != Vocab {
		t.Errorf("Vocab = %d, want %d", cfg.Vocab, Vocab)
	}
	if cfg.HeadDim() != Dim/Heads {
		t.Errorf("HeadDim = %d, want %d", cfg.HeadDim(), Dim/Heads)
	}
	if cfg.IsGQA() {
		t.Errorf("default config should not be GQA")
	}
	// Verify derived sizes match compile-time constants.
	if cfg.WqSize() != WQSize {
		t.Errorf("WqSize = %d, want %d", cfg.WqSize(), WQSize)
	}
	if cfg.W1Size() != W1Size {
		t.Errorf("W1Size = %d, want %d", cfg.W1Size(), W1Size)
	}
}

func TestQwen3Config(t *testing.T) {
	cfg := ModelConfig{
		Dim:     2560,
		Hidden:  6912,
		Heads:   32,
		KVHeads: 8,
		NLayers: 36,
		Vocab:   151936,
		Seq:     4096,
	}
	if !cfg.IsGQA() {
		t.Error("Qwen3-4B should be GQA")
	}
	if cfg.HeadDim() != 80 { // 2560 / 32
		t.Errorf("HeadDim = %d, want 80", cfg.HeadDim())
	}
	if cfg.KVDim() != 640 { // 8 * 80
		t.Errorf("KVDim = %d, want 640", cfg.KVDim())
	}
	if cfg.WkSize() != 640*2560 {
		t.Errorf("WkSize = %d, want %d", cfg.WkSize(), 640*2560)
	}
	if cfg.WqSize() != 2560*2560 {
		t.Errorf("WqSize = %d, want %d", cfg.WqSize(), 2560*2560)
	}
}

func TestNewModelWeightsFromConfig(t *testing.T) {
	cfg := ModelConfig{
		Dim:     128,
		Hidden:  256,
		Heads:   4,
		KVHeads: 2,
		NLayers: 2,
		Vocab:   100,
		Seq:     32,
	}
	mw := NewModelWeightsFromConfig(cfg)
	if len(mw.Layers) != 2 {
		t.Fatalf("Layers = %d, want 2", len(mw.Layers))
	}
	if len(mw.RMSFinal) != 128 {
		t.Errorf("RMSFinal = %d, want 128", len(mw.RMSFinal))
	}
	if len(mw.Embed) != 100*128 {
		t.Errorf("Embed = %d, want %d", len(mw.Embed), 100*128)
	}
	// GQA: Wk should be kvDim * dim = (2*32) * 128 = 8192
	kvDim := cfg.KVDim() // 2 * 32 = 64
	if len(mw.Layers[0].Wk) != kvDim*128 {
		t.Errorf("Wk = %d, want %d", len(mw.Layers[0].Wk), kvDim*128)
	}
	// Wq should be dim * dim = 128 * 128
	if len(mw.Layers[0].Wq) != 128*128 {
		t.Errorf("Wq = %d, want %d", len(mw.Layers[0].Wq), 128*128)
	}
	if mw.Config.Dim != 128 {
		t.Errorf("Config.Dim = %d, want 128", mw.Config.Dim)
	}
}

func TestConfigFromLlama2(t *testing.T) {
	hdr := Llama2Config{
		Dim:       768,
		HiddenDim: 2048,
		NLayers:   12,
		NHeads:    12,
		NKVHeads:  12,
		VocabSize: 32000,
		SeqLen:    256,
	}
	cfg := ConfigFromLlama2(hdr)
	def := DefaultConfig()
	if cfg.Dim != def.Dim || cfg.Hidden != def.Hidden || cfg.NLayers != def.NLayers {
		t.Errorf("ConfigFromLlama2 mismatch: got %+v", cfg)
	}
	// Negative vocab means non-shared classifier.
	hdr2 := hdr
	hdr2.VocabSize = -32000
	cfg2 := ConfigFromLlama2(hdr2)
	if cfg2.Vocab != 32000 {
		t.Errorf("negative vocab: got %d, want 32000", cfg2.Vocab)
	}
}
