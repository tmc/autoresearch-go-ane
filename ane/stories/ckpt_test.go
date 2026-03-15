package stories

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCheckpointRoundTrip(t *testing.T) {
	cfg := ModelConfig{
		Dim:     64,
		Hidden:  128,
		Heads:   4,
		KVHeads: 4,
		NLayers: 2,
		Vocab:   50,
		Seq:     32,
	}
	mw := NewModelWeightsFromConfig(cfg)
	RandomInit(mw, 42)

	opt := NewOptimStateFromConfig(cfg)
	meta := TrainMeta{
		Step:       10,
		TotalSteps: 100,
		LR:         3e-4,
		Loss:       2.5,
		CumSteps:   10,
		CumBatches: 5,
		AdamT:      5,
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "test.ckpt")

	if err := SaveCheckpoint(path, meta, mw, opt); err != nil {
		t.Fatalf("SaveCheckpoint: %v", err)
	}

	mw2 := NewModelWeightsFromConfig(cfg)
	opt2 := NewOptimStateFromConfig(cfg)
	meta2, err := LoadCheckpoint(path, mw2, opt2)
	if err != nil {
		t.Fatalf("LoadCheckpoint: %v", err)
	}

	if meta2.Step != meta.Step || meta2.LR != meta.LR || meta2.Loss != meta.Loss {
		t.Errorf("meta mismatch: got step=%d lr=%f loss=%f", meta2.Step, meta2.LR, meta2.Loss)
	}

	// Verify some weights match.
	for i := 0; i < len(mw.Layers[0].Wq) && i < 10; i++ {
		if mw.Layers[0].Wq[i] != mw2.Layers[0].Wq[i] {
			t.Errorf("Wq[0][%d]: got %f, want %f", i, mw2.Layers[0].Wq[i], mw.Layers[0].Wq[i])
		}
	}
	for i := 0; i < len(mw.RMSFinal) && i < 10; i++ {
		if mw.RMSFinal[i] != mw2.RMSFinal[i] {
			t.Errorf("RMSFinal[%d]: got %f, want %f", i, mw2.RMSFinal[i], mw.RMSFinal[i])
		}
	}
}

func TestCheckpointRoundTripGQA(t *testing.T) {
	cfg := ModelConfig{
		Dim:     128,
		Hidden:  256,
		Heads:   4,
		KVHeads: 2,
		NLayers: 2,
		Vocab:   50,
		Seq:     32,
	}
	mw := NewModelWeightsFromConfig(cfg)
	RandomInit(mw, 123)

	meta := TrainMeta{Step: 5, TotalSteps: 50, LR: 1e-3}

	dir := t.TempDir()
	path := filepath.Join(dir, "gqa.ckpt")

	if err := SaveCheckpoint(path, meta, mw, nil); err != nil {
		t.Fatalf("SaveCheckpoint GQA: %v", err)
	}

	mw2 := NewModelWeightsFromConfig(cfg)
	meta2, err := LoadCheckpoint(path, mw2, nil)
	if err != nil {
		t.Fatalf("LoadCheckpoint GQA: %v", err)
	}
	if meta2.Step != 5 {
		t.Errorf("step = %d, want 5", meta2.Step)
	}

	// Verify GQA weights (Wk should be smaller).
	kvDim := cfg.KVDim()
	wkSize := kvDim * cfg.Dim
	if len(mw2.Layers[0].Wk) != wkSize {
		t.Errorf("Wk size = %d, want %d", len(mw2.Layers[0].Wk), wkSize)
	}
	for i := 0; i < 10 && i < len(mw.Layers[0].Wk); i++ {
		if mw.Layers[0].Wk[i] != mw2.Layers[0].Wk[i] {
			t.Errorf("Wk[0][%d]: got %f, want %f", i, mw2.Layers[0].Wk[i], mw.Layers[0].Wk[i])
		}
	}
}

func TestCheckpointConfigMismatch(t *testing.T) {
	cfg := ModelConfig{
		Dim:     128,
		Hidden:  256,
		Heads:   4,
		KVHeads: 4,
		NLayers: 2,
		Vocab:   50,
		Seq:     32,
	}
	mw := NewModelWeightsFromConfig(cfg)
	RandomInit(mw, 42)

	dir := t.TempDir()
	path := filepath.Join(dir, "test.ckpt")
	if err := SaveCheckpoint(path, TrainMeta{}, mw, nil); err != nil {
		t.Fatalf("SaveCheckpoint: %v", err)
	}

	// Try to load with a different config — should fail.
	cfg2 := cfg
	cfg2.Dim = 256 // Wrong dim.
	mw2 := NewModelWeightsFromConfig(cfg2)
	_, err := LoadCheckpoint(path, mw2, nil)
	if err == nil {
		t.Error("expected config mismatch error, got nil")
	}
}

func TestCheckpointSize(t *testing.T) {
	cfg := ModelConfig{
		Dim:     64,
		Hidden:  128,
		Heads:   4,
		KVHeads: 4,
		NLayers: 2,
		Vocab:   50,
		Seq:     32,
	}
	mw := NewModelWeightsFromConfig(cfg)
	RandomInit(mw, 42)

	dir := t.TempDir()
	path := filepath.Join(dir, "test.ckpt")
	if err := SaveCheckpoint(path, TrainMeta{}, mw, nil); err != nil {
		t.Fatalf("SaveCheckpoint: %v", err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	expected := CheckpointSize(cfg)
	if int(info.Size()) != expected {
		t.Errorf("file size = %d, CheckpointSize = %d", info.Size(), expected)
	}
}
