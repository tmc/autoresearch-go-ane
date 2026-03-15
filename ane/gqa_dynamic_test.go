package ane

import (
	"strings"
	"testing"
)

func TestGenGQASDPAForwardInference_MHA(t *testing.T) {
	// When kvHeads == heads, should fall back to standard MHA path.
	mha := GenGQASDPAForwardInference(768, 12, 12, 256, 0.2)
	std := genStoriesSDPAForwardDynamicInference(768, 12, 256, 0.2)
	if mha != std {
		t.Error("GQA with kvHeads==heads should produce identical MIL to standard MHA")
	}
}

func TestGenGQASDPAForwardInference_GQA(t *testing.T) {
	// Qwen3-4B style: 32 Q heads, 8 KV heads, dim=2560
	mil := GenGQASDPAForwardInference(2560, 32, 8, 64, 0.2)

	if mil == "" {
		t.Fatal("generated MIL is empty")
	}

	// Should contain GQA-specific operations.
	if !strings.Contains(mil, "tile") {
		t.Error("GQA MIL should contain tile op for KV head expansion")
	}

	// Verify input tensor dimensions: kvDim = 8 * 80 = 640
	// spatial = seq + 1 + 2*dim + 2*kvDim = 64 + 1 + 5120 + 1280 = 6465
	if !strings.Contains(mil, "6465") {
		t.Error("GQA MIL should have correct spatial dimension (6465)")
	}

	// Should reference correct head counts.
	if !strings.Contains(mil, "kv_reps") {
		t.Error("GQA MIL should define kv_reps for tiling")
	}

	// KV head expansion factor should be 32/8 = 4.
	if !strings.Contains(mil, "[1,4,1,1]") {
		t.Error("GQA MIL should have tile reps [1,4,1,1] for 32/8 head ratio")
	}
}

func TestGenGQASDPAForwardInference_SmallGQA(t *testing.T) {
	// Small model: 4 Q heads, 2 KV heads, dim=128
	mil := GenGQASDPAForwardInference(128, 4, 2, 32, 0.3)

	if mil == "" {
		t.Fatal("generated MIL is empty")
	}

	// headDim = 128/4 = 32, kvDim = 2*32 = 64
	// tile reps should be [1,2,1,1] for 4/2 head ratio
	if !strings.Contains(mil, "[1,2,1,1]") {
		t.Error("small GQA MIL should have tile reps [1,2,1,1]")
	}

	// Verify it's valid MIL (has opening and closing braces).
	if !strings.HasSuffix(strings.TrimSpace(mil), "}") {
		t.Error("MIL should end with closing brace")
	}
	if !strings.Contains(mil, "func main<ios18>") {
		t.Error("MIL should contain main function")
	}
}

func TestGenGQASDPAForwardInference_KVZero(t *testing.T) {
	// kvHeads=0 should fall back to MHA.
	mil := GenGQASDPAForwardInference(768, 12, 0, 256, 0.2)
	std := genStoriesSDPAForwardDynamicInference(768, 12, 256, 0.2)
	if mil != std {
		t.Error("GQA with kvHeads=0 should produce identical MIL to standard MHA")
	}
}

func TestGenStoriesFFNForwardInference(t *testing.T) {
	mil := genStoriesFFNForwardDynamicInference(768, 2048, 256, 0.2)
	if mil == "" {
		t.Fatal("generated FFN MIL is empty")
	}
	if !strings.Contains(mil, "silu") {
		t.Error("FFN MIL should contain SiLU activation")
	}
	if !strings.Contains(mil, "func main<ios18>") {
		t.Error("FFN MIL should contain main function")
	}
}
