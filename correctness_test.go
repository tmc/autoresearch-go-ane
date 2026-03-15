//go:build darwin

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"
)

const goldenLogitsPath = "testdata/golden_logits.bin"

// TestSaveGoldenLogits generates the golden reference logits file.
// Run once to establish the baseline:
//
//	go test -run TestSaveGoldenLogits -v
//
// The file is regenerated only when the CPU reference path changes.
func TestSaveGoldenLogits(t *testing.T) {
	if testEngine == nil {
		t.Skip("engine not initialized")
	}

	window := testTokens[:evalSeqLen]
	logits, err := testEngine.EvalLogits(window)
	if err != nil {
		t.Fatalf("EvalLogits: %v", err)
	}

	f, err := os.Create(goldenLogitsPath)
	if err != nil {
		t.Fatalf("create golden file: %v", err)
	}
	defer f.Close()

	if err := binary.Write(f, binary.LittleEndian, int32(len(logits))); err != nil {
		t.Fatalf("write length: %v", err)
	}
	if err := binary.Write(f, binary.LittleEndian, logits); err != nil {
		t.Fatalf("write logits: %v", err)
	}

	t.Logf("saved %d golden logits to %s", len(logits), goldenLogitsPath)
}

// TestInferenceCorrectness compares EvalLogits output against golden reference.
// Uses fp16-level tolerance (~1e-3 relative error) since ANE operates in fp16.
func TestInferenceCorrectness(t *testing.T) {
	if testEngine == nil {
		t.Skip("engine not initialized")
	}

	// Load golden logits
	golden, err := loadGoldenLogits(goldenLogitsPath)
	if err != nil {
		t.Fatalf("load golden logits: %v (run TestSaveGoldenLogits first)", err)
	}

	// Run inference
	window := testTokens[:evalSeqLen]
	logits, err := testEngine.EvalLogits(window)
	if err != nil {
		t.Fatalf("EvalLogits: %v", err)
	}

	if len(logits) != len(golden) {
		t.Fatalf("logits length mismatch: got %d, want %d", len(logits), len(golden))
	}

	// Compare with fp16 tolerance
	const relTol = 1e-3
	const absTol = 1e-4
	var maxRelErr float64
	var maxAbsErr float64
	var mismatches int

	for i := range logits {
		diff := math.Abs(float64(logits[i]) - float64(golden[i]))
		if diff > maxAbsErr {
			maxAbsErr = diff
		}

		denom := math.Max(math.Abs(float64(golden[i])), 1e-8)
		rel := diff / denom
		if rel > maxRelErr {
			maxRelErr = rel
		}

		// Check: either absolute or relative error must be within tolerance
		if diff > absTol && rel > relTol {
			mismatches++
			if mismatches <= 5 {
				t.Errorf("logit[%d]: got %g, want %g (rel=%g, abs=%g)",
					i, logits[i], golden[i], rel, diff)
			}
		}
	}

	total := len(logits)
	mismatchPct := 100.0 * float64(mismatches) / float64(total)

	t.Logf("correctness: %d/%d match (%.4f%% mismatch), maxRelErr=%.6f, maxAbsErr=%.6f",
		total-mismatches, total, mismatchPct, maxRelErr, maxAbsErr)

	// Allow up to 0.1% mismatches for fp16 rounding
	if mismatchPct > 0.1 {
		t.Fatalf("too many mismatches: %.4f%% > 0.1%% threshold (%d/%d)",
			mismatchPct, mismatches, total)
	}
}

func loadGoldenLogits(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open: %w", err)
	}
	defer f.Close()

	var n int32
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		return nil, fmt.Errorf("read length: %w", err)
	}
	if n <= 0 || n > 100_000_000 {
		return nil, fmt.Errorf("invalid logits count: %d", n)
	}

	logits := make([]float32, n)
	if err := binary.Read(f, binary.LittleEndian, logits); err != nil {
		return nil, fmt.Errorf("read logits: %w", err)
	}
	return logits, nil
}
