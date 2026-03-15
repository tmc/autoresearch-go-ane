//go:build darwin

package main

import (
	"fmt"
	"os"
	"testing"

	"github.com/tmc/aneperf"
	"github.com/tmc/autoresearch-go-ane/ane"
)

var (
	testEngine  *ane.Engine
	testTokens  []uint16
	testSampler *aneperf.Sampler
)

func TestMain(m *testing.M) {
	dataPath := os.Getenv("DATA")
	if dataPath == "" {
		dataPath = "tinystories_data00.bin"
	}
	if _, err := os.Stat(dataPath); err != nil {
		fmt.Fprintf(os.Stderr, "skipping benchmarks: %s not found (set DATA env var)\n", dataPath)
		os.Exit(0)
	}

	fmt.Fprintf(os.Stderr, "loading tokens from %s...\n", dataPath)
	var err error
	testTokens, err = loadTokens(dataPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load tokens: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "loaded %d tokens\n", len(testTokens))

	// Default to pretrained model (fast to load). Set MODEL=randinit to
	// create a random-init checkpoint instead (slow: writes ~600MB).
	modelPath := os.Getenv("MODEL")
	if modelPath == "" {
		modelPath = "stories110M.bin"
	}
	if modelPath == "randinit" {
		modelPath = "stories110M_randinit.bin"
		fmt.Fprintf(os.Stderr, "creating random-init checkpoint (this takes a few minutes)...\n")
		if err := ensureRandomInitModel(modelPath, Seed); err != nil {
			fmt.Fprintf(os.Stderr, "random init: %v\n", err)
			os.Exit(1)
		}
	}

	fmt.Fprintf(os.Stderr, "opening engine with %s...\n", modelPath)
	opts := experimentConfig(modelPath, testTokens)
	testEngine, err = ane.Open(opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open engine: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "engine ready\n")

	testSampler, err = aneperf.NewSampler()
	if err != nil {
		fmt.Fprintf(os.Stderr, "aneperf sampler: %v\n", err)
		// Non-fatal: benchmarks still run, just without ANE metrics.
	}

	code := m.Run()
	testEngine.Close()
	if testSampler != nil {
		testSampler.Close()
	}
	os.Exit(code)
}

// BenchmarkStep measures training step throughput. Each iteration is one
// gradient accumulation cycle (AccumSteps micro-batches). Reports:
//   - tokens/s:  training throughput
//   - loss:      final training loss of the last iteration
//   - step_ms:   per-step wall time
//   - ane_ms:    ANE eval time per step
//   - adam_ms:   optimizer time per step
//   - ane-watts, ane-compute-%, etc. from aneperf
func BenchmarkStep(b *testing.B) {
	// Warmup: first few steps have compilation overhead.
	for range 3 {
		if _, err := testEngine.Step(); err != nil {
			b.Fatal(err)
		}
	}

	tokensPerStep := int64(SequenceLength) * int64(AccumSteps)
	var lastRes ane.StepResult

	b.SetBytes(tokensPerStep * 2) // uint16 tokens
	b.ResetTimer()
	for b.Loop() {
		snap := startANESample()
		res, err := testEngine.Step()
		if err != nil {
			b.Fatal(err)
		}
		stopANESample(snap, b)
		lastRes = res
	}
	b.StopTimer()

	b.ReportMetric(float64(lastRes.Loss), "loss")
	b.ReportMetric(float64(lastRes.StepDuration.Milliseconds()), "step_ms")
	b.ReportMetric(float64(lastRes.ANEEvalDuration.Milliseconds()), "ane_ms")
	b.ReportMetric(float64(lastRes.AdamDuration.Milliseconds()), "adam_ms")
}

// BenchmarkEvalLogits measures single-window inference throughput. Reports:
//   - tokens/s: inference throughput
//   - ane-watts, ane-compute-%, etc. from aneperf
func BenchmarkEvalLogits(b *testing.B) {
	window := testTokens[:evalSeqLen]

	// Warmup.
	if _, err := testEngine.EvalLogits(window); err != nil {
		b.Fatal(err)
	}

	b.SetBytes(int64(evalSeqLen) * 2)
	b.ResetTimer()
	for b.Loop() {
		snap := startANESample()
		if _, err := testEngine.EvalLogits(window); err != nil {
			b.Fatal(err)
		}
		stopANESample(snap, b)
	}
}

// BenchmarkEvalLoss measures the full validation pass over evalTokens tokens.
// Reports:
//   - val_loss: cross-entropy in nats
//   - windows:  number of eval windows processed
//   - ane-watts, ane-compute-%, etc. from aneperf
func BenchmarkEvalLoss(b *testing.B) {
	windows := (evalTokens - 1) / (evalSeqLen - 1)
	for b.Loop() {
		snap := startANESample()
		loss, err := evalLoss(testEngine, testTokens)
		if err != nil {
			b.Fatal(err)
		}
		stopANESample(snap, b)
		b.ReportMetric(loss, "val_loss")
	}
	b.ReportMetric(float64(windows), "windows")
}

// BenchmarkLRSchedule validates the LR schedule is cheap (no allocation, fast math).
func BenchmarkLRSchedule(b *testing.B) {
	var i int
	for b.Loop() {
		_ = lrSchedule(float64(i) / 1000.0)
		i++
	}
}

func startANESample() aneperf.Snapshot {
	if testSampler == nil {
		return aneperf.Snapshot{}
	}
	return testSampler.Start()
}

func stopANESample(snap aneperf.Snapshot, b *testing.B) {
	if testSampler == nil {
		return
	}
	delta := testSampler.Stop(snap)
	delta.ReportMetrics(b)
}
