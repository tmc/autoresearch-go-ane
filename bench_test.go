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
	}

	code := m.Run()
	testEngine.Close()
	if testSampler != nil {
		testSampler.Close()
	}
	os.Exit(code)
}

// BenchmarkStep measures one training step (AccumSteps micro-batches).
//
// Key metrics:
//   - loss:       training loss
//   - tok/s:      training throughput
//   - step_ms:    wall time per step
//   - ane_ms:     ANE forward eval time
//   - adam_ms:    optimizer time
//   - ane-watts:  ANE power draw
//   - ane-compute-%: ANE utilization
func BenchmarkStep(b *testing.B) {
	for range 3 {
		if _, err := testEngine.Step(); err != nil {
			b.Fatal(err)
		}
	}

	tokensPerStep := int64(SequenceLength) * int64(AccumSteps)
	var lastRes ane.StepResult

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

	stepSec := lastRes.StepDuration.Seconds()
	if stepSec > 0 {
		b.ReportMetric(float64(tokensPerStep)/stepSec, "tok/s")
	}
	b.ReportMetric(float64(lastRes.Loss), "loss")
	b.ReportMetric(float64(lastRes.StepDuration.Milliseconds()), "step_ms")
	b.ReportMetric(float64(lastRes.ANEEvalDuration.Milliseconds()), "ane_ms")
	b.ReportMetric(float64(lastRes.AdamDuration.Milliseconds()), "adam_ms")
}

// BenchmarkEvalLogits measures single-window inference speed.
//
// Key metrics:
//   - tok/s:         inference throughput (256 tokens per call)
//   - ane-watts:     ANE power draw
//   - ane-compute-%: ANE utilization
func BenchmarkEvalLogits(b *testing.B) {
	window := testTokens[:evalSeqLen]

	if _, err := testEngine.EvalLogits(window); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for b.Loop() {
		snap := startANESample()
		if _, err := testEngine.EvalLogits(window); err != nil {
			b.Fatal(err)
		}
		stopANESample(snap, b)
	}
}

// BenchmarkEvalLoss measures the full validation pass.
//
// Key metrics:
//   - val_loss:      cross-entropy in nats (THE optimization target)
//   - ane-watts:     ANE power draw
//   - ane-compute-%: ANE utilization
func BenchmarkEvalLoss(b *testing.B) {
	for b.Loop() {
		snap := startANESample()
		loss, err := evalLoss(testEngine, testTokens)
		if err != nil {
			b.Fatal(err)
		}
		stopANESample(snap, b)
		b.ReportMetric(loss, "val_loss")
	}
}

// BenchmarkLRSchedule validates the LR schedule is cheap.
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
