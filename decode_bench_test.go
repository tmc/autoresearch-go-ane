//go:build darwin

package main

import (
	"testing"
	"time"

	"github.com/tmc/autoresearch-go-ane/ane"
)

// BenchmarkDecode measures single-token decode throughput using the KV cache.
// This is the autoregressive "tokens out" loop: each iteration generates one
// token via EvalNextToken, reusing cached K/V from previous positions.
func BenchmarkDecode(b *testing.B) {
	if testEngine == nil {
		b.Skip("engine not initialized")
	}
	// Reset KV cache and prefill with a few tokens as prompt.
	testEngine.ResetCache()
	prompt := testTokens[:8]
	for _, tok := range prompt {
		if _, err := testEngine.EvalNextToken(tok); err != nil {
			b.Fatalf("prefill: %v", err)
		}
	}

	// Warmup decode.
	for range 3 {
		if _, err := testEngine.EvalNextToken(prompt[0]); err != nil {
			b.Fatalf("warmup: %v", err)
		}
	}

	b.ResetTimer()
	for b.Loop() {
		if _, err := testEngine.EvalNextToken(prompt[0]); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()

	timings := testEngine.LastTokenTimings()
	b.ReportMetric(float64(timings.Total.Milliseconds()), "ms/token")
	b.ReportMetric(float64(timings.QKV.Milliseconds()), "qkv_ms")
	b.ReportMetric(float64(timings.Attention.Milliseconds()), "attn_ms")
	b.ReportMetric(float64(timings.Wo.Milliseconds()), "wo_ms")
	b.ReportMetric(float64(timings.FFN.Milliseconds()), "ffn_ms")
	b.ReportMetric(float64(timings.Classifier.Milliseconds()), "cls_ms")
	b.ReportMetric(1e9/float64(timings.Total.Nanoseconds()), "tokens_per_sec")
}

// BenchmarkMPSGraphDecode measures MPSGraph compiled decode throughput.
// This is a MATMUL-ONLY benchmark (no RoPE/attention/KV cache yet).
// It measures how fast the compiled GPU graph can process the entire
// transformer's matmul chain in a single dispatch.
func BenchmarkMPSGraphDecode(b *testing.B) {
	if testEngine == nil {
		b.Skip("engine not initialized")
	}
	cfg := testEngine.Config()
	dim := cfg.Dim
	vocab := cfg.Vocab

	b.Log("compiling MPSGraph transformer (this takes a while)...")
	start := time.Now()
	decoder, err := ane.NewMPSGraphDecoder(cfg, testEngine.Weights())
	if err != nil {
		b.Skipf("MPSGraph compilation failed: %v", err)
	}
	defer decoder.Close()
	b.Logf("MPSGraph compiled in %v", time.Since(start))

	x := make([]float32, dim)
	logits := make([]float32, vocab)
	// Fill x with test data.
	for i := range x {
		x[i] = 0.01
	}

	// Warmup
	for range 3 {
		if err := decoder.Exec(logits, x); err != nil {
			b.Fatalf("warmup: %v", err)
		}
	}

	b.ResetTimer()
	for b.Loop() {
		if err := decoder.Exec(logits, x); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}
