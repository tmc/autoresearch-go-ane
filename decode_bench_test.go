//go:build darwin

package main

import "testing"

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
