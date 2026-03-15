package storiesane

import (
	"fmt"
	"math/rand"

	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

// GenerateCallback is called after each generated token.
// Return false to stop generation early.
type GenerateCallback func(token int32, step int) bool

// Generate runs autoregressive decoding using KV-cached incremental evaluation.
// It returns the full token sequence (prompt + generated).
//
// The engine has a fixed sequence length (e.seq). The prompt is prefilled
// through the KV cache, then each new token is generated in O(n) time
// instead of O(n*seq) full-sequence recomputation.
func Generate(e *Engine, prompt []int32, opts stories.GenerateOptions) ([]int32, error) {
	var result []int32
	err := GenerateStream(e, prompt, opts, func(token int32, step int) bool {
		result = append(result, token)
		return true
	})
	if err != nil {
		return nil, err
	}
	// Prepend prompt to match expected output (prompt + generated).
	out := make([]int32, 0, len(prompt)+len(result))
	out = append(out, prompt...)
	out = append(out, result...)
	return out, nil
}

// GenerateStream runs autoregressive decoding with KV cache and calls cb
// after each new token. If cb returns false, generation stops early.
//
// The KV cache is reset at the start of each call, then the prompt is
// prefilled. Each subsequent token is generated incrementally.
func GenerateStream(e *Engine, prompt []int32, opts stories.GenerateOptions, cb GenerateCallback) error {
	if e == nil || e.mw == nil {
		return fmt.Errorf("storiesane generate: engine is closed")
	}
	cfg := e.Config()
	seq := e.seq
	vocab := cfg.Vocab
	if vocab <= 0 {
		return fmt.Errorf("storiesane generate: invalid vocab size %d", vocab)
	}

	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 64
	}
	temp := opts.Temperature
	if temp < 0 {
		temp = 0
	}

	// Build the stop-token set for fast lookup.
	stopSet := make(map[int32]bool, len(opts.StopTokens))
	for _, t := range opts.StopTokens {
		stopSet[t] = true
	}

	// Seed the RNG. Use the engine seed for reproducibility.
	rng := rand.New(rand.NewSource(e.seed))

	// Prepare prompt tokens.
	promptToks := make([]int32, len(prompt))
	copy(promptToks, prompt)
	if len(promptToks) == 0 {
		promptToks = []int32{storiesBOS}
	}

	// Truncate prompt if it exceeds sequence length (leave room for at least 1 generated token).
	maxPrompt := seq - 1
	if maxPrompt < 1 {
		maxPrompt = 1
	}
	if len(promptToks) > maxPrompt {
		promptToks = promptToks[len(promptToks)-maxPrompt:]
	}

	// Reset and prefill the KV cache with the prompt.
	e.ResetCache()

	var logits []float32
	var err error
	logits, err = e.EvalPrefill(promptToks)
	if err != nil {
		return fmt.Errorf("storiesane generate: prefill: %w", err)
	}

	row := make([]float32, vocab)

	for step := 0; step < maxTokens; step++ {
		// EvalNextToken returns flat [vocab] logits — copy directly.
		copy(row, logits)

		next := stories.SampleToken(row, vocab, temp, opts.TopP, rng)

		// Check stop tokens before reporting.
		if stopSet[next] {
			if cb != nil {
				cb(next, step)
			}
			return nil
		}

		if cb != nil {
			if !cb(next, step) {
				return nil
			}
		}

		// Check if we have room in the cache for more tokens.
		if e.kvc != nil && e.kvc.pos >= seq {
			// Cache is full; cannot generate more tokens.
			break
		}

		// Evaluate the next token incrementally.
		logits, err = e.EvalNextToken(next)
		if err != nil {
			return fmt.Errorf("storiesane generate: step %d: %w", step, err)
		}
	}

	return nil
}

// GenerateStreamNocache runs autoregressive decoding using full-sequence
// recomputation (no KV cache). This is the legacy O(n^2) path, kept for
// correctness testing.
func GenerateStreamNocache(e *Engine, prompt []int32, opts stories.GenerateOptions, cb GenerateCallback) error {
	if e == nil || e.mw == nil {
		return fmt.Errorf("storiesane generate: engine is closed")
	}
	cfg := e.Config()
	seq := e.seq
	vocab := cfg.Vocab
	if vocab <= 0 {
		return fmt.Errorf("storiesane generate: invalid vocab size %d", vocab)
	}

	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 64
	}
	temp := opts.Temperature
	if temp < 0 {
		temp = 0
	}

	stopSet := make(map[int32]bool, len(opts.StopTokens))
	for _, t := range opts.StopTokens {
		stopSet[t] = true
	}

	rng := rand.New(rand.NewSource(e.seed))

	tokens := make([]int32, len(prompt))
	copy(tokens, prompt)
	if len(tokens) == 0 {
		tokens = []int32{storiesBOS}
	}
	if len(tokens) > seq {
		tokens = tokens[len(tokens)-seq:]
	}

	window := make([]int32, seq)
	row := make([]float32, vocab)

	for step := 0; step < maxTokens; step++ {
		fillWindow(window, tokens)

		logits, err := e.EvalLogits(window)
		if err != nil {
			return fmt.Errorf("storiesane generate: step %d: %w", step, err)
		}

		lastPos := seq - 1
		for v := 0; v < vocab; v++ {
			row[v] = logits[v*seq+lastPos]
		}

		next := stories.SampleToken(row, vocab, temp, opts.TopP, rng)

		if stopSet[next] {
			if cb != nil {
				cb(next, step)
			}
			return nil
		}

		tokens = append(tokens, next)
		if len(tokens) > seq {
			tokens = tokens[len(tokens)-seq:]
		}

		if cb != nil {
			if !cb(next, step) {
				return nil
			}
		}
	}

	return nil
}

// fillWindow places tokens right-aligned in dst, padding the left with BOS.
func fillWindow(dst []int32, tokens []int32) {
	for i := range dst {
		dst[i] = storiesBOS
	}
	src := tokens
	if len(src) > len(dst) {
		src = src[len(src)-len(dst):]
	}
	start := len(dst) - len(src)
	copy(dst[start:], src)
}
