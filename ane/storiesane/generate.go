package storiesane

import (
	"fmt"
	"math/rand"

	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

// GenerateCallback is called after each generated token.
// Return false to stop generation early.
type GenerateCallback func(token uint16, step int) bool

// Generate runs autoregressive decoding using repeated full-sequence EvalLogits
// calls (no KV cache). It returns the full token sequence (prompt + generated).
//
// The engine has a fixed sequence length (e.seq). The prompt is placed at the
// right side of the window (left-padded with BOS). New tokens are appended,
// shifting the window as needed.
func Generate(e *Engine, prompt []uint16, opts stories.GenerateOptions) ([]uint16, error) {
	var result []uint16
	err := GenerateStream(e, prompt, opts, func(token uint16, step int) bool {
		result = append(result, token)
		return true
	})
	if err != nil {
		return nil, err
	}
	// Prepend prompt to match expected output (prompt + generated).
	out := make([]uint16, 0, len(prompt)+len(result))
	out = append(out, prompt...)
	out = append(out, result...)
	return out, nil
}

// GenerateStream runs autoregressive decoding and calls cb after each new token.
// If cb returns false, generation stops early. This uses full-sequence
// recomputation at each step (O(n^2)) — correct but not optimized.
func GenerateStream(e *Engine, prompt []uint16, opts stories.GenerateOptions, cb GenerateCallback) error {
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
	stopSet := make(map[uint16]bool, len(opts.StopTokens))
	for _, t := range opts.StopTokens {
		stopSet[t] = true
	}

	// Seed the RNG. Use the engine seed for reproducibility.
	rng := rand.New(rand.NewSource(e.seed))

	// Working token buffer: all tokens generated so far (prompt + new).
	tokens := make([]uint16, len(prompt))
	copy(tokens, prompt)
	if len(tokens) == 0 {
		tokens = []uint16{storiesBOS}
	}

	// Truncate prompt if it already exceeds sequence length.
	if len(tokens) > seq {
		tokens = tokens[len(tokens)-seq:]
	}

	window := make([]uint16, seq)
	row := make([]float32, vocab)

	for step := 0; step < maxTokens; step++ {
		// Fill the fixed-size window: left-pad with BOS, right-align tokens.
		fillWindow(window, tokens)

		logits, err := e.EvalLogits(window)
		if err != nil {
			return fmt.Errorf("storiesane generate: step %d: %w", step, err)
		}

		// Extract logits for the last position in the window.
		// Layout: logits[v*seq + p] for vocab item v at position p.
		lastPos := seq - 1
		for v := 0; v < vocab; v++ {
			row[v] = logits[v*seq+lastPos]
		}

		next := stories.SampleToken(row, vocab, temp, opts.TopP, rng)

		// Check stop tokens before appending.
		if stopSet[next] {
			// Optionally report the stop token.
			if cb != nil {
				cb(next, step)
			}
			return nil
		}

		tokens = append(tokens, next)

		// Keep the window from exceeding seq length.
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
func fillWindow(dst []uint16, tokens []uint16) {
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
