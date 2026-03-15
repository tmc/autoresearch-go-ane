//go:build !darwin

package storiesane

import (
	"fmt"
	"time"
)

// TokenTimings holds per-token timing breakdown for EvalNextToken.
type TokenTimings struct {
	Total      time.Duration
	Embed      time.Duration
	LayerTotal time.Duration
	QKV        time.Duration
	RoPE       time.Duration
	Attention  time.Duration
	Wo         time.Duration
	FFN        time.Duration
	Residual   time.Duration
	RMSNorm    time.Duration
	Classifier time.Duration
	FP16Conv   time.Duration
}

// LastTokenTimings returns the per-component timing breakdown from the most recent EvalNextToken call.
func (e *Engine) LastTokenTimings() TokenTimings {
	return TokenTimings{}
}

// ResetCache clears the KV cache (no-op on non-darwin).
func (e *Engine) ResetCache() {}

// EvalNextToken is unavailable on non-darwin platforms.
func (e *Engine) EvalNextToken(token int32) ([]float32, error) {
	return nil, fmt.Errorf("storiesane eval next token: unavailable on this platform")
}

// EvalPrefill is unavailable on non-darwin platforms.
func (e *Engine) EvalPrefill(tokens []int32) ([]float32, error) {
	return nil, fmt.Errorf("storiesane eval prefill: unavailable on this platform")
}

func (e *Engine) cleanupANEExecutors() {}
