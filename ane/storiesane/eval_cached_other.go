//go:build !darwin

package storiesane

import "fmt"

// ResetCache clears the KV cache (no-op on non-darwin).
func (e *Engine) ResetCache() {}

// EvalNextToken is unavailable on non-darwin platforms.
func (e *Engine) EvalNextToken(token uint16) ([]float32, error) {
	return nil, fmt.Errorf("storiesane eval next token: unavailable on this platform")
}

// EvalPrefill is unavailable on non-darwin platforms.
func (e *Engine) EvalPrefill(tokens []uint16) ([]float32, error) {
	return nil, fmt.Errorf("storiesane eval prefill: unavailable on this platform")
}
