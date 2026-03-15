package storiesane

import (
	"fmt"
	"math/rand"

	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

const (
	storiesBOS = 1
	storiesEOS = 2
)

// Decode runs autoregressive sampling using repeated full-sequence EvalLogits calls.
//
// This path is intended for parity and experimentation, not KV-cache-efficient decoding.
func (e *Engine) Decode(tok *stories.Tokenizer, opts stories.DecodeOptions) (stories.DecodeResult, error) {
	if e == nil || e.mw == nil {
		return stories.DecodeResult{}, fmt.Errorf("storiesane decode: engine is closed")
	}
	maxNew := opts.MaxNewTokens
	if maxNew <= 0 {
		maxNew = 64
	}
	temp := opts.Temperature
	if temp < 0 {
		temp = 0
	}
	seed := opts.Seed
	if seed == 0 {
		seed = e.seed
	}
	rng := rand.New(rand.NewSource(seed))

	prompt := append([]int(nil), opts.PromptTokens...)
	if len(prompt) == 0 {
		prompt = []int{storiesBOS}
	}
	if len(prompt) > e.seq {
		prompt = prompt[len(prompt)-e.seq:]
	}
	for i, t := range prompt {
		if t < 0 || t >= e.cfg.Vocab {
			return stories.DecodeResult{}, fmt.Errorf("storiesane decode: prompt token[%d]=%d out of range", i, t)
		}
	}

	tokens := append([]int(nil), prompt...)
	text := ""
	stoppedOnEOS := false
	window := make([]uint16, e.seq)
	for step := 0; step < maxNew; step++ {
		fillDecodeWindow(window, tokens)
		logits, err := e.EvalLogits(window)
		if err != nil {
			return stories.DecodeResult{}, err
		}
		last := len(window) - 1
		row := make([]float32, e.cfg.Vocab)
		for v := 0; v < e.cfg.Vocab; v++ {
			row[v] = logits[v*e.seq+last]
		}
		next := sampleFromLogits(rng, row, temp)
		tokens = append(tokens, next)
		if tok != nil {
			text += tok.Decode(next)
		}
		if next == storiesEOS {
			stoppedOnEOS = true
			break
		}
	}

	return stories.DecodeResult{
		Tokens:       tokens,
		PromptLength: len(prompt),
		Text:         text,
		StoppedOnEOS: stoppedOnEOS,
	}, nil
}

func fillDecodeWindow(dst []uint16, tokens []int) {
	for i := range dst {
		dst[i] = storiesBOS
	}
	if len(tokens) >= len(dst) {
		tokens = tokens[len(tokens)-len(dst):]
	}
	start := len(dst) - len(tokens)
	for i, tok := range tokens {
		dst[start+i] = uint16(tok)
	}
}
