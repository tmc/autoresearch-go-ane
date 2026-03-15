//go:build darwin

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/tmc/autoresearch-go-ane/ane"
	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

// --- token loading ---

func loadTokens(path string) ([]uint16, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokens: %w", err)
	}
	if len(data)%2 != 0 {
		return nil, fmt.Errorf("read tokens: file size %d is not even", len(data))
	}
	n := len(data) / 2
	tokens := make([]uint16, n)
	for i := range tokens {
		tokens[i] = binary.LittleEndian.Uint16(data[i*2:])
	}
	return tokens, nil
}

// --- evaluation ---

const (
	evalTokens = 10_000
	evalSeqLen = 256

	// avgBytesPerToken is the mean UTF-8 byte length per token for the
	// Llama2 32K sentencepiece tokenizer on English text. Used to convert
	// cross-entropy loss (nats/token) to bits per byte (BPB), a
	// vocab-size-independent metric: bpb = loss / ln(2) / avgBytesPerToken.
	avgBytesPerToken = 3.7
)

func evalLoss(engine *ane.Engine, tokens []uint16) (float64, error) {
	if len(tokens) < evalTokens+evalSeqLen {
		return 0, fmt.Errorf("eval: need at least %d tokens, got %d", evalTokens+evalSeqLen, len(tokens))
	}

	valTokens := tokens[len(tokens)-evalTokens:]

	var totalLoss float64
	var totalPositions int

	for pos := 0; pos+evalSeqLen < len(valTokens); pos += evalSeqLen - 1 {
		window := valTokens[pos : pos+evalSeqLen]
		logits, err := engine.EvalLogits(window)
		if err != nil {
			return 0, fmt.Errorf("eval logits at pos %d: %w", pos, err)
		}

		vocabSize := len(logits) / evalSeqLen
		for i := 0; i < evalSeqLen-1; i++ {
			row := logits[i*vocabSize : (i+1)*vocabSize]

			maxVal := float64(math.SmallestNonzeroFloat64)
			for _, v := range row {
				if float64(v) > maxVal {
					maxVal = float64(v)
				}
			}
			var sumExp float64
			for _, v := range row {
				sumExp += math.Exp(float64(v) - maxVal)
			}

			target := window[i+1]
			logProb := float64(row[target]) - maxVal - math.Log(sumExp)
			totalLoss -= logProb
			totalPositions++
		}
	}

	if totalPositions == 0 {
		return 0, fmt.Errorf("eval: no positions evaluated")
	}
	return totalLoss / float64(totalPositions), nil
}

// --- random init ---

func ensureRandomInitModel(path string, seed int64) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	}
	mw := stories.NewModelWeights(stories.Vocab)
	stories.RandomInit(mw, seed)
	opt := stories.NewOptimState(stories.Vocab)
	meta := stories.TrainMeta{Step: 0, LR: LearningRate}
	if err := stories.SaveCheckpoint(path, meta, mw, opt); err != nil {
		return fmt.Errorf("save random init checkpoint: %w", err)
	}
	return nil
}
