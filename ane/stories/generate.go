package stories

import (
	"math"
	"math/rand"
	"sort"
)

// GenerateOptions controls autoregressive token generation.
type GenerateOptions struct {
	MaxTokens  int       // maximum number of tokens to generate (excluding prompt)
	Temperature float32  // sampling temperature; 0 = greedy argmax
	TopP       float32   // nucleus sampling threshold; 0 = disabled
	StopTokens []uint16  // tokens that end generation (e.g., EOS)
}

// SampleToken samples a single token from logits using the specified strategy.
//
// If temp == 0: greedy argmax.
// If topP > 0: nucleus (top-p) sampling after temperature scaling.
// Otherwise: temperature-scaled softmax sampling.
func SampleToken(logits []float32, vocab int, temp float32, topP float32, rng *rand.Rand) uint16 {
	if vocab <= 0 {
		return 0
	}
	if vocab > len(logits) {
		vocab = len(logits)
	}

	// Greedy.
	if temp <= 1e-6 {
		best := 0
		bestVal := logits[0]
		for i := 1; i < vocab; i++ {
			if logits[i] > bestVal {
				bestVal = logits[i]
				best = i
			}
		}
		return uint16(best)
	}

	// Compute temperature-scaled log-probabilities.
	maxVal := logits[0]
	for i := 1; i < vocab; i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
		}
	}
	invT := float64(1.0 / temp)

	// Nucleus sampling.
	if topP > 0 && topP < 1 {
		return sampleNucleus(logits, vocab, maxVal, invT, topP, rng)
	}

	// Plain temperature sampling.
	total := 0.0
	for i := 0; i < vocab; i++ {
		total += math.Exp((float64(logits[i]) - float64(maxVal)) * invT)
	}
	target := rng.Float64() * total
	acc := 0.0
	for i := 0; i < vocab; i++ {
		acc += math.Exp((float64(logits[i]) - float64(maxVal)) * invT)
		if acc >= target {
			return uint16(i)
		}
	}
	return uint16(vocab - 1)
}

// tokenProb pairs a token index with its probability for sorting.
type tokenProb struct {
	idx  int
	prob float64
}

func sampleNucleus(logits []float32, vocab int, maxVal float32, invT float64, topP float32, rng *rand.Rand) uint16 {
	// Build probability distribution.
	probs := make([]tokenProb, vocab)
	total := 0.0
	for i := 0; i < vocab; i++ {
		p := math.Exp((float64(logits[i]) - float64(maxVal)) * invT)
		probs[i] = tokenProb{idx: i, prob: p}
		total += p
	}

	// Normalize.
	invTotal := 1.0 / total
	for i := range probs {
		probs[i].prob *= invTotal
	}

	// Sort descending by probability.
	sort.Slice(probs, func(i, j int) bool {
		return probs[i].prob > probs[j].prob
	})

	// Accumulate until we exceed topP.
	cumulative := 0.0
	cutoff := 0
	for i := range probs {
		cumulative += probs[i].prob
		cutoff = i + 1
		if cumulative >= float64(topP) {
			break
		}
	}

	// Re-normalize the nucleus set and sample.
	nucleusTotal := 0.0
	for i := 0; i < cutoff; i++ {
		nucleusTotal += probs[i].prob
	}
	target := rng.Float64() * nucleusTotal
	acc := 0.0
	for i := 0; i < cutoff; i++ {
		acc += probs[i].prob
		if acc >= target {
			return uint16(probs[i].idx)
		}
	}
	return uint16(probs[cutoff-1].idx)
}

// TopKLogits returns a copy of logits with all entries outside the top-k set to -Inf.
// If k <= 0 or k >= len(logits), the original slice is returned unchanged (no copy).
func TopKLogits(logits []float32, k int) []float32 {
	if k <= 0 || k >= len(logits) {
		return logits
	}

	// Find the k-th largest value.
	// We sort indices by value descending and mask everything below the cutoff.
	n := len(logits)
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return logits[indices[i]] > logits[indices[j]]
	})

	threshold := logits[indices[k-1]]
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		if logits[i] >= threshold {
			out[i] = logits[i]
		} else {
			out[i] = float32(math.Inf(-1))
		}
	}
	return out
}
