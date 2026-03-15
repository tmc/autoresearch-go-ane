package ane

import (
	"math"
	"runtime"
	"sync"

	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

// --- cross-entropy loss ---

func crossEntropyLossFromProbs(dLogits, probs []float32, targets []uint16, vocab, seq int) float32 {
	loss, valid := crossEntropyLossFromProbsUnscaled(dLogits, probs, targets, vocab, seq)
	if valid == 0 {
		return 0
	}
	scale := float32(1.0 / float64(valid))
	parallelForCF(len(dLogits[:vocab*seq]), func(start, end int) {
		scaleCrossEntropyGradSlice(dLogits, scale, start, end)
	})
	return loss
}

func crossEntropyLossFromProbsUnscaled(dLogits, probs []float32, targets []uint16, vocab, seq int) (float32, int) {
	if vocab <= 0 || seq <= 0 {
		return 0, 0
	}
	if len(probs) < vocab*seq || len(dLogits) < vocab*seq || len(targets) < seq {
		for i := range dLogits {
			dLogits[i] = 0
		}
		return 0, 0
	}
	if !sameBackingSlice(dLogits[:vocab*seq], probs[:vocab*seq]) {
		copy(dLogits[:vocab*seq], probs[:vocab*seq])
	}
	loss := 0.0
	valid := 0
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 || seq < workers*4 {
		loss, valid = crossEntropyLossFromProbsRange(dLogits, probs, targets, vocab, seq, 0, seq)
		if valid == 0 {
			return 0, 0
		}
		return float32(loss / float64(valid)), valid
	}
	if workers > seq {
		workers = seq
	}
	chunk := (seq + workers - 1) / workers
	type shard struct {
		loss  float64
		valid int
	}
	shards := make([]shard, workers)
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		start := worker * chunk
		if start >= seq {
			break
		}
		end := start + chunk
		if end > seq {
			end = seq
		}
		wg.Add(1)
		go func(start, end, worker int) {
			defer wg.Done()
			shards[worker].loss, shards[worker].valid = crossEntropyLossFromProbsRange(dLogits, probs, targets, vocab, seq, start, end)
		}(start, end, worker)
	}
	wg.Wait()
	for _, shard := range shards {
		loss += shard.loss
		valid += shard.valid
	}
	if valid == 0 {
		return 0, 0
	}
	return float32(loss / float64(valid)), valid
}

func sameBackingSlice(a, b []float32) bool {
	if len(a) == 0 || len(b) == 0 {
		return len(a) == 0 && len(b) == 0
	}
	return &a[0] == &b[0]
}

func crossEntropyLossFromProbsRange(dLogits, probs []float32, targets []uint16, vocab, seq, start, end int) (float64, int) {
	loss := 0.0
	valid := 0
	for t := start; t < end; t++ {
		tgt := int(targets[t])
		if tgt < 0 || tgt >= vocab {
			for v := 0; v < vocab; v++ {
				dLogits[v*seq+t] = 0
			}
			continue
		}
		p := probs[tgt*seq+t]
		if p < 1e-10 {
			p = 1e-10
		}
		loss -= math.Log(float64(p))
		dLogits[tgt*seq+t] -= 1
		valid++
	}
	return loss, valid
}

func scaleCrossEntropyGradSlice(dLogits []float32, scale float32, start, end int) {
	for i := start; i < end; i++ {
		dLogits[i] *= scale
	}
}

// --- rms norm gradient weights ---

func rmsNormGradWeights(dw, dy, x, w []float32, d, s int) {
	rmsNormGradWeightsWithRRMS(dw, dy, x, nil, d, s)
}

func rmsNormGradWeightsWithRRMS(dw, dy, x, rrms []float32, d, s int) {
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 || s < workers*4 {
		rmsNormGradWeightsRange(dw, dy, x, rrms, d, s, 0, s)
		return
	}
	if workers > s {
		workers = s
	}
	shards := make([][]float32, workers)
	chunk := (s + workers - 1) / workers
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		start := worker * chunk
		if start >= s {
			break
		}
		end := start + chunk
		if end > s {
			end = s
		}
		shards[worker] = make([]float32, d)
		wg.Add(1)
		go func(start, end, worker int) {
			defer wg.Done()
			rmsNormGradWeightsRange(shards[worker], dy, x, rrms, d, s, start, end)
		}(start, end, worker)
	}
	wg.Wait()
	for _, shard := range shards {
		if shard == nil {
			continue
		}
		for i := range dw {
			dw[i] += shard[i]
		}
	}
}

func rmsNormGradWeightsRange(dw, dy, x, rrms []float32, d, s, start, end int) {
	for t := start; t < end; t++ {
		scale := 0.0
		if len(rrms) > t {
			scale = float64(rrms[t])
		} else {
			sum := 0.0
			for i := 0; i < d; i++ {
				v := float64(x[i*s+t])
				sum += v * v
			}
			scale = 1.0 / math.Sqrt(sum/float64(d)+1e-5)
		}
		for i := 0; i < d; i++ {
			dw[i] += float32(float64(dy[i*s+t]*x[i*s+t]) * scale)
		}
	}
}

// --- residual scaling ---

var layerResidualScale = float32(1.0 / math.Sqrt(2.0*float64(stories.NLayers)))

func blendResidualInPlace(sum, base []float32) {
	for i := range sum {
		sum[i] = base[i] + (sum[i]-base[i])*layerResidualScale
	}
}

func addScaledResidual(dst, base, branch []float32) {
	addScaledResidualAccel(dst, base, branch, layerResidualScale)
}

func scaleInto(dst, src []float32, scale float32) {
	scaleIntoAccel(dst, src, scale)
}

// --- RoPE ---

func buildRoPETables(seq, headDim int) ([]float32, []float32) {
	if seq <= 0 || headDim <= 0 {
		return nil, nil
	}
	half := headDim / 2
	if half <= 0 {
		return nil, nil
	}
	cos := make([]float32, seq*half)
	sin := make([]float32, seq*half)
	for pos := 0; pos < seq; pos++ {
		base := pos * half
		for i := 0; i < half; i++ {
			freq := float64(pos) / math.Pow(10000, float64(2*i)/float64(headDim))
			cos[base+i] = float32(math.Cos(freq))
			sin[base+i] = float32(math.Sin(freq))
		}
	}
	return cos, sin
}

func applyRoPECFInPlace(x []float32, heads, headDim, seq int, ropeCos, ropeSin []float32) {
	if heads <= 0 || headDim <= 1 || seq <= 0 {
		return
	}
	half := headDim / 2
	for h := 0; h < heads; h++ {
		headBase := h * headDim
		for i := 0; i < half; i++ {
			even := (headBase + 2*i) * seq
			odd := even + seq
			for t := 0; t < seq; t++ {
				c := ropeCos[t*half+i]
				s := ropeSin[t*half+i]
				e := x[even+t]
				o := x[odd+t]
				x[even+t] = e*c - o*s
				x[odd+t] = o*c + e*s
			}
		}
	}
}
