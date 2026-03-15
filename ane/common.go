package ane

import (
	"math"
	"runtime"
)

// --- grad task concurrency ---

const defaultGradTaskConcurrency = 4

var gradTaskLimit = defaultGradTaskConcurrency

// SetGradTaskConcurrency sets the maximum number of concurrent gradient tasks.
//
// Values <= 0 restore the default.
func SetGradTaskConcurrency(n int) {
	if n <= 0 {
		gradTaskLimit = defaultGradTaskConcurrency
		return
	}
	gradTaskLimit = n
}

func gradTaskConcurrency() int {
	n := runtime.GOMAXPROCS(0)
	if n < 1 {
		n = 1
	}
	if n > gradTaskLimit {
		n = gradTaskLimit
	}
	return n
}

// --- offload tile ranges ---

type tileRange struct {
	start int
	size  int
}

func classifierTileRanges(vocab, tile int) []tileRange {
	if vocab <= 0 || tile <= 0 {
		return nil
	}
	n := (vocab + tile - 1) / tile
	ranges := make([]tileRange, 0, n)
	for start := 0; start < vocab; start += tile {
		size := tile
		if rem := vocab - start; rem < size {
			size = rem
		}
		ranges = append(ranges, tileRange{start: start, size: size})
	}
	return ranges
}

// --- forward common (CPU fallback) ---

func causalAttentionCF(out, qf, kf, vf []float32, heads, headDim, seq int) {
	scores := make([]float32, seq)
	probs := make([]float32, seq)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	for h := 0; h < heads; h++ {
		base := h * headDim
		for t := 0; t < seq; t++ {
			for j := 0; j < seq; j++ {
				if j > t {
					scores[j] = -65504
					continue
				}
				sum := float32(0)
				for i := 0; i < headDim; i++ {
					sum += qf[(base+i)*seq+t] * kf[(base+i)*seq+j]
				}
				scores[j] = sum * scale
			}
			softmaxRow(probs, scores)
			for i := 0; i < headDim; i++ {
				sum := float32(0)
				for j := 0; j < seq; j++ {
					sum += probs[j] * vf[(base+i)*seq+j]
				}
				out[(base+i)*seq+t] = sum
			}
		}
	}
}

func linearCF(out, weights, in []float32, outCh, inCh, seq int) {
	for o := 0; o < outCh; o++ {
		row := weights[o*inCh : (o+1)*inCh]
		for t := 0; t < seq; t++ {
			sum := float32(0)
			for i := 0; i < inCh; i++ {
				sum += row[i] * in[i*seq+t]
			}
			out[o*seq+t] = sum
		}
	}
}

func rmsNormCF(out, x, w []float32, dim, seq int) {
	rmsNormCFWithRRMS(out, nil, x, w, dim, seq)
}

func rmsNormCFWithRRMS(out, rrms, x, w []float32, dim, seq int) {
	parallelForCF(seq, func(start, end int) {
		for t := start; t < end; t++ {
			sum := 0.0
			for i := 0; i < dim; i++ {
				v := float64(x[i*seq+t])
				sum += v * v
			}
			scale := float32(1.0 / math.Sqrt(sum/float64(dim)+1e-5))
			if rrms != nil {
				rrms[t] = scale
			}
			for i := 0; i < dim; i++ {
				out[i*seq+t] = x[i*seq+t] * scale * w[i]
			}
		}
	})
}

func rmsNormRRMS(rrms, x []float32, dim, seq int) {
	if len(rrms) < seq {
		return
	}
	parallelForCF(seq, func(start, end int) {
		for t := start; t < end; t++ {
			sum := 0.0
			for i := 0; i < dim; i++ {
				v := float64(x[i*seq+t])
				sum += v * v
			}
			rrms[t] = float32(1.0 / math.Sqrt(sum/float64(dim)+1e-5))
		}
	})
}

func silu32(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

func softmaxRow(out, in []float32) {
	softmaxRowAccel(out, in)
}
