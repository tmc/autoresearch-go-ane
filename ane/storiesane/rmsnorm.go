package storiesane

import (
	"math"
	"runtime"
	"sync"
)

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
