package stories

import (
	"math"
	"runtime"
	"sync"
)

func parallelFor(n int, fn func(start, end int)) {
	if n <= 0 {
		return
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 || n < workers*4 {
		fn(0, n)
		return
	}
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunk
		if start >= n {
			break
		}
		end := start + chunk
		if end > n {
			end = n
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			fn(s, e)
		}(start, end)
	}
	wg.Wait()
}

func EmbedLookup(out, embed []float32, tokens []int32, dim, seq int) {
	if dim <= 0 || seq <= 0 {
		return
	}
	if len(out) < dim*seq || len(tokens) < seq {
		return
	}
	vocab := len(embed) / dim
	if vocab <= 0 {
		for t := 0; t < seq; t++ {
			for d := 0; d < dim; d++ {
				out[d*seq+t] = 0
			}
		}
		return
	}
	parallelFor(seq, func(start, end int) {
		for t := start; t < end; t++ {
			tok := int(tokens[t])
			if tok < 0 || tok >= vocab {
				for d := 0; d < dim; d++ {
					out[d*seq+t] = 0
				}
				continue
			}
			base := tok * dim
			for d := 0; d < dim; d++ {
				out[d*seq+t] = embed[base+d]
			}
		}
	})
}

func EmbedBackward(dEmbed, dx []float32, tokens []int32, dim, seq int) {
	if dim <= 0 || seq <= 0 {
		return
	}
	if len(tokens) < seq || len(dx) < dim*seq || len(dEmbed) < dim {
		return
	}
	vocab := len(dEmbed) / dim
	if vocab <= 0 {
		return
	}
	for t := 0; t < seq; t++ {
		tok := int(tokens[t])
		if tok < 0 || tok >= vocab {
			continue
		}
		base := tok * dim
		for d := 0; d < dim; d++ {
			dEmbed[base+d] += dx[d*seq+t]
		}
	}
}

func RMSNorm(out, x, w []float32, d, s int) {
	parallelFor(s, func(start, end int) {
		for t := start; t < end; t++ {
			sum := 0.0
			for i := 0; i < d; i++ {
				v := float64(x[i*s+t])
				sum += v * v
			}
			rrms := 1.0 / math.Sqrt(sum/float64(d)+1e-5)
			for i := 0; i < d; i++ {
				out[i*s+t] = float32(float64(x[i*s+t]) * rrms * float64(w[i]))
			}
		}
	})
}

func RMSNormBackward(dx, dw, dy, x, w []float32, d, s int) {
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 || s < workers*4 {
		rmsNormBackwardRange(dx, dw, dy, x, w, d, s, 0, s)
		return
	}
	if workers > s {
		workers = s
	}
	chunk := (s + workers - 1) / workers
	shards := make([][]float32, workers)
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
			rmsNormBackwardRange(dx, shards[worker], dy, x, w, d, s, start, end)
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

func rmsNormBackwardRange(dx, dw, dy, x, w []float32, d, s, start, end int) {
	for t := start; t < end; t++ {
		sum := 0.0
		for i := 0; i < d; i++ {
			v := float64(x[i*s+t])
			sum += v * v
		}
		rrms := 1.0 / math.Sqrt(sum/float64(d)+1e-5)
		rrms2InvD := (rrms * rrms) / float64(d)
		dot := 0.0
		for i := 0; i < d; i++ {
			dot += float64(dy[i*s+t] * x[i*s+t] * w[i])
		}
		for i := 0; i < d; i++ {
			v := float64(dy[i*s+t]) - float64(x[i*s+t])*dot*rrms2InvD
			dx[i*s+t] = float32(v * rrms * float64(w[i]))
			dw[i] += float32(float64(dy[i*s+t]*x[i*s+t]) * rrms)
		}
	}
}

func CrossEntropyLoss(dLogits, logits []float32, targets []int32, v, s int) float32 {
	if v <= 0 || s <= 0 {
		return 0
	}
	if len(logits) < v*s || len(dLogits) < v*s || len(targets) < s {
		for i := range dLogits {
			dLogits[i] = 0
		}
		return 0
	}
	loss := 0.0
	valid := 0
	type shard struct {
		loss  float64
		valid int
	}
	shards := make([]shard, runtime.GOMAXPROCS(0))
	var shardMu sync.Mutex
	parallelFor(s, func(start, end int) {
		idx := start * len(shards) / s
		localLoss := 0.0
		localValid := 0
		for t := start; t < end; t++ {
			mx := logits[t]
			for i := 1; i < v; i++ {
				val := logits[i*s+t]
				if val > mx {
					mx = val
				}
			}
			sum := 0.0
			for i := 0; i < v; i++ {
				e := float32(math.Exp(float64(logits[i*s+t] - mx)))
				dLogits[i*s+t] = e
				sum += float64(e)
			}
			inv := float32(1.0 / sum)
			tgt := int(targets[t])
			for i := 0; i < v; i++ {
				dLogits[i*s+t] *= inv
			}
			if tgt < 0 || tgt >= v {
				for i := 0; i < v; i++ {
					dLogits[i*s+t] = 0
				}
				continue
			}
			p := dLogits[tgt*s+t]
			if p < 1e-10 {
				p = 1e-10
			}
			localLoss -= math.Log(float64(p))
			dLogits[tgt*s+t] -= 1
			localValid++
		}
		shardMu.Lock()
		shards[idx].loss += localLoss
		shards[idx].valid += localValid
		shardMu.Unlock()
	})
	for i := range shards {
		loss += shards[i].loss
		valid += shards[i].valid
	}
	if valid == 0 {
		return 0
	}
	invValid := float32(1.0 / float64(valid))
	parallelFor(s, func(start, end int) {
		for t := start; t < end; t++ {
			tgt := int(targets[t])
			if tgt < 0 || tgt >= v {
				continue
			}
			for i := 0; i < v; i++ {
				dLogits[i*s+t] *= invValid
			}
		}
	})
	return float32(loss / float64(valid))
}

type AdamState struct {
	M []float32
	V []float32
}

func NewAdamState(n int) AdamState {
	return AdamState{M: make([]float32, n), V: make([]float32, n)}
}

func AdamUpdate(w, g []float32, st *AdamState, t int, lr, b1, b2, eps float32) {
	bc1 := float32(1.0 - math.Pow(float64(b1), float64(t)))
	bc2 := float32(1.0 - math.Pow(float64(b2), float64(t)))
	for i := range w {
		st.M[i] = b1*st.M[i] + (1-b1)*g[i]
		st.V[i] = b2*st.V[i] + (1-b2)*g[i]*g[i]
		mh := st.M[i] / bc1
		vh := st.V[i] / bc2
		w[i] -= lr * mh / (float32(math.Sqrt(float64(vh))) + eps)
	}
}

func MatMulVocabSeq(logits, embed, x []float32, vocab, dim, seq int) {
	if matMulVocabSeqAccelerate(logits, embed, x, vocab, dim, seq) {
		return
	}
	parallelFor(vocab, func(start, end int) {
		acc := make([]float32, seq)
		for v := start; v < end; v++ {
			eb := embed[v*dim : (v+1)*dim]
			for t := 0; t < seq; t++ {
				acc[t] = 0
			}
			for d := 0; d < dim; d++ {
				w := eb[d]
				xd := x[d*seq : (d+1)*seq]
				for t := 0; t < seq; t++ {
					acc[t] += w * xd[t]
				}
			}
			out := logits[v*seq : (v+1)*seq]
			for t := 0; t < seq; t++ {
				out[t] = acc[t]
			}
		}
	})
}

func MatMulEmbedT(dx, embed, dLogits []float32, vocab, dim, seq int) {
	MatMulEmbedTScale(dx, embed, dLogits, vocab, dim, seq, 1)
}

func MatMulEmbedTScale(dx, embed, dLogits []float32, vocab, dim, seq int, scale float32) {
	if scale == 0 {
		clear(dx)
		return
	}
	if matMulEmbedTAccelerateScale(dx, embed, dLogits, vocab, dim, seq, scale) {
		return
	}
	parallelFor(dim, func(start, end int) {
		acc := make([]float32, seq)
		for d := start; d < end; d++ {
			for t := 0; t < seq; t++ {
				acc[t] = 0
			}
			for v := 0; v < vocab; v++ {
				w := embed[v*dim+d]
				dlv := dLogits[v*seq : (v+1)*seq]
				for t := 0; t < seq; t++ {
					acc[t] += w * dlv[t]
				}
			}
			out := dx[d*seq : (d+1)*seq]
			for t := 0; t < seq; t++ {
				out[t] = acc[t] * scale
			}
		}
	})
}

func MatMulGradEmbed(dEmbed, dLogits, x []float32, vocab, dim, seq int) {
	MatMulGradEmbedScale(dEmbed, dLogits, x, vocab, dim, seq, 1)
}

func MatMulGradEmbedScale(dEmbed, dLogits, x []float32, vocab, dim, seq int, scale float32) {
	if scale == 0 {
		clear(dEmbed)
		return
	}
	if matMulGradEmbedAccelerateScale(dEmbed, dLogits, x, vocab, dim, seq, scale) {
		return
	}
	parallelFor(vocab, func(start, end int) {
		for v := start; v < end; v++ {
			for d := 0; d < dim; d++ {
				sum := float32(0)
				for t := 0; t < seq; t++ {
					sum += dLogits[v*seq+t] * x[d*seq+t]
				}
				dEmbed[v*dim+d] = sum * scale
			}
		}
	})
}
