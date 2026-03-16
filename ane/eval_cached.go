//go:build darwin

package ane

import (
	"fmt"
	"math"
	"sync"
	"time"

	xane "github.com/tmc/apple/x/ane"
	"github.com/tmc/apple/x/ane/dynamicmatmul"
	"github.com/tmc/apple/x/ane/model"
)

// TokenTimings holds per-token timing breakdown for EvalNextToken.
type TokenTimings struct {
	Total      time.Duration
	Embed      time.Duration
	LayerTotal time.Duration
	QKV        time.Duration // Fused QKV matmul
	RoPE       time.Duration
	Attention  time.Duration // Single-query attention against KV cache
	Wo         time.Duration
	FFN        time.Duration // Fused W1W3 + SiLU + W2
	Residual   time.Duration
	RMSNorm    time.Duration
	Classifier time.Duration
	FP16Conv   time.Duration // fp16→fp32 conversion time
}

// LastTokenTimings returns the per-component timing breakdown from the most recent EvalNextToken call.
func (e *Engine) LastTokenTimings() TokenTimings {
	if e == nil {
		return TokenTimings{}
	}
	return e.lastTokenTimings
}

// ResetCache clears the KV cache, forcing the next EvalNextToken call to
// re-process the full prompt from scratch.
func (e *Engine) ResetCache() {
	if e == nil {
		return
	}
	e.kvc = nil
}

// EvalNextToken runs a single-token forward pass using a KV cache.
//
// On the first call (or after ResetCache), it performs a full prefill over
// the provided prompt tokens (populating the cache for every position).
// On subsequent calls it processes only the single new token at the next
// position, reusing cached K/V from all previous positions. This reduces
// per-token cost from O(n*seq) to O(n) where n is the model width.
//
// Returns the vocab-sized logits for the new token's position.
func (e *Engine) EvalNextToken(token int32) ([]float32, error) {
	if e == nil || e.mw == nil {
		return nil, fmt.Errorf("storiesane eval next token: engine is closed")
	}

	cfg := e.cfg
	dim := cfg.Dim
	qDim := cfg.QDim()
	kvDim := cfg.KVDim()
	hidden := cfg.Hidden
	heads := cfg.Heads
	kvHeads := cfg.EffectiveKVHeads()
	headDim := cfg.HeadDim()
	vocab := cfg.Vocab

	useFP16 := e.mw.FP16Layers != nil

	// Lazily allocate the KV cache and scratch buffers.
	if e.kvc == nil {
		e.kvc = newKVCache(cfg.NLayers, kvDim, e.seq)
		e.cachedRopeCos, e.cachedRopeSin = buildRoPETables(e.seq, headDim)
		// Single-position scratch buffers.
		e.cacheX = make([]float32, dim)
		e.cacheXNorm = make([]float32, dim)
		e.cacheQ = make([]float32, qDim)
		e.cacheK = make([]float32, kvDim)
		e.cacheV = make([]float32, kvDim)
		e.cacheAttOut = make([]float32, qDim)
		e.cacheX2 = make([]float32, dim)
		e.cacheH1 = make([]float32, hidden)
		e.cacheH3 = make([]float32, hidden)
		e.cacheGate = make([]float32, hidden)
		e.cacheFFOut = make([]float32, dim)
		e.cacheNext = make([]float32, dim)
		e.cacheLogits = make([]float32, vocab)
		// Fused QKV buffer: [qDim + 2*kvDim] for one BLAS call instead of 3.
		e.cacheQKV = make([]float32, qDim+2*kvDim)
		// Fused W1W3 buffer: [2*hidden] for one BLAS call instead of 2.
		e.cacheH1H3 = make([]float32, 2*hidden)
		// Pre-build fused weight matrices to avoid cold-start overhead.
		e.ensureFusedWeights()
		// Note: ANE dispatch overhead (~20ms/kernel) makes batch=1 ANE slower
		// than CPU BLAS. ANE is only beneficial at batch>=32 or with fused
		// multi-layer kernels. Keep CPU BLAS for single-token path.
	}

	// Lazily allocate fp16 scratch and fused fp16 weight arrays.
	if useFP16 && e.fp16Scratch == nil {
		maxSize := 2 * hidden * dim // fused W1W3
		if qkvSize := (qDim + 2*kvDim) * dim; qkvSize > maxSize {
			maxSize = qkvSize
		}
		if woSize := dim * qDim; woSize > maxSize {
			maxSize = woSize
		}
		if w2Size := dim * hidden; w2Size > maxSize {
			maxSize = w2Size
		}
		e.fp16Scratch = make([]float32, maxSize)
		e.fusedQKVFP16 = make([][]uint16, cfg.NLayers)
		e.fusedW1W3FP16 = make([][]uint16, cfg.NLayers)
	}

	if e.kvc.pos >= e.kvc.maxSeq {
		return nil, fmt.Errorf("storiesane eval next token: cache full (pos=%d, max=%d)", e.kvc.pos, e.kvc.maxSeq)
	}

	pos := e.kvc.pos

	var timings TokenTimings
	t0 := time.Now()

	// --- Embedding lookup for single token ---
	tEmbed := time.Now()
	tokInt := int(token)
	if tokInt >= vocab {
		tokInt = 0
	}
	for d := 0; d < dim; d++ {
		e.cacheX[d] = e.mw.Embed[tokInt*dim+d]
	}
	timings.Embed = time.Since(tEmbed)

	cur := e.cacheX
	next := e.cacheNext
	residualScale := float32(e.cfg.ResidualScale())

	// --- Transformer layers ---
	tLayerStart := time.Now()
	for li := range e.mw.Layers {
		layer := e.mw.Layers[li]

		// --- CPU BLAS path (ANE dispatch overhead too high for batch=1) ---

		// Attention sub-block: RMSNorm → Q,K,V → RoPE → cache → attention → Wo → residual
		tRMS := time.Now()
		if useFP16 {
			rmsNormSingle(e.cacheXNorm, cur, e.mw.FP16Layers[li].RMSAtt, dim)
		} else {
			rmsNormSingle(e.cacheXNorm, cur, layer.RMSAtt, dim)
		}
		timings.RMSNorm += time.Since(tRMS)

		// Unfused QKV: run 3 smaller GEMVs concurrently for better core utilization.
		tQKV := time.Now()
		var qkvWG sync.WaitGroup
		qkvWG.Add(2)
		go func() {
			linearSingle(e.cacheQKV[qDim:qDim+kvDim], layer.Wk, e.cacheXNorm, kvDim, dim)
			qkvWG.Done()
		}()
		go func() {
			linearSingle(e.cacheQKV[qDim+kvDim:], layer.Wv, e.cacheXNorm, kvDim, dim)
			qkvWG.Done()
		}()
		linearSingle(e.cacheQKV[:qDim], layer.Wq, e.cacheXNorm, qDim, dim)
		qkvWG.Wait()
		timings.QKV += time.Since(tQKV)

		fusedQ := e.cacheQKV[:qDim]
		fusedK := e.cacheQKV[qDim : qDim+kvDim]
		fusedV := e.cacheQKV[qDim+kvDim:]

		tRoPE := time.Now()
		applyRoPESinglePos(fusedQ, heads, headDim, pos, e.cachedRopeCos, e.cachedRopeSin)
		applyRoPESinglePos(fusedK, kvHeads, headDim, pos, e.cachedRopeCos, e.cachedRopeSin)
		timings.RoPE += time.Since(tRoPE)

		// Append K, V to cache.
		e.kvc.appendKV(li, fusedK, fusedV)

		// Single-query attention against cached KV.
		tAttn := time.Now()
		cachedK := e.kvc.getK(li)
		cachedV := e.kvc.getV(li)
		singleQueryGQAAttention(e.cacheAttOut, fusedQ, cachedK, cachedV,
			heads, kvHeads, headDim, pos+1, e.kvc.maxSeq)
		timings.Attention += time.Since(tAttn)

		// Wo projection: [dim, qDim] @ attOut[qDim] → x2[dim]
		tWo := time.Now()
		if useFP16 {
			woSize := dim * qDim
			tFP16 := time.Now()
			convertFP16ToF32(e.fp16Scratch[:woSize], e.mw.FP16Layers[li].Wo)
			timings.FP16Conv += time.Since(tFP16)
			linearSingle(e.cacheX2, e.fp16Scratch[:woSize], e.cacheAttOut, dim, qDim)
		} else {
			linearSingle(e.cacheX2, layer.Wo, e.cacheAttOut, dim, qDim)
		}
		timings.Wo += time.Since(tWo)

		// Residual connection: x2 = cur + scale * x2
		tRes := time.Now()
		addScaledResidualSingle(e.cacheX2, cur, e.cacheX2, residualScale)
		timings.Residual += time.Since(tRes)

		// FFN sub-block: RMSNorm → W1+W3 fused → SiLU → W2 → residual
		tRMS = time.Now()
		if useFP16 {
			rmsNormSingle(e.cacheXNorm, e.cacheX2, e.mw.FP16Layers[li].RMSFFN, dim)
		} else {
			rmsNormSingle(e.cacheXNorm, e.cacheX2, layer.RMSFFN, dim)
		}
		timings.RMSNorm += time.Since(tRMS)

		// Unfused W1+W3: run concurrently for better core utilization at batch=1.
		tFFN := time.Now()
		var ffnWG sync.WaitGroup
		ffnWG.Add(1)
		go func() {
			linearSingle(e.cacheH1H3[hidden:], layer.W3, e.cacheXNorm, hidden, dim)
			ffnWG.Done()
		}()
		linearSingle(e.cacheH1H3[:hidden], layer.W1, e.cacheXNorm, hidden, dim)
		ffnWG.Wait()
		siluMulAccel(e.cacheGate, e.cacheH1H3[:hidden], e.cacheH1H3[hidden:])

		if useFP16 {
			w2Size := dim * hidden
			tFP16 := time.Now()
			convertFP16ToF32(e.fp16Scratch[:w2Size], e.mw.FP16Layers[li].W2)
			timings.FP16Conv += time.Since(tFP16)
			linearSingle(e.cacheFFOut, e.fp16Scratch[:w2Size], e.cacheGate, dim, hidden)
		} else {
			linearSingle(e.cacheFFOut, layer.W2, e.cacheGate, dim, hidden)
		}
		timings.FFN += time.Since(tFFN)

		tRes = time.Now()
		addScaledResidualSingle(next, e.cacheX2, e.cacheFFOut, residualScale)
		timings.Residual += time.Since(tRes)

		cur, next = next, cur
	}
	timings.LayerTotal = time.Since(tLayerStart)

	// Final RMSNorm + classifier.
	tRMS := time.Now()
	rmsNormSingle(e.cacheXNorm, cur, e.mw.RMSFinal, dim)
	timings.RMSNorm += time.Since(tRMS)

	// Classifier: logits = Embed @ xNorm where Embed is [vocab, dim] row-major.
	tCls := time.Now()
	linearSingle(e.cacheLogits, e.mw.Embed, e.cacheXNorm, vocab, dim)
	timings.Classifier = time.Since(tCls)

	e.kvc.advancePos()

	timings.Total = time.Since(t0)
	e.lastTokenTimings = timings

	// Return the internal buffer directly — caller must consume before next call.
	return e.cacheLogits, nil
}

// ensureFusedWeights pre-builds the fused QKV and W1W3 weight matrices for
// all layers. Called once at first use to avoid cold-start overhead.
func (e *Engine) ensureFusedWeights() {
	cfg := e.cfg
	dim := cfg.Dim
	qDim := cfg.QDim()
	kvDim := cfg.KVDim()
	hidden := cfg.Hidden

	if e.fusedQKVW == nil {
		e.fusedQKVW = make([][]float32, cfg.NLayers)
		e.fusedW1W3 = make([][]float32, cfg.NLayers)
	}
	for li := range e.mw.Layers {
		if e.fusedQKVW[li] == nil {
			layer := e.mw.Layers[li]
			qkvSize := (qDim + 2*kvDim) * dim
			fused := make([]float32, qkvSize)
			copy(fused[0:], layer.Wq)
			copy(fused[qDim*dim:], layer.Wk)
			copy(fused[(qDim+kvDim)*dim:], layer.Wv)
			e.fusedQKVW[li] = fused
		}
		if e.fusedW1W3[li] == nil {
			layer := e.mw.Layers[li]
			w1w3Size := 2 * hidden * dim
			fused := make([]float32, w1w3Size)
			copy(fused[0:], layer.W1)
			copy(fused[hidden*dim:], layer.W3)
			e.fusedW1W3[li] = fused
		}
	}
}

// ensureANEExecutors compiles and primes ANE executors for single-token inference.
func (e *Engine) ensureANEExecutors() bool {
	if e.aneReady {
		return true
	}
	if !e.useANE {
		return false
	}

	cfg := e.cfg
	dim := cfg.Dim
	qDim := cfg.QDim()
	kvDim := cfg.KVDim()
	hidden := cfg.Hidden
	vocab := cfg.Vocab

	opts := dynamicmatmul.Options{}

	e.aneQKV = make([]*dynamicmatmul.Executor, cfg.NLayers)
	e.aneWo = make([]*dynamicmatmul.Executor, cfg.NLayers)
	e.aneW1W3 = make([]*dynamicmatmul.Executor, cfg.NLayers)
	e.aneW2 = make([]*dynamicmatmul.Executor, cfg.NLayers)

	for li := range e.mw.Layers {
		layer := e.mw.Layers[li]
		var err error

		// Fused QKV: [dim → qDim+2*kvDim]
		e.aneQKV[li], err = dynamicmatmul.New(1, dim, qDim+2*kvDim, opts)
		if err != nil {
			e.cleanupANEExecutors()
			return false
		}
		fusedQKV := make([]float32, (qDim+2*kvDim)*dim)
		copy(fusedQKV[0:], layer.Wq)
		copy(fusedQKV[qDim*dim:], layer.Wk)
		copy(fusedQKV[(qDim+kvDim)*dim:], layer.Wv)
		fusedQKVT := transposeWeights(fusedQKV, qDim+2*kvDim, dim)
		if err := e.aneQKV[li].PrimeWeightsIO(fusedQKVT); err != nil {
			e.cleanupANEExecutors()
			return false
		}

		// Wo: [qDim → dim]
		e.aneWo[li], err = dynamicmatmul.New(1, qDim, dim, opts)
		if err != nil {
			e.cleanupANEExecutors()
			return false
		}
		woT := transposeWeights(layer.Wo, dim, qDim)
		if err := e.aneWo[li].PrimeWeightsIO(woT); err != nil {
			e.cleanupANEExecutors()
			return false
		}

		// Fused W1+W3: [dim → 2*hidden]
		e.aneW1W3[li], err = dynamicmatmul.New(1, dim, 2*hidden, opts)
		if err != nil {
			e.cleanupANEExecutors()
			return false
		}
		fusedW1W3 := make([]float32, 2*hidden*dim)
		copy(fusedW1W3[0:], layer.W1)
		copy(fusedW1W3[hidden*dim:], layer.W3)
		fusedW1W3T := transposeWeights(fusedW1W3, 2*hidden, dim)
		if err := e.aneW1W3[li].PrimeWeightsIO(fusedW1W3T); err != nil {
			e.cleanupANEExecutors()
			return false
		}

		// W2: [hidden → dim]
		e.aneW2[li], err = dynamicmatmul.New(1, hidden, dim, opts)
		if err != nil {
			e.cleanupANEExecutors()
			return false
		}
		w2T := transposeWeights(layer.W2, dim, hidden)
		if err := e.aneW2[li].PrimeWeightsIO(w2T); err != nil {
			e.cleanupANEExecutors()
			return false
		}
	}

	// Classifier: [dim → vocab]
	var err error
	e.aneCls, err = dynamicmatmul.New(1, dim, vocab, opts)
	if err != nil {
		// Classifier may be too large for ANE — fall back to CPU for it.
		e.aneCls = nil
	} else {
		embedT := transposeWeights(e.mw.Embed, vocab, dim)
		if err := e.aneCls.PrimeWeightsIO(embedT); err != nil {
			e.aneCls.Close()
			e.aneCls = nil
		}
	}

	e.aneReady = true
	return true
}

func (e *Engine) cleanupANEExecutors() {
	for _, ex := range e.aneQKV {
		if ex != nil {
			ex.Close()
		}
	}
	e.aneQKV = nil
	for _, ex := range e.aneWo {
		if ex != nil {
			ex.Close()
		}
	}
	e.aneWo = nil
	for _, ex := range e.aneW1W3 {
		if ex != nil {
			ex.Close()
		}
	}
	e.aneW1W3 = nil
	for _, ex := range e.aneW2 {
		if ex != nil {
			ex.Close()
		}
	}
	e.aneW2 = nil
	if e.aneCls != nil {
		e.aneCls.Close()
		e.aneCls = nil
	}
	e.aneReady = false
}

// EvalPrefill processes a full prompt through EvalNextToken, populating the
// KV cache. It returns the logits for the last prompt position.
func (e *Engine) EvalPrefill(tokens []int32) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("storiesane eval prefill: empty prompt")
	}
	var logits []float32
	var err error
	for _, tok := range tokens {
		logits, err = e.EvalNextToken(tok)
		if err != nil {
			return nil, fmt.Errorf("storiesane eval prefill: %w", err)
		}
	}
	return logits, nil
}

// --- Single-position helper functions ---

// rmsNormSingle computes RMSNorm for a single position.
// out[d] = x[d] * (1/rms) * w[d]
func rmsNormSingle(out, x, w []float32, dim int) {
	sum := 0.0
	for i := 0; i < dim; i++ {
		v := float64(x[i])
		sum += v * v
	}
	scale := float32(1.0 / math.Sqrt(sum/float64(dim)+1e-5))
	for i := 0; i < dim; i++ {
		out[i] = x[i] * scale * w[i]
	}
}

// linearSingle computes out = W @ x for a single position.
// W is [outDim, inDim] row-major, x is [inDim], out is [outDim].
func linearSingle(out, w, x []float32, outDim, inDim int) {
	linearCF(out, w, x, outDim, inDim, 1)
}

// applyRoPESinglePos applies rotary position embeddings to a single position.
// x is laid out as [nHeads * headDim] flat (CF with seq=1).
func applyRoPESinglePos(x []float32, nHeads, headDim, pos int, ropeCos, ropeSin []float32) {
	half := headDim / 2
	if half <= 0 || pos < 0 {
		return
	}
	// ropeCos/Sin tables: [seq, half], row `pos` starts at pos*half.
	ropeBase := pos * half
	for h := 0; h < nHeads; h++ {
		headOff := h * headDim
		for i := 0; i < half; i++ {
			c := ropeCos[ropeBase+i]
			s := ropeSin[ropeBase+i]
			// In CF layout with seq=1, element at channel c is just x[c].
			// The even/odd interleaving in the full CF layout collapses to
			// consecutive pairs: x[headOff + 2*i] and x[headOff + 2*i + 1].
			even := headOff + 2*i
			odd := even + 1
			e := x[even]
			o := x[odd]
			x[even] = e*c - o*s
			x[odd] = o*c + e*s
		}
	}
}

// addScaledResidualSingle: dst[i] = base[i] + scale * branch[i]
func addScaledResidualSingle(dst, base, branch []float32, scale float32) {
	addScaledResidualAccel(dst, base, branch, scale)
}

// singleQueryGQAAttention computes attention for a single query position
// against the KV cache. This is the decode-time attention kernel.
//
// q:      [qDim] flat = [heads * headDim]
// kCache: [kvDim, maxSeq] CF layout, only columns [0, npos) are valid
// vCache: [kvDim, maxSeq] CF layout, only columns [0, npos) are valid
// out:    [qDim] flat
// npos:   number of valid cached positions (including current)
// maxSeq: stride of the cache (second dimension)
func singleQueryGQAAttention(out, q, kCache, vCache []float32,
	heads, kvHeads, headDim, npos, maxSeq int) {

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	groupSize := 1
	if kvHeads > 0 {
		groupSize = heads / kvHeads
	}

	scores := make([]float32, npos)
	probs := make([]float32, npos)

	for h := 0; h < heads; h++ {
		kvH := h / groupSize
		qBase := h * headDim
		kBase := kvH * headDim

		// Compute scores: q[h] dot k_cached[kvH, j] for j in [0, npos)
		for j := 0; j < npos; j++ {
			sum := float32(0)
			for d := 0; d < headDim; d++ {
				sum += q[qBase+d] * kCache[(kBase+d)*maxSeq+j]
			}
			scores[j] = sum * scale
		}

		// Softmax
		softmaxRow(probs, scores)

		// Weighted sum of v[kvH]
		for d := 0; d < headDim; d++ {
			sum := float32(0)
			for j := 0; j < npos; j++ {
				sum += probs[j] * vCache[(kBase+d)*maxSeq+j]
			}
			out[qBase+d] = sum
		}
	}
}

// writeANEActivations writes activations x[dim, seq] to the first seq columns
// of an ANE kernel's input IOSurface. Config-aware: uses actual dim, not stories.Dim.
func writeANEActivations(k *model.Kernel, seq int, x []float32) error {
	return withLockedFP16Input(k, 0, func(layout xane.TensorLayout, data []uint16) error {
		writeChannelFirstActsFP16(data, layout, seq, x)
		return nil
	})
}

// writeANEFFNActivations writes activations to an FFN kernel's input.
func writeANEFFNActivations(k *model.Kernel, seq int, x []float32) error {
	return withLockedFP16Input(k, 0, func(layout xane.TensorLayout, data []uint16) error {
		writeChannelFirstActsFP16(data, layout, seq, x)
		return nil
	})
}
