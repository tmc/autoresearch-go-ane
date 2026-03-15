//go:build darwin

package storiesane

import (
	"fmt"
	"math"
)

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
func (e *Engine) EvalNextToken(token uint16) ([]float32, error) {
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
	}

	if e.kvc.pos >= e.kvc.maxSeq {
		return nil, fmt.Errorf("storiesane eval next token: cache full (pos=%d, max=%d)", e.kvc.pos, e.kvc.maxSeq)
	}

	pos := e.kvc.pos

	// --- Embedding lookup for single token ---
	tokInt := int(token)
	if tokInt >= vocab {
		tokInt = 0
	}
	for d := 0; d < dim; d++ {
		e.cacheX[d] = e.mw.Embed[tokInt*dim+d]
	}

	cur := e.cacheX
	next := e.cacheNext
	residualScale := float32(e.cfg.ResidualScale())

	// --- Transformer layers ---
	for li := range e.mw.Layers {
		layer := e.mw.Layers[li]

		// Attention sub-block: RMSNorm → Q,K,V → RoPE → cache → attention → Wo → residual
		rmsNormSingle(e.cacheXNorm, cur, layer.RMSAtt, dim)

		// Fused QKV: one BLAS call instead of 3.
		if e.fusedQKVW == nil {
			e.fusedQKVW = make([][]float32, cfg.NLayers)
			e.fusedW1W3 = make([][]float32, cfg.NLayers)
		}
		if e.fusedQKVW[li] == nil {
			// Build fused weight matrix: stack Wq, Wk, Wv vertically → [(qDim+2*kvDim), dim]
			fused := make([]float32, (qDim+2*kvDim)*dim)
			copy(fused[0:], layer.Wq)
			copy(fused[qDim*dim:], layer.Wk)
			copy(fused[(qDim+kvDim)*dim:], layer.Wv)
			e.fusedQKVW[li] = fused
		}
		linearSingle(e.cacheQKV, e.fusedQKVW[li], e.cacheXNorm, qDim+2*kvDim, dim)
		// Use slices directly from fused output to avoid copies.
		fusedQ := e.cacheQKV[:qDim]
		fusedK := e.cacheQKV[qDim : qDim+kvDim]
		fusedV := e.cacheQKV[qDim+kvDim:]

		applyRoPESinglePos(fusedQ, heads, headDim, pos, e.cachedRopeCos, e.cachedRopeSin)
		applyRoPESinglePos(fusedK, kvHeads, headDim, pos, e.cachedRopeCos, e.cachedRopeSin)

		// Append K, V to cache.
		e.kvc.appendKV(li, fusedK, fusedV)

		// Single-query attention against cached KV.
		cachedK := e.kvc.getK(li)
		cachedV := e.kvc.getV(li)
		singleQueryGQAAttention(e.cacheAttOut, fusedQ, cachedK, cachedV,
			heads, kvHeads, headDim, pos+1, e.kvc.maxSeq)

		// Wo projection: [dim, qDim] @ attOut[qDim] → x2[dim]
		linearSingle(e.cacheX2, layer.Wo, e.cacheAttOut, dim, qDim)

		// Residual connection: x2 = cur + scale * x2
		addScaledResidualSingle(e.cacheX2, cur, e.cacheX2, residualScale)

		// FFN sub-block: RMSNorm → W1+W3 fused → SiLU → W2 → residual
		rmsNormSingle(e.cacheXNorm, e.cacheX2, layer.RMSFFN, dim)
		if e.fusedW1W3[li] == nil {
			fused := make([]float32, 2*hidden*dim)
			copy(fused[0:], layer.W1)
			copy(fused[hidden*dim:], layer.W3)
			e.fusedW1W3[li] = fused
		}
		linearSingle(e.cacheH1H3, e.fusedW1W3[li], e.cacheXNorm, 2*hidden, dim)
		siluMulAccel(e.cacheGate, e.cacheH1H3[:hidden], e.cacheH1H3[hidden:])
		linearSingle(e.cacheFFOut, layer.W2, e.cacheGate, dim, hidden)
		addScaledResidualSingle(next, e.cacheX2, e.cacheFFOut, residualScale)

		cur, next = next, cur
	}

	// Final RMSNorm + classifier.
	rmsNormSingle(e.cacheXNorm, cur, e.mw.RMSFinal, dim)

	// Classifier: logits = Embed @ xNorm where Embed is [vocab, dim] row-major.
	// Use BLAS matmul via linearCF with seq=1.
	linearSingle(e.cacheLogits, e.mw.Embed, e.cacheXNorm, vocab, dim)

	e.kvc.advancePos()

	// Return the internal buffer directly — caller must consume before next call.
	return e.cacheLogits, nil
}

// EvalPrefill processes a full prompt through EvalNextToken, populating the
// KV cache. It returns the logits for the last prompt position.
func (e *Engine) EvalPrefill(tokens []uint16) ([]float32, error) {
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
	// Use BLAS via linearCF with seq=1 — the CF layout with seq=1
	// is equivalent to a flat vector.
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
