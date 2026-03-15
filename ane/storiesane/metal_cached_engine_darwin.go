//go:build darwin && cgo

package storiesane

// metalLayerHandles holds per-layer cached Metal GPU matmul handles.
type metalLayerHandles struct {
	qkv [][3]*MetalMatmul // per-layer [Wq, Wk, Wv]
	wo  []*MetalMatmul    // per-layer Wo
	w1  []*MetalMatmul    // per-layer W1
	w3  []*MetalMatmul    // per-layer W3
	w2  []*MetalMatmul    // per-layer W2
	cls *MetalMatmul      // classifier
}

// ensureMetalMatmuls lazily creates cached Metal GPU matmul handles for all
// layer weight matrices. Only enabled for large models where GPU throughput
// exceeds CPU BLAS (Accelerate sgemv). Returns true when Metal handles are
// ready to use.
func (e *Engine) ensureMetalMatmuls() bool {
	if h, ok := e.metalHandles.(*metalLayerHandles); ok && h != nil {
		return true
	}

	cfg := e.cfg
	// Only beneficial for large models where Metal throughput beats CPU AMX sgemv.
	// Threshold: total weight data > 4 GB.
	weightBytes := int64(cfg.NLayers) * int64(cfg.WqSize()+cfg.WkSize()+cfg.WvSize()+cfg.WoSize()+cfg.W1Size()+cfg.W2Size()+cfg.W3Size()) * 4
	if weightBytes <= 4*1024*1024*1024 {
		return false
	}

	dim := cfg.Dim
	qDim := cfg.QDim()
	kvDim := cfg.KVDim()
	hidden := cfg.Hidden
	vocab := cfg.Vocab

	h := &metalLayerHandles{
		qkv: make([][3]*MetalMatmul, cfg.NLayers),
		wo:  make([]*MetalMatmul, cfg.NLayers),
		w1:  make([]*MetalMatmul, cfg.NLayers),
		w3:  make([]*MetalMatmul, cfg.NLayers),
		w2:  make([]*MetalMatmul, cfg.NLayers),
	}

	for li := range e.mw.Layers {
		l := &e.mw.Layers[li]
		h.qkv[li][0] = NewMetalMatmul(l.Wq, qDim, dim)
		h.qkv[li][1] = NewMetalMatmul(l.Wk, kvDim, dim)
		h.qkv[li][2] = NewMetalMatmul(l.Wv, kvDim, dim)
		h.wo[li] = NewMetalMatmul(l.Wo, dim, qDim)
		h.w1[li] = NewMetalMatmul(l.W1, hidden, dim)
		h.w3[li] = NewMetalMatmul(l.W3, hidden, dim)
		h.w2[li] = NewMetalMatmul(l.W2, dim, hidden)

		// Verify critical handles were created successfully.
		if h.qkv[li][0] == nil || h.qkv[li][1] == nil || h.qkv[li][2] == nil ||
			h.wo[li] == nil || h.w1[li] == nil || h.w3[li] == nil || h.w2[li] == nil {
			cleanupMetalHandles(h)
			return false
		}
	}

	// Classifier: Embed is [vocab, dim] row-major.
	h.cls = NewMetalMatmul(e.mw.Embed, vocab, dim)

	e.metalHandles = h
	return true
}

// metalExecQKV runs Q, K, V matmuls on the GPU for layer li.
func (e *Engine) metalExecQKV(li int, q, k, v, xNorm []float32) {
	h := e.metalHandles.(*metalLayerHandles)
	h.qkv[li][0].Exec(q, xNorm)
	h.qkv[li][1].Exec(k, xNorm)
	h.qkv[li][2].Exec(v, xNorm)
}

// metalExecWo runs the Wo projection on the GPU for layer li.
func (e *Engine) metalExecWo(li int, out, attOut []float32) {
	h := e.metalHandles.(*metalLayerHandles)
	h.wo[li].Exec(out, attOut)
}

// metalExecFFN runs W1, W3, W2 matmuls on the GPU for layer li.
func (e *Engine) metalExecFFN(li int, h1, h3, ffOut, xNorm, gate []float32) {
	h := e.metalHandles.(*metalLayerHandles)
	h.w1[li].Exec(h1, xNorm)
	h.w3[li].Exec(h3, xNorm)
	siluMulAccel(gate, h1, h3)
	h.w2[li].Exec(ffOut, gate)
}

// metalExecClassifier runs the classifier matmul on the GPU.
// Returns false if no Metal classifier is available.
func (e *Engine) metalExecClassifier(logits, xNorm []float32) bool {
	h := e.metalHandles.(*metalLayerHandles)
	if h.cls == nil {
		return false
	}
	return h.cls.Exec(logits, xNorm)
}

// cleanupMetalMatmuls releases all cached Metal GPU handles.
func (e *Engine) cleanupMetalMatmuls() {
	h, ok := e.metalHandles.(*metalLayerHandles)
	if !ok || h == nil {
		e.metalHandles = nil
		return
	}
	cleanupMetalHandles(h)
	e.metalHandles = nil
}

func cleanupMetalHandles(h *metalLayerHandles) {
	for _, qkv := range h.qkv {
		for _, m := range qkv {
			if m != nil {
				m.Close()
			}
		}
	}
	h.qkv = nil
	for _, m := range h.wo {
		if m != nil {
			m.Close()
		}
	}
	h.wo = nil
	for _, m := range h.w1 {
		if m != nil {
			m.Close()
		}
	}
	h.w1 = nil
	for _, m := range h.w3 {
		if m != nil {
			m.Close()
		}
	}
	h.w3 = nil
	for _, m := range h.w2 {
		if m != nil {
			m.Close()
		}
	}
	h.w2 = nil
	if h.cls != nil {
		h.cls.Close()
		h.cls = nil
	}
}
