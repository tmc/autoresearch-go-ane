package storiesane

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

type layerCache struct {
	x         []float32
	xNorm     []float32
	attRRMS   []float32
	q         []float32
	k         []float32
	v         []float32
	attOut    []float32
	x2        []float32
	x2Norm    []float32
	ffnRRMS   []float32
	h1        []float32
	h3        []float32
	gate      []float32
	dOut      []float32
	dh1       []float32
	dh3       []float32
	dx2       []float32
	dx2Scaled []float32
	dq        []float32
	dk        []float32
	dv        []float32

	attTapsReady bool
	ffnTapsReady bool
}

type modelGrad struct {
	Layers   []stories.LayerWeights
	RMSFinal []float32
	Embed    []float32
}

func newLayerCache(seq int) layerCache {
	return layerCache{
		x:         make([]float32, stories.Dim*seq),
		xNorm:     make([]float32, stories.Dim*seq),
		attRRMS:   make([]float32, seq),
		q:         make([]float32, stories.Dim*seq),
		k:         make([]float32, stories.Dim*seq),
		v:         make([]float32, stories.Dim*seq),
		attOut:    make([]float32, stories.Dim*seq),
		x2:        make([]float32, stories.Dim*seq),
		x2Norm:    make([]float32, stories.Dim*seq),
		ffnRRMS:   make([]float32, seq),
		h1:        make([]float32, stories.Hidden*seq),
		h3:        make([]float32, stories.Hidden*seq),
		gate:      make([]float32, stories.Hidden*seq),
		dOut:      make([]float32, stories.Dim*seq),
		dh1:       make([]float32, stories.Hidden*seq),
		dh3:       make([]float32, stories.Hidden*seq),
		dx2:       make([]float32, stories.Dim*seq),
		dx2Scaled: make([]float32, stories.Dim*seq),
		dq:        make([]float32, stories.Dim*seq),
		dk:        make([]float32, stories.Dim*seq),
		dv:        make([]float32, stories.Dim*seq),
	}
}

func newLayerGrad() stories.LayerWeights {
	return stories.LayerWeights{
		Wq:     make([]float32, stories.WQSize),
		Wk:     make([]float32, stories.WQSize),
		Wv:     make([]float32, stories.WQSize),
		Wo:     make([]float32, stories.WOSize),
		W1:     make([]float32, stories.W1Size),
		W2:     make([]float32, stories.W2Size),
		W3:     make([]float32, stories.W3Size),
		RMSAtt: make([]float32, stories.Dim),
		RMSFFN: make([]float32, stories.Dim),
	}
}

func newModelGrad(vocab int) *modelGrad {
	g := &modelGrad{
		Layers:   make([]stories.LayerWeights, stories.NLayers),
		RMSFinal: make([]float32, stories.Dim),
		Embed:    make([]float32, vocab*stories.Dim),
	}
	for i := range g.Layers {
		g.Layers[i] = newLayerGrad()
	}
	return g
}

func newLayerGradFromConfig(cfg stories.ModelConfig) stories.LayerWeights {
	return stories.LayerWeights{
		Wq:     make([]float32, cfg.WqSize()),
		Wk:     make([]float32, cfg.WkSize()),
		Wv:     make([]float32, cfg.WvSize()),
		Wo:     make([]float32, cfg.WoSize()),
		W1:     make([]float32, cfg.W1Size()),
		W2:     make([]float32, cfg.W2Size()),
		W3:     make([]float32, cfg.W3Size()),
		RMSAtt: make([]float32, cfg.Dim),
		RMSFFN: make([]float32, cfg.Dim),
	}
}

func newModelGradFromConfig(cfg stories.ModelConfig) *modelGrad {
	g := &modelGrad{
		Layers:   make([]stories.LayerWeights, cfg.NLayers),
		RMSFinal: make([]float32, cfg.Dim),
		Embed:    make([]float32, cfg.Vocab*cfg.Dim),
	}
	for i := range g.Layers {
		g.Layers[i] = newLayerGradFromConfig(cfg)
	}
	return g
}

func clearLayerGrad(g *stories.LayerWeights) {
	clear(g.Wq)
	clear(g.Wk)
	clear(g.Wv)
	clear(g.Wo)
	clear(g.W1)
	clear(g.W2)
	clear(g.W3)
	clear(g.RMSAtt)
	clear(g.RMSFFN)
}

func clearModelGrad(g *modelGrad) {
	if g == nil {
		return
	}
	for i := range g.Layers {
		clearLayerGrad(&g.Layers[i])
	}
	clear(g.RMSFinal)
	clear(g.Embed)
}

func scaleLayerGrad(g *stories.LayerWeights, scale float32) {
	scaleGradSlice(g.Wq, scale)
	scaleGradSlice(g.Wk, scale)
	scaleGradSlice(g.Wv, scale)
	scaleGradSlice(g.Wo, scale)
	scaleGradSlice(g.W1, scale)
	scaleGradSlice(g.W2, scale)
	scaleGradSlice(g.W3, scale)
	scaleGradSlice(g.RMSAtt, scale)
	scaleGradSlice(g.RMSFFN, scale)
}

func scaleModelGrad(g *modelGrad, scale float32) {
	for i := range g.Layers {
		scaleLayerGrad(&g.Layers[i], scale)
	}
	scaleSlice(g.RMSFinal, scale)
	scaleSlice(g.Embed, scale)
}

func scaleSlice(v []float32, scale float32) {
	scaleSliceAccel(v, scale)
}

func sumSquares(v []float32) float64 {
	return sumSquaresGrad(v)
}

func (e *Engine) clipLayerGradients(layers []stories.LayerWeights, gRMS, gEmbed []float32) {
	if e == nil || e.gradClip <= 0 {
		return
	}
	var norm2 float64
	for i := range layers {
		g := &layers[i]
		norm2 += sumSquares(g.Wq)
		norm2 += sumSquares(g.Wk)
		norm2 += sumSquares(g.Wv)
		norm2 += sumSquares(g.Wo)
		norm2 += sumSquares(g.W1)
		norm2 += sumSquares(g.W2)
		norm2 += sumSquares(g.W3)
		norm2 += sumSquares(g.RMSAtt)
		norm2 += sumSquares(g.RMSFFN)
	}
	norm2 += sumSquares(gRMS)
	norm2 += sumSquares(gEmbed)
	if norm2 <= 0 {
		return
	}
	norm := float32(math.Sqrt(norm2))
	if norm <= e.gradClip {
		return
	}
	scale := e.gradClip / norm
	for i := range layers {
		scaleLayerGrad(&layers[i], scale)
	}
	scaleSlice(gRMS, scale)
	scaleSlice(gEmbed, scale)
}

func addLayerGrad(dst, src *stories.LayerWeights) {
	addSlice(dst.Wq, src.Wq)
	addSlice(dst.Wk, src.Wk)
	addSlice(dst.Wv, src.Wv)
	addSlice(dst.Wo, src.Wo)
	addSlice(dst.W1, src.W1)
	addSlice(dst.W2, src.W2)
	addSlice(dst.W3, src.W3)
	addSlice(dst.RMSAtt, src.RMSAtt)
	addSlice(dst.RMSFFN, src.RMSFFN)
}

func addSlice(dst, src []float32) {
	addSliceAccel(dst, src)
}

func accumLinearGradCF(dW, dy, x []float32, outCh, inCh, seq int) {
	if accumLinearGradCFAccelerate(dW, dy, x, outCh, inCh, seq) {
		return
	}
	for o := 0; o < outCh; o++ {
		row := dW[o*inCh : (o+1)*inCh]
		dyRow := dy[o*seq : (o+1)*seq]
		for i := 0; i < inCh; i++ {
			xRow := x[i*seq : (i+1)*seq]
			sum := float32(0)
			for t := 0; t < seq; t++ {
				sum += dyRow[t] * xRow[t]
			}
			row[i] += sum
		}
	}
}

// accumLinearGrad3CF fuses three weight gradient accumulations sharing the
// same input x into a single call, reducing CGo crossings.
func accumLinearGrad3CF(dW1, dy1, dW2, dy2, dW3, dy3, x []float32, outCh, inCh, seq int) {
	if accumLinearGrad3CFAccelerate(dW1, dy1, dW2, dy2, dW3, dy3, x, outCh, inCh, seq) {
		return
	}
	accumLinearGradCF(dW1, dy1, x, outCh, inCh, seq)
	accumLinearGradCF(dW2, dy2, x, outCh, inCh, seq)
	accumLinearGradCF(dW3, dy3, x, outCh, inCh, seq)
}

func linearBackwardDXCF(dx, w, dy []float32, outCh, inCh, seq int) {
	if linearBackwardDXCFAccelerate(dx, w, dy, outCh, inCh, seq) {
		return
	}
	for i := 0; i < inCh; i++ {
		row := dx[i*seq : (i+1)*seq]
		for t := range row {
			row[t] = 0
		}
		for o := 0; o < outCh; o++ {
			weight := w[o*inCh+i]
			dyRow := dy[o*seq : (o+1)*seq]
			for t := 0; t < seq; t++ {
				row[t] += weight * dyRow[t]
			}
		}
	}
}

func causalAttentionBackwardCF(dq, dk, dv, dOut, q, k, v []float32, heads, headDim, seq int) {
	clear(dq)
	clear(dk)
	clear(dv)
	scores := make([]float32, seq)
	probs := make([]float32, seq)
	dScores := make([]float32, seq)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	for h := 0; h < heads; h++ {
		base := h * headDim
		for t := 0; t < seq; t++ {
			maxv := float32(math.Inf(-1))
			for s := 0; s <= t; s++ {
				sum := float32(0)
				for i := 0; i < headDim; i++ {
					sum += q[(base+i)*seq+t] * k[(base+i)*seq+s]
				}
				scores[s] = sum * scale
				if scores[s] > maxv {
					maxv = scores[s]
				}
			}
			total := float64(0)
			for s := 0; s <= t; s++ {
				e := math.Exp(float64(scores[s] - maxv))
				probs[s] = float32(e)
				total += e
			}
			invTotal := float32(1.0 / total)
			for s := 0; s <= t; s++ {
				probs[s] *= invTotal
			}

			dsSum := float32(0)
			for s := 0; s <= t; s++ {
				dot := float32(0)
				for i := 0; i < headDim; i++ {
					dot += dOut[(base+i)*seq+t] * v[(base+i)*seq+s]
				}
				dScores[s] = dot
				dsSum += probs[s] * dot
			}

			for s := 0; s <= t; s++ {
				ds := probs[s] * (dScores[s] - dsSum) * scale
				for i := 0; i < headDim; i++ {
					qIdx := (base + i) * seq
					dq[qIdx+t] += ds * k[qIdx+s]
					dk[qIdx+s] += ds * q[qIdx+t]
					dv[qIdx+s] += probs[s] * dOut[qIdx+t]
				}
			}
		}
	}
}

func sampleFromLogits(rng *rand.Rand, logits []float32, temperature float32) int {
	if temperature <= 1e-6 {
		return argmax(logits)
	}
	maxv := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > maxv {
			maxv = logits[i]
		}
	}
	invTemp := float64(1 / temperature)
	total := 0.0
	for i := range logits {
		total += math.Exp((float64(logits[i]) - float64(maxv)) * invTemp)
	}
	target := rng.Float64() * total
	acc := 0.0
	for i := range logits {
		acc += math.Exp((float64(logits[i]) - float64(maxv)) * invTemp)
		if acc >= target {
			return i
		}
	}
	return len(logits) - 1
}

func argmax(v []float32) int {
	best := 0
	bestVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > bestVal {
			bestVal = v[i]
			best = i
		}
	}
	return best
}

func (e *Engine) disableLayerForward(err error) {
	if err == nil {
		return
	}
	for i := range e.layers {
		if e.layers[i] != nil {
			e.layers[i].close()
		}
	}
	e.layers = nil
	e.layersInit = true
	e.layerInitErr = err
}

func (e *Engine) forwardTraining(input []int32) ([]float32, error) {
	// Wait for any pending async weight refresh before reading kernels.
	e.waitAsyncRefresh()
	if e.useANE && e.ensureLayers() == nil {
		if out, err := e.forwardTrainingANE(input); err == nil {
			return out, nil
		} else {
			e.disableLayerForward(err)
		}
	}
	return e.forwardTrainingCPU(input), nil
}

func (e *Engine) forwardTrainingCPU(input []int32) []float32 {
	stories.EmbedLookup(e.x, e.mw.Embed, input, stories.Dim, e.seq)
	cur := e.x
	next := e.tmpHidden
	for i := range e.mw.Layers {
		layer := e.mw.Layers[i]
		cache := &e.caches[i]
		copy(cache.x, cur)
		rmsNormCFWithRRMS(cache.xNorm, cache.attRRMS, cache.x, layer.RMSAtt, stories.Dim, e.seq)
		linearCF(cache.q, layer.Wq, cache.xNorm, stories.Dim, stories.Dim, e.seq)
		linearCF(cache.k, layer.Wk, cache.xNorm, stories.Dim, stories.Dim, e.seq)
		applyRoPECFInPlace(cache.q, stories.Heads, stories.Dim/stories.Heads, e.seq, e.ropeCos, e.ropeSin)
		applyRoPECFInPlace(cache.k, stories.Heads, stories.Dim/stories.Heads, e.seq, e.ropeCos, e.ropeSin)
		linearCF(cache.v, layer.Wv, cache.xNorm, stories.Dim, stories.Dim, e.seq)
		causalAttentionCF(cache.attOut, cache.q, cache.k, cache.v, stories.Heads, stories.Dim/stories.Heads, e.seq)
		linearCF(next, layer.Wo, cache.attOut, stories.Dim, stories.Dim, e.seq)
		addScaledResidual(cache.x2, cache.x, next)
		rmsNormCFWithRRMS(cache.x2Norm, cache.ffnRRMS, cache.x2, layer.RMSFFN, stories.Dim, e.seq)
		linearCF(cache.h1, layer.W1, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
		linearCF(cache.h3, layer.W3, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
		for j := range cache.gate {
			cache.gate[j] = silu32(cache.h1[j]) * cache.h3[j]
		}
		cache.attTapsReady = true
		cache.ffnTapsReady = true
		linearCF(next, layer.W2, cache.gate, stories.Dim, stories.Hidden, e.seq)
		addScaledResidual(next, cache.x2, next)
		cur, next = next, cur
	}
	return cur
}

func (e *Engine) forwardTrainingANE(input []int32) ([]float32, error) {
	stories.EmbedLookup(e.x, e.mw.Embed, input, stories.Dim, e.seq)
	cur := e.x
	next := e.tmpHidden
	for i := range e.layers {
		cache := &e.caches[i]
		copy(cache.x, cur)
		if err := e.layers[i].runWithTaps(next, cur, cache); err != nil {
			return nil, fmt.Errorf("storiesane step: layer %d: %w", i, err)
		}
		cur, next = next, cur
	}
	return cur, nil
}

func (e *Engine) runFinalHead(finalHidden []float32, target []int32) (float32, error) {
	start := time.Now()
	e.ensureOffload()
	clear(e.gRMS)

	if e.off == nil || !e.off.hasRMSForward() {
		stories.RMSNorm(e.xNorm, finalHidden, e.mw.RMSFinal, stories.Dim, e.seq)
	} else if err := e.off.runRMSForward(e.xNorm, finalHidden); err != nil {
		e.off.disableRMSForward()
		stories.RMSNorm(e.xNorm, finalHidden, e.mw.RMSFinal, stories.Dim, e.seq)
	}

	loss := float32(0)
	validTargets := 0
	logitsScaled := false
	combinedSoftmax := false
	if e.off != nil && e.off.hasClassifierForward() && e.off.hasSoftmax() {
		if err := e.off.runClassifierSoftmax(e.logits, e.xNorm); err != nil {
			e.off.disableClassifierForward()
			e.off.disableSoftmax()
		} else {
			loss, validTargets = crossEntropyLossFromProbsUnscaled(e.logits, e.logits, target, stories.Vocab, e.seq)
			combinedSoftmax = true
		}
	}
	if !combinedSoftmax {
		if e.off == nil || !e.off.hasClassifierForward() {
			stories.MatMulVocabSeq(e.logits, e.mw.Embed, e.xNorm, stories.Vocab, stories.Dim, e.seq)
		} else if err := e.off.runClassifierForward(e.logits, e.xNorm); err != nil {
			e.off.disableClassifierForward()
			stories.MatMulVocabSeq(e.logits, e.mw.Embed, e.xNorm, stories.Vocab, stories.Dim, e.seq)
		}
		if e.off == nil || !e.off.hasSoftmax() {
			loss = stories.CrossEntropyLoss(e.logits, e.logits, target, stories.Vocab, e.seq)
			logitsScaled = true
		} else if err := e.off.runSoftmax(e.logits); err != nil {
			e.off.disableSoftmax()
			loss = stories.CrossEntropyLoss(e.logits, e.logits, target, stories.Vocab, e.seq)
			logitsScaled = true
		} else {
			loss, validTargets = crossEntropyLossFromProbsUnscaled(e.logits, e.logits, target, stories.Vocab, e.seq)
		}
	}
	scale := float32(1)
	if !logitsScaled && validTargets == 0 {
		scale = 0
	}
	if !logitsScaled && validTargets > 0 {
		scale = float32(1.0 / float64(validTargets))
	}
	gradScale := scale * e.lossScale

	embedAsync := e.off != nil && e.off.hasClassifierBackward()
	e.embedGradDone = nil
	gradLogits := e.logits
	if embedAsync {
		e.embedGradDone = make(chan struct{})
		done := e.embedGradDone
		embedScale := gradScale
		go func() {
			begin := time.Now()
			stories.MatMulGradEmbedScale(e.gEmbed, gradLogits, e.xNorm, stories.Vocab, stories.Dim, e.seq, embedScale)
			e.stepMetrics.addEmbedGrad(time.Since(begin))
			close(done)
		}()
	}
	if gradScale == 0 {
		clear(e.dy)
	} else if e.off == nil || !e.off.hasClassifierBackward() {
		stories.MatMulEmbedTScale(e.dy, e.mw.Embed, gradLogits, stories.Vocab, stories.Dim, e.seq, gradScale)
	} else if err := e.off.runClassifierBackward(e.dy, gradLogits); err != nil {
		e.off.disableClassifierBackward()
		stories.MatMulEmbedTScale(e.dy, e.mw.Embed, gradLogits, stories.Vocab, stories.Dim, e.seq, gradScale)
	} else if gradScale != 1 {
		scaleSlice(e.dy, gradScale)
	}
	if !embedAsync {
		begin := time.Now()
		stories.MatMulGradEmbedScale(e.gEmbed, gradLogits, e.xNorm, stories.Vocab, stories.Dim, e.seq, gradScale)
		e.stepMetrics.addEmbedGrad(time.Since(begin))
	}
	if e.off == nil || !e.off.hasRMSBackward() {
		stories.RMSNormBackward(e.dx, e.gRMS, e.dy, finalHidden, e.mw.RMSFinal, stories.Dim, e.seq)
	} else if err := e.off.runRMSBackward(e.dx, e.dy, finalHidden); err != nil {
		e.off.disableRMSBackward()
		stories.RMSNormBackward(e.dx, e.gRMS, e.dy, finalHidden, e.mw.RMSFinal, stories.Dim, e.seq)
	} else {
		begin := time.Now()
		rmsNormGradWeights(e.gRMS, e.dy, finalHidden, e.mw.RMSFinal, stories.Dim, e.seq)
		e.stepMetrics.addRMSDW(time.Since(begin))
	}
	e.stepMetrics.addFinalHead(time.Since(start))
	return loss, nil
}

func (e *Engine) runRMSBackwardLayer(dx, dw, dy, x, w, rrms []float32) {
	if e.off == nil || !e.off.hasRMSBackward() {
		stories.RMSNormBackward(dx, dw, dy, x, w, stories.Dim, e.seq)
		return
	}
	if err := e.off.runRMSBackwardWithWeights(dx, dy, x, w); err != nil {
		e.off.disableRMSBackward()
		stories.RMSNormBackward(dx, dw, dy, x, w, stories.Dim, e.seq)
		return
	}
	begin := time.Now()
	rmsNormGradWeightsWithRRMS(dw, dy, x, rrms, stories.Dim, e.seq)
	e.stepMetrics.addRMSDW(time.Since(begin))
}

func (e *Engine) ensureAttentionCache(layer *stories.LayerWeights, cache *layerCache) {
	if cache.attTapsReady {
		return
	}
	rmsNormCFWithRRMS(cache.xNorm, cache.attRRMS, cache.x, layer.RMSAtt, stories.Dim, e.seq)
	linearCF(cache.q, layer.Wq, cache.xNorm, stories.Dim, stories.Dim, e.seq)
	linearCF(cache.k, layer.Wk, cache.xNorm, stories.Dim, stories.Dim, e.seq)
	applyRoPECFInPlace(cache.q, stories.Heads, stories.Dim/stories.Heads, e.seq, e.ropeCos, e.ropeSin)
	applyRoPECFInPlace(cache.k, stories.Heads, stories.Dim/stories.Heads, e.seq, e.ropeCos, e.ropeSin)
	linearCF(cache.v, layer.Wv, cache.xNorm, stories.Dim, stories.Dim, e.seq)
	causalAttentionCF(cache.attOut, cache.q, cache.k, cache.v, stories.Heads, stories.Dim/stories.Heads, e.seq)
	cache.attTapsReady = true
}

func (e *Engine) ensureFFNCache(layer *stories.LayerWeights, cache *layerCache) {
	if cache.ffnTapsReady {
		return
	}
	rmsNormCFWithRRMS(cache.x2Norm, cache.ffnRRMS, cache.x2, layer.RMSFFN, stories.Dim, e.seq)
	linearCF(cache.h1, layer.W1, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
	linearCF(cache.h3, layer.W3, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
	for i := range cache.gate {
		cache.gate[i] = silu32(cache.h1[i]) * cache.h3[i]
	}
	cache.ffnTapsReady = true
}

func (e *Engine) backwardFFNCPU(layer *stories.LayerWeights, cache *layerCache, grad *stories.LayerWeights, dFFN, dPrev []float32) {
	e.ensureFFNCache(layer, cache)
	linearBackwardDXCF(e.gradGate, layer.W2, dFFN, stories.Dim, stories.Hidden, e.seq)
	siluBackwardAccel(e.gradH1, e.gradH3, e.gradGate, cache.h1, cache.h3)
	linearBackwardDXCF(e.gradXNorm, layer.W1, e.gradH1, stories.Hidden, stories.Dim, e.seq)
	linearBackwardDXCF(dPrev, layer.W3, e.gradH3, stories.Hidden, stories.Dim, e.seq)
	for i := range e.gradXNorm {
		e.gradXNorm[i] += dPrev[i]
	}
	stories.RMSNormBackward(dPrev, grad.RMSFFN, e.gradXNorm, cache.x2, layer.RMSFFN, stories.Dim, e.seq)
	for i := range e.gradX2 {
		e.gradX2[i] += dPrev[i]
	}
}

func (e *Engine) backwardFFNHybrid(lb *layerBackward, layer *stories.LayerWeights, cache *layerCache, grad *stories.LayerWeights, dFFN, dPrev []float32) error {
	e.ensureFFNCache(layer, cache)
	if err := lb.runFFN(e.gradXNorm, cache.dh1, cache.dh3, dFFN, cache.h1, cache.h3); err != nil {
		return err
	}
	// Submit dW jobs immediately after ANE outputs are ready, before RMS
	// backward CPU work, so CBLAS runs concurrently with the CPU reduction.
	e.submitDWJob(func() {
		accumLinearGradCF(grad.W2, cache.dOut, cache.gate, stories.Dim, stories.Hidden, e.seq)
		accumLinearGradCF(grad.W1, cache.dh1, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
		accumLinearGradCF(grad.W3, cache.dh3, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
	})
	e.runRMSBackwardLayer(dPrev, grad.RMSFFN, e.gradXNorm, cache.x2, layer.RMSFFN, cache.ffnRRMS)
	for i := range cache.dx2 {
		cache.dx2[i] += dPrev[i]
	}
	return nil
}

func (e *Engine) backwardAttentionHybridWithDW(lb *layerBackward, layer *stories.LayerWeights, cache *layerCache, grad *stories.LayerWeights, dx2, dx2Scaled, dPrev []float32) error {
	e.ensureAttentionCache(layer, cache)
	if err := lb.runAttention(e.gradXNorm, cache.dq, cache.dk, cache.dv, cache.q, cache.k, cache.v, dx2Scaled); err != nil {
		return err
	}
	// Submit dW jobs immediately after ANE outputs are ready, before RMS
	// backward CPU work, so CBLAS runs concurrently with the CPU reduction.
	e.submitDWJob(func() {
		accumLinearGradCF(grad.Wo, dx2Scaled, cache.attOut, stories.Dim, stories.Dim, e.seq)
	})
	e.submitDWJob(func() {
		accumLinearGrad3CF(grad.Wq, cache.dq, grad.Wk, cache.dk, grad.Wv, cache.dv, cache.xNorm, stories.Dim, stories.Dim, e.seq)
	})
	e.runRMSBackwardLayer(dPrev, grad.RMSAtt, e.gradXNorm, cache.x, layer.RMSAtt, cache.attRRMS)
	for i := range dPrev {
		dPrev[i] += dx2[i]
	}
	return nil
}

func (e *Engine) backwardAttentionCPU(layer *stories.LayerWeights, cache *layerCache, grad *stories.LayerWeights, dx2, dx2Scaled, dPrev []float32) {
	e.ensureAttentionCache(layer, cache)
	linearBackwardDXCF(e.gradAtt, layer.Wo, dx2Scaled, stories.Dim, stories.Dim, e.seq)
	causalAttentionBackwardCF(e.gradQ, e.gradK, e.gradV, e.gradAtt, cache.q, cache.k, cache.v, stories.Heads, stories.Dim/stories.Heads, e.seq)
	// Fuse Wq+Wk+Wv backward: dx = Wq^T @ dQ + Wk^T @ dK + Wv^T @ dV
	if !linearBackwardDX3AccumAccelerate(e.gradXNorm,
		layer.Wq, e.gradQ,
		layer.Wk, e.gradK,
		layer.Wv, e.gradV,
		stories.Dim, stories.Dim, e.seq) {
		linearBackwardDXCF(e.gradXNorm, layer.Wq, e.gradQ, stories.Dim, stories.Dim, e.seq)
		linearBackwardDXCF(dPrev, layer.Wk, e.gradK, stories.Dim, stories.Dim, e.seq)
		for i := range e.gradXNorm {
			e.gradXNorm[i] += dPrev[i]
		}
		linearBackwardDXCF(dPrev, layer.Wv, e.gradV, stories.Dim, stories.Dim, e.seq)
		for i := range e.gradXNorm {
			e.gradXNorm[i] += dPrev[i]
		}
	}
	stories.RMSNormBackward(dPrev, grad.RMSAtt, e.gradXNorm, cache.x, layer.RMSAtt, stories.Dim, e.seq)
	for i := range dPrev {
		dPrev[i] += dx2[i]
	}
}


func (e *Engine) backwardAndUpdate(input []int32) time.Duration {
	stepT := int(e.state.AdamT) + 1
	useHybrid := false
	if e.hybridBackwardRequested {
		if err := e.ensureBackward(); err == nil {
			useHybrid = true
		}
	}
	if e.accum != nil {
		return e.backwardAndAccumulate(input, useHybrid)
	}
	return e.backwardAndApply(input, stepT, useHybrid)
}

func (e *Engine) backwardAndAccumulate(input []int32, useHybrid bool) time.Duration {
	dCur := e.dx
	dPrev := e.gradPrev
	for l := stories.NLayers - 1; l >= 0; l-- {
		layer := &e.mw.Layers[l]
		cache := &e.caches[l]
		grad := &e.accum.Layers[l]

		copy(cache.dOut, dCur)
		scaleSlice(cache.dOut, layerResidualScale)
		if useHybrid {
			copy(cache.dx2, dCur)
			// backwardFFNHybrid submits dW jobs internally, right after
			// ANE outputs are ready, overlapping CBLAS with RMS backward.
			if err := e.backwardFFNHybrid(e.backward[l], layer, cache, grad, cache.dOut, dPrev); err != nil {
				e.disableHybridBackward(fmt.Errorf("storiesane step: layer %d hybrid ffn backward: %w", l, err))
				useHybrid = false
			}
		}
		if !useHybrid {
			copy(e.gradX2, dCur)
			e.backwardFFNCPU(layer, cache, grad, cache.dOut, dPrev)
			copy(cache.dh1, e.gradH1)
			copy(cache.dh3, e.gradH3)
			copy(cache.dx2, e.gradX2)
			e.submitDWJob(func() {
				accumLinearGradCF(grad.W2, cache.dOut, cache.gate, stories.Dim, stories.Hidden, e.seq)
				accumLinearGradCF(grad.W1, cache.dh1, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
				accumLinearGradCF(grad.W3, cache.dh3, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
			})
		}

		scaleInto(cache.dx2Scaled, cache.dx2, layerResidualScale)
		if useHybrid {
			// backwardAttentionHybridWithDW submits dW jobs internally,
			// right after ANE outputs are ready, overlapping CBLAS with
			// the RMS backward CPU work.
			if err := e.backwardAttentionHybridWithDW(e.backward[l], layer, cache, grad, cache.dx2, cache.dx2Scaled, dPrev); err != nil {
				e.disableHybridBackward(fmt.Errorf("storiesane step: layer %d hybrid attention backward: %w", l, err))
				useHybrid = false
			}
		}
		if !useHybrid {
			e.backwardAttentionCPU(layer, cache, grad, cache.dx2, cache.dx2Scaled, dPrev)
			copy(cache.dq, e.gradQ)
			copy(cache.dk, e.gradK)
			copy(cache.dv, e.gradV)
			e.submitDWJob(func() {
				accumLinearGradCF(grad.Wo, cache.dx2Scaled, cache.attOut, stories.Dim, stories.Dim, e.seq)
			})
			e.submitDWJob(func() {
				accumLinearGrad3CF(grad.Wq, cache.dq, grad.Wk, cache.dk, grad.Wv, cache.dv, cache.xNorm, stories.Dim, stories.Dim, e.seq)
			})
		}
		dCur, dPrev = dPrev, dCur
	}
	e.waitDWJobs()

	if e.embedGradDone != nil {
		<-e.embedGradDone
		e.embedGradDone = nil
	}
	begin := time.Now()
	stories.EmbedBackward(e.gEmbed, dCur, input, stories.Dim, e.seq)
	e.stepMetrics.addEmbedGrad(time.Since(begin))
	addSlice(e.accum.RMSFinal, e.gRMS)
	addSlice(e.accum.Embed, e.gEmbed)
	e.state.PendingSteps++
	if int(e.state.PendingSteps) >= e.accumSteps {
		return e.flushPending()
	}
	return 0
}

func (e *Engine) backwardAndApply(input []int32, stepT int, useHybrid bool) time.Duration {
	dCur := e.dx
	dPrev := e.gradPrev
	for l := stories.NLayers - 1; l >= 0; l-- {
		layer := &e.mw.Layers[l]
		cache := &e.caches[l]
		grad := &e.applyGrads[l]
		clearLayerGrad(grad)

		copy(cache.dOut, dCur)
		scaleSlice(cache.dOut, layerResidualScale)
		if useHybrid {
			copy(cache.dx2, dCur)
			if err := e.backwardFFNHybrid(e.backward[l], layer, cache, grad, cache.dOut, dPrev); err != nil {
				e.disableHybridBackward(fmt.Errorf("storiesane step: layer %d hybrid ffn backward: %w", l, err))
				useHybrid = false
			}
		}
		if !useHybrid {
			copy(e.gradX2, dCur)
			e.backwardFFNCPU(layer, cache, grad, cache.dOut, dPrev)
			copy(cache.dh1, e.gradH1)
			copy(cache.dh3, e.gradH3)
			copy(cache.dx2, e.gradX2)
			e.submitDWJob(func() {
				accumLinearGradCF(grad.W2, cache.dOut, cache.gate, stories.Dim, stories.Hidden, e.seq)
				accumLinearGradCF(grad.W1, cache.dh1, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
				accumLinearGradCF(grad.W3, cache.dh3, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
			})
		}
		scaleInto(cache.dx2Scaled, cache.dx2, layerResidualScale)
		if useHybrid {
			if err := e.backwardAttentionHybridWithDW(e.backward[l], layer, cache, grad, cache.dx2, cache.dx2Scaled, dPrev); err != nil {
				e.disableHybridBackward(fmt.Errorf("storiesane step: layer %d hybrid attention backward: %w", l, err))
				useHybrid = false
			}
		}
		if !useHybrid {
			e.backwardAttentionCPU(layer, cache, grad, cache.dx2, cache.dx2Scaled, dPrev)
			copy(cache.dq, e.gradQ)
			copy(cache.dk, e.gradK)
			copy(cache.dv, e.gradV)
			e.submitDWJob(func() {
				accumLinearGradCF(grad.Wo, cache.dx2Scaled, cache.attOut, stories.Dim, stories.Dim, e.seq)
			})
			e.submitDWJob(func() {
				accumLinearGrad3CF(grad.Wq, cache.dq, grad.Wk, cache.dk, grad.Wv, cache.dv, cache.xNorm, stories.Dim, stories.Dim, e.seq)
			})
		}
		dCur, dPrev = dPrev, dCur
	}
	e.waitDWJobs()

	if e.embedGradDone != nil {
		<-e.embedGradDone
		e.embedGradDone = nil
	}
	begin := time.Now()
	stories.EmbedBackward(e.gEmbed, dCur, input, stories.Dim, e.seq)
	e.stepMetrics.addEmbedGrad(time.Since(begin))
	if e.lossScale != 0 && e.lossScale != 1 {
		invLossScale := float32(1.0 / float64(e.lossScale))
		for l := range e.applyGrads {
			scaleLayerGrad(&e.applyGrads[l], invLossScale)
		}
		scaleSlice(e.gRMS, invLossScale)
		scaleSlice(e.gEmbed, invLossScale)
	}
	e.clipLayerGradients(e.applyGrads, e.gRMS, e.gEmbed)
	adamStart := time.Now()
	invBC1, invBC2 := adamBiasCorrectionInv(stepT, e.adamBeta1, e.adamBeta2)
	e.applyLayerAdamAll(e.applyGrads, stepT, invBC1, invBC2)
	adamUpdateCFWithInv(e.mw.RMSFinal, e.gRMS, &e.opt.RMSFinal, e.lr, e.adamBeta1, e.adamBeta2, e.adamEps, 0, invBC1, invBC2, false)
	adamUpdateCFWithInv(e.mw.Embed, e.gEmbed, &e.opt.Embed, e.lr, e.adamBeta1, e.adamBeta2, e.adamEps, e.weightDecay, invBC1, invBC2, true)
	e.stepMetrics.addAdam(time.Since(adamStart))
	compileDur := e.refreshANERuntimeForWeights()
	e.state.AdamT = uint32(stepT)
	e.state.CumBatches++
	return compileDur
}

func (e *Engine) submitDWJob(fn func()) {
	if e == nil || fn == nil {
		return
	}
	run := func() {
		begin := time.Now()
		fn()
		e.stepMetrics.addDWGEMM(time.Since(begin))
	}
	if e.gradTasks == nil {
		run()
		return
	}
	e.gradTasks.Go(run)
}

func (e *Engine) waitDWJobs() {
	if e == nil || e.gradTasks == nil {
		return
	}
	begin := time.Now()
	e.gradTasks.Wait()
	e.stepMetrics.addDWWait(time.Since(begin))
}

func (e *Engine) applyLayerAdamAll(grads []stories.LayerWeights, t int, invBC1, invBC2 float32) {
	_, err := compileParallel(len(e.mw.Layers), func(i int) (struct{}, error) {
		applyLayerAdam(
			&e.mw.Layers[i],
			&grads[i],
			&e.opt.Layers[i],
			e.lr,
			e.adamBeta1,
			e.adamBeta2,
			e.adamEps,
			e.weightDecay,
			invBC1,
			invBC2,
		)
		return struct{}{}, nil
	}, func(struct{}) {})
	if err != nil {
		for i := range e.mw.Layers {
			applyLayerAdam(
				&e.mw.Layers[i],
				&grads[i],
				&e.opt.Layers[i],
				e.lr,
				e.adamBeta1,
				e.adamBeta2,
				e.adamEps,
				e.weightDecay,
				invBC1,
				invBC2,
			)
		}
	}
}

func applyLayerAdam(dst *stories.LayerWeights, grad *stories.LayerWeights, st *stories.LayerOptimState, lr, b1, b2, eps, wd, invBC1, invBC2 float32) {
	adamUpdateCFWithInv(dst.Wq, grad.Wq, &st.Wq, lr, b1, b2, eps, wd, invBC1, invBC2, false)
	adamUpdateCFWithInv(dst.Wk, grad.Wk, &st.Wk, lr, b1, b2, eps, wd, invBC1, invBC2, false)
	adamUpdateCFWithInv(dst.Wv, grad.Wv, &st.Wv, lr, b1, b2, eps, wd, invBC1, invBC2, false)
	adamUpdateCFWithInv(dst.Wo, grad.Wo, &st.Wo, lr, b1, b2, eps, wd, invBC1, invBC2, false)
	adamUpdateCFWithInv(dst.W1, grad.W1, &st.W1, lr, b1, b2, eps, wd, invBC1, invBC2, false)
	adamUpdateCFWithInv(dst.W2, grad.W2, &st.W2, lr, b1, b2, eps, wd, invBC1, invBC2, false)
	adamUpdateCFWithInv(dst.W3, grad.W3, &st.W3, lr, b1, b2, eps, wd, invBC1, invBC2, false)
	adamUpdateCFWithInv(dst.RMSAtt, grad.RMSAtt, &st.RMSAtt, lr, b1, b2, eps, 0, invBC1, invBC2, false)
	adamUpdateCFWithInv(dst.RMSFFN, grad.RMSFFN, &st.RMSFFN, lr, b1, b2, eps, 0, invBC1, invBC2, false)
}

func adamUpdateCF(w, g []float32, st *stories.AdamState, t int, lr, b1, b2, eps, wd float32) {
	invBC1, invBC2 := adamBiasCorrectionInv(t, b1, b2)
	adamUpdateCFWithInv(w, g, st, lr, b1, b2, eps, wd, invBC1, invBC2, true)
}

func adamBiasCorrectionInv(t int, b1, b2 float32) (invBC1, invBC2 float32) {
	bc1 := float32(1.0 - math.Pow(float64(b1), float64(t)))
	bc2 := float32(1.0 - math.Pow(float64(b2), float64(t)))
	return 1 / bc1, 1 / bc2
}

func adamUpdateCFWithInv(w, g []float32, st *stories.AdamState, lr, b1, b2, eps, wd, invBC1, invBC2 float32, parallel bool) {
	if len(w) == 0 {
		return
	}
	update := func(start, end int) {
		if adamUpdateCFAccelerateChunk(
			w[start:end],
			g[start:end],
			st.M[start:end],
			st.V[start:end],
			b1,
			b2,
			invBC1,
			invBC2,
			lr,
			eps,
			wd,
		) {
			return
		}
		for i := start; i < end; i++ {
			st.M[i] = b1*st.M[i] + (1-b1)*g[i]
			st.V[i] = b2*st.V[i] + (1-b2)*g[i]*g[i]
			mh := st.M[i] * invBC1
			vh := st.V[i] * invBC2
			update := mh / (float32(math.Sqrt(float64(vh))) + eps)
			if wd != 0 {
				update += wd * w[i]
			}
			w[i] -= lr * update
		}
	}
	if !parallel || len(w) < 1<<20 {
		update(0, len(w))
		return
	}
	parallelForCF(len(w), update)
}
