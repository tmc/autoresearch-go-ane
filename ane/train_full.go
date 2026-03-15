package ane

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

type layerCache struct {
	x        []float32
	xNorm    []float32
	q        []float32
	k        []float32
	v        []float32
	attOut   []float32
	x2       []float32
	x2Norm   []float32
	h1       []float32
	gate     []float32
	preMix   []float32 // pre-residual-lambda output for backward
	qPreNorm []float32 // Q before QK-norm (saved for backward)
	kPreNorm []float32 // K before QK-norm (saved for backward)
	dOut     []float32
	dh1      []float32
	dx2      []float32
	dq       []float32
	dk       []float32
	dv       []float32
	ve       []float32 // VE lookup result, dim*seq, VE layers only
	veGateAct []float32 // per-head sigmoid gate activation, Heads*seq, VE layers only

	attTapsReady bool
	ffnTapsReady bool
}

type modelGrad struct {
	Layers        []stories.LayerWeights
	Embed         []float32
	ResidLambdas  []float32
	X0Lambdas     []float32
	SmearGate     []float32
	SmearLambda   []float32
	BackoutLambda []float32
}

func newLayerCache(seq, layer int) layerCache {
	lc := layerCache{
		x:        make([]float32, stories.Dim*seq),
		xNorm:    make([]float32, stories.Dim*seq),
		q:        make([]float32, stories.Dim*seq),
		k:        make([]float32, stories.Dim*seq),
		v:        make([]float32, stories.Dim*seq),
		attOut:   make([]float32, stories.Dim*seq),
		x2:       make([]float32, stories.Dim*seq),
		x2Norm:   make([]float32, stories.Dim*seq),
		h1:       make([]float32, stories.Hidden*seq),
		gate:     make([]float32, stories.Hidden*seq),
		preMix:   make([]float32, stories.Dim*seq),
		qPreNorm: make([]float32, stories.Dim*seq),
		kPreNorm: make([]float32, stories.Dim*seq),
		dOut:     make([]float32, stories.Dim*seq),
		dh1:      make([]float32, stories.Hidden*seq),
		dx2:      make([]float32, stories.Dim*seq),
		dq:       make([]float32, stories.Dim*seq),
		dk:       make([]float32, stories.Dim*seq),
		dv:       make([]float32, stories.Dim*seq),
	}
	if stories.IsVELayer(layer) {
		lc.ve = make([]float32, stories.Dim*seq)
		lc.veGateAct = make([]float32, stories.Heads*seq)
	}
	return lc
}

func newLayerGrad(layer int) stories.LayerWeights {
	lw := stories.LayerWeights{
		Wq: make([]float32, stories.WQSize),
		Wk: make([]float32, stories.WQSize),
		Wv: make([]float32, stories.WQSize),
		Wo: make([]float32, stories.WOSize),
		W1: make([]float32, stories.W1Size),
		W2: make([]float32, stories.W2Size),
	}
	if stories.IsVELayer(layer) {
		lw.VEEmbed = make([]float32, stories.VEEmbedSize(stories.Vocab))
		lw.VEGate = make([]float32, stories.VEGateSize())
	}
	return lw
}

func newModelGrad(vocab int) *modelGrad {
	g := &modelGrad{
		Layers:        make([]stories.LayerWeights, stories.NLayers),
		Embed:         make([]float32, vocab*stories.Dim),
		ResidLambdas:  make([]float32, stories.NLayers),
		X0Lambdas:     make([]float32, stories.NLayers),
		SmearGate:     make([]float32, stories.Dim*stories.Dim),
		SmearLambda:   make([]float32, 1),
		BackoutLambda: make([]float32, 1),
	}
	for i := range g.Layers {
		g.Layers[i] = newLayerGrad(i)
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
	clear(g.VEEmbed)
	clear(g.VEGate)
}

func clearModelGrad(g *modelGrad) {
	if g == nil {
		return
	}
	for i := range g.Layers {
		clearLayerGrad(&g.Layers[i])
	}
	clear(g.Embed)
	clear(g.ResidLambdas)
	clear(g.X0Lambdas)
	clear(g.SmearGate)
	clear(g.SmearLambda)
	clear(g.BackoutLambda)
}

func scaleLayerGrad(g *stories.LayerWeights, scale float32) {
	scaleGradSlice(g.Wq, scale)
	scaleGradSlice(g.Wk, scale)
	scaleGradSlice(g.Wv, scale)
	scaleGradSlice(g.Wo, scale)
	scaleGradSlice(g.W1, scale)
	scaleGradSlice(g.W2, scale)
	scaleGradSlice(g.VEEmbed, scale)
	scaleGradSlice(g.VEGate, scale)
}

func scaleModelGrad(g *modelGrad, scale float32) {
	for i := range g.Layers {
		scaleLayerGrad(&g.Layers[i], scale)
	}
	scaleSlice(g.Embed, scale)
	scaleSlice(g.ResidLambdas, scale)
	scaleSlice(g.X0Lambdas, scale)
	scaleSlice(g.SmearGate, scale)
	scaleSlice(g.SmearLambda, scale)
	scaleSlice(g.BackoutLambda, scale)
}

func scaleSlice(v []float32, scale float32) {
	scaleSliceAccel(v, scale)
}

func sumSquares(v []float32) float64 {
	return sumSquaresGrad(v)
}

func (e *Engine) clipLayerGradients(layers []stories.LayerWeights, gEmbed []float32) {
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
		norm2 += sumSquares(g.VEEmbed)
		norm2 += sumSquares(g.VEGate)
	}
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
	scaleSlice(gEmbed, scale)
}

func addLayerGrad(dst, src *stories.LayerWeights) {
	addSlice(dst.Wq, src.Wq)
	addSlice(dst.Wk, src.Wk)
	addSlice(dst.Wv, src.Wv)
	addSlice(dst.Wo, src.Wo)
	addSlice(dst.W1, src.W1)
	addSlice(dst.W2, src.W2)
	addSlice(dst.VEEmbed, src.VEEmbed)
	addSlice(dst.VEGate, src.VEGate)
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
	// QK-norm means attention scale is 1.0 (no 1/sqrt(headDim)).
	scale := float32(1.0)
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

func (e *Engine) forwardTraining(input []uint16) ([]float32, error) {
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

func (e *Engine) forwardTrainingCPU(input []uint16) []float32 {
	stories.EmbedLookup(e.x, e.mw.Embed, input, stories.Dim, e.seq)
	// Save pre-norm embedding for backward.
	copy(e.xPreEmbedNorm, e.x)
	// Embedding norm (parameterless).
	rmsNormNoWeightCF(e.x, e.x, stories.Dim, e.seq)
	// Save x0 for residual lambdas.
	copy(e.x0, e.x)
	smearForwardWithSaveCF(e.x, e.smearXPre, e.mw.SmearGate, e.mw.SmearLambda[0],
		e.smearGatePre, e.smearShifted, e.smearGateAct, stories.Dim, e.seq)
	cur := e.x
	next := e.tmpHidden
	midLayer := stories.NLayers / 2
	for i := range e.mw.Layers {
		layer := e.mw.Layers[i]
		cache := &e.caches[i]
		copy(cache.x, cur)
		rmsNormNoWeightCF(cache.xNorm, cache.x, stories.Dim, e.seq)
		linearCF(cache.q, layer.Wq, cache.xNorm, stories.Dim, stories.Dim, e.seq)
		linearCF(cache.k, layer.Wk, cache.xNorm, stories.Dim, stories.Dim, e.seq)
		applyRoPECFInPlace(cache.q, stories.Heads, stories.Dim/stories.Heads, e.seq, e.ropeCos, e.ropeSin)
		applyRoPECFInPlace(cache.k, stories.Heads, stories.Dim/stories.Heads, e.seq, e.ropeCos, e.ropeSin)
		// QK-norm: save pre-norm Q,K for backward, then apply per-head RMSNorm*1.2.
		copy(cache.qPreNorm, cache.q)
		copy(cache.kPreNorm, cache.k)
		qkNormCF(cache.q, cache.k, stories.Dim, stories.Heads, e.seq)
		linearCF(cache.v, layer.Wv, cache.xNorm, stories.Dim, stories.Dim, e.seq)
		if stories.IsVELayer(i) {
			veForwardCF(cache.v, cache.ve, cache.veGateAct, layer.VEEmbed, layer.VEGate, cache.x, input, stories.Dim, e.seq)
		}
		causalAttentionCF(cache.attOut, cache.q, cache.k, cache.v, stories.Heads, stories.Dim/stories.Heads, e.seq)
		linearCF(next, layer.Wo, cache.attOut, stories.Dim, stories.Dim, e.seq)
		// Residual with per-layer lambdas: x2 = rl*x + next.
		rl := e.mw.ResidLambdas[i]
		for j := 0; j < stories.Dim*e.seq; j++ {
			cache.x2[j] = rl*cache.x[j] + next[j]
		}
		// Add x0 lambda contribution: x2 += xl*x0.
		xl := e.mw.X0Lambdas[i]
		if xl != 0 {
			for j := 0; j < stories.Dim*e.seq; j++ {
				cache.x2[j] += xl * e.x0[j]
			}
		}
		rmsNormNoWeightCF(cache.x2Norm, cache.x2, stories.Dim, e.seq)
		linearCF(cache.h1, layer.W1, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
		for j := range cache.gate {
			cache.gate[j] = reluSquared32(cache.h1[j])
		}
		cache.attTapsReady = true
		cache.ffnTapsReady = true
		linearCF(next, layer.W2, cache.gate, stories.Dim, stories.Hidden, e.seq)
		// Save pre-mix for backward.
		copy(cache.preMix, next)
		// FFN residual with per-layer lambdas: out = rl*x2 + next + xl*x0.
		for j := 0; j < stories.Dim*e.seq; j++ {
			next[j] = rl*cache.x2[j] + next[j]
		}
		if xl != 0 {
			for j := 0; j < stories.Dim*e.seq; j++ {
				next[j] += xl * e.x0[j]
			}
		}
		cur, next = next, cur
		if i == midLayer-1 {
			copy(e.xMid, cur)
		}
	}
	backoutForwardCF(cur, e.xMid, e.mw.BackoutLambda[0], stories.Dim, e.seq)
	return cur
}

func (e *Engine) forwardTrainingANE(input []uint16) ([]float32, error) {
	stories.EmbedLookup(e.x, e.mw.Embed, input, stories.Dim, e.seq)
	copy(e.xPreEmbedNorm, e.x)
	rmsNormNoWeightCF(e.x, e.x, stories.Dim, e.seq)
	copy(e.x0, e.x)
	smearForwardWithSaveCF(e.x, e.smearXPre, e.mw.SmearGate, e.mw.SmearLambda[0],
		e.smearGatePre, e.smearShifted, e.smearGateAct, stories.Dim, e.seq)
	cur := e.x
	next := e.tmpHidden
	midLayer := stories.NLayers / 2
	for i := range e.layers {
		cache := &e.caches[i]
		copy(cache.x, cur)
		if err := e.layers[i].runWithTaps(next, cur, cache); err != nil {
			return nil, fmt.Errorf("storiesane step: layer %d: %w", i, err)
		}
		cur, next = next, cur
		if i == midLayer-1 {
			copy(e.xMid, cur)
		}
	}
	backoutForwardCF(cur, e.xMid, e.mw.BackoutLambda[0], stories.Dim, e.seq)
	return cur, nil
}

func (e *Engine) runFinalHead(finalHidden []float32, target []uint16) (float32, error) {
	start := time.Now()
	e.ensureOffload()

	// Parameterless final RMSNorm.
	stories.RMSNormNoWeight(e.xNorm, finalHidden, stories.Dim, e.seq)

	// Classifier: logits = Embed @ xNorm.
	if e.off == nil || !e.off.hasClassifierForward() {
		stories.MatMulVocabSeq(e.logits, e.mw.Embed, e.xNorm, stories.Vocab, stories.Dim, e.seq)
	} else if err := e.off.runClassifierForward(e.logits, e.xNorm); err != nil {
		e.off.disableClassifierForward()
		stories.MatMulVocabSeq(e.logits, e.mw.Embed, e.xNorm, stories.Vocab, stories.Dim, e.seq)
	}

	// Logit softcap: 15*tanh(logits/15).
	copy(e.logitsPreSoftcap, e.logits)
	logitSoftcap(e.logits)

	loss := float32(0)
	validTargets := 0
	logitsScaled := false
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

	scale := float32(1)
	if !logitsScaled && validTargets == 0 {
		scale = 0
	}
	if !logitsScaled && validTargets > 0 {
		scale = float32(1.0 / float64(validTargets))
	}
	gradScale := scale * e.lossScale

	// Softcap backward: multiply grad by 1-tanh^2(x/15).
	gradLogits := e.logits
	logitSoftcapBackward(gradLogits, e.logitsPreSoftcap)

	embedAsync := e.off != nil && e.off.hasClassifierBackward()
	e.embedGradDone = nil
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
	// Parameterless final RMSNorm backward.
	stories.RMSNormNoWeightBackward(e.dx, e.dy, finalHidden, stories.Dim, e.seq)
	e.stepMetrics.addFinalHead(time.Since(start))
	return loss, nil
}

func (e *Engine) ensureAttentionCache(layerIdx int, layer *stories.LayerWeights, cache *layerCache, input []uint16) {
	if cache.attTapsReady {
		return
	}
	rmsNormNoWeightCF(cache.xNorm, cache.x, stories.Dim, e.seq)
	linearCF(cache.q, layer.Wq, cache.xNorm, stories.Dim, stories.Dim, e.seq)
	linearCF(cache.k, layer.Wk, cache.xNorm, stories.Dim, stories.Dim, e.seq)
	applyRoPECFInPlace(cache.q, stories.Heads, stories.Dim/stories.Heads, e.seq, e.ropeCos, e.ropeSin)
	applyRoPECFInPlace(cache.k, stories.Heads, stories.Dim/stories.Heads, e.seq, e.ropeCos, e.ropeSin)
	copy(cache.qPreNorm, cache.q)
	copy(cache.kPreNorm, cache.k)
	qkNormCF(cache.q, cache.k, stories.Dim, stories.Heads, e.seq)
	linearCF(cache.v, layer.Wv, cache.xNorm, stories.Dim, stories.Dim, e.seq)
	if stories.IsVELayer(layerIdx) {
		veForwardCF(cache.v, cache.ve, cache.veGateAct, layer.VEEmbed, layer.VEGate, cache.x, input, stories.Dim, e.seq)
	}
	causalAttentionCF(cache.attOut, cache.q, cache.k, cache.v, stories.Heads, stories.Dim/stories.Heads, e.seq)
	cache.attTapsReady = true
}

func (e *Engine) ensureFFNCache(layer *stories.LayerWeights, cache *layerCache) {
	if cache.ffnTapsReady {
		return
	}
	rmsNormNoWeightCF(cache.x2Norm, cache.x2, stories.Dim, e.seq)
	linearCF(cache.h1, layer.W1, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
	for i := range cache.gate {
		cache.gate[i] = reluSquared32(cache.h1[i])
	}
	cache.ffnTapsReady = true
}

func (e *Engine) backwardFFNCPU(layerIdx int, layer *stories.LayerWeights, cache *layerCache, grad *stories.LayerWeights, dFFN, dPrev []float32) {
	e.ensureFFNCache(layer, cache)
	linearBackwardDXCF(e.gradGate, layer.W2, dFFN, stories.Dim, stories.Hidden, e.seq)
	reluSquaredBackwardSimple(e.gradH1, e.gradGate, cache.h1)
	linearBackwardDXCF(e.gradXNorm, layer.W1, e.gradH1, stories.Hidden, stories.Dim, e.seq)
	rmsNormNoWeightBackwardCF(dPrev, e.gradXNorm, cache.x2, stories.Dim, e.seq)
	for i := range e.gradX2 {
		e.gradX2[i] += dPrev[i]
	}
}

func (e *Engine) backwardFFNHybrid(lb *layerBackward, layerIdx int, layer *stories.LayerWeights, cache *layerCache, grad *stories.LayerWeights, dFFN, dPrev []float32) error {
	e.ensureFFNCache(layer, cache)
	linearBackwardDXCF(e.gradGate, layer.W2, dFFN, stories.Dim, stories.Hidden, e.seq)
	reluSquaredBackwardSimple(cache.dh1, e.gradGate, cache.h1)
	linearBackwardDXCF(e.gradXNorm, layer.W1, cache.dh1, stories.Hidden, stories.Dim, e.seq)
	e.submitDWJob(func() {
		accumLinearGradCF(grad.W2, cache.dOut, cache.gate, stories.Dim, stories.Hidden, e.seq)
		accumLinearGradCF(grad.W1, cache.dh1, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
	})
	rmsNormNoWeightBackwardCF(dPrev, e.gradXNorm, cache.x2, stories.Dim, e.seq)
	for i := range cache.dx2 {
		cache.dx2[i] += dPrev[i]
	}
	return nil
}

func (e *Engine) backwardAttentionHybridWithDW(layerIdx int, lb *layerBackward, layer *stories.LayerWeights, cache *layerCache, grad *stories.LayerWeights, dx2, dPrev []float32, input []uint16) error {
	e.ensureAttentionCache(layerIdx, layer, cache, input)
	if err := lb.runAttention(e.gradXNorm, cache.dq, cache.dk, cache.dv, cache.q, cache.k, cache.v, dx2); err != nil {
		return err
	}
	e.submitDWJob(func() {
		accumLinearGradCF(grad.Wo, dx2, cache.attOut, stories.Dim, stories.Dim, e.seq)
	})
	// QK-norm backward on dq/dk.
	qkNormBackwardCF(cache.dq, cache.dk, cache.qPreNorm, cache.kPreNorm, stories.Dim, stories.Heads, e.seq)
	e.submitDWJob(func() {
		accumLinearGrad3CF(grad.Wq, cache.dq, grad.Wk, cache.dk, grad.Wv, cache.dv, cache.xNorm, stories.Dim, stories.Dim, e.seq)
	})
	rmsNormNoWeightBackwardCF(dPrev, e.gradXNorm, cache.x, stories.Dim, e.seq)
	if stories.IsVELayer(layerIdx) {
		veBackwardCF(grad.VEEmbed, grad.VEGate, dPrev, cache.dv, cache.ve, cache.veGateAct, layer.VEEmbed, layer.VEGate, cache.x, input, stories.Dim, e.seq)
	}
	// Residual lambda backward for attention block.
	rl := e.mw.ResidLambdas[layerIdx]
	for i := range dPrev {
		dPrev[i] += rl * dx2[i]
	}
	return nil
}

func (e *Engine) backwardAttentionCPU(layerIdx int, layer *stories.LayerWeights, cache *layerCache, grad *stories.LayerWeights, dx2, dPrev []float32, input []uint16) {
	e.ensureAttentionCache(layerIdx, layer, cache, input)
	linearBackwardDXCF(e.gradAtt, layer.Wo, dx2, stories.Dim, stories.Dim, e.seq)
	causalAttentionBackwardCF(e.gradQ, e.gradK, e.gradV, e.gradAtt, cache.q, cache.k, cache.v, stories.Heads, stories.Dim/stories.Heads, e.seq)
	// QK-norm backward.
	qkNormBackwardCF(e.gradQ, e.gradK, cache.qPreNorm, cache.kPreNorm, stories.Dim, stories.Heads, e.seq)
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
	rmsNormNoWeightBackwardCF(dPrev, e.gradXNorm, cache.x, stories.Dim, e.seq)
	if stories.IsVELayer(layerIdx) {
		veBackwardCF(grad.VEEmbed, grad.VEGate, dPrev, e.gradV, cache.ve, cache.veGateAct, layer.VEEmbed, layer.VEGate, cache.x, input, stories.Dim, e.seq)
	}
	// Residual lambda backward for attention block.
	rl := e.mw.ResidLambdas[layerIdx]
	for i := range dPrev {
		dPrev[i] += rl * dx2[i]
	}
}

func (e *Engine) backwardAndUpdate(input []uint16) time.Duration {
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

func (e *Engine) backwardAndAccumulate(input []uint16, useHybrid bool) time.Duration {
	dCur := e.dx
	dPrev := e.gradPrev
	// Backout backward.
	clear(e.gBackoutLambda)
	backoutBackwardCF(dCur, e.xMid, e.mw.BackoutLambda[0], e.gBackoutLambda, stories.Dim, e.seq)
	backoutLam := e.mw.BackoutLambda[0]
	for i := range dCur {
		e.xMid[i] = -backoutLam * dCur[i]
	}
	midLayer := stories.NLayers / 2
	// Accumulate x0 gradient from all layers.
	clear(e.gX0)
	for l := stories.NLayers - 1; l >= 0; l-- {
		layer := &e.mw.Layers[l]
		cache := &e.caches[l]
		grad := &e.accum.Layers[l]
		rl := e.mw.ResidLambdas[l]

		// FFN residual backward: dOut comes scaled by rl for x2, rest is dPreMix.
		copy(cache.dOut, dCur)
		if useHybrid {
			copy(cache.dx2, dCur)
			scaleSlice(cache.dx2, rl)
			if err := e.backwardFFNHybrid(e.backward[l], l, layer, cache, grad, cache.dOut, dPrev); err != nil {
				e.disableHybridBackward(fmt.Errorf("storiesane step: layer %d hybrid ffn backward: %w", l, err))
				useHybrid = false
			}
		}
		if !useHybrid {
			copy(e.gradX2, dCur)
			scaleSlice(e.gradX2, rl)
			e.backwardFFNCPU(l, layer, cache, grad, cache.dOut, dPrev)
			copy(cache.dh1, e.gradH1)
			copy(cache.dx2, e.gradX2)
			e.submitDWJob(func() {
				accumLinearGradCF(grad.W2, cache.dOut, cache.gate, stories.Dim, stories.Hidden, e.seq)
				accumLinearGradCF(grad.W1, cache.dh1, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
			})
		}

		// Compute dResidLambda for FFN residual: sum(dCur * x2).
		var dRL float64
		for j := 0; j < stories.Dim*e.seq; j++ {
			dRL += float64(dCur[j] * cache.x2[j])
		}
		// Compute dX0Lambda for FFN residual: sum(dCur * x0).
		xl := e.mw.X0Lambdas[l]
		var dXL float64
		if xl != 0 || true {
			for j := 0; j < stories.Dim*e.seq; j++ {
				dXL += float64(dCur[j] * e.x0[j])
			}
		}
		// x0 gradient from FFN residual.
		for j := 0; j < stories.Dim*e.seq; j++ {
			e.gX0[j] += xl * dCur[j]
		}

		// Attention residual backward.
		attDx2 := cache.dx2
		if useHybrid {
			if err := e.backwardAttentionHybridWithDW(l, e.backward[l], layer, cache, grad, attDx2, dPrev, input); err != nil {
				e.disableHybridBackward(fmt.Errorf("storiesane step: layer %d hybrid attention backward: %w", l, err))
				useHybrid = false
			}
		}
		if !useHybrid {
			e.backwardAttentionCPU(l, layer, cache, grad, attDx2, dPrev, input)
			copy(cache.dq, e.gradQ)
			copy(cache.dk, e.gradK)
			copy(cache.dv, e.gradV)
			e.submitDWJob(func() {
				accumLinearGradCF(grad.Wo, attDx2, cache.attOut, stories.Dim, stories.Dim, e.seq)
			})
			e.submitDWJob(func() {
				accumLinearGrad3CF(grad.Wq, cache.dq, grad.Wk, cache.dk, grad.Wv, cache.dv, cache.xNorm, stories.Dim, stories.Dim, e.seq)
			})
		}
		// Attention residual lambda grads: sum(dx2 * x).
		for j := 0; j < stories.Dim*e.seq; j++ {
			dRL += float64(attDx2[j] * cache.x[j])
		}
		for j := 0; j < stories.Dim*e.seq; j++ {
			dXL += float64(attDx2[j] * e.x0[j])
		}
		// x0 gradient from attention residual.
		for j := 0; j < stories.Dim*e.seq; j++ {
			e.gX0[j] += xl * attDx2[j]
		}
		e.accum.ResidLambdas[l] += float32(dRL)
		e.accum.X0Lambdas[l] += float32(dXL)

		dCur, dPrev = dPrev, dCur
		if l == midLayer {
			addSlice(dCur, e.xMid)
		}
	}
	// Propagate x0 gradient into dCur.
	addSlice(dCur, e.gX0)
	e.waitDWJobs()

	// Smear backward.
	clear(e.gSmearGate)
	clear(e.gSmearLambda)
	smearBackwardCF(dCur, e.smearXPre, e.mw.SmearGate, e.mw.SmearLambda[0],
		e.smearGatePre, e.smearShifted, e.smearGateAct,
		e.gSmearGate, e.gSmearLambda, stories.Dim, e.seq)

	// Embedding norm backward.
	rmsNormNoWeightBackwardCF(dPrev, dCur, e.xPreEmbedNorm, stories.Dim, e.seq)
	copy(dCur, dPrev)

	if e.embedGradDone != nil {
		<-e.embedGradDone
		e.embedGradDone = nil
	}
	begin := time.Now()
	stories.EmbedBackward(e.gEmbed, dCur, input, stories.Dim, e.seq)
	e.stepMetrics.addEmbedGrad(time.Since(begin))
	addSlice(e.accum.Embed, e.gEmbed)
	addSlice(e.accumGSmearGate, e.gSmearGate)
	addSlice(e.accumGSmearLam, e.gSmearLambda)
	addSlice(e.accumGBackoutLam, e.gBackoutLambda)
	e.state.PendingSteps++
	if int(e.state.PendingSteps) >= e.accumSteps {
		return e.flushPending()
	}
	return 0
}

func (e *Engine) backwardAndApply(input []uint16, stepT int, useHybrid bool) time.Duration {
	dCur := e.dx
	dPrev := e.gradPrev
	// Backout backward.
	clear(e.gBackoutLambda)
	backoutBackwardCF(dCur, e.xMid, e.mw.BackoutLambda[0], e.gBackoutLambda, stories.Dim, e.seq)
	backoutLam := e.mw.BackoutLambda[0]
	for i := range dCur {
		e.xMid[i] = -backoutLam * dCur[i]
	}
	midLayer := stories.NLayers / 2
	clear(e.gX0)
	clear(e.gResidLambdas)
	clear(e.gX0Lambdas)
	for l := stories.NLayers - 1; l >= 0; l-- {
		layer := &e.mw.Layers[l]
		cache := &e.caches[l]
		grad := &e.applyGrads[l]
		clearLayerGrad(grad)
		rl := e.mw.ResidLambdas[l]

		copy(cache.dOut, dCur)
		if useHybrid {
			copy(cache.dx2, dCur)
			scaleSlice(cache.dx2, rl)
			if err := e.backwardFFNHybrid(e.backward[l], l, layer, cache, grad, cache.dOut, dPrev); err != nil {
				e.disableHybridBackward(fmt.Errorf("storiesane step: layer %d hybrid ffn backward: %w", l, err))
				useHybrid = false
			}
		}
		if !useHybrid {
			copy(e.gradX2, dCur)
			scaleSlice(e.gradX2, rl)
			e.backwardFFNCPU(l, layer, cache, grad, cache.dOut, dPrev)
			copy(cache.dh1, e.gradH1)
			copy(cache.dx2, e.gradX2)
			e.submitDWJob(func() {
				accumLinearGradCF(grad.W2, cache.dOut, cache.gate, stories.Dim, stories.Hidden, e.seq)
				accumLinearGradCF(grad.W1, cache.dh1, cache.x2Norm, stories.Hidden, stories.Dim, e.seq)
			})
		}

		var dRL float64
		for j := 0; j < stories.Dim*e.seq; j++ {
			dRL += float64(dCur[j] * cache.x2[j])
		}
		xl := e.mw.X0Lambdas[l]
		var dXL float64
		for j := 0; j < stories.Dim*e.seq; j++ {
			dXL += float64(dCur[j] * e.x0[j])
		}
		for j := 0; j < stories.Dim*e.seq; j++ {
			e.gX0[j] += xl * dCur[j]
		}

		attDx2 := cache.dx2
		if useHybrid {
			if err := e.backwardAttentionHybridWithDW(l, e.backward[l], layer, cache, grad, attDx2, dPrev, input); err != nil {
				e.disableHybridBackward(fmt.Errorf("storiesane step: layer %d hybrid attention backward: %w", l, err))
				useHybrid = false
			}
		}
		if !useHybrid {
			e.backwardAttentionCPU(l, layer, cache, grad, attDx2, dPrev, input)
			copy(cache.dq, e.gradQ)
			copy(cache.dk, e.gradK)
			copy(cache.dv, e.gradV)
			e.submitDWJob(func() {
				accumLinearGradCF(grad.Wo, attDx2, cache.attOut, stories.Dim, stories.Dim, e.seq)
			})
			e.submitDWJob(func() {
				accumLinearGrad3CF(grad.Wq, cache.dq, grad.Wk, cache.dk, grad.Wv, cache.dv, cache.xNorm, stories.Dim, stories.Dim, e.seq)
			})
		}
		for j := 0; j < stories.Dim*e.seq; j++ {
			dRL += float64(attDx2[j] * cache.x[j])
		}
		for j := 0; j < stories.Dim*e.seq; j++ {
			dXL += float64(attDx2[j] * e.x0[j])
		}
		for j := 0; j < stories.Dim*e.seq; j++ {
			e.gX0[j] += xl * attDx2[j]
		}
		e.gResidLambdas[l] += float32(dRL)
		e.gX0Lambdas[l] += float32(dXL)

		dCur, dPrev = dPrev, dCur
		if l == midLayer {
			addSlice(dCur, e.xMid)
		}
	}
	addSlice(dCur, e.gX0)
	e.waitDWJobs()

	// Smear backward.
	clear(e.gSmearGate)
	clear(e.gSmearLambda)
	smearBackwardCF(dCur, e.smearXPre, e.mw.SmearGate, e.mw.SmearLambda[0],
		e.smearGatePre, e.smearShifted, e.smearGateAct,
		e.gSmearGate, e.gSmearLambda, stories.Dim, e.seq)

	// Embedding norm backward.
	rmsNormNoWeightBackwardCF(dPrev, dCur, e.xPreEmbedNorm, stories.Dim, e.seq)
	copy(dCur, dPrev)

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
		scaleSlice(e.gEmbed, invLossScale)
		scaleSlice(e.gSmearGate, invLossScale)
		scaleSlice(e.gSmearLambda, invLossScale)
		scaleSlice(e.gBackoutLambda, invLossScale)
		scaleSlice(e.gResidLambdas, invLossScale)
		scaleSlice(e.gX0Lambdas, invLossScale)
	}
	e.clipLayerGradients(e.applyGrads, e.gEmbed)
	adamStart := time.Now()
	invBC1, invBC2 := adamBiasCorrectionInv(stepT, e.adamBeta1, e.adamBeta2)
	e.applyLayerAdamAll(e.applyGrads, stepT, invBC1, invBC2)
	adamUpdateCFWithInv(e.mw.Embed, e.gEmbed, &e.opt.Embed, e.lr, e.adamBeta1, e.adamBeta2, e.adamEps, e.weightDecay, invBC1, invBC2, true)
	adamUpdateCFWithInv(e.mw.ResidLambdas, e.gResidLambdas, &e.opt.ResidLambdas, e.lr, e.adamBeta1, e.adamBeta2, e.adamEps, 0, invBC1, invBC2, false)
	adamUpdateCFWithInv(e.mw.X0Lambdas, e.gX0Lambdas, &e.opt.X0Lambdas, e.lr, e.adamBeta1, e.adamBeta2, e.adamEps, 0, invBC1, invBC2, false)
	adamUpdateCFWithInv(e.mw.SmearGate, e.gSmearGate, &e.opt.SmearGate, e.lr, e.adamBeta1, e.adamBeta2, e.adamEps, 0, invBC1, invBC2, false)
	adamUpdateCFWithInv(e.mw.SmearLambda, e.gSmearLambda, &e.opt.SmearLambda, e.lr, e.adamBeta1, e.adamBeta2, e.adamEps, 0, invBC1, invBC2, false)
	adamUpdateCFWithInv(e.mw.BackoutLambda, e.gBackoutLambda, &e.opt.BackoutLambda, e.lr, e.adamBeta1, e.adamBeta2, e.adamEps, 0, invBC1, invBC2, false)
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

// veForwardCF applies per-head value embeddings: looks up VE embeddings by
// input token, computes a per-head gated value via sigmoid over the first
// VEGateChannels of x, and adds to V. Channel-first layout.
func veForwardCF(v, ve, veGateAct, veEmbed, veGate, x []float32, input []uint16, dim, seq int) {
	heads := stories.Heads
	headDim := stories.HeadDim
	gc := stories.VEGateChannels
	for t := 0; t < seq; t++ {
		tok := int(input[t])
		for d := 0; d < dim; d++ {
			ve[d*seq+t] = veEmbed[tok*dim+d]
		}
		for h := 0; h < heads; h++ {
			gatePre := float32(0)
			gateBase := h * gc
			for c := 0; c < gc; c++ {
				gatePre += veGate[gateBase+c] * x[c*seq+t]
			}
			act := float32(1.0 / (1.0 + math.Exp(-float64(gatePre))))
			veGateAct[h*seq+t] = act
			gate := 3.0 * act
			dBase := h * headDim
			for d := dBase; d < dBase+headDim; d++ {
				v[d*seq+t] += gate * ve[d*seq+t]
			}
		}
	}
}

// veBackwardCF computes per-head VE gradients from dV.
func veBackwardCF(dVEEmbed, dVEGate, dPrev, dv, ve, veGateAct, veEmbed, veGate, x []float32, input []uint16, dim, seq int) {
	heads := stories.Heads
	headDim := stories.HeadDim
	gc := stories.VEGateChannels
	for t := 0; t < seq; t++ {
		tok := int(input[t])
		for h := 0; h < heads; h++ {
			act := veGateAct[h*seq+t]
			gate := float32(3.0) * act
			dGate := float32(0)
			dBase := h * headDim
			for d := dBase; d < dBase+headDim; d++ {
				dvdt := dv[d*seq+t]
				dVEEmbed[tok*dim+d] += dvdt * gate
				dGate += dvdt * ve[d*seq+t]
			}
			dGatePre := dGate * 3.0 * act * (1 - act)
			gateBase := h * gc
			for c := 0; c < gc; c++ {
				dVEGate[gateBase+c] += dGatePre * x[c*seq+t]
				dPrev[c*seq+t] += dGatePre * veGate[gateBase+c]
			}
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
	adamUpdateCFWithInv(dst.VEEmbed, grad.VEEmbed, &st.VEEmbed, lr, b1, b2, eps, 0, invBC1, invBC2, true)
	adamUpdateCFWithInv(dst.VEGate, grad.VEGate, &st.VEGate, lr, b1, b2, eps, 0, invBC1, invBC2, false)
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
