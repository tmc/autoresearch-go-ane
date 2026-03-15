//go:build darwin

package ane

import (
	"fmt"
	"math"
	"sync"

	"github.com/tmc/apple/x/ane/dynamicmatmul"
	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

// tiledExecutors holds dynamicmatmul.Executor instances for each matmul
// in a tiled transformer layer. Each executor handles its own internal
// tiling so that large weight matrices (exceeding ANE SRAM) work correctly.
type tiledExecutors struct {
	wq *dynamicmatmul.Executor
	wk *dynamicmatmul.Executor
	wv *dynamicmatmul.Executor
	wo *dynamicmatmul.Executor
	w1 *dynamicmatmul.Executor
	w3 *dynamicmatmul.Executor
	w2 *dynamicmatmul.Executor
}

func (te *tiledExecutors) close() {
	if te == nil {
		return
	}
	if te.wq != nil {
		te.wq.Close()
	}
	if te.wk != nil {
		te.wk.Close()
	}
	if te.wv != nil {
		te.wv.Close()
	}
	if te.wo != nil {
		te.wo.Close()
	}
	if te.w1 != nil {
		te.w1.Close()
	}
	if te.w3 != nil {
		te.w3.Close()
	}
	if te.w2 != nil {
		te.w2.Close()
	}
}

// compileStoriesLayerForwardTiled compiles a tiled layer forward pass using
// dynamicmatmul.Executor for each projection. This path works for arbitrarily
// large models where the monolithic MIL program would exceed ANE SRAM limits.
//
// Each matmul (Wq, Wk, Wv, Wo, W1, W3, W2) is an independent
// dynamicmatmul.Executor. The CPU handles RMSNorm, RoPE, causal attention,
// SiLU gating, and residual connections between ANE matmuls.
func compileStoriesLayerForwardTiled(cfg stories.ModelConfig, layer stories.LayerWeights, seq int) (_ *layerForward, err error) {
	dim := cfg.Dim
	hidden := cfg.Hidden
	heads := cfg.Heads
	kvHeads := cfg.EffectiveKVHeads()
	kvDim := cfg.KVDim()
	headDim := cfg.HeadDim()

	if dim <= 0 || hidden <= 0 || heads <= 0 || seq <= 0 {
		return nil, fmt.Errorf("compile tiled layer: invalid shape dim=%d hidden=%d heads=%d seq=%d", dim, hidden, heads, seq)
	}
	if dim%heads != 0 {
		return nil, fmt.Errorf("compile tiled layer: dim=%d not divisible by heads=%d", dim, heads)
	}
	if kvHeads > 0 && heads%kvHeads != 0 {
		return nil, fmt.Errorf("compile tiled layer: heads=%d not divisible by kvHeads=%d", heads, kvHeads)
	}

	// Validate weight dimensions.
	check := func(name string, got, want int) error {
		if got != want {
			return fmt.Errorf("compile tiled layer: %s len=%d want=%d", name, got, want)
		}
		return nil
	}
	if err := check("rms_att", len(layer.RMSAtt), dim); err != nil {
		return nil, err
	}
	if err := check("wq", len(layer.Wq), dim*dim); err != nil {
		return nil, err
	}
	if err := check("wk", len(layer.Wk), kvDim*dim); err != nil {
		return nil, err
	}
	if err := check("wv", len(layer.Wv), kvDim*dim); err != nil {
		return nil, err
	}
	if err := check("wo", len(layer.Wo), dim*dim); err != nil {
		return nil, err
	}
	if err := check("rms_ffn", len(layer.RMSFFN), dim); err != nil {
		return nil, err
	}
	if err := check("w1", len(layer.W1), hidden*dim); err != nil {
		return nil, err
	}
	if err := check("w2", len(layer.W2), dim*hidden); err != nil {
		return nil, err
	}
	if err := check("w3", len(layer.W3), hidden*dim); err != nil {
		return nil, err
	}

	_ = headDim // used implicitly via heads/dim

	// Compile executors. batch=seq for channel-first layout.
	// Weight layout for EvalCFIOInto: weights are [inDim, outDim] row-major.
	// For Wq: input is xnorm[dim, seq], output is q[dim, seq].
	//   weights are Wq[dim, dim] (row-major), but linearCF treats weights as
	//   [outDim, inDim] row-major. The dynamicmatmul executor expects [inDim, outDim].
	//   Since Wq is stored as [outDim, inDim] = [dim, dim] for linearCF,
	//   we need to transpose it for the executor: [inDim, outDim].
	//   Actually, PrimeWeightsIO expects [inDim, outDim] row-major.
	//   The stored weights are [outDim, inDim] row-major (like linearCF).
	//   So we need to transpose when priming.

	var te tiledExecutors
	defer func() {
		if err != nil {
			te.close()
		}
	}()

	opts := dynamicmatmul.Options{}

	// Wq: [dim, dim] → q[dim, seq]
	te.wq, err = dynamicmatmul.New(seq, dim, dim, opts)
	if err != nil {
		return nil, fmt.Errorf("compile tiled layer: wq executor: %w", err)
	}

	// Wk: [dim, kvDim] → k[kvDim, seq]
	te.wk, err = dynamicmatmul.New(seq, dim, kvDim, opts)
	if err != nil {
		return nil, fmt.Errorf("compile tiled layer: wk executor: %w", err)
	}

	// Wv: [dim, kvDim] → v[kvDim, seq]
	te.wv, err = dynamicmatmul.New(seq, dim, kvDim, opts)
	if err != nil {
		return nil, fmt.Errorf("compile tiled layer: wv executor: %w", err)
	}

	// Wo: [dim, dim] → oo[dim, seq]
	te.wo, err = dynamicmatmul.New(seq, dim, dim, opts)
	if err != nil {
		return nil, fmt.Errorf("compile tiled layer: wo executor: %w", err)
	}

	// W1 (gate): [dim, hidden] → h1[hidden, seq]
	te.w1, err = dynamicmatmul.New(seq, dim, hidden, opts)
	if err != nil {
		return nil, fmt.Errorf("compile tiled layer: w1 executor: %w", err)
	}

	// W3 (up): [dim, hidden] → h3[hidden, seq]
	te.w3, err = dynamicmatmul.New(seq, dim, hidden, opts)
	if err != nil {
		return nil, fmt.Errorf("compile tiled layer: w3 executor: %w", err)
	}

	// W2 (down): [hidden, dim] → ff[dim, seq]
	te.w2, err = dynamicmatmul.New(seq, hidden, dim, opts)
	if err != nil {
		return nil, fmt.Errorf("compile tiled layer: w2 executor: %w", err)
	}

	// Prime weights. The executor expects [inDim, outDim] row-major.
	// Our stored weights are [outDim, inDim] row-major (for linearCF).
	// We need to transpose them.
	wqT := transposeWeights(layer.Wq, dim, dim)
	if err := te.wq.PrimeWeightsIO(wqT); err != nil {
		return nil, fmt.Errorf("compile tiled layer: prime wq: %w", err)
	}

	wkT := transposeWeights(layer.Wk, kvDim, dim)
	if err := te.wk.PrimeWeightsIO(wkT); err != nil {
		return nil, fmt.Errorf("compile tiled layer: prime wk: %w", err)
	}

	wvT := transposeWeights(layer.Wv, kvDim, dim)
	if err := te.wv.PrimeWeightsIO(wvT); err != nil {
		return nil, fmt.Errorf("compile tiled layer: prime wv: %w", err)
	}

	woT := transposeWeights(layer.Wo, dim, dim)
	if err := te.wo.PrimeWeightsIO(woT); err != nil {
		return nil, fmt.Errorf("compile tiled layer: prime wo: %w", err)
	}

	w1T := transposeWeights(layer.W1, hidden, dim)
	if err := te.w1.PrimeWeightsIO(w1T); err != nil {
		return nil, fmt.Errorf("compile tiled layer: prime w1: %w", err)
	}

	w3T := transposeWeights(layer.W3, hidden, dim)
	if err := te.w3.PrimeWeightsIO(w3T); err != nil {
		return nil, fmt.Errorf("compile tiled layer: prime w3: %w", err)
	}

	w2T := transposeWeights(layer.W2, dim, hidden)
	if err := te.w2.PrimeWeightsIO(w2T); err != nil {
		return nil, fmt.Errorf("compile tiled layer: prime w2: %w", err)
	}

	lf := &layerForward{
		dim:    dim,
		hidden: hidden,
		heads:  heads,
		seq:    seq,
		tiled:  &te,
		rmsAtt: layer.RMSAtt,
		rmsFFN: layer.RMSFFN,
		x2:     make([]float32, dim*seq),
	}
	// Allocate tiled scratch buffers.
	lf.tiledScratch = &tiledScratchBuffers{
		xnorm:  make([]float32, dim*seq),
		q:      make([]float32, dim*seq),
		k:      make([]float32, kvDim*seq),
		v:      make([]float32, kvDim*seq),
		attOut: make([]float32, dim*seq),
		oo:     make([]float32, dim*seq),
		x2norm: make([]float32, dim*seq),
		h1:     make([]float32, hidden*seq),
		h3:     make([]float32, hidden*seq),
		gate:   make([]float32, hidden*seq),
		ff:     make([]float32, dim*seq),

		kvHeads: kvHeads,
		headDim: cfg.HeadDim(),

		residualScale: float32(cfg.ResidualScale()),
	}

	return lf, nil
}

// tiledScratchBuffers holds intermediate buffers for the tiled forward path.
type tiledScratchBuffers struct {
	xnorm  []float32
	q      []float32
	k      []float32
	v      []float32
	attOut []float32
	oo     []float32
	x2norm []float32
	h1     []float32
	h3     []float32
	gate   []float32
	ff     []float32

	ropeCos []float32
	ropeSin []float32

	kvHeads       int
	headDim       int
	residualScale float32
}

// transposeWeights transposes a [rows, cols] row-major matrix to [cols, rows] row-major.
func transposeWeights(w []float32, rows, cols int) []float32 {
	out := make([]float32, len(w))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = w[r*cols+c]
		}
	}
	return out
}

// runTiled executes a tiled layer forward pass. Each matmul runs on ANE via
// dynamicmatmul.Executor, while CPU handles RMSNorm, RoPE, attention, SiLU,
// and residual connections.
func (lf *layerForward) runTiled(out, x []float32) error {
	if lf == nil || lf.tiled == nil || lf.tiledScratch == nil {
		return fmt.Errorf("run tiled layer: layer is not tiled")
	}
	dim := lf.dim
	seq := lf.seq
	heads := lf.heads
	te := lf.tiled
	sc := lf.tiledScratch
	want := dim * seq

	if len(x) != want || len(out) != want {
		return fmt.Errorf("run tiled layer: len mismatch in=%d out=%d want=%d", len(x), len(out), want)
	}

	// ---- Attention block ----

	// CPU: RMSNorm
	rmsNormCF(sc.xnorm, x, lf.rmsAtt, dim, seq)

	// ANE: Wq @ xnorm → q [dim, seq]
	if _, err := te.wq.EvalCFIOInto(sc.q, sc.xnorm); err != nil {
		return fmt.Errorf("run tiled layer: wq: %w", err)
	}

	// ANE: Wk @ xnorm → k [kvDim, seq]
	if _, err := te.wk.EvalCFIOInto(sc.k, sc.xnorm); err != nil {
		return fmt.Errorf("run tiled layer: wk: %w", err)
	}

	// ANE: Wv @ xnorm → v [kvDim, seq]
	if _, err := te.wv.EvalCFIOInto(sc.v, sc.xnorm); err != nil {
		return fmt.Errorf("run tiled layer: wv: %w", err)
	}

	// CPU: RoPE on q and k
	headDim := sc.headDim
	kvHeads := sc.kvHeads

	// Cache RoPE tables on first call.
	if sc.ropeCos == nil {
		sc.ropeCos, sc.ropeSin = buildRoPETables(seq, headDim)
	}
	applyRoPECFInPlace(sc.q, heads, headDim, seq, sc.ropeCos, sc.ropeSin)
	applyRoPECFInPlace(sc.k, kvHeads, headDim, seq, sc.ropeCos, sc.ropeSin)

	// CPU: causal attention (GQA-aware)
	gqaCausalAttentionCF(sc.attOut, sc.q, sc.k, sc.v, heads, kvHeads, headDim, seq)

	// ANE: Wo @ attOut → oo [dim, seq]
	if _, err := te.wo.EvalCFIOInto(sc.oo, sc.attOut); err != nil {
		return fmt.Errorf("run tiled layer: wo: %w", err)
	}

	// CPU: residual connection: x2 = x + scale * oo
	addScaledResidualWith(lf.x2, x, sc.oo, sc.residualScale)

	// ---- FFN block ----

	// CPU: RMSNorm
	rmsNormCF(sc.x2norm, lf.x2, lf.rmsFFN, dim, seq)

	// ANE: W1 @ x2norm → h1 [hidden, seq]
	if _, err := te.w1.EvalCFIOInto(sc.h1, sc.x2norm); err != nil {
		return fmt.Errorf("run tiled layer: w1: %w", err)
	}

	// ANE: W3 @ x2norm → h3 [hidden, seq]
	if _, err := te.w3.EvalCFIOInto(sc.h3, sc.x2norm); err != nil {
		return fmt.Errorf("run tiled layer: w3: %w", err)
	}

	// CPU: SiLU gate = silu(h1) * h3
	siluMulAccel(sc.gate, sc.h1, sc.h3)

	// ANE: W2 @ gate → ff [dim, seq]
	if _, err := te.w2.EvalCFIOInto(sc.ff, sc.gate); err != nil {
		return fmt.Errorf("run tiled layer: w2: %w", err)
	}

	// CPU: residual connection: out = x2 + scale * ff
	addScaledResidualWith(out, lf.x2, sc.ff, sc.residualScale)

	return nil
}

// closeTiled releases tiled executor resources.
func (lf *layerForward) closeTiled() {
	if lf == nil {
		return
	}
	if lf.tiled != nil {
		lf.tiled.close()
		lf.tiled = nil
	}
	lf.tiledScratch = nil
}

// addScaledResidualWith computes dst[i] = base[i] + scale*branch[i].
func addScaledResidualWith(dst, base, branch []float32, scale float32) {
	addScaledResidualAccel(dst, base, branch, scale)
}

// gqaCausalAttentionCF computes grouped-query causal attention in channel-first layout.
//
// q has shape [heads*headDim, seq], k and v have shape [kvHeads*headDim, seq].
// When kvHeads < heads, each KV head is shared by (heads/kvHeads) query heads.
// Output has shape [heads*headDim, seq].
//
// Uses BLAS (cblas_sgemm) for the two expensive matmuls:
//   - scores = Q_h^T @ K_kvH  (attention scores)
//   - out_h  = V_kvH @ scores^T  (value gather)
// Causal mask and softmax are applied row-by-row between the two BLAS calls.
func gqaCausalAttentionCF(out, qf, kf, vf []float32, heads, kvHeads, headDim, seq int) {
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	groupSize := 1
	if kvHeads > 0 && kvHeads < heads {
		groupSize = heads / kvHeads
	}

	// Process heads in parallel groups of 2 to overlap BLAS + softmax work.
	var wg sync.WaitGroup
	for h := 0; h < heads; h += 2 {
		endH := h + 2
		if endH > heads {
			endH = heads
		}
		wg.Add(1)
		go func(startH, endH int) {
			defer wg.Done()
			scores := make([]float32, seq*seq)
			for hh := startH; hh < endH; hh++ {
				kvH := hh / groupSize
				qOff := hh * headDim * seq
				kOff := kvH * headDim * seq
				vOff := kvH * headDim * seq
				outOff := hh * headDim * seq

				gqaAttentionScoresBLAS(scores, qf[qOff:], kf[kOff:], headDim, seq, scale)

				for t := 0; t < seq; t++ {
					row := scores[t*seq : (t+1)*seq]
					for j := t + 1; j < seq; j++ {
						row[j] = -65504
					}
					softmaxRow(row, row)
				}

				gqaAttentionValueBLAS(out[outOff:], vf[vOff:], scores, headDim, seq)
			}
		}(h, endH)
	}
	wg.Wait()
}
