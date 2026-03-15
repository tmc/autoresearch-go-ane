//go:build darwin

package storiesane

import (
	"fmt"
	"math"
	"sync"

	"github.com/tmc/autoresearch-go-ane/ane/dynamicmatmul"
	"github.com/tmc/autoresearch-go-ane/ane/stories"
)

// moeExpertExecutor holds ANE executors for a single expert's FFN.
// Executors are compiled lazily on first use to avoid compiling all 512 experts upfront.
type moeExpertExecutor struct {
	w1 *dynamicmatmul.Executor // gate_proj: [dim, expertHidden]
	w2 *dynamicmatmul.Executor // down_proj: [expertHidden, dim]
	w3 *dynamicmatmul.Executor // up_proj: [dim, expertHidden]
}

func (e *moeExpertExecutor) close() {
	if e == nil {
		return
	}
	if e.w1 != nil {
		e.w1.Close()
	}
	if e.w2 != nil {
		e.w2.Close()
	}
	if e.w3 != nil {
		e.w3.Close()
	}
}

// moeLayer holds the compiled executors for a single MoE transformer layer.
type moeLayer struct {
	cfg stories.MoEConfig

	// Attention executors (same as dense tiled path).
	attExec tiledExecutors

	// Attention scratch buffers.
	attScratch *tiledScratchBuffers
	rmsAtt     []float32
	rmsFFN     []float32

	// Router weights for CPU-side gating. [numExperts, dim] row-major.
	routerWeight []float32

	// Expert executors, lazily compiled.
	mu      sync.Mutex
	experts []*moeExpertExecutor // len == numExperts, nil until compiled
	shared  *moeExpertExecutor   // nil if no shared expert
	weights *stories.MoELayerWeights
	seq     int

	// Scratch buffers for MoE forward.
	x2      []float32 // [dim * seq] residual after attention
	x2norm  []float32 // [dim * seq] RMSNorm of x2
	logits  []float32 // [numExperts] router logits per token
	h1      []float32 // [expertHidden * seq] gate projection scratch
	h3      []float32 // [expertHidden * seq] up projection scratch
	gate    []float32 // [expertHidden * seq] silu(h1) * h3
	ffExp   []float32 // [dim * seq] single expert output scratch
	ffAccum []float32 // [dim * seq] accumulated expert outputs
}

// compileMoELayer compiles a MoE layer's attention executors and prepares
// lazy expert compilation. Expert ANE executors are compiled on first use.
func compileMoELayer(cfg stories.MoEConfig, layer stories.MoELayerWeights, seq int) (_ *moeLayer, err error) {
	dim := cfg.Dim
	hidden := cfg.ExpertHidden
	heads := cfg.Heads
	kvHeads := cfg.EffectiveKVHeads()
	kvDim := cfg.KVDim()

	if dim <= 0 || hidden <= 0 || heads <= 0 || seq <= 0 {
		return nil, fmt.Errorf("compile moe layer: invalid shape dim=%d hidden=%d heads=%d seq=%d", dim, hidden, heads, seq)
	}
	if dim%heads != 0 {
		return nil, fmt.Errorf("compile moe layer: dim=%d not divisible by heads=%d", dim, heads)
	}
	if kvHeads > 0 && heads%kvHeads != 0 {
		return nil, fmt.Errorf("compile moe layer: heads=%d not divisible by kvHeads=%d", heads, kvHeads)
	}
	if cfg.NumExperts <= 0 || cfg.NumActiveExperts <= 0 {
		return nil, fmt.Errorf("compile moe layer: invalid expert config numExperts=%d numActive=%d", cfg.NumExperts, cfg.NumActiveExperts)
	}
	if len(layer.Experts) != cfg.NumExperts {
		return nil, fmt.Errorf("compile moe layer: experts len=%d want=%d", len(layer.Experts), cfg.NumExperts)
	}
	if len(layer.RouterWeight) != cfg.RouterSize() {
		return nil, fmt.Errorf("compile moe layer: router weight len=%d want=%d", len(layer.RouterWeight), cfg.RouterSize())
	}

	// Compile attention executors (reuse tiled path).
	var te tiledExecutors
	defer func() {
		if err != nil {
			te.close()
		}
	}()

	opts := dynamicmatmul.Options{}

	te.wq, err = dynamicmatmul.New(seq, dim, dim, opts)
	if err != nil {
		return nil, fmt.Errorf("compile moe layer: wq executor: %w", err)
	}
	te.wk, err = dynamicmatmul.New(seq, dim, kvDim, opts)
	if err != nil {
		return nil, fmt.Errorf("compile moe layer: wk executor: %w", err)
	}
	te.wv, err = dynamicmatmul.New(seq, dim, kvDim, opts)
	if err != nil {
		return nil, fmt.Errorf("compile moe layer: wv executor: %w", err)
	}
	te.wo, err = dynamicmatmul.New(seq, dim, dim, opts)
	if err != nil {
		return nil, fmt.Errorf("compile moe layer: wo executor: %w", err)
	}

	// Prime attention weights (transposed for dynamicmatmul).
	wqT := transposeWeights(layer.Wq, dim, dim)
	if err := te.wq.PrimeWeightsIO(wqT); err != nil {
		return nil, fmt.Errorf("compile moe layer: prime wq: %w", err)
	}
	wkT := transposeWeights(layer.Wk, kvDim, dim)
	if err := te.wk.PrimeWeightsIO(wkT); err != nil {
		return nil, fmt.Errorf("compile moe layer: prime wk: %w", err)
	}
	wvT := transposeWeights(layer.Wv, kvDim, dim)
	if err := te.wv.PrimeWeightsIO(wvT); err != nil {
		return nil, fmt.Errorf("compile moe layer: prime wv: %w", err)
	}
	woT := transposeWeights(layer.Wo, dim, dim)
	if err := te.wo.PrimeWeightsIO(woT); err != nil {
		return nil, fmt.Errorf("compile moe layer: prime wo: %w", err)
	}

	headDim := cfg.HeadDim()
	residualScale := float32(cfg.ResidualScale())

	ml := &moeLayer{
		cfg:     cfg,
		attExec: te,
		attScratch: &tiledScratchBuffers{
			xnorm:         make([]float32, dim*seq),
			q:             make([]float32, dim*seq),
			k:             make([]float32, kvDim*seq),
			v:             make([]float32, kvDim*seq),
			attOut:        make([]float32, dim*seq),
			oo:            make([]float32, dim*seq),
			kvHeads:       kvHeads,
			headDim:       headDim,
			residualScale: residualScale,
		},
		rmsAtt:       layer.RMSAtt,
		rmsFFN:       layer.RMSFFN,
		routerWeight: layer.RouterWeight,
		experts: make([]*moeExpertExecutor, cfg.NumExperts),
		weights: &layer,
		seq:     seq,
		x2:          make([]float32, dim*seq),
		x2norm:      make([]float32, dim*seq),
		logits:      make([]float32, cfg.NumExperts),
		h1:          make([]float32, hidden*seq),
		h3:          make([]float32, hidden*seq),
		gate:        make([]float32, hidden*seq),
		ffExp:       make([]float32, dim*seq),
		ffAccum:     make([]float32, dim*seq),
	}

	// Compile shared expert eagerly if present (it runs every token).
	if cfg.HasSharedExpert && layer.SharedExpert != nil {
		ml.shared, err = compileExpertExecutor(cfg, *layer.SharedExpert, seq)
		if err != nil {
			return nil, fmt.Errorf("compile moe layer: shared expert: %w", err)
		}
	}

	return ml, nil
}

// compileExpertExecutor compiles ANE executors for a single expert's FFN
// and primes their weights.
func compileExpertExecutor(cfg stories.MoEConfig, ew stories.ExpertWeights, seq int) (*moeExpertExecutor, error) {
	dim := cfg.Dim
	hidden := cfg.ExpertHidden
	opts := dynamicmatmul.Options{}

	w1Exec, err := dynamicmatmul.New(seq, dim, hidden, opts)
	if err != nil {
		return nil, fmt.Errorf("expert w1: %w", err)
	}
	w3Exec, err := dynamicmatmul.New(seq, dim, hidden, opts)
	if err != nil {
		w1Exec.Close()
		return nil, fmt.Errorf("expert w3: %w", err)
	}
	w2Exec, err := dynamicmatmul.New(seq, hidden, dim, opts)
	if err != nil {
		w1Exec.Close()
		w3Exec.Close()
		return nil, fmt.Errorf("expert w2: %w", err)
	}

	// Prime weights. Stored as [outDim, inDim] row-major, executor wants [inDim, outDim].
	w1T := transposeWeights(ew.W1, hidden, dim)
	if err := w1Exec.PrimeWeightsIO(w1T); err != nil {
		w1Exec.Close()
		w3Exec.Close()
		w2Exec.Close()
		return nil, fmt.Errorf("prime expert w1: %w", err)
	}
	w3T := transposeWeights(ew.W3, hidden, dim)
	if err := w3Exec.PrimeWeightsIO(w3T); err != nil {
		w1Exec.Close()
		w3Exec.Close()
		w2Exec.Close()
		return nil, fmt.Errorf("prime expert w3: %w", err)
	}
	w2T := transposeWeights(ew.W2, dim, hidden)
	if err := w2Exec.PrimeWeightsIO(w2T); err != nil {
		w1Exec.Close()
		w3Exec.Close()
		w2Exec.Close()
		return nil, fmt.Errorf("prime expert w2: %w", err)
	}

	return &moeExpertExecutor{w1: w1Exec, w2: w2Exec, w3: w3Exec}, nil
}

// ensureExpertCompiled lazily compiles expert idx if not already compiled.
func (ml *moeLayer) ensureExpertCompiled(idx int) (*moeExpertExecutor, error) {
	ml.mu.Lock()
	defer ml.mu.Unlock()

	if ml.experts[idx] != nil {
		return ml.experts[idx], nil
	}

	exec, err := compileExpertExecutor(ml.cfg, ml.weights.Experts[idx], ml.seq)
	if err != nil {
		return nil, fmt.Errorf("lazy compile expert %d: %w", idx, err)
	}
	ml.experts[idx] = exec
	return exec, nil
}

// routerTopK computes router logits on CPU and returns the top-k expert
// indices and their softmax-normalized gating weights.
//
// xnorm is the RMSNorm'd hidden state for a single token position: [dim].
// routerW is [numExperts, dim] row-major.
// Returns topIndices[k] and topWeights[k].
func routerTopK(xnorm []float32, routerW []float32, dim, numExperts, k int) ([]int, []float32) {
	// Compute router logits: routerW @ xnorm (CPU matmul, tiny).
	logits := make([]float32, numExperts)
	for e := 0; e < numExperts; e++ {
		sum := float32(0)
		row := routerW[e*dim : (e+1)*dim]
		for d := 0; d < dim; d++ {
			sum += row[d] * xnorm[d]
		}
		logits[e] = sum
	}

	// Select top-k using simple O(n*k) selection.
	topIndices := make([]int, k)
	topLogits := make([]float32, k)
	used := make([]bool, numExperts)

	for i := 0; i < k; i++ {
		bestIdx := -1
		bestVal := float32(-math.MaxFloat32)
		for e := 0; e < numExperts; e++ {
			if !used[e] && logits[e] > bestVal {
				bestVal = logits[e]
				bestIdx = e
			}
		}
		topIndices[i] = bestIdx
		topLogits[i] = bestVal
		if bestIdx >= 0 {
			used[bestIdx] = true
		}
	}

	// Softmax over selected top-k logits only.
	topWeights := make([]float32, k)
	maxLogit := topLogits[0]
	for i := 1; i < k; i++ {
		if topLogits[i] > maxLogit {
			maxLogit = topLogits[i]
		}
	}
	sumExp := float32(0)
	for i := 0; i < k; i++ {
		topWeights[i] = float32(math.Exp(float64(topLogits[i] - maxLogit)))
		sumExp += topWeights[i]
	}
	if sumExp > 0 {
		for i := 0; i < k; i++ {
			topWeights[i] /= sumExp
		}
	}

	return topIndices, topWeights
}

// runExpertFFN runs a single expert FFN on ANE.
// Input xnormCF is channel-first [dim, seq], output ffOut is channel-first [dim, seq].
func runExpertFFN(exec *moeExpertExecutor, ffOut, xnormCF, h1, h3, gateBuf []float32) error {
	// W1 @ xnorm -> h1 [expertHidden, seq]
	if _, err := exec.w1.EvalCFIOInto(h1, xnormCF); err != nil {
		return fmt.Errorf("expert w1: %w", err)
	}
	// W3 @ xnorm -> h3 [expertHidden, seq]
	if _, err := exec.w3.EvalCFIOInto(h3, xnormCF); err != nil {
		return fmt.Errorf("expert w3: %w", err)
	}
	// SiLU gate = silu(h1) * h3
	siluMulAccel(gateBuf, h1, h3)
	// W2 @ gate -> ffOut [dim, seq]
	if _, err := exec.w2.EvalCFIOInto(ffOut, gateBuf); err != nil {
		return fmt.Errorf("expert w2: %w", err)
	}
	return nil
}

// runMoEForward executes a full MoE layer forward pass.
//
// x is [dim * seq] channel-first input, out is [dim * seq] channel-first output.
func (ml *moeLayer) runMoEForward(out, x []float32) error {
	dim := ml.cfg.Dim
	seq := ml.seq
	hidden := ml.cfg.ExpertHidden
	want := dim * seq

	if len(x) != want || len(out) != want {
		return fmt.Errorf("run moe forward: len mismatch in=%d out=%d want=%d", len(x), len(out), want)
	}

	// ---- Attention block (same as dense tiled path) ----

	te := &ml.attExec
	sc := ml.attScratch

	// CPU: RMSNorm
	rmsNormCF(sc.xnorm, x, ml.rmsAtt, dim, seq)

	// ANE: Wq @ xnorm -> q
	if _, err := te.wq.EvalCFIOInto(sc.q, sc.xnorm); err != nil {
		return fmt.Errorf("run moe forward: wq: %w", err)
	}
	// ANE: Wk @ xnorm -> k
	if _, err := te.wk.EvalCFIOInto(sc.k, sc.xnorm); err != nil {
		return fmt.Errorf("run moe forward: wk: %w", err)
	}
	// ANE: Wv @ xnorm -> v
	if _, err := te.wv.EvalCFIOInto(sc.v, sc.xnorm); err != nil {
		return fmt.Errorf("run moe forward: wv: %w", err)
	}

	// CPU: RoPE
	headDim := sc.headDim
	kvHeads := sc.kvHeads
	ropeCos, ropeSin := buildRoPETables(seq, headDim)
	applyRoPECFInPlace(sc.q, ml.cfg.Heads, headDim, seq, ropeCos, ropeSin)
	applyRoPECFInPlace(sc.k, kvHeads, headDim, seq, ropeCos, ropeSin)

	// CPU: causal attention (GQA-aware)
	gqaCausalAttentionCF(sc.attOut, sc.q, sc.k, sc.v, ml.cfg.Heads, kvHeads, headDim, seq)

	// ANE: Wo @ attOut -> oo
	if _, err := te.wo.EvalCFIOInto(sc.oo, sc.attOut); err != nil {
		return fmt.Errorf("run moe forward: wo: %w", err)
	}

	// CPU: residual connection: x2 = x + scale * oo
	addScaledResidualWith(ml.x2, x, sc.oo, sc.residualScale)

	// ---- MoE FFN block ----

	// CPU: RMSNorm
	rmsNormCF(ml.x2norm, ml.x2, ml.rmsFFN, dim, seq)

	// Zero the accumulator.
	for i := range ml.ffAccum {
		ml.ffAccum[i] = 0
	}

	// Shared expert runs once for all tokens (not per-token).
	if ml.shared != nil {
		if err := runExpertFFN(ml.shared, ml.ffAccum, ml.x2norm, ml.h1[:hidden*seq], ml.h3[:hidden*seq], ml.gate[:hidden*seq]); err != nil {
			return fmt.Errorf("run moe forward: shared expert ffn: %w", err)
		}
	}

	// For each token position, route to top-k experts.
	// We extract per-token hidden states from channel-first layout for the router.
	tokenHidden := make([]float32, dim)

	for t := 0; t < seq; t++ {
		// Extract token t's hidden state from channel-first x2norm[dim, seq].
		for d := 0; d < dim; d++ {
			tokenHidden[d] = ml.x2norm[d*seq+t]
		}

		// CPU: router top-k selection.
		topIdx, topW := routerTopK(tokenHidden, ml.routerWeight, dim, ml.cfg.NumExperts, ml.cfg.NumActiveExperts)

		// Run each selected expert.
		for i, expertIdx := range topIdx {
			if expertIdx < 0 {
				continue
			}
			exec, err := ml.ensureExpertCompiled(expertIdx)
			if err != nil {
				return fmt.Errorf("run moe forward: token %d expert %d: %w", t, expertIdx, err)
			}

			// The executor is compiled for full seq, so we run the expert on full x2norm
			// and then extract position t's output.
			if err := runExpertFFN(exec, ml.ffExp, ml.x2norm, ml.h1[:hidden*seq], ml.h3[:hidden*seq], ml.gate[:hidden*seq]); err != nil {
				return fmt.Errorf("run moe forward: token %d expert %d ffn: %w", t, expertIdx, err)
			}

			// Accumulate: ffAccum[:, t] += topW[i] * ffExp[:, t]
			w := topW[i]
			for d := 0; d < dim; d++ {
				ml.ffAccum[d*seq+t] += w * ml.ffExp[d*seq+t]
			}
		}
	}

	// CPU: residual connection: out = x2 + scale * ffAccum
	addScaledResidualWith(out, ml.x2, ml.ffAccum, sc.residualScale)

	return nil
}

// close releases all resources held by the MoE layer.
func (ml *moeLayer) close() {
	if ml == nil {
		return
	}
	ml.attExec.close()
	ml.mu.Lock()
	for _, e := range ml.experts {
		e.close()
	}
	ml.experts = nil
	ml.mu.Unlock()
	if ml.shared != nil {
		ml.shared.close()
		ml.shared = nil
	}
}
